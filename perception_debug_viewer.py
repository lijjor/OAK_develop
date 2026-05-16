"""
perception_debug_viewer.py

最小三图调试入口。

输入链路：
- 与 `get_depth_claude_avoid.py` 共享同一套 depth / disparity 获取逻辑

处理链路：
1. Layer 1：`ground_filter.segment_ground()` 在 disparity 域识别地面
2. Layer 2：`height_filter.HeightFilter.filter()` 在 depth_mm 域按相对机身顶部过滤

显示链路：
- `Disparity Raw`        = 原始 `latest_disp`
- `Disparity No Ground`  = Layer 1 输出的 `disp_no_ground`
- `Disparity For Avoid`  = `disp_no_ground` 再套 Layer 2 的 `kept_mask`
"""

import argparse
import time
from pathlib import Path

import cv2
import depthai as dai
import numpy as np

from get_depth_claude_avoid import (
    _MEDIAN_MAP,
    _RESOLUTION_MAP,
    add_common_depth_args,
    build_colormap_with_zero_black,
    compute_max_disparity,
    create_device_context,
    create_depth_pipeline,
    fit_to_window,
    update_latest_stereo_frames,
)
from ground_filter import GroundTracker, segment_ground
from height_filter import HeightFilter, HeightFilterConfig


def colorize_disparity(disp_frame: np.ndarray, disp_mul: float, color_lut: np.ndarray) -> np.ndarray:
    """把真实 disparity 帧映射成伪彩色，0 保持黑色。"""
    disp_u8 = np.clip(disp_frame.astype(np.float32) * disp_mul, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(disp_u8, color_lut)
    color[disp_frame <= 0] = 0
    return color


def apply_kept_mask_to_disparity(disp_no_ground: np.ndarray, kept_mask: np.ndarray) -> np.ndarray:
    """把 Layer 2 的 kept_mask 作用到 disp_no_ground，得到最终用于显示的 disp_for_avoid。"""
    if kept_mask.shape != disp_no_ground.shape:
        kept_mask = cv2.resize(
            kept_mask.astype(np.uint8),
            (disp_no_ground.shape[1], disp_no_ground.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    disp_for_avoid = disp_no_ground.copy()
    disp_for_avoid[~kept_mask] = 0
    return disp_for_avoid


def choose_intrinsic_socket(args, calib):
    if args.align:
        mapping = {
            "rgb": dai.CameraBoardSocket.CAM_A,
            "left": dai.CameraBoardSocket.CAM_B,
            "right": dai.CameraBoardSocket.CAM_C,
        }
        return mapping[args.align_target]
    return calib.getStereoLeftCameraId()


def read_height_filter_intrinsics(device, args, depth_shape):
    """优先从设备标定读取 depth 对应相机内参；命令行给了 override 则使用 override。"""
    if args.fy_pixels > 0 and args.cy_pixels > 0:
        return float(args.fy_pixels), float(args.cy_pixels), "cli"

    calib = device.readCalibration()
    socket = choose_intrinsic_socket(args, calib)
    h, w = depth_shape
    k = np.array(calib.getCameraIntrinsics(socket, w, h), dtype=np.float32)
    fy = float(k[1, 1])
    cy = float(k[1, 2])
    return fy, cy, "device"

def parse_args():
    p = argparse.ArgumentParser(description="主流程一致的三图调试：Disparity Raw / No Ground / For Avoid")

    # 与 get_depth_claude_avoid.py 共享同一套 depth/disparity 参数定义
    add_common_depth_args(p)

    # Layer 1: ground_filter
    p.add_argument("--gnd-y-start", type=float, default=0.45)
    p.add_argument("--gnd-min-peak-count", type=int, default=5)
    p.add_argument("--gnd-smooth-kernel", type=int, default=3)
    p.add_argument("--gnd-max-peaks", type=int, default=2)
    p.add_argument("--gnd-peak-distance", type=int, default=3)
    p.add_argument(
        "--gnd-column-norm",
        choices=["subtract_percentile", "subtract_median", "subtract_min", "none"],
        default="subtract_percentile",
    )
    p.add_argument("--gnd-norm-pct", type=float, default=50.0)
    p.add_argument("--gnd-min-useful-disp", type=int, default=2)
    p.add_argument("--gnd-residual", type=float, default=2.5)
    p.add_argument("--gnd-iters", type=int, default=300)
    p.add_argument("--gnd-min-inliers", type=int, default=30)
    p.add_argument("--gnd-min-inlier-ratio", type=float, default=0.35)
    p.add_argument("--gnd-y-gap", type=float, default=0.3)
    p.add_argument("--gnd-min-slope", type=float, default=0.12)
    p.add_argument("--gnd-max-slope", type=float, default=0.5)
    p.add_argument("--gnd-min-bottom-disp", type=float, default=8.0)
    p.add_argument("--track-hold-sec", type=float, default=1.5)
    p.add_argument("--track-ema", type=float, default=0.4)
    p.add_argument("--track-slope-jump", type=float, default=0.6)
    p.add_argument("--track-intercept-jump", type=float, default=15.0)
    p.add_argument("--track-hold-min-ratio", type=float, default=0.4)
    p.add_argument("--remove-ground-tolerance", type=float, default=2.5)

    # Layer 2: height_filter（机身参考系）
    p.add_argument(
        "--camera-to-top-m", "--robot-height-m",
        dest="camera_to_top_m",
        type=float,
        default=0.20,
        help="相机到机身顶部的固定距离（米）",
    )
    p.add_argument(
        "--cam-height-m",
        type=float,
        default=0.30,
        help=argparse.SUPPRESS,  # 兼容旧命令行；当前逻辑已不再使用
    )
    p.add_argument("--cam-pitch-deg", type=float, default=0.0, help="相机相对机身俯仰角，向下看为负")
    p.add_argument("--top-tolerance-m", type=float, default=0.05)
    p.add_argument("--valid-min-mm", type=int, default=200)
    p.add_argument("--valid-max-mm", type=int, default=8000)
    p.add_argument("--fy-pixels", type=float, default=-1.0, help="可选 override；默认从设备标定读取")
    p.add_argument("--cy-pixels", type=float, default=-1.0, help="可选 override；默认从设备标定读取")

    # 显示与保存相关参数已由 add_common_depth_args(p) 提供
    return p.parse_args()


def main():
    args = parse_args()
    resolution = _RESOLUTION_MAP[args.resolution]
    median = _MEDIAN_MAP[args.median]
    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    if args.aggressive:
        hole_fill, spatial_iter, use_alpha_delta = max(args.hole_fill, 4), max(args.spatial_iter, 2), True
    else:
        hole_fill, spatial_iter, use_alpha_delta = args.hole_fill, args.spatial_iter, False

    pipeline = create_depth_pipeline(
        mono_resolution=resolution,
        confidence_threshold=args.confidence,
        median_filter=median,
        enable_lrc=not args.no_lrc,
        enable_subpixel=args.subpixel,
        enable_extended=args.extended,
        enable_rgb_depth_align=args.align,
        align_target=args.align_target,
        bilateral_sigma=args.sigma,
        lrc_threshold=args.lrc_threshold,
        spatial_hole_fill=hole_fill,
        spatial_iterations=spatial_iter,
        use_spatial_alpha_delta=use_alpha_delta,
        enable_threshold_filter=not args.no_threshold,
    )

    max_disparity = compute_max_disparity(args.subpixel, args.extended)
    subpixel_scale = 32 if args.subpixel else 1
    int_max_disp = max_disparity // subpixel_scale
    cv_cm_id = getattr(cv2, f"COLORMAP_{args.colormap}")
    color_lut = build_colormap_with_zero_black(cv_cm_id)
    disp_mul = 255.0 / max(float(max_disparity), 1.0)

    tracker = GroundTracker(
        hold_seconds=args.track_hold_sec,
        ema_alpha=args.track_ema,
        max_slope_jump_ratio=args.track_slope_jump,
        max_intercept_jump=args.track_intercept_jump,
        hold_min_inlier_ratio=args.track_hold_min_ratio,
    )
    ground_params = {
        "y_start_ratio": args.gnd_y_start,
        "min_useful_disp": args.gnd_min_useful_disp,
        "min_peak_count": args.gnd_min_peak_count,
        "smooth_kernel": args.gnd_smooth_kernel,
        "max_peaks_per_row": args.gnd_max_peaks,
        "peak_min_distance": args.gnd_peak_distance,
        "column_normalize": args.gnd_column_norm,
        "column_norm_percentile": args.gnd_norm_pct,
        "n_iterations": args.gnd_iters,
        "residual_threshold": args.gnd_residual,
        "min_inliers": args.gnd_min_inliers,
        "min_inlier_ratio": args.gnd_min_inlier_ratio,
        "min_y_gap_ratio": args.gnd_y_gap,
        "min_slope": args.gnd_min_slope,
        "max_slope": args.gnd_max_slope,
        "min_bottom_disp": args.gnd_min_bottom_disp,
    }

    print("=" * 60)
    print(" Perception Debug Viewer")
    print("=" * 60)
    print(f" 深度链路: 复用 get_depth_claude_avoid.py pipeline")
    print(f" 分辨率: {args.resolution}  subpixel={args.subpixel}  extended={args.extended}  align={args.align}")
    print(f" Layer1: tol={args.remove_ground_tolerance}  hold={args.track_hold_sec}s  min_ratio={args.gnd_min_inlier_ratio}")
    print(f" Layer2: camera_to_top={args.camera_to_top_m}m  pitch={args.cam_pitch_deg}deg")
    print(" 窗口: Disparity Raw / Disparity No Ground / Disparity For Avoid")
    print(" 按键: [q]退出 [s]保存 [r]重置地面跟踪")
    print()

    device_ctx = create_device_context(pipeline, args.usb_speed)

    with device_ctx as device:
        print(f"USB: {device.getUsbSpeed()}")
        depth_q = device.getOutputQueue("depth", maxSize=4, blocking=False)
        disp_q = device.getOutputQueue("disparity", maxSize=4, blocking=False)

        cv2.namedWindow("Disparity Raw", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Disparity No Ground", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Disparity For Avoid", cv2.WINDOW_NORMAL)

        fps_count, fps_start, fps = 0, time.time(), 0.0
        latest_depth, latest_disp = None, None
        last_print_t = 0.0
        first_intrinsics = False
        height_filter = None
        latest_debug_pack = None

        while True:
            # 原始 depth / disparity 输入链路与主文件共享。
            latest_depth, latest_disp = update_latest_stereo_frames(
                depth_q, disp_q, latest_depth, latest_disp
            )
            if latest_disp is None or latest_depth is None:
                continue

            if not first_intrinsics:
                # Layer 2 默认从设备标定中读取 fy / cy；只有显式传参才覆盖。
                fy_pixels, cy_pixels, source = read_height_filter_intrinsics(device, args, latest_depth.shape)
                hf_cfg = HeightFilterConfig(
                    camera_to_top_m=args.camera_to_top_m,
                    fy_pixels=fy_pixels,
                    cy_pixels=cy_pixels,
                    cam_pitch_deg=args.cam_pitch_deg,
                    valid_min_mm=args.valid_min_mm,
                    valid_max_mm=args.valid_max_mm,
                    top_tolerance_m=args.top_tolerance_m,
                )
                height_filter = HeightFilter(hf_cfg)
                print(f"[HeightFilter] fy={fy_pixels:.3f} cy={cy_pixels:.3f}  source={source}")
                first_intrinsics = True

            fps_count += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                fps = fps_count / elapsed
                fps_count, fps_start = 0, time.time()

            # Layer 1：在 disparity 域识别地面，并同步给出 disparity / depth 两侧的去地面结果。
            ground_result = segment_ground(
                latest_disp,
                latest_depth,
                tracker=tracker,
                max_disp=int_max_disp,
                subpixel_scale=subpixel_scale,
                params=ground_params,
                ground_tolerance=args.remove_ground_tolerance,
            )
            depth_no_ground = ground_result["depth_no_ground"]

            # Layer 2：在 depth_mm 域里按相对机身顶部做过滤。
            height_result = height_filter.filter(depth_no_ground)
            disp_no_ground = ground_result["disp_no_ground"]

            # 三图统一显示真实 disparity：
            #   Raw          = latest_disp
            #   No Ground    = disp_no_ground
            #   For Avoid    = disp_no_ground 再套 kept_mask
            disp_for_avoid = apply_kept_mask_to_disparity(disp_no_ground, height_result["kept_mask"])

            latest_debug_pack = {
                "raw_depth": latest_depth,
                "disp": latest_disp,
                "disp_no_ground": disp_no_ground,
                "disp_for_avoid": disp_for_avoid,
                "ground_result": ground_result,
                "height_result": height_result,
            }

            disp_raw_color = colorize_disparity(latest_disp, disp_mul, color_lut)
            disp_nogr_color = colorize_disparity(disp_no_ground, disp_mul, color_lut)
            disp_avoid_color = colorize_disparity(disp_for_avoid, disp_mul, color_lut)

            raw_shown = fit_to_window(disp_raw_color, args.window_width)
            nogr_shown = fit_to_window(disp_nogr_color, args.window_width)
            avoid_shown = fit_to_window(disp_avoid_color, args.window_width)

            raw_zero_count = int((latest_depth == 0).sum())
            l1_removed = int((depth_no_ground == 0).sum()) - raw_zero_count
            l1_removed_ratio = max(0.0, l1_removed / latest_depth.size)
            l2_kept_ratio = float(height_result["kept_mask"].sum()) / max(height_result["kept_mask"].size, 1)

            cv2.putText(raw_shown, f"Disparity Raw  FPS:{fps:.1f}", (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(raw_shown, "真实 disparity 显示；输入链路与 get_depth_claude_avoid.py 共享", (10, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(nogr_shown,
                        f"Disparity No Ground  state={ground_result['tracked'].state}  removed={l1_removed_ratio*100:.1f}%",
                        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(avoid_shown,
                        f"Disparity For Avoid  kept={l2_kept_ratio*100:.1f}%  range=[{args.valid_min_mm},{args.valid_max_mm}]mm",
                        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            now = time.time()
            if now - last_print_t > 1.0:
                print(
                    f"[{time.strftime('%H:%M:%S')}] "
                    f"Ground={ground_result['tracked'].state} "
                    f"L1_removed={l1_removed_ratio*100:.1f}% "
                    f"L2_kept={l2_kept_ratio*100:.1f}%"
                )
                last_print_t = now

            cv2.imshow("Disparity Raw", raw_shown)
            cv2.imshow("Disparity No Ground", nogr_shown)
            cv2.imshow("Disparity For Avoid", avoid_shown)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                tracker.reset()
                print("[GroundTracker] 已重置")
            if key == ord("s") and latest_debug_pack is not None:
                if save_dir is None:
                    print("[!] 未设置 --save-dir，跳过保存。")
                else:
                    stamp = time.strftime("%Y%m%d_%H%M%S")
                    # 保存只保留最核心的 disparity / depth 中间结果；
                    # raw_depth 与 height_map 不再单独保存，避免目录过乱。
                    np.save(save_dir / f"disp_raw_{stamp}.npy", latest_debug_pack["disp"])
                    np.save(save_dir / f"disp_no_ground_{stamp}.npy", latest_debug_pack["disp_no_ground"])
                    np.save(save_dir / f"disp_for_avoid_{stamp}.npy", latest_debug_pack["disp_for_avoid"])
                    np.save(save_dir / f"depth_no_ground_{stamp}.npy", latest_debug_pack["ground_result"]["depth_no_ground"])
                    np.save(save_dir / f"depth_for_avoid_{stamp}.npy", height_result["depth_for_avoid"])
                    print(f"[✓] 已保存到 {save_dir}（{stamp}）")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
