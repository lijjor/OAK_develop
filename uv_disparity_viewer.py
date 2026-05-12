"""
uv_disparity.py — U/V-Disparity 观察 + 地面拟合 v3

v3 重大改动
============
1. 默认 min_slope 从 0.005 大幅提高到 0.08（关键）
   - 真实地面斜率约 0.10-0.30，远低于此的"线"是墙体/天花板
2. 新增 --cam-height-m 参数，用于在终端提示理论斜率范围
3. 新增 --gnd-min-bottom-disp（地面合理性验证）
4. 收紧 inlier 比例下限
"""
import argparse
import time
from pathlib import Path

import cv2
import depthai as dai
import numpy as np

from ground_fitter import (
    extract_ground_candidates, fit_ground_line_ransac,
    GroundTracker, TrackedGround, estimate_ground_slope_range,
)


# ─────────────────────────────────────────────────────────────────────────────
# UV-Disparity 计算
# ─────────────────────────────────────────────────────────────────────────────
def compute_v_disparity(disp_int: np.ndarray, max_disp: int) -> np.ndarray:
    h, _ = disp_int.shape
    v_hist = np.zeros((h, max_disp + 1), dtype=np.uint32)
    valid = disp_int > 0
    for y in range(h):
        row = disp_int[y][valid[y]]
        if row.size:
            v_hist[y] = np.bincount(row, minlength=max_disp + 1)[:max_disp + 1]
    return v_hist


def compute_u_disparity(disp_int: np.ndarray, max_disp: int) -> np.ndarray:
    _, w = disp_int.shape
    u_hist = np.zeros((max_disp + 1, w), dtype=np.uint32)
    valid = disp_int > 0
    for x in range(w):
        col = disp_int[:, x][valid[:, x]]
        if col.size:
            u_hist[:, x] = np.bincount(col, minlength=max_disp + 1)[:max_disp + 1]
    return u_hist


def disparity_to_int(disp_frame, max_disp, subpixel_scale=1):
    if subpixel_scale > 1:
        disp_int = disp_frame.astype(np.int32) // subpixel_scale
    else:
        disp_int = disp_frame.astype(np.int32)
    return np.clip(disp_int, 0, max_disp)


def render_hist(hist, axis, target_size, log_scale=True, colormap=cv2.COLORMAP_HOT):
    img = hist.astype(np.float32)
    if log_scale:
        img = np.log1p(img)
    img = img / max(img.max(), 1e-6)
    img_u8 = (img * 255).astype(np.uint8)
    colored = cv2.applyColorMap(img_u8, colormap)
    if target_size is not None:
        colored = cv2.resize(colored, target_size, interpolation=cv2.INTER_NEAREST)
    label = "V-Disparity (X=disp, Y=row)" if axis == "v" else "U-Disparity (X=col, Y=disp)"
    h, w = colored.shape[:2]
    cv2.rectangle(colored, (0, 0), (w, 18), (0, 0, 0), -1)
    cv2.putText(colored, label, (5, 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return colored


# ─────────────────────────────────────────────────────────────────────────────
# 地面拟合可视化
# ─────────────────────────────────────────────────────────────────────────────
STATE_LINE_COLOR = {
    "LOCKED": (255, 255, 0),
    "HELD":   (0, 165, 255),
    "LOST":   (0, 0, 255),
}


def overlay_ground_line(v_disp_img, v_hist_shape, candidates, tracked, inliers_mask=None):
    disp_h, max_disp_plus_1 = v_hist_shape
    img_h, img_w = v_disp_img.shape[:2]
    sx = img_w / max_disp_plus_1
    sy = img_h / disp_h

    if candidates is not None and len(candidates) > 0:
        for i, (y, d) in enumerate(candidates):
            color = (0, 255, 0) if (inliers_mask is not None and inliers_mask[i]) else (0, 255, 255)
            cx = int(d * sx + sx / 2)
            cy = int(y * sy + sy / 2)
            cv2.circle(v_disp_img, (cx, cy), 1, color, -1, cv2.LINE_AA)

    line = tracked.line
    state = tracked.state
    line_color = STATE_LINE_COLOR.get(state, (255, 255, 255))

    if line is not None and state in ("LOCKED", "HELD"):
        y_top, y_bot = 0, disp_h - 1
        d_top = line.slope * y_top + line.intercept
        d_bot = line.slope * y_bot + line.intercept
        pt1 = (int(d_top * sx + sx / 2), int(y_top * sy + sy / 2))
        pt2 = (int(d_bot * sx + sx / 2), int(y_bot * sy + sy / 2))
        cv2.line(v_disp_img, pt1, pt2, line_color, 2, cv2.LINE_AA)

    cv2.rectangle(v_disp_img, (0, img_h - 38), (img_w, img_h), (0, 0, 0), -1)
    n_cand = 0 if candidates is None else len(candidates)
    if line is not None:
        row1 = f"[{state}] slope={line.slope:.3f} intercept={line.intercept:.1f}"
        if state == "HELD":
            row1 += f"  age={tracked.age_seconds:.1f}s"
        elif state == "LOCKED":
            row1 += f"  inl={line.num_inliers}/{line.num_candidates}({line.inlier_ratio*100:.0f}%)"
    else:
        row1 = f"[{state}] no ground line  (cands={n_cand})"

    extra_marks = []
    if tracked.rejected_outlier:
        extra_marks.append("OUTLIER_REJECTED")
    if not tracked.raw_fit_succeeded and state != "LOST":
        extra_marks.append("raw_fail")
    row2 = " ".join(extra_marks)

    cv2.putText(v_disp_img, row1, (5, img_h - 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, line_color, 1, cv2.LINE_AA)
    if row2:
        cv2.putText(v_disp_img, row2, (5, img_h - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)


def overlay_ground_on_disparity(disp_color, tracked, depth_h, thickness=1):
    line = tracked.line
    if line is None or abs(line.slope) < 1e-6:
        return
    color = STATE_LINE_COLOR.get(tracked.state, (255, 255, 255))
    img_h, img_w = disp_color.shape[:2]
    sy = img_h / depth_h
    for d_mark in (5, 15, 25):
        y_pred = (d_mark - line.intercept) / line.slope
        if 0 <= y_pred < depth_h:
            y_show = int(y_pred * sy)
            cv2.line(disp_color, (0, y_show), (img_w - 1, y_show), color, thickness, cv2.LINE_AA)
            cv2.putText(disp_color, f"d={d_mark}",
                        (img_w - 60, y_show - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline 构建（★ 与主代码完全一致）
# ─────────────────────────────────────────────────────────────────────────────
def create_depth_pipeline_aligned_with_main(
    mono_resolution,
    confidence_threshold=240,
    median_filter=dai.MedianFilter.KERNEL_7x7,
    enable_lrc=True,
    enable_subpixel=False,
    enable_extended=False,
    enable_rgb_depth_align=False,
    align_target="rgb",
    bilateral_sigma=0,
    lrc_threshold=10,
    spatial_hole_fill=2,
    spatial_iterations=1,
    use_spatial_alpha_delta=False,
    enable_threshold_filter=True,
    threshold_min_mm=200,
    threshold_max_mm=10000,
):
    pipeline = dai.Pipeline()

    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_right = pipeline.create(dai.node.MonoCamera)
    mono_left.setResolution(mono_resolution)
    mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    mono_right.setResolution(mono_resolution)
    mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

    if enable_rgb_depth_align and align_target == "rgb":
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.initialConfig.setConfidenceThreshold(confidence_threshold)
    if median_filter is not None:
        stereo.initialConfig.setMedianFilter(median_filter)
    stereo.setLeftRightCheck(enable_lrc)
    stereo.initialConfig.setLeftRightCheckThreshold(lrc_threshold)
    stereo.setExtendedDisparity(enable_extended)
    stereo.setSubpixel(enable_subpixel)

    cfg = stereo.initialConfig.get()
    cfg.postProcessing.spatialFilter.enable = True
    cfg.postProcessing.spatialFilter.holeFillingRadius = int(spatial_hole_fill)
    cfg.postProcessing.spatialFilter.numIterations = int(spatial_iterations)
    if use_spatial_alpha_delta:
        try:
            cfg.postProcessing.spatialFilter.alpha = 0.5
            cfg.postProcessing.spatialFilter.delta = 20
        except AttributeError:
            pass
    cfg.postProcessing.speckleFilter.enable = True
    cfg.postProcessing.speckleFilter.speckleRange = 50
    if enable_threshold_filter:
        cfg.postProcessing.thresholdFilter.minRange = int(threshold_min_mm)
        cfg.postProcessing.thresholdFilter.maxRange = int(threshold_max_mm)
    cfg.postProcessing.bilateralSigmaValue = bilateral_sigma
    stereo.initialConfig.set(cfg)
    stereo.setRectifyEdgeFillColor(0)

    if enable_rgb_depth_align:
        if align_target == "left":
            stereo.setDepthAlign(dai.CameraBoardSocket.CAM_B)
        elif align_target == "right":
            stereo.setDepthAlign(dai.CameraBoardSocket.CAM_C)
        else:
            stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

    xout_disp = pipeline.create(dai.node.XLinkOut)
    xout_disp.setStreamName("disparity")
    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)
    stereo.disparity.link(xout_disp.input)
    return pipeline


# ─────────────────────────────────────────────────────────────────────────────
def build_colormap_with_zero_black(cv_colormap_id=cv2.COLORMAP_JET):
    color_map = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv_colormap_id)
    color_map[0] = [0, 0, 0]
    return color_map


def fit_to_window(img, target_w):
    if target_w <= 0:
        return img
    h, w = img.shape[:2]
    if w == target_w:
        return img
    scale = target_w / w
    return cv2.resize(img, (target_w, int(round(h * scale))),
                      interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)


_RESOLUTION_MAP = {
    "400p": dai.MonoCameraProperties.SensorResolution.THE_400_P,
    "480p": dai.MonoCameraProperties.SensorResolution.THE_480_P,
    "720p": dai.MonoCameraProperties.SensorResolution.THE_720_P,
    "800p": dai.MonoCameraProperties.SensorResolution.THE_800_P,
}
_MEDIAN_MAP = {
    "off": dai.MedianFilter.MEDIAN_OFF,
    "3":   dai.MedianFilter.KERNEL_3x3,
    "5":   dai.MedianFilter.KERNEL_5x5,
    "7":   dai.MedianFilter.KERNEL_7x7,
}


def parse_args():
    p = argparse.ArgumentParser(description="UV-Disparity + 地面 RANSAC v3")
    p.add_argument("--resolution", choices=list(_RESOLUTION_MAP.keys()), default="400p")
    p.add_argument("--confidence", type=int, default=240)
    p.add_argument("--median", choices=list(_MEDIAN_MAP.keys()), default="7")
    p.add_argument("--sigma", type=int, default=0)
    p.add_argument("--lrc-threshold", type=int, default=10)
    p.add_argument("--no-lrc", action="store_true")
    p.add_argument("--subpixel", action="store_true")
    p.add_argument("--extended", action="store_true")
    p.add_argument("--align", action="store_true")
    p.add_argument("--align-target", choices=["rgb", "left", "right"], default="rgb")
    p.add_argument("--hole-fill", type=int, default=2)
    p.add_argument("--spatial-iter", type=int, default=1)
    p.add_argument("--aggressive", action="store_true")
    p.add_argument("--no-threshold", action="store_true")

    p.add_argument("--colormap", choices=["JET", "TURBO", "HOT", "VIRIDIS"], default="JET")
    p.add_argument("--window-width", type=int, default=720)
    p.add_argument("--vdisp-width", type=int, default=400)
    p.add_argument("--udisp-height", type=int, default=300)
    p.add_argument("--no-log", action="store_true")
    p.add_argument("--save-dir", type=str, default="")
    p.add_argument("--usb-speed", choices=["auto", "usb2", "usb3"], default="auto")

    # ★ 相机安装信息（用于辅助计算理论斜率）
    p.add_argument("--cam-height-m", type=float, default=0.30,
                   help="相机离地高度（米），用于估算地面斜率范围")

    # 候选点提取
    p.add_argument("--no-ground-fit", action="store_true")
    p.add_argument("--gnd-y-start", type=float, default=0.45)
    p.add_argument("--gnd-min-peak-count", type=int, default=5)
    p.add_argument("--gnd-smooth-kernel", type=int, default=3)
    p.add_argument("--gnd-max-peaks", type=int, default=2,
                   help="每行最多取几个峰值（v3 默认 2）")
    p.add_argument("--gnd-peak-distance", type=int, default=3)

    # RANSAC（v3 严格默认）
    p.add_argument("--gnd-residual", type=float, default=1.5)
    p.add_argument("--gnd-iters", type=int, default=300)
    p.add_argument("--gnd-min-inliers", type=int, default=30)
    p.add_argument("--gnd-min-inlier-ratio", type=float, default=0.25,
                   help="inlier 比例下限（默认 0.25）")
    p.add_argument("--gnd-y-gap", type=float, default=0.3)
    p.add_argument("--gnd-min-slope", type=float, default=0.08,
                   help="地面斜率下限（v3 默认 0.08，大幅高于以前的 0.005）")
    p.add_argument("--gnd-max-slope", type=float, default=0.5,
                   help="地面斜率上限（默认 0.5）")
    p.add_argument("--gnd-min-bottom-disp", type=float, default=8.0,
                   help="拟合直线在最底行的最小预测视差（地面应该近=disp 大）")

    # 时序平滑
    p.add_argument("--track-hold-sec", type=float, default=2.0)
    p.add_argument("--track-ema", type=float, default=0.4)
    p.add_argument("--track-slope-jump", type=float, default=0.6)
    p.add_argument("--track-intercept-jump", type=float, default=15.0)
    return p.parse_args()


def main():
    args = parse_args()
    resolution = _RESOLUTION_MAP[args.resolution]
    median = _MEDIAN_MAP[args.median]
    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    if args.subpixel and args.extended:
        print("[!] subpixel 与 extended 互斥，自动关闭 extended")
        args.extended = False

    if args.aggressive:
        hole_fill, spatial_iter, use_alpha_delta = max(args.hole_fill, 4), max(args.spatial_iter, 2), True
    else:
        hole_fill, spatial_iter, use_alpha_delta = args.hole_fill, args.spatial_iter, False

    pipeline = create_depth_pipeline_aligned_with_main(
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

    max_disparity = 95
    if args.extended:
        max_disparity *= 2
    if args.subpixel:
        max_disparity *= 32
    subpixel_scale = 32 if args.subpixel else 1
    int_max_disp = max_disparity // subpixel_scale

    cv_cm_id = getattr(cv2, f"COLORMAP_{args.colormap}")
    color_lut = build_colormap_with_zero_black(cv_cm_id)
    disp_color_mul = 255.0 / max_disparity

    # 估算理论斜率范围
    slope_lo, slope_hi = estimate_ground_slope_range(
        fy_pixels=453.0, baseline_m=0.075,
        camera_height_m=args.cam_height_m, height_h=400,
    )

    tracker = GroundTracker(
        hold_seconds=args.track_hold_sec,
        ema_alpha=args.track_ema,
        max_slope_jump_ratio=args.track_slope_jump,
        max_intercept_jump=args.track_intercept_jump,
    )

    print("=" * 60)
    print(" UV-Disparity + 地面 RANSAC v3")
    print("=" * 60)
    print(f" 分辨率: {args.resolution}    相机高: {args.cam_height_m}m")
    print(f" 整数视差范围: 0..{int_max_disp}    subpixel_scale={subpixel_scale}")
    print(f" 理论地面斜率: 约 {slope_lo:.3f} ~ {slope_hi:.3f}")
    print(f" 拟合 slope 范围: [{args.gnd_min_slope}, {args.gnd_max_slope}]")
    print(f" min_inlier_ratio: {args.gnd_min_inlier_ratio}   "
          f"min_bottom_disp: {args.gnd_min_bottom_disp}")
    print(" 按键: [q]退出 [s]保存 [l]切换log [g]切换拟合 [r]重置跟踪")
    print()

    if args.usb_speed == "usb2":
        device_ctx = dai.Device(pipeline, dai.UsbSpeed.HIGH)
    elif args.usb_speed == "usb3":
        device_ctx = dai.Device(pipeline, dai.UsbSpeed.SUPER)
    else:
        device_ctx = dai.Device(pipeline)

    with device_ctx as device:
        print(f"USB: {device.getUsbSpeed()}")
        disp_q = device.getOutputQueue("disparity", maxSize=4, blocking=False)

        cv2.namedWindow("Disparity", cv2.WINDOW_NORMAL)
        cv2.namedWindow("V-Disparity", cv2.WINDOW_NORMAL)
        cv2.namedWindow("U-Disparity", cv2.WINDOW_NORMAL)

        fps_count, fps_start, fps = 0, time.time(), 0.0
        latest_disp = None
        first_disp = first_v = first_u = False
        log_scale = not args.no_log
        vdisp_width = args.vdisp_width
        udisp_height = args.udisp_height
        do_ground_fit = not args.no_ground_fit
        last_state_print_t = 0.0
        last_state_str = ""

        while True:
            in_disp = disp_q.tryGet()
            if in_disp is not None:
                latest_disp = in_disp.getFrame()
            if latest_disp is None:
                continue

            fps_count += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                fps = fps_count / elapsed
                fps_count, fps_start = 0, time.time()

            disp_u8 = np.clip(latest_disp * disp_color_mul, 0, 255).astype(np.uint8)
            disp_color = cv2.applyColorMap(disp_u8, color_lut)
            disp_shown = fit_to_window(disp_color, args.window_width)

            disp_int = disparity_to_int(latest_disp, int_max_disp, subpixel_scale)
            v_hist = compute_v_disparity(disp_int, int_max_disp)
            u_hist = compute_u_disparity(disp_int, int_max_disp)
            v_img = render_hist(v_hist, "v",
                                target_size=(vdisp_width, disp_shown.shape[0]),
                                log_scale=log_scale)
            u_img = render_hist(u_hist, "u",
                                target_size=(disp_shown.shape[1], udisp_height),
                                log_scale=log_scale)

            tracked = TrackedGround(None, "LOST", float("inf"), False, False)
            candidates = np.empty((0, 2), np.float32)
            inliers_mask = None

            if do_ground_fit:
                candidates = extract_ground_candidates(
                    v_hist,
                    y_start_ratio=args.gnd_y_start,
                    min_peak_count=args.gnd_min_peak_count,
                    smooth_kernel=args.gnd_smooth_kernel,
                    max_peaks_per_row=args.gnd_max_peaks,
                    peak_min_distance=args.gnd_peak_distance,
                )
                raw_line = None
                if len(candidates) >= args.gnd_min_inliers // 2:
                    raw_line = fit_ground_line_ransac(
                        candidates,
                        n_iterations=args.gnd_iters,
                        residual_threshold=args.gnd_residual,
                        min_inliers=args.gnd_min_inliers,
                        min_inlier_ratio=args.gnd_min_inlier_ratio,
                        min_slope=args.gnd_min_slope,
                        max_slope=args.gnd_max_slope,
                        min_y_gap_ratio=args.gnd_y_gap,
                        min_bottom_disp=args.gnd_min_bottom_disp,
                    )
                tracked = tracker.update(raw_line)

                if raw_line is not None and len(candidates) > 0:
                    residuals = np.abs(
                        candidates[:, 1] - (raw_line.slope * candidates[:, 0] + raw_line.intercept)
                    )
                    inliers_mask = residuals <= args.gnd_residual

                overlay_ground_line(v_img, v_hist.shape, candidates, tracked, inliers_mask)
                overlay_ground_on_disparity(disp_shown, tracked, depth_h=v_hist.shape[0])

                now = time.time()
                state_str = f"{tracked.state}"
                if tracked.line is not None:
                    state_str += f"(slope={tracked.line.slope:.3f}, inter={tracked.line.intercept:.1f}"
                    if tracked.state == "HELD":
                        state_str += f", age={tracked.age_seconds:.1f}s"
                    state_str += ")"
                state_str += f"  cands={len(candidates)}"
                if tracked.rejected_outlier:
                    state_str += " [OUTLIER_REJECTED]"

                if state_str != last_state_str or (now - last_state_print_t) > 2.0:
                    print(f"[{time.strftime('%H:%M:%S')}] ground={state_str}")
                    last_state_str = state_str
                    last_state_print_t = now

            cv2.putText(disp_shown, f"FPS:{fps:.1f}", (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

            cv2.imshow("Disparity",   disp_shown)
            cv2.imshow("V-Disparity", v_img)
            cv2.imshow("U-Disparity", u_img)

            if not first_disp:
                cv2.resizeWindow("Disparity", disp_shown.shape[1], disp_shown.shape[0]); first_disp = True
            if not first_v:
                cv2.resizeWindow("V-Disparity", v_img.shape[1], v_img.shape[0]); first_v = True
            if not first_u:
                cv2.resizeWindow("U-Disparity", u_img.shape[1], u_img.shape[0]); first_u = True

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("l"):
                log_scale = not log_scale
                print(f"[显示] log 拉伸: {'开' if log_scale else '关'}")
            elif key == ord("g"):
                do_ground_fit = not do_ground_fit
                print(f"[显示] 地面拟合: {'开' if do_ground_fit else '关'}")
            elif key == ord("r"):
                tracker.reset()
                print("[跟踪] 已重置")
            elif key in (ord("+"), ord("=")):
                vdisp_width = min(vdisp_width + 50, 800)
                udisp_height = min(udisp_height + 30, 500)
                print(f"[显示] V宽={vdisp_width}  U高={udisp_height}")
            elif key in (ord("-"), ord("_")):
                vdisp_width = max(vdisp_width - 50, 150)
                udisp_height = max(udisp_height - 30, 100)
                print(f"[显示] V宽={vdisp_width}  U高={udisp_height}")
            elif key == ord("s"):
                if save_dir is None:
                    print("[!] 未设置 --save-dir，跳过保存。")
                else:
                    stamp = time.strftime("%Y%m%d_%H%M%S")
                    np.save(save_dir / f"disp_{stamp}.npy", latest_disp)
                    cv2.imwrite(str(save_dir / f"disp_{stamp}.png"), disp_shown)
                    cv2.imwrite(str(save_dir / f"vdisp_{stamp}.png"), v_img)
                    cv2.imwrite(str(save_dir / f"udisp_{stamp}.png"), u_img)
                    if tracked.line is not None:
                        with open(save_dir / f"ground_{stamp}.txt", "w") as f:
                            f.write(f"state={tracked.state}\n")
                            f.write(f"slope={tracked.line.slope:.6f}\n")
                            f.write(f"intercept={tracked.line.intercept:.6f}\n")
                            f.write(f"age_seconds={tracked.age_seconds:.3f}\n")
                    print(f"[✓] 已保存到 {save_dir}（{stamp}）")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
