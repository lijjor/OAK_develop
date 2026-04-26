"""
get_depth_claude.py — OAK 深度图采集脚本（与 depthai_demo 显示效果对齐）

【本次调整】把后处理改回温和参数，避免大块黑色：
  - holeFillingRadius: 5 → 2（demo 默认值）
  - numIterations: 2 → 1（demo 默认值）
  - 不设置 spatial.alpha/delta（让 SDK 用内置默认，避免激进平滑）
  - 保留 thresholdFilter（无副作用，只切掉无效远点）
  - 保留 RGB 对齐默认关闭（这是上版关键修复）

总体思路回归 demo 默认配置，只保留两个明确无副作用的改动。
如果还想填洞更激进，提供 --aggressive 开关一键切换。
"""
import argparse
import time
from pathlib import Path

import cv2
import depthai as dai
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────
def create_depth_pipeline(
    mono_resolution,
    confidence_threshold=240,
    median_filter=dai.MedianFilter.KERNEL_7x7,
    enable_lrc=True,
    enable_subpixel=False,
    enable_extended=False,
    enable_rgb_depth_align=False,    # 默认关（关键修复）
    align_target="rgb",
    bilateral_sigma=0,
    lrc_threshold=10,
    # ★ 后处理参数（温和模式默认值）
    spatial_hole_fill=2,             # demo 默认 2
    spatial_iterations=1,            # demo 默认 1
    use_spatial_alpha_delta=False,   # 默认不设置 alpha/delta
    enable_threshold_filter=True,
    threshold_min_mm=200,
    threshold_max_mm=10000,
):
    pipeline = dai.Pipeline()

    # ── 双目相机 ─────────────────────────────────────────────────────
    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_right = pipeline.create(dai.node.MonoCamera)
    mono_left.setResolution(mono_resolution)
    mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    mono_right.setResolution(mono_resolution)
    mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

    # ── RGB 相机（仅当开启对齐时） ───────────────────────────────────
    if enable_rgb_depth_align and align_target == "rgb":
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    # ── StereoDepth ──────────────────────────────────────────────────
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)

    stereo.initialConfig.setConfidenceThreshold(confidence_threshold)
    if median_filter is not None:
        stereo.initialConfig.setMedianFilter(median_filter)
    stereo.setLeftRightCheck(enable_lrc)
    stereo.initialConfig.setLeftRightCheckThreshold(lrc_threshold)
    stereo.setExtendedDisparity(enable_extended)
    stereo.setSubpixel(enable_subpixel)

    # ── 后处理 ────────────────────────────────────────────────────────
    cfg = stereo.initialConfig.get()

    # Spatial Filter（温和参数）
    cfg.postProcessing.spatialFilter.enable = True
    cfg.postProcessing.spatialFilter.holeFillingRadius = int(spatial_hole_fill)
    cfg.postProcessing.spatialFilter.numIterations = int(spatial_iterations)
    if use_spatial_alpha_delta:
        try:
            cfg.postProcessing.spatialFilter.alpha = 0.5
            cfg.postProcessing.spatialFilter.delta = 20
        except AttributeError:
            pass

    # Speckle Filter
    cfg.postProcessing.speckleFilter.enable = True
    cfg.postProcessing.speckleFilter.speckleRange = 50

    # Threshold Filter（无副作用，仅切掉远近无效点）
    if enable_threshold_filter:
        cfg.postProcessing.thresholdFilter.minRange = int(threshold_min_mm)
        cfg.postProcessing.thresholdFilter.maxRange = int(threshold_max_mm)

    cfg.postProcessing.bilateralSigmaValue = bilateral_sigma
    stereo.initialConfig.set(cfg)

    stereo.setRectifyEdgeFillColor(0)

    # 对齐目标
    if enable_rgb_depth_align:
        if align_target == "left":
            stereo.setDepthAlign(dai.CameraBoardSocket.CAM_B)
        elif align_target == "right":
            stereo.setDepthAlign(dai.CameraBoardSocket.CAM_C)
        else:
            stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

    # ── 输出 ──────────────────────────────────────────────────────────
    xout_depth = pipeline.create(dai.node.XLinkOut)
    xout_depth.setStreamName("depth")
    xout_disp = pipeline.create(dai.node.XLinkOut)
    xout_disp.setStreamName("disparity")

    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)
    stereo.depth.link(xout_depth.input)
    stereo.disparity.link(xout_disp.input)

    return pipeline


# ─────────────────────────────────────────────────────────────────────────────
# 显示工具
# ─────────────────────────────────────────────────────────────────────────────
def compute_max_disparity(enable_subpixel, enable_extended):
    max_disp = 95
    if enable_extended:
        max_disp *= 2
    if enable_subpixel:
        max_disp *= 32
    return max_disp


def build_colormap_with_zero_black(cv_colormap_id=cv2.COLORMAP_JET):
    color_map = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv_colormap_id)
    color_map[0] = [0, 0, 0]
    return color_map


def fit_to_window(img, target_w):
    h, w = img.shape[:2]
    if w == target_w:
        return img
    scale = target_w / w
    return cv2.resize(img, (target_w, int(round(h * scale))),
                      interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)


# ─────────────────────────────────────────────────────────────────────────────
# 参数与主循环
# ─────────────────────────────────────────────────────────────────────────────
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
    p = argparse.ArgumentParser(description="OAK 深度采集（对齐 demo 显示）")
    p.add_argument("--resolution", choices=list(_RESOLUTION_MAP.keys()), default="400p")
    p.add_argument("--confidence", type=int, default=240)
    p.add_argument("--median", choices=list(_MEDIAN_MAP.keys()), default="7")
    p.add_argument("--sigma", type=int, default=0)
    p.add_argument("--lrc-threshold", type=int, default=10)
    p.add_argument("--no-lrc", action="store_true")
    p.add_argument("--subpixel", action="store_true")
    p.add_argument("--extended", action="store_true")
    p.add_argument("--align", action="store_true",
                   help="开启 RGB-Depth 对齐（默认关，与 demo 默认一致）")
    p.add_argument("--align-target", choices=["rgb", "left", "right"], default="rgb")

    # ★ 后处理调节（默认温和，提供 --aggressive 开关）
    p.add_argument("--hole-fill", type=int, default=2,
                   help="空间滤波填洞半径 0-5（默认 2，过大会导致大块变黑）")
    p.add_argument("--spatial-iter", type=int, default=1,
                   help="空间滤波迭代次数 1-5（默认 1）")
    p.add_argument("--aggressive", action="store_true",
                   help="一键开启激进填洞 (hole-fill=4, iter=2, alpha/delta on)")
    p.add_argument("--no-threshold", action="store_true", help="关闭 200-10000mm 阈值过滤")

    p.add_argument("--colormap", choices=["JET", "TURBO", "HOT", "VIRIDIS"], default="JET")
    p.add_argument("--window-width", type=int, default=720)
    p.add_argument("--show", choices=["disparity", "depth", "both"], default="disparity")
    p.add_argument("--max-display-mm", type=int, default=8000)
    p.add_argument("--save-dir", type=str, default="")
    p.add_argument("--usb-speed", choices=["auto", "usb2", "usb3"], default="auto")
    return p.parse_args()


def main():
    args = parse_args()
    resolution = _RESOLUTION_MAP[args.resolution]
    median = _MEDIAN_MAP[args.median]
    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    # 处理 aggressive 开关
    if args.aggressive:
        hole_fill = max(args.hole_fill, 4)
        spatial_iter = max(args.spatial_iter, 2)
        use_alpha_delta = True
    else:
        hole_fill = args.hole_fill
        spatial_iter = args.spatial_iter
        use_alpha_delta = False

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
    cv_cm_id = getattr(cv2, f"COLORMAP_{args.colormap}")
    color_lut = build_colormap_with_zero_black(cv_cm_id)
    disp_mul = 255.0 / max_disparity

    print("=" * 60)
    print(" OAK 深度采集（对齐 depthai_demo）")
    print("=" * 60)
    print(f" 分辨率: {args.resolution}    显示: {args.show}    色图: {args.colormap}")
    print(f" maxDisparity: {max_disparity}    confidence: {args.confidence}")
    print(f" subpixel: {args.subpixel}   extended: {args.extended}   LRC: {not args.no_lrc}")
    print(f" RGB-Depth 对齐: {args.align}（demo 默认关）")
    print(f" Spatial: holeFill={hole_fill}  iter={spatial_iter}  alpha/delta={use_alpha_delta}")
    print(f" Threshold: {'关' if args.no_threshold else '200-10000mm'}")
    print(f" 模式: {'激进填洞' if args.aggressive else '温和（demo 默认）'}")
    print(" 按键: [q] 退出   [s] 保存当前帧")
    print()

    if args.usb_speed == "usb2":
        device_ctx = dai.Device(pipeline, dai.UsbSpeed.HIGH)
    elif args.usb_speed == "usb3":
        device_ctx = dai.Device(pipeline, dai.UsbSpeed.SUPER)
    else:
        device_ctx = dai.Device(pipeline)

    with device_ctx as device:
        print(f"USB: {device.getUsbSpeed()}")
        depth_q = device.getOutputQueue("depth", maxSize=4, blocking=False)
        disp_q  = device.getOutputQueue("disparity", maxSize=4, blocking=False)

        if args.show in ("disparity", "both"):
            cv2.namedWindow("Disparity", cv2.WINDOW_NORMAL)
        if args.show in ("depth", "both"):
            cv2.namedWindow("Depth", cv2.WINDOW_NORMAL)

        fps_count, fps_start, fps = 0, time.time(), 0.0
        min_mm, mean_mm, valid_pct, stat_n = 0, 0.0, 0.0, 0
        latest_depth = None
        latest_disp = None
        first_disp_shown = False
        first_depth_shown = False

        while True:
            in_disp = disp_q.tryGet()
            in_depth = depth_q.tryGet()
            if in_disp is not None:
                latest_disp = in_disp.getFrame()
            if in_depth is not None:
                latest_depth = in_depth.getFrame()
            if latest_disp is None and latest_depth is None:
                continue

            if latest_depth is not None:
                stat_n += 1
                if stat_n >= 5:
                    valid = latest_depth[latest_depth > 0]
                    min_mm = int(valid.min()) if valid.size else 0
                    mean_mm = float(valid.mean()) if valid.size else 0.0
                    valid_pct = 100.0 * valid.size / latest_depth.size
                    stat_n = 0

            fps_count += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                fps = fps_count / elapsed
                fps_count, fps_start = 0, time.time()

            if args.show in ("disparity", "both") and latest_disp is not None:
                disp_u8 = np.clip(latest_disp * disp_mul, 0, 255).astype(np.uint8)
                disp_color = cv2.applyColorMap(disp_u8, color_lut)

                cv2.putText(disp_color, f"FPS:{fps:.1f}",         (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
                cv2.putText(disp_color, f"Min:{min_mm}mm",        (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
                cv2.putText(disp_color, f"Mean:{mean_mm:.0f}mm",  (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
                cv2.putText(disp_color, f"Valid:{valid_pct:.1f}%",(10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

                shown = fit_to_window(disp_color, args.window_width) if args.window_width > 0 else disp_color
                cv2.imshow("Disparity", shown)
                if not first_disp_shown and args.window_width > 0:
                    cv2.resizeWindow("Disparity", shown.shape[1], shown.shape[0])
                    first_disp_shown = True

            if args.show in ("depth", "both") and latest_depth is not None:
                clipped = np.clip(latest_depth, 0, args.max_display_mm)
                depth_u8 = (clipped * (255.0 / args.max_display_mm)).astype(np.uint8)
                depth_color = cv2.applyColorMap(depth_u8, color_lut)
                cv2.putText(depth_color, f"FPS:{fps:.1f}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
                shown = fit_to_window(depth_color, args.window_width) if args.window_width > 0 else depth_color
                cv2.imshow("Depth", shown)
                if not first_depth_shown and args.window_width > 0:
                    cv2.resizeWindow("Depth", shown.shape[1], shown.shape[0])
                    first_depth_shown = True

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                if save_dir is None:
                    print("[!] 未设置 --save-dir，跳过保存。")
                else:
                    stamp = time.strftime("%Y%m%d_%H%M%S")
                    if latest_depth is not None:
                        np.save(save_dir / f"depth_{stamp}.npy", latest_depth)
                    if latest_disp is not None:
                        np.save(save_dir / f"disp_{stamp}.npy", latest_disp)
                        disp_u8 = np.clip(latest_disp * disp_mul, 0, 255).astype(np.uint8)
                        cv2.imwrite(str(save_dir / f"disp_color_{stamp}.png"),
                                    cv2.applyColorMap(disp_u8, color_lut))
                    print(f"[✓] 已保存到 {save_dir}（{stamp}）")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
