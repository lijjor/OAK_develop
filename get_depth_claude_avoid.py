"""
get_depth_claude_avoid.py — 在原 get_depth_claude.py 基础上集成左中右三区避障

主要改动：
  1. import obstacle_avoidance 模块
  2. 主循环中对每帧 depth_mm 调用 ObstacleAvoidance.process()
  3. 在 disparity 显示窗口上叠加避障可视化（区域框 + 状态 + 决策横幅）
  4. 终端打印每秒一次的避障决策（避免刷屏）
  5. 新增命令行参数 --stop-mm / --warn-mm / --no-avoid 等

获取深度逻辑、显示逻辑保持与你当前版本完全一致。
"""
import argparse
import time
from pathlib import Path

import cv2
import depthai as dai
import numpy as np

from obstacle_avoidance import ObstacleAvoidance, draw_avoidance_overlay


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline（与你当前版本完全一致）
# ─────────────────────────────────────────────────────────────────────────────
def create_depth_pipeline(
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
# 显示工具（与你版本一致）
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
# 参数
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
    p = argparse.ArgumentParser(description="OAK 深度采集 + 三区避障")

    # 深度参数
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

    # ★ 避障参数
    p.add_argument("--no-avoid", action="store_true", help="关闭避障算法")
    p.add_argument("--stop-mm",  type=int, default=600,  help="紧急停车距离阈值(mm)")
    p.add_argument("--warn-mm",  type=int, default=1200, help="警戒距离阈值(mm)")
    p.add_argument("--avoid-percentile", type=float, default=10.0,
                   help="区域代表距离取第几百分位(0-100，越小越敏感)")
    p.add_argument("--avoid-min-valid", type=float, default=0.10,
                   help="区域有效像素占比低于此值视为 UNKNOWN")
    p.add_argument("--roi-top",    type=float, default=0.20, help="ROI 顶部裁剪比例")
    p.add_argument("--roi-bottom", type=float, default=0.85, help="ROI 底部裁剪比例")

    # 显示
    p.add_argument("--colormap", choices=["JET", "TURBO", "HOT", "VIRIDIS"], default="JET")
    p.add_argument("--window-width", type=int, default=720)
    p.add_argument("--max-display-mm", type=int, default=8000)
    p.add_argument("--save-dir", type=str, default="")
    p.add_argument("--usb-speed", choices=["auto", "usb2", "usb3"], default="auto")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 主循环
# ─────────────────────────────────────────────────────────────────────────────
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

    # ★ 避障实例
    avoider = ObstacleAvoidance(
        stop_mm=args.stop_mm,
        warn_mm=args.warn_mm,
        valid_min_mm=200,
        valid_max_mm=8000,
        percentile=args.avoid_percentile,
        min_valid_ratio=args.avoid_min_valid,
        roi_top=args.roi_top,
        roi_bottom=args.roi_bottom,
    )

    max_disparity = compute_max_disparity(args.subpixel, args.extended)
    cv_cm_id = getattr(cv2, f"COLORMAP_{args.colormap}")
    color_lut = build_colormap_with_zero_black(cv_cm_id)
    disp_mul = 255.0 / max_disparity

    print("=" * 60)
    print(" OAK 深度采集 + 三区避障")
    print("=" * 60)
    print(f" 分辨率: {args.resolution}    色图: {args.colormap}")
    print(f" 深度: confidence={args.confidence}  subpixel={args.subpixel}  "
          f"extended={args.extended}  align={args.align}")
    print(f" 避障: STOP<{args.stop_mm}mm  WARN<{args.warn_mm}mm  "
          f"P{args.avoid_percentile:.0f}  ROI=[{args.roi_top:.2f},{args.roi_bottom:.2f}]")
    print(f" 模式: {'激进填洞' if args.aggressive else '温和'}    避障: {'关' if args.no_avoid else '开'}")
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

        cv2.namedWindow("Disparity + Avoidance", cv2.WINDOW_NORMAL)

        fps_count, fps_start, fps = 0, time.time(), 0.0
        latest_depth, latest_disp = None, None
        last_print_t = 0.0
        first_shown = False

        while True:
            in_disp = disp_q.tryGet()
            in_depth = depth_q.tryGet()
            if in_disp is not None:
                latest_disp = in_disp.getFrame()
            if in_depth is not None:
                latest_depth = in_depth.getFrame()
            if latest_disp is None or latest_depth is None:
                continue

            # FPS
            fps_count += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                fps = fps_count / elapsed
                fps_count, fps_start = 0, time.time()

            # ── 渲染 disparity ──
            disp_u8 = np.clip(latest_disp * disp_mul, 0, 255).astype(np.uint8)
            disp_color = cv2.applyColorMap(disp_u8, color_lut)

            # ── 缩放到显示尺寸（避障 bbox 用 depth 原始尺寸算的，需要 scale） ──
            shown = fit_to_window(disp_color, args.window_width) if args.window_width > 0 else disp_color
            scale_x = shown.shape[1] / disp_color.shape[1]
            scale_y = shown.shape[0] / disp_color.shape[0]

            # ── 避障 ──
            if not args.no_avoid:
                # 注意：depth 和 disparity 尺寸应当一致（同一 stereo 节点输出）
                # 如果尺寸不一致，需要按 depth 尺寸做避障，再按 scale 投到 shown 上
                result = avoider.process(latest_depth)
                # 避障 bbox 是按 depth 原图坐标算的；shown 已被缩放，所以需要 scale
                bbox_scale_x = shown.shape[1] / latest_depth.shape[1]
                bbox_scale_y = shown.shape[0] / latest_depth.shape[0]
                draw_avoidance_overlay(shown, result, scale_x=bbox_scale_x, scale_y=bbox_scale_y)

                # 每秒打印一次决策
                now = time.time()
                if now - last_print_t > 1.0:
                    z = result.zones
                    print(f"[{time.strftime('%H:%M:%S')}] "
                          f"L:{z[0].state}({z[0].rep_mm}mm) "
                          f"C:{z[1].state}({z[1].rep_mm}mm) "
                          f"R:{z[2].state}({z[2].rep_mm}mm) "
                          f"=> {result.action}  | {result.reason}")
                    last_print_t = now

            # ── FPS 角标 ──
            cv2.putText(shown, f"FPS:{fps:.1f}", (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

            cv2.imshow("Disparity + Avoidance", shown)
            if not first_shown and args.window_width > 0:
                cv2.resizeWindow("Disparity + Avoidance", shown.shape[1], shown.shape[0])
                first_shown = True

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                if save_dir is None:
                    print("[!] 未设置 --save-dir，跳过保存。")
                else:
                    stamp = time.strftime("%Y%m%d_%H%M%S")
                    np.save(save_dir / f"depth_{stamp}.npy", latest_depth)
                    cv2.imwrite(str(save_dir / f"avoid_{stamp}.png"), shown)
                    print(f"[✓] 已保存到 {save_dir}（{stamp}）")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
