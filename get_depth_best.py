"""
get_depth_claude.py — OAK 深度图采集脚本（与 depthai_demo 显示效果对齐）

核心对齐点（参照 depthai_demo 源码）：
1. 同时输出 depth(uint16, mm) 和 disparity(uint8/uint16)；显示用 disparity，统计用 depth。
   —— 这是 demo 默认 disparityColor 视图效果好的根本原因：disparity 经过 maxDisparity 归一化
      到 0..255 再上色，对比度好、近场细节多；而直接用固定 max_mm 归一化 depth 会糊。
2. dispMultiplier = 255 / maxDisparity，maxDisparity 随 subpixel/extended 动态变化：
   - 默认: 95
   - extended: 95 * 2 = 190
   - subpixel: 95 * 32 = 3040
   - subpixel + extended: 95 * 64 = 6080
3. 颜色表首行置零 (cvColorMap[0] = [0,0,0])，无效像素呈黑色，与 demo 完全一致。
4. HIGH_DENSITY 预设 + LRC + Subpixel + Spatial/Speckle 后处理 + setRectifyEdgeFillColor(0)。
5. 窗口尺寸固定为 720x405（不占满屏幕），可由 --window-width 调整。
6. 去掉之前错误创建却未使用的 ColorCamera（默认 align_target=rgb 时仍按 demo 思路创建并对齐）。
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
    confidence_threshold=240,    # demo GUI 默认 240（不是 245）
    median_filter=dai.MedianFilter.KERNEL_7x7,  # demo 默认 7x7
    enable_lrc=True,
    enable_subpixel=False,        # demo 默认关闭（开启会大幅增加计算量）
    enable_extended=False,
    enable_rgb_depth_align=True,
    align_target="rgb",
    bilateral_sigma=0,            # demo 默认 0
    lrc_threshold=10,             # demo GUI 默认 10
):
    pipeline = dai.Pipeline()

    # ── 双目相机 ─────────────────────────────────────────────────────
    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_right = pipeline.create(dai.node.MonoCamera)
    mono_left.setResolution(mono_resolution)
    mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    mono_right.setResolution(mono_resolution)
    mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

    # ── RGB 相机（仅做对齐参考，无 XLinkOut） ─────────────────────────
    cam_rgb = None
    if enable_rgb_depth_align and align_target == "rgb":
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    # ── StereoDepth ──────────────────────────────────────────────────
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)

    # 基础配置（与 demo updateDepthConfig 对应）
    stereo.initialConfig.setConfidenceThreshold(confidence_threshold)
    if median_filter is not None:
        stereo.initialConfig.setMedianFilter(median_filter)
    stereo.setLeftRightCheck(enable_lrc)
    stereo.initialConfig.setLeftRightCheckThreshold(lrc_threshold)
    stereo.setExtendedDisparity(enable_extended)
    stereo.setSubpixel(enable_subpixel)

    # 后处理 + bilateral sigma + 滤镜
    cfg = stereo.initialConfig.get()
    cfg.postProcessing.spatialFilter.enable = True
    cfg.postProcessing.spatialFilter.holeFillingRadius = 2
    cfg.postProcessing.spatialFilter.numIterations = 1
    cfg.postProcessing.speckleFilter.enable = True
    cfg.postProcessing.speckleFilter.speckleRange = 50
    cfg.postProcessing.bilateralSigmaValue = bilateral_sigma  # demo 用同一字段
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

    # ── 输出：同时输出 depth 和 disparity ──────────────────────────────
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
    """与 demo ConfigManager.maxDisparity 一致。"""
    max_disp = 95
    if enable_extended:
        max_disp *= 2
    if enable_subpixel:
        max_disp *= 32
    return max_disp


def build_colormap_with_zero_black(cv_colormap_id=cv2.COLORMAP_JET):
    """与 demo getColorMap 一致：256 级颜色表，第 0 项强制黑色（无效像素）。"""
    color_map = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv_colormap_id)
    color_map[0] = [0, 0, 0]
    return color_map


def colorize_disparity(disp_frame, max_disparity, color_map):
    """
    模拟 demo 的 disparityColor 渲染：
      1. 乘 dispMultiplier (255/maxDisp) 归一化到 0..255
      2. 通过 256 级颜色表上色
    """
    disp_mul = 255.0 / max_disparity
    disp_u8 = np.clip(disp_frame * disp_mul, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(disp_u8, color_map[:, 0])  # 用同色彩映射


def colorize_disparity_lut(disp_frame, max_disparity, color_map_lut):
    """LUT 版本（与 demo 完全一致）：先归一化到 uint8，再用 LUT 查表。"""
    disp_mul = 255.0 / max_disparity
    disp_u8 = np.clip(disp_frame * disp_mul, 0, 255).astype(np.uint8)
    # color_map_lut shape: (256, 1, 3)
    return color_map_lut[disp_u8].reshape(*disp_u8.shape, 3).astype(np.uint8)


def fit_to_window(img, target_w):
    """按宽度等比缩放到 target_w（仅用于显示，不影响数据本身）。"""
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
    p = argparse.ArgumentParser(description="OAK 深度采集（对齐 depthai_demo 显示效果）")
    p.add_argument("--resolution", choices=list(_RESOLUTION_MAP.keys()), default="400p")
    p.add_argument("--confidence", type=int, default=240, help="0-255，越低越严格（demo 默认 240）")
    p.add_argument("--median", choices=list(_MEDIAN_MAP.keys()), default="7", help="中值滤波核大小")
    p.add_argument("--sigma", type=int, default=0, help="Bilateral sigma，0 关闭（demo 默认 0）")
    p.add_argument("--lrc-threshold", type=int, default=10, help="LR-Check 阈值（demo 默认 10）")
    p.add_argument("--no-lrc", action="store_true", help="关闭 Left-Right Check")
    p.add_argument("--subpixel", action="store_true", help="开启亚像素（默认关）")
    p.add_argument("--extended", action="store_true", help="开启扩展视差（默认关）")
    p.add_argument("--no-align", action="store_true", help="禁用 RGB-Depth 对齐")
    p.add_argument("--align-target", choices=["rgb", "left", "right"], default="rgb")
    p.add_argument("--colormap", choices=["JET", "TURBO", "HOT", "VIRIDIS"], default="JET")
    p.add_argument("--window-width", type=int, default=720,
                   help="显示窗口宽度（像素），默认 720。设为 0 表示不缩放。")
    p.add_argument("--show", choices=["disparity", "depth", "both"], default="disparity",
                   help="显示哪个画面：disparity（与 demo 默认一致）/ depth / both")
    p.add_argument("--max-display-mm", type=int, default=8000,
                   help="depth 显示模式下的最大映射距离（mm）")
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

    pipeline = create_depth_pipeline(
        mono_resolution=resolution,
        confidence_threshold=args.confidence,
        median_filter=median,
        enable_lrc=not args.no_lrc,
        enable_subpixel=args.subpixel,
        enable_extended=args.extended,
        enable_rgb_depth_align=not args.no_align,
        align_target=args.align_target,
        bilateral_sigma=args.sigma,
        lrc_threshold=args.lrc_threshold,
    )

    max_disparity = compute_max_disparity(args.subpixel, args.extended)
    cv_cm_id = getattr(cv2, f"COLORMAP_{args.colormap}")
    color_lut = build_colormap_with_zero_black(cv_cm_id)
    disp_mul = 255.0 / max_disparity

    print("=" * 60)
    print(" OAK 深度采集（对齐 depthai_demo 风格）")
    print("=" * 60)
    print(f" 分辨率: {args.resolution}    显示: {args.show}    色图: {args.colormap}")
    print(f" maxDisparity: {max_disparity}    confidence: {args.confidence}")
    print(f" subpixel: {args.subpixel}   extended: {args.extended}   LRC: {not args.no_lrc}")
    print(f" 中值滤波: {args.median}    Bilateral sigma: {args.sigma}")
    print(f" 显示窗宽: {args.window_width}px")
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

        # 创建窗口（NORMAL 模式可手动调整大小，AUTOSIZE 强制按图像大小）
        if args.show in ("disparity", "both"):
            cv2.namedWindow("Disparity", cv2.WINDOW_NORMAL)
        if args.show in ("depth", "both"):
            cv2.namedWindow("Depth", cv2.WINDOW_NORMAL)

        fps_count, fps_start, fps = 0, time.time(), 0.0
        min_mm, mean_mm, stat_n = 0, 0.0, 0
        latest_depth = None
        latest_disp = None

        while True:
            in_disp = disp_q.tryGet()
            in_depth = depth_q.tryGet()

            if in_disp is not None:
                latest_disp = in_disp.getFrame()
            if in_depth is not None:
                latest_depth = in_depth.getFrame()

            if latest_disp is None and latest_depth is None:
                continue

            # 统计（每 5 帧）
            if latest_depth is not None:
                stat_n += 1
                if stat_n >= 5:
                    valid = latest_depth[latest_depth > 0]
                    min_mm = int(valid.min()) if valid.size else 0
                    mean_mm = float(valid.mean()) if valid.size else 0.0
                    stat_n = 0

            # FPS
            fps_count += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                fps = fps_count / elapsed
                fps_count, fps_start = 0, time.time()

            # ── 显示 disparity（与 demo disparityColor 完全一致的渲染） ──
            if args.show in ("disparity", "both") and latest_disp is not None:
                # disparity 输出在 subpixel=True 时为 uint16
                disp_u8 = np.clip(latest_disp * disp_mul, 0, 255).astype(np.uint8)
                # 用 LUT 上色（首项黑色 → 无效像素显示黑）
                disp_color = cv2.applyColorMap(disp_u8, color_lut)

                cv2.putText(disp_color, f"FPS:{fps:.1f}",       (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
                cv2.putText(disp_color, f"Min:{min_mm}mm",      (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
                cv2.putText(disp_color, f"Mean:{mean_mm:.0f}mm",(10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

                shown = fit_to_window(disp_color, args.window_width) if args.window_width > 0 else disp_color
                cv2.imshow("Disparity", shown)
                # 第一次显示时把窗口尺寸设到与图像一致
                if fps_count == 1 and args.window_width > 0:
                    cv2.resizeWindow("Disparity", shown.shape[1], shown.shape[0])

            # ── 显示 depth（基于固定 mm 范围归一化） ──
            if args.show in ("depth", "both") and latest_depth is not None:
                clipped = np.clip(latest_depth, 0, args.max_display_mm)
                depth_u8 = (clipped * (255.0 / args.max_display_mm)).astype(np.uint8)
                depth_color = cv2.applyColorMap(depth_u8, color_lut)

                cv2.putText(depth_color, f"FPS:{fps:.1f}",       (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
                cv2.putText(depth_color, f"Range:0-{args.max_display_mm}mm", (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

                shown = fit_to_window(depth_color, args.window_width) if args.window_width > 0 else depth_color
                cv2.imshow("Depth", shown)
                if fps_count == 1 and args.window_width > 0:
                    cv2.resizeWindow("Depth", shown.shape[1], shown.shape[0])

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