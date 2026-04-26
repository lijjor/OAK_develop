"""
get_depth.py — OAK 深度图采集脚本
- 修复 Python 3.7/3.8 不支持 tuple[...] 类型注解的问题
- 深度管道逻辑与 depthai_demo 保持一致:
    * HIGH_DENSITY 预设
    * LRC + Subpixel + Speckle/Spatial 后处理滤镜
    * RGB-Depth 对齐（与 demo 的 rgbDepthAlignment 选项一致）
    * setRectifyEdgeFillColor(0) 消除右侧边缘噪声
"""
import argparse
import time
from pathlib import Path

import cv2
import depthai as dai
import numpy as np


def create_depth_pipeline(
    mono_resolution,   # dai.MonoCameraProperties.SensorResolution
    confidence_threshold=245,
    median_filter=dai.MedianFilter.KERNEL_5x5,
    enable_lrc=True,
    enable_subpixel=True,
    enable_rgb_depth_align=True,
    align_target="rgb",
    enable_extended=False,
):
    """
    创建深度管道。逻辑与 depthai_demo 的 PipelineManager.createDepth() 对齐：
      - HIGH_DENSITY 预设（对应 demo 默认模式）
      - LRC、Subpixel、Speckle Filter、Spatial Filter
      - RGB 对齐（对应 demo 的 toggleRgbDepthAlignment）
    """
    pipeline = dai.Pipeline()

    # ── 双目相机 ──────────────────────────────────────────────────────
    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_right = pipeline.create(dai.node.MonoCamera)
    mono_left.setResolution(mono_resolution)
    mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    mono_right.setResolution(mono_resolution)
    mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

    # ── RGB 相机（仅用于对齐，不做预览输出） ──────────────────────────
    cam_rgb = None
    if enable_lrc and enable_rgb_depth_align and align_target == "rgb":
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam_rgb.setInterleaved(False)
        # demo 里 setIspScale(1,3) → 1920x1080→640x360，这里保持 1080p 全幅对齐
        # 如果你的设备是 OAK-D-Lite，RGB 最高只到 4K，不需要降采样

    # ── StereoDepth ───────────────────────────────────────────────────
    stereo = pipeline.create(dai.node.StereoDepth)

    # 1. 预设：HIGH_DENSITY（与 demo 默认一致）
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)

    # 2. 基础参数
    stereo.initialConfig.setConfidenceThreshold(confidence_threshold)
    stereo.initialConfig.setMedianFilter(median_filter)
    stereo.setLeftRightCheck(enable_lrc)
    stereo.initialConfig.setLeftRightCheckThreshold(4)   # demo 默认值
    stereo.setExtendedDisparity(enable_extended)
    stereo.setSubpixel(enable_subpixel)

    # 3. 后处理滤镜（解决闪烁 / 重影 / 噪点）
    cfg = stereo.initialConfig.get()

    # Spatial Filter：填充小空洞、平滑边缘
    cfg.postProcessing.spatialFilter.enable = True
    cfg.postProcessing.spatialFilter.holeFillingRadius = 2
    cfg.postProcessing.spatialFilter.numIterations = 1

    # Speckle Filter：消除右侧噪点团块（重影的主要来源之一）
    cfg.postProcessing.speckleFilter.enable = True
    cfg.postProcessing.speckleFilter.speckleRange = 50

    stereo.initialConfig.set(cfg)

    # 4. 边缘填充为 0，消除校正边缘的无效像素噪声
    stereo.setRectifyEdgeFillColor(0)

    # 5. RGB-Depth 对齐（与 demo 的 toggleRgbDepthAlignment 选项一致）
    if enable_lrc and enable_rgb_depth_align:
        if align_target == "left":
            stereo.setDepthAlign(dai.CameraBoardSocket.CAM_B)
        elif align_target == "right":
            stereo.setDepthAlign(dai.CameraBoardSocket.CAM_C)
        else:  # rgb（默认）
            stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

    # ── 输出节点 ──────────────────────────────────────────────────────
    xout_depth = pipeline.create(dai.node.XLinkOut)
    xout_depth.setStreamName("depth")

    # ── 连线 ──────────────────────────────────────────────────────────
    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)
    stereo.depth.link(xout_depth.input)
    # 注意：cam_rgb 不需要连接到 stereo，setDepthAlign 只需要知道 socket 编号

    return pipeline


# ─────────────────────────────────────────────────────────────────────────────

def depth_to_colormap(depth_mm, max_display_mm):
    """16-bit 深度图 (mm) → JET 伪彩色图，与 demo 的 colorMap 逻辑一致。"""
    clipped = np.clip(depth_mm, 0, max_display_mm)
    depth_u8 = cv2.convertScaleAbs(clipped, alpha=255.0 / max_display_mm)
    return cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)


def parse_args():
    parser = argparse.ArgumentParser(description="OAK 深度图采集脚本")
    parser.add_argument(
        "--resolution", choices=["400p", "480p", "720p", "800p"], default="400p",
        help="双目相机分辨率，默认 400p（与 demo 的 mono default 一致）",
    )
    parser.add_argument(
        "--max-display-mm", type=int, default=5000,
        help="伪彩色映射的最大深度 (mm)，默认 5000",
    )
    parser.add_argument(
        "--save-dir", type=str, default="",
        help="按 s 保存当前帧：.npy 原始深度 + .png 伪彩图",
    )
    parser.add_argument(
        "--confidence", type=int, default=245,
        help="深度置信度阈值 0-255，默认 245",
    )
    parser.add_argument("--no-lrc", action="store_true", help="关闭 Left-Right Check")
    parser.add_argument("--no-subpixel", action="store_true", help="关闭亚像素精度")
    parser.add_argument("--extended", action="store_true", help="开启扩展视差")
    parser.add_argument(
        "--no-rgb-depth-align", action="store_true",
        help="禁用 RGB-Depth 对齐（默认开启，与 demo 一致）",
    )
    parser.add_argument(
        "--align-target", choices=["rgb", "left", "right"], default="rgb",
        help="对齐目标相机，默认 rgb",
    )
    parser.add_argument(
        "--usb-speed", choices=["auto", "usb2", "usb3"], default="auto",
        help="USB 速率：auto / usb2 / usb3",
    )
    return parser.parse_args()


_RESOLUTION_MAP = {
    "400p": dai.MonoCameraProperties.SensorResolution.THE_400_P,
    "480p": dai.MonoCameraProperties.SensorResolution.THE_480_P,
    "720p": dai.MonoCameraProperties.SensorResolution.THE_720_P,
    "800p": dai.MonoCameraProperties.SensorResolution.THE_800_P,
}


def main():
    args = parse_args()
    resolution = _RESOLUTION_MAP[args.resolution]
    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    pipeline = create_depth_pipeline(
        resolution,
        confidence_threshold=args.confidence,
        enable_lrc=not args.no_lrc,
        enable_subpixel=not args.no_subpixel,
        enable_rgb_depth_align=not args.no_rgb_depth_align,
        align_target=args.align_target,
        enable_extended=args.extended,
    )

    print("=" * 50)
    print(" OAK 深度图采集脚本")
    print("=" * 50)
    print(" 按键: [q] 退出   [s] 保存当前帧")
    print()

    if args.usb_speed == "usb2":
        device_ctx = dai.Device(pipeline, dai.UsbSpeed.HIGH)
    elif args.usb_speed == "usb3":
        device_ctx = dai.Device(pipeline, dai.UsbSpeed.SUPER)
    else:
        device_ctx = dai.Device(pipeline)

    with device_ctx as device:
        print(f"USB 速度: {device.getUsbSpeed()}")
        # maxSize=1 + blocking=False：始终只保留最新帧，避免队列积压导致延迟
        depth_queue = device.getOutputQueue(name="depth", maxSize=1, blocking=False)
        print("开始接收深度图...\n")

        fps_count = 0
        fps_start = time.time()
        fps = 0.0
        min_mm = 0
        mean_mm = 0.0
        stat_count = 0

        while True:
            in_depth = depth_queue.tryGet()
            if in_depth is None:
                continue

            depth_mm = in_depth.getFrame()   # shape (H, W), dtype uint16, 单位 mm

            # 每 5 帧更新一次统计，降低 CPU 占用
            stat_count += 1
            if stat_count >= 5:
                valid = depth_mm[depth_mm > 0]
                min_mm = int(valid.min()) if valid.size else 0
                mean_mm = float(valid.mean()) if valid.size else 0.0
                stat_count = 0

            depth_color = depth_to_colormap(depth_mm, args.max_display_mm)

            fps_count += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                fps = fps_count / elapsed
                fps_count = 0
                fps_start = time.time()

            cv2.putText(depth_color, f"FPS: {fps:.1f}",         (20, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(depth_color, f"Min:  {min_mm} mm",       (20, 60),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(depth_color, f"Mean: {mean_mm:.0f} mm",  (20, 90),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("OAK Depth (Colorized)", depth_color)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                if save_dir is None:
                    print("[!] 未设置 --save-dir，跳过保存。")
                else:
                    stamp = time.strftime("%Y%m%d_%H%M%S")
                    npy_path = save_dir / f"depth_{stamp}.npy"
                    png_path = save_dir / f"depth_{stamp}.png"
                    np.save(npy_path, depth_mm)
                    cv2.imwrite(str(png_path), depth_color)
                    print(f"[✓] 已保存: {npy_path}  |  {png_path}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
