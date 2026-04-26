import argparse
import time
from pathlib import Path

import cv2
import depthai as dai
import numpy as np


def create_depth_pipeline(
    mono_resolution: dai.MonoCameraProperties.SensorResolution,
    confidence_threshold: int = 245,
    median_filter: dai.MedianFilter = dai.MedianFilter.KERNEL_5x5,
    enable_lrc: bool = True,
    enable_subpixel: bool = True,
    enable_align: bool = False,
    enable_extended: bool = False,
) -> dai.Pipeline:
    """
    创建深度图管道，并配置后处理滤镜以解决闪烁和重影。
    """
    pipeline = dai.Pipeline()

    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_right = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)
    xout_depth = pipeline.create(dai.node.XLinkOut)

    # 与官方 demo 思路一致：先套用高密度预设，再按需覆盖细节参数。
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)

    mono_left.setResolution(mono_resolution)
    mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    mono_right.setResolution(mono_resolution)
    mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

    # 1. 基础配置
    stereo.initialConfig.setConfidenceThreshold(confidence_threshold)
    if median_filter is not None:
        stereo.initialConfig.setMedianFilter(median_filter)

    stereo.setLeftRightCheck(enable_lrc)
    # 官方 demo 默认更接近 4，阈值过严会导致更多空洞和不稳定。
    stereo.initialConfig.setLeftRightCheckThreshold(4)
    
    stereo.setExtendedDisparity(enable_extended)
    stereo.setSubpixel(enable_subpixel)

    # 2. 后处理滤镜配置 (解决闪烁和重影的关键)
    config = stereo.initialConfig.get()
    
    # 空间滤波 (Spatial Filter)：平滑边缘，填充微小空洞。
    config.postProcessing.spatialFilter.enable = True
    config.postProcessing.spatialFilter.holeFillingRadius = 2
    config.postProcessing.spatialFilter.numIterations = 1
    
    # 杂散斑点滤波 (Speckle Filter)：移除右侧常见的噪点团块（重影）。
    config.postProcessing.speckleFilter.enable = True
    config.postProcessing.speckleFilter.speckleRange = 50
    
    # 边缘处理：将校正边缘无效像素设为0，消除右侧边缘的混乱。
    stereo.setRectifyEdgeFillColor(0)
    
    stereo.initialConfig.set(config)

    if enable_align:
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_C)

    xout_depth.setStreamName("depth")
    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)
    stereo.depth.link(xout_depth.input)

    return pipeline


def depth_to_colormap(depth_mm: np.ndarray, max_display_mm: int) -> np.ndarray:
    """
    把 16bit 深度图(单位:mm)映射成伪彩色图，便于人眼观察。
    """
    clipped = np.clip(depth_mm, 0, max_display_mm)
    depth_u8 = cv2.convertScaleAbs(clipped, alpha=255.0 / max_display_mm)
    depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)
    return depth_color


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OAK-D-LITE-FF 深度图采集（仅深度，不含避障算法）")
    parser.add_argument(
        "--resolution",
        choices=["400p", "480p", "720p", "800p"],
        default="480p",
        help="双目相机分辨率，默认 480p",
    )
    parser.add_argument(
        "--max-display-mm",
        type=int,
        default=5000,
        help="可视化时映射的最大深度(mm)，默认 5000",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="",
        help="按 s 保存当前深度帧到该目录（保存 .npy 原始毫米图 + .png 伪彩图）",
    )
    parser.add_argument(
        "--confidence",
        type=int,
        default=245,
        help="深度置信度阈值 0-255，越大深度点越少但更可靠（默认 245）",
    )
    parser.add_argument(
        "--no-lrc",
        action="store_true",
        help="关闭左右一致性检查（LeftRightCheck），减少重影但错误匹配增多",
    )
    parser.add_argument(
        "--no-subpixel",
        action="store_true",
        help="关闭亚像素精度（Subpixel），默认开启以获得更高深度精度",
    )
    parser.add_argument(
        "--extended",
        action="store_true",
        help="开启扩展视差（Extended Disparity），提升近处深度精度",
    )
    parser.add_argument(
        "--align",
        action="store_true",
        help="开启深度对齐到右相机，解决物理重影问题",
    )
    parser.add_argument(
        "--usb-speed",
        choices=["auto", "usb2", "usb3"],
        default="auto",
        help="设备 USB 速率模式：auto(自动)、usb2(强制 HIGH)、usb3(强制 SUPER)",
    )
    return parser.parse_args()


def get_resolution(name: str) -> dai.MonoCameraProperties.SensorResolution:
    mapping = {
        "400p": dai.MonoCameraProperties.SensorResolution.THE_400_P,
        "480p": dai.MonoCameraProperties.SensorResolution.THE_480_P,
        "720p": dai.MonoCameraProperties.SensorResolution.THE_720_P,
        "800p": dai.MonoCameraProperties.SensorResolution.THE_800_P,
    }
    return mapping[name]


def main() -> None:
    args = parse_args()
    resolution = get_resolution(args.resolution)
    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    pipeline = create_depth_pipeline(
        resolution,
        confidence_threshold=args.confidence,
        enable_lrc=not args.no_lrc,
        enable_subpixel=not args.no_subpixel,
        enable_align=args.align,
        enable_extended=args.extended,
    )

    print("==============================================")
    print(" OAK-D-LITE-FF 深度图采集脚本（仅深度）")
    print("==============================================")
    print("[按键说明] q:退出  s:保存当前深度帧")

    if args.usb_speed == "usb2":
        requested_usb_speed = dai.UsbSpeed.HIGH
    elif args.usb_speed == "usb3":
        requested_usb_speed = dai.UsbSpeed.SUPER
    else:
        requested_usb_speed = None

    if requested_usb_speed is None:
        device_ctx = dai.Device(pipeline)
    else:
        device_ctx = dai.Device(pipeline, requested_usb_speed)

    with device_ctx as device:
        # 只保留最新帧，避免显示端处理不过来时累积队列导致延迟越来越大。
        depth_queue = device.getOutputQueue(name="depth", maxSize=1, blocking=False)
        print(f"USB 请求模式: {args.usb_speed}")
        print(f"USB 实际连接速度: {device.getUsbSpeed()}")
        print("开始接收深度图...")

        fps_count = 0
        fps_time_start = time.time()
        fps = 0.0
        min_mm = 0
        mean_mm = 0.0
        stat_frame_count = 0

        while True:
            in_depth = depth_queue.tryGet()
            if in_depth is None:
                continue
            depth_mm = in_depth.getFrame()  # uint16, 单位 mm

            # 统计信息不必每帧都算，降低 CPU 压力和显示端延迟。
            stat_frame_count += 1
            if stat_frame_count >= 5:
                valid = depth_mm[depth_mm > 0]
                min_mm = int(valid.min()) if valid.size else 0
                mean_mm = float(valid.mean()) if valid.size else 0.0
                stat_frame_count = 0

            depth_color = depth_to_colormap(depth_mm, args.max_display_mm)

            fps_count += 1
            elapsed = time.time() - fps_time_start
            if elapsed >= 1.0:
                fps = fps_count / elapsed
                fps_count = 0
                fps_time_start = time.time()

            cv2.putText(depth_color, f"FPS: {fps:.1f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(depth_color, f"Min: {min_mm} mm", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(depth_color, f"Mean: {mean_mm:.0f} mm", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("OAK Depth (Colorized)", depth_color)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                if save_dir is None:
                    print("未设置 --save-dir，跳过保存。")
                else:
                    stamp = time.strftime("%Y%m%d_%H%M%S")
                    npy_path = save_dir / f"depth_{stamp}.npy"
                    png_path = save_dir / f"depth_{stamp}.png"
                    np.save(npy_path, depth_mm)
                    cv2.imwrite(str(png_path), depth_color)
                    print(f"已保存: {npy_path} 和 {png_path}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
