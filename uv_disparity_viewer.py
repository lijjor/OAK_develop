"""
uv_disparity.py — U/V-Disparity 观察工具 + 地面拟合可视化

用途
====
1. 提供 compute_v_disparity / compute_u_disparity / render_hist 三个纯函数,
   主代码 get_depth_claude_avoid.py 将来需要时直接 from uv_disparity import ...
2. 自带 main() 可以独立跑起来验证效果 —— 但管道参数与主代码完全一致,
   所以这里看到的 disparity 与主代码看到的 disparity 完全等价。
3. ★ 新增：在 V-Disparity 窗口上实时叠加 RANSAC 拟合出的地面直线（青色）+
   候选点（黄色）+ inlier 点（绿色），用于评估拟合效果。

★ 管道参数对齐说明
==================
下列参数必须与 get_depth_claude_avoid.py / get_depth_claude.py 保持一致:
  - confidence_threshold = 240
  - median_filter        = KERNEL_7x7
  - enable_lrc           = True, lrc_threshold = 10
  - enable_subpixel      = False (默认)
  - enable_extended      = False (默认)
  - bilateral_sigma      = 0
  - spatial_hole_fill    = 2  (温和)
  - spatial_iterations   = 1  (温和)
  - speckle_range        = 50
  - threshold filter     = 200..10000 mm
  - setRectifyEdgeFillColor(0)
  - 默认不开 RGB-Depth 对齐
任何主代码参数变更请同步到这里, 否则两边 disparity 不一致, 观察结果失真。
"""
import argparse
import time
from pathlib import Path

import cv2
import depthai as dai
import numpy as np

from ground_fitter import (
    extract_ground_candidates, fit_ground_line_ransac, GroundLine,
)


# ─────────────────────────────────────────────────────────────────────────────
# UV-Disparity 计算（纯函数，主代码可直接 import 复用）
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


def disparity_to_int(disp_frame: np.ndarray, max_disp: int,
                     subpixel_scale: int = 1) -> np.ndarray:
    if subpixel_scale > 1:
        disp_int = disp_frame.astype(np.int32) // subpixel_scale
    else:
        disp_int = disp_frame.astype(np.int32)
    return np.clip(disp_int, 0, max_disp)


def render_hist(hist: np.ndarray, axis: str, target_size,
                log_scale: bool = True, colormap: int = cv2.COLORMAP_HOT) -> np.ndarray:
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
# ★ 地面拟合可视化：在 V-Disparity 显示图上叠加候选点 + inlier + 拟合直线
# ─────────────────────────────────────────────────────────────────────────────
def overlay_ground_line(
    v_disp_img: np.ndarray,           # 已 render_hist 上色并 resize 过的 BGR 图
    v_hist_shape,                     # (H_disp, max_disp+1)：v_hist 原始尺寸
    candidates: np.ndarray,           # (N, 2) [y, disp]，可能为空
    line: GroundLine,                 # 可能为 None
    inliers_mask: np.ndarray = None,  # bool (N,)，可选
):
    """
    把候选点（黄）、inlier 点（绿）、拟合直线（青）画到 v_disp_img 上。
    v_disp_img 是显示尺寸，需要把 v_hist 坐标映射到显示坐标。
    """
    disp_h, max_disp_plus_1 = v_hist_shape
    img_h, img_w = v_disp_img.shape[:2]
    sx = img_w / max_disp_plus_1
    sy = img_h / disp_h

    # 候选点（黄）
    if candidates is not None and len(candidates) > 0:
        for i, (y, d) in enumerate(candidates):
            color = (0, 255, 0) if (inliers_mask is not None and inliers_mask[i]) else (0, 255, 255)
            cx = int(d * sx + sx / 2)
            cy = int(y * sy + sy / 2)
            cv2.circle(v_disp_img, (cx, cy), 1, color, -1, cv2.LINE_AA)

    # 拟合直线（青色，鲜艳）
    if line is not None:
        # 在 v_hist 坐标系下，直线 disp = slope*y + intercept，y ∈ [0, disp_h-1]
        y_top, y_bot = 0, disp_h - 1
        d_top = line.slope * y_top + line.intercept
        d_bot = line.slope * y_bot + line.intercept
        # 映射到显示坐标
        pt1 = (int(d_top * sx + sx / 2), int(y_top * sy + sy / 2))
        pt2 = (int(d_bot * sx + sx / 2), int(y_bot * sy + sy / 2))
        cv2.line(v_disp_img, pt1, pt2, (255, 255, 0), 2, cv2.LINE_AA)  # 青色

        # 拟合信息文字
        info = f"slope={line.slope:.3f} intercept={line.intercept:.1f} " \
               f"inliers={line.num_inliers}/{line.num_candidates} ({line.inlier_ratio*100:.0f}%)"
        cv2.rectangle(v_disp_img, (0, img_h - 22), (img_w, img_h), (0, 0, 0), -1)
        cv2.putText(v_disp_img, info, (5, img_h - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1, cv2.LINE_AA)
    else:
        cv2.rectangle(v_disp_img, (0, img_h - 22), (img_w, img_h), (0, 0, 0), -1)
        cv2.putText(v_disp_img, "Ground fit: FAILED", (5, img_h - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)


def overlay_ground_on_disparity(
    disp_color: np.ndarray,           # 原 disparity 上色图
    line: GroundLine,                 # 拟合的地面直线（v_hist 坐标系）
    depth_h: int,                     # depth/disparity 原始高度
    color=(0, 255, 255),              # 黄色
    thickness: int = 1,
):
    """
    把"每行预测的地面 disparity 位置"标记到原 disparity 显示图上。
    实际上是把斜线投回到原图：对于每个像素行 y，地面 disparity = line.predict(y)
    但这只是 V-Disparity 中的拟合信息，画到原图上意义有限。
    简化：画一条水平参考线，标注地面起始行（disp = 0 对应的 y）。
    实际不画水平线，只标"开始有地面信号的 y"用于参考。
    """
    if line is None:
        return
    # 求 line 在 depth 坐标系下的"地面起始 y"：disp = min_useful_disp (例如 5)
    # 即 y = (5 - intercept) / slope
    if abs(line.slope) < 1e-6:
        return
    img_h, img_w = disp_color.shape[:2]
    sy = img_h / depth_h
    # 标 disparity = 5, 15, 25 对应的 y 行（不同距离的地面位置）
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
# 显示工具
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
    p = argparse.ArgumentParser(description="UV-Disparity 观察 + 地面 RANSAC 拟合")
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

    # UV 显示相关
    p.add_argument("--colormap", choices=["JET", "TURBO", "HOT", "VIRIDIS"], default="JET")
    p.add_argument("--window-width", type=int, default=720)
    p.add_argument("--vdisp-width", type=int, default=400)
    p.add_argument("--udisp-height", type=int, default=300)
    p.add_argument("--no-log", action="store_true")
    p.add_argument("--save-dir", type=str, default="")
    p.add_argument("--usb-speed", choices=["auto", "usb2", "usb3"], default="auto")

    # ★ 地面拟合相关
    p.add_argument("--no-ground-fit", action="store_true",
                   help="关闭地面拟合可视化")
    p.add_argument("--gnd-y-start", type=float, default=0.45,
                   help="候选点提取的起始 y 比例（图像下方区域，默认 0.45）")
    p.add_argument("--gnd-relative-peak", type=float, default=0.5,
                   help="行内峰值占比阈值（默认 0.5，越小越容易接受散开行）")
    p.add_argument("--gnd-min-peak-count", type=int, default=5,
                   help="行峰值最小像素数（默认 5）")
    p.add_argument("--gnd-residual", type=float, default=1.5,
                   help="RANSAC 残差阈值（视差单位，默认 1.5）")
    p.add_argument("--gnd-iters", type=int, default=200,
                   help="RANSAC 迭代次数（默认 200）")
    p.add_argument("--gnd-min-inliers", type=int, default=30,
                   help="最少 inlier 数才认为拟合成功（默认 30）")
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

    print("=" * 60)
    print(" UV-Disparity 观察 + 地面 RANSAC 拟合")
    print("=" * 60)
    print(f" 分辨率: {args.resolution}    confidence: {args.confidence}")
    print(f" subpixel={args.subpixel}  extended={args.extended}  LRC={not args.no_lrc}")
    print(f" 整数视差范围: 0..{int_max_disp}    subpixel_scale={subpixel_scale}")
    print(f" 地面拟合: {'关' if args.no_ground_fit else '开'}  "
          f"(y_start={args.gnd_y_start}, residual={args.gnd_residual}, "
          f"iters={args.gnd_iters})")
    print(" 按键: [q]退出 [s]保存 [l]切换log [+/-]调UV显示尺寸 [g]切换地面拟合")
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

            # ── 主 disparity 上色 ──
            disp_u8 = np.clip(latest_disp * disp_color_mul, 0, 255).astype(np.uint8)
            disp_color = cv2.applyColorMap(disp_u8, color_lut)
            disp_shown = fit_to_window(disp_color, args.window_width)

            # ── 反 subpixel 得整数视差 ──
            disp_int = disparity_to_int(latest_disp, int_max_disp, subpixel_scale)

            # ── V/U 直方图 ──
            v_hist = compute_v_disparity(disp_int, int_max_disp)
            u_hist = compute_u_disparity(disp_int, int_max_disp)
            v_img = render_hist(v_hist, "v",
                                target_size=(vdisp_width, disp_shown.shape[0]),
                                log_scale=log_scale)
            u_img = render_hist(u_hist, "u",
                                target_size=(disp_shown.shape[1], udisp_height),
                                log_scale=log_scale)

            # ── ★ 地面拟合 ──
            ground_line = None
            candidates = np.empty((0, 2), np.float32)
            inliers_mask = None
            if do_ground_fit:
                candidates = extract_ground_candidates(
                    v_hist,
                    y_start_ratio=args.gnd_y_start,
                    min_peak_count=args.gnd_min_peak_count,
                    relative_peak_ratio=args.gnd_relative_peak,
                )
                if len(candidates) >= args.gnd_min_inliers // 2:
                    ground_line = fit_ground_line_ransac(
                        candidates,
                        n_iterations=args.gnd_iters,
                        residual_threshold=args.gnd_residual,
                        min_inliers=args.gnd_min_inliers,
                    )
                    # 重新计算 inliers_mask 用于显示
                    if ground_line is not None:
                        residuals = np.abs(
                            candidates[:, 1] - (ground_line.slope * candidates[:, 0] + ground_line.intercept)
                        )
                        inliers_mask = residuals <= args.gnd_residual

                overlay_ground_line(v_img, v_hist.shape, candidates, ground_line, inliers_mask)
                # 在原 disparity 显示图上也叠加 d=5/15/25 的地面参考线
                overlay_ground_on_disparity(disp_shown, ground_line, depth_h=v_hist.shape[0])

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
                    if ground_line is not None:
                        with open(save_dir / f"ground_{stamp}.txt", "w") as f:
                            f.write(f"slope={ground_line.slope:.6f}\n")
                            f.write(f"intercept={ground_line.intercept:.6f}\n")
                            f.write(f"inliers={ground_line.num_inliers}/{ground_line.num_candidates}\n")
                            f.write(f"inlier_ratio={ground_line.inlier_ratio:.3f}\n")
                    print(f"[✓] 已保存到 {save_dir}（{stamp}）")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
