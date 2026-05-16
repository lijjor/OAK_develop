"""
height_filter.py — Layer 2：按真实世界高度过滤 depth

职责
====
**输入**：depth_no_ground（毫米深度图，0=无效，且地面已由 Layer 1 剔除）
**输出**：depth_for_avoid —— 仅保留"可能撞到机身"的像素（其余置 0）

坐标约定
========
相机系：z 朝前（光轴），x 朝右，y 朝下（OpenCV 标准）。
世界系：与 pitch=0 的相机系重合，原点投影在地面 cam_height_m 正下方。
pitch 角 θ（cam_pitch_deg）：相机绕世界 x 轴的旋转。
  **θ < 0 ⇒ 相机向下看**（光轴从水平面朝地面倾斜）。
  **θ > 0 ⇒ 相机向上看**。

公式推导
========
像素 (u, v) 在相机系：
    X_c = depth · (u - cx) / fx
    Y_c = depth · (v - cy) / fy
    Z_c = depth

把相机系点投到世界系（绕 x 轴 R_x(θ)）：
    Y_w = cos(θ)·Y_c - sin(θ)·Z_c

"地面以上高度" = cam_height_m - Y_w：
    Y_above_ground = cam_h - depth · [ cos(θ)·(v-cy)/fy  -  sin(θ) ]
                                       ^^^^^^^^^^^^^^^^^^^^^^^^^^
                                       注意是减 sin(θ)，不是加！
    θ=0 时退化为：cam_h - depth · (v-cy)/fy

【符号自检】
  θ = -10°（相机向下看），中心像素 v=cy，depth=1m：
    Y_above = cam_h - 1·[1·0 - sin(-10°)] = cam_h - 1·[0 - (-0.174)] = cam_h - 0.174m
  → 相机向下看，前方 1m 中心射线落点比相机低 0.174m。✓ 与直觉一致。

过滤规则
========
保留当且仅当：
  1. valid_min_mm ≤ depth ≤ valid_max_mm
  2. -underfloor_tolerance_m ≤ Y_above_ground ≤ robot_height_m + overhead_tolerance_m

公开 API
========
HeightFilterConfig（dataclass）
HeightFilter:
    .filter(depth_no_ground) -> Dict 包含：
        - depth_for_avoid         : 过滤后的 depth_mm
        - height_above_ground_m  : 离地高度（float32，米；NaN=无效）
        - kept_mask              : bool，True=保留
"""
from dataclasses import dataclass
from typing import Dict, Optional, Any

import numpy as np


@dataclass
class HeightFilterConfig:
    cam_height_m: float                  # 相机离地高度
    robot_height_m: float                # 机身高度（高于此值的物体不算障碍）
    fy_pixels: float                     # 相机内参 fy
    cy_pixels: float                     # 相机内参 cy
    cam_pitch_deg: float = 0.0           # 相机俯仰角（向下看为负）
    valid_min_mm: int = 200              # 太近 → 噪声
    valid_max_mm: int = 8000             # 太远 → 不参与判断
    overhead_tolerance_m: float = 0.05   # 机身高度上方多留 5cm 容差
    underfloor_tolerance_m: float = 0.05 
    # 地面下方多留 5cm 容差，- ground_fitter 的确已经把“地面类像素”剔除了
    # 但 height_filter 里剩下像素的“计算高度”仍然可能因为边界、噪声、标定误差而略小于 0
    # 这不是重复去地面，而是为了不把这些“轻微负高度的真实障碍边缘点”误删掉
    # 如果地面残留多，减参数；障碍物被削薄，放大参数


class HeightFilter:
    def __init__(self, cfg: HeightFilterConfig):
        assert cfg.fy_pixels > 0 and cfg.cy_pixels > 0, "需要相机内参 fy/cy"
        assert cfg.cam_height_m > 0
        assert cfg.robot_height_m > 0
        assert cfg.valid_min_mm < cfg.valid_max_mm
        self.cfg = cfg
        # 预计算俯仰角的 cos/sin
        theta = np.deg2rad(cfg.cam_pitch_deg)
        self._cos_t = float(np.cos(theta))
        self._sin_t = float(np.sin(theta))
        # 预生成行向量（每张图都用得到）；按需 lazy init
        self._cached_v = None    # (h, 1) 的列向量
        self._cached_shape = None

    def _get_v_minus_cy_over_fy(self, h):
        """返回 shape (h, 1) 的 (v-cy)/fy（每行一个值）。"""
        if self._cached_shape != h:
            v = np.arange(h, dtype=np.float32) - self.cfg.cy_pixels
            self._cached_v = (v / self.cfg.fy_pixels)[:, None]   # (h, 1)
            self._cached_shape = h
        return self._cached_v

    def compute_height_above_ground_map(self, depth_mm: np.ndarray) -> np.ndarray:
        """
        计算每像素的离地高度（米，地面以上为正）。
        无效像素（depth=0）返回 NaN。
        """
        h, _ = depth_mm.shape
        depth_m = depth_mm.astype(np.float32) / 1000.0   # mm → m
        v_term = self._get_v_minus_cy_over_fy(h)          # (h, 1)

        # Pitch=0 时：height_above_ground = cam_h - depth * (v-cy)/fy
        # Pitch≠0 时：height_above_ground = cam_h - depth * ( cos*(v-cy)/fy - sin )
        #   注意是「减 sin(θ)」。θ<0(相机向下看) 时 -sin(θ)>0，
        #   使中心射线落点离地更低，符合几何直觉。
        if self._sin_t == 0.0 and self._cos_t == 1.0:
            height_above_ground_m = self.cfg.cam_height_m - depth_m * v_term
        else:
            height_above_ground_m = self.cfg.cam_height_m - depth_m * (self._cos_t * v_term - self._sin_t)

        # 无效像素标 NaN
        height_above_ground_m = np.where(depth_mm > 0, height_above_ground_m, np.nan).astype(np.float32)
        return height_above_ground_m

    def filter(
        self,
        depth_no_ground: np.ndarray,
    ) -> Dict[str, Any]:
        """
        主入口。
        参数：
          depth_no_ground : uint16 HxW（毫米），已由 Layer 1 去地面；0=无效/已剔除

        返回 dict：
          depth_for_avoid        : uint16 HxW，过滤后；0=被剔除
          height_above_ground_m : float32 HxW，离地高度（NaN=无效）
          kept_mask             : bool HxW，True=被保留
        """
        cfg = self.cfg
        height_above_ground_m = self.compute_height_above_ground_map(depth_no_ground)

        # 各项过滤条件
        in_range = (depth_no_ground >= cfg.valid_min_mm) & (depth_no_ground <= cfg.valid_max_mm)

        upper = cfg.robot_height_m + cfg.overhead_tolerance_m
        lower = -cfg.underfloor_tolerance_m
        height_ok = (height_above_ground_m >= lower) & (height_above_ground_m <= upper)
        # height_above_ground_m 里 NaN 与任何比较都得 False，所以已自动排除无效像素

        kept_mask = in_range & height_ok

        depth_for_avoid = np.where(kept_mask, depth_no_ground, 0).astype(depth_no_ground.dtype)

        return {
            "depth_for_avoid":         depth_for_avoid,
            "height_above_ground_m":  height_above_ground_m,
            "kept_mask":              kept_mask,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 调试/可视化辅助
# ─────────────────────────────────────────────────────────────────────────────
def colorize_height_above_ground_map(height_above_ground_m: np.ndarray,
                                      min_m: float = -0.2,
                                      max_m: float = 2.0) -> np.ndarray:
    """
    把离地高度图可视化：低处偏蓝，高处偏红，无效（NaN）为黑。
    返回 BGR uint8。
    """
    import cv2
    h = height_above_ground_m.copy()
    valid = np.isfinite(h)
    h_norm = np.where(valid,
                      np.clip((h - min_m) / max(max_m - min_m, 1e-6), 0.0, 1.0),
                      0.0)
    h_u8 = (h_norm * 255).astype(np.uint8)
    color = cv2.applyColorMap(h_u8, cv2.COLORMAP_JET)
    color[~valid] = 0
    return color


def make_kept_mask_overlay(canvas_bgr: np.ndarray,
                            kept_mask: np.ndarray,
                            alpha: float = 0.55) -> np.ndarray:
    """
    在 canvas 上把"被保留的像素"染成绿色叠加，被剔除的不变。
    用于直观看高度过滤效果。
    """
    import cv2
    if canvas_bgr.shape[:2] != kept_mask.shape:
        kept_mask_r = cv2.resize(kept_mask.astype(np.uint8),
                                 (canvas_bgr.shape[1], canvas_bgr.shape[0]),
                                 interpolation=cv2.INTER_NEAREST).astype(bool)
    else:
        kept_mask_r = kept_mask
    overlay = canvas_bgr.copy()
    overlay[kept_mask_r] = (
        canvas_bgr[kept_mask_r] * (1 - alpha)
        + np.array([0, 255, 0], dtype=np.float32) * alpha
    ).astype(np.uint8)
    return overlay
