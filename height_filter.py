"""
height_filter.py — Layer 2：按机身顶部过滤 depth

职责
====
**输入**：depth_no_ground（毫米深度图，0=无效，且地面已由 Layer 1 剔除）
**输出**：depth_for_avoid —— 去掉"高于机身顶部"的点后得到的 depth_mm

为什么不用离地高度
================
对轮腿机器人来说，相机离地高度会随着腿长 / 机身升降变化，因此：
- 不能再把"地面"当作固定参考
- 更稳定的参考量是"相机到机身顶部的固定距离"

所以 Layer 2 不再回答"这个点离地多高"，而是回答：
"这个点相对机身顶部，是不是已经太高了？"

坐标约定
========
相机系 c：z 朝前，x 朝右，y 朝下（OpenCV 标准）。
机身系 b：与机身刚性绑定，z 朝前，x 朝右，y 朝下。

当 cam_pitch_deg = 0 时，认为相机系与机身系对齐。
若相机相对机身有固定俯仰角 θ（向下看为负），则绕 x 轴做一次固定旋转：

    Y_b = cos(θ)·Y_c - sin(θ)·Z_c

像素 (u, v) 的相机系纵向分量为：

    Y_c = depth · (v - cy) / fy
    Z_c = depth

代入得：

    Y_b = depth · [ cos(θ)·(v-cy)/fy - sin(θ) ]

机身顶部参考
============
设：
- d_top = camera_to_top_m > 0，表示"相机到机身顶部"的固定距离

因为 y 轴向下为正，所以机身顶部平面在机身系中的 y 坐标是：

    Y_top = -d_top

定义"距机身顶部余量"（clearance_to_top）：

    C_top = Y_b - Y_top = Y_b + d_top

其物理意义：
- C_top > 0 ：点在机身顶部平面以下（保留）
- C_top = 0 ：点恰好落在机身顶部平面
- C_top < 0 ：点高于机身顶部（应剔除）

过滤规则
========
基础保留条件：
  1. valid_min_mm ≤ depth ≤ valid_max_mm
  2. clearance_to_top_m ≥ -top_tolerance_m

也就是：允许点比机身顶部再高出一点点容差，但太高的点直接剔除。

可选二重保障：动态地面下界
========================
如果后续下位机 / 电控能提供"髋电机中点到地面"的实时距离，那么可以进一步构造
机身系中的地面纵向坐标：

    Y_ground = origin_to_hip_mid_m + hip_mid_to_ground_m

其中：
  - origin_to_hip_mid_m：机身原点到髋电机中点的固定距离（实测）
  - hip_mid_to_ground_m：髋电机中点到地面的实时距离（电控提供）

定义点相对地面的净空：

    clearance_to_ground = Y_ground - Y_b

在 y 轴向下为正的坐标系里：
  - clearance_to_ground > 0 ：点在地面以上
  - clearance_to_ground = 0 ：点在地面上
  - clearance_to_ground < 0 ：点在地面以下

于是可选地再加一条保留条件：

    clearance_to_ground ≥ ground_tolerance_m

这样可以把"贴地/穿地"的点再过滤一遍，作为 Layer 1 去地面的二重保障。

注意：
  - 这个逻辑默认关闭（enable_dynamic_ground_guard = False）
  - 在上位机和电控数据没打通前，不会参与实际过滤
  - 只有同时满足下面两件事时，这条逻辑才会真正生效：
      1. cfg.enable_dynamic_ground_guard = True
      2. 每帧调用 set_dynamic_ground_measurement(...) 注入实时值

后续接电控时的最小使用方式
==========================
初始化时：

    cfg = HeightFilterConfig(
        camera_to_top_m=0.20,
        fy_pixels=fy,
        cy_pixels=cy,
        cam_pitch_deg=-8.0,
        enable_dynamic_ground_guard=True,
        origin_to_hip_mid_m=0.12,   # 例子：机身原点到髋电机中点的固定距离
        ground_tolerance_m=0.03,
    )
    hf = HeightFilter(cfg)

运行时每帧：

    hf.set_dynamic_ground_measurement(hip_mid_to_ground_m=x2)
    result = hf.filter(depth_no_ground)

如果当前帧电控没有给出有效数据，则传：

    hf.set_dynamic_ground_measurement(None)

这样 Layer 2 会自动退回到"只按机身顶部过滤"的基础模式。

公开 API
========
HeightFilterConfig（dataclass）
HeightFilter:
    .filter(depth_no_ground) -> Dict 包含：
        - depth_for_avoid       : 过滤后的 depth_mm
        - clearance_to_top_m    : 相对机身顶部的余量（米；NaN=无效）
        - clearance_to_ground_m : 相对地面净空（米；未启用时为 None）
        - body_vertical_m       : 点在机身系中的纵向坐标 Y_b（米；NaN=无效）
        - kept_mask             : bool，True=保留
"""
from dataclasses import dataclass
from typing import Dict, Optional, Any

import numpy as np


@dataclass
class HeightFilterConfig:
    camera_to_top_m: float               # 相机到机身顶部的固定距离（米，正值）
    fy_pixels: float                     # 相机内参 fy
    cy_pixels: float                     # 相机内参 cy
    cam_pitch_deg: float = 0.0           # 相机相对机身的固定俯仰角（向下看为负）
    valid_min_mm: int = 200              # 太近 → 噪声
    valid_max_mm: int = 8000             # 太远 → 不参与判断
    top_tolerance_m: float = 0.05        # 机身顶部上方仍允许多留 5cm 容差
    enable_dynamic_ground_guard: bool = False   # 默认关闭；打开后才会启用动态地面下界保障
    origin_to_hip_mid_m: Optional[float] = None # 机身原点到髋电机中点的固定距离（米，实测）
    ground_tolerance_m: float = 0.03            # 与动态地面的最小净空，小于此值的点也剔除


class HeightFilter:
    def __init__(self, cfg: HeightFilterConfig):
        assert cfg.fy_pixels > 0 and cfg.cy_pixels > 0, "需要相机内参 fy/cy"
        assert cfg.camera_to_top_m >= 0
        assert cfg.valid_min_mm < cfg.valid_max_mm
        assert cfg.ground_tolerance_m >= 0
        if cfg.enable_dynamic_ground_guard:
            assert cfg.origin_to_hip_mid_m is not None, (
                "启用动态地面保障时，必须显式填写 origin_to_hip_mid_m；"
                "不要依赖默认值。"
            )
        self.cfg = cfg
        # 预计算相机相对机身俯仰角的 cos/sin
        theta = np.deg2rad(cfg.cam_pitch_deg)
        self._cos_t = float(np.cos(theta))
        self._sin_t = float(np.sin(theta))
        # 预生成行向量（每张图都用得到）；按需 lazy init
        self._cached_v = None    # (h, 1) 的列向量
        self._cached_shape = None
        # 运行时由上层注入：髋电机中点到地面的实时距离（米）
        self._hip_mid_to_ground_m: Optional[float] = None

    def _get_v_minus_cy_over_fy(self, h):
        """返回 shape (h, 1) 的 (v-cy)/fy（每行一个值）。"""
        if self._cached_shape != h:
            v = np.arange(h, dtype=np.float32) - self.cfg.cy_pixels
            self._cached_v = (v / self.cfg.fy_pixels)[:, None]   # (h, 1)
            self._cached_shape = h
        return self._cached_v

    def compute_body_vertical_map(self, depth_mm: np.ndarray) -> np.ndarray:
        """
        计算每像素在机身系中的纵向坐标 Y_b（米，向下为正）。
        无效像素（depth=0）返回 NaN。
        """
        h, _ = depth_mm.shape
        depth_m = depth_mm.astype(np.float32) / 1000.0   # mm → m
        v_term = self._get_v_minus_cy_over_fy(h)          # (h, 1)

        # Y_b = depth * [ cos(theta)*(v-cy)/fy - sin(theta) ]
        if self._sin_t == 0.0 and self._cos_t == 1.0:
            body_vertical_m = depth_m * v_term
        else:
            body_vertical_m = depth_m * (self._cos_t * v_term - self._sin_t)

        # 无效像素标 NaN
        body_vertical_m = np.where(depth_mm > 0, body_vertical_m, np.nan).astype(np.float32)
        return body_vertical_m

    def compute_clearance_to_top_map(
        self,
        depth_mm: np.ndarray,
        body_vertical_m: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        计算每像素相对机身顶部的余量（米）：
          > 0  : 在机身顶部以下
          = 0  : 恰好落在机身顶部平面
          < 0  : 高于机身顶部
        无效像素（depth=0）返回 NaN。
        """
        if body_vertical_m is None:
            body_vertical_m = self.compute_body_vertical_map(depth_mm)
        # compute_body_vertical_map() 已经把无效像素标成 NaN，这里直接平移即可。
        return (body_vertical_m + self.cfg.camera_to_top_m).astype(np.float32)

    def set_dynamic_ground_measurement(self, hip_mid_to_ground_m: Optional[float]) -> None:
        """
        注入电控提供的"髋电机中点到地面"实时距离（米）。
        传 None 表示当前帧无可用数据。
        注意：即使这里传了值，若 cfg.enable_dynamic_ground_guard=False，
        这路数据也不会参与实际过滤。
        """
        self._hip_mid_to_ground_m = hip_mid_to_ground_m

    def get_dynamic_ground_y_m(self) -> Optional[float]:
        """
        返回机身系下的地面纵向坐标 Y_ground（米，向下为正）。
        若功能关闭或当前没有实时测量，则返回 None。
        """
        if not self.cfg.enable_dynamic_ground_guard:
            return None
        if self._hip_mid_to_ground_m is None:
            return None
        return float(self.cfg.origin_to_hip_mid_m + self._hip_mid_to_ground_m)

    def compute_clearance_to_ground_map(
        self,
        depth_mm: np.ndarray,
        body_vertical_m: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """
        若已启用并注入动态地面测量，返回每像素相对地面的净空（米）：
          > 0 ：在地面以上
          = 0 ：在地面上
          < 0 ：在地面以下
        否则返回 None。
        """
        ground_y_m = self.get_dynamic_ground_y_m()
        if ground_y_m is None:
            return None
        if body_vertical_m is None:
            body_vertical_m = self.compute_body_vertical_map(depth_mm)
        return (ground_y_m - body_vertical_m).astype(np.float32)

    def filter(
        self,
        depth_no_ground: np.ndarray,
    ) -> Dict[str, Any]:
        """
        主入口。
        参数：
          depth_no_ground : uint16 HxW（毫米），已由 Layer 1 去地面；0=无效/已剔除

        返回 dict：
          depth_for_avoid       : uint16 HxW，过滤后；0=被剔除
          clearance_to_top_m    : float32 HxW，相对机身顶部的余量（NaN=无效）
          clearance_to_ground_m : Optional[float32 HxW]，相对地面净空（未启用时为 None）
          body_vertical_m       : float32 HxW，点在机身系中的纵向坐标 Y_b（NaN=无效）
          kept_mask             : bool HxW，True=被保留
        """
        cfg = self.cfg
        body_vertical_m = self.compute_body_vertical_map(depth_no_ground)
        clearance_to_top_m = self.compute_clearance_to_top_map(depth_no_ground, body_vertical_m=body_vertical_m)

        # 各项过滤条件
        in_range = (depth_no_ground >= cfg.valid_min_mm) & (depth_no_ground <= cfg.valid_max_mm)
        top_ok = clearance_to_top_m >= -cfg.top_tolerance_m
        # clearance_to_top_m 里 NaN 与任何比较都得 False，所以已自动排除无效像素
        kept_mask = in_range & top_ok

        clearance_to_ground_m = self.compute_clearance_to_ground_map(
            depth_no_ground,
            body_vertical_m=body_vertical_m,
        )
        if clearance_to_ground_m is not None:
            ground_ok = clearance_to_ground_m >= cfg.ground_tolerance_m
            kept_mask &= ground_ok

        depth_for_avoid = np.where(kept_mask, depth_no_ground, 0).astype(depth_no_ground.dtype)

        return {
            "depth_for_avoid":       depth_for_avoid,
            "clearance_to_top_m":    clearance_to_top_m,
            "clearance_to_ground_m": clearance_to_ground_m,
            "body_vertical_m":       body_vertical_m,
            "kept_mask":             kept_mask,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 调试/可视化辅助
# ─────────────────────────────────────────────────────────────────────────────
def colorize_clearance_to_top_map(clearance_to_top_m: np.ndarray,
                                  min_m: float = -0.2,
                                  max_m: float = 0.2) -> np.ndarray:
    """
    把"相对机身顶部余量"可视化：
      - 负值（高于顶部）偏蓝
      - 正值（低于顶部）偏红
      - 无效（NaN）为黑

    默认把范围设成 [-0.2, +0.2]，使 C_top = 0（恰在机身顶部平面）
    落在配色中点，读图更直观。
    返回 BGR uint8。
    """
    import cv2
    h = clearance_to_top_m.copy()
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
