"""
ground_fitter.py — V-Disparity 地面拟合 v3

v3 重大改进（解决"墙被识别成地面"问题）
==========================================
1. 收紧 min_slope（默认 0.005 → 0.08）
   - 真实地面在 V-Disparity 上斜率约 0.10 ~ 0.30（相机高度 0.2-0.5m，400p 分辨率）
   - 之前 0.005 太宽容，水平方向的"远处墙体"也被拟合成"地面"

2. 根据相机内参 + 安装高度计算理论地面斜率
   - 提供 estimate_ground_slope_range() 辅助函数
   - 知道相机高度即可推出 slope 大致范围

3. 拟合后的合理性验证
   - 拟合的直线如果在画面下半（y_bottom）处，预测 disparity 必须 ≥ min_bottom_disp
     否则说明"近处地面应该 disparity 很大，但这条线预测的 disparity 太小"
     → 拒绝该拟合结果

4. inlier 比例要求收紧
   - inlier_ratio < 0.25 视为可疑，拒绝

5. 候选点提取更严格：只取每行 1-2 个最强峰值（之前最多 3 个）
   - 减少非地面（如墙体的水平窄峰）混入候选池
"""
import time
from dataclasses import dataclass, replace
from typing import Optional, Tuple

import numpy as np


@dataclass
class GroundLine:
    slope: float
    intercept: float
    num_inliers: int
    num_candidates: int
    inlier_ratio: float

    def predict_disp(self, y):
        return self.slope * y + self.intercept


@dataclass
class TrackedGround:
    line: Optional[GroundLine]
    state: str
    age_seconds: float
    raw_fit_succeeded: bool
    rejected_outlier: bool


# ─────────────────────────────────────────────────────────────────────────────
# 理论地面斜率估算（辅助函数）
# ─────────────────────────────────────────────────────────────────────────────
def estimate_ground_slope_range(
    fy_pixels: float,
    baseline_m: float,
    camera_height_m: float,
    height_h: int,
    cy_pixels: float = None,
):
    """
    根据相机参数估算地面在 V-Disparity 上的理论斜率。

    几何推导（相机光轴水平时）：
      物理点 (Y, Z)：Y = -camera_height（地面）
      像素 y - cy = fy * Y / Z = -fy * camera_height / Z
      disp = fx * baseline / Z  → Z = fx * baseline / disp
      代入：(y - cy) = -fy * camera_height * disp / (fx * baseline)
                    ≈ -fy * camera_height * disp / (fx * baseline)  (fx≈fy)
      所以：disp ≈ -(fy * baseline) / (camera_height * fy) * (y - cy)
                 ≈ baseline / camera_height * (y - cy) / 1.0  ... 简化版

      实际斜率 slope = d(disp)/d(y) ≈ baseline / camera_height （但单位与像素相关）

    精确版（针对 OAK 标定后的内参）：
      slope = baseline_m / camera_height_m × (像素/米相关因子)
            ≈ fx_pixels * baseline_m / (camera_height_m * fy_pixels) × ratio

    简化估算（OAK-D 400p 经验值）：
      相机高 0.3m → slope ≈ 0.15
      相机高 0.5m → slope ≈ 0.09
      相机高 0.2m → slope ≈ 0.22
    """
    # 经验公式：slope ∝ 1 / camera_height
    # 系数 0.045 是 OAK-D 400p 经验值（baseline 75mm）
    typical_slope = 0.045 / max(camera_height_m, 0.05)
    # 给一个上下浮动 50% 的范围
    return typical_slope * 0.5, typical_slope * 1.8


# ─────────────────────────────────────────────────────────────────────────────
# 候选点提取（v3 — 更严格）
# ─────────────────────────────────────────────────────────────────────────────
def _find_local_peaks_1d(arr: np.ndarray, min_height: float,
                          min_distance: int = 2) -> np.ndarray:
    n = arr.size
    if n < 3:
        return np.empty(0, dtype=np.int64)

    left  = arr[1:-1] >  arr[:-2]
    right = arr[1:-1] >= arr[2:]
    above = arr[1:-1] >= min_height
    peaks = np.where(left & right & above)[0] + 1

    if peaks.size == 0:
        return peaks

    if min_distance > 1 and peaks.size > 1:
        order = np.argsort(-arr[peaks])
        keep = np.ones(peaks.size, dtype=bool)
        for i in order:
            if not keep[i]:
                continue
            mask = np.abs(peaks - peaks[i]) < min_distance
            mask[i] = False
            keep &= ~mask
        peaks = peaks[keep]
        peaks.sort()

    return peaks


def extract_ground_candidates(
    v_hist: np.ndarray,
    y_start_ratio: float = 0.45,
    y_end_ratio: float = 1.0,
    min_disp: int = 1,
    min_peak_count: int = 5,
    smooth_kernel: int = 3,
    max_peaks_per_row: int = 2,      # v3: 3 → 2（减少非地面峰混入）
    peak_min_distance: int = 3,      # v3: 2 → 3（相邻峰更稀疏）
) -> np.ndarray:
    """每行多个局部峰值，返回 (N, 2) [y, disp]。"""
    h, n_disp_bins = v_hist.shape
    y_start = max(0, int(h * y_start_ratio))
    y_end = min(h, int(h * y_end_ratio))

    if smooth_kernel > 1:
        k = smooth_kernel
        kernel = np.ones(k, dtype=np.float32) / k

    candidates = []
    for y in range(y_start, y_end):
        row = v_hist[y, min_disp:].astype(np.float32)
        if row.size == 0:
            continue

        if smooth_kernel > 1:
            row_smooth = np.convolve(row, kernel, mode="same")
        else:
            row_smooth = row

        peaks = _find_local_peaks_1d(row_smooth,
                                      min_height=float(min_peak_count),
                                      min_distance=peak_min_distance)
        if peaks.size == 0:
            continue

        if peaks.size > max_peaks_per_row:
            top = np.argsort(-row_smooth[peaks])[:max_peaks_per_row]
            peaks = peaks[top]

        for p_idx in peaks:
            d = int(p_idx + min_disp)
            candidates.append((y, d))

    return np.asarray(candidates, dtype=np.float32) if candidates else np.empty((0, 2), np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# RANSAC 拟合（v3 — 严格斜率约束 + 合理性验证）
# ─────────────────────────────────────────────────────────────────────────────
def fit_ground_line_ransac(
    points: np.ndarray,
    n_iterations: int = 300,
    residual_threshold: float = 1.5,
    min_inliers: int = 30,
    min_inlier_ratio: float = 0.25,   # ★ v3 新增：inlier 比例下限
    min_slope: float = 0.08,          # ★ v3：0.005 → 0.08（关键）
    max_slope: float = 0.5,           # ★ v3：2.0 → 0.5（地面不会很陡）
    min_y_gap_ratio: float = 0.3,
    # ★ v3 新增：地面合理性验证 — 最底行预测视差必须 ≥ min_bottom_disp
    bottom_disp_check: bool = True,
    min_bottom_disp: float = 8.0,
    seed: Optional[int] = None,
) -> Optional[GroundLine]:
    """
    RANSAC 拟合 disp = slope × y + intercept。

    v3 多重过滤：
      - slope ∈ [min_slope, max_slope]：排除水平线（墙）和过陡线
      - inlier_ratio ≥ min_inlier_ratio：排除弱拟合
      - bottom_disp_check：最底行预测 disp 必须 ≥ min_bottom_disp（地面应该近）
    """
    n = len(points)
    if n < max(2, min_inliers // 5):
        return None

    rng = np.random.default_rng(seed)
    y_coords = points[:, 0]
    d_coords = points[:, 1]

    y_min, y_max = float(y_coords.min()), float(y_coords.max())
    y_range = y_max - y_min
    min_y_gap = y_range * min_y_gap_ratio

    best_inliers_mask = None
    best_slope = 0.0
    best_intercept = 0.0
    best_inlier_count = 0

    sort_idx = np.argsort(y_coords)
    y_sorted = y_coords[sort_idx]
    d_sorted = d_coords[sort_idx]

    half = n // 2
    upper = np.arange(0, half)
    lower = np.arange(half, n)

    for _ in range(n_iterations):
        if upper.size > 0 and lower.size > 0:
            i1 = rng.choice(upper)
            i2 = rng.choice(lower)
        else:
            ii = rng.choice(n, size=2, replace=False)
            i1, i2 = ii[0], ii[1]

        y1, d1 = y_sorted[i1], d_sorted[i1]
        y2, d2 = y_sorted[i2], d_sorted[i2]
        if y2 - y1 < min_y_gap or y1 == y2:
            continue

        slope = (d2 - d1) / (y2 - y1)
        # ★ 严格斜率过滤
        if slope < min_slope or slope > max_slope:
            continue
        intercept = d1 - slope * y1

        predicted = slope * y_coords + intercept
        residuals = np.abs(d_coords - predicted)
        inliers_mask = residuals <= residual_threshold
        inlier_count = int(inliers_mask.sum())

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_inliers_mask = inliers_mask
            best_slope = slope
            best_intercept = intercept

    if best_inlier_count < min_inliers:
        return None

    inlier_ratio = best_inlier_count / n
    if inlier_ratio < min_inlier_ratio:
        return None

    # 用所有 inlier 做最小二乘精炼
    y_in = y_coords[best_inliers_mask]
    d_in = d_coords[best_inliers_mask]
    a, b = np.polyfit(y_in, d_in, 1)
    if a < min_slope or a > max_slope:
        a, b = best_slope, best_intercept

    # ★ 合理性验证：在画面下方某 y，预测的 disparity 应该足够大（地面近 → disp 大）
    if bottom_disp_check:
        # 取所有候选点中最大的 y 作为"底部行"参考
        y_bottom = float(y_coords.max())
        disp_at_bottom = a * y_bottom + b
        if disp_at_bottom < min_bottom_disp:
            return None

    return GroundLine(
        slope=float(a),
        intercept=float(b),
        num_inliers=best_inlier_count,
        num_candidates=n,
        inlier_ratio=inlier_ratio,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 时序平滑跟踪器（不变）
# ─────────────────────────────────────────────────────────────────────────────
class GroundTracker:
    def __init__(
        self,
        hold_seconds: float = 2.0,
        ema_alpha: float = 0.4,
        max_slope_jump_ratio: float = 0.6,
        max_intercept_jump: float = 15.0,
        warmup_frames: int = 3,
    ):
        self.hold_seconds = hold_seconds
        self.ema_alpha = ema_alpha
        self.max_slope_jump_ratio = max_slope_jump_ratio
        self.max_intercept_jump = max_intercept_jump
        self.warmup_frames = warmup_frames

        self._smoothed: Optional[GroundLine] = None
        self._last_success_t: Optional[float] = None
        self._success_count: int = 0

    def reset(self):
        self._smoothed = None
        self._last_success_t = None
        self._success_count = 0

    def update(self, raw_line: Optional[GroundLine], now: Optional[float] = None) -> TrackedGround:
        if now is None:
            now = time.time()

        if raw_line is not None:
            rejected = False
            if self._smoothed is not None and self._success_count >= self.warmup_frames:
                slope_jump = abs(raw_line.slope - self._smoothed.slope)
                slope_jump_ratio = slope_jump / max(abs(self._smoothed.slope), 1e-6)
                intercept_jump = abs(raw_line.intercept - self._smoothed.intercept)

                if (slope_jump_ratio > self.max_slope_jump_ratio or
                        intercept_jump > self.max_intercept_jump):
                    rejected = True

            if rejected:
                return self._held_or_lost(now, raw_fit_succeeded=True, rejected_outlier=True)

            if self._smoothed is None:
                self._smoothed = raw_line
            else:
                a = self.ema_alpha
                self._smoothed = replace(
                    raw_line,
                    slope    = a * raw_line.slope     + (1 - a) * self._smoothed.slope,
                    intercept= a * raw_line.intercept + (1 - a) * self._smoothed.intercept,
                )
            self._last_success_t = now
            self._success_count += 1

            return TrackedGround(
                line=self._smoothed, state="LOCKED",
                age_seconds=0.0,
                raw_fit_succeeded=True, rejected_outlier=False,
            )

        return self._held_or_lost(now, raw_fit_succeeded=False, rejected_outlier=False)

    def _held_or_lost(self, now, raw_fit_succeeded, rejected_outlier):
        if self._smoothed is None or self._last_success_t is None:
            return TrackedGround(
                line=None, state="LOST",
                age_seconds=float("inf"),
                raw_fit_succeeded=raw_fit_succeeded,
                rejected_outlier=rejected_outlier,
            )
        age = now - self._last_success_t
        if age <= self.hold_seconds:
            return TrackedGround(
                line=self._smoothed, state="HELD",
                age_seconds=age,
                raw_fit_succeeded=raw_fit_succeeded,
                rejected_outlier=rejected_outlier,
            )
        self._smoothed = None
        self._last_success_t = None
        return TrackedGround(
            line=None, state="LOST",
            age_seconds=age,
            raw_fit_succeeded=raw_fit_succeeded,
            rejected_outlier=rejected_outlier,
        )
