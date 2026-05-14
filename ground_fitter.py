"""
ground_fitter.py — V-Disparity 地面拟合

本次改动（仅两处）
==================
1. normalize_v_hist_by_column 新增 "subtract_percentile" 方法，可指定百分位
   - 保留原有 subtract_median / subtract_min / none
2. extract_ground_candidates 新增 min_useful_disp（默认 2，屏蔽远景背景）

其他逻辑（RANSAC、突变拒绝、HELD/LOST 跟踪器）保持原样。
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


def estimate_ground_slope_range(fy_pixels, baseline_m, camera_height_m, height_h, cy_pixels=None):
    typical_slope = 0.045 / max(camera_height_m, 0.05)
    return typical_slope * 0.5, typical_slope * 1.8


# ─────────────────────────────────────────────────────────────────────────────
# ★ 列归一化：抑制 V-Disparity 中的墙体垂直亮带
# ─────────────────────────────────────────────────────────────────────────────
def normalize_v_hist_by_column(v_hist: np.ndarray,
                                method: str = "subtract_median",
                                percentile: float = 25.0) -> np.ndarray:
    """
    对 V-Disparity 直方图每列减去某个统计量，削弱墙体垂直亮带。

    method:
      "subtract_percentile"：减去每列 percentile% 分位数（百分位可调）
      "subtract_median"    ：减去每列中位数（等价于 percentile=50）
      "subtract_min"       ：减去每列最小值（等价于 percentile=0，最弱）
      "none"               ：不归一化

    percentile 仅在 method="subtract_percentile" 时生效，取值 0-100。
    """
    if method == "none":
        return v_hist

    v = v_hist.astype(np.float32)
    if method == "subtract_percentile":
        col_stat = np.percentile(v, percentile, axis=0, keepdims=True)
    elif method == "subtract_median":
        col_stat = np.median(v, axis=0, keepdims=True)
    elif method == "subtract_min":
        col_stat = v.min(axis=0, keepdims=True)
    else:
        raise ValueError(f"Unknown method: {method}")

    v_norm = np.clip(v - col_stat, 0, None)
    return v_norm.astype(np.uint32)


# ─────────────────────────────────────────────────────────────────────────────
# 候选点提取
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
    min_useful_disp: int = 2,                          # ★ 新增：屏蔽远景背景
    min_peak_count: int = 5,
    smooth_kernel: int = 3,
    max_peaks_per_row: int = 2,
    peak_min_distance: int = 3,
    column_normalize: str = "subtract_median",
    column_norm_percentile: float = 25.0,              # ★ 新增：百分位参数
) -> np.ndarray:
    """
    每行多个局部峰值，返回 (N, 2) [y, disp]。
    """
    v_hist_proc = normalize_v_hist_by_column(
        v_hist, method=column_normalize, percentile=column_norm_percentile
    )

    h, n_disp_bins = v_hist_proc.shape
    y_start = max(0, int(h * y_start_ratio))
    y_end = min(h, int(h * y_end_ratio))

    if smooth_kernel > 1:
        k = smooth_kernel
        kernel = np.ones(k, dtype=np.float32) / k

    # ★ 屏蔽 disp 太小的列（远景背景）
    actual_start = max(min_disp, min_useful_disp)

    candidates = []
    for y in range(y_start, y_end):
        row = v_hist_proc[y, actual_start:].astype(np.float32)
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
            d = int(p_idx + actual_start)
            candidates.append((y, d))

    return np.asarray(candidates, dtype=np.float32) if candidates else np.empty((0, 2), np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# RANSAC 拟合（未改）
# ─────────────────────────────────────────────────────────────────────────────
def fit_ground_line_ransac(
    points: np.ndarray,
    n_iterations: int = 300,
    residual_threshold: float = 1.5,
    min_inliers: int = 30,
    min_inlier_ratio: float = 0.25,
    min_slope: float = 0.09,
    max_slope: float = 0.5,
    min_y_gap_ratio: float = 0.3,
    bottom_disp_check: bool = True,
    min_bottom_disp: float = 8.0,
    seed: Optional[int] = None,
) -> Optional[GroundLine]:
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

    y_in = y_coords[best_inliers_mask]
    d_in = d_coords[best_inliers_mask]
    a, b = np.polyfit(y_in, d_in, 1)
    if a < min_slope or a > max_slope:
        a, b = best_slope, best_intercept

    if bottom_disp_check:
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
# 时序平滑跟踪器（未改）
# ─────────────────────────────────────────────────────────────────────────────
class GroundTracker:
    def __init__(
        self,
        hold_seconds: float = 1.5,
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
