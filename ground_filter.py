"""
ground_filter.py — V-Disparity 地面拟合

本次改动（仅 GroundTracker 一处）
==================================
新增 hold_min_inlier_ratio（默认 0.4）：只有"高质量拟合"才配进入 HELD 持有。

问题背景：
  之前只要 RANSAC 拟合成功就 HELD 1.5 秒。但 min_inlier_ratio 只要 0.25，
  在实际没有地面的场景，RANSAC 会偶然凑出一条 25%~40% 内点的低质量线，
  照样触发 HELD —— 一不小心又凑一条，又 HELD，赖着不走。

解决：
  - 拟合成功时记住这条线的 inlier_ratio
  - 本帧失败 / 被突变拒绝时，检查"上一条成功线"的质量：
      inlier_ratio >= hold_min_inlier_ratio  → 高质量，值得 HELD
      inlier_ratio <  hold_min_inlier_ratio  → 偶然凑出的低质量线，直接 LOST
  这样真地面（ratio 通常 0.4~0.8）失败时能平滑 HELD；
  偶然假线（ratio 0.35~0.4）失败时立刻 LOST，不赖着。
"""
import time
from dataclasses import dataclass, replace
from typing import Any, Dict, Optional, Tuple

import numpy as np


def disparity_to_int(disp_frame: np.ndarray, max_disp: int, subpixel_scale: int = 1) -> np.ndarray:
    if subpixel_scale > 1:
        disp_int = disp_frame.astype(np.int32) // subpixel_scale
    else:
        disp_int = disp_frame.astype(np.int32)
    return np.clip(disp_int, 0, max_disp)


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
                                method: str = "subtract_percentile",
                                percentile: float = 50.0) -> np.ndarray:
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
    column_normalize: str = "subtract_percentile",
    column_norm_percentile: float = 50.0,              # ★ 新增：百分位参数
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
    residual_threshold: float = 2.5,
    min_inliers: int = 30,
    min_inlier_ratio: float = 0.35,
    min_slope: float = 0.12,
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
        hold_min_inlier_ratio: float = 0.4,   # ★ 新增：低于此质量的拟合不配 HELD
    ):
        self.hold_seconds = hold_seconds
        self.ema_alpha = ema_alpha
        self.max_slope_jump_ratio = max_slope_jump_ratio
        self.max_intercept_jump = max_intercept_jump
        self.warmup_frames = warmup_frames
        self.hold_min_inlier_ratio = hold_min_inlier_ratio

        self._smoothed: Optional[GroundLine] = None
        self._last_success_t: Optional[float] = None
        self._success_count: int = 0
        # ★ 记住"上一条成功拟合线"的 inlier_ratio，用于判断它配不配被 HELD
        self._last_inlier_ratio: float = 0.0

    def reset(self):
        self._smoothed = None
        self._last_success_t = None
        self._success_count = 0
        self._last_inlier_ratio = 0.0

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
            # ★ 记住这条成功线的质量（用 raw_line 的 ratio，不是 smoothed 的）
            self._last_inlier_ratio = raw_line.inlier_ratio

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

        # ★ 质量门槛：上一条成功线如果质量不够好（inlier_ratio 太低），
        #   说明它多半是偶然凑出来的，不值得 HELD，直接 LOST。
        if self._last_inlier_ratio < self.hold_min_inlier_ratio:
            self._smoothed = None
            self._last_success_t = None
            self._last_inlier_ratio = 0.0
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
        self._last_inlier_ratio = 0.0
        return TrackedGround(
            line=None, state="LOST",
            age_seconds=age,
            raw_fit_succeeded=raw_fit_succeeded,
            rejected_outlier=rejected_outlier,
        )


def compute_ground_mask_from_disparity(
    disparity: np.ndarray,
    ground_line: Optional[GroundLine],
    tolerance: float = 2.5,
    subpixel_scale: int = 1,
) -> np.ndarray:
    if ground_line is None:
        return np.zeros(disparity.shape, dtype=bool)

    if subpixel_scale > 1:
        disp_int = disparity.astype(np.float32) / subpixel_scale
    else:
        disp_int = disparity.astype(np.float32)

    y_idx = np.arange(disparity.shape[0], dtype=np.float32)
    threshold = (ground_line.slope * y_idx + ground_line.intercept + tolerance)[:, None]
    valid = disparity > 0
    return (disp_int <= threshold) & valid


def remove_ground_from_depth(depth_mm: np.ndarray, ground_mask: np.ndarray) -> np.ndarray:
    out = depth_mm.copy()
    out[ground_mask] = 0
    return out


def estimate_tracked_ground(
    disparity: np.ndarray,
    tracker: GroundTracker,
    max_disp: int,
    subpixel_scale: int = 1,
    params: Optional[Dict[str, Any]] = None,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    p = params or {}
    disp_int = disparity_to_int(disparity, max_disp, subpixel_scale)
    v_hist = compute_v_disparity(disp_int, max_disp)
    candidates = extract_ground_candidates(
        v_hist,
        y_start_ratio=p.get("y_start_ratio", 0.45),
        min_useful_disp=p.get("min_useful_disp", 2),
        min_peak_count=p.get("min_peak_count", 5),
        smooth_kernel=p.get("smooth_kernel", 3),
        max_peaks_per_row=p.get("max_peaks_per_row", 2),
        peak_min_distance=p.get("peak_min_distance", 3),
        column_normalize=p.get("column_normalize", "subtract_percentile"),
        column_norm_percentile=p.get("column_norm_percentile", 50.0),
    )
    raw_line = None
    min_inliers = int(p.get("min_inliers", 30))
    residual_threshold = float(p.get("residual_threshold", 2.5))
    if len(candidates) >= max(2, min_inliers // 2):
        raw_line = fit_ground_line_ransac(
            candidates,
            n_iterations=p.get("n_iterations", 300),
            residual_threshold=residual_threshold,
            min_inliers=min_inliers,
            min_inlier_ratio=p.get("min_inlier_ratio", 0.35),
            min_slope=p.get("min_slope", 0.12),
            max_slope=p.get("max_slope", 0.5),
            min_y_gap_ratio=p.get("min_y_gap_ratio", 0.3),
            min_bottom_disp=p.get("min_bottom_disp", 8.0),
        )
    tracked = tracker.update(raw_line, now=now)
    inliers_mask = None
    if raw_line is not None and len(candidates) > 0:
        residuals = np.abs(candidates[:, 1] - (raw_line.slope * candidates[:, 0] + raw_line.intercept))
        inliers_mask = residuals <= residual_threshold
    return {"disp_int": disp_int, "v_hist": v_hist, "candidates": candidates, "raw_line": raw_line, "tracked": tracked, "inliers_mask": inliers_mask}


def segment_ground(
    disparity: np.ndarray,
    depth_mm: Optional[np.ndarray],
    tracker: GroundTracker,
    max_disp: int,
    subpixel_scale: int = 1,
    params: Optional[Dict[str, Any]] = None,
    ground_tolerance: float = 2.5,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Layer 1 统一入口：识别地面，并输出 ground_mask / disp_no_ground / depth_no_ground。

    约定：
      - 地面识别、去地面都在本层完成
      - 下游 Layer 2（height_filter）应优先消费 depth_no_ground
      - 若 depth_mm is None，则仅返回 disparity 侧结果，便于 uv_disparity_viewer 做可视化
    """
    result = estimate_tracked_ground(
        disparity,
        tracker,
        max_disp,
        subpixel_scale=subpixel_scale,
        params=params,
        now=now,
    )
    tracked = result["tracked"]
    if tracked.line is not None and tracked.state in ("LOCKED", "HELD"):
        ground_mask = compute_ground_mask_from_disparity(
            disparity,
            tracked.line,
            tolerance=ground_tolerance,
            subpixel_scale=subpixel_scale,
        )
    else:
        ground_mask = np.zeros(disparity.shape, dtype=bool)

    disp_no_ground = disparity.copy()
    disp_no_ground[ground_mask] = 0

    result["ground_mask"] = ground_mask
    result["disp_no_ground"] = disp_no_ground
    result["depth_no_ground"] = None if depth_mm is None else remove_ground_from_depth(depth_mm, ground_mask)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 地面点去除
# ─────────────────────────────────────────────────────────────────────────────
def remove_ground_from_disparity(
    disparity: np.ndarray,
    ground_line: GroundLine,
    tolerance: float = 2.5,
    subpixel_scale: int = 1,
) -> np.ndarray:
    """
    根据已拟合的地面线，把 disparity 图里"贴着地面或更远"的像素置 0。

    原理：
      对每个像素 (y, x)：
        ground_disp = slope * y + intercept   # 这一行的地面理论视差
        actual_disp = disparity[y, x]
        如果 actual_disp <= ground_disp + tolerance  →  地面/背景，置 0
        如果 actual_disp >  ground_disp + tolerance  →  障碍物（站在地上），保留

      "<=" 而不是 "in ± tolerance"：比地面线小的视差表示更远，仍归为地面或背景，
      一并剔除。只有"比地面更近"（视差更大）的像素才是真的障碍物。

    参数：
      disparity      : 原始 disparity 图（任意整数 dtype，含 subpixel 时也行）
      ground_line    : 拟合得到的 GroundLine（slope/intercept 都是整数视差单位）
      tolerance      : 容差（整数视差单位，建议和 RANSAC 的 residual_threshold 一致）
      subpixel_scale : 如果 disparity 是 subpixel（× 32 整数存储），传 32；否则 1

    返回：
      与 disparity 同 shape 同 dtype，地面像素已置 0 的新数组。原数组不动。
      原本就是 0 的像素（无效区域）保持 0。
    """
    if ground_line is None:
        return disparity.copy()

    ground_mask = compute_ground_mask_from_disparity(
        disparity,
        ground_line,
        tolerance=tolerance,
        subpixel_scale=subpixel_scale,
    )
    out = disparity.copy()
    out[ground_mask] = 0
    return out
