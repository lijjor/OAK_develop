"""
ground_fitter.py — V-Disparity 地面直线拟合（RANSAC）

数据流
======
输入：V-Disparity 直方图 v_hist (H, max_disp+1) uint32
  v_hist[y, d] = 原图第 y 行中 disparity=d 的像素个数

步骤：
  1. extract_ground_candidates(v_hist, ...)
       从每行选最亮的 disparity 作为地面候选点 → [(y, disp), ...]
  2. fit_ground_line_ransac(points, ...)
       RANSAC 拟合直线 disp = slope * y + intercept
       返回 GroundLine(slope, intercept, inliers_mask, num_inliers)

地面直线方程
============
  disp = slope × y + intercept
  其中 y 是像素行号（0=画面顶），disp 是整数视差
  - slope > 0：y 越大（画面下方）disparity 越大（地面更近）→ 这是相机俯视/平视的常态
  - slope < 0：地面在画面上方，相机倒装时会出现
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class GroundLine:
    """V-Disparity 中拟合出的地面直线 disp = slope × y + intercept。"""
    slope: float
    intercept: float
    num_inliers: int
    num_candidates: int
    inlier_ratio: float

    def predict_disp(self, y):
        """给定像素行 y（标量或数组），返回该行地面的视差值。"""
        return self.slope * y + self.intercept


# ─────────────────────────────────────────────────────────────────────────────
# 步骤 1：候选点提取
# ─────────────────────────────────────────────────────────────────────────────
def extract_ground_candidates(
    v_hist: np.ndarray,
    y_start_ratio: float = 0.45,      # 从图像下方 55% 开始（地面通常在画面下半部）
    y_end_ratio: float = 1.0,          # 到最底
    min_disp: int = 1,                 # 排除 disparity=0（无效）
    min_peak_count: int = 5,           # 该行最大计数至少 5 个像素，否则跳过
    relative_peak_ratio: float = 0.5,  # 最大值必须占该行总和的 50% 以上，去除平坦行
) -> np.ndarray:
    """
    从 V-Disparity 直方图中提取地面候选点。
    每一行：找最大计数所在的 disparity 列；若该最大值满足条件，则记为一个候选点。

    返回 (N, 2) 的 ndarray，每行为 [y, disp]。N 可能为 0。
    """
    h, n_disp_bins = v_hist.shape
    y_start = max(0, int(h * y_start_ratio))
    y_end = min(h, int(h * y_end_ratio))

    candidates = []
    for y in range(y_start, y_end):
        row = v_hist[y, min_disp:]   # 跳过 disparity=0
        if row.size == 0:
            continue

        row_sum = int(row.sum())
        if row_sum < min_peak_count:
            continue

        peak_idx = int(np.argmax(row))
        peak_val = int(row[peak_idx])
        if peak_val < min_peak_count:
            continue
        if peak_val < row_sum * relative_peak_ratio:
            # 该行没有明显主导 disparity，是个"散开行"，跳过
            continue

        # 还原成 v_hist 中的 disparity 索引
        candidates.append((y, peak_idx + min_disp))

    return np.asarray(candidates, dtype=np.float32) if candidates else np.empty((0, 2), np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 步骤 2：RANSAC 直线拟合
# ─────────────────────────────────────────────────────────────────────────────
def fit_ground_line_ransac(
    points: np.ndarray,                # (N, 2) 列为 [y, disp]
    n_iterations: int = 200,
    residual_threshold: float = 1.5,   # 视差残差阈值（disp 单位）
    min_inliers: int = 30,             # 最少 inlier 数量才认为拟合成功
    min_slope: float = 0.005,          # 斜率下限（去除水平线，那是天花板/远处墙的特征）
    max_slope: float = 2.0,            # 斜率上限
    seed: Optional[int] = None,
) -> Optional[GroundLine]:
    """
    RANSAC 直线拟合：disp = slope × y + intercept

    返回 None 表示拟合失败（候选点太少 / 没找到足够 inlier）。
    """
    n = len(points)
    if n < max(2, min_inliers // 5):
        return None

    rng = np.random.default_rng(seed)
    y_coords = points[:, 0]
    d_coords = points[:, 1]

    best_inliers_mask = None
    best_slope = 0.0
    best_intercept = 0.0
    best_inlier_count = 0

    for _ in range(n_iterations):
        # 随机选两点
        idx = rng.choice(n, size=2, replace=False)
        y1, d1 = y_coords[idx[0]], d_coords[idx[0]]
        y2, d2 = y_coords[idx[1]], d_coords[idx[1]]
        if y1 == y2:
            continue

        slope = (d2 - d1) / (y2 - y1)
        if slope < min_slope or slope > max_slope:
            continue
        intercept = d1 - slope * y1

        # 计算所有点的残差
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

    # 用所有 inlier 做最小二乘精炼
    y_in = y_coords[best_inliers_mask]
    d_in = d_coords[best_inliers_mask]
    # 一次多项式拟合：d = a*y + b
    a, b = np.polyfit(y_in, d_in, 1)
    if a < min_slope or a > max_slope:
        # 精炼后斜率越界，退回到 RANSAC 原始结果
        a, b = best_slope, best_intercept

    return GroundLine(
        slope=float(a),
        intercept=float(b),
        num_inliers=best_inlier_count,
        num_candidates=n,
        inlier_ratio=best_inlier_count / n,
    )
