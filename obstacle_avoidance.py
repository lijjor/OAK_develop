"""
obstacle_avoidance.py — 简单的左中右三区避障算法

设计思路
========
1. 把深度图横向均分三列（左/中/右），可选只取中间高度区间（去掉天花板/地板）
2. 每个区域计算"代表距离":
   - 用低分位数 (默认 10%) 而非 min(避免单点噪声) 也非 mean(被远处稀释)
   - 忽略 0 (无效像素) 和超出阈值范围的值
3. 与三档阈值比较得出每个区域的状态: SAFE / WARN / STOP
4. 综合决策给出建议动作: FORWARD / TURN_LEFT / TURN_RIGHT / STOP

输出结构
========
ZoneInfo:
  name: "left" / "center" / "right"
  rep_mm: 代表距离 (mm)
  valid_ratio: 该区域有效像素占比 (0.0-1.0)
  state: "SAFE" / "WARN" / "STOP" / "UNKNOWN"
  bbox: (x1, y1, x2, y2) 用于可视化

AvoidanceResult:
  zones: [ZoneInfo, ZoneInfo, ZoneInfo]
  action: "FORWARD" / "TURN_LEFT" / "TURN_RIGHT" / "STOP"
  reason: 决策理由（人类可读）
"""
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class ZoneInfo:
    name: str
    rep_mm: int          # 代表距离, 0 表示无有效数据
    valid_ratio: float   # 有效像素占比
    state: str           # SAFE / WARN / STOP / UNKNOWN
    bbox: Tuple[int, int, int, int]  # (x1,y1,x2,y2)


@dataclass
class AvoidanceResult:
    zones: List[ZoneInfo]
    action: str          # FORWARD / TURN_LEFT / TURN_RIGHT / STOP
    reason: str


# ─────────────────────────────────────────────────────────────────────────────
# 状态颜色（BGR），用于可视化
# ─────────────────────────────────────────────────────────────────────────────
STATE_COLORS = {
    "SAFE":    (0, 200, 0),      # 绿
    "WARN":    (0, 200, 255),    # 黄
    "STOP":    (0, 0, 255),      # 红
    "UNKNOWN": (128, 128, 128),  # 灰
}


class ObstacleAvoidance:
    def __init__(
        self,
        stop_mm: int = 600,           # 小于此距离 -> STOP（紧急停车）
        warn_mm: int = 1200,          # 小于此距离 -> WARN（减速/警戒）
        # 大于 warn_mm -> SAFE
        valid_min_mm: int = 200,      # 小于此距离的点视为噪声/盲区
        valid_max_mm: int = 8000,     # 大于此距离的点不参与避障判断
        percentile: float = 10.0,     # 用第几百分位数作为代表距离 (0-100)
        min_valid_ratio: float = 0.10,  # 区域有效像素占比低于此值视为 UNKNOWN
        roi_top: float = 0.20,        # ROI 顶部裁剪比例 (去天花板)
        roi_bottom: float = 0.85,     # ROI 底部裁剪比例 (去地板)
        zone_split: Tuple[float, float] = (1/3, 2/3),  # 左中分界、中右分界
    ):
        assert valid_min_mm < valid_max_mm
        assert stop_mm < warn_mm
        assert 0 <= roi_top < roi_bottom <= 1.0
        assert 0 < zone_split[0] < zone_split[1] < 1
        self.stop_mm = stop_mm
        self.warn_mm = warn_mm
        self.valid_min = valid_min_mm
        self.valid_max = valid_max_mm
        self.percentile = percentile
        self.min_valid_ratio = min_valid_ratio
        self.roi_top = roi_top
        self.roi_bottom = roi_bottom
        self.zone_split = zone_split

    # ─────────────────────────────────────────────────────────────────
    # 单帧主入口
    # ─────────────────────────────────────────────────────────────────
    def process(self, depth_mm: np.ndarray) -> AvoidanceResult:
        """depth_mm: HxW uint16 深度图（毫米）"""
        h, w = depth_mm.shape[:2]
        y1 = int(h * self.roi_top)
        y2 = int(h * self.roi_bottom)
        x_l = int(w * self.zone_split[0])
        x_r = int(w * self.zone_split[1])

        zones = [
            self._analyze_zone("left",   depth_mm[y1:y2, 0:x_l],   (0,   y1, x_l, y2)),
            self._analyze_zone("center", depth_mm[y1:y2, x_l:x_r], (x_l, y1, x_r, y2)),
            self._analyze_zone("right",  depth_mm[y1:y2, x_r:w],   (x_r, y1, w,   y2)),
        ]
        action, reason = self._decide(zones)
        return AvoidanceResult(zones=zones, action=action, reason=reason)

    # ─────────────────────────────────────────────────────────────────
    # 区域分析
    # ─────────────────────────────────────────────────────────────────
    def _analyze_zone(self, name, region, bbox) -> ZoneInfo:
        # 过滤无效像素 + 范围外
        mask = (region >= self.valid_min) & (region <= self.valid_max)
        valid = region[mask]
        total = region.size
        valid_ratio = valid.size / total if total > 0 else 0.0

        if valid_ratio < self.min_valid_ratio:
            # 有效像素太少，无法判断（可能是黑墙或全部超距）
            return ZoneInfo(name=name, rep_mm=0, valid_ratio=valid_ratio,
                            state="UNKNOWN", bbox=bbox)

        rep = int(np.percentile(valid, self.percentile))

        if rep < self.stop_mm:
            state = "STOP"
        elif rep < self.warn_mm:
            state = "WARN"
        else:
            state = "SAFE"

        return ZoneInfo(name=name, rep_mm=rep, valid_ratio=valid_ratio,
                        state=state, bbox=bbox)

    # ─────────────────────────────────────────────────────────────────
    # 决策（左中右三区简单版）
    # ─────────────────────────────────────────────────────────────────
    def _decide(self, zones: List[ZoneInfo]) -> Tuple[str, str]:
        left, center, right = zones[0], zones[1], zones[2]

        # 1) 中间 STOP -> 必须避让
        if center.state == "STOP":
            # 左右选一个 SAFE/UNKNOWN 的方向
            #   UNKNOWN 也可考虑（可能是远处空旷），但优先 SAFE
            left_ok  = left.state in ("SAFE", "WARN")
            right_ok = right.state in ("SAFE", "WARN")
            if left_ok and right_ok:
                # 两边都可走，选距离更远的一侧
                if left.rep_mm >= right.rep_mm:
                    return "TURN_LEFT", f"中间过近({center.rep_mm}mm)，左侧更宽敞"
                return "TURN_RIGHT", f"中间过近({center.rep_mm}mm)，右侧更宽敞"
            if left_ok:
                return "TURN_LEFT",  f"中间过近({center.rep_mm}mm)，向左避让"
            if right_ok:
                return "TURN_RIGHT", f"中间过近({center.rep_mm}mm)，向右避让"
            return "STOP", f"三区皆危险（左{left.state}/中{center.state}/右{right.state}）"

        # 2) 中间 WARN -> 减速但仍可前进；如某侧明显更远可主动绕
        if center.state == "WARN":
            # 比较左右是否有明显更优的路径（距离 > 中间 + 500mm）
            if left.state == "SAFE" and left.rep_mm > center.rep_mm + 500:
                return "TURN_LEFT", f"中间警戒({center.rep_mm}mm)，左侧更优"
            if right.state == "SAFE" and right.rep_mm > center.rep_mm + 500:
                return "TURN_RIGHT", f"中间警戒({center.rep_mm}mm)，右侧更优"
            return "FORWARD", f"中间警戒({center.rep_mm}mm)，谨慎前进"

        # 3) 中间 SAFE -> 前进
        if center.state == "SAFE":
            return "FORWARD", f"前方畅通({center.rep_mm}mm)"

        # 4) 中间 UNKNOWN -> 谨慎前进或停止
        #   如果两侧都 SAFE 也可继续，否则停下
        if left.state == "SAFE" and right.state == "SAFE":
            return "FORWARD", "中间数据缺失，但两侧畅通"
        return "STOP", "中间数据缺失且两侧不安全"


# ─────────────────────────────────────────────────────────────────────────────
# 可视化：把避障结果画到一张已上色的图（如 disparity_color）上
# ─────────────────────────────────────────────────────────────────────────────
ACTION_TEXT_COLOR = {
    "FORWARD":    (0, 255, 0),
    "TURN_LEFT":  (255, 200, 0),
    "TURN_RIGHT": (255, 200, 0),
    "STOP":       (0, 0, 255),
}


def draw_avoidance_overlay(
    canvas: np.ndarray,
    result: AvoidanceResult,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    show_bbox: bool = True,
    show_zone_text: bool = True,
):
    """
    在 canvas 上绘制三区域框 + 状态 + 距离 + 顶部决策动作
    canvas: BGR 图（disparity_color 等）
    scale_x/y: 如果 canvas 已被 resize 过，需要传入缩放比例（depth 坐标 × scale）
    """
    overlay = canvas.copy()

    # 区域框
    if show_bbox:
        for z in result.zones:
            x1, y1, x2, y2 = z.bbox
            x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
            y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
            color = STATE_COLORS[z.state]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

    # 半透明叠加
    cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)

    # 区域文字（在每个 bbox 顶部居中显示）
    if show_zone_text:
        for z in result.zones:
            x1, y1, x2, y2 = z.bbox
            x1s, x2s = int(x1 * scale_x), int(x2 * scale_x)
            y1s = int(y1 * scale_y)
            color = STATE_COLORS[z.state]
            label = f"{z.name.upper()}: {z.state}"
            dist  = f"{z.rep_mm}mm" if z.rep_mm > 0 else "N/A"
            cx = (x1s + x2s) // 2
            # 文字阴影 + 主色
            for txt, dy in [(label, 18), (dist, 38)]:
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                tx = cx - tw // 2
                ty = y1s + dy
                cv2.putText(canvas, txt, (tx, ty),   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(canvas, txt, (tx, ty),   cv2.FONT_HERSHEY_SIMPLEX, 0.55, color,     1, cv2.LINE_AA)

    # 顶部决策横幅
    h, w = canvas.shape[:2]
    banner = f"ACTION: {result.action}   |   {result.reason}"
    color = ACTION_TEXT_COLOR.get(result.action, (255, 255, 255))
    cv2.rectangle(canvas, (0, h - 36), (w, h), (0, 0, 0), -1)
    cv2.putText(canvas, banner, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
