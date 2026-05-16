"""
obstacle_avoidance_3.py — Layer 3：纯三区评估和动作决策

职责（精简）
============
**只做**「左中右三区距离评估 + 动作决策」。
**不再做** ROI 裁剪、不再做天花板/地面过滤 —— 那些活归 Layer 2 (height_filter.py)。

输入约定
========
depth_for_avoid：已经过 Layer 1 (地面剔除) + Layer 2 (机身高度过滤) 的 depth_mm 图。
                 任何为 0 的像素都视为"无效/已剔除"，不参与判断。

输出结构
========
ZoneInfo:
  name: "left" / "center" / "right"
  rep_mm: 代表距离 (mm)
  valid_count: 区域内有效像素数量
  valid_ratio: 该区域有效像素占比 (0.0-1.0)
  state: "SAFE" / "WARN" / "STOP" / "UNKNOWN"
  bbox: (x1, y1, x2, y2)  整张图（无 ROI）的三列

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
    rep_mm: int
    valid_count: int
    valid_ratio: float
    state: str
    bbox: Tuple[int, int, int, int]


@dataclass
class AvoidanceResult:
    zones: List[ZoneInfo]
    action: str
    reason: str


STATE_COLORS = {
    "SAFE":    (0, 200, 0),
    "WARN":    (0, 200, 255),
    "STOP":    (0, 0, 255),
    "UNKNOWN": (128, 128, 128),
}


class ObstacleAvoidance:
    def __init__(
        self,
        stop_mm: int = 600,
        warn_mm: int = 1200,
        percentile: float = 10.0,
        min_valid_ratio: float = 0.10,
        min_valid_pixels: int = 200,
        zone_split: Tuple[float, float] = (1/3, 2/3),
    ):
        """
        参数：
          stop_mm        : 代表距离 < 此值 → STOP
          warn_mm        : 代表距离 < 此值 → WARN（stop_mm < warn_mm < SAFE）
          percentile     : 用第几百分位作代表距离（默认 10：取较近的 10% 像素）
          min_valid_ratio: 区域有效像素占比门槛
          min_valid_pixels: 区域最少有效像素数量门槛
          zone_split     : (左中分界, 中右分界) 的比例
        """
        assert stop_mm < warn_mm
        assert min_valid_pixels >= 1
        assert 0 < zone_split[0] < zone_split[1] < 1
        self.stop_mm = stop_mm
        self.warn_mm = warn_mm
        self.percentile = percentile
        self.min_valid_ratio = min_valid_ratio
        self.min_valid_pixels = min_valid_pixels
        self.zone_split = zone_split

    def process(self, depth_for_avoid: np.ndarray) -> AvoidanceResult:
        """
        输入 depth_for_avoid: HxW uint16，已经过 Layer 1+2 过滤，0=无效像素。
        """
        h, w = depth_for_avoid.shape[:2]
        x_l = int(w * self.zone_split[0])
        x_r = int(w * self.zone_split[1])

        zones = [
            self._analyze_zone("left",   depth_for_avoid[:, 0:x_l],   (0,   0, x_l, h)),
            self._analyze_zone("center", depth_for_avoid[:, x_l:x_r], (x_l, 0, x_r, h)),
            self._analyze_zone("right",  depth_for_avoid[:, x_r:w],   (x_r, 0, w,   h)),
        ]
        action, reason = self._decide(zones)
        return AvoidanceResult(zones=zones, action=action, reason=reason)

    def _analyze_zone(self, name, region, bbox) -> ZoneInfo:
        # Layer 1/2 已完成地面、机身高度与距离范围过滤。
        # 这里把 0 统一视为无效/已剔除，其余非 0 像素直接参与避障统计。
        mask = region > 0
        valid = region[mask]
        total = region.size
        valid_count = int(valid.size)
        valid_ratio = valid_count / total if total > 0 else 0.0

        valid_enough = (
            valid_ratio >= self.min_valid_ratio or
            valid_count >= self.min_valid_pixels
        )
        if not valid_enough:
            return ZoneInfo(name=name, rep_mm=0, valid_count=valid_count, valid_ratio=valid_ratio,
                            state="UNKNOWN", bbox=bbox)

        rep = int(np.percentile(valid, self.percentile))
        if rep < self.stop_mm:
            state = "STOP"
        elif rep < self.warn_mm:
            state = "WARN"
        else:
            state = "SAFE"

        return ZoneInfo(name=name, rep_mm=rep, valid_count=valid_count, valid_ratio=valid_ratio,
                        state=state, bbox=bbox)

    def _decide(self, zones: List[ZoneInfo]) -> Tuple[str, str]:
        left, center, right = zones[0], zones[1], zones[2]

        if center.state == "STOP":
            left_ok  = left.state in ("SAFE", "WARN")
            right_ok = right.state in ("SAFE", "WARN")
            if left_ok and right_ok:
                if left.rep_mm >= right.rep_mm:
                    return "TURN_LEFT", f"中间过近({center.rep_mm}mm)，左侧更宽敞"
                return "TURN_RIGHT", f"中间过近({center.rep_mm}mm)，右侧更宽敞"
            if left_ok:
                return "TURN_LEFT",  f"中间过近({center.rep_mm}mm)，向左避让"
            if right_ok:
                return "TURN_RIGHT", f"中间过近({center.rep_mm}mm)，向右避让"
            return "STOP", f"三区皆危险（左{left.state}/中{center.state}/右{right.state}）"

        if center.state == "WARN":
            if left.state == "SAFE" and left.rep_mm > center.rep_mm + 500:
                return "TURN_LEFT", f"中间警戒({center.rep_mm}mm)，左侧更优"
            if right.state == "SAFE" and right.rep_mm > center.rep_mm + 500:
                return "TURN_RIGHT", f"中间警戒({center.rep_mm}mm)，右侧更优"
            return "FORWARD", f"中间警戒({center.rep_mm}mm)，谨慎前进"

        if center.state == "SAFE":
            return "FORWARD", f"前方畅通({center.rep_mm}mm)"

        # center UNKNOWN
        if left.state == "SAFE" and right.state == "SAFE":
            if left.rep_mm >= right.rep_mm:
                return "TURN_LEFT", "中间数据缺失，优先向更宽敞的左侧试探绕行"
            return "TURN_RIGHT", "中间数据缺失，优先向更宽敞的右侧试探绕行"
        if left.state == "SAFE" and right.state == "UNKNOWN":
            return "TURN_LEFT", "中间数据缺失，左侧安全，优先向左绕行"
        if right.state == "SAFE" and left.state == "UNKNOWN":
            return "TURN_RIGHT", "中间数据缺失，右侧安全，优先向右绕行"
        if left.state == "SAFE":
            return "TURN_LEFT", "中间数据缺失，左侧安全，优先向左绕行"
        if right.state == "SAFE":
            return "TURN_RIGHT", "中间数据缺失，右侧安全，优先向右绕行"
        if left.state == "WARN" and right.state == "WARN":
            return "STOP", "中间数据缺失，且左右都仅为警戒，保守停车"
        return "STOP", "中间数据缺失且两侧不安全"


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
    在 canvas 上绘制三区域框 + 状态 + 距离 + 顶部决策动作。
    scale_x/y: 如果 canvas 已被 resize（depth 坐标 × scale 才能映射到 canvas）。
    """
    overlay = canvas.copy()

    if show_bbox:
        for z in result.zones:
            x1, y1, x2, y2 = z.bbox
            x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
            y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
            color = STATE_COLORS[z.state]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

    cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)

    if show_zone_text:
        for z in result.zones:
            x1, y1, x2, y2 = z.bbox
            x1s, x2s = int(x1 * scale_x), int(x2 * scale_x)
            y1s = int(y1 * scale_y)
            color = STATE_COLORS[z.state]
            label = f"{z.name.upper()}: {z.state}"
            dist  = f"{z.rep_mm}mm" if z.rep_mm > 0 else "N/A"
            cx = (x1s + x2s) // 2
            for txt, dy in [(label, 18), (dist, 38)]:
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                tx = cx - tw // 2
                ty = y1s + dy
                cv2.putText(canvas, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(canvas, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color,     1, cv2.LINE_AA)

    h, w = canvas.shape[:2]
    banner = f"ACTION: {result.action}   |   {result.reason}"
    color = ACTION_TEXT_COLOR.get(result.action, (255, 255, 255))
    cv2.rectangle(canvas, (0, h - 36), (w, h), (0, 0, 0), -1)
    cv2.putText(canvas, banner, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
