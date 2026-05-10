
from typing import Dict, Tuple


class Box:
    def __init__(self, x1: float, y1: float, x2: float, y2: float) -> None:
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    @property
    def w(self) -> float:
        return max(1.0, self.x2 - self.x1)

    @property
    def h(self) -> float:
        return max(1.0, self.y2 - self.y1)

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0


def _clamp10(v: float) -> float:
    return max(0.0, min(10.0, v))


def _rule_of_thirds_score(box: Box, image_size: Tuple[int, int]) -> float:
    h, w = image_size
    points = [
        (w / 3.0, h / 3.0),
        (2 * w / 3.0, h / 3.0),
        (w / 3.0, 2 * h / 3.0),
        (2 * w / 3.0, 2 * h / 3.0),
    ]
    min_dist = min(((box.cx - px) ** 2 + (box.cy - py) ** 2) ** 0.5 for px, py in points)
    max_dist = (w**2 + h**2) ** 0.5 / 2.0
    return _clamp10(10.0 * (1.0 - min_dist / max_dist))


def _headroom_score(box: Box, image_size: Tuple[int, int]) -> float:
    h, _ = image_size
    top_margin = box.y1 / max(1.0, h)
    target = 0.12
    err = abs(top_margin - target)
    return _clamp10(10.0 * (1.0 - err / 0.2))


def _subject_ratio_score(box: Box, image_size: Tuple[int, int]) -> float:
    h, w = image_size
    ratio = (box.w * box.h) / max(1.0, (w * h))
    target = 0.28
    err = abs(ratio - target)
    return _clamp10(10.0 * (1.0 - err / 0.25))


def _horizontal_balance_score(box: Box, image_size: Tuple[int, int]) -> float:
    _, w = image_size
    center_offset = abs(box.cx / max(1.0, w) - 0.5)
    return _clamp10(10.0 * (1.0 - center_offset / 0.5))


def compute_rule_based_score(bbox: Dict[str, float], image_size: Tuple[int, int]) -> Dict[str, float]:
    box = Box(**bbox)
    score = (
        _rule_of_thirds_score(box, image_size)
        + _headroom_score(box, image_size)
        + _subject_ratio_score(box, image_size)
        + _horizontal_balance_score(box, image_size)
    ) / 4.0
    return {"score": round(_clamp10(score), 4)}
