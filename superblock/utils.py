from __future__ import annotations

import math
import random
from typing import Iterable


GRID_MAX = 40
GRID_CELLS = 40
CENTER_CELL = (19, 19)


def set_seed(seed: int) -> None:
    random.seed(seed)


def shoelace_area(points: Iterable[tuple[int, int]]) -> float:
    pts = list(points)
    area2 = 0
    for i, (x1, y1) in enumerate(pts):
        x2, y2 = pts[(i + 1) % len(pts)]
        area2 += x1 * y2 - y1 * x2
    return abs(area2) / 2.0


def mse(pred: list[list[float]], target: list[list[float]]) -> float:
    acc = 0.0
    count = 0
    for prow, trow in zip(pred, target):
        for p, t in zip(prow, trow):
            diff = p - t
            acc += diff * diff
            count += 1
    return acc / max(1, count)


def exp(value: float) -> float:
    return math.exp(value)
