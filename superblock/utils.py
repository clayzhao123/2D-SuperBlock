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


def _to_points(points_or_state: list[tuple[int, int]] | list[int]) -> list[tuple[int, int]]:
    if points_or_state and isinstance(points_or_state[0], int):
        state = points_or_state  # type: ignore[assignment]
        return [(int(state[i]), int(state[i + 1])) for i in range(0, len(state), 2)]
    points = points_or_state  # type: ignore[assignment]
    return [(int(x), int(y)) for x, y in points]


def occupied_cells(points_or_state: list[tuple[int, int]] | list[int]) -> list[tuple[int, int]]:
    """Return the four integer grid cells currently occupied by the block."""
    return _to_points(points_or_state)


def position_key(points_or_state: list[tuple[int, int]] | list[int]) -> tuple[int, int]:
    """Return a stable discrete position key shared by foraging + curiosity logic.

    We use the shape anchor (min_x, min_y) to avoid Python banker rounding at *.5.
    """
    pts = _to_points(points_or_state)
    return min(x for x, _ in pts), min(y for _, y in pts)
