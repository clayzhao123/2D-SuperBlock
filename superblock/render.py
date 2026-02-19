from __future__ import annotations

from .utils import CENTER_CELL, GRID_MAX


def _rotate_minus_90(vx: float, vy: float) -> tuple[float, float]:
    return vy, -vx


def render_view(
    points: list[tuple[int, int]],
    *,
    height: int = 32,
    width: int = 16,
    strip_len: float = 5.0,
    perspective_p: float = 2.0,
    alpha: float = 0.35,
    subsamples: int = 2,
    visible_cells: list[tuple[int, int]] | None = None,
) -> list[list[float]]:
    p0x, p0y = points[0]
    if len(points) > 1:
        p1x, p1y = points[1]
    else:
        p1x, p1y = p0x + 1, p0y

    ex, ey = p1x - p0x, p1y - p0y
    e_norm = (ex * ex + ey * ey) ** 0.5
    if e_norm == 0:
        raise ValueError("Observation edge has zero length.")
    eux, euy = ex / e_norm, ey / e_norm
    nux, nuy = _rotate_minus_90(eux, euy)

    targets = set(visible_cells or [CENTER_CELL])

    img = [[0.0 for _ in range(width)] for _ in range(height)]
    offsets = [(k + 0.5) / subsamples for k in range(subsamples)]

    for i in range(height):
        for j in range(width):
            pix_acc = 0.0
            for oy in offsets:
                row_pos = (i + (oy - 0.5)) / max(height - 1, 1)
                row_pos = min(1.0, max(0.0, row_pos))
                d_sub = (row_pos**perspective_p) * strip_len
                scale_sub = 1.0 / (1.0 + alpha * d_sub)
                for ox in offsets:
                    x_img = ((j + (ox - 0.5)) / width) - 0.5
                    s = 0.5 + x_img / scale_sub
                    if s < 0.0 or s > 1.0:
                        continue
                    x = p0x + eux * s + nux * d_sub
                    y = p0y + euy * s + nuy * d_sub
                    if x < 0.0 or y < 0.0 or x >= GRID_MAX or y >= GRID_MAX:
                        continue
                    cell = (int(x), int(y))
                    if cell in targets:
                        pix_acc += 1.0
            img[i][j] = min(1.0, max(0.0, pix_acc / (subsamples * subsamples)))

    return img
