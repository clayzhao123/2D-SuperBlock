from __future__ import annotations

def _line_svg(values: list[float], *, width: int = 560, height: int = 180, color: str = "#4f46e5") -> str:
    if not values:
        return ""
    lo = min(values)
    hi = max(values)
    span = hi - lo if hi != lo else 1.0
    points = []
    for idx, value in enumerate(values):
        x = (idx / max(1, len(values) - 1)) * (width - 10) + 5
        y = height - (((value - lo) / span) * (height - 20) + 10)
        points.append(f"{x:.2f},{y:.2f}")
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#f8fafc" stroke="#cbd5e1"/>'
        f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{" ".join(points)}"/>'
        "</svg>"
    )
