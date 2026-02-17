from __future__ import annotations

import csv
import html
import json
import os
import pickle
from pathlib import Path

from .buffer import ReplayBuffer
from .models import ForwardModel


def save_checkpoint(
    path: str,
    *,
    day_idx: int,
    model: ForwardModel,
    buffer: ReplayBuffer,
    history: list[dict[str, float]],
    visible_cells: list[tuple[int, int]],
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "day_idx": day_idx,
        "model": {
            "w1": model.w1,
            "b1": model.b1,
            "w2": model.w2,
            "b2": model.b2,
        },
        "buffer": buffer.items,
        "history": history,
        "visible_cells": visible_cells,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def load_checkpoint(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def write_history_csv(path: str, history: list[dict[str, float]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fields = ["day_idx", "invalid_move_ratio", "pos_mse", "exact_acc", "near_acc", "score", "visible_hits"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in history:
            out = {name: row.get(name, 0.0) for name in fields}
            out["day_idx"] = int(out["day_idx"])
            writer.writerow(out)


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


def write_dashboard(path: str, history: list[dict[str, float]], visible_cells: list[tuple[int, int]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    latest = history[-1] if history else None
    score_svg = _line_svg([row["score"] for row in history], color="#2563eb")
    mse_svg = _line_svg([row["pos_mse"] for row in history], color="#16a34a")
    trust_svg = _line_svg([row.get("visible_hits", 0.0) for row in history], color="#0f172a")

    rows = "\n".join(
        (
            "<tr>"
            f"<td>{int(row['day_idx'])}</td>"
            f"<td>{row['invalid_move_ratio']:.3f}</td>"
            f"<td>{row['pos_mse']:.6f}</td>"
            f"<td>{row['exact_acc']:.3f}</td>"
            f"<td>{row['near_acc']:.3f}</td>"
            f"<td>{row.get('visible_hits', 0.0):.0f}</td>"
            f"<td>{row['score']:.3f}</td>"
            "</tr>"
        )
        for row in history[-30:]
    )

    latest_block = "暂无数据"
    if latest:
        latest_block = (
            f"day={int(latest['day_idx'])} "
            f"score={latest['score']:.3f} "
            f"mse={latest['pos_mse']:.6f} "
            f"visible_hits={latest.get('visible_hits', 0.0):.0f}"
        )

    visible = html.escape(json.dumps(visible_cells, ensure_ascii=False))

    content = f"""<!doctype html>
<html lang=\"zh\">
<head>
  <meta charset=\"utf-8\" />
  <meta http-equiv=\"refresh\" content=\"3\" />
  <title>Superblock Training Dashboard</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; font-size: 13px; }}
    th {{ background: #f1f5f9; }}
    .meta {{ margin-bottom: 16px; color: #334155; }}
  </style>
</head>
<body>
  <h1>Superblock 训练可视化</h1>
  <div class=\"meta\">最新状态: {latest_block}</div>
  <div class=\"meta\">当前可视单元格: {visible}</div>
  <div class=\"grid\">
    <div>
      <h3>Score 曲线</h3>
      {score_svg}
    </div>
    <div>
      <h3>Pos MSE 曲线</h3>
      {mse_svg}
    </div>
    <div>
      <h3>信任图（可视单元格命中次数/天）</h3>
      {trust_svg}
    </div>
  </div>
  <h3>最近 30 天</h3>
  <table>
    <thead>
      <tr><th>day</th><th>invalid_ratio</th><th>pos_mse</th><th>exact_acc</th><th>near_acc</th><th>visible_hits</th><th>score</th></tr>
    </thead>
    <tbody>
      {rows}
    </tbody>
  </table>
</body>
</html>
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def checkpoint_exists(path: str) -> bool:
    return os.path.exists(path)
