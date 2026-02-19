from __future__ import annotations

import csv
import html
import json
from pathlib import Path

from .monitor import _line_svg, load_checkpoint


def _load_motion(ckpt_path: str) -> tuple[list[dict[str, float]], list[tuple[int, int]]]:
    path = Path(ckpt_path)
    if not path.exists():
        return [], []
    payload = load_checkpoint(str(path))
    history = payload.get("history", [])
    visible_cells = [tuple(cell) for cell in payload.get("visible_cells", [])]
    return history, visible_cells


def _parse_curiosity_report(report_path: str) -> str:
    path = Path(report_path)
    if not path.exists():
        return "未运行 explore"
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return "未运行 explore"
    # 格式示例: visited_cells=123 total_cells=1681 coverage=0.0732
    parts = dict(item.split("=", 1) for item in text.split() if "=" in item)
    visited = parts.get("visited_cells", "?")
    coverage = parts.get("coverage", "?")
    return f"visited_cells={visited}, coverage={coverage}"


def _load_forage_history(metrics_csv_path: str, forage_ckpt_path: str) -> list[dict[str, float]]:
    csv_path = Path(metrics_csv_path)
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        out: list[dict[str, float]] = []
        for row in rows:
            converted: dict[str, float] = {}
            for k, v in row.items():
                if v is None or v == "":
                    continue
                try:
                    converted[k] = float(v)
                except ValueError:
                    continue
            out.append(converted)
        return out

    ckpt_path = Path(forage_ckpt_path)
    if not ckpt_path.exists():
        return []
    payload = load_checkpoint(str(ckpt_path))
    return payload.get("history", [])




def _load_simple_rate_history(metrics_csv_path: str) -> list[dict[str, float]]:
    csv_path = Path(metrics_csv_path)
    if not csv_path.exists():
        return []
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out: list[dict[str, float]] = []
    for row in rows:
        converted: dict[str, float] = {}
        for k, v in row.items():
            if v is None or v == "":
                continue
            try:
                converted[k] = float(v)
            except ValueError:
                continue
        out.append(converted)
    return out


def write_master_dashboard(
    out_path: str = "artifacts/master_dashboard.html",
    *,
    motion_ckpt_path: str = "artifacts/train.ckpt",
    curiosity_report_path: str = "artifacts/curiosity_report.txt",
    forage_metrics_csv_path: str = "artifacts/forage_metrics.csv",
    forage_ckpt_path: str = "artifacts/forage.ckpt",
    motion_dashboard_path: str = "artifacts/dashboard.html",
    forage_dashboard_path: str = "artifacts/forage_dashboard.html",
    evade_metrics_csv_path: str = "artifacts/evade_metrics.csv",
    evade_dashboard_path: str = "artifacts/evade_dashboard.html",
) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    motion_history, visible_cells = _load_motion(motion_ckpt_path)
    motion_score_svg = _line_svg([row.get("score", 0.0) for row in motion_history], color="#2563eb")
    motion_mse_svg = _line_svg([row.get("pos_mse", 0.0) for row in motion_history], color="#16a34a")

    curiosity_block = _parse_curiosity_report(curiosity_report_path)

    forage_history = _load_forage_history(forage_metrics_csv_path, forage_ckpt_path)
    rate_values = [
        row.get("hungry_success_rate_total", row.get("success_rate", 0.0))
        for row in forage_history
    ]
    latency_values = [
        row.get("hungry_latency_mean_total", -1.0)
        for row in forage_history
        if row.get("hungry_latency_mean_total", -1.0) >= 0
    ]
    deaths_values = [row.get("deaths_total", 0.0) for row in forage_history]

    forage_rate_svg = _line_svg(rate_values, color="#16a34a")
    forage_latency_svg = _line_svg(latency_values, color="#f59e0b")
    forage_deaths_svg = _line_svg(deaths_values, color="#dc2626")

    evade_history = _load_simple_rate_history(evade_metrics_csv_path)
    evade_rate_svg = _line_svg([row.get("success_rate_total", 0.0) for row in evade_history], color="#ef4444")

    visible_text = html.escape(json.dumps(visible_cells, ensure_ascii=False))

    content = f"""<!doctype html>
<html lang=\"zh\">
<head>
  <meta charset=\"utf-8\" />
  <meta http-equiv=\"refresh\" content=\"3\" />
  <title>Superblock Master Dashboard</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #0f172a; }}
    .links a {{ margin-right: 14px; }}
    .section {{ margin-top: 20px; padding: 12px; border: 1px solid #cbd5e1; border-radius: 8px; background: #f8fafc; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; color: #334155; }}
  </style>
</head>
<body>
  <h1>Superblock Master Dashboard</h1>
  <div class=\"links\">
    <a href=\"{motion_dashboard_path}\">Motion Dashboard</a>
    <a href=\"{forage_dashboard_path}\">Forage Dashboard</a>
  </div>

  <div class=\"section\">
    <h2>Motion</h2>
    <div class=\"mono\">visible_cells={visible_text}</div>
    <div class=\"grid\">
      <div>
        <h4>score 曲线</h4>
        {motion_score_svg}
      </div>
      <div>
        <h4>pos_mse 曲线</h4>
        {motion_mse_svg}
      </div>
    </div>
  </div>

  <div class=\"section\">
    <h2>Curiosity</h2>
    <div class=\"mono\">{html.escape(curiosity_block)}</div>
  </div>

  <div class=\"section\">
    <h2>Forage</h2>
    <div class=\"grid\">
      <div>
        <h4>hungry_success_rate 曲线</h4>
        {forage_rate_svg}
      </div>
      <div>
        <h4>hungry_latency_mean 曲线</h4>
        {forage_latency_svg}
      </div>
    </div>
    <div>
      <h4>deaths 曲线</h4>
      {forage_deaths_svg}
    </div>
  </div>

  <div class="section">
    <h2>Evade</h2>
    <div>
      <h4>survival_success_rate 曲线</h4>
      {evade_rate_svg}
    </div>
  </div>
</body>
</html>
"""
    Path(out_path).write_text(content, encoding="utf-8")
