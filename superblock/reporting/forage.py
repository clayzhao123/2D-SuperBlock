from __future__ import annotations

import csv
from pathlib import Path

from .charts import _line_svg

def write_dashboard(path: str, history: list[dict[str, float]], *, show_motion: bool, policy_name: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    success_svg = _line_svg([row["hungry_success_rate_total"] for row in history], color="#16a34a")
    latency_svg = _line_svg([row["hungry_latency_mean_total"] for row in history if row["hungry_latency_mean_total"] >= 0], color="#f59e0b")
    deaths_svg = _line_svg([row["deaths_total"] for row in history], color="#dc2626")
    curiosity_svg = _line_svg([row["curiosity_score"] for row in history], color="#7c3aed")

    latest = history[-1] if history else None
    eval_line = "暂无评估"
    if latest is not None:
        eval_line = (
            f"success_rate_total={latest.get('hungry_success_rate_total', 0.0):.3f}, "
            f"latency_mean_total={latest.get('hungry_latency_mean_total', -1.0):.2f}, "
            f"deaths_total={int(latest.get('deaths_total', 0.0))}"
        )

    motion_block = ""
    if show_motion:
        motion_svg = _line_svg([row.get("motion_score", 0.0) for row in history], color="#2563eb")
        motion_block = f"""
        <h3>运动模型分数（仅在显式启用 motion 训练时显示）</h3>
        {motion_svg}
        """

    rows = "\n".join(
        "<tr>"
        f"<td>{int(row['day_idx'])}</td>"
        f"<td>{row['hungry_success_rate_today']:.3f}</td>"
        f"<td>{row['hungry_latency_mean_today']:.2f}</td>"
        f"<td>{int(row['hungry_attempts_today'])}</td>"
        f"<td>{int(row['hungry_success_today'])}</td>"
        f"<td>{int(row['food_seen_flag_today'])}</td>"
        f"<td>{int(row['steps_to_first_food_seen_today'])}</td>"
        f"<td>{int(row['deaths_total'])}</td>"
        f"<td>{row['curiosity_score']:.3f}</td>"
        "</tr>"
        for row in history[-30:]
    )

    content = f"""<!doctype html>
<html lang=\"zh\"><head><meta charset=\"utf-8\"><meta http-equiv=\"refresh\" content=\"3\">
<title>Forage Dashboard</title></head><body>
<h1>Forage 训练面板（觅食指标优先）</h1>
<div><b>policy:</b> {policy_name}</div>
<div><b>eval:</b> {eval_line}</div>
<h3>累计觅食成功率（越高越好）</h3>
{success_svg}
<h3>累计平均觅食耗时 hungry→eat（越低越好）</h3>
{latency_svg}
<h3>累计死亡次数</h3>
{deaths_svg}
<h3>好奇心分数（辅助指标）</h3>
{curiosity_svg}
{motion_block}
<table border=\"1\" cellspacing=\"0\" cellpadding=\"4\">
<tr><th>day</th><th>rate_today</th><th>lat_mean_today</th><th>attempts_today</th><th>success_today</th><th>food_seen</th><th>first_seen_step</th><th>deaths_total</th><th>curiosity</th></tr>
{rows}
</table></body></html>"""
    Path(path).write_text(content, encoding="utf-8")

def write_csv(path: str, history: list[dict[str, float]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not history:
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n")
        return

    fieldnames = list(history[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

def write_attempts_csv(path: str, attempts: list[dict[str, str]]) -> None:
    if not path:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["day_idx", "attempt_idx", "start_center", "target_food", "success", "latency_steps"],
        )
        writer.writeheader()
        writer.writerows(attempts)
