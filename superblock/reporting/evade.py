from __future__ import annotations

import csv
from pathlib import Path

from .charts import _line_svg

def write_evade_dashboard(path: str, history: list[dict[str, float]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    succ = _line_svg([row.get("success_rate_total", 0.0) for row in history], color="#16a34a")
    hungry_succ = _line_svg([row.get("hungry_success_rate_total", 0.0) for row in history], color="#22c55e")
    hacker_death = _line_svg([row.get("death_by_hacker_total", 0.0) for row in history], color="#ef4444")
    hunger_death = _line_svg([row.get("death_by_hunger_total", 0.0) for row in history], color="#f97316")
    rows = "\n".join(
        "<tr>"
        f"<td>{int(r['day_idx'])}</td>"
        f"<td>{int(r.get('success_today', 0.0))}</td>"
        f"<td>{r.get('success_rate_total', 0.0):.3f}</td>"
        f"<td>{int(r.get('hungry_attempts_today', 0.0))}</td>"
        f"<td>{int(r.get('hungry_success_today', 0.0))}</td>"
        f"<td>{r.get('hungry_success_rate_today', 0.0):.3f}</td>"
        f"<td>{int(r.get('death_by_hacker_today', 0.0))}</td>"
        f"<td>{int(r.get('death_by_hunger_today', 0.0))}</td>"
        "</tr>"
        for r in history[-30:]
    )
    Path(path).write_text(
        f"""<!doctype html><html lang='zh'><head><meta charset='utf-8'><meta http-equiv='refresh' content='3'><title>Evade Dashboard</title></head>
<body><h1>Evade + Forage Dashboard</h1>
<h3>survival_success_rate_total</h3>{succ}
<h3>hungry_success_rate_total</h3>{hungry_succ}
<h3>death_by_hacker_total</h3>{hacker_death}
<h3>death_by_hunger_total</h3>{hunger_death}
<table border='1'><tr><th>day</th><th>survive_today</th><th>survive_rate_total</th><th>hungry_attempts_today</th><th>hungry_success_today</th><th>hungry_success_rate_today</th><th>death_hacker_today</th><th>death_hunger_today</th></tr>{rows}</table></body></html>""",
        encoding="utf-8",
    )

def write_evade_csv(path: str, history: list[dict[str, float]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not history:
        Path(path).write_text("\n", encoding="utf-8")
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
