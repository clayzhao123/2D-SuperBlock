from __future__ import annotations

import argparse
import csv
import pickle
import random
from pathlib import Path

from .forage_agent import FoodMemory, ForagePolicy
from .forage_env import ForageEnv
from .policy_curiosity import CuriosityMemory, CuriosityPolicy, load_forward_model_from_ckpt


def write_dashboard(path: str, history: list[dict[str, float]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    rates = [row["success_rate"] for row in history]
    points = []
    width, height = 560, 180
    if rates:
        lo, hi = min(rates), max(rates)
        span = hi - lo if hi != lo else 1.0
        for i, value in enumerate(rates):
            x = (i / max(1, len(rates) - 1)) * (width - 10) + 5
            y = height - (((value - lo) / span) * (height - 20) + 10)
            points.append(f"{x:.2f},{y:.2f}")
    rows = "\n".join(
        f"<tr><td>{int(row['day_idx'])}</td><td>{int(row['forage_attempts_total'])}</td>"
        f"<td>{int(row['forage_success_total'])}</td><td>{row['success_rate']:.3f}</td>"
        f"<td>{int(row['deaths_total'])}</td><td>{int(row['food_found_count'])}</td></tr>"
        for row in history[-30:]
    )
    content = f"""<!doctype html>
<html lang=\"zh\"><head><meta charset=\"utf-8\"><meta http-equiv=\"refresh\" content=\"3\">
<title>Forage Dashboard</title></head><body>
<h1>Forage 训练面板</h1>
<svg width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">
<rect x=\"0\" y=\"0\" width=\"{width}\" height=\"{height}\" fill=\"#f8fafc\" stroke=\"#cbd5e1\"/>
<polyline fill=\"none\" stroke=\"#16a34a\" stroke-width=\"2\" points=\"{' '.join(points)}\"/>
</svg>
<table border=\"1\" cellspacing=\"0\" cellpadding=\"4\">
<tr><th>day</th><th>attempts</th><th>success</th><th>rate</th><th>deaths</th><th>food_found</th></tr>
{rows}
</table></body></html>"""
    Path(path).write_text(content, encoding="utf-8")


def write_csv(path: str, history: list[dict[str, float]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "day_idx",
                "forage_attempts_total",
                "forage_success_total",
                "success_rate",
                "deaths_total",
                "food_found_count",
            ],
        )
        writer.writeheader()
        writer.writerows(history)


def save_ckpt(path: str, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def run(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    model = load_forward_model_from_ckpt(args.motion_checkpoint_path)
    env = ForageEnv(
        seed=args.seed,
        food_count=args.food_count,
        hunger_interval=args.hunger_interval,
        hunger_death_steps=args.hunger_death_steps,
    )
    curiosity_memory = CuriosityMemory()
    curiosity_policy = CuriosityPolicy(forward_model=model, memory=curiosity_memory)
    food_memory = FoodMemory()
    forage_policy = ForagePolicy(curiosity_policy, food_memory)

    history: list[dict[str, float]] = []
    forage_attempts_total = 0
    forage_success_total = 0
    deaths_total = 0

    for day_idx in range(1, args.days + 1):
        state, _ = env.reset_day(day_idx)
        food_memory.reset_day()
        curiosity_memory.update(state)

        day_seen_food: set[tuple[int, int]] = set()
        for _ in range(args.steps_per_day):
            visible = env.observe_visible_food(args.vision_radius)
            food_memory.update(visible)
            day_seen_food.update(visible)

            action = forage_policy.select_action(state, env, rng)
            state, _, _, done, _ = env.step(action)
            curiosity_memory.update(state)
            if done:
                deaths_total += 1
                break

        forage_attempts_total = env.forage_attempts
        forage_success_total = env.forage_success
        success_rate = forage_success_total / max(1, forage_attempts_total)
        row = {
            "day_idx": float(day_idx),
            "forage_attempts_total": float(forage_attempts_total),
            "forage_success_total": float(forage_success_total),
            "success_rate": success_rate,
            "deaths_total": float(deaths_total),
            "food_found_count": float(len(day_seen_food)),
        }
        history.append(row)

        write_csv(args.out_metrics_csv, history)
        write_dashboard(args.out_dashboard, history)
        save_ckpt(
            args.out_checkpoint,
            {
                "day_idx": day_idx,
                "history": history,
                "forage_attempts_total": forage_attempts_total,
                "forage_success_total": forage_success_total,
                "deaths_total": deaths_total,
                "motion_checkpoint_path": args.motion_checkpoint_path,
                "params": {
                    "food_count": args.food_count,
                    "hunger_interval": args.hunger_interval,
                    "hunger_death_steps": args.hunger_death_steps,
                    "vision_radius": args.vision_radius,
                    "seed": args.seed,
                },
            },
        )
        print(
            f"day={day_idx} attempts={forage_attempts_total} successes={forage_success_total} "
            f"success_rate={success_rate:.3f} deaths={deaths_total}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Foraging training loop")
    parser.add_argument("--days", type=int, default=100)
    parser.add_argument("--steps-per-day", type=int, default=100)
    parser.add_argument("--food-count", type=int, default=3)
    parser.add_argument("--hunger-interval", type=int, default=25)
    parser.add_argument("--hunger-death-steps", type=int, default=75)
    parser.add_argument("--vision-radius", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--motion-checkpoint-path", type=str, default="artifacts/train.ckpt")
    parser.add_argument("--out-checkpoint", type=str, default="artifacts/forage.ckpt")
    parser.add_argument("--out-metrics-csv", type=str, default="artifacts/forage_metrics.csv")
    parser.add_argument("--out-dashboard", type=str, default="artifacts/forage_dashboard.html")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
