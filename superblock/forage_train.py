from __future__ import annotations

import argparse
import csv
import pickle
import random
from pathlib import Path

from .buffer import ReplayBuffer, Transition
from .forage_agent import FoodMemory, ForagePolicy
from .forage_env import ForageEnv
from .monitor import load_checkpoint, save_checkpoint
from .policy_curiosity import CuriosityMemory, CuriosityPolicy, load_forward_model_from_ckpt
from .train import action_to_onehot, train_night


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
        f"<td>{int(row['deaths_total'])}</td><td>{int(row['food_found_count'])}</td>"
        f"<td>{row['curiosity_score']:.3f}</td><td>{row['motion_score']:.3f}</td></tr>"
        for row in history[-30:]
    )
    curiosity_svg = _line_svg([row["curiosity_score"] for row in history], color="#f59e0b")
    motion_svg = _line_svg([row["motion_score"] for row in history], color="#2563eb")
    content = f"""<!doctype html>
<html lang=\"zh\"><head><meta charset=\"utf-8\"><meta http-equiv=\"refresh\" content=\"3\">
<title>Forage Dashboard</title></head><body>
<h1>Forage 训练面板</h1>
<h3>运动训练分数（来自觅食过程中的持续校准）</h3>
{motion_svg}
<h3>感知力/好奇心曲线（横轴天数，纵轴当日平均新颖度）</h3>
{curiosity_svg}
<h3>觅食成功率曲线</h3>
<svg width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">
<rect x=\"0\" y=\"0\" width=\"{width}\" height=\"{height}\" fill=\"#f8fafc\" stroke=\"#cbd5e1\"/>
<polyline fill=\"none\" stroke=\"#16a34a\" stroke-width=\"2\" points=\"{' '.join(points)}\"/>
</svg>
<table border=\"1\" cellspacing=\"0\" cellpadding=\"4\">
<tr><th>day</th><th>attempts</th><th>success</th><th>rate</th><th>deaths</th><th>food_found</th><th>curiosity</th><th>motion_score</th></tr>
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
                "curiosity_score",
                "motion_pos_mse",
                "motion_exact_acc",
                "motion_near_acc",
                "motion_score",
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
    motion_payload = load_checkpoint(args.motion_checkpoint_path)
    model = load_forward_model_from_ckpt(args.motion_checkpoint_path)
    motion_history = motion_payload.get("history", [])
    motion_visible_cells = [tuple(cell) for cell in motion_payload.get("visible_cells", [(19, 19)])]
    motion_buffer = ReplayBuffer()
    for item in motion_payload.get("buffer", []):
        motion_buffer.add(item)

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
        curiosity_sum = 0.0
        for _ in range(args.steps_per_day):
            visible = env.observe_visible_food(args.vision_radius)
            food_memory.update(visible)
            day_seen_food.update(visible)

            prev_state = state.copy()
            prev_img = env.base_env.render()
            action = forage_policy.select_action(state, env, rng)
            state, _, _, done, _ = env.step(action)
            cell = (sum(state[::2]) / 4.0, sum(state[1::2]) / 4.0)
            center = (round(cell[0]), round(cell[1]))
            curiosity_sum += 1.0 / (1.0 + curiosity_memory.count(center))
            curiosity_memory.update(state)

            motion_buffer.add(
                Transition(
                    state_t=prev_state,
                    action_vec=action_to_onehot(action),
                    invalid_move=False,
                    img_t=[row[:] for row in prev_img],
                    state_tp1=state.copy(),
                    img_tp1=[row[:] for row in env.base_env.render()],
                )
            )
            if done:
                deaths_total += 1
                break

        motion_pos_mse = 0.0
        motion_exact_acc = 0.0
        motion_near_acc = 0.0
        if args.motion_epochs_per_day > 0:
            motion_pos_mse, motion_exact_acc, motion_near_acc = train_night(
                model,
                motion_buffer,
                batch_size=args.motion_batch_size,
                lr=args.motion_lr,
                epochs=args.motion_epochs_per_day,
                seed=args.seed + day_idx,
            )
            motion_score = (
                args.motion_score_w_mse * (2.718281828 ** (-args.motion_score_k * motion_pos_mse))
                + args.motion_score_w_exact * motion_exact_acc
                + args.motion_score_w_near * motion_near_acc
            )
        else:
            motion_score = motion_history[-1]["score"] if motion_history else 0.0

        forage_attempts_total = env.forage_attempts
        forage_success_total = env.forage_success
        success_rate = forage_success_total / max(1, forage_attempts_total)
        curiosity_score = curiosity_sum / max(1, args.steps_per_day)
        row = {
            "day_idx": float(day_idx),
            "forage_attempts_total": float(forage_attempts_total),
            "forage_success_total": float(forage_success_total),
            "success_rate": success_rate,
            "deaths_total": float(deaths_total),
            "food_found_count": float(len(day_seen_food)),
            "curiosity_score": curiosity_score,
            "motion_pos_mse": motion_pos_mse,
            "motion_exact_acc": motion_exact_acc,
            "motion_near_acc": motion_near_acc,
            "motion_score": motion_score,
        }
        history.append(row)

        motion_history.append(
            {
                "day_idx": float(len(motion_history) + 1),
                "invalid_move_ratio": 0.0,
                "pos_mse": motion_pos_mse,
                "exact_acc": motion_exact_acc,
                "near_acc": motion_near_acc,
                "score": motion_score,
            }
        )

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
        save_checkpoint(
            args.motion_output_checkpoint,
            day_idx=len(motion_history),
            model=model,
            buffer=motion_buffer,
            history=motion_history,
            visible_cells=motion_visible_cells,
        )
        print(
            f"day={day_idx} attempts={forage_attempts_total} successes={forage_success_total} "
            f"success_rate={success_rate:.3f} curiosity={curiosity_score:.3f} "
            f"motion_score={motion_score:.3f} deaths={deaths_total}"
        )


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
    parser.add_argument("--motion-output-checkpoint", type=str, default="artifacts/train_forage_tuned.ckpt")
    parser.add_argument("--motion-epochs-per-day", type=int, default=10)
    parser.add_argument("--motion-batch-size", type=int, default=64)
    parser.add_argument("--motion-lr", type=float, default=1e-2)
    parser.add_argument("--motion-score-k", type=float, default=50.0)
    parser.add_argument("--motion-score-w-mse", type=float, default=0.5)
    parser.add_argument("--motion-score-w-exact", type=float, default=0.2)
    parser.add_argument("--motion-score-w-near", type=float, default=0.3)
    parser.add_argument("--out-checkpoint", type=str, default="artifacts/forage.ckpt")
    parser.add_argument("--out-metrics-csv", type=str, default="artifacts/forage_metrics.csv")
    parser.add_argument("--out-dashboard", type=str, default="artifacts/forage_dashboard.html")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
