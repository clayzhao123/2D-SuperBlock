from __future__ import annotations

import argparse
import random

from .buffer import ReplayBuffer, Transition
from .env import Action, SuperblockEnv
from .models import ForwardModel
from .monitor import checkpoint_exists, load_checkpoint, save_checkpoint, write_dashboard, write_history_csv
from .utils import GRID_MAX, exp, mse, set_seed


def action_to_onehot(action: Action) -> list[float]:
    vec = [0.0] * 10
    vec[action.leg_id] = 1.0
    move_idx = {"+x": 0, "-x": 1, "+y": 2, "-y": 3}[action.move_dir]
    vec[4 + move_idx] = 1.0
    turn_idx = {"CW": 0, "CCW": 1}[action.turn_dir]
    vec[8 + turn_idx] = 1.0
    return vec


def _split_indices(n: int, seed: int) -> tuple[list[int], list[int]]:
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    split = max(1, int(n * 0.8))
    train_idx = idx[:split]
    val_idx = idx[split:] or idx[:1]
    return train_idx, val_idx


def parse_visible_cells(spec: str) -> list[tuple[int, int]]:
    spec = spec.strip()
    if not spec:
        return [(19, 19)]
    cells = []
    for token in spec.split(","):
        x_str, y_str = token.split(":")
        x = int(x_str)
        y = int(y_str)
        if x < 0 or y < 0 or x > GRID_MAX or y > GRID_MAX:
            raise ValueError(f"Cell out of range: ({x},{y})")
        cells.append((x, y))
    if not cells:
        raise ValueError("visible_cells cannot be empty")
    return cells


def train_night(
    model: ForwardModel,
    buffer: ReplayBuffer,
    *,
    batch_size: int,
    lr: float,
    epochs: int,
    seed: int,
) -> tuple[float, float, float]:
    x, y = buffer.as_arrays()
    n = len(x)
    if n < 5:
        return float("inf"), 0.0, 0.0

    train_idx, val_idx = _split_indices(n, seed)
    x_train = [x[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    x_val = [x[i] for i in val_idx]
    y_val = [y[i] for i in val_idx]

    for epoch in range(epochs):
        order = list(range(len(x_train)))
        random.Random(seed + epoch).shuffle(order)
        for s in range(0, len(order), batch_size):
            chunk = order[s : s + batch_size]
            xb = [x_train[i] for i in chunk]
            yb = [y_train[i] for i in chunk]
            model.train_batch(xb, yb, lr)

    pred_val = model.predict_batch(x_val)
    pos_mse = mse(pred_val, y_val)

    exact_hits = 0
    near_hits = 0
    for pred, true in zip(pred_val, y_val):
        p_int = [round(v * 40.0) for v in pred]
        t_int = [round(v * 40.0) for v in true]
        if p_int == t_int:
            exact_hits += 1
        if all(abs(p - t) <= 1 for p, t in zip(p_int, t_int)):
            near_hits += 1
    exact_acc = exact_hits / len(y_val)
    near_acc = near_hits / len(y_val)

    return pos_mse, exact_acc, near_acc


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    rng = random.Random(args.seed)
    visible_cells = parse_visible_cells(args.visible_cells)

    env = SuperblockEnv(visible_cells=visible_cells)
    model = ForwardModel(seed=args.seed)
    buffer = ReplayBuffer()
    history: list[dict[str, float]] = []
    start_day = 0

    if args.resume and checkpoint_exists(args.checkpoint_path):
        payload = load_checkpoint(args.checkpoint_path)
        start_day = int(payload["day_idx"])
        model.w1 = payload["model"]["w1"]
        model.b1 = payload["model"]["b1"]
        model.w2 = payload["model"]["w2"]
        model.b2 = payload["model"]["b2"]
        buffer.items = payload["buffer"]
        history = payload.get("history", [])
        if not args.override_visible_cells:
            visible_cells = [tuple(cell) for cell in payload.get("visible_cells", visible_cells)]
        env.set_visible_cells(visible_cells)
        print(f"Resumed from day {start_day} with {len(buffer)} transitions.")

    streak = 0
    if history:
        for row in reversed(history):
            if row["score"] >= args.stop_score:
                streak += 1
            else:
                break

    write_dashboard(args.dashboard_path, history, visible_cells)

    try:
        for day_idx in range(start_day + 1, args.max_days + 1):
            invalid_count = 0
            visible_hits = 0

            state_t, img_t = env.reset()
            for _ in range(args.steps_per_day):
                action = env.sample_action(rng)
                state_tp1, img_tp1, invalid = env.step(action)
                if invalid:
                    invalid_count += 1
                if any(px > 0.0 for row in img_tp1 for px in row):
                    visible_hits += 1
                buffer.add(
                    Transition(
                        state_t=state_t.copy(),
                        action_vec=action_to_onehot(action),
                        invalid_move=invalid,
                        img_t=[row[:] for row in img_t],
                        state_tp1=state_tp1.copy(),
                        img_tp1=[row[:] for row in img_tp1],
                    )
                )
                state_t, img_t = state_tp1, img_tp1

            night_epochs = 0 if args.daytime_only else args.epochs_per_night
            pos_mse, exact_acc, near_acc = train_night(
                model,
                buffer,
                batch_size=args.batch_size,
                lr=args.lr,
                epochs=night_epochs,
                seed=args.seed + day_idx,
            )
            score = (
                args.score_w_mse * exp(-args.score_k * pos_mse)
                + args.score_w_exact * exact_acc
                + args.score_w_near * near_acc
            )
            invalid_move_ratio = invalid_count / float(args.steps_per_day)

            row = {
                "day_idx": float(day_idx),
                "invalid_move_ratio": invalid_move_ratio,
                "pos_mse": pos_mse,
                "exact_acc": exact_acc,
                "near_acc": near_acc,
                "score": score,
                "visible_hits": float(visible_hits),
            }
            history.append(row)

            print(
                f"day_idx={day_idx} invalid_move_ratio={invalid_move_ratio:.3f} "
                f"pos_mse={pos_mse:.6f} exact_acc={exact_acc:.3f} "
                f"near_acc={near_acc:.3f} score={score:.3f}"
            )

            write_dashboard(args.dashboard_path, history, visible_cells)
            write_history_csv(args.metrics_csv_path, history)
            if day_idx % args.save_every_days == 0:
                save_checkpoint(
                    args.checkpoint_path,
                    day_idx=day_idx,
                    model=model,
                    buffer=buffer,
                    history=history,
                    visible_cells=visible_cells,
                )

            streak = streak + 1 if score >= args.stop_score else 0
            if streak >= args.stop_streak:
                print(f"Converged for {streak} consecutive nights. Stop at day {day_idx}.")
                break
    except KeyboardInterrupt:
        print("Interrupted by user, saving checkpoint...")
    finally:
        last_day = int(history[-1]["day_idx"]) if history else start_day
        save_checkpoint(
            args.checkpoint_path,
            day_idx=last_day,
            model=model,
            buffer=buffer,
            history=history,
            visible_cells=visible_cells,
        )
        write_dashboard(args.dashboard_path, history, visible_cells)
        write_history_csv(args.metrics_csv_path, history)
        print(f"Checkpoint saved to {args.checkpoint_path}")
        print(f"Dashboard updated at {args.dashboard_path}")
        print(f"Metrics CSV updated at {args.metrics_csv_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Superblock forward model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps-per-day", type=int, default=100)
    parser.add_argument("--max-days", type=int, default=200)
    parser.add_argument("--epochs-per-night", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--score-k", type=float, default=50.0)
    parser.add_argument("--score-w-mse", type=float, default=0.5)
    parser.add_argument("--score-w-exact", type=float, default=0.2)
    parser.add_argument("--score-w-near", type=float, default=0.3)
    parser.add_argument("--stop-score", type=float, default=0.95)
    parser.add_argument("--stop-streak", type=int, default=3)
    parser.add_argument("--checkpoint-path", type=str, default="artifacts/train.ckpt")
    parser.add_argument("--dashboard-path", type=str, default="artifacts/dashboard.html")
    parser.add_argument("--save-every-days", type=int, default=1)
    parser.add_argument("--metrics-csv-path", type=str, default="artifacts/metrics.csv")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--daytime-only", action="store_true")
    parser.add_argument(
        "--visible-cells",
        type=str,
        default="19:19",
        help="Comma-separated x:y list, e.g. 19:19,20:19",
    )
    parser.add_argument(
        "--override-visible-cells",
        action="store_true",
        help="When resuming, use --visible-cells instead of checkpoint value.",
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
