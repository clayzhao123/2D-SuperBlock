from __future__ import annotations

import argparse
import random

from .buffer import ReplayBuffer, Transition
from .env import Action, SuperblockEnv
from .models import ForwardModel
from .utils import exp, mse, set_seed


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


def train_night(
    model: ForwardModel,
    buffer: ReplayBuffer,
    *,
    batch_size: int,
    lr: float,
    epochs: int,
    seed: int,
) -> tuple[float, float]:
    x, y = buffer.as_arrays()
    n = len(x)
    if n < 5:
        return float("inf"), 0.0

    train_idx, val_idx = _split_indices(n, seed)
    x_train = [x[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    x_val = [x[i] for i in val_idx]
    y_val = [y[i] for i in val_idx]

    for _ in range(epochs):
        order = list(range(len(x_train)))
        random.Random(seed + _).shuffle(order)
        for s in range(0, len(order), batch_size):
            chunk = order[s : s + batch_size]
            xb = [x_train[i] for i in chunk]
            yb = [y_train[i] for i in chunk]
            model.train_batch(xb, yb, lr)

    pred_val = model.predict_batch(x_val)
    pos_mse = mse(pred_val, y_val)

    exact_hits = 0
    for pred, true in zip(pred_val, y_val):
        p_int = [round(v * 40.0) for v in pred]
        t_int = [round(v * 40.0) for v in true]
        if p_int == t_int:
            exact_hits += 1
    exact_acc = exact_hits / len(y_val)

    return pos_mse, exact_acc


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    rng = random.Random(args.seed)

    env = SuperblockEnv()
    model = ForwardModel(seed=args.seed)
    buffer = ReplayBuffer()

    streak = 0
    for day_idx in range(1, args.max_days + 1):
        invalid_count = 0

        state_t, img_t = env.reset()
        for _ in range(args.steps_per_day):
            action = env.sample_action(rng)
            state_tp1, img_tp1, invalid = env.step(action)
            if invalid:
                invalid_count += 1
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

        pos_mse, exact_acc = train_night(
            model,
            buffer,
            batch_size=args.batch_size,
            lr=args.lr,
            epochs=args.epochs_per_night,
            seed=args.seed + day_idx,
        )
        score = 0.5 * exp(-args.score_k * pos_mse) + 0.5 * exact_acc
        invalid_move_ratio = invalid_count / float(args.steps_per_day)

        print(
            f"day_idx={day_idx} invalid_move_ratio={invalid_move_ratio:.3f} "
            f"pos_mse={pos_mse:.6f} exact_acc={exact_acc:.3f} score={score:.3f}"
        )

        streak = streak + 1 if score >= args.stop_score else 0
        if streak >= args.stop_streak:
            print(f"Converged for {streak} consecutive nights. Stop at day {day_idx}.")
            break


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Superblock forward model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps-per-day", type=int, default=100)
    parser.add_argument("--max-days", type=int, default=200)
    parser.add_argument("--epochs-per-night", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--score-k", type=float, default=50.0)
    parser.add_argument("--stop-score", type=float, default=0.95)
    parser.add_argument("--stop-streak", type=int, default=3)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
