from __future__ import annotations

import argparse
import csv
import random
from collections.abc import Callable
from pathlib import Path

from .env import Action
from .evade_env import EvadeEnv
from .master_dashboard import write_master_dashboard
from .monitor import _line_svg, load_checkpoint
from .policy_curiosity import CuriosityPolicy, CuriosityMemory, load_forward_model_from_ckpt
from .utils import position_key


def write_evade_dashboard(path: str, history: list[dict[str, float]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    succ = _line_svg([row["success_rate_total"] for row in history], color="#16a34a")
    rows = "\n".join(
        f"<tr><td>{int(r['day_idx'])}</td><td>{int(r['success_today'])}</td><td>{r['success_rate_total']:.3f}</td></tr>"
        for r in history[-30:]
    )
    Path(path).write_text(
        f"""<!doctype html><html lang='zh'><head><meta charset='utf-8'><meta http-equiv='refresh' content='3'><title>Evade Dashboard</title></head>
<body><h1>躲避训练面板</h1><h3>累计成功率曲线</h3>{succ}<table border='1'><tr><th>day</th><th>success_today</th><th>success_rate_total</th></tr>{rows}</table></body></html>""",
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


def _is_loop(path: list[tuple[int, int]], current: tuple[int, int]) -> bool:
    return len(path) >= 6 and current in path[-6:]


def _resolve_motion_checkpoint_path(path: str) -> str:
    payload = load_checkpoint(path)
    if "model" in payload:
        return path

    motion_path = payload.get("motion_checkpoint_path")
    if not isinstance(motion_path, str) or not motion_path:
        raise ValueError(f"checkpoint {path} 不包含 model，且没有 motion_checkpoint_path")

    candidate = Path(path).parent / motion_path
    if candidate.exists():
        return str(candidate)
    return motion_path


def _can_see_superhacker(prey: tuple[int, int], hacker: tuple[int, int], vision_length: int) -> bool:
    if prey[0] == hacker[0]:
        return abs(prey[1] - hacker[1]) <= vision_length
    if prey[1] == hacker[1]:
        return abs(prey[0] - hacker[0]) <= vision_length
    return False


def _observe_grass(
    anchor: tuple[int, int],
    grass_cells: set[tuple[int, int]],
    vision_length: int,
    memory: set[tuple[int, int]],
) -> None:
    ax, ay = anchor
    for gx, gy in grass_cells:
        if abs(gx - ax) + abs(gy - ay) <= vision_length:
            memory.add((gx, gy))


def _choose_grass_escape_action(
    *,
    state: list[int],
    env: EvadeEnv,
    policy: CuriosityPolicy,
    memory_grass: set[tuple[int, int]],
    rng: random.Random,
) -> Action | None:
    anchor = position_key(state)
    _observe_grass(anchor, env.grass_cells, env.vision_length, memory_grass)

    visible_grass = [
        cell for cell in memory_grass if abs(cell[0] - anchor[0]) + abs(cell[1] - anchor[1]) <= env.vision_length
    ]
    target = min(visible_grass or list(memory_grass), key=lambda c: abs(c[0] - anchor[0]) + abs(c[1] - anchor[1]), default=None)
    if target is None:
        return None

    candidates = []
    for action in policy.candidate_actions():
        next_points, invalid = env.base_env.peek_step(env.base_env.points, action)
        if invalid:
            continue
        pred_anchor = position_key(next_points)
        dist = abs(pred_anchor[0] - target[0]) + abs(pred_anchor[1] - target[1])
        candidates.append((dist, action))

    if not candidates:
        return None

    best_dist = min(dist for dist, _ in candidates)
    best_actions = [action for dist, action in candidates if dist == best_dist]
    return rng.choice(best_actions)


def run_with_callbacks(
    args: argparse.Namespace,
    *,
    on_step: Callable[[int, int, EvadeEnv, list[tuple[int, int]]], None] | None = None,
    on_day_end: Callable[[dict[str, float]], None] | None = None,
) -> None:
    rng = random.Random(args.seed)
    env = EvadeEnv(seed=args.seed, grass_area=args.grass_area, grass_count=args.grass_count, vision_length=3)
    motion_ckpt_path = _resolve_motion_checkpoint_path(args.motion_checkpoint_path)
    model = load_forward_model_from_ckpt(motion_ckpt_path)
    policy = CuriosityPolicy(forward_model=model, memory=CuriosityMemory(), epsilon=args.epsilon)
    history: list[dict[str, float]] = []
    success_total = 0

    for day_idx in range(1, args.days + 1):
        state, _ = env.reset_day(day_idx)
        policy.memory = CuriosityMemory()
        policy.recent_path.clear()
        policy.memory.update(state)
        survived = 1
        grass_memory: set[tuple[int, int]] = set()
        walk: list[tuple[int, int]] = [position_key(state)]
        for step_idx in range(args.steps_per_day):
            anchor = position_key(state)
            hacker_seen = _can_see_superhacker(anchor, env.superhacker_pos, env.vision_length)
            action = None
            if hacker_seen:
                action = _choose_grass_escape_action(state=state, env=env, policy=policy, memory_grass=grass_memory, rng=rng)
            if action is None:
                action = policy.select_action(state, env.base_env, rng)

            next_points, will_invalid = env.base_env.peek_step(env.base_env.points, action)
            if _is_loop(walk, position_key(next_points)) or will_invalid:
                action = env.base_env.sample_action(rng)
            state, _img, _invalid, done, _caught = env.step(action)
            pos = position_key(state)
            walk.append(pos)
            policy.memory.update(state)
            _observe_grass(pos, env.grass_cells, env.vision_length, grass_memory)
            if on_step is not None:
                on_step(day_idx, step_idx, env, walk)
            if done:
                survived = 0
                break
        success_total += survived
        history.append(
            {
                "day_idx": float(day_idx),
                "success_today": float(survived),
                "success_rate_total": success_total / day_idx,
            }
        )

        write_evade_dashboard(args.dashboard_path, history)
        write_evade_csv(args.metrics_csv_path, history)
        write_master_dashboard(evade_metrics_csv_path=args.metrics_csv_path, evade_dashboard_path=args.dashboard_path)
        if on_day_end is not None:
            on_day_end(history[-1])


def run(args: argparse.Namespace) -> None:
    run_with_callbacks(args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evade training")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--days", type=int, default=100)
    p.add_argument("--steps-per-day", type=int, default=100)
    p.add_argument("--grass-area", type=int, default=10)
    p.add_argument("--grass-count", type=int, default=3)
    p.add_argument("--epsilon", type=float, default=0.08)
    p.add_argument("--motion-checkpoint-path", type=str, default="artifacts/train.ckpt")
    p.add_argument("--dashboard-path", type=str, default="artifacts/evade_dashboard.html")
    p.add_argument("--metrics-csv-path", type=str, default="artifacts/evade_metrics.csv")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
