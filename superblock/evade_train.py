from __future__ import annotations

import argparse
import random
from collections.abc import Callable
from pathlib import Path

from .env import Action
from .evade_env import EvadeEnv
from .forage_agent import FoodMemory, ForagePolicy
from .reporting.summary import write_training_summary
from .reporting.evade import write_evade_dashboard, write_evade_csv
from .monitor import load_checkpoint
from .policy_curiosity import CuriosityMemory, CuriosityPolicy, load_forward_model_from_ckpt
from .utils import position_key


def _is_loop(path: list[tuple[int, int]], current: tuple[int, int]) -> bool:
    return len(path) >= 6 and current in path[-6:]


def _resolve_motion_checkpoint_path(path: str) -> str:
    payload = load_checkpoint(path)
    if "model" in payload:
        return path

    motion_path = payload.get("motion_checkpoint_path")
    if not isinstance(motion_path, str) or not motion_path:
        raise ValueError(f"checkpoint {path} does not contain model and has no motion_checkpoint_path")

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
    target = min(
        visible_grass or list(memory_grass),
        key=lambda c: abs(c[0] - anchor[0]) + abs(c[1] - anchor[1]),
        default=None,
    )
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
    food_count = getattr(args, "food_count", 3)
    hunger_interval = getattr(args, "hunger_interval", 25)
    hunger_death_steps = getattr(args, "hunger_death_steps", 75)
    vision_radius = getattr(args, "vision_radius", 3)
    food_spawn_mode = getattr(args, "food_spawn_mode", "static")
    food_spawn_radius = getattr(args, "food_spawn_radius", 8)
    max_food_on_map = getattr(args, "max_food_on_map", 6)

    rng = random.Random(args.seed)
    env = EvadeEnv(
        seed=args.seed,
        grass_area=args.grass_area,
        grass_count=args.grass_count,
        vision_length=3,
        food_count=food_count,
        hunger_interval=hunger_interval,
        hunger_death_steps=hunger_death_steps,
        food_spawn_mode=food_spawn_mode,
        food_spawn_radius=food_spawn_radius,
        max_food_on_map=max_food_on_map,
    )
    motion_ckpt_path = _resolve_motion_checkpoint_path(args.motion_checkpoint_path)
    model = load_forward_model_from_ckpt(motion_ckpt_path)
    policy = CuriosityPolicy(forward_model=model, memory=CuriosityMemory(), epsilon=args.epsilon)
    food_memory = FoodMemory()
    forage_policy = ForagePolicy(policy, food_memory)

    history: list[dict[str, float]] = []
    success_total = 0
    death_by_hacker_total = 0
    death_by_hunger_total = 0

    for day_idx in range(1, args.days + 1):
        state, _ = env.reset_day(day_idx)
        policy.memory = CuriosityMemory()
        policy.recent_path.clear()
        policy.memory.update(state)
        food_memory.reset_day()

        survived = 1
        death_by_hacker_today = 0
        death_by_hunger_today = 0
        first_food_seen_step = -1
        hungry_attempts_before_day = env.forage_attempts
        hungry_success_before_day = env.forage_success

        grass_memory: set[tuple[int, int]] = set()
        walk: list[tuple[int, int]] = [position_key(state)]

        for step_idx in range(args.steps_per_day):
            visible_food = env.observe_visible_food(vision_radius)
            food_memory.retain(set(env.food_cells))
            food_memory.update(visible_food)
            if visible_food and first_food_seen_step < 0:
                first_food_seen_step = step_idx + 1

            anchor = position_key(state)
            hacker_seen = _can_see_superhacker(anchor, env.superhacker_pos, env.vision_length)
            action = None
            if hacker_seen:
                action = _choose_grass_escape_action(
                    state=state,
                    env=env,
                    policy=policy,
                    memory_grass=grass_memory,
                    rng=rng,
                )
            if action is None:
                action = forage_policy.select_action(state, env, rng)

            next_points, will_invalid = env.base_env.peek_step(env.base_env.points, action)
            if _is_loop(walk, position_key(next_points)) or will_invalid:
                action = env.base_env.sample_action(rng)

            state, _img, _invalid, done, _caught = env.step(action)
            pos = position_key(state)
            walk.append(pos)
            policy.memory.update(state)
            _observe_grass(pos, env.grass_cells, env.vision_length, grass_memory)
            food_memory.retain(set(env.food_cells))
            if on_step is not None:
                on_step(day_idx, step_idx, env, walk)
            if done:
                survived = 0
                if env.last_caught:
                    death_by_hacker_today = 1
                elif env.last_hunger_death:
                    death_by_hunger_today = 1
                break

        success_total += survived
        death_by_hacker_total += death_by_hacker_today
        death_by_hunger_total += death_by_hunger_today

        hungry_attempts_today = env.forage_attempts - hungry_attempts_before_day
        hungry_success_today = env.forage_success - hungry_success_before_day
        hungry_success_rate_today = hungry_success_today / max(1, hungry_attempts_today)
        hungry_success_rate_total = env.forage_success / max(1, env.forage_attempts)

        history.append(
            {
                "day_idx": float(day_idx),
                "success_today": float(survived),
                "success_rate_total": success_total / day_idx,
                "death_by_hacker_today": float(death_by_hacker_today),
                "death_by_hunger_today": float(death_by_hunger_today),
                "death_by_hacker_total": float(death_by_hacker_total),
                "death_by_hunger_total": float(death_by_hunger_total),
                "hungry_attempts_today": float(hungry_attempts_today),
                "hungry_success_today": float(hungry_success_today),
                "hungry_success_rate_today": float(hungry_success_rate_today),
                "hungry_attempts_total": float(env.forage_attempts),
                "hungry_success_total": float(env.forage_success),
                "hungry_success_rate_total": float(hungry_success_rate_total),
                "steps_to_first_food_seen_today": float(first_food_seen_step),
                "food_seen_flag_today": float(1 if first_food_seen_step > 0 else 0),
                "food_count_on_map_today": float(len(env.food_cells)),
            }
        )

        write_evade_dashboard(args.dashboard_path, history)
        write_evade_csv(args.metrics_csv_path, history)
        write_training_summary(args, "evade")
        if on_day_end is not None:
            on_day_end(history[-1])


def run(args: argparse.Namespace) -> None:
    run_with_callbacks(args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evade + forage joint training")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--days", type=int, default=100)
    p.add_argument("--steps-per-day", type=int, default=100)
    p.add_argument("--grass-area", type=int, default=10)
    p.add_argument("--grass-count", type=int, default=3)
    p.add_argument("--food-count", type=int, default=3)
    p.add_argument("--hunger-interval", type=int, default=25)
    p.add_argument("--hunger-death-steps", type=int, default=75)
    p.add_argument("--vision-radius", type=int, default=3)
    p.add_argument("--food-spawn-mode", type=str, choices=["static", "spawn_on_hunger"], default="static")
    p.add_argument("--food-spawn-radius", type=int, default=8)
    p.add_argument("--max-food-on-map", type=int, default=6)
    p.add_argument("--epsilon", type=float, default=0.08)
    p.add_argument("--motion-checkpoint-path", type=str, default="artifacts/train.ckpt")
    p.add_argument("--dashboard-path", type=str, default="artifacts/evade_dashboard.html")
    p.add_argument("--metrics-csv-path", type=str, default="artifacts/evade_metrics.csv")
    p.add_argument("--master-dashboard-path", default="artifacts/master_dashboard.html")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
