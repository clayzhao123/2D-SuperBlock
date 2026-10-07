from __future__ import annotations

import argparse
import random
import warnings
from collections.abc import Callable

from .agents.qlearn import QLearnForagePolicy, _min_food_distance, _nearest_food
from .checkpoint import save_payload_checkpoint as save_ckpt
from .reporting.forage import write_dashboard, write_csv, write_attempts_csv
from .buffer import ReplayBuffer, Transition
from .env import Action
from .forage_agent import FoodMemory, ForagePolicy
from .reporting.summary import write_training_summary
from .forage_env import ForageEnv
from .monitor import load_checkpoint, save_checkpoint
from .policy_curiosity import CuriosityMemory, CuriosityPolicy, load_forward_model_from_ckpt
from .utils import position_key
from .train import action_to_onehot, train_night


def _percentile(values: list[int], q: float) -> float:
    if not values:
        return -1.0
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * q))
    return float(ordered[idx])


def run_with_callbacks(
    args: argparse.Namespace,
    *,
    on_step: Callable[[int, int, ForageEnv, list[tuple[float, float]]], None] | None = None,
    on_day_end: Callable[[dict[str, float]], None] | None = None,
) -> None:
    policy_name = getattr(args, "policy", "heuristic")
    if policy_name == "imitation":
        warnings.warn("imitation is not implemented; using heuristic policy", UserWarning, stacklevel=2)
        policy_name = "heuristic"

    q_alpha = getattr(args, "q_alpha", 0.3)
    q_gamma = getattr(args, "q_gamma", 0.95)
    q_epsilon = getattr(args, "q_epsilon", 0.2)
    q_epsilon_min = getattr(args, "q_epsilon_min", 0.02)
    q_epsilon_decay = getattr(args, "q_epsilon_decay", 0.99)
    q_cell_div = getattr(args, "q_cell_div", 4)
    occupy_mode = getattr(args, "occupy_mode", "single_cell")
    vision_from = getattr(args, "vision_from", "all_edges")

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
        food_spawn_mode=args.food_spawn_mode,
        food_spawn_radius=args.food_spawn_radius,
        max_food_on_map=args.max_food_on_map,
        eat_mode=args.eat_mode,
        occupy_mode=occupy_mode,
        vision_from=vision_from,
    )
    curiosity_memory = CuriosityMemory()
    curiosity_policy = CuriosityPolicy(forward_model=model, memory=curiosity_memory)
    food_memory = FoodMemory()
    forage_policy = ForagePolicy(curiosity_policy, food_memory)
    qlearn_policy: QLearnForagePolicy | None = None
    if policy_name == "qlearn":
        qlearn_policy = QLearnForagePolicy(
            actions=curiosity_policy.candidate_actions(),
            alpha=q_alpha,
            gamma=q_gamma,
            epsilon=q_epsilon,
            epsilon_min=q_epsilon_min,
            epsilon_decay=q_epsilon_decay,
            cell_div=q_cell_div,
        )

    history: list[dict[str, float]] = []
    attempts_history: list[dict[str, str]] = []
    deaths_total = 0
    all_success_latencies: list[int] = []
    food_seen_days_total = 0
    first_food_seen_steps_sum_total = 0

    for day_idx in range(1, args.days + 1):
        state, _ = env.reset_day(day_idx)
        food_memory.reset_day()
        curiosity_policy.recent_path.clear()
        curiosity_memory.update(state)

        attempts_before_day = env.forage_attempts
        successes_before_day = env.forage_success

        curiosity_sum = 0.0
        first_food_seen_step = -1
        day_success_latencies: list[int] = []
        day_attempt_idx = 0
        active_attempt: dict[str, object] | None = None
        day_path: list[tuple[float, float]] = [
            (sum(state[::2]) / max(1, len(state[::2])), sum(state[1::2]) / max(1, len(state[1::2])))
        ]

        for step_idx in range(1, args.steps_per_day + 1):
            visible = env.observe_visible_food(args.vision_radius)
            food_memory.retain(set(env.food_cells))
            food_memory.update(visible)
            if visible and first_food_seen_step < 0:
                first_food_seen_step = step_idx

            prev_state = state.copy()
            prev_img = env.base_env.render()
            prev_attempts = env.forage_attempts
            prev_successes = env.forage_success

            prev_cells = env.occupied_cells(state)
            prev_target = _nearest_food(prev_cells, food_memory.food_cells) or _nearest_food(prev_cells, set(env.food_cells))
            if qlearn_policy is not None:
                feat = qlearn_policy._feature(state, env, food_memory)
                action_idx, action = qlearn_policy.select_action(feat, rng)
            else:
                action = forage_policy.select_action(state, env, rng)
                action_idx = -1
            state, _, _, done, ate_food = env.step(action)
            food_memory.retain(set(env.food_cells))

            pos_key = position_key(state)
            curiosity_sum += 1.0 / (1.0 + curiosity_memory.count(pos_key))
            curiosity_memory.update(state)
            day_path.append((sum(state[::2]) / max(1, len(state[::2])), sum(state[1::2]) / max(1, len(state[1::2]))))
            if on_step is not None:
                on_step(day_idx, step_idx, env, day_path)

            if env.forage_attempts > prev_attempts:
                day_attempt_idx += 1
                target_food = _nearest_food(env.occupied_cells(state), food_memory.food_cells)
                active_attempt = {
                    "day_idx": day_idx,
                    "attempt_idx": day_attempt_idx,
                    "start_center": pos_key,
                    "target_food": target_food,
                    "start_step": step_idx,
                }

            if env.forage_success > prev_successes and active_attempt is not None:
                latency_steps = step_idx - int(active_attempt["start_step"]) + 1
                day_success_latencies.append(latency_steps)
                all_success_latencies.append(latency_steps)
                attempts_history.append(
                    {
                        "day_idx": str(day_idx),
                        "attempt_idx": str(active_attempt["attempt_idx"]),
                        "start_center": str(active_attempt["start_center"]),
                        "target_food": str(active_attempt["target_food"]),
                        "success": "1",
                        "latency_steps": str(latency_steps),
                    }
                )
                active_attempt = None

            if qlearn_policy is not None:
                reward = -0.01
                if prev_target is not None and env.hungry:
                    dist = _min_food_distance(env.occupied_cells(state), prev_target)
                    reward += -0.05 * dist
                if ate_food:
                    reward += 10.0
                if done:
                    reward -= 10.0
                next_feat = qlearn_policy._feature(state, env, food_memory)
                qlearn_policy.update(feat, action_idx, reward, next_feat, done)

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
                if active_attempt is not None:
                    attempts_history.append(
                        {
                            "day_idx": str(day_idx),
                            "attempt_idx": str(active_attempt["attempt_idx"]),
                            "start_center": str(active_attempt["start_center"]),
                            "target_food": str(active_attempt["target_food"]),
                            "success": "0",
                            "latency_steps": "-1",
                        }
                    )
                    active_attempt = None
                break

        if active_attempt is not None:
            attempts_history.append(
                {
                    "day_idx": str(day_idx),
                    "attempt_idx": str(active_attempt["attempt_idx"]),
                    "start_center": str(active_attempt["start_center"]),
                    "target_food": str(active_attempt["target_food"]),
                    "success": "0",
                    "latency_steps": "-1",
                }
            )

        run_motion_training = args.motion_epochs_per_day > 0
        if run_motion_training:
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

        hungry_attempts_total = env.forage_attempts
        hungry_success_total = env.forage_success
        hungry_attempts_today = hungry_attempts_total - attempts_before_day
        hungry_success_today = hungry_success_total - successes_before_day
        hungry_success_rate_today = hungry_success_today / max(1, hungry_attempts_today)
        hungry_success_rate_total = hungry_success_total / max(1, hungry_attempts_total)

        food_seen_flag_today = 1 if first_food_seen_step > 0 else 0
        if food_seen_flag_today:
            food_seen_days_total += 1
            first_food_seen_steps_sum_total += first_food_seen_step

        if qlearn_policy is not None:
            qlearn_policy.end_day()

        unreachable_food_seen_today = 0
        if args.eat_mode == "center" and first_food_seen_step > 0 and food_memory.food_cells:
            key = env.center_cell()
            if not any((abs(fx - key[0]) + abs(fy - key[1])) % 2 == 0 for fx, fy in food_memory.food_cells):
                unreachable_food_seen_today = 1

        row = {
            "day_idx": float(day_idx),
            "policy": 0.0 if policy_name == "heuristic" else 1.0,
            "hungry_attempts_today": float(hungry_attempts_today),
            "hungry_success_today": float(hungry_success_today),
            "hungry_success_rate_today": float(hungry_success_rate_today),
            "hungry_latency_mean_today": float(sum(day_success_latencies) / len(day_success_latencies)) if day_success_latencies else -1.0,
            "hungry_latency_p50_today": _percentile(day_success_latencies, 0.5),
            "hungry_latency_p90_today": _percentile(day_success_latencies, 0.9),
            "steps_to_first_food_seen_today": float(first_food_seen_step),
            "food_seen_flag_today": float(food_seen_flag_today),
            "food_count_on_map_today": float(len(env.food_cells)),
            "unreachable_food_seen_today": float(unreachable_food_seen_today),
            "hungry_attempts_total": float(hungry_attempts_total),
            "hungry_success_total": float(hungry_success_total),
            "hungry_success_rate_total": float(hungry_success_rate_total),
            "hungry_latency_mean_total": float(sum(all_success_latencies) / len(all_success_latencies)) if all_success_latencies else -1.0,
            "hungry_latency_p50_total": _percentile(all_success_latencies, 0.5),
            "hungry_latency_p90_total": _percentile(all_success_latencies, 0.9),
            "food_seen_days_total": float(food_seen_days_total),
            "steps_to_first_food_seen_mean_total": float(first_food_seen_steps_sum_total / food_seen_days_total)
            if food_seen_days_total
            else -1.0,
            "deaths_total": float(deaths_total),
            "curiosity_score": float(curiosity_sum / max(1, args.steps_per_day)),
        }
        if run_motion_training:
            row.update(
                {
                    "motion_pos_mse": float(motion_pos_mse),
                    "motion_exact_acc": float(motion_exact_acc),
                    "motion_near_acc": float(motion_near_acc),
                    "motion_score": float(motion_score),
                }
            )
        history.append(row)

        write_csv(args.out_metrics_csv, history)
        write_attempts_csv(args.out_attempts_csv, attempts_history)
        write_dashboard(args.out_dashboard, history, show_motion=run_motion_training, policy_name=policy_name)
        save_ckpt(
            args.out_checkpoint,
            {
                "day_idx": day_idx,
                "history": history,
                "hungry_attempts_total": hungry_attempts_total,
                "hungry_success_total": hungry_success_total,
                "deaths_total": deaths_total,
                "motion_checkpoint_path": args.motion_checkpoint_path,
                "policy": policy_name,
                "q_table": qlearn_policy.serialize() if qlearn_policy is not None else {},
                "params": {
                    "food_count": args.food_count,
                    "hunger_interval": args.hunger_interval,
                    "hunger_death_steps": args.hunger_death_steps,
                    "vision_radius": args.vision_radius,
                    "seed": args.seed,
                    "food_spawn_mode": args.food_spawn_mode,
                    "food_spawn_radius": args.food_spawn_radius,
                    "max_food_on_map": args.max_food_on_map,
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
        write_training_summary(args, "forage")
        if on_day_end is not None:
            on_day_end(row)

        print(
            f"policy={policy_name} day={day_idx} attempts_today={hungry_attempts_today} successes_today={hungry_success_today} "
            f"rate_today={hungry_success_rate_today:.3f} latency_mean_today={row['hungry_latency_mean_today']:.2f} "
            f"rate_total={hungry_success_rate_total:.3f} deaths={deaths_total}"
        )


def run(args: argparse.Namespace) -> None:
    run_with_callbacks(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Foraging training loop")
    parser.add_argument("--days", type=int, default=100)
    parser.add_argument("--steps-per-day", type=int, default=100)
    parser.add_argument("--food-count", type=int, default=3)
    parser.add_argument("--hunger-interval", type=int, default=25)
    parser.add_argument("--hunger-death-steps", type=int, default=75)
    parser.add_argument("--vision-radius", type=int, default=3)
    parser.add_argument("--food-spawn-mode", type=str, default="static")
    parser.add_argument("--food-spawn-radius", type=int, default=8)
    parser.add_argument("--max-food-on-map", type=int, default=6)
    parser.add_argument("--eat-mode", type=str, choices=["center", "overlap"], default="overlap")
    parser.add_argument("--occupy-mode", type=str, choices=["single_cell", "shape"], default="single_cell")
    parser.add_argument("--vision-from", type=str, choices=["center", "all_edges"], default="all_edges")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--policy", type=str, choices=["heuristic", "qlearn", "imitation"], default="heuristic")
    parser.add_argument("--q-alpha", type=float, default=0.3)
    parser.add_argument("--q-gamma", type=float, default=0.95)
    parser.add_argument("--q-epsilon", type=float, default=0.2)
    parser.add_argument("--q-epsilon-min", type=float, default=0.02)
    parser.add_argument("--q-epsilon-decay", type=float, default=0.99)
    parser.add_argument("--q-cell-div", type=int, default=4)
    parser.add_argument("--motion-checkpoint-path", type=str, default="artifacts/train.ckpt")
    parser.add_argument("--motion-output-checkpoint", type=str, default="artifacts/train_forage_tuned.ckpt")
    # 默认关闭 motion 日训练，避免 forage_train 输出看起来像“运动训练”。
    parser.add_argument("--motion-epochs-per-day", type=int, default=0)
    parser.add_argument("--motion-batch-size", type=int, default=64)
    parser.add_argument("--motion-lr", type=float, default=1e-2)
    parser.add_argument("--motion-score-k", type=float, default=50.0)
    parser.add_argument("--motion-score-w-mse", type=float, default=0.5)
    parser.add_argument("--motion-score-w-exact", type=float, default=0.2)
    parser.add_argument("--motion-score-w-near", type=float, default=0.3)
    parser.add_argument("--out-checkpoint", type=str, default="artifacts/forage.ckpt")
    parser.add_argument("--out-metrics-csv", type=str, default="artifacts/forage_metrics.csv")
    parser.add_argument("--out-attempts-csv", type=str, default="")
    parser.add_argument("--out-dashboard", type=str, default="artifacts/forage_dashboard.html")
    parser.add_argument("--master-dashboard-path", default="artifacts/master_dashboard.html")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
