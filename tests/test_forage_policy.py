from __future__ import annotations

import random

from superblock.forage_agent import FoodMemory, ForagePolicy
from superblock.forage_env import ForageEnv
from superblock.models import ForwardModel
from superblock.policy_curiosity import CuriosityMemory, CuriosityPolicy, state_to_center_cell


def _center_dist(state_t: list[int], food_cell: tuple[int, int]) -> int:
    cx, cy = state_to_center_cell(state_t)
    return abs(cx - food_cell[0]) + abs(cy - food_cell[1])


def _build_policy(*, use_forward_model_for_food_nav: bool = False) -> ForagePolicy:
    curiosity = CuriosityPolicy(forward_model=ForwardModel(), memory=CuriosityMemory(), epsilon=0.0)
    return ForagePolicy(
        curiosity_policy=curiosity,
        food_memory=FoodMemory(),
        use_forward_model_for_food_nav=use_forward_model_for_food_nav,
    )


def test_hungry_food_navigation_reduces_distance_quickly() -> None:
    env = ForageEnv(init_points=[(2, 2), (3, 2), (3, 3), (2, 3)])
    state_t, _ = env.reset_day(day_idx=0)

    target_food = (6, 2)
    policy = _build_policy()
    policy.food_memory.update([target_food])

    env.hungry = True
    rng = random.Random(0)

    prev_dist = _center_dist(state_t, target_food)
    distances = [prev_dist]
    for _ in range(3):
        action = policy.select_action(state_t, env, rng)
        state_t, _, _, _, _ = env.step(action)
        next_dist = _center_dist(state_t, target_food)
        distances.append(next_dist)
        assert next_dist <= prev_dist
        prev_dist = next_dist
        if next_dist == 0:
            break

    assert min(distances[1:]) < distances[0]


def test_food_navigation_avoids_invalid_translate_near_boundary() -> None:
    env = ForageEnv(init_points=[(0, 0), (1, 0), (1, 1), (0, 1)])
    state_t, _ = env.reset_day(day_idx=0)

    policy = _build_policy()
    policy.food_memory.update([(10, 0)])

    env.hungry = True
    action = policy.select_action(state_t, env, random.Random(0))

    _, invalid = env.base_env.peek_step(env.points, action)
    assert not invalid
