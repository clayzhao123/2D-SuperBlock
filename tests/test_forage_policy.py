from __future__ import annotations

import random

from superblock.forage_agent import FoodMemory, ForagePolicy
from superblock.forage_env import ForageEnv
from superblock.models import ForwardModel
from superblock.policy_curiosity import CuriosityMemory, CuriosityPolicy
from superblock.utils import occupied_cells


def _food_dist(state_t: list[int], food_cell: tuple[int, int]) -> int:
    return min(abs(x - food_cell[0]) + abs(y - food_cell[1]) for x, y in occupied_cells(state_t))


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

    prev_dist = _food_dist(state_t, target_food)
    distances = [prev_dist]
    for _ in range(3):
        action = policy.select_action(state_t, env, rng)
        state_t, _, _, _, _ = env.step(action)
        next_dist = _food_dist(state_t, target_food)
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


def test_curiosity_policy_does_not_always_pick_same_line_action() -> None:
    env = ForageEnv(init_points=[(20, 20), (21, 20), (21, 21), (20, 21)])
    state_t, _ = env.reset_day(day_idx=0)
    curiosity = CuriosityPolicy(forward_model=ForwardModel(), memory=CuriosityMemory(), epsilon=0.0)

    actions = {curiosity.select_action(state_t, env.base_env, random.Random(seed)).move_dir for seed in range(20)}
    assert len(actions) >= 2


def test_food_navigation_ignores_stale_memory_and_tracks_live_food() -> None:
    env = ForageEnv(init_points=[(5, 5), (6, 5), (6, 6), (5, 6)])
    state_t, _ = env.reset_day(day_idx=0)

    policy = _build_policy()
    policy.food_memory.update([(30, 30)])  # stale memory
    env.food_cells = [(8, 5)]  # live food on map
    policy.food_memory.retain(set(env.food_cells))

    env.hungry = True
    action = policy.select_action(state_t, env, random.Random(0))
    next_points, _ = env.base_env.peek_step(env.points, action)

    cur_dist = min(abs(x - 8) + abs(y - 5) for x, y in env.occupied_cells(state_t))
    next_dist = min(abs(x - 8) + abs(y - 5) for x, y in env.occupied_cells(next_points))
    assert next_dist <= cur_dist


def test_food_memory_retain_removes_disappeared_food() -> None:
    memory = FoodMemory()
    memory.update([(1, 1), (2, 2), (3, 3)])

    memory.retain({(2, 2), (4, 4)})

    assert memory.food_cells == {(2, 2)}
