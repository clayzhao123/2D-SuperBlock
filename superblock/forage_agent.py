from __future__ import annotations

import random

from .env import Action
from .forage_env import ForageEnv
from .policy_curiosity import CuriosityMemory, CuriosityPolicy


def _min_food_distance(cells: list[tuple[int, int]], target: tuple[int, int]) -> int:
    return min(abs(x - target[0]) + abs(y - target[1]) for x, y in cells)


class FoodMemory:
    def __init__(self) -> None:
        self.food_cells: set[tuple[int, int]] = set()

    def update(self, cells: list[tuple[int, int]]) -> None:
        self.food_cells.update(cells)

    def retain(self, live_cells: set[tuple[int, int]]) -> None:
        self.food_cells.intersection_update(live_cells)

    def reset_day(self) -> None:
        self.food_cells.clear()


class ForagePolicy:
    def __init__(
        self,
        curiosity_policy: CuriosityPolicy,
        food_memory: FoodMemory,
        use_forward_model_for_food_nav: bool = False,
    ) -> None:
        self.curiosity_policy = curiosity_policy
        self.food_memory = food_memory
        # 默认关闭：觅食导航优先使用 env.peek_step 的真实下一步结果，
        # 避免 forward model 误差导致“越走越远”。保留该开关用于后续 A/B 对比。
        self.use_forward_model_for_food_nav = use_forward_model_for_food_nav

    def _select_towards_food(self, state_t: list[int], env: ForageEnv, rng: random.Random) -> Action:
        current_cells = env.occupied_cells(state_t)
        live_food = set(env.food_cells)
        tracked_food = self.food_memory.food_cells & live_food
        candidate_food = tracked_food or self.food_memory.food_cells or live_food
        if not candidate_food:
            return self.curiosity_policy.select_action(state_t, env.base_env, rng)

        target = min(
            candidate_food,
            key=lambda cell: _min_food_distance(current_cells, cell),
        )

        best_action = self.curiosity_policy.candidate_actions()[0]
        best_dist = float("inf")
        for action in self.curiosity_policy.candidate_actions():
            if self.use_forward_model_for_food_nav:
                pred_state = self.curiosity_policy._predict_next_state(state_t, action)
                pred_cells = env.occupied_cells(pred_state)
                dist = _min_food_distance(pred_cells, target)
                _, invalid = env.base_env.peek_step(env.points, action)
            else:
                next_points, invalid = env.base_env.peek_step(env.points, action)
                dist = _min_food_distance(env.occupied_cells(next_points), target)
            if invalid:
                dist += 1e6
            if dist < best_dist:
                best_dist = dist
                best_action = action
        return best_action

    def select_action(self, state_t: list[int], env: ForageEnv, rng: random.Random) -> Action:
        if env.hungry and (self.food_memory.food_cells or env.food_cells):
            return self._select_towards_food(state_t, env, rng)
        return self.curiosity_policy.select_action(state_t, env.base_env, rng)
