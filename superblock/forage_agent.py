from __future__ import annotations

import random

from .env import Action
from .forage_env import ForageEnv
from .policy_curiosity import CuriosityMemory, CuriosityPolicy, state_to_center_cell


class FoodMemory:
    def __init__(self) -> None:
        self.food_cells: set[tuple[int, int]] = set()

    def update(self, cells: list[tuple[int, int]]) -> None:
        self.food_cells.update(cells)

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

    def _select_towards_food(self, state_t: list[int], env: ForageEnv) -> Action:
        current = state_to_center_cell(state_t)
        target = min(
            self.food_memory.food_cells,
            key=lambda cell: abs(cell[0] - current[0]) + abs(cell[1] - current[1]),
        )

        best_action = self.curiosity_policy.candidate_actions()[0]
        best_dist = float("inf")
        for action in self.curiosity_policy.candidate_actions():
            if self.use_forward_model_for_food_nav:
                pred_state = self.curiosity_policy._predict_next_state(state_t, action)
                pred_center = state_to_center_cell(pred_state)
                dist = abs(pred_center[0] - target[0]) + abs(pred_center[1] - target[1])
                _, invalid = env.base_env.peek_step(env.points, action)
            else:
                next_points, invalid = env.base_env.peek_step(env.points, action)
                cx = sum(x for x, _ in next_points) / len(next_points)
                cy = sum(y for _, y in next_points) / len(next_points)
                next_center = round(cx), round(cy)
                dist = abs(next_center[0] - target[0]) + abs(next_center[1] - target[1])
            if invalid:
                dist += 1e6
            if dist < best_dist:
                best_dist = dist
                best_action = action
        return best_action

    def select_action(self, state_t: list[int], env: ForageEnv, rng: random.Random) -> Action:
        if env.hungry and self.food_memory.food_cells:
            return self._select_towards_food(state_t, env)
        return self.curiosity_policy.select_action(state_t, env.base_env, rng)
