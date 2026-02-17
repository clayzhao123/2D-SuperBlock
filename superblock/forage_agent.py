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
    def __init__(self, curiosity_policy: CuriosityPolicy, food_memory: FoodMemory) -> None:
        self.curiosity_policy = curiosity_policy
        self.food_memory = food_memory

    def _select_towards_food(self, state_t: list[int], env: ForageEnv) -> Action:
        current = state_to_center_cell(state_t)
        target = min(
            self.food_memory.food_cells,
            key=lambda cell: abs(cell[0] - current[0]) + abs(cell[1] - current[1]),
        )

        best_action = self.curiosity_policy.candidate_actions()[0]
        best_dist = float("inf")
        for action in self.curiosity_policy.candidate_actions():
            pred_state = self.curiosity_policy._predict_next_state(state_t, action)
            pred_center = state_to_center_cell(pred_state)
            dist = abs(pred_center[0] - target[0]) + abs(pred_center[1] - target[1])
            _, invalid = env.base_env.peek_step(env.points, action)
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
