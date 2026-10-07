"""Tabular reward-driven forage route; selected with --policy qlearn."""

from __future__ import annotations

import random

from ..env import Action
from .heuristic import FoodMemory
from ..forage_env import ForageEnv
from ..utils import position_key


def _min_food_distance(cells: list[tuple[int, int]], food: tuple[int, int]) -> int:
    return min(abs(x - food[0]) + abs(y - food[1]) for x, y in cells)


def _nearest_food(cells: list[tuple[int, int]], foods: set[tuple[int, int]]) -> tuple[int, int] | None:
    if not foods:
        return None
    return min(foods, key=lambda cell: _min_food_distance(cells, cell))


def _sgn(value: int) -> int:
    return 1 if value > 0 else (-1 if value < 0 else 0)


class QLearnForagePolicy:
    def __init__(
        self,
        *,
        actions: list[Action],
        alpha: float,
        gamma: float,
        epsilon: float,
        epsilon_min: float,
        epsilon_decay: float,
        cell_div: int,
    ) -> None:
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.cell_div = max(1, cell_div)
        self.q_table: dict[tuple[int, int, int, int, int, int], list[float]] = {}

    def _feature(self, state_t: list[int], env: ForageEnv, food_memory: FoodMemory) -> tuple[int, int, int, int, int, int]:
        cells = env.occupied_cells(state_t)
        px, py = position_key(state_t)
        target = _nearest_food(cells, food_memory.food_cells) or _nearest_food(cells, set(env.food_cells))
        dx_s, dy_s = 0, 0
        if target is not None:
            nearest = min(cells, key=lambda cell: abs(cell[0] - target[0]) + abs(cell[1] - target[1]))
            dx_s = _sgn(target[0] - nearest[0])
            dy_s = _sgn(target[1] - nearest[1])

        remaining = max(0, env.hunger_death_steps - env.hungry_steps)
        remaining_bucket = min(4, remaining // max(1, env.hunger_death_steps // 5 or 1))
        return (px // self.cell_div, py // self.cell_div, 1 if env.hungry else 0, dx_s, dy_s, remaining_bucket)

    def _ensure_row(self, feat: tuple[int, int, int, int, int, int]) -> list[float]:
        if feat not in self.q_table:
            self.q_table[feat] = [0.0 for _ in self.actions]
        return self.q_table[feat]

    def select_action(self, feat: tuple[int, int, int, int, int, int], rng: random.Random) -> tuple[int, Action]:
        q_values = self._ensure_row(feat)
        if rng.random() < self.epsilon:
            idx = rng.randrange(len(self.actions))
            return idx, self.actions[idx]
        best_idx = max(range(len(self.actions)), key=lambda i: q_values[i])
        return best_idx, self.actions[best_idx]

    def update(
        self,
        feat: tuple[int, int, int, int, int, int],
        action_idx: int,
        reward: float,
        next_feat: tuple[int, int, int, int, int, int],
        done: bool,
    ) -> None:
        q_values = self._ensure_row(feat)
        target = reward
        if not done:
            target += self.gamma * max(self._ensure_row(next_feat))
        q_values[action_idx] += self.alpha * (target - q_values[action_idx])

    def end_day(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def serialize(self) -> dict[str, list[float]]:
        return {"|".join(str(v) for v in key): values for key, values in self.q_table.items()}
