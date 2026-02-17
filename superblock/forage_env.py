from __future__ import annotations

import random

from .env import Action, SuperblockEnv
from .utils import GRID_MAX


class ForageEnv:
    def __init__(
        self,
        *,
        seed: int = 42,
        food_count: int = 3,
        hunger_interval: int = 25,
        hunger_death_steps: int = 75,
        init_points: list[tuple[int, int]] | None = None,
    ) -> None:
        self.seed = seed
        self.food_count = food_count
        self.hunger_interval = hunger_interval
        self.hunger_death_steps = hunger_death_steps

        self.base_env = SuperblockEnv(init_points=init_points)
        self.points = self.base_env.points
        self.food_cells: list[tuple[int, int]] = []

        self.steps_since_last_meal = 0
        self.hungry = False
        self.hungry_steps = 0
        self.done = False

        self.forage_attempts = 0
        self.forage_success = 0

    def reset_day(self, day_idx: int) -> tuple[list[int], list[list[float]]]:
        state, img = self.base_env.reset()
        self.points = self.base_env.points
        self.steps_since_last_meal = 0
        self.hungry = False
        self.hungry_steps = 0
        self.done = False
        day_rng = random.Random(self.seed + day_idx)
        self.food_cells = [
            (day_rng.randint(0, GRID_MAX), day_rng.randint(0, GRID_MAX)) for _ in range(self.food_count)
        ]
        return state, img

    def center_cell(self) -> tuple[int, int]:
        cx = sum(x for x, _ in self.points) / len(self.points)
        cy = sum(y for _, y in self.points) / len(self.points)
        return round(cx), round(cy)

    def observe_visible_food(self, vision_radius: int = 0) -> list[tuple[int, int]]:
        cx, cy = self.center_cell()
        visible = []
        for fx, fy in self.food_cells:
            if abs(fx - cx) <= vision_radius and abs(fy - cy) <= vision_radius:
                visible.append((fx, fy))
        return visible

    def step(self, action: Action) -> tuple[list[int], list[list[float]], bool, bool, bool]:
        if self.done:
            return self.base_env.get_state(), self.base_env.render(), False, True, False

        state, img, invalid = self.base_env.step(action)
        self.points = self.base_env.points

        self.steps_since_last_meal += 1
        if not self.hungry and self.steps_since_last_meal % self.hunger_interval == 0:
            self.hungry = True
            self.hungry_steps = 0
            self.forage_attempts += 1

        if self.hungry:
            self.hungry_steps += 1

        ate_food = self.center_cell() in self.food_cells
        if ate_food and self.hungry:
            self.forage_success += 1
            self.hungry = False
            self.steps_since_last_meal = 0
            self.hungry_steps = 0

        if self.hungry and self.hungry_steps >= self.hunger_death_steps:
            self.done = True

        return state, img, invalid, self.done, ate_food
