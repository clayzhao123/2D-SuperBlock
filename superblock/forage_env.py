from __future__ import annotations

import random

from .env import Action, SuperblockEnv
from .utils import GRID_MAX, occupied_cells as shape_occupied_cells, position_key


class ForageEnv:
    def __init__(
        self,
        *,
        seed: int = 42,
        food_count: int = 3,
        hunger_interval: int = 25,
        hunger_death_steps: int = 75,
        init_points: list[tuple[int, int]] | None = None,
        food_spawn_mode: str = "static",
        food_spawn_radius: int = 8,
        max_food_on_map: int = 6,
        eat_mode: str = "overlap",
        occupy_mode: str = "single_cell",
        vision_from: str = "all_edges",
    ) -> None:
        self.seed = seed
        self.food_count = food_count
        self.hunger_interval = hunger_interval
        self.hunger_death_steps = hunger_death_steps
        self.food_spawn_mode = food_spawn_mode
        self.food_spawn_radius = max(0, food_spawn_radius)
        self.max_food_on_map = max(1, max_food_on_map)
        if eat_mode not in {"center", "overlap"}:
            raise ValueError(f"Unsupported eat_mode: {eat_mode}")
        self.eat_mode = eat_mode
        if occupy_mode not in {"single_cell", "shape"}:
            raise ValueError(f"Unsupported occupy_mode: {occupy_mode}")
        if vision_from not in {"center", "all_edges"}:
            raise ValueError(f"Unsupported vision_from: {vision_from}")
        self.occupy_mode = occupy_mode
        self.vision_from = vision_from

        self.base_env = SuperblockEnv(init_points=init_points)
        self.points = self.base_env.points
        self.food_cells: list[tuple[int, int]] = []
        self._day_rng = random.Random(self.seed)

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
        self._day_rng = random.Random(self.seed + day_idx)
        self.food_cells = [
            (self._day_rng.randint(0, GRID_MAX), self._day_rng.randint(0, GRID_MAX)) for _ in range(self.food_count)
        ]
        return state, img

    def center_cell(self) -> tuple[int, int]:
        # Backward-compatible API: center_cell now uses the shared stable key helper.
        return position_key(self.points)

    def occupied_cells(self, points_or_state: list[tuple[int, int]] | list[int] | None = None) -> list[tuple[int, int]]:
        source: list[tuple[int, int]] | list[int] = self.points if points_or_state is None else points_or_state
        if self.occupy_mode == "single_cell":
            return [position_key(source)]
        return shape_occupied_cells(source)

    def _spawn_food_near(self, center: tuple[int, int], rng: random.Random) -> tuple[int, int]:
        occupied = set(self.occupied_cells())
        existing = set(self.food_cells)
        cx, cy = center

        # 优先尝试随机采样，保持生成位置多样性。
        for _ in range(64):
            x = rng.randint(max(0, cx - self.food_spawn_radius), min(GRID_MAX, cx + self.food_spawn_radius))
            y = rng.randint(max(0, cy - self.food_spawn_radius), min(GRID_MAX, cy + self.food_spawn_radius))
            if abs(x - cx) + abs(y - cy) > self.food_spawn_radius:
                continue
            if (x, y) in occupied or (x, y) in existing:
                continue
            return x, y

        # 回退：在半径内按距离扫描，尽量找到合法空位。
        for d in range(self.food_spawn_radius + 1):
            for dx in range(-d, d + 1):
                dy = d - abs(dx)
                for sy in {-1, 1} if dy else {1}:
                    x = cx + dx
                    y = cy + sy * dy
                    if x < 0 or y < 0 or x > GRID_MAX or y > GRID_MAX:
                        continue
                    if (x, y) in occupied or (x, y) in existing:
                        continue
                    return x, y

        # 退无可退：返回中心（调用方会去重，避免重复追加）。
        return center

    def _maybe_spawn_food_on_hunger(self) -> None:
        if self.food_spawn_mode != "spawn_on_hunger":
            return
        if len(self.food_cells) >= self.max_food_on_map:
            return
        new_food = self._spawn_food_near(self.center_cell(), self._day_rng)
        if new_food not in self.food_cells and new_food not in self.points:
            self.food_cells.append(new_food)

    def observe_visible_food(self, vision_radius: int = 0) -> list[tuple[int, int]]:
        if self.vision_from == "center":
            view_sources = {self.center_cell()}
        else:
            view_sources = set(self.points)
            view_sources.add(self.center_cell())
        visible = []
        for fx, fy in self.food_cells:
            if any(abs(fx - sx) <= vision_radius and abs(fy - sy) <= vision_radius for sx, sy in view_sources):
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
            self._maybe_spawn_food_on_hunger()

        if self.hungry:
            self.hungry_steps += 1

        if self.eat_mode == "center":
            eaten_cells = {self.center_cell()} if self.center_cell() in self.food_cells else set()
        else:
            occupied = set(self.occupied_cells())
            eaten_cells = {cell for cell in occupied if cell in self.food_cells}
        ate_food = bool(eaten_cells)
        if ate_food and self.hungry:
            self.food_cells = [cell for cell in self.food_cells if cell not in eaten_cells]
            self.forage_success += 1
            self.hungry = False
            self.steps_since_last_meal = 0
            self.hungry_steps = 0

        if self.hungry and self.hungry_steps >= self.hunger_death_steps:
            self.done = True

        return state, img, invalid, self.done, ate_food
