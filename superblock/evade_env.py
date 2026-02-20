from __future__ import annotations

import random

from .env import Action, SuperblockEnv
from .utils import GRID_MAX, position_key


class EvadeEnv:
    def __init__(
        self,
        *,
        seed: int = 42,
        grass_area: int = 10,
        grass_count: int = 3,
        vision_length: int = 3,
        food_count: int = 3,
        hunger_interval: int = 25,
        hunger_death_steps: int = 75,
        food_spawn_mode: str = "static",
        food_spawn_radius: int = 8,
        max_food_on_map: int = 6,
        init_points: list[tuple[int, int]] | None = None,
    ) -> None:
        self.seed = seed
        self.grass_area = max(1, grass_area)
        self.grass_count = max(0, grass_count)
        self.vision_length = max(1, vision_length)
        self.food_count = max(0, food_count)
        self.hunger_interval = max(1, hunger_interval)
        self.hunger_death_steps = max(1, hunger_death_steps)
        if food_spawn_mode not in {"static", "spawn_on_hunger"}:
            raise ValueError(f"Unsupported food_spawn_mode: {food_spawn_mode}")
        self.food_spawn_mode = food_spawn_mode
        self.food_spawn_radius = max(0, food_spawn_radius)
        self.max_food_on_map = max(1, max_food_on_map)
        self.base_env = SuperblockEnv(init_points=init_points or [(2, 2)])
        self.points = self.base_env.points
        self._day_rng = random.Random(seed)

        self.superhacker_pos = (GRID_MAX, GRID_MAX)
        self.superhacker_dir = (-1, 0)
        self.mode = "patrol"
        self.search_center = self.superhacker_pos
        self.search_points: list[tuple[int, int]] = []
        self.search_idx = 0
        self.last_seen = self.superhacker_pos
        self.grass_cells: set[tuple[int, int]] = set()
        self.food_cells: list[tuple[int, int]] = []
        self.steps_since_last_meal = 0
        self.hungry = False
        self.hungry_steps = 0
        self.forage_attempts = 0
        self.forage_success = 0
        self.last_ate_food = False
        self.last_hunger_death = False
        self.last_caught = False
        self.done = False

    def reset_day(self, day_idx: int) -> tuple[list[int], list[list[float]]]:
        state, img = self.base_env.reset()
        self.points = self.base_env.points
        self._day_rng = random.Random(self.seed + day_idx)
        self.superhacker_pos = (GRID_MAX, GRID_MAX)
        self.superhacker_dir = (-1, 0)
        self.mode = "patrol"
        self.search_center = self.superhacker_pos
        self.search_points = []
        self.search_idx = 0
        self.last_seen = self.superhacker_pos
        self.grass_cells = self._spawn_grass(self._day_rng)
        self.food_cells = [
            (self._day_rng.randint(0, GRID_MAX), self._day_rng.randint(0, GRID_MAX)) for _ in range(self.food_count)
        ]
        self.steps_since_last_meal = 0
        self.hungry = False
        self.hungry_steps = 0
        self.last_ate_food = False
        self.last_hunger_death = False
        self.last_caught = False
        self.done = False
        return state, img

    def center_cell(self) -> tuple[int, int]:
        return position_key(self.base_env.points)

    def occupied_cells(self, points_or_state: list[tuple[int, int]] | list[int] | None = None) -> list[tuple[int, int]]:
        source: list[tuple[int, int]] | list[int] = self.base_env.points if points_or_state is None else points_or_state
        return [position_key(source)]

    def _spawn_food_near(self, center: tuple[int, int], rng: random.Random) -> tuple[int, int]:
        occupied = set(self.occupied_cells())
        existing = set(self.food_cells)
        cx, cy = center

        for _ in range(64):
            x = rng.randint(max(0, cx - self.food_spawn_radius), min(GRID_MAX, cx + self.food_spawn_radius))
            y = rng.randint(max(0, cy - self.food_spawn_radius), min(GRID_MAX, cy + self.food_spawn_radius))
            if abs(x - cx) + abs(y - cy) > self.food_spawn_radius:
                continue
            if (x, y) in occupied or (x, y) in existing:
                continue
            return x, y

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
        return center

    def _maybe_spawn_food_on_hunger(self) -> None:
        if self.food_spawn_mode != "spawn_on_hunger":
            return
        if len(self.food_cells) >= self.max_food_on_map:
            return
        new_food = self._spawn_food_near(self.center_cell(), self._day_rng)
        if new_food not in self.food_cells and new_food not in self.base_env.points:
            self.food_cells.append(new_food)

    def observe_visible_food(self, vision_radius: int = 0) -> list[tuple[int, int]]:
        center = self.center_cell()
        visible = []
        for fx, fy in self.food_cells:
            if abs(fx - center[0]) <= vision_radius and abs(fy - center[1]) <= vision_radius:
                visible.append((fx, fy))
        return visible

    def _spawn_grass(self, rng: random.Random) -> set[tuple[int, int]]:
        grass: set[tuple[int, int]] = set()
        for _ in range(self.grass_count):
            sx, sy = rng.randint(0, GRID_MAX), rng.randint(0, GRID_MAX)
            patch = {(sx, sy)}
            frontier = [(sx, sy)]
            while len(patch) < self.grass_area and frontier:
                cx, cy = frontier[rng.randrange(len(frontier))]
                dx, dy = rng.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
                nx, ny = max(0, min(GRID_MAX, cx + dx)), max(0, min(GRID_MAX, cy + dy))
                if (nx, ny) not in patch:
                    patch.add((nx, ny))
                    frontier.append((nx, ny))
                if len(frontier) > self.grass_area * 3:
                    frontier.pop(0)
            grass.update(patch)
        return grass

    def _in_vision(self, hunter: tuple[int, int], prey: tuple[int, int]) -> bool:
        if prey in self.grass_cells:
            return False
        hx, hy = hunter
        px, py = prey
        if hx == px:
            return abs(py - hy) <= self.vision_length
        if hy == py:
            return abs(px - hx) <= self.vision_length
        return False

    def _step_towards(self, src: tuple[int, int], dst: tuple[int, int]) -> tuple[int, int]:
        sx, sy = src
        dx = dst[0] - sx
        dy = dst[1] - sy
        if abs(dx) >= abs(dy) and dx != 0:
            sx += 1 if dx > 0 else -1
        elif dy != 0:
            sy += 1 if dy > 0 else -1
        return max(0, min(GRID_MAX, sx)), max(0, min(GRID_MAX, sy))

    def _patrol_step(self) -> None:
        x, y = self.superhacker_pos
        dx, dy = self.superhacker_dir
        nx, ny = x + dx, y + dy
        if nx < 0 or ny < 0 or nx > GRID_MAX or ny > GRID_MAX:
            self.superhacker_dir = (-dy, dx)
            dx, dy = self.superhacker_dir
            nx, ny = x + dx, y + dy
        self.superhacker_pos = (max(0, min(GRID_MAX, nx)), max(0, min(GRID_MAX, ny)))

    def _build_search_route(self, center: tuple[int, int]) -> list[tuple[int, int]]:
        cx, cy = center
        route = [
            (cx + 1, cy),
            (cx + 1, cy + 1),
            (cx, cy + 1),
            (cx - 1, cy + 1),
            (cx - 1, cy),
            (cx - 1, cy - 1),
            (cx, cy - 1),
            (cx + 1, cy - 1),
        ]
        return [(max(0, min(GRID_MAX, x)), max(0, min(GRID_MAX, y))) for x, y in route]

    def step(self, action: Action) -> tuple[list[int], list[list[float]], bool, bool, bool]:
        if self.done:
            return self.base_env.get_state(), self.base_env.render(), False, True, False

        self.last_ate_food = False
        self.last_hunger_death = False
        self.last_caught = False

        state, img, invalid = self.base_env.step(action)
        self.points = self.base_env.points
        prey = position_key(state)
        seen = self._in_vision(self.superhacker_pos, prey)

        self.steps_since_last_meal += 1
        if not self.hungry and self.steps_since_last_meal % self.hunger_interval == 0:
            self.hungry = True
            self.hungry_steps = 0
            self.forage_attempts += 1
            self._maybe_spawn_food_on_hunger()
        if self.hungry:
            self.hungry_steps += 1

        eaten_cells = {cell for cell in self.occupied_cells(state) if cell in self.food_cells}
        self.last_ate_food = bool(eaten_cells)
        if self.last_ate_food and self.hungry:
            self.food_cells = [cell for cell in self.food_cells if cell not in eaten_cells]
            self.forage_success += 1
            self.hungry = False
            self.steps_since_last_meal = 0
            self.hungry_steps = 0

        if seen:
            self.mode = "chase"
            self.last_seen = prey

        if self.mode == "chase":
            self.superhacker_pos = self._step_towards(self.superhacker_pos, prey)
            if not self._in_vision(self.superhacker_pos, prey):
                self.mode = "search"
                self.search_center = self.last_seen
                self.search_points = self._build_search_route(self.search_center)
                self.search_idx = 0
        elif self.mode == "search":
            if self.search_idx < len(self.search_points):
                self.superhacker_pos = self.search_points[self.search_idx]
                self.search_idx += 1
            if self._in_vision(self.superhacker_pos, prey):
                self.mode = "chase"
                self.last_seen = prey
            elif self.search_idx >= len(self.search_points):
                self.mode = "patrol"
        else:
            self._patrol_step()

        caught = self.superhacker_pos == prey
        self.last_caught = caught
        self.last_hunger_death = (self.hungry and self.hungry_steps >= self.hunger_death_steps) and not caught
        self.done = self.last_caught or self.last_hunger_death
        return state, img, invalid, self.done, caught
