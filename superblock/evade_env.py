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
        init_points: list[tuple[int, int]] | None = None,
    ) -> None:
        self.seed = seed
        self.grass_area = max(1, grass_area)
        self.grass_count = max(0, grass_count)
        self.vision_length = max(1, vision_length)
        self.base_env = SuperblockEnv(init_points=init_points or [(2, 2)])
        self._day_rng = random.Random(seed)

        self.superhacker_pos = (GRID_MAX, GRID_MAX)
        self.superhacker_dir = (-1, 0)
        self.mode = "patrol"
        self.search_center = self.superhacker_pos
        self.search_points: list[tuple[int, int]] = []
        self.search_idx = 0
        self.last_seen = self.superhacker_pos
        self.grass_cells: set[tuple[int, int]] = set()
        self.done = False

    def reset_day(self, day_idx: int) -> tuple[list[int], list[list[float]]]:
        state, img = self.base_env.reset()
        self._day_rng = random.Random(self.seed + day_idx)
        self.superhacker_pos = (GRID_MAX, GRID_MAX)
        self.superhacker_dir = (-1, 0)
        self.mode = "patrol"
        self.search_center = self.superhacker_pos
        self.search_points = []
        self.search_idx = 0
        self.last_seen = self.superhacker_pos
        self.grass_cells = self._spawn_grass(self._day_rng)
        self.done = False
        return state, img

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

        state, img, invalid = self.base_env.step(action)
        prey = position_key(state)
        seen = self._in_vision(self.superhacker_pos, prey)

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
        self.done = caught
        return state, img, invalid, self.done, caught
