from __future__ import annotations

import random
from dataclasses import dataclass

from .render import render_view
from .utils import CENTER_CELL, GRID_MAX

MOVE_DELTAS = {
    "+x": (1, 0),
    "-x": (-1, 0),
    "+y": (0, 1),
    "-y": (0, -1),
}
TURN_DIRS = {"CW", "CCW"}


@dataclass(frozen=True)
class Action:
    leg_id: int
    move_dir: str
    turn_dir: str


class SuperblockEnv:
    def __init__(
        self,
        init_points: list[tuple[int, int]] | None = None,
        render_height: int = 32,
        render_width: int = 16,
        visible_cells: list[tuple[int, int]] | None = None,
    ) -> None:
        self.init_points = init_points or [(2, 2)]
        self.render_height = render_height
        self.render_width = render_width
        self.visible_cells = visible_cells or [CENTER_CELL]
        self.points: list[tuple[int, int]] = []
        self.reset()

    def reset(self) -> tuple[list[int], list[list[float]]]:
        self.points = list(self.init_points)
        return self.get_state(), self.render()

    def get_state(self) -> list[int]:
        return [coord for p in self.points for coord in p]

    def set_visible_cells(self, cells: list[tuple[int, int]]) -> None:
        self.visible_cells = list(cells)

    def render(self) -> list[list[float]]:
        return render_view(
            self.points,
            height=self.render_height,
            width=self.render_width,
            visible_cells=self.visible_cells,
        )

    @staticmethod
    def sample_action(rng: random.Random) -> Action:
        return Action(
            leg_id=rng.randint(0, 3),
            move_dir=rng.choice(list(MOVE_DELTAS.keys())),
            turn_dir=rng.choice(list(TURN_DIRS)),
        )

    def peek_step(self, points: list[tuple[int, int]], action: Action) -> tuple[list[tuple[int, int]], bool]:
        if action.move_dir not in MOVE_DELTAS:
            raise ValueError(f"Unsupported move_dir: {action.move_dir}")
        if action.turn_dir not in TURN_DIRS:
            raise ValueError(f"Unsupported turn_dir: {action.turn_dir}")
        if action.leg_id not in {0, 1, 2, 3}:
            raise ValueError(f"Unsupported leg_id: {action.leg_id}")

        dx, dy = MOVE_DELTAS[action.move_dir]
        moved_points = [(x + dx, y + dy) for x, y in points]
        invalid_move = any(x < 0 or y < 0 or x > GRID_MAX or y > GRID_MAX for x, y in moved_points)
        next_points = list(points) if invalid_move else moved_points

        if len(next_points) > 1:
            if action.turn_dir == "CW":
                next_points = [next_points[-1], *next_points[:-1]]
            else:
                next_points = [*next_points[1:], next_points[0]]
        return next_points, invalid_move

    def step(self, action: Action) -> tuple[list[int], list[list[float]], bool]:
        self.points, invalid_move = self.peek_step(self.points, action)
        return self.get_state(), self.render(), invalid_move
