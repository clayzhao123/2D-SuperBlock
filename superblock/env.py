from __future__ import annotations

import random
from dataclasses import dataclass

from .render import render_view
from .utils import GRID_MAX

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
    ) -> None:
        self.init_points = init_points or [(2, 2), (3, 2), (3, 3), (2, 3)]
        self.render_height = render_height
        self.render_width = render_width
        self.points: list[tuple[int, int]] = []
        self.reset()

    def reset(self) -> tuple[list[int], list[list[float]]]:
        self.points = list(self.init_points)
        return self.get_state(), self.render()

    def get_state(self) -> list[int]:
        return [coord for p in self.points for coord in p]

    def render(self) -> list[list[float]]:
        return render_view(self.points, height=self.render_height, width=self.render_width)

    @staticmethod
    def sample_action(rng: random.Random) -> Action:
        return Action(
            leg_id=rng.randint(0, 3),
            move_dir=rng.choice(list(MOVE_DELTAS.keys())),
            turn_dir=rng.choice(list(TURN_DIRS)),
        )

    def step(self, action: Action) -> tuple[list[int], list[list[float]], bool]:
        if action.move_dir not in MOVE_DELTAS:
            raise ValueError(f"Unsupported move_dir: {action.move_dir}")
        if action.turn_dir not in TURN_DIRS:
            raise ValueError(f"Unsupported turn_dir: {action.turn_dir}")
        if action.leg_id not in {0, 1, 2, 3}:
            raise ValueError(f"Unsupported leg_id: {action.leg_id}")

        dx, dy = MOVE_DELTAS[action.move_dir]
        moved_points = [(x + dx, y + dy) for x, y in self.points]
        invalid_move = any(x < 0 or y < 0 or x > GRID_MAX or y > GRID_MAX for x, y in moved_points)
        if not invalid_move:
            self.points = moved_points

        if action.turn_dir == "CW":
            self.points = [self.points[3], self.points[0], self.points[1], self.points[2]]
        else:
            self.points = [self.points[1], self.points[2], self.points[3], self.points[0]]

        return self.get_state(), self.render(), invalid_move
