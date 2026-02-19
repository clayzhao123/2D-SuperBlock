from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field

from .env import Action, MOVE_DELTAS, SuperblockEnv, TURN_DIRS
from .models import ForwardModel
from .monitor import load_checkpoint
from .utils import position_key


_COORD_SCALE = 40.0


def state_to_center_cell(state: list[int]) -> tuple[int, int]:
    return position_key(state)


def action_to_onehot(action: Action) -> list[float]:
    vec = [0.0] * 10
    vec[action.leg_id] = 1.0
    move_idx = {"+x": 0, "-x": 1, "+y": 2, "-y": 3}[action.move_dir]
    vec[4 + move_idx] = 1.0
    turn_idx = {"CW": 0, "CCW": 1}[action.turn_dir]
    vec[8 + turn_idx] = 1.0
    return vec


def load_forward_model_from_ckpt(path: str) -> ForwardModel:
    payload = load_checkpoint(path)
    model = ForwardModel()
    model.w1 = payload["model"]["w1"]
    model.b1 = payload["model"]["b1"]
    model.w2 = payload["model"]["w2"]
    model.b2 = payload["model"]["b2"]
    return model


class CuriosityMemory:
    def __init__(self) -> None:
        self.visits: dict[tuple[int, int], int] = {}

    def update(self, points_or_state: list[tuple[int, int]] | list[int]) -> None:
        if points_or_state and isinstance(points_or_state[0], int):
            cell = state_to_center_cell(points_or_state)  # type: ignore[arg-type]
        else:
            cell = position_key(points_or_state)  # type: ignore[arg-type]
        self.visits[cell] = self.visits.get(cell, 0) + 1

    def count(self, center_cell: tuple[int, int]) -> int:
        return self.visits.get(center_cell, 0)


@dataclass
class CuriosityPolicy:
    forward_model: ForwardModel
    memory: CuriosityMemory
    epsilon: float = 0.05
    invalid_penalty: float = 1e6
    loop_penalty: float = 0.35
    recent_path: deque[tuple[int, int]] = field(default_factory=lambda: deque(maxlen=10))

    def candidate_actions(self) -> list[Action]:
        return [Action(leg_id, move_dir, turn_dir) for leg_id in range(4) for move_dir in MOVE_DELTAS for turn_dir in TURN_DIRS]

    def _predict_next_state(self, state_t: list[int], action: Action) -> list[int]:
        model_in = [v / _COORD_SCALE for v in state_t] + action_to_onehot(action)
        pred = self.forward_model.predict_batch([model_in])[0]
        return [round(v * _COORD_SCALE) for v in pred]

    def _is_invalid(self, env: SuperblockEnv, action: Action) -> bool:
        _, invalid = env.peek_step(env.points, action)
        return invalid

    def select_action(self, state_t: list[int], env: SuperblockEnv, rng: random.Random) -> Action:
        actions = self.candidate_actions()
        if rng.random() < self.epsilon:
            return rng.choice(actions)

        if not self.recent_path:
            self.recent_path.append(state_to_center_cell(state_t))

        best_score = float("-inf")
        best_actions: list[Action] = []
        for action in actions:
            pred_state = self._predict_next_state(state_t, action)
            pred_center = state_to_center_cell(pred_state)
            novelty = 1.0 / (1.0 + self.memory.count(pred_center))
            invalid = self._is_invalid(env, action)
            loop_hit = pred_center in self.recent_path
            score = novelty - (self.invalid_penalty if invalid else 0.0) - (self.loop_penalty if loop_hit else 0.0)
            if score > best_score + 1e-12:
                best_score = score
                best_actions = [action]
            elif abs(score - best_score) <= 1e-12:
                best_actions.append(action)
        chosen = rng.choice(best_actions or actions)
        next_points, _ = env.peek_step(env.points, chosen)
        self.recent_path.append(position_key(next_points))
        return chosen
