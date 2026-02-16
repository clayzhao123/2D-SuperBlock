from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Transition:
    state_t: list[int]
    action_vec: list[float]
    invalid_move: bool
    img_t: list[list[float]]
    state_tp1: list[int]
    img_tp1: list[list[float]]


class ReplayBuffer:
    def __init__(self) -> None:
        self.items: list[Transition] = []

    def add(self, t: Transition) -> None:
        self.items.append(t)

    def __len__(self) -> int:
        return len(self.items)

    def as_arrays(self) -> tuple[list[list[float]], list[list[float]]]:
        x = []
        y = []
        for item in self.items:
            norm_state_t = [v / 40.0 for v in item.state_t]
            x.append(norm_state_t + list(item.action_vec))
            y.append([v / 40.0 for v in item.state_tp1])
        return x, y
