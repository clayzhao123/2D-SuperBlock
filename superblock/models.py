from __future__ import annotations

import random


def _relu(x: float) -> float:
    return x if x > 0 else 0.0


def _relu_grad(x: float) -> float:
    return 1.0 if x > 0 else 0.0


class ForwardModel:
    """Tiny pure-Python MLP for CPU-only no-dependency environments."""

    def __init__(self, input_dim: int = 18, hidden_dim: int = 64, output_dim: int = 8, seed: int = 42) -> None:
        rng = random.Random(seed)
        self.w1 = [[rng.uniform(-0.1, 0.1) for _ in range(input_dim)] for _ in range(hidden_dim)]
        self.b1 = [0.0 for _ in range(hidden_dim)]
        self.w2 = [[rng.uniform(-0.1, 0.1) for _ in range(hidden_dim)] for _ in range(output_dim)]
        self.b2 = [0.0 for _ in range(output_dim)]

    def predict_batch(self, x_batch: list[list[float]]) -> list[list[float]]:
        outputs = []
        for x in x_batch:
            z1 = [sum(w * xi for w, xi in zip(w_row, x)) + b for w_row, b in zip(self.w1, self.b1)]
            h = [_relu(v) for v in z1]
            y = [sum(w * hi for w, hi in zip(w_row, h)) + b for w_row, b in zip(self.w2, self.b2)]
            outputs.append(y)
        return outputs

    def train_batch(self, x_batch: list[list[float]], y_batch: list[list[float]], lr: float) -> None:
        n = len(x_batch)
        if n == 0:
            return

        grad_w1 = [[0.0 for _ in row] for row in self.w1]
        grad_b1 = [0.0 for _ in self.b1]
        grad_w2 = [[0.0 for _ in row] for row in self.w2]
        grad_b2 = [0.0 for _ in self.b2]

        for x, y_true in zip(x_batch, y_batch):
            z1 = [sum(w * xi for w, xi in zip(w_row, x)) + b for w_row, b in zip(self.w1, self.b1)]
            h = [_relu(v) for v in z1]
            y_pred = [sum(w * hi for w, hi in zip(w_row, h)) + b for w_row, b in zip(self.w2, self.b2)]

            dy = [2.0 * (yp - yt) / len(y_true) for yp, yt in zip(y_pred, y_true)]

            for o in range(len(self.w2)):
                for j in range(len(self.w2[o])):
                    grad_w2[o][j] += dy[o] * h[j]
                grad_b2[o] += dy[o]

            dh = [0.0 for _ in h]
            for j in range(len(h)):
                dh[j] = sum(self.w2[o][j] * dy[o] for o in range(len(self.w2)))

            dz1 = [dh_j * _relu_grad(z) for dh_j, z in zip(dh, z1)]
            for j in range(len(self.w1)):
                for i in range(len(self.w1[j])):
                    grad_w1[j][i] += dz1[j] * x[i]
                grad_b1[j] += dz1[j]

        scale = lr / n
        for j in range(len(self.w1)):
            for i in range(len(self.w1[j])):
                self.w1[j][i] -= scale * grad_w1[j][i]
            self.b1[j] -= scale * grad_b1[j]

        for o in range(len(self.w2)):
            for j in range(len(self.w2[o])):
                self.w2[o][j] -= scale * grad_w2[o][j]
            self.b2[o] -= scale * grad_b2[o]
