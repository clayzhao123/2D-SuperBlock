from __future__ import annotations

import argparse
import random
from pathlib import Path

from .env import SuperblockEnv
from .policy_curiosity import CuriosityMemory, CuriosityPolicy, load_forward_model_from_ckpt
from .utils import GRID_CELLS


def run(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    env = SuperblockEnv()
    model = load_forward_model_from_ckpt(args.checkpoint_path)
    memory = CuriosityMemory()
    policy = CuriosityPolicy(forward_model=model, memory=memory)

    state, _ = env.reset()
    memory.update(state)
    for _ in range(args.steps):
        action = policy.select_action(state, env, rng)
        state, _, _ = env.step(action)
        memory.update(state)

    visited = len(memory.visits)
    total = GRID_CELLS * GRID_CELLS
    coverage = visited / float(total)
    msg = f"visited_cells={visited} total_cells={total} coverage={coverage:.4f}"
    print(msg)

    if args.report_path:
        Path(args.report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_path).write_text(msg + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Curiosity-driven exploration with trained motion model")
    parser.add_argument("--checkpoint-path", type=str, default="artifacts/train.ckpt")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-path", type=str, default="artifacts/curiosity_report.txt")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
