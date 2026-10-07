"""Trusted local checkpoints, written atomically without changing their format."""

from __future__ import annotations

import os
import pickle
import tempfile
from pathlib import Path


def save_payload_checkpoint(path: str, payload: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            pickle.dump(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_payload_checkpoint(path: str) -> dict:
    """Load a checkpoint you trust; pickle must not be used for untrusted input."""
    with open(path, "rb") as stream:
        return pickle.load(stream)
