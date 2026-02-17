import csv

from superblock.monitor import write_history_csv


def test_write_history_csv(tmp_path) -> None:
    path = tmp_path / "metrics.csv"
    history = [
        {
            "day_idx": 1.0,
            "invalid_move_ratio": 0.1,
            "pos_mse": 0.001,
            "exact_acc": 0.0,
            "near_acc": 0.2,
            "score": 0.4,
            "visible_hits": 33.0,
        }
    ]
    write_history_csv(str(path), history)

    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["day_idx"] == "1"
    assert rows[0]["visible_hits"] == "33.0"
