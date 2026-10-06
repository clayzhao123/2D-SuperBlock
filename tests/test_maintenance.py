"""Regression checks for archived runs and behavior-preserving module extraction."""

import json
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

from superblock import train
from superblock.agents.qlearn import QLearnForagePolicy
from superblock.buffer import ReplayBuffer
from superblock.checkpoint import load_payload_checkpoint, save_payload_checkpoint
from superblock.demo import run_demo
from superblock.env import Action
from superblock.experiment import run_experiment
from superblock.forage_train import QLearnForagePolicy as LegacyQLearnForagePolicy, build_parser, run
from superblock.master_dashboard import _dashboard_link
from superblock.models import ForwardModel
from superblock.monitor import save_checkpoint


class MaintenanceTests(unittest.TestCase):
    def test_existing_run_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "same"
            path.mkdir()
            marker = path / "keep.txt"
            marker.write_text("original")
            with self.assertRaises(FileExistsError):
                run_experiment("motion", [], runs_dir=directory, run_id="same")
            self.assertEqual(marker.read_text(), "original")
            self.assertEqual(list(path.iterdir()), [marker])

    def test_failed_training_is_recorded_and_propagated(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(train, "run", side_effect=RuntimeError("training failed")):
            with self.assertRaisesRegex(RuntimeError, "training failed"):
                run_experiment("motion", [], runs_dir=directory, run_id="failed")
            state = json.loads((Path(directory) / "failed/run.json").read_text())
            self.assertEqual(state["status"], "failed")
            self.assertEqual(state["error_type"], "RuntimeError")
            self.assertIn("finished_at", state)

    def test_interruption_is_recorded_and_propagated(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(train, "run", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                run_experiment("motion", [], runs_dir=directory, run_id="interrupted")
            state = json.loads((Path(directory) / "interrupted/run.json").read_text())
            self.assertEqual(state["status"], "interrupted")

    def test_missing_input_checkpoint_leaves_a_failed_run(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                run_experiment("forage", ["--motion-checkpoint-path", str(Path(directory) / "missing.ckpt")], runs_dir=directory, run_id="missing")
            state = json.loads((Path(directory) / "missing/run.json").read_text())
            self.assertEqual(state["status"], "failed")

    def test_managed_paths_and_unsafe_names_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for options in (["--checkpoint-path=elsewhere.ckpt"], ["--resume"], ["--steps-per-day", "0"]):
                with self.assertRaises(ValueError):
                    run_experiment("motion", options, runs_dir=directory)
            with self.assertRaises(ValueError):
                run_experiment("motion", [], runs_dir=directory, run_id="../outside")
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_complete_demo_is_isolated_and_records_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = run_demo(directory)
            self.assertEqual(len(set(paths)), 3)
            for path in paths:
                state = json.loads((path / "run.json").read_text())
                config = json.loads((path / "config.json").read_text())
                self.assertEqual(state["status"], "completed")
                self.assertEqual(config["parameters"]["seed"], 42)
                self.assertTrue((path / "master_dashboard.html").is_file())
                for name, value in config["parameters"].items():
                    if name in {"checkpoint_path", "out_checkpoint", "dashboard_path", "out_dashboard", "master_dashboard_path"}:
                        self.assertEqual(Path(value).parent, path)
                if config["stage"] != "motion":
                    self.assertEqual(len(config["input_checkpoint"]["sha256"]), 64)
                    self.assertEqual((path / "input_motion.ckpt").read_bytes(), (paths[0] / "train.ckpt").read_bytes())
            forage = load_payload_checkpoint(str(paths[1] / "forage.ckpt"))
            self.assertEqual(forage["policy"], "qlearn")
            self.assertIsInstance(forage["q_table"], dict)

    def test_failed_checkpoint_write_keeps_previous_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.ckpt"
            save_payload_checkpoint(str(path), {"day_idx": 1})
            original = path.read_bytes()
            with patch("superblock.checkpoint.pickle.dump", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    save_payload_checkpoint(str(path), {"day_idx": 2})
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_summary_links_use_relative_paths_and_handle_missing_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chart = root / "chart with space.html"
            chart.write_text("chart")
            self.assertIn('href="chart%20with%20space.html"', _dashboard_link(str(chart), "Chart", str(root / "master.html")))
            self.assertEqual(_dashboard_link(str(root / "missing.html"), "Forage", str(root / "master.html")), "<span>Forage（未生成）</span>")

    def test_qlearning_legacy_import_and_terminal_update_stay_valid(self):
        self.assertIs(LegacyQLearnForagePolicy, QLearnForagePolicy)
        policy = QLearnForagePolicy(actions=[Action(0, "+x", "CW")], alpha=0.5, gamma=0.9, epsilon=0, epsilon_min=0, epsilon_decay=1, cell_div=4)
        state, next_state = (0, 0, 1, 1, 0, 3), (1, 0, 0, 0, 0, 4)
        policy.q_table[next_state] = [100]
        policy.update(state, 0, 2, next_state, done=True)
        self.assertEqual(policy.q_table[state], [1.0])

    def test_imitation_fallback_is_visible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "motion.ckpt"
            save_checkpoint(str(source), day_idx=0, model=ForwardModel(), buffer=ReplayBuffer(), history=[], visible_cells=[(19, 19)])
            args = build_parser().parse_args(["--policy", "imitation", "--days", "1", "--steps-per-day", "4", "--motion-checkpoint-path", str(source), "--out-checkpoint", str(root / "forage.ckpt"), "--out-metrics-csv", str(root / "metrics.csv"), "--out-dashboard", str(root / "forage.html"), "--motion-output-checkpoint", str(root / "tuned.ckpt"), "--master-dashboard-path", str(root / "master.html")])
            with warnings.catch_warnings(record=True) as seen:
                warnings.simplefilter("always")
                run(args)
            self.assertTrue(any("imitation is not implemented" in str(item.message) for item in seen))
            self.assertEqual(load_payload_checkpoint(str(root / "forage.ckpt"))["policy"], "heuristic")
