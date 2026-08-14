import tempfile
import unittest
from pathlib import Path
from unittest import mock

from conftest import pack_module


run_store = pack_module("training.run_store")


def dataset():
    return {"root": "dataset", "manifest_hash": "abc", "tracks": [{"audio": "a.wav"}], "total_seconds": 10.0, "settings": {}}


def config(rank=16):
    return {"max_train_steps": 10, "lora_rank": rank, "lora_alpha": rank, "base_model_precision": "no_change", "text_encoder_1_precision": "no_change", "mixed_precision": "bf16"}


class RunStoreTests(unittest.TestCase):
    def test_create_update_and_list_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(run_store, "runs_root", return_value=root), mock.patch.object(run_store, "ensure_directories"):
                run_dir, spec = run_store.create_run(dataset(), config(), None, "demo")
                state = run_store.update_state(run_dir, status="running", phase="training", current=2)
                listed = run_store.list_runs()
            self.assertEqual(spec["output_name"], "demo")
            self.assertEqual(state["sequence"], 1)
            self.assertEqual(listed[0]["current"], 2)

    def test_resume_rejects_changed_rank_or_dataset(self):
        spec = {"dataset": {"manifest_hash": "abc"}, "train_config": config(), "output_name": "demo"}
        with self.assertRaisesRegex(ValueError, "dataset manifest"):
            run_store.validate_resume(spec, {**dataset(), "manifest_hash": "different"}, config(), "demo")
        with self.assertRaisesRegex(ValueError, "lora_rank"):
            run_store.validate_resume(spec, dataset(), config(rank=32), "demo")

    def test_recent_metrics_skips_partial_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "metrics.jsonl").write_text('{"step": 1, "loss": 0.5}\nnot-json\n', encoding="utf-8")
            self.assertEqual(run_store.recent_metrics(root), [{"step": 1, "loss": 0.5}])
