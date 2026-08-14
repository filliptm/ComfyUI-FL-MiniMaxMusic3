import tempfile
import unittest
from pathlib import Path
from unittest import mock

from conftest import pack_module


trainer_module = pack_module("nodes.trainer")
run_store = pack_module("training.run_store")


class FakeWorker:
    def __init__(self, returncode=0):
        self.returncode = returncode

    def poll(self):
        return self.returncode


class SequencedWorker(FakeWorker):
    def __init__(self):
        super().__init__(0)
        self.poll_results = iter((None, None, 0))

    def poll(self):
        return next(self.poll_results)


def dataset():
    return {
        "root": "dataset",
        "manifest_hash": "manifest",
        "tracks": [{"audio": "song.wav"}],
        "total_seconds": 10.0,
        "settings": {"caption_extension": ".txt", "lyrics_extension": ".lyrics", "duration_interval": 3.0, "max_duration": 60.0},
    }


def train_config():
    return {
        "schema_version": 1, "max_train_steps": 2, "lora_rank": 16, "lora_alpha": 16,
        "base_model_precision": "no_change", "text_encoder_1_precision": "no_change", "mixed_precision": "bf16",
    }


class TrainerTests(unittest.TestCase):
    def test_trainer_records_completion_and_returns_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            runs = Path(directory) / "runs"
            adapter = Path(directory) / "demo.safetensors"
            adapter.write_bytes(b"lora")
            with mock.patch.object(run_store, "runs_root", return_value=runs), mock.patch.object(run_store, "ensure_directories"), mock.patch.object(trainer_module, "require_environment", return_value={"python": "python", "simpletuner": "simpletuner"}), mock.patch.object(trainer_module, "write_simpletuner_config"), mock.patch.object(trainer_module, "launch_worker", return_value=FakeWorker()), mock.patch.object(trainer_module, "worker_state", return_value={"status": "completed"}), mock.patch.object(trainer_module, "export_adapter", return_value=adapter), mock.patch.object(trainer_module.comfy.model_management, "unload_all_models"), mock.patch.object(trainer_module.comfy.model_management, "soft_empty_cache"):
                handle, lora_path = trainer_module.FL_MiniMaxMusic3LoRATrainer().train(dataset(), train_config(), "demo", "require_installed")
            self.assertEqual(lora_path, str(adapter))
            self.assertEqual(handle["state"]["status"], "completed")
            self.assertEqual(handle["state"]["adapter_path"], str(adapter))

    def test_trainer_surfaces_worker_failure_and_keeps_run_state(self):
        with tempfile.TemporaryDirectory() as directory:
            runs = Path(directory) / "runs"
            with mock.patch.object(run_store, "runs_root", return_value=runs), mock.patch.object(run_store, "ensure_directories"), mock.patch.object(trainer_module, "require_environment", return_value={"python": "python", "simpletuner": "simpletuner"}), mock.patch.object(trainer_module, "write_simpletuner_config"), mock.patch.object(trainer_module, "launch_worker", return_value=FakeWorker(1)), mock.patch.object(trainer_module, "worker_state", return_value={"status": "failed"}), mock.patch.object(trainer_module.comfy.model_management, "unload_all_models"), mock.patch.object(trainer_module.comfy.model_management, "soft_empty_cache"):
                with self.assertRaisesRegex(RuntimeError, "training failed"):
                    trainer_module.FL_MiniMaxMusic3LoRATrainer().train(dataset(), train_config(), "demo", "require_installed")
            states = [run_store.read_json(path / "state.json") for path in runs.iterdir()]
            self.assertEqual(states[0]["status"], "failed")

    def test_trainer_only_persists_training_metrics(self):
        preprocessing = {
            "timestamp": "2026-08-14T00:00:00Z", "phase": "preprocessing", "message": "Caching 4399 samples",
            "current": 4399, "total": 4399, "metrics": {},
        }
        training = {
            "timestamp": "2026-08-14T00:00:01Z", "phase": "training", "message": "Step 1/2 loss=0.5",
            "current": 1, "total": 2, "metrics": {"loss": 0.5},
        }
        progress = mock.Mock()
        progress.poll.side_effect = (preprocessing, training, None)
        with tempfile.TemporaryDirectory() as directory:
            runs = Path(directory) / "runs"
            adapter = Path(directory) / "demo.safetensors"
            adapter.write_bytes(b"lora")
            with mock.patch.object(run_store, "runs_root", return_value=runs), mock.patch.object(run_store, "ensure_directories"), mock.patch.object(trainer_module, "require_environment", return_value={"python": "python", "simpletuner": "simpletuner"}), mock.patch.object(trainer_module, "write_simpletuner_config"), mock.patch.object(trainer_module, "launch_worker", return_value=SequencedWorker()), mock.patch.object(trainer_module, "worker_state", return_value={"status": "completed"}), mock.patch.object(trainer_module, "export_adapter", return_value=adapter), mock.patch.object(trainer_module, "LogProgress", return_value=progress), mock.patch.object(trainer_module, "append_metric") as append_metric, mock.patch.object(trainer_module.time, "sleep"), mock.patch.object(trainer_module.comfy.model_management, "unload_all_models"), mock.patch.object(trainer_module.comfy.model_management, "soft_empty_cache"):
                trainer_module.FL_MiniMaxMusic3LoRATrainer().train(dataset(), train_config(), "demo", "require_installed")
        append_metric.assert_called_once()
        self.assertEqual(append_metric.call_args.args[1]["step"], 1)
        self.assertEqual(append_metric.call_args.args[1]["loss"], 0.5)
