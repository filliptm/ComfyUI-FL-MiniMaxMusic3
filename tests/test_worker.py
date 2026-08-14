import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from conftest import pack_module


worker_module = pack_module("training.worker")


FAKE_TRAINER = '''
import pathlib
import sys
import time

print("Building VAE cache for audio dataset", flush=True)
print("Steps: 50%| 1/2 loss=0.5 lr=5e-05", flush=True)
if "slow" in pathlib.Path(__file__).stem:
    time.sleep(30)
output = pathlib.Path.cwd() / "backend_output"
output.mkdir(exist_ok=True)
(output / "checkpoint-1").mkdir(exist_ok=True)
(output / "pytorch_lora_weights.safetensors").write_bytes(b"fake-lora")
print("Steps: 100%| 2/2 loss=0.25 lr=5e-05", flush=True)
'''


def worker_command(run_dir, trainer):
    return [
        sys.executable,
        str(Path(worker_module.__file__)),
        "--run-dir",
        str(run_dir),
        "--simpletuner",
        str(trainer),
        "--model-cache",
        str(run_dir / "models"),
    ]


class WorkerTests(unittest.TestCase):
    def test_worker_state_write_retries_windows_replace_collision(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "worker.json"
            original_replace = worker_module.os.replace
            original_name = worker_module.os.name
            attempts = []

            def replace(source, destination):
                attempts.append((source, destination))
                if len(attempts) < 3:
                    raise PermissionError("busy")
                return original_replace(source, destination)

            try:
                worker_module.os.replace = replace
                worker_module.os.name = "nt"
                worker_module._write(target, {"status": "running"})
            finally:
                worker_module.os.replace = original_replace
                worker_module.os.name = original_name
            self.assertEqual(len(attempts), 3)
            self.assertEqual(json.loads(target.read_text(encoding="utf-8")), {"status": "running"})

    def test_managed_environment_precedes_comfy_scripts(self):
        environment = worker_module._training_environment(
            Path("backend") / "Scripts" / "simpletuner.exe",
            Path("models"),
        )
        path_key = next(key for key in environment if key.upper() == "PATH")
        self.assertEqual(Path(environment[path_key].split(__import__("os").pathsep)[0]), Path("backend") / "Scripts")
        self.assertEqual(Path(environment["VIRTUAL_ENV"]), Path("backend"))
        self.assertEqual(environment["PYTHONNOUSERSITE"], "1")

    def test_managed_command_uses_the_active_python(self):
        original = worker_module.importlib.util.find_spec
        try:
            worker_module.importlib.util.find_spec = lambda name: type(
                "Spec", (), {"submodule_search_locations": [str(Path("managed") / "simpletuner")]}
            )()
            command = worker_module._trainer_command(Path("backend") / "Scripts" / "simpletuner.exe")
        finally:
            worker_module.importlib.util.find_spec = original
        self.assertEqual(command[:4], [sys.executable, "-m", "accelerate.commands.accelerate_cli", "launch"])
        self.assertEqual(Path(command[-2]).name, "simpletuner_entry.py")
        self.assertEqual(command[-1], str(Path("managed") / "simpletuner" / "train.py"))

    def test_python_trainer_remains_supported(self):
        trainer = Path("fake_trainer.py")
        self.assertEqual(worker_module._trainer_command(trainer), [sys.executable, str(trainer)])

    def test_worker_runs_trainer_and_records_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "config/job").mkdir(parents=True)
            trainer = run_dir / "fake_trainer.py"
            trainer.write_text(FAKE_TRAINER, encoding="utf-8")
            result = subprocess.run(worker_command(run_dir, trainer), timeout=30)
            state = json.loads((run_dir / "worker.json").read_text(encoding="utf-8"))
            self.assertEqual(result.returncode, 0)
            self.assertEqual(state["status"], "completed")
            self.assertTrue((run_dir / "backend_output/pytorch_lora_weights.safetensors").is_file())
            self.assertIn("loss=0.25", (run_dir / "logs/trainer.log").read_text(encoding="utf-8"))

    def test_worker_honors_stop_request(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "config/job").mkdir(parents=True)
            trainer = run_dir / "slow_trainer.py"
            trainer.write_text(FAKE_TRAINER, encoding="utf-8")
            process = subprocess.Popen(worker_command(run_dir, trainer))
            deadline = time.monotonic() + 10
            while not (run_dir / "worker.json").is_file() and time.monotonic() < deadline:
                time.sleep(0.05)
            (run_dir / "stop.request").write_text("stop\n", encoding="utf-8")
            process.wait(timeout=30)
            state = json.loads((run_dir / "worker.json").read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "interrupted")
