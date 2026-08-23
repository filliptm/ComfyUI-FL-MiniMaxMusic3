import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from conftest import pack_module


environment = pack_module("training.environment")
preprocessing_environment = pack_module("preprocessing.environment")


class EnvironmentTests(unittest.TestCase):
    def test_status_modules_import_without_venv(self):
        original_import = __import__

        def reject_venv(name, *args, **kwargs):
            if name == "venv":
                raise ModuleNotFoundError("No module named 'venv'")
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=reject_venv):
            importlib.reload(environment)
            importlib.reload(preprocessing_environment)

    def test_status_requires_exact_manifest_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {"backend_id": "backend", "simpletuner_commit": "abc"}
            backend = root / "backend"
            (backend / "Scripts").mkdir(parents=True)
            (backend / "Scripts/python.exe").touch()
            (backend / "Scripts/simpletuner.exe").touch()
            (backend / "fl_backend.json").write_text(json.dumps(manifest), encoding="utf-8")
            with mock.patch.object(environment, "backend_root", return_value=root), mock.patch.object(environment, "ensure_directories"), mock.patch.object(environment, "load_backend_manifest", return_value=manifest):
                self.assertTrue(environment.environment_status()["installed"])
                (backend / "fl_backend.json").write_text(json.dumps({**manifest, "simpletuner_commit": "wrong"}), encoding="utf-8")
                self.assertFalse(environment.environment_status()["installed"])

    def test_require_environment_does_not_install_by_default(self):
        with mock.patch.object(environment, "environment_status", return_value={"verified": False}), self.assertRaisesRegex(RuntimeError, "not installed"):
            environment.require_environment("require_installed")

    def test_windows_install_uses_pinned_cuda_torch_without_linux_triton_extra(self):
        manifest = {
            "simpletuner_source": "https://example.test/SimpleTuner.git",
            "simpletuner_commit": "abc",
            "extra": "cuda13",
            "extra_index_url": "https://download.pytorch.org/whl/cu130",
            "windows_torch_packages": ["torch==2.11.0+cu130", "torchvision==0.26.0+cu130"],
            "windows_runtime_packages": ["windows-curses==2.4.2"],
        }
        with mock.patch.object(environment.os, "name", "nt"):
            commands = environment._install_commands(Path("backend"), manifest)
        self.assertIn("torch==2.11.0+cu130", commands[0])
        self.assertNotIn("simpletuner[cuda13]", " ".join(commands[1]))
        self.assertIn("windows-curses==2.4.2", commands[2])

    def test_windows_trainingsample_build_removes_only_opencv_feature(self):
        with tempfile.TemporaryDirectory() as directory:
            pyproject = Path(directory) / "pyproject.toml"
            pyproject.write_text(
                '[tool.maturin]\nfeatures = ["pyo3/extension-module", "python-bindings", "opencv", "simd"]\n',
                encoding="utf-8",
            )
            environment._disable_trainingsample_opencv(pyproject)
            result = pyproject.read_text(encoding="utf-8")
            self.assertNotIn('"opencv"', result)
            self.assertIn('"python-bindings"', result)
