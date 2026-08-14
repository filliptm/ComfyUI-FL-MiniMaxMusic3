import json
import hashlib
import os
import shutil
import subprocess
import tarfile
import threading
import venv
from pathlib import Path

import requests

from .paths import backend_root, ensure_directories, load_backend_manifest, pack_root


_install_lock = threading.Lock()


def _python_path(environment):
    return environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _simpletuner_path(environment):
    return environment / ("Scripts/simpletuner.exe" if os.name == "nt" else "bin/simpletuner")


def environment_path():
    manifest = load_backend_manifest()
    return backend_root() / manifest["backend_id"]


def environment_status(refresh=False):
    ensure_directories()
    manifest = load_backend_manifest()
    environment = environment_path()
    marker_path = environment / "fl_backend.json"
    marker = None
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    installed = (
        _python_path(environment).is_file()
        and _simpletuner_path(environment).is_file()
        and marker == manifest
    )
    status = {
        "schema_version": 1,
        "backend_id": manifest["backend_id"],
        "path": str(environment),
        "python": str(_python_path(environment)),
        "simpletuner": str(_simpletuner_path(environment)),
        "installed": installed,
        "manifest": manifest,
    }
    if installed and refresh:
        command = [str(_python_path(environment)), "-c", "import simpletuner; from simpletuner.helpers.models.minimaxmusic.model import MiniMaxMusic; print('ok')"]
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
        status["verified"] = result.returncode == 0 and result.stdout.strip().endswith("ok")
        status["verification_error"] = result.stderr[-4000:] if not status["verified"] else None
    else:
        status["verified"] = installed
        status["verification_error"] = None
    return status


def _run_install(command, callback, cwd=None):
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=creationflags,
        cwd=cwd,
    )
    assert process.stdout is not None
    for line in process.stdout:
        if callback:
            callback(line.rstrip())
    returncode = process.wait()
    if returncode:
        raise RuntimeError(f"Training backend installation failed with exit code {returncode}")


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_checked(url, target, expected_sha256, callback):
    target = Path(target)
    if target.is_file() and _file_sha256(target) == expected_sha256:
        return
    partial = target.with_suffix(target.suffix + ".partial")
    try:
        with requests.get(url, stream=True, timeout=(30, 300)) as response:
            response.raise_for_status()
            with partial.open("wb") as file:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        file.write(chunk)
    except requests.RequestException as error:
        raise RuntimeError(f"Could not download the pinned TrainingSample source: {error}") from error
    if _file_sha256(partial) != expected_sha256:
        partial.unlink(missing_ok=True)
        raise RuntimeError("The Windows TrainingSample source archive failed SHA-256 verification")
    partial.replace(target)
    if callback:
        callback("Verified TrainingSample source archive")


def _disable_trainingsample_opencv(pyproject):
    pyproject = Path(pyproject)
    text = pyproject.read_text(encoding="utf-8")
    original = 'features = ["pyo3/extension-module", "python-bindings", "opencv", "simd"]'
    replacement = 'features = ["pyo3/extension-module", "python-bindings", "simd"]'
    if original in text:
        pyproject.write_text(text.replace(original, replacement, 1), encoding="utf-8")
    elif replacement not in text:
        raise RuntimeError("The pinned TrainingSample build features changed; refusing to modify an unknown source layout")


def _install_windows_trainingsample(environment, manifest, callback):
    package = manifest["windows_trainingsample"]
    sources = environment / "sources"
    sources.mkdir(parents=True, exist_ok=True)
    archive = sources / f"trainingsample-{package['version']}.tar.gz"
    source = sources / f"trainingsample-{package['version']}"
    wheels = sources / "wheels"
    wheel = next(wheels.glob(f"trainingsample-{package['version']}-cp*-win_amd64.whl"), None) if wheels.is_dir() else None
    if wheel is None:
        if shutil.which("cargo") is None:
            raise RuntimeError("Rust is required once to build the pinned Windows audio-only TrainingSample wheel. Install Rust from rustup.rs and retry.")
        if callback:
            callback("Preparing the pinned Windows TrainingSample compatibility build")
        _download_checked(package["source_url"], archive, package["source_sha256"], callback)
        if not (source / "pyproject.toml").is_file():
            with tarfile.open(archive, "r:gz") as file:
                file.extractall(sources, filter="data")
        _disable_trainingsample_opencv(source / "pyproject.toml")
        _run_install([str(_python_path(environment)), "-m", "pip", "install", package["maturin"]], callback)
        wheels.mkdir(exist_ok=True)
        _run_install([
            str(_python_path(environment)),
            "-m",
            "maturin",
            "build",
            "--release",
            "--interpreter",
            str(_python_path(environment)),
            "--out",
            str(wheels),
        ], callback, cwd=source)
        wheel = next(wheels.glob(f"trainingsample-{package['version']}-cp*-win_amd64.whl"), None)
        if wheel is None:
            raise RuntimeError("The Windows TrainingSample compatibility build did not produce a wheel")
    _run_install([str(_python_path(environment)), "-m", "pip", "install", str(wheel)], callback)


def _install_windows_fcntl(environment):
    target = environment / "Lib" / "site-packages" / "fcntl.py"
    shutil.copy2(pack_root() / "training" / "compat" / "fcntl.py", target)


def _install_commands(environment, manifest):
    python = str(_python_path(environment))
    source = f"simpletuner @ git+{manifest['simpletuner_source']}@{manifest['simpletuner_commit']}"
    if os.name == "nt":
        return [
            [python, "-m", "pip", "install", *manifest["windows_torch_packages"], "--extra-index-url", manifest["extra_index_url"]],
            [python, "-m", "pip", "install", source],
            [python, "-m", "pip", "install", *manifest["windows_runtime_packages"]],
        ]
    source = f"simpletuner[{manifest['extra']}] @ git+{manifest['simpletuner_source']}@{manifest['simpletuner_commit']}"
    return [[python, "-m", "pip", "install", source, "--extra-index-url", manifest["extra_index_url"]]]


def install_environment(callback=None):
    with _install_lock:
        current = environment_status(refresh=True)
        if current["verified"]:
            return current
        ensure_directories()
        environment = environment_path()
        free = shutil.disk_usage(backend_root()).free
        if free < 20 * 1024 ** 3:
            raise RuntimeError("At least 20 GiB of free disk space is required to install the Music 3 training backend")
        if callback:
            callback("Creating isolated Python environment")
        if not _python_path(environment).is_file():
            venv.EnvBuilder(with_pip=True, clear=False).create(environment)
        manifest = load_backend_manifest()
        commands = _install_commands(environment, manifest)
        if os.name == "nt":
            _run_install(commands[0], callback)
            _install_windows_trainingsample(environment, manifest, callback)
            _run_install(commands[1], callback)
            _run_install(commands[2], callback)
            _install_windows_fcntl(environment)
        else:
            _run_install(commands[0], callback)
        (environment / "fl_backend.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        status = environment_status(refresh=True)
        if not status["verified"]:
            raise RuntimeError(status["verification_error"] or "The Music 3 training backend did not verify after installation")
        return status


def require_environment(policy, callback=None):
    status = environment_status(refresh=True)
    if status["verified"]:
        return status
    if policy == "install_pinned_if_missing":
        return install_environment(callback)
    raise RuntimeError(
        "The pinned MiniMax Music 3 training backend is not installed. "
        "Set backend_policy to install_pinned_if_missing for one explicitly authorized queued run."
    )
