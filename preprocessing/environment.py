import json
import os
import shutil
import subprocess
import sys
import threading
import venv
from pathlib import Path

from ..training.paths import moss_backend_root, pack_root


_install_lock = threading.Lock()


def load_backend_manifest():
    return json.loads((pack_root() / "preprocessing" / "backend_manifest.json").read_text(encoding="utf-8"))


def environment_path():
    manifest = load_backend_manifest()
    return moss_backend_root() / manifest["backend_id"]


def _python_path(environment):
    return environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _marker_path(environment):
    return environment / "fl_moss_backend.json"


def _manifest_matches(environment, manifest):
    try:
        return json.loads(_marker_path(environment).read_text(encoding="utf-8")) == manifest
    except (FileNotFoundError, json.JSONDecodeError):
        return False


def _verify(environment, manifest):
    python = _python_path(environment)
    if not python.is_file() or not _manifest_matches(environment, manifest):
        return False, "The pinned environment is not installed"
    code = (
        "import importlib.metadata as m, torch, torchaudio, transformers; "
        "assert torch.__version__ == '" + manifest["torch"] + "'; "
        "assert torchaudio.__version__ == '" + manifest["torchaudio"] + "'; "
        "assert transformers.__version__ == '" + manifest["transformers"] + "'; "
        "assert torch.cuda.is_available(); "
        "print(torch.cuda.get_device_name(0))"
    )
    result = subprocess.run([str(python), "-I", "-c", code], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=90)
    return result.returncode == 0, (result.stdout.strip() if result.returncode == 0 else result.stderr.strip())


def environment_status(refresh=False):
    manifest = load_backend_manifest()
    environment = environment_path()
    if refresh:
        verified, message = _verify(environment, manifest)
    elif _manifest_matches(environment, manifest) and _python_path(environment).is_file():
        verified, message = True, "Pinned environment installed"
    else:
        verified, message = False, "The pinned environment is not installed"
    return {
        "backend_id": manifest["backend_id"],
        "path": str(environment),
        "python": str(_python_path(environment)),
        "verified": verified,
        "message": message,
    }


def _run(command, callback):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    output = []
    for line in process.stdout:
        line = line.rstrip()
        output.append(line)
        if callback and line:
            callback(line)
    returncode = process.wait()
    if returncode:
        raise RuntimeError("MOSS backend installation failed: " + "\n".join(output[-20:]))


def install_environment(callback=None):
    with _install_lock:
        status = environment_status(refresh=True)
        if status["verified"]:
            return status
        manifest = load_backend_manifest()
        environment = environment_path()
        environment.parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(environment.parent).free < 20 * 1024 ** 3:
            raise RuntimeError("At least 20 GiB of free disk space is required for the isolated MOSS backend")
        if sys.version_info[:2] != (3, 12):
            raise RuntimeError("The managed MOSS backend requires ComfyUI to run with Python 3.12")
        if callback:
            callback("Creating the isolated MOSS environment")
        if not _python_path(environment).is_file():
            venv.EnvBuilder(with_pip=True, clear=False).create(environment)
        python = str(_python_path(environment))
        _run([python, "-I", "-m", "pip", "install", "--upgrade", "pip", "wheel"], callback)
        _run([
            python, "-I", "-m", "pip", "install",
            f"torch=={manifest['torch']}", f"torchaudio=={manifest['torchaudio']}",
            "--index-url", manifest["torch_index_url"],
        ], callback)
        packages = [
            f"transformers=={manifest['transformers']}",
            f"accelerate=={manifest['accelerate']}",
            f"safetensors=={manifest['safetensors']}",
            f"numpy=={manifest['numpy']}",
            f"soundfile=={manifest['soundfile']}",
            f"tiktoken=={manifest['tiktoken']}",
            f"einops=={manifest['einops']}",
            f"scipy=={manifest['scipy']}",
            f"tqdm=={manifest['tqdm']}",
            f"packaging=={manifest['packaging']}",
        ]
        _run([python, "-I", "-m", "pip", "install", *packages], callback)
        _marker_path(environment).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        status = environment_status(refresh=True)
        if not status["verified"]:
            raise RuntimeError(status["message"] or "The MOSS backend did not verify after installation")
        return status


def require_environment(policy, callback=None):
    status = environment_status(refresh=True)
    if status["verified"]:
        return status
    if policy == "install_if_missing":
        return install_environment(callback)
    raise RuntimeError("The pinned MOSS backend is not installed. Choose install_if_missing or use Install in the preprocessor dashboard.")
