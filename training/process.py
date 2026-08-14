import json
import os
import subprocess
from pathlib import Path

from .paths import model_cache_root, pack_root


def launch_worker(run_dir, backend):
    run_dir = Path(run_dir)
    stop_path = run_dir / "stop.request"
    if stop_path.exists():
        stop_path.unlink()
    command = [
        backend["python"],
        str(pack_root() / "training" / "worker.py"),
        "--run-dir",
        str(run_dir),
        "--simpletuner",
        backend["simpletuner"],
        "--model-cache",
        str(model_cache_root()),
    ]
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / "worker.log").open("a", encoding="utf-8", buffering=1) as log:
        return subprocess.Popen(
            command,
            cwd=run_dir,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
        )


def worker_state(run_dir):
    try:
        return json.loads((Path(run_dir) / "worker.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def request_stop(run_dir):
    (Path(run_dir) / "stop.request").write_text("stop\n", encoding="utf-8")
