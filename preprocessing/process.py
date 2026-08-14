import os
import subprocess
from pathlib import Path

from ..training.paths import pack_root


def launch_worker(run_dir, backend, model_path):
    run_dir = Path(run_dir)
    stop_path = run_dir / "stop.request"
    stop_path.unlink(missing_ok=True)
    command = [
        backend["python"],
        str(pack_root() / "preprocessing" / "moss_worker.py"),
        "--run-dir", str(run_dir),
        "--model-path", str(model_path),
    ]
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log = (log_dir / "worker.log").open("a", encoding="utf-8", buffering=1)
    process = subprocess.Popen(
        command,
        cwd=run_dir,
        stdout=log,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
    )
    process._fl_log = log
    return process


def close_worker(process):
    log = getattr(process, "_fl_log", None)
    if log is not None:
        log.close()
