import argparse
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write(path, payload):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for attempt in range(20):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if os.name != "nt" or attempt == 19:
                raise
            time.sleep(0.05)


def _stop_process(process):
    if process.poll() is not None:
        return
    if os.name == "nt":
        try:
            process.send_signal(signal.CTRL_BREAK_EVENT)
            process.wait(timeout=15)
            return
        except (OSError, subprocess.TimeoutExpired):
            pass
    else:
        process.terminate()
        try:
            process.wait(timeout=15)
            return
        except subprocess.TimeoutExpired:
            pass
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()


def _training_environment(simpletuner, model_cache):
    environment = os.environ.copy()
    environment.update({
        "HF_HOME": str(model_cache),
        "HF_HUB_CACHE": str(Path(model_cache) / "hub"),
        "SIMPLETUNER_LOG_LEVEL": "INFO",
        "PYTHONUTF8": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONNOUSERSITE": "1",
    })
    environment.pop("PYTHONPATH", None)
    simpletuner = Path(simpletuner)
    if simpletuner.suffix.lower() != ".py":
        scripts = simpletuner.parent
        path_key = next((key for key in environment if key.upper() == "PATH"), "PATH")
        environment[path_key] = str(scripts) + os.pathsep + environment.get(path_key, "")
        environment["VIRTUAL_ENV"] = str(scripts.parent)
    return environment


def _trainer_command(simpletuner):
    simpletuner = Path(simpletuner)
    if simpletuner.suffix.lower() == ".py":
        return [sys.executable, str(simpletuner)]
    spec = importlib.util.find_spec("simpletuner")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("SimpleTuner is not installed in the training environment")
    train_path = Path(next(iter(spec.submodule_search_locations))) / "train.py"
    entry_path = Path(__file__).with_name("simpletuner_entry.py")
    return [
        sys.executable,
        "-m",
        "accelerate.commands.accelerate_cli",
        "launch",
        "--mixed_precision=bf16",
        "--num_processes=1",
        "--num_machines=1",
        "--dynamo_backend=no",
        str(entry_path),
        str(train_path),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--simpletuner", required=True)
    parser.add_argument("--model-cache", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    worker_path = run_dir / "worker.json"
    stop_path = run_dir / "stop.request"
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "trainer.log"
    environment = _training_environment(args.simpletuner, args.model_cache)
    environment["ENV"] = "job"
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    process = None
    payload = {
        "schema_version": 1,
        "pid": os.getpid(),
        "child_pid": None,
        "status": "starting",
        "started_at": _now(),
        "heartbeat": _now(),
        "returncode": None,
    }
    _write(worker_path, payload)
    try:
        with log_path.open("a", encoding="utf-8", buffering=1) as log:
            process = subprocess.Popen(
                _trainer_command(args.simpletuner),
                cwd=run_dir,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
            )
            payload["child_pid"] = process.pid
            payload["status"] = "running"
            _write(worker_path, payload)
            while process.poll() is None:
                if stop_path.exists():
                    payload["status"] = "stop_requested"
                    payload["heartbeat"] = _now()
                    _write(worker_path, payload)
                    _stop_process(process)
                    break
                payload["heartbeat"] = _now()
                _write(worker_path, payload)
                time.sleep(0.5)
            returncode = process.wait()
    except BaseException as error:
        if process is not None and process.poll() is None:
            _stop_process(process)
        payload["status"] = "failed"
        payload["returncode"] = process.returncode if process is not None else None
        payload["error"] = f"{type(error).__name__}: {error}"
        payload["heartbeat"] = _now()
        payload["finished_at"] = _now()
        _write(worker_path, payload)
        raise
    payload["status"] = "interrupted" if stop_path.exists() else ("completed" if returncode == 0 else "failed")
    payload["returncode"] = returncode
    payload["heartbeat"] = _now()
    payload["finished_at"] = _now()
    _write(worker_path, payload)
    raise SystemExit(returncode)


if __name__ == "__main__":
    main()
