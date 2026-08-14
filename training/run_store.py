import json
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path

from .paths import ensure_directories, resolve_run, runs_root, validate_output_name


TERMINAL_STATES = {"completed", "failed", "interrupted"}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def create_run(dataset, train_config, validation_config, output_name):
    ensure_directories()
    output_name = validate_output_name(output_name)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{timestamp}-{output_name}-{secrets.token_hex(3)}"
    run_dir = runs_root() / run_id
    run_dir.mkdir(parents=True)
    spec = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": utc_now(),
        "output_name": output_name,
        "dataset": {
            "root": dataset["root"],
            "manifest_hash": dataset["manifest_hash"],
            "track_count": len(dataset["tracks"]),
            "total_seconds": dataset["total_seconds"],
            "settings": dataset["settings"],
        },
        "train_config": train_config,
        "validation_config": validation_config,
    }
    _write_json_atomic(run_dir / "run.json", spec)
    _write_json_atomic(run_dir / "state.json", {
        "schema_version": 1,
        "run_id": run_id,
        "status": "created",
        "phase": "created",
        "message": "Training run created",
        "current": 0,
        "total": train_config["max_train_steps"],
        "created_at": spec["created_at"],
        "updated_at": spec["created_at"],
        "sequence": 0,
        "last_checkpoint": None,
        "adapter_path": None,
        "error": None,
    })
    return run_dir, spec


def load_run(run_id):
    run_dir = resolve_run(run_id)
    spec = read_json(run_dir / "run.json")
    state = read_json(run_dir / "state.json")
    if not spec or not state:
        raise ValueError(f"Training run is incomplete: {run_id}")
    return run_dir, spec, state


def update_state(run_dir, **changes):
    path = Path(run_dir) / "state.json"
    state = read_json(path, {})
    state.update(changes)
    state["updated_at"] = utc_now()
    state["sequence"] = int(state.get("sequence", 0)) + 1
    _write_json_atomic(path, state)
    return state


def append_event(run_dir, state):
    event = {
        "event_version": 1,
        "run_id": state["run_id"],
        "sequence": state["sequence"],
        "timestamp": state["updated_at"],
        "phase": state.get("phase"),
        "state": state.get("status"),
        "current": state.get("current"),
        "total": state.get("total"),
        "metrics": state.get("metrics", {}),
        "message": state.get("message", ""),
        "artifact": state.get("artifact"),
    }
    with (Path(run_dir) / "events.jsonl").open("a", encoding="utf-8") as file:
        file.write(json.dumps(event, ensure_ascii=False) + "\n")
    return event


def append_metric(run_dir, metric):
    with (Path(run_dir) / "metrics.jsonl").open("a", encoding="utf-8") as file:
        file.write(json.dumps(metric, ensure_ascii=False) + "\n")


def recent_metrics(run_dir, limit=500):
    path = Path(run_dir) / "metrics.jsonl"
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]
    result = []
    for line in lines:
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return result


def list_runs(limit=100):
    ensure_directories()
    result = []
    directories = sorted((path for path in runs_root().iterdir() if path.is_dir()), reverse=True)
    for run_dir in directories[:limit]:
        spec = read_json(run_dir / "run.json")
        state = read_json(run_dir / "state.json")
        if spec and state:
            result.append({
                "run_id": run_dir.name,
                "output_name": spec.get("output_name"),
                "dataset": spec.get("dataset", {}).get("root"),
                "status": state.get("status"),
                "phase": state.get("phase"),
                "current": state.get("current"),
                "total": state.get("total"),
                "updated_at": state.get("updated_at"),
                "last_checkpoint": state.get("last_checkpoint"),
                "adapter_path": state.get("adapter_path"),
            })
    return result


def validate_resume(spec, dataset, train_config, output_name):
    mismatches = []
    if spec["dataset"]["manifest_hash"] != dataset["manifest_hash"]:
        mismatches.append("dataset manifest")
    previous = spec["train_config"]
    for key in ("lora_rank", "lora_alpha", "base_model_precision", "text_encoder_1_precision", "mixed_precision"):
        if previous.get(key) != train_config.get(key):
            mismatches.append(key)
    if spec["output_name"] != validate_output_name(output_name):
        mismatches.append("output name")
    if mismatches:
        raise ValueError(f"Cannot resume because these fields changed: {', '.join(mismatches)}")


def mark_stale_runs():
    for item in list_runs():
        if item["status"] in TERMINAL_STATES:
            continue
        run_dir = resolve_run(item["run_id"])
        worker = read_json(run_dir / "worker.json", {})
        pid = worker.get("pid")
        alive = False
        if isinstance(pid, int) and pid > 0:
            try:
                os.kill(pid, 0)
                alive = True
            except OSError:
                pass
        if not alive:
            update_state(run_dir, status="interrupted", phase="interrupted", message="Training stopped before completion and can be resumed")
