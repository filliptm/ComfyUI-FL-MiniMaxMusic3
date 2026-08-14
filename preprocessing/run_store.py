import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from ..training.paths import contained_path, preprocess_runs_root


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write(path, payload):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def create_run(spec):
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    run_dir = preprocess_runs_root() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    spec = {"schema_version": 1, "run_id": run_id, "created_at": _now(), **spec}
    state = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "created",
        "phase": "preflight",
        "message": "Preparing MOSS preprocessing",
        "current": 0,
        "total": len(spec.get("sources", [])),
        "track": None,
        "error": None,
        "warnings": [],
        "updated_at": _now(),
    }
    _write(run_dir / "job.json", spec)
    _write(run_dir / "state.json", state)
    return run_dir, spec, state


def update_state(run_dir, **changes):
    run_dir = Path(run_dir)
    try:
        state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        state = {"schema_version": 1, "run_id": run_dir.name}
    state.update(changes)
    state["updated_at"] = _now()
    _write(run_dir / "state.json", state)
    return state


def load_run(run_id):
    run_dir = contained_path(preprocess_runs_root(), run_id)
    if not run_dir.is_dir():
        raise ValueError(f"Preprocessing run does not exist: {run_id}")
    spec = json.loads((run_dir / "job.json").read_text(encoding="utf-8"))
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    return run_dir, spec, state


def list_runs(limit=30):
    root = preprocess_runs_root()
    root.mkdir(parents=True, exist_ok=True)
    result = []
    for path in sorted((item for item in root.iterdir() if item.is_dir()), key=lambda item: item.stat().st_mtime_ns, reverse=True):
        try:
            _run_dir, spec, state = load_run(path.name)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        result.append({
            "run_id": path.name,
            "source_folder": spec.get("source_folder"),
            "output_dataset": spec.get("output_dataset"),
            "status": state.get("status", "unknown"),
            "phase": state.get("phase", "unknown"),
            "current": state.get("current", 0),
            "total": state.get("total", 0),
            "message": state.get("message", ""),
            "updated_at": state.get("updated_at"),
        })
        if len(result) >= limit:
            break
    return result


def request_stop(run_dir):
    (Path(run_dir) / "stop.request").write_text("stop\n", encoding="utf-8")


def wait_for_state(run_dir, timeout=2.0):
    deadline = time.monotonic() + timeout
    path = Path(run_dir) / "state.json"
    while time.monotonic() < deadline:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(0.05)
    return None
