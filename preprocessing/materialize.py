import json
import os
import shutil
from pathlib import Path


OWNED_EXTENSIONS = {".wav", ".flac", ".txt", ".lyrics", ".json"}


def _generated(path):
    metadata = path if path.name.endswith(".music3.json") else path.with_suffix(".music3.json")
    try:
        payload = json.loads(metadata.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    return (
        payload.get("provenance", {}).get("generator") == "FL MiniMax Music 3 Dataset Preprocessor"
        and payload.get("review", {}).get("status") not in {"edited", "approved"}
    )


def _copy_atomic(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def materialize_dataset(staging, destination, write_policy):
    staging = Path(staging).resolve()
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    written = []
    skipped = []
    for source in sorted(path for path in staging.rglob("*") if path.is_file()):
        relative = source.relative_to(staging)
        target = destination / relative
        if target.suffix.lower() not in OWNED_EXTENSIONS:
            continue
        replace = not target.exists() or write_policy == "replace_all"
        if write_policy == "replace_generated" and target.exists():
            replace = _generated(target)
        if source.name == "dataset.music3.json":
            replace = True
        if replace:
            _copy_atomic(source, target)
            written.append(relative.as_posix())
        else:
            skipped.append(relative.as_posix())
    return {"written": written, "skipped": skipped}
