import hashlib
import json
import os
import shutil
import threading
import time
from pathlib import Path

import requests

from ..training.paths import moss_model_root, pack_root


CHUNK_SIZE = 4 * 1024 * 1024
_model_lock = threading.Lock()


def load_model_manifest():
    return json.loads((pack_root() / "preprocessing" / "model_manifest.json").read_text(encoding="utf-8"))


def _sha256(path, callback=None, artifact=None):
    digest = hashlib.sha256()
    total = path.stat().st_size
    current = 0
    last_event = 0.0
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(CHUNK_SIZE), b""):
            digest.update(chunk)
            current += len(chunk)
            now = time.monotonic()
            if callback and (now - last_event >= 0.2 or current == total):
                callback({"state": "verifying", "artifact": artifact, "value": current, "max": total, "message": f"Verifying {artifact}"})
                last_event = now
    return digest.hexdigest()


def _marker_path():
    return moss_model_root() / ".fl_verified.json"


def _marker_valid(manifest):
    try:
        marker = json.loads(_marker_path().read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    if marker.get("revision") != manifest["revision"]:
        return False
    recorded = {item["path"]: item for item in marker.get("files", [])}
    for artifact in manifest["files"]:
        path = moss_model_root() / artifact["path"]
        item = recorded.get(artifact["path"])
        if not path.is_file() or item is None:
            return False
        stat = path.stat()
        if stat.st_size != artifact["size"] or item.get("size") != stat.st_size or item.get("mtime_ns") != stat.st_mtime_ns:
            return False
    return True


def model_inventory():
    manifest = load_model_manifest()
    files = []
    for artifact in manifest["files"]:
        path = moss_model_root() / artifact["path"]
        partial = Path(f"{path}.part")
        available = path.stat().st_size if path.is_file() else (partial.stat().st_size if partial.is_file() else 0)
        state = "present" if path.is_file() and available == artifact["size"] else ("partial" if 0 < available < artifact["size"] else "missing")
        if available > artifact["size"]:
            state = "invalid"
        files.append({
            "path": artifact["path"],
            "size": artifact["size"],
            "available": available,
            "state": state,
        })
    return {
        "repo_id": manifest["repo_id"],
        "revision": manifest["revision"],
        "model_path": str(moss_model_root()),
        "verified": _marker_valid(manifest),
        "total_size": sum(item["size"] for item in manifest["files"]),
        "files": files,
    }


def verify_model(callback=None):
    manifest = load_model_manifest()
    if _marker_valid(manifest):
        return model_inventory()
    records = []
    for artifact in manifest["files"]:
        path = moss_model_root() / artifact["path"]
        if not path.is_file() or path.stat().st_size != artifact["size"]:
            raise RuntimeError(f"MOSS-Music model file is missing or incomplete: {artifact['path']}")
        if _sha256(path, callback, artifact["path"]) != artifact["sha256"]:
            raise RuntimeError(f"MOSS-Music checksum mismatch: {artifact['path']}")
        stat = path.stat()
        records.append({"path": artifact["path"], "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    marker = {"schema_version": 1, "revision": manifest["revision"], "files": records}
    temporary = _marker_path().with_suffix(".tmp")
    temporary.write_text(json.dumps(marker, indent=2), encoding="utf-8")
    os.replace(temporary, _marker_path())
    return model_inventory()


def _download_url(manifest, artifact):
    return f"https://huggingface.co/{manifest['repo_id']}/resolve/{manifest['revision']}/{artifact['path']}?download=true"


def _download_artifact(manifest, artifact, callback=None, stop=None):
    target = moss_model_root() / artifact["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(f"{target}.part")
    expected = artifact["size"]
    if target.is_file() and target.stat().st_size == expected:
        return
    if target.exists():
        target.unlink()
    if partial.is_file() and partial.stat().st_size > expected:
        partial.unlink()
    current = partial.stat().st_size if partial.is_file() else 0
    if current == expected:
        if _sha256(partial, callback, artifact["path"]) == artifact["sha256"]:
            os.replace(partial, target)
            return
        partial.unlink()
        current = 0
    if shutil.disk_usage(target.parent).free < expected - current:
        raise RuntimeError(f"Not enough free space to download {artifact['path']}")
    headers = {"Range": f"bytes={current}-"} if current else {}
    if callback:
        callback({"state": "downloading", "artifact": artifact["path"], "value": current, "max": expected, "message": f"Downloading {artifact['path']}"})
    with requests.get(_download_url(manifest, artifact), headers=headers, stream=True, timeout=(15, 300)) as response:
        mode = "ab" if current and response.status_code == 206 else "wb"
        if current and response.status_code == 206:
            content_range = response.headers.get("Content-Range", "")
            if not content_range.startswith(f"bytes {current}-"):
                raise RuntimeError(f"Invalid resume response for {artifact['path']}")
        else:
            response.raise_for_status()
            if current:
                current = 0
        last_event = 0.0
        with partial.open(mode) as file:
            for chunk in response.iter_content(CHUNK_SIZE):
                if stop and stop():
                    raise InterruptedError("MOSS-Music download stopped")
                if not chunk:
                    continue
                file.write(chunk)
                current += len(chunk)
                if current > expected:
                    raise RuntimeError(f"Download exceeded the expected size for {artifact['path']}")
                now = time.monotonic()
                if callback and (now - last_event >= 0.2 or current == expected):
                    callback({"state": "downloading", "artifact": artifact["path"], "value": current, "max": expected, "message": f"Downloading {artifact['path']}"})
                    last_event = now
    if current != expected:
        raise RuntimeError(f"Incomplete MOSS-Music download for {artifact['path']}: {current} of {expected} bytes")
    if _sha256(partial, callback, artifact["path"]) != artifact["sha256"]:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"MOSS-Music checksum mismatch: {artifact['path']}")
    os.replace(partial, target)


def ensure_model(policy, callback=None, stop=None):
    manifest = load_model_manifest()
    with _model_lock:
        if _marker_valid(manifest):
            return {"path": str(moss_model_root()), **model_inventory()}
        complete = all((moss_model_root() / item["path"]).is_file() and (moss_model_root() / item["path"]).stat().st_size == item["size"] for item in manifest["files"])
        if not complete:
            if policy != "download_if_missing":
                raise RuntimeError("MOSS-Music is not installed. Choose download_if_missing or use Download in the preprocessor dashboard.")
            moss_model_root().mkdir(parents=True, exist_ok=True)
            for artifact in manifest["files"]:
                _download_artifact(manifest, artifact, callback, stop)
        if callback:
            callback({"state": "verifying", "artifact": None, "value": 0, "max": 0, "message": "Verifying the pinned MOSS-Music snapshot"})
        result = verify_model(callback)
        if callback:
            callback({"state": "ready", "artifact": None, "value": result["total_size"], "max": result["total_size"], "message": "MOSS-Music is verified"})
        return {"path": str(moss_model_root()), **result}
