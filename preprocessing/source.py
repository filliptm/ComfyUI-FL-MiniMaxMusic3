import hashlib
import json
import re
import shutil
import subprocess

from ..training.dataset import AUDIO_EXTENSIONS
from ..training.paths import resolve_source_folder, source_root


SLUG_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


def source_folders():
    root = source_root()
    root.mkdir(parents=True, exist_ok=True)
    folders = set()
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            folders.add(path.parent.relative_to(root).as_posix() or ".")
    return sorted(folders) or ["<no source audio found>"]


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _probe(path):
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError("FFprobe was not found. Install FFmpeg and ensure it is on PATH.")
    result = subprocess.run([
        ffprobe, "-v", "error", "-show_entries", "format=duration:stream=codec_type,sample_rate,channels",
        "-of", "json", str(path),
    ], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
    if result.returncode:
        raise ValueError(result.stderr.strip() or "FFprobe could not read the source audio")
    payload = json.loads(result.stdout)
    stream = next((item for item in payload.get("streams", []) if item.get("codec_type") == "audio"), None)
    if stream is None:
        raise ValueError("No audio stream was found")
    return {
        "duration": float(payload.get("format", {}).get("duration") or 0.0),
        "sample_rate": int(stream.get("sample_rate") or 0),
        "channels": int(stream.get("channels") or 0),
    }


def discover_sources(folder):
    root = resolve_source_folder(folder)
    tracks = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.suffix.lower() in AUDIO_EXTENSIONS):
        metadata = _probe(path)
        if metadata["duration"] <= 0:
            raise ValueError(f"Source audio has no duration: {path.name}")
        relative = path.relative_to(root).as_posix()
        slug = SLUG_PATTERN.sub("_", str(path.relative_to(root).with_suffix("")).replace("/", "__")).strip("._") or "track"
        tracks.append({
            "path": str(path),
            "relative_path": relative,
            "slug": slug[:120],
            "sha256": _sha256(path),
            **metadata,
        })
    if not tracks:
        raise ValueError(f"No supported audio files were found in {root}")
    return root, tracks


def source_change_token(folder):
    root = resolve_source_folder(folder)
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.suffix.lower() in AUDIO_EXTENSIONS):
        stat = path.stat()
        digest.update(f"{path.relative_to(root).as_posix()}|{stat.st_size}|{stat.st_mtime_ns}".encode())
    return digest.hexdigest()
