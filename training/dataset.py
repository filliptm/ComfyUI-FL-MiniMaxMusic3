import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

from .paths import resolve_dataset_folder


AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac"}
SIDECAR_EXTENSION = re.compile(r"^\.[A-Za-z0-9_-]{1,16}$")
VOLUME_PATTERN = re.compile(r"(mean_volume|max_volume):\s*(-?(?:inf|\d+(?:\.\d+)?))\s*dB", re.IGNORECASE)


def _validate_sidecar_extension(value, label):
    value = str(value).strip()
    if not SIDECAR_EXTENSION.fullmatch(value):
        raise ValueError(f"{label} must be a simple extension such as .txt or .lyrics")
    return value


def _audio_files(root, recursive):
    iterator = root.rglob("*") if recursive else root.iterdir()
    return sorted(path for path in iterator if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS)


def _probe_audio(path):
    executable = shutil.which("ffprobe")
    if executable is None:
        raise RuntimeError("FFprobe was not found. Install FFmpeg and ensure ffprobe is on PATH.")
    command = [
        executable,
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=codec_type,sample_rate,channels",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
    if result.returncode != 0:
        raise ValueError(result.stderr.strip() or "FFprobe could not decode the file")
    payload = json.loads(result.stdout)
    stream = next((item for item in payload.get("streams", []) if item.get("codec_type") == "audio"), None)
    if stream is None:
        raise ValueError("No audio stream was found")
    duration = float(payload.get("format", {}).get("duration", 0.0))
    return {
        "duration": duration,
        "sample_rate": int(stream.get("sample_rate") or 0),
        "channels": int(stream.get("channels") or 0),
    }


def _volume_levels(path):
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise RuntimeError("FFmpeg was not found. Install FFmpeg and ensure it is on PATH.")
    command = [executable, "-hide_banner", "-nostdin", "-i", str(path), "-af", "volumedetect", "-f", "null", "-"]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
    values = {}
    for name, value in VOLUME_PATTERN.findall(result.stderr):
        values[name.lower()] = float("-inf") if value.lower() == "-inf" else float(value)
    return values


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_change_token(folder, recursive, caption_extension, lyrics_extension, missing_lyrics):
    root = resolve_dataset_folder(folder)
    caption_extension = _validate_sidecar_extension(caption_extension, "Caption extension")
    lyrics_extension = _validate_sidecar_extension(lyrics_extension, "Lyrics extension")
    digest = hashlib.sha256(f"{recursive}|{caption_extension}|{lyrics_extension}|{missing_lyrics}".encode())
    for path in _audio_files(root, recursive):
        stat = path.stat()
        digest.update(f"{path.relative_to(root).as_posix()}|{stat.st_size}|{stat.st_mtime_ns}".encode())
        for extension in (caption_extension, lyrics_extension):
            sidecar = path.with_suffix(extension)
            if sidecar.is_file():
                stat = sidecar.stat()
                digest.update(f"{sidecar.relative_to(root).as_posix()}|{stat.st_size}|{stat.st_mtime_ns}".encode())
    return digest.hexdigest()


def scan_dataset(
    folder,
    recursive=True,
    caption_extension=".txt",
    lyrics_extension=".lyrics",
    missing_lyrics="instrumental",
    min_duration=1.0,
    max_duration=60.0,
    duration_interval=3.0,
    audio_analysis="metadata",
    include_invalid=False,
):
    root = resolve_dataset_folder(folder)
    caption_extension = _validate_sidecar_extension(caption_extension, "Caption extension")
    lyrics_extension = _validate_sidecar_extension(lyrics_extension, "Lyrics extension")
    if missing_lyrics not in {"reject", "instrumental"}:
        raise ValueError("Missing lyrics policy must be reject or instrumental")
    if min_duration < 0 or max_duration <= min_duration:
        raise ValueError("Maximum duration must be greater than minimum duration")
    if duration_interval <= 0:
        raise ValueError("Duration interval must be positive")

    tracks = []
    errors = []
    warnings = []
    hashes = {}
    audio_files = _audio_files(root, recursive)
    for audio_path in audio_files:
        relative = audio_path.relative_to(root).as_posix()
        caption_path = audio_path.with_suffix(caption_extension)
        lyrics_path = audio_path.with_suffix(lyrics_extension)
        track_errors = []
        track_warnings = []
        try:
            metadata = _probe_audio(audio_path)
        except (OSError, ValueError, subprocess.SubprocessError, json.JSONDecodeError) as error:
            metadata = {"duration": 0.0, "sample_rate": 0, "channels": 0}
            track_errors.append(f"decode failed: {error}")

        caption = caption_path.read_text(encoding="utf-8").strip() if caption_path.is_file() else ""
        lyrics = lyrics_path.read_text(encoding="utf-8").strip() if lyrics_path.is_file() else ""
        if not caption:
            track_errors.append(f"missing or empty {caption_extension} caption")
        if not lyrics and missing_lyrics == "reject":
            track_errors.append(f"missing or empty {lyrics_extension} lyrics")
        elif not lyrics:
            track_warnings.append("no lyrics; treated as instrumental")
        duration = metadata["duration"]
        if duration and duration < min_duration:
            track_errors.append(f"duration {duration:.2f}s is below {min_duration:.2f}s")
        if duration > max_duration:
            track_errors.append(f"duration {duration:.2f}s exceeds {max_duration:.2f}s")
        if metadata["channels"] not in {1, 2}:
            track_warnings.append(f"unusual channel count: {metadata['channels']}")

        levels = {}
        content_hash = None
        if audio_analysis == "full" and not any(item.startswith("decode failed") for item in track_errors):
            try:
                levels = _volume_levels(audio_path)
                content_hash = _sha256(audio_path)
                hashes.setdefault(content_hash, []).append(relative)
                if levels.get("mean_volume", 0.0) <= -70.0:
                    track_errors.append("audio appears silent")
                if levels.get("max_volume", -100.0) >= -0.1:
                    track_warnings.append("audio peaks at or near digital full scale")
            except (OSError, subprocess.SubprocessError) as error:
                track_warnings.append(f"deep audio analysis failed: {error}")

        track = {
            "audio": str(audio_path),
            "relative_audio": relative,
            "caption": caption,
            "lyrics": lyrics,
            "duration": duration,
            "sample_rate": metadata["sample_rate"],
            "channels": metadata["channels"],
            "levels": levels,
            "content_hash": content_hash,
            "valid": not track_errors,
            "errors": track_errors,
            "warnings": track_warnings,
        }
        if track["valid"] or include_invalid:
            tracks.append(track)
        errors.extend(f"{relative}: {message}" for message in track_errors)
        warnings.extend(f"{relative}: {message}" for message in track_warnings)

    for duplicates in hashes.values():
        if len(duplicates) > 1:
            warnings.append(f"duplicate audio content: {', '.join(duplicates)}")

    manifest_hash = dataset_change_token(folder, recursive, caption_extension, lyrics_extension, missing_lyrics)
    valid_tracks = [track for track in tracks if track["valid"]]
    total_seconds = sum(track["duration"] for track in valid_tracks)
    report = {
        "schema_version": 1,
        "root": str(root),
        "manifest_hash": manifest_hash,
        "audio_files": len(audio_files),
        "valid_tracks": len(valid_tracks),
        "invalid_tracks": len(audio_files) - len(valid_tracks),
        "total_seconds": total_seconds,
        "estimated_vae_cache_bytes": int(total_seconds * 128 * 75 * 2),
        "errors": errors,
        "warnings": warnings,
    }
    dataset = {
        "schema_version": 1,
        "root": str(root),
        "manifest_hash": manifest_hash,
        "tracks": valid_tracks,
        "total_seconds": total_seconds,
        "settings": {
            "recursive": bool(recursive),
            "caption_extension": caption_extension,
            "lyrics_extension": lyrics_extension,
            "missing_lyrics": missing_lyrics,
            "min_duration": float(min_duration),
            "max_duration": float(max_duration),
            "duration_interval": float(duration_interval),
        },
        "report": report,
    }
    return dataset
