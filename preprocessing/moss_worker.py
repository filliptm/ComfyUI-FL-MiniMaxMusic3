import argparse
import hashlib
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


sys.path.insert(0, str(Path(__file__).resolve().parent))
from compiler import compile_caption, compile_lyrics, normalize_lyrics
from segmenter import plan_segments


ANALYSIS_PROMPT = """Analyze this music recording for a generative music training dataset. Return only one valid JSON object with this exact shape:
{
  "caption": "detailed natural-language musical description",
  "genres": ["genre or style"],
  "moods": ["mood or energy"],
  "bpm": 0,
  "meter": "time signature",
  "key": "musical key if supported by the audio, otherwise unknown",
  "instruments": ["audible instrument or sound"],
  "vocals": {"present": true, "language": "language or unknown", "character": "voice character", "delivery": "delivery style"},
  "harmony": ["harmonic characteristic"],
  "melody": ["melodic characteristic"],
  "production": ["production or mix characteristic"],
  "arrangement": ["arrangement characteristic"],
  "structure": [{"label": "Intro", "start_seconds": 0.0, "end_seconds": 12.0}]
}
Describe only audible evidence. Do not identify artists, titles, or source recordings. Use numeric seconds for all section boundaries. Keep the caption under 120 words, each descriptive list to at most 8 concise items, and the structure to at most 24 major sections. Do not include markdown or commentary."""

COMPACT_ANALYSIS_PROMPT = """Analyze this recording and return only compact valid JSON in this shape:
{"caption":"audible musical description under 120 words","genres":[],"moods":[],"bpm":0,"meter":"","key":"unknown","instruments":[],"vocals":{"present":false,"language":"unknown","character":"","delivery":""},"harmony":[],"melody":[],"production":[],"arrangement":[],"structure":[]}
Use at most 8 concise items per list and at most 16 major structure sections with label, start_seconds, and end_seconds. Do not include markdown."""

LYRICS_PROMPT = """Transcribe the lyrics and song sections in this recording. Return only one valid JSON object with this exact shape:
{
  "instrumental": false,
  "language": "language or unknown",
  "sections": [
    {"label": "Verse", "start_seconds": 0.0, "end_seconds": 12.0, "lines": ["one lyric line", "next lyric line"]}
  ]
}
If there are no intelligible lyrics, return {"instrumental": true, "language": "none", "sections": []}. Use numeric seconds, combine repeated choruses under their own timestamped sections, and keep the response concise enough to finish. Do not include markdown or commentary."""

COMPACT_LYRICS_PROMPT = """Return only compact valid JSON describing the intelligible lyrics in this recording:
{"instrumental":false,"language":"language or unknown","sections":[{"label":"Verse","start_seconds":0.0,"end_seconds":12.0,"lines":["lyric line"]}]}
Use no more than 32 timestamped sections. If no lyrics are intelligible, return {"instrumental":true,"language":"none","sections":[]}. Do not include markdown."""

SEGMENT_PROMPT = """Write a precise training caption for only this audio excerpt. Describe genre, tempo, mood, instrumentation, vocals, harmony, production, and the current arrangement section when audible. Do not name artists or recordings. Return only JSON: {"caption": "caption text"}."""
COMPACT_SEGMENT_PROMPT = """Return only valid JSON with one caption under 120 words for this excerpt: {"caption":"concise audible musical description"}. Do not include markdown."""


class MossMusicEncoderConfig:
    def __init__(self, values):
        self.__dict__.update(values)

    def to_dict(self):
        return dict(self.__dict__)


class MossJSONError(ValueError):
    def __init__(self, error, responses):
        super().__init__(str(error))
        self.responses = responses


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    for attempt in range(20):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if os.name != "nt" or attempt == 19:
                raise
            time.sleep(0.05)


class State:
    def __init__(self, run_dir):
        self.path = Path(run_dir) / "state.json"
        try:
            self.payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            self.payload = {"schema_version": 1, "run_id": Path(run_dir).name}

    def update(self, **changes):
        self.payload.update(changes)
        self.payload["updated_at"] = _now()
        _write_json(self.path, self.payload)


def _extract_json(text):
    text = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", str(text).strip(), flags=re.IGNORECASE)
    start = text.find("{")
    if start < 0:
        raise ValueError("MOSS did not return a JSON object")
    value, _end = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(value, dict):
        raise ValueError("MOSS returned JSON that is not an object")
    return value


def _load_audio(path, sample_rate=16000, start=None, duration=None):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("FFmpeg was not found in the MOSS worker environment")
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin"]
    if start is not None:
        command.extend(["-ss", f"{start:.6f}"])
    command.extend(["-i", str(path)])
    if duration is not None:
        command.extend(["-t", f"{duration:.6f}"])
    command.extend(["-map", "0:a:0", "-ac", "1", "-ar", str(sample_rate), "-f", "f32le", "-"])
    result = subprocess.run(command, capture_output=True, timeout=1800)
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", "replace").strip() or "FFmpeg could not decode the source audio")
    audio = np.frombuffer(result.stdout, dtype=np.float32).copy()
    if not audio.size:
        raise RuntimeError("Decoded source audio is empty")
    return audio


def _write_segment(source, destination, start, duration, sample_rate, preserve_channels):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("FFmpeg was not found in the MOSS worker environment")
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", str(source),
        "-ss", f"{start:.6f}", "-t", f"{duration:.6f}", "-map", "0:a:0", "-ar", str(sample_rate),
    ]
    if not preserve_channels:
        command.extend(["-ac", "1"])
    command.extend(["-c:a", "pcm_s16le", str(destination)])
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1800)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "FFmpeg could not write the training segment")


class MossEngine:
    def __init__(self, model_path, settings):
        if not torch.cuda.is_available():
            raise RuntimeError("MOSS-Music preprocessing requires an NVIDIA CUDA GPU")
        self.settings = settings
        model_path = Path(model_path).resolve()
        sys.path.insert(0, str(model_path.parent))
        configuration = importlib.import_module(f"{model_path.name}.configuration_moss_music")
        modeling = importlib.import_module(f"{model_path.name}.modeling_moss_music")
        processing = importlib.import_module(f"{model_path.name}.processing_moss_music")
        config = configuration.MossMusicConfig.from_pretrained(str(model_path), local_files_only=True)
        if isinstance(config.audio_config, dict):
            config.audio_config = MossMusicEncoderConfig(config.audio_config)
        self.model = modeling.MossMusicModel.from_pretrained(
            str(model_path), config=config, local_files_only=True, dtype="auto", device_map="cuda:0"
        )
        self.model.eval()
        self.processor = processing.MossMusicProcessor.from_pretrained(
            str(model_path), local_files_only=True, enable_time_marker=True
        )

    def generate(self, audio, prompt, temperature=None, max_new_tokens=None):
        inputs = self.processor(text=prompt, audios=[audio], return_tensors="pt")
        inputs = inputs.to(self.model.device)
        if inputs.get("audio_data") is not None:
            inputs["audio_data"] = inputs["audio_data"].to(self.model.dtype)
        inputs["audio_input_mask"] = inputs["input_ids"] == self.processor.audio_token_id
        temperature = self.settings["temperature"] if temperature is None else temperature
        kwargs = {
            "max_new_tokens": self.settings["max_new_tokens"] if max_new_tokens is None else max_new_tokens,
            "do_sample": temperature > 0,
            "num_beams": 1,
            "use_cache": True,
        }
        if temperature > 0:
            kwargs.update({"temperature": temperature, "top_p": 0.9, "top_k": 50})
        generated = self.model.generate(**inputs, **kwargs)
        input_length = inputs["input_ids"].shape[1]
        return self.processor.decode(generated[0, input_length:], skip_special_tokens=True).strip()

    def generate_json(self, audio, prompt, compact_prompt):
        responses = []
        first = self.generate(audio, prompt)
        responses.append(first)
        try:
            return _extract_json(first), first
        except (ValueError, json.JSONDecodeError) as first_error:
            repair = prompt + "\nThe response below was invalid or truncated. Return a corrected, concise JSON object only. Shorten lists and structure if needed:\n" + first[:6000]
            repair_tokens = max(2048, self.settings["max_new_tokens"])
            second = self.generate(audio, repair, temperature=0.0, max_new_tokens=repair_tokens)
            responses.append(second)
            try:
                return _extract_json(second), second
            except (ValueError, json.JSONDecodeError):
                third = self.generate(audio, compact_prompt, temperature=0.0, max_new_tokens=repair_tokens)
                responses.append(third)
                try:
                    return _extract_json(third), third
                except (ValueError, json.JSONDecodeError) as final_error:
                    raise MossJSONError(final_error, responses) from first_error


def _cache_key(source, settings, kind, segment=None):
    payload = {
        "source": source["sha256"],
        "settings": {
            "analysis_profile": settings["analysis_profile"],
            "temperature": settings["temperature"],
            "max_new_tokens": settings["max_new_tokens"],
        },
        "kind": kind,
        "segment": segment,
        "prompt_version": 2,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _cached_json(cache_root, key):
    path = Path(cache_root) / f"{key}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _save_cache(cache_root, key, payload):
    _write_json(Path(cache_root) / f"{key}.json", payload)


def _analysis(engine, audio, source, settings, cache_root):
    key = _cache_key(source, settings, "analysis")
    cached = _cached_json(cache_root, key)
    if cached:
        return cached["parsed"], cached.get("raw", ""), True
    parsed, raw = engine.generate_json(audio, ANALYSIS_PROMPT, COMPACT_ANALYSIS_PROMPT)
    _save_cache(cache_root, key, {"parsed": parsed, "raw": raw})
    return parsed, raw, False


def _lyrics(engine, audio, source, settings, cache_root):
    key = _cache_key(source, settings, "lyrics")
    cached = _cached_json(cache_root, key)
    if cached:
        return cached["parsed"], cached.get("raw", ""), True
    parsed, raw = engine.generate_json(audio, LYRICS_PROMPT, COMPACT_LYRICS_PROMPT)
    _save_cache(cache_root, key, {"parsed": parsed, "raw": raw})
    return parsed, raw, False


def _segment_caption(engine, audio, source, segment, settings, cache_root):
    key = _cache_key(source, settings, "segment_caption", {"start": round(segment["start"], 3), "end": round(segment["end"], 3)})
    cached = _cached_json(cache_root, key)
    if cached:
        return cached["parsed"].get("caption", ""), cached.get("raw", ""), True
    parsed, raw = engine.generate_json(audio, SEGMENT_PROMPT, COMPACT_SEGMENT_PROMPT)
    _save_cache(cache_root, key, {"parsed": parsed, "raw": raw})
    return parsed.get("caption", ""), raw, False


def _process_source(engine, source, settings, staging, cache_root, state, stop_path, completed, total):
    if stop_path.exists():
        raise InterruptedError("MOSS preprocessing stopped")
    state.update(status="running", phase="analyzing", message="Analyzing musical content", track=source["relative_path"], current=completed, total=total)
    audio = _load_audio(source["path"])
    analysis, raw_analysis, analysis_cached = _analysis(engine, audio, source, settings, cache_root)
    lyrics = {"instrumental": True, "language": "none", "sections": []}
    raw_lyrics = ""
    lyrics_cached = False
    if settings["analysis_profile"] != "caption_only":
        state.update(phase="transcribing", message="Transcribing lyrics and song sections")
        lyrics, raw_lyrics, lyrics_cached = _lyrics(engine, audio, source, settings, cache_root)
        lyrics = normalize_lyrics(lyrics, analysis)
    sections = analysis.get("structure") or lyrics.get("sections") or []
    segments = plan_segments(
        source["duration"], sections,
        settings["min_segment_seconds"], settings["target_segment_seconds"], settings["max_segment_seconds"],
        settings["segment_long_tracks"],
    )
    outputs = []
    for segment in segments:
        if stop_path.exists():
            raise InterruptedError("MOSS preprocessing stopped")
        stem = f"{source['slug']}__s{segment['index']:03d}"
        audio_path = staging / f"{stem}.wav"
        state.update(phase="segmenting", message=f"Writing segment {segment['index'] + 1} of {len(segments)}")
        _write_segment(
            source["path"], audio_path, segment["start"], segment["duration"],
            settings["output_sample_rate"], settings["preserve_channels"],
        )
        caption_text = analysis.get("caption", "")
        raw_segment = ""
        caption_cached = analysis_cached
        if len(segments) > 1:
            state.update(phase="captioning", message=f"Captioning segment {segment['index'] + 1} of {len(segments)}")
            segment_audio = _load_audio(source["path"], start=segment["start"], duration=segment["duration"])
            caption_text, raw_segment, caption_cached = _segment_caption(engine, segment_audio, source, segment, settings, cache_root)
        caption = compile_caption(analysis, caption_text)
        if not caption:
            raise ValueError("MOSS returned an empty music caption")
        audio_path.with_suffix(".txt").write_text(caption + "\n", encoding="utf-8")
        lyrics_text = None
        if settings["analysis_profile"] != "caption_only":
            lyrics_text = compile_lyrics(lyrics, segment["start"], segment["end"])
            audio_path.with_suffix(".lyrics").write_text(lyrics_text + "\n", encoding="utf-8")
        metadata = {
            "schema_version": 1,
            "source": {
                "relative_path": source["relative_path"],
                "sha256": source["sha256"],
                "duration_seconds": source["duration"],
                "sample_rate": source["sample_rate"],
                "channels": source["channels"],
            },
            "segment": segment,
            "analysis": analysis,
            "lyrics": lyrics,
            "outputs": {"caption": caption, "lyrics": lyrics_text},
            "provenance": {
                "generator": "FL MiniMax Music 3 Dataset Preprocessor",
                "model_id": "OpenMOSS-Team/MOSS-Music-8B-Instruct",
                "model_revision": "fce7f8304e96cc2d3398b8106456cbb2ecec3139",
                "prompt_schema": 1,
                "analysis_cached": analysis_cached,
                "lyrics_cached": lyrics_cached,
                "caption_cached": caption_cached,
            },
            "raw": {"analysis": raw_analysis, "lyrics": raw_lyrics, "segment_caption": raw_segment},
            "review": {"status": "generated", "warnings": []},
        }
        metadata_path = audio_path.with_suffix(".music3.json")
        _write_json(metadata_path, metadata)
        outputs.append({"audio": audio_path.name, "caption": audio_path.with_suffix(".txt").name, "lyrics": audio_path.with_suffix(".lyrics").name if lyrics_text is not None else None, "metadata": metadata_path.name})
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--model-path", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    spec = json.loads((run_dir / "job.json").read_text(encoding="utf-8"))
    settings = spec["settings"]
    staging = run_dir / "dataset"
    staging.mkdir(parents=True, exist_ok=True)
    stop_path = run_dir / "stop.request"
    state = State(run_dir)
    errors = []
    all_outputs = []
    state.update(status="running", phase="loading_model", message="Loading MOSS-Music", current=0, total=len(spec["sources"]), error=None)
    try:
        engine = MossEngine(args.model_path, settings)
        for index, source in enumerate(spec["sources"]):
            try:
                outputs = _process_source(engine, source, settings, staging, spec["cache_root"], state, stop_path, index, len(spec["sources"]))
                all_outputs.extend(outputs)
            except InterruptedError:
                raise
            except Exception as error:
                if isinstance(error, MossJSONError):
                    _write_json(run_dir / "failures" / f"{source['slug']}.json", {"source": source["relative_path"], "responses": error.responses})
                errors.append(f"{source['relative_path']}: {type(error).__name__}: {error}")
            state.update(current=index + 1, total=len(spec["sources"]), message=f"Processed {index + 1} of {len(spec['sources'])} source tracks")
        if not all_outputs:
            raise RuntimeError("MOSS preprocessing produced no valid training segments. " + (errors[0] if errors else ""))
        manifest = {
            "schema_version": 1,
            "generator": "FL MiniMax Music 3 Dataset Preprocessor",
            "run_id": spec["run_id"],
            "source_folder": spec["source_folder"],
            "output_dataset": spec["output_dataset"],
            "settings": settings,
            "segments": all_outputs,
            "errors": errors,
            "created_at": _now(),
        }
        _write_json(staging / "dataset.music3.json", manifest)
        _write_json(run_dir / "result.json", manifest)
        state.update(status="completed", phase="completed", message=f"Created {len(all_outputs)} training segments", current=len(spec["sources"]), total=len(spec["sources"]), warnings=errors, result=str(run_dir / "result.json"), finished_at=_now())
    except InterruptedError as error:
        state.update(status="interrupted", phase="interrupted", message=str(error), finished_at=_now())
        raise SystemExit(130)
    except BaseException as error:
        state.update(status="failed", phase="failed", message="MOSS preprocessing failed", error=f"{type(error).__name__}: {error}"[:4000], warnings=errors, finished_at=_now())
        raise


if __name__ == "__main__":
    main()
