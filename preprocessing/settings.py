import json


DEFAULT_SETTINGS = {
    "schema_version": 1,
    "analysis_profile": "caption_and_lyrics",
    "segment_long_tracks": True,
    "min_segment_seconds": 8.0,
    "target_segment_seconds": 42.0,
    "max_segment_seconds": 60.0,
    "output_sample_rate": 44100,
    "preserve_channels": True,
    "write_policy": "fill_missing",
    "execution_mode": "auto_process_and_write",
    "model_policy": "download_if_missing",
    "backend_policy": "install_if_missing",
    "temperature": 0.2,
    "max_new_tokens": 1024,
}


def normalize_settings(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(f"Preprocessor settings are not valid JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("Preprocessor settings must be a JSON object")
    settings = {**DEFAULT_SETTINGS, **value}
    if int(settings.get("schema_version", 0)) != 1:
        raise ValueError("Unsupported preprocessor settings schema")
    if settings["analysis_profile"] not in {"caption_only", "caption_and_lyrics", "full_analysis"}:
        raise ValueError("Unknown MOSS analysis profile")
    if settings["write_policy"] not in {"fill_missing", "replace_generated", "replace_all"}:
        raise ValueError("Unknown dataset write policy")
    if settings["execution_mode"] not in {"auto_process_and_write", "require_review"}:
        raise ValueError("Unknown preprocessor execution mode")
    if settings["model_policy"] not in {"require_installed", "download_if_missing"}:
        raise ValueError("Unknown MOSS model policy")
    if settings["backend_policy"] not in {"require_installed", "install_if_missing"}:
        raise ValueError("Unknown MOSS backend policy")
    minimum = float(settings["min_segment_seconds"])
    target = float(settings["target_segment_seconds"])
    maximum = float(settings["max_segment_seconds"])
    if minimum < 1.0 or not minimum <= target <= maximum or maximum > 1800.0:
        raise ValueError("Segment durations must satisfy 1 <= minimum <= target <= maximum <= 1800")
    sample_rate = int(settings["output_sample_rate"])
    if sample_rate not in {32000, 44100, 48000}:
        raise ValueError("Output sample rate must be 32000, 44100, or 48000 Hz")
    temperature = float(settings["temperature"])
    if not 0.0 <= temperature <= 2.0:
        raise ValueError("Temperature must be between 0 and 2")
    max_new_tokens = int(settings["max_new_tokens"])
    if not 128 <= max_new_tokens <= 4096:
        raise ValueError("Maximum new tokens must be between 128 and 4096")
    settings.update({
        "segment_long_tracks": bool(settings["segment_long_tracks"]),
        "min_segment_seconds": minimum,
        "target_segment_seconds": target,
        "max_segment_seconds": maximum,
        "output_sample_rate": sample_rate,
        "preserve_channels": bool(settings["preserve_channels"]),
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
    })
    return settings


def settings_json(settings=None):
    return json.dumps(normalize_settings(settings or DEFAULT_SETTINGS), separators=(",", ":"), ensure_ascii=False)
