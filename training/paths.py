import json
import re
from pathlib import Path

import folder_paths


OUTPUT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def pack_root():
    return Path(__file__).resolve().parents[1]


def dataset_root():
    return Path(folder_paths.get_input_directory()).resolve() / "fl_minimax_music3" / "datasets"


def source_root():
    return Path(folder_paths.get_input_directory()).resolve() / "fl_minimax_music3" / "sources"


def user_root():
    return Path(folder_paths.get_user_directory()).resolve() / "fl_minimax_music3"


def backend_root():
    return user_root() / "backends"


def runs_root():
    return user_root() / "runs"


def dataset_cache_root():
    return user_root() / "dataset_cache"


def preprocess_root():
    return user_root() / "preprocess"


def preprocess_runs_root():
    return preprocess_root() / "runs"


def preprocess_cache_root():
    return preprocess_root() / "cache"


def moss_backend_root():
    return backend_root() / "moss_music"


def moss_model_root():
    return Path(folder_paths.models_dir).resolve() / "moss_music" / "MOSS-Music-8B-Instruct"


def model_cache_root():
    return Path(folder_paths.models_dir).resolve() / "minimax_music3_training" / "huggingface"


def lora_root():
    paths = folder_paths.get_folder_paths("loras")
    if not paths:
        raise RuntimeError("ComfyUI has no configured LoRA model directory")
    return Path(paths[0]).resolve() / "MiniMaxMusic3"


def ensure_directories():
    for path in (
        dataset_root(),
        source_root(),
        backend_root(),
        runs_root(),
        dataset_cache_root(),
        preprocess_runs_root(),
        preprocess_cache_root(),
        moss_backend_root(),
        moss_model_root(),
        model_cache_root(),
        lora_root(),
    ):
        path.mkdir(parents=True, exist_ok=True)


def contained_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Path must remain inside {root}") from error
    return path


def resolve_dataset_folder(value):
    if not value or value == "<no datasets found>":
        raise ValueError(f"Choose a dataset folder under {dataset_root()}")
    path = contained_path(dataset_root(), value)
    if not path.is_dir():
        raise ValueError(f"Dataset folder does not exist: {path}")
    return path


def resolve_source_folder(value):
    if not value or value == "<no source audio found>":
        raise ValueError("Enter an absolute path to a source audio folder")
    candidate = Path(str(value).strip())
    path = candidate.resolve() if candidate.is_absolute() else contained_path(source_root(), candidate)
    if not path.is_dir():
        raise ValueError(f"Source folder does not exist: {path}")
    return path


def resolve_output_dataset(value):
    return contained_path(dataset_root(), validate_output_name(value))


def validate_output_name(value):
    value = str(value).strip()
    if not OUTPUT_NAME_PATTERN.fullmatch(value):
        raise ValueError("Output name must be 1-80 letters, numbers, dots, underscores, or hyphens")
    return value


def resolve_run(run_id):
    if not RUN_ID_PATTERN.fullmatch(str(run_id)):
        raise ValueError("Invalid training run ID")
    path = contained_path(runs_root(), run_id)
    if not path.is_dir():
        raise ValueError(f"Training run does not exist: {run_id}")
    return path


def load_backend_manifest():
    return json.loads((pack_root() / "training" / "backend_manifest.json").read_text(encoding="utf-8"))
