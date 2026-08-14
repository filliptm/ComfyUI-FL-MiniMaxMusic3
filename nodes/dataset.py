import json

from ..training.dataset import AUDIO_EXTENSIONS, dataset_change_token, scan_dataset
from ..training.paths import dataset_root


def dataset_folders():
    root = dataset_root()
    root.mkdir(parents=True, exist_ok=True)
    folders = set()
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            folders.add(path.parent.relative_to(root).as_posix() or ".")
    return sorted(folders) or ["<no datasets found>"]


class FL_MiniMaxMusic3Dataset:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset_folder": (dataset_folders(),),
                "recursive": ("BOOLEAN", {"default": True}),
                "caption_extension": ("STRING", {"default": ".txt"}),
                "lyrics_extension": ("STRING", {"default": ".lyrics"}),
                "missing_lyrics": (["instrumental", "reject"],),
                "min_duration": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 600.0, "step": 0.5}),
                "max_duration": ("FLOAT", {"default": 60.0, "min": 1.0, "max": 1800.0, "step": 1.0}),
                "duration_interval": ("FLOAT", {"default": 3.0, "min": 0.5, "max": 30.0, "step": 0.5}),
                "audio_analysis": (["metadata", "full"],),
                "include_invalid": ("BOOLEAN", {"default": False}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("FL_MINIMAX_MUSIC3_DATASET", "STRING")
    RETURN_NAMES = ("dataset", "report")
    FUNCTION = "load_dataset"
    CATEGORY = "FL/MiniMax Music 3/Training"
    DESCRIPTION = "Validates a local caption-and-lyrics Music 3 audio dataset without decoding it into workflow tensors."

    @classmethod
    def IS_CHANGED(cls, dataset_folder, recursive, caption_extension, lyrics_extension, missing_lyrics, **_kwargs):
        try:
            return dataset_change_token(dataset_folder, recursive, caption_extension, lyrics_extension, missing_lyrics)
        except (OSError, ValueError):
            return float("NaN")

    def load_dataset(self, dataset_folder, recursive, caption_extension, lyrics_extension, missing_lyrics, min_duration, max_duration, duration_interval, audio_analysis, include_invalid, unique_id=None):
        dataset = scan_dataset(
            dataset_folder,
            recursive,
            caption_extension,
            lyrics_extension,
            missing_lyrics,
            min_duration,
            max_duration,
            duration_interval,
            audio_analysis,
            include_invalid,
        )
        if not dataset["tracks"]:
            raise ValueError("MiniMax Music 3 dataset has no valid tracks. Inspect the report for caption, lyrics, duration, or decode errors.")
        return dataset, json.dumps(dataset["report"], indent=2, ensure_ascii=False)
