import shutil
from pathlib import Path

from .paths import dataset_cache_root, lora_root, load_backend_manifest, model_cache_root
from .run_store import _write_json_atomic


def write_simpletuner_config(run_dir, spec, resume=False):
    run_dir = Path(run_dir)
    manifest = load_backend_manifest()
    train = spec["train_config"]
    dataset = spec["dataset"]
    validation = spec.get("validation_config")
    if dataset["settings"]["caption_extension"] != ".txt":
        raise ValueError("The pinned SimpleTuner textfile backend requires .txt captions")

    config_dir = run_dir / "config" / "job"
    config_dir.mkdir(parents=True, exist_ok=True)
    backend_output = run_dir / "backend_output"
    vae_cache = dataset_cache_root() / dataset["manifest_hash"] / "vae"
    text_cache = dataset_cache_root() / dataset["manifest_hash"] / "text"
    data_config = [
        {
            "id": "fl-minimax-music3-audio",
            "type": "local",
            "dataset_type": "audio",
            "instance_data_dir": dataset["root"],
            "metadata_backend": "discovery",
            "caption_strategy": "textfile",
            "audio": {
                "bucket_strategy": "duration",
                "duration_interval": dataset["settings"]["duration_interval"],
                "max_duration_seconds": dataset["settings"]["max_duration"],
                "lyrics_filename_format": f"{{filename}}{dataset['settings']['lyrics_extension']}",
                "channels": 1,
                "truncation_mode": "beginning",
            },
            "cache_dir_vae": vae_cache.as_posix(),
        },
        {
            "id": "text-embeds",
            "dataset_type": "text_embeds",
            "default": True,
            "type": "local",
            "cache_dir": text_cache.as_posix(),
        },
    ]
    config = {
        "model_family": "minimaxmusic",
        "model_type": "lora",
        "model_flavour": "music3",
        "pretrained_model_name_or_path": manifest["pretrained_model"],
        "pretrained_vae_model_name_or_path": manifest["pretrained_vae"],
        "output_dir": backend_output.as_posix(),
        "logging_dir": (run_dir / "logs" / "tensorboard").as_posix(),
        "cache_dir": model_cache_root().as_posix(),
        "data_backend_config": (config_dir / "multidatabackend.json").as_posix(),
        "resolution": 512,
        "mixed_precision": train["mixed_precision"],
        "base_model_precision": train["base_model_precision"],
        "text_encoder_1_precision": train["text_encoder_1_precision"],
        "gradient_checkpointing": train["gradient_checkpointing"],
        "lora_rank": train["lora_rank"],
        "lora_alpha": train["lora_alpha"],
        "lora_dropout": train["lora_dropout"],
        "lora_format": "comfyui",
        "optimizer": train["optimizer"],
        "learning_rate": train["learning_rate"],
        "train_batch_size": train["train_batch_size"],
        "gradient_accumulation_steps": train["gradient_accumulation_steps"],
        "lr_scheduler": train["lr_scheduler"],
        "lr_warmup_steps": train["lr_warmup_steps"],
        "adam_weight_decay": train["weight_decay"],
        "max_grad_norm": 1.0,
        "max_train_steps": train["max_train_steps"],
        "num_train_epochs": 0,
        "vae_batch_size": 1,
        "checkpoint_step_interval": train["checkpoint_step_interval"],
        "checkpoints_total_limit": train["checkpoints_total_limit"],
        "preserve_data_backend_cache": train["preserve_data_backend_cache"],
        "compress_disk_cache": True,
        "seed": train["seed"],
        "report_to": "none",
        "push_to_hub": False,
        "push_checkpoints_to_hub": False,
        "tracker_project_name": "fl-minimax-music3",
        "tracker_run_name": spec["run_id"],
    }
    if validation:
        config.update({
            "validation_prompt": validation["caption"],
            "validation_lyrics": validation["lyrics"],
            "validation_audio_duration": validation["duration"],
            "validation_seed": validation["seed"],
            "validation_num_inference_steps": validation["inference_steps"],
            "validation_guidance": validation["guidance"],
            "validation_steps": validation["interval"],
            "validation_disable_unconditional": True,
            "num_validation_images": validation["samples"],
        })
    if resume:
        config["resume_from_checkpoint"] = "latest"

    _write_json_atomic(config_dir / "config.json", config)
    _write_json_atomic(config_dir / "multidatabackend.json", data_config)
    _write_json_atomic(run_dir / "resolved" / "config.json", config)
    _write_json_atomic(run_dir / "resolved" / "multidatabackend.json", data_config)
    if validation:
        _write_json_atomic(run_dir / "resolved" / "validation.json", validation)
    return config


def find_latest_adapter(run_dir):
    output = Path(run_dir) / "backend_output"
    if not output.is_dir():
        return None
    candidates = [
        path for path in output.rglob("*.safetensors")
        if "optimizer" not in path.name.lower() and "ema" not in path.name.lower()
    ]
    return max(candidates, key=lambda path: path.stat().st_mtime_ns) if candidates else None


def export_adapter(run_dir, output_name):
    source = find_latest_adapter(run_dir)
    if source is None:
        raise RuntimeError("SimpleTuner completed without producing a LoRA safetensors file")
    target_dir = lora_root() / output_name
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{output_name}.safetensors"
    temporary = target.with_suffix(".safetensors.partial")
    shutil.copy2(source, temporary)
    temporary.replace(target)
    return target
