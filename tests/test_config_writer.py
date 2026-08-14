import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from conftest import pack_module


config_writer = pack_module("training.config_writer")


def training_spec(root, steps=10):
    return {
        "run_id": "test-run",
        "output_name": "demo",
        "dataset": {
            "root": str(root / "dataset"),
            "manifest_hash": "abc",
            "settings": {"caption_extension": ".txt", "lyrics_extension": ".lyrics", "duration_interval": 3.0, "max_duration": 60.0},
        },
        "train_config": {
            "mixed_precision": "bf16", "base_model_precision": "int8-quanto", "text_encoder_1_precision": "int8-quanto",
            "gradient_checkpointing": True, "lora_rank": 64, "lora_alpha": 64, "lora_dropout": 0.0,
            "optimizer": "adamw_bf16", "learning_rate": 5e-5, "train_batch_size": 1, "gradient_accumulation_steps": 4,
            "lr_scheduler": "cosine", "lr_warmup_steps": 50, "weight_decay": 0.01, "max_train_steps": steps,
            "checkpoint_step_interval": 5, "checkpoints_total_limit": 2, "preserve_data_backend_cache": True, "seed": 42,
        },
        "validation_config": {"caption": "funky house", "lyrics": "", "duration": 10, "seed": 42, "inference_steps": 4, "guidance": 1.7, "interval": 5, "samples": 1},
    }


class ConfigWriterTests(unittest.TestCase):
    def test_simpletuner_config_matches_music3_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(config_writer, "dataset_cache_root", return_value=root / "cache"), mock.patch.object(config_writer, "model_cache_root", return_value=root / "models"), mock.patch.object(config_writer, "load_backend_manifest", return_value={"pretrained_model": "MiniMaxAI/MiniMax-Music3", "pretrained_vae": "SimpleTuner/MiniMax-Music-3-Encoder"}):
                config = config_writer.write_simpletuner_config(root / "run", training_spec(root))
            self.assertEqual(config["model_family"], "minimaxmusic")
            self.assertEqual(config["model_flavour"], "music3")
            self.assertEqual(config["lora_format"], "comfyui")
            self.assertEqual(config["adam_weight_decay"], 0.01)
            self.assertNotIn("weight_decay", config)
            self.assertEqual(config["report_to"], "none")
            self.assertFalse(config["push_to_hub"])
            data = json.loads((root / "run/config/job/multidatabackend.json").read_text(encoding="utf-8"))
            self.assertEqual(data[0]["dataset_type"], "audio")
            self.assertEqual(data[0]["audio"]["lyrics_filename_format"], "{filename}.lyrics")
            self.assertEqual(data[0]["audio"]["channels"], 1)
            self.assertEqual(data[0]["audio"]["truncation_mode"], "beginning")
            self.assertEqual(data[1]["dataset_type"], "text_embeds")

    def test_resume_sets_full_checkpoint_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = training_spec(root, steps=2)
            spec["validation_config"] = None
            with mock.patch.object(config_writer, "dataset_cache_root", return_value=root / "cache"), mock.patch.object(config_writer, "model_cache_root", return_value=root / "models"), mock.patch.object(config_writer, "load_backend_manifest", return_value={"pretrained_model": "model", "pretrained_vae": "vae"}):
                config = config_writer.write_simpletuner_config(root / "run", spec, resume=True)
            self.assertEqual(config["resume_from_checkpoint"], "latest")
