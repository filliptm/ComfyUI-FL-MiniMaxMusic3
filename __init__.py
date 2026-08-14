from .nodes.dataset import FL_MiniMaxMusic3Dataset
from .nodes.loaders.FL_MiniMaxMusic3Loader import FL_MiniMaxMusic3AudioVAELoader, FL_MiniMaxMusic3Loader
from .nodes.preprocessor import FL_MiniMaxMusic3DatasetPreprocessor
from .nodes.train_config import FL_MiniMaxMusic3TrainConfig, FL_MiniMaxMusic3ValidationConfig
from .nodes.trainer import FL_MiniMaxMusic3LoRATrainer, FL_MiniMaxMusic3TrainingRun
from .training.run_store import mark_stale_runs
from . import routes


mark_stale_runs()


NODE_CLASS_MAPPINGS = {
    "FL_MiniMaxMusic3Loader": FL_MiniMaxMusic3Loader,
    "FL_MiniMaxMusic3AudioVAELoader": FL_MiniMaxMusic3AudioVAELoader,
    "FL_MiniMaxMusic3Dataset": FL_MiniMaxMusic3Dataset,
    "FL_MiniMaxMusic3DatasetPreprocessor": FL_MiniMaxMusic3DatasetPreprocessor,
    "FL_MiniMaxMusic3TrainConfig": FL_MiniMaxMusic3TrainConfig,
    "FL_MiniMaxMusic3ValidationConfig": FL_MiniMaxMusic3ValidationConfig,
    "FL_MiniMaxMusic3TrainingRun": FL_MiniMaxMusic3TrainingRun,
    "FL_MiniMaxMusic3LoRATrainer": FL_MiniMaxMusic3LoRATrainer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FL_MiniMaxMusic3Loader": "FL MiniMax Music 3 Loader",
    "FL_MiniMaxMusic3AudioVAELoader": "FL MiniMax Music 3 Audio VAE Loader",
    "FL_MiniMaxMusic3Dataset": "FL MiniMax Music 3 Dataset",
    "FL_MiniMaxMusic3DatasetPreprocessor": "FL MiniMax Music 3 Dataset Preprocessor",
    "FL_MiniMaxMusic3TrainConfig": "FL MiniMax Music 3 Train Config",
    "FL_MiniMaxMusic3ValidationConfig": "FL MiniMax Music 3 Validation Config",
    "FL_MiniMaxMusic3TrainingRun": "FL MiniMax Music 3 Training Run",
    "FL_MiniMaxMusic3LoRATrainer": "FL MiniMax Music 3 LoRA Trainer",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
