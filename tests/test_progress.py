import tempfile
import unittest
from pathlib import Path

from conftest import pack_module


progress_module = pack_module("training.progress")


class ProgressTests(unittest.TestCase):
    def test_progress_parser_reads_step_loss_and_learning_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "trainer.log"
            parser = progress_module.LogProgress(log, 10)
            log.write_text("Training started\nSteps: 10%| 1/10 loss=0.321 lr=5e-05\n", encoding="utf-8")
            update = parser.poll()
            self.assertEqual(update["phase"], "training")
            self.assertEqual(update["current"], 1)
            self.assertEqual(update["total"], 10)
            self.assertEqual(update["metrics"], {"loss": 0.321, "learning_rate": 5e-05})
            self.assertIsNone(parser.poll())

    def test_progress_parser_tracks_cache_phases(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "trainer.log"
            parser = progress_module.LogProgress(log, 5)
            log.write_text("Building VAE cache for audio dataset\n", encoding="utf-8")
            self.assertEqual(parser.poll()["phase"], "caching_audio")
            with log.open("a", encoding="utf-8") as file:
                file.write("Building text embeds cache\n")
            self.assertEqual(parser.poll()["phase"], "caching_text")

    def test_progress_parser_ignores_non_training_progress_bars(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "trainer.log"
            parser = progress_module.LogProgress(log, 2)
            log.write_text("Loading weights: 100%| 399/399 [00:00<00:00]\n", encoding="utf-8")
            self.assertIsNone(parser.poll())
            with log.open("a", encoding="utf-8") as file:
                file.write("Epoch 1/1, Steps: 50%| 1/2 [00:03<00:03, lr=5e-5, step_loss=0.686]\n")
            update = parser.poll()
            self.assertEqual(update["phase"], "training")
            self.assertEqual(update["current"], 1)
            self.assertEqual(update["total"], 2)
