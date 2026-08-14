import importlib
import json
import unittest
from pathlib import Path

from conftest import PACKAGE


class PackTests(unittest.TestCase):
    def test_pack_registers_expected_nodes(self):
        package = importlib.import_module(PACKAGE)
        self.assertEqual(set(package.NODE_CLASS_MAPPINGS), {
            "FL_MiniMaxMusic3Loader",
            "FL_MiniMaxMusic3AudioVAELoader",
            "FL_MiniMaxMusic3Dataset",
            "FL_MiniMaxMusic3DatasetPreprocessor",
            "FL_MiniMaxMusic3TrainConfig",
            "FL_MiniMaxMusic3ValidationConfig",
            "FL_MiniMaxMusic3TrainingRun",
            "FL_MiniMaxMusic3LoRATrainer",
        })
        self.assertEqual(package.WEB_DIRECTORY, "./web")

    def test_example_workflow_serializes_seed_controls(self):
        workflow_path = Path(__file__).parents[1] / "example_workflows" / "MiniMax Music 3 LoRA Training.json"
        workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
        nodes = {node["type"]: node for node in workflow["nodes"]}
        self.assertEqual(nodes["FL_MiniMaxMusic3TrainConfig"]["widgets_values"][15], "fixed")
        self.assertEqual(nodes["FL_MiniMaxMusic3ValidationConfig"]["widgets_values"][4], "fixed")
