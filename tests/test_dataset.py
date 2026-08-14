import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

from conftest import pack_module


dataset_module = pack_module("training.dataset")
paths_module = pack_module("training.paths")


def write_wav(path, seconds=1, sample_rate=8000):
    frames = b"\0\0" * sample_rate * seconds
    with wave.open(str(path), "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(sample_rate)
        file.writeframes(frames)


class DatasetTests(unittest.TestCase):
    def test_dataset_scan_accepts_instrumental_missing_lyrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "datasets"
            folder = root / "demo"
            folder.mkdir(parents=True)
            write_wav(folder / "song.wav")
            (folder / "song.txt").write_text("lo-fi instrumental beat", encoding="utf-8")
            with mock.patch.object(paths_module, "dataset_root", return_value=root):
                result = dataset_module.scan_dataset("demo", missing_lyrics="instrumental", min_duration=0.5)
            self.assertEqual(len(result["tracks"]), 1)
            self.assertEqual(result["tracks"][0]["lyrics"], "")
            self.assertTrue(result["report"]["warnings"])

    def test_dataset_scan_rejects_missing_caption_and_lyrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "datasets"
            folder = root / "demo"
            folder.mkdir(parents=True)
            write_wav(folder / "song.wav")
            with mock.patch.object(paths_module, "dataset_root", return_value=root):
                result = dataset_module.scan_dataset("demo", missing_lyrics="reject", min_duration=0.5)
            self.assertEqual(result["tracks"], [])
            self.assertEqual(result["report"]["invalid_tracks"], 1)
            self.assertEqual(len(result["report"]["errors"]), 2)

    def test_dataset_path_cannot_escape_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "datasets"
            root.mkdir()
            with mock.patch.object(paths_module, "dataset_root", return_value=root):
                with self.assertRaisesRegex(ValueError, "inside"):
                    paths_module.resolve_dataset_folder("../outside")

    def test_manifest_changes_when_sidecar_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "datasets"
            folder = root / "demo"
            folder.mkdir(parents=True)
            write_wav(folder / "song.wav")
            caption = folder / "song.txt"
            caption.write_text("first", encoding="utf-8")
            with mock.patch.object(paths_module, "dataset_root", return_value=root):
                first = dataset_module.dataset_change_token("demo", True, ".txt", ".lyrics", "instrumental")
                caption.write_text("second caption", encoding="utf-8")
                second = dataset_module.dataset_change_token("demo", True, ".txt", ".lyrics", "instrumental")
            self.assertNotEqual(first, second)
