import hashlib
import json
import pathlib
import tempfile
import unittest
from unittest import mock

import requests

from conftest import pack_module


compiler = pack_module("preprocessing.compiler")
materialize = pack_module("preprocessing.materialize")
model = pack_module("preprocessing.model")
moss_worker = pack_module("preprocessing.moss_worker")
segmenter = pack_module("preprocessing.segmenter")
settings = pack_module("preprocessing.settings")
paths = pack_module("training.paths")


class FakeResponse:
    def __init__(self, data, status_code=200, headers=None):
        self.data = data
        self.status_code = status_code
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def iter_content(self, _chunk_size):
        yield self.data


class PreprocessingTests(unittest.TestCase):
    def test_json_generation_repairs_truncated_output_with_larger_deterministic_budget(self):
        engine = object.__new__(moss_worker.MossEngine)
        engine.settings = {"temperature": 0.2, "max_new_tokens": 1024}
        engine.generate = mock.Mock(side_effect=['{"caption":"cut off"', '{"caption":"repaired"}'])
        parsed, raw = engine.generate_json(None, "full schema", "compact schema")
        self.assertEqual(parsed, {"caption": "repaired"})
        self.assertEqual(raw, '{"caption":"repaired"}')
        self.assertEqual(engine.generate.call_args_list[1].kwargs, {"temperature": 0.0, "max_new_tokens": 2048})

    def test_json_generation_uses_compact_final_retry(self):
        engine = object.__new__(moss_worker.MossEngine)
        engine.settings = {"temperature": 0.2, "max_new_tokens": 1024}
        engine.generate = mock.Mock(side_effect=["bad", "still bad", '{"caption":"compact"}'])
        parsed, _raw = engine.generate_json(None, "full schema", "compact schema")
        self.assertEqual(parsed, {"caption": "compact"})
        self.assertEqual(engine.generate.call_args_list[2].args[1], "compact schema")

    def test_json_extraction_accepts_trailing_model_commentary(self):
        self.assertEqual(moss_worker._extract_json('{"caption":"valid"}\nextra text'), {"caption": "valid"})

    def test_absolute_source_folder_is_read_in_place(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(paths.resolve_source_folder(directory), pathlib.Path(directory).resolve())

    def test_relative_source_folder_remains_managed_for_compatibility(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            source = root / "library"
            source.mkdir()
            with mock.patch.object(paths, "source_root", return_value=root):
                self.assertEqual(paths.resolve_source_folder("library"), source.resolve())

    def test_settings_validate_segment_order(self):
        with self.assertRaisesRegex(ValueError, "minimum <= target <= maximum"):
            settings.normalize_settings({"min_segment_seconds": 40, "target_segment_seconds": 20, "max_segment_seconds": 60})

    def test_structure_boundaries_drive_segments(self):
        sections = [
            {"label": "Intro", "start_seconds": 0, "end_seconds": 20},
            {"label": "Verse", "start_seconds": 20, "end_seconds": 55},
            {"label": "Chorus", "start_seconds": 55, "end_seconds": 90},
        ]
        result = segmenter.plan_segments(90, sections, minimum=8, target=55, maximum=60)
        self.assertEqual([(item["start"], item["end"]) for item in result], [(0, 55), (55, 90)])
        self.assertEqual(result[0]["labels"], ["Intro", "Verse"])

    def test_oversized_section_is_split_under_maximum(self):
        result = segmenter.plan_segments(125, [{"label": "Jam", "start": 0, "end": 125}], maximum=60)
        self.assertEqual(len(result), 3)
        self.assertTrue(all(item["duration"] <= 60 for item in result))

    def test_caption_and_lyrics_compile_to_trainer_text(self):
        caption = compiler.compile_caption({
            "genres": ["funky house"],
            "moods": ["energetic"],
            "bpm": 125,
            "meter": "4/4",
            "instruments": ["electric bass", "disco strings"],
        })
        lyrics = compiler.compile_lyrics({
            "instrumental": False,
            "sections": [{"label": "pre-chorus", "start_seconds": 10, "end_seconds": 20, "lines": ["Lift it up"]}],
        })
        self.assertIn("125 BPM", caption)
        self.assertIn("electric bass", caption)
        self.assertEqual(lyrics, "[Pre Chorus]\nLift it up")
        self.assertEqual(compiler.compile_lyrics({"instrumental": True}), "[Inst]")

    def test_lyrics_normalization_reconciles_instrumental_analysis(self):
        lyrics = {
            "instrumental": False,
            "language": "none",
            "sections": [{"label": "inst1", "lines": [{"lyrics": "one"}]}],
        }
        normalized = compiler.normalize_lyrics(lyrics, {"vocals": {"present": False}})
        self.assertEqual(normalized, {"instrumental": True, "language": "none", "sections": []})
        self.assertEqual(compiler.compile_lyrics(normalized), "[Inst]")

    def test_dictionary_lyric_lines_are_unwrapped(self):
        lyrics = {"instrumental": False, "sections": [{"label": "Verse", "lines": [{"lyrics": "A real line"}]}]}
        self.assertEqual(compiler.compile_lyrics(lyrics), "[Verse]\nA real line")

    def test_model_download_resumes_and_verifies(self):
        data = b"moss-model-test-data"
        artifact = {"path": "model.bin", "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        manifest = {"repo_id": "example/model", "revision": "revision", "files": [artifact]}
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            pathlib.Path(f"{root / 'model.bin'}.part").write_bytes(data[:5])
            response = FakeResponse(data[5:], 206, {"Content-Range": f"bytes 5-{len(data) - 1}/{len(data)}"})
            with (
                mock.patch.object(model, "moss_model_root", return_value=root),
                mock.patch.object(model.requests, "get", return_value=response) as request,
            ):
                model._download_artifact(manifest, artifact)
            self.assertEqual((root / "model.bin").read_bytes(), data)
            self.assertEqual(request.call_args.kwargs["headers"], {"Range": "bytes=5-"})

    def test_model_download_bad_checksum_is_not_promoted(self):
        data = b"bad-data"
        artifact = {"path": "model.bin", "size": len(data), "sha256": hashlib.sha256(b"expected").hexdigest()}
        manifest = {"repo_id": "example/model", "revision": "revision", "files": [artifact]}
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            with (
                mock.patch.object(model, "moss_model_root", return_value=root),
                mock.patch.object(model.requests, "get", return_value=FakeResponse(data)),
            ):
                with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                    model._download_artifact(manifest, artifact)
            self.assertFalse((root / "model.bin").exists())
            self.assertFalse(pathlib.Path(f"{root / 'model.bin'}.part").exists())

    def test_complete_partial_is_verified_without_network(self):
        data = b"complete-partial"
        artifact = {"path": "model.bin", "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        manifest = {"repo_id": "example/model", "revision": "revision", "files": [artifact]}
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            pathlib.Path(f"{root / 'model.bin'}.part").write_bytes(data)
            with (
                mock.patch.object(model, "moss_model_root", return_value=root),
                mock.patch.object(model.requests, "get") as request,
            ):
                model._download_artifact(manifest, artifact)
            request.assert_not_called()
            self.assertEqual((root / "model.bin").read_bytes(), data)

    def test_download_stop_keeps_resumable_partial(self):
        data = b"interruptible-download"
        artifact = {"path": "model.bin", "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        manifest = {"repo_id": "example/model", "revision": "revision", "files": [artifact]}
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            with (
                mock.patch.object(model, "moss_model_root", return_value=root),
                mock.patch.object(model.requests, "get", return_value=FakeResponse(data)),
            ):
                with self.assertRaisesRegex(InterruptedError, "stopped"):
                    model._download_artifact(manifest, artifact, stop=lambda: True)
            self.assertFalse((root / "model.bin").exists())
            self.assertTrue(pathlib.Path(f"{root / 'model.bin'}.part").exists())

    def test_materialize_preserves_human_files_and_replaces_generated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            staging = root / "staging"
            target = root / "dataset"
            staging.mkdir()
            target.mkdir()
            for base in (staging, target):
                (base / "track.flac").write_bytes(b"new" if base == staging else b"old")
                (base / "track.txt").write_text("new" if base == staging else "old", encoding="utf-8")
            (staging / "track.music3.json").write_text(json.dumps({"schema_version": 2, "provenance": {"generator": "FL MiniMax Music 3 Dataset Preprocessor"}}), encoding="utf-8")
            first = materialize.materialize_dataset(staging, target, "fill_missing")
            self.assertEqual((target / "track.txt").read_text(encoding="utf-8"), "old")
            self.assertIn("track.txt", first["skipped"])
            (target / "track.music3.json").write_text(json.dumps({"provenance": {"generator": "FL MiniMax Music 3 Dataset Preprocessor"}}), encoding="utf-8")
            materialize.materialize_dataset(staging, target, "replace_generated")
            self.assertEqual((target / "track.txt").read_text(encoding="utf-8"), "new")
            self.assertEqual(json.loads((target / "track.music3.json").read_text(encoding="utf-8"))["schema_version"], 2)

    def test_replace_generated_preserves_approved_sidecars(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            staging = root / "staging"
            target = root / "dataset"
            staging.mkdir()
            target.mkdir()
            (staging / "track.txt").write_text("new", encoding="utf-8")
            (target / "track.txt").write_text("approved", encoding="utf-8")
            (target / "track.music3.json").write_text(json.dumps({
                "provenance": {"generator": "FL MiniMax Music 3 Dataset Preprocessor"},
                "review": {"status": "approved"},
            }), encoding="utf-8")
            materialize.materialize_dataset(staging, target, "replace_generated")
            self.assertEqual((target / "track.txt").read_text(encoding="utf-8"), "approved")


if __name__ == "__main__":
    unittest.main()
