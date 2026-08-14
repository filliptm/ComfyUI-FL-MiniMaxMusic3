import hashlib
import importlib.util
import pathlib
import tempfile
import unittest
from unittest import mock

import requests


MODULE_PATH = (
    pathlib.Path(__file__).parents[1]
    / "nodes"
    / "loaders"
    / "FL_MiniMaxMusic3Loader.py"
)
SPEC = importlib.util.spec_from_file_location("fl_minimax_music3_loader_tests", MODULE_PATH)
loader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(loader)


def artifact(data, filename="model.safetensors"):
    return {
        "key": "model",
        "label": "Diffusion model",
        "folder": "diffusion_models",
        "subdir": "diffusion_models",
        "filename": filename,
        "repo_path": f"diffusion_models/{filename}",
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


class FakeProgress:
    def __init__(self, total=0, node_id=None):
        self.total = total
        self.node_id = node_id
        self.values = []

    def update_absolute(self, value, total=None, preview=None):
        self.values.append(value)


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

    def iter_content(self, chunk_size):
        return (
            self.data[index : index + chunk_size]
            for index in range(0, len(self.data), chunk_size)
        )


class MiniMaxMusic3LoaderTests(unittest.TestCase):
    def tearDown(self):
        loader._verified_file_stat.cache_clear()

    def test_missing_artifact_downloads_verifies_and_promotes(self):
        data = b"minimax-music-model"
        spec = artifact(data)
        with tempfile.TemporaryDirectory() as directory:
            target = pathlib.Path(directory) / spec["filename"]
            progress = FakeProgress(len(data))
            with mock.patch.object(
                loader.requests,
                "get",
                return_value=FakeResponse(data),
            ) as request:
                result = loader._download_artifact(spec, target, progress, 0, len(data))

            self.assertEqual(result, target)
            self.assertEqual(target.read_bytes(), data)
            self.assertFalse(pathlib.Path(f"{target}.part").exists())
            self.assertEqual(progress.values[-1], len(data))
            self.assertIn(loader.REPO_REVISION, request.call_args.args[0])

    def test_verified_artifact_skips_network(self):
        data = b"already-installed"
        spec = artifact(data)
        with tempfile.TemporaryDirectory() as directory:
            target = pathlib.Path(directory) / spec["filename"]
            target.write_bytes(data)
            with (
                mock.patch.object(loader, "ARTIFACTS", (spec,)),
                mock.patch.object(loader, "_candidate_paths", return_value=[target]),
                mock.patch.object(loader.comfy.utils, "ProgressBar", FakeProgress),
                mock.patch.object(loader.requests, "get") as request,
            ):
                result = loader._ensure_artifacts("node-1")

            self.assertEqual(result, (target,))
            request.assert_not_called()

    def test_artifact_can_select_its_own_repository_and_revision(self):
        spec = artifact(b"full-dav")
        spec.update(
            {
                "repo_id": "SimpleTuner/MiniMax-Music-3-Encoder",
                "revision": "encoder-revision",
                "repo_path": "audio_vae/model.safetensors",
            }
        )

        self.assertEqual(
            loader._download_url(spec),
            "https://huggingface.co/SimpleTuner/MiniMax-Music-3-Encoder/resolve/encoder-revision/audio_vae/model.safetensors?download=true",
        )

    def test_partial_download_resumes(self):
        data = b"resumable-model-data"
        partial_size = 7
        spec = artifact(data)
        with tempfile.TemporaryDirectory() as directory:
            target = pathlib.Path(directory) / spec["filename"]
            temporary = pathlib.Path(f"{target}.part")
            temporary.write_bytes(data[:partial_size])
            response = FakeResponse(
                data[partial_size:],
                status_code=206,
                headers={"Content-Range": f"bytes {partial_size}-{len(data) - 1}/{len(data)}"},
            )
            with mock.patch.object(loader.requests, "get", return_value=response) as request:
                loader._download_artifact(spec, target, FakeProgress(len(data)), 0, len(data))

            self.assertEqual(target.read_bytes(), data)
            self.assertEqual(request.call_args.kwargs["headers"], {"Range": f"bytes={partial_size}-"})

    def test_download_emits_artifact_progress_and_verification_states(self):
        data = b"download-status-model"
        spec = artifact(data)
        events = []
        with tempfile.TemporaryDirectory() as directory:
            target = pathlib.Path(directory) / spec["filename"]
            status = loader._LoaderStatus("node-status")
            with (
                mock.patch.object(loader.requests, "get", return_value=FakeResponse(data)),
                mock.patch.object(
                    loader,
                    "_send_status_event",
                    side_effect=lambda node_id, payload: events.append((node_id, payload)),
                ),
            ):
                loader._download_artifact(spec, target, FakeProgress(len(data)), 0, len(data), status)

        self.assertEqual(events[0][0], "node-status")
        self.assertEqual(events[0][1]["state"], "downloading")
        self.assertEqual(events[0][1]["artifact"], "model")
        self.assertEqual(events[0][1]["value"], 0)
        self.assertEqual(events[0][1]["max"], len(data))
        self.assertEqual(events[-2][1]["state"], "verifying")
        self.assertEqual(events[-1][1]["state"], "verified")

    def test_inventory_distinguishes_present_partial_and_invalid_files(self):
        data = b"inventory-model"
        spec = artifact(data)
        with tempfile.TemporaryDirectory() as directory:
            target = pathlib.Path(directory) / spec["filename"]
            with (
                mock.patch.object(loader, "ARTIFACTS", (spec,)),
                mock.patch.object(loader, "_candidate_paths", return_value=[target]),
                mock.patch.object(loader, "_target_path", return_value=target),
            ):
                missing = loader.minimax_music3_inventory()["artifacts"][0]
                pathlib.Path(f"{target}.part").write_bytes(data[:5])
                partial = loader.minimax_music3_inventory()["artifacts"][0]
                pathlib.Path(f"{target}.part").unlink()
                target.write_bytes(b"wrong-size")
                invalid = loader.minimax_music3_inventory()["artifacts"][0]
                target.write_bytes(data)
                present = loader.minimax_music3_inventory()["artifacts"][0]

        self.assertEqual(missing["state"], "missing")
        self.assertEqual(partial["state"], "partial")
        self.assertEqual(partial["available_bytes"], 5)
        self.assertEqual(invalid["state"], "invalid_size")
        self.assertEqual(present["state"], "present")

    def test_server_ignoring_range_restarts_partial(self):
        data = b"complete-model"
        spec = artifact(data)
        with tempfile.TemporaryDirectory() as directory:
            target = pathlib.Path(directory) / spec["filename"]
            pathlib.Path(f"{target}.part").write_bytes(b"partial")
            with mock.patch.object(loader.requests, "get", return_value=FakeResponse(data)):
                loader._download_artifact(spec, target, FakeProgress(len(data)), 0, len(data))

            self.assertEqual(target.read_bytes(), data)

    def test_incomplete_download_stays_partial(self):
        data = b"complete-model"
        spec = artifact(data)
        with tempfile.TemporaryDirectory() as directory:
            target = pathlib.Path(directory) / spec["filename"]
            with mock.patch.object(loader.requests, "get", return_value=FakeResponse(data[:5])):
                with self.assertRaisesRegex(RuntimeError, "Queue again to resume"):
                    loader._download_artifact(spec, target, FakeProgress(len(data)), 0, len(data))

            self.assertFalse(target.exists())
            self.assertEqual(pathlib.Path(f"{target}.part").read_bytes(), data[:5])

    def test_bad_checksum_is_not_installed(self):
        data = b"invalid-model"
        spec = artifact(b"expected-model")
        spec["size"] = len(data)
        with tempfile.TemporaryDirectory() as directory:
            target = pathlib.Path(directory) / spec["filename"]
            with mock.patch.object(loader.requests, "get", return_value=FakeResponse(data)):
                with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                    loader._download_artifact(spec, target, FakeProgress(len(data)), 0, len(data))

            self.assertFalse(target.exists())
            self.assertFalse(pathlib.Path(f"{target}.part").exists())

    def test_loader_uses_minimax_clip_type_and_returns_three_outputs(self):
        paths = tuple(pathlib.Path(name) for name in ("model.safetensors", "clip.safetensors", "vae.safetensors"))
        model = object()
        clip = object()
        vae = object()
        with (
            mock.patch.object(loader, "_ensure_artifacts", return_value=paths),
            mock.patch.object(loader.comfy.sd, "load_diffusion_model", return_value=model) as load_model,
            mock.patch.object(loader.comfy.sd, "load_clip", return_value=clip) as load_clip,
            mock.patch.object(loader, "_load_vae", return_value=vae) as load_vae,
        ):
            result = loader.FL_MiniMaxMusic3Loader().load_models("node-2")

        self.assertEqual(result, (model, clip, vae))
        load_model.assert_called_once_with("model.safetensors", model_options={})
        self.assertEqual(load_clip.call_args.kwargs["ckpt_paths"], ["clip.safetensors"])
        self.assertEqual(load_clip.call_args.kwargs["clip_type"], loader.comfy.sd.CLIPType.MINIMAX)
        self.assertEqual(load_clip.call_args.kwargs["model_options"], {})
        load_vae.assert_called_once_with(paths[2])

    def test_loader_maps_core_weight_dtype_options(self):
        paths = tuple(pathlib.Path(name) for name in ("model.safetensors", "clip.safetensors", "vae.safetensors"))
        cases = {
            "default": {},
            "fp8_e4m3fn": {"dtype": loader.torch.float8_e4m3fn},
            "fp8_e4m3fn_fast": {"dtype": loader.torch.float8_e4m3fn, "fp8_optimizations": True},
            "fp8_e5m2": {"dtype": loader.torch.float8_e5m2},
        }

        for weight_dtype, expected_options in cases.items():
            with self.subTest(weight_dtype=weight_dtype):
                with (
                    mock.patch.object(loader, "_ensure_artifacts", return_value=paths),
                    mock.patch.object(loader.comfy.sd, "load_diffusion_model", return_value=object()) as load_model,
                    mock.patch.object(loader.comfy.sd, "load_clip", return_value=object()),
                    mock.patch.object(loader, "_load_vae", return_value=object()),
                ):
                    loader.FL_MiniMaxMusic3Loader().load_models(weight_dtype=weight_dtype)

                load_model.assert_called_once_with("model.safetensors", model_options=expected_options)

    def test_loader_can_keep_clip_on_cpu(self):
        paths = tuple(pathlib.Path(name) for name in ("model.safetensors", "clip.safetensors", "vae.safetensors"))
        with (
            mock.patch.object(loader, "_ensure_artifacts", return_value=paths),
            mock.patch.object(loader.comfy.sd, "load_diffusion_model", return_value=object()),
            mock.patch.object(loader.comfy.sd, "load_clip", return_value=object()) as load_clip,
            mock.patch.object(loader, "_load_vae", return_value=object()),
        ):
            loader.FL_MiniMaxMusic3Loader().load_models(clip_device="cpu")

        self.assertEqual(
            load_clip.call_args.kwargs["model_options"],
            {"load_device": loader.torch.device("cpu"), "offload_device": loader.torch.device("cpu")},
        )

    def test_full_dav_loader_returns_encoding_vae(self):
        vae_path = pathlib.Path("minimax_music3_dav_full.safetensors")
        vae = object()
        with (
            mock.patch.object(loader, "_ensure_artifacts", return_value=(vae_path,)) as ensure,
            mock.patch.object(loader, "_load_vae", return_value=vae) as load_vae,
        ):
            result = loader.FL_MiniMaxMusic3AudioVAELoader().load_vae("node-encoder")

        self.assertEqual(result, (vae,))
        ensure.assert_called_once_with("node-encoder", mock.ANY, (loader.FULL_DAV_ARTIFACT,))
        load_vae.assert_called_once_with(vae_path)

    def test_loader_emits_loading_sequence_and_completion(self):
        paths = tuple(pathlib.Path(name) for name in ("model.safetensors", "clip.safetensors", "vae.safetensors"))
        events = []
        with (
            mock.patch.object(loader, "_ensure_artifacts", return_value=paths),
            mock.patch.object(loader.comfy.sd, "load_diffusion_model", return_value=object()),
            mock.patch.object(loader.comfy.sd, "load_clip", return_value=object()),
            mock.patch.object(loader, "_load_vae", return_value=object()),
            mock.patch.object(
                loader,
                "_send_status_event",
                side_effect=lambda node_id, payload: events.append((node_id, payload)),
            ),
        ):
            loader.FL_MiniMaxMusic3Loader().load_models("node-load")

        states = [(payload["state"], payload["artifact"]) for _, payload in events]
        self.assertEqual(
            states,
            [
                ("loading", "model"),
                ("ready", "model"),
                ("loading", "clip"),
                ("ready", "clip"),
                ("loading", "vae"),
                ("ready", "vae"),
                ("complete", None),
            ],
        )

    def test_loader_emits_error_without_hiding_the_exception(self):
        events = []
        with (
            mock.patch.object(loader, "_ensure_artifacts", side_effect=RuntimeError("disk unavailable")),
            mock.patch.object(
                loader,
                "_send_status_event",
                side_effect=lambda node_id, payload: events.append((node_id, payload)),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "disk unavailable"):
                loader.FL_MiniMaxMusic3Loader().load_models("node-error")

        self.assertEqual(events[-1][0], "node-error")
        self.assertEqual(events[-1][1]["state"], "error")
        self.assertEqual(events[-1][1]["message"], "disk unavailable")


if __name__ == "__main__":
    unittest.main()
