import hashlib
import logging
import os
import shutil
import threading
import time
from functools import lru_cache
from pathlib import Path

import requests
import torch

import comfy.model_management
import comfy.sd
import comfy.utils
import folder_paths


REPO_ID = "Comfy-Org/MiniMax-Music-3"
REPO_REVISION = "6444666eb6edfb2c7fcab5f8b81da8b84b4b17b6"
DOWNLOAD_CHUNK_SIZE = 4 * 1024 * 1024
STATUS_EVENT = "fl_minimax_music3_loader_status"
ARTIFACTS = (
    {
        "key": "model",
        "label": "Diffusion model",
        "folder": "diffusion_models",
        "subdir": "diffusion_models",
        "filename": "minimax_music3_dit_fp16.safetensors",
        "repo_path": "diffusion_models/minimax_music3_dit_fp16.safetensors",
        "size": 4914197682,
        "sha256": "45494a2b6b69af115902ff28eaf54118d19067aa54da01000f3e3efce7ba0e34",
    },
    {
        "key": "clip",
        "label": "Text encoder",
        "folder": "text_encoders",
        "subdir": "text_encoders",
        "filename": "minimax_music3_text_encoder_pruned_int8_convrot.safetensors",
        "repo_path": "text_encoders/minimax_music3_text_encoder_pruned_int8_convrot.safetensors",
        "size": 9196611886,
        "sha256": "010b7416d2336a08c711bc22ee65849c9623069ddb7d89bec011a75699e52014",
    },
    {
        "key": "vae",
        "label": "DAV VAE",
        "folder": "vae",
        "subdir": "vae",
        "filename": "minimax_music3_dav.safetensors",
        "repo_path": "vae/minimax_music3_dav.safetensors",
        "size": 216696128,
        "sha256": "2a32155b769be01445fcc2a8663b910fc9e1751e18dc1c3ec528064512d9ef0c",
    },
)
FULL_DAV_ARTIFACT = {
    "key": "audio_vae",
    "label": "Full DAV audio VAE",
    "folder": "vae",
    "subdir": "vae",
    "filename": "minimax_music3_dav_full.safetensors",
    "repo_path": "audio_vae/diffusion_pytorch_model.safetensors",
    "repo_id": "SimpleTuner/MiniMax-Music-3-Encoder",
    "revision": "fce0d00b1ae42ee47874babb8c06fb859eb01443",
    "size": 306466152,
    "sha256": "ea6d2458de8d71e3d8b8210362ab31c547ac3c99bafa53ba004f3751acb5428e",
}


logger = logging.getLogger("fl_minimax_music3.loader")
_download_lock = threading.Lock()


def _send_status_event(node_id, payload):
    if node_id is None:
        return
    try:
        from server import PromptServer

        server = PromptServer.instance
        server.send_sync(STATUS_EVENT, {"node": str(node_id), **payload}, server.client_id)
    except Exception:
        logger.debug("MiniMax Music 3 status event failed", exc_info=True)


class _LoaderStatus:
    def __init__(self, node_id):
        self.node_id = node_id
        self.current_artifact = None
        self.last_download_event = 0.0

    def emit(
        self,
        state,
        artifact=None,
        message="",
        value=None,
        maximum=None,
        overall_value=None,
        overall_max=None,
        resumed=False,
        force=True,
    ):
        if artifact is not None:
            self.current_artifact = artifact
        if state == "downloading" and not force:
            now = time.monotonic()
            if now - self.last_download_event < 0.2 and value != maximum:
                return
            self.last_download_event = now

        payload = {
            "state": state,
            "artifact": artifact.get("key", artifact["filename"]) if artifact else None,
            "label": artifact.get("label", artifact["filename"]) if artifact else None,
            "filename": artifact["filename"] if artifact else None,
            "expected_size": artifact["size"] if artifact else None,
            "message": message,
            "resumed": bool(resumed),
        }
        if value is not None:
            payload["value"] = int(value)
        if maximum is not None:
            payload["max"] = int(maximum)
        if overall_value is not None:
            payload["overall_value"] = int(overall_value)
        if overall_max is not None:
            payload["overall_max"] = int(overall_max)
        _send_status_event(self.node_id, payload)


def _file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(DOWNLOAD_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=16)
def _verified_file_stat(path, size, modified_ns, expected_size, expected_sha256):
    return size == expected_size and _file_sha256(Path(path)) == expected_sha256


def _verified_file(path, artifact):
    if not path.is_file():
        return False
    stat = path.stat()
    return _verified_file_stat(
        str(path),
        stat.st_size,
        stat.st_mtime_ns,
        artifact["size"],
        artifact["sha256"],
    )


def _target_path(artifact):
    return Path(folder_paths.models_dir) / artifact["subdir"] / artifact["filename"]


def _candidate_paths(artifact):
    paths = [_target_path(artifact)]
    for relative_path in folder_paths.get_filename_list(artifact["folder"]):
        if Path(relative_path).name != artifact["filename"]:
            continue
        full_path = folder_paths.get_full_path(artifact["folder"], relative_path)
        if full_path is not None:
            paths.append(Path(full_path))

    unique = []
    seen = set()
    for path in paths:
        key = os.path.normcase(os.path.abspath(path))
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def minimax_music3_inventory():
    inventory = []
    for artifact in ARTIFACTS:
        state = "missing"
        available_bytes = 0
        invalid_size = None
        for path in _candidate_paths(artifact):
            if not path.is_file():
                continue
            size = path.stat().st_size
            if size == artifact["size"]:
                state = "present"
                available_bytes = size
                break
            invalid_size = size

        if state != "present":
            temporary_path = Path(f"{_target_path(artifact)}.part")
            if temporary_path.is_file():
                available_bytes = temporary_path.stat().st_size
                state = "partial" if 0 < available_bytes <= artifact["size"] else "invalid_size"
            elif invalid_size is not None:
                available_bytes = invalid_size
                state = "invalid_size"

        inventory.append(
            {
                "key": artifact["key"],
                "label": artifact["label"],
                "filename": artifact["filename"],
                "expected_size": artifact["size"],
                "available_bytes": available_bytes,
                "state": state,
            }
        )

    return {
        "repository": REPO_ID,
        "revision": REPO_REVISION,
        "total_size": sum(artifact["size"] for artifact in ARTIFACTS),
        "artifacts": inventory,
    }


def _find_verified_artifact(artifact, status=None):
    for path in _candidate_paths(artifact):
        if not path.is_file():
            continue
        logger.info("Verifying %s", path.name)
        if status is not None:
            status.emit(
                "verifying",
                artifact,
                f"Verifying {artifact['label']}",
                value=0,
                maximum=artifact["size"],
            )
        if _verified_file(path, artifact):
            if status is not None:
                status.emit(
                    "verified",
                    artifact,
                    f"{artifact['label']} verified",
                    value=artifact["size"],
                    maximum=artifact["size"],
                )
            return path
        logger.warning("Ignoring invalid MiniMax Music 3 model file: %s", path)
        if status is not None:
            status.emit("invalid", artifact, f"{artifact['label']} needs to be downloaded again")
    return None


def _download_url(artifact):
    repo_id = artifact.get("repo_id", REPO_ID)
    revision = artifact.get("revision", REPO_REVISION)
    return f"https://huggingface.co/{repo_id}/resolve/{revision}/{artifact['repo_path']}?download=true"


def _download_artifact(artifact, target, progress, completed_bytes, total_bytes, status=None):
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = Path(f"{target}.part")
    expected_size = artifact["size"]

    if temporary_path.is_file():
        partial_size = temporary_path.stat().st_size
        if partial_size == expected_size:
            if status is not None:
                status.emit(
                    "verifying",
                    artifact,
                    f"Verifying downloaded {artifact['label']}",
                    value=expected_size,
                    maximum=expected_size,
                )
            if _verified_file(temporary_path, artifact):
                temporary_path.replace(target)
                progress.update_absolute(completed_bytes + expected_size, total_bytes)
                if status is not None:
                    status.emit(
                        "verified",
                        artifact,
                        f"{artifact['label']} verified",
                        value=expected_size,
                        maximum=expected_size,
                        overall_value=completed_bytes + expected_size,
                        overall_max=total_bytes,
                    )
                return target
            temporary_path.unlink()
        elif partial_size > expected_size:
            temporary_path.unlink()

    partial_size = temporary_path.stat().st_size if temporary_path.is_file() else 0
    remaining = expected_size - partial_size
    free_space = shutil.disk_usage(target.parent).free
    if free_space < remaining:
        raise RuntimeError(
            f"MiniMax Music 3 requires {remaining / (1024 ** 3):.2f} GiB of additional disk space "
            f"in {target.parent}; {free_space / (1024 ** 3):.2f} GiB is available."
        )

    headers = {"Range": f"bytes={partial_size}-"} if partial_size else {}
    resumed = bool(partial_size)
    logger.info(
        "%s %s (%.2f GiB)",
        "Resuming" if partial_size else "Downloading",
        artifact["filename"],
        expected_size / (1024 ** 3),
    )
    if status is not None:
        status.emit(
            "downloading",
            artifact,
            f"{'Resuming' if resumed else 'Downloading'} {artifact['label']}",
            value=partial_size,
            maximum=expected_size,
            overall_value=completed_bytes + partial_size,
            overall_max=total_bytes,
            resumed=resumed,
        )

    try:
        with requests.get(
            _download_url(artifact),
            headers=headers,
            stream=True,
            timeout=(10, 300),
        ) as response:
            mode = "wb"
            if partial_size and response.status_code == 206:
                content_range = response.headers.get("Content-Range", "")
                if not content_range.startswith(f"bytes {partial_size}-"):
                    raise RuntimeError(
                        f"MiniMax Music 3 received an invalid resume response for {artifact['filename']}."
                    )
                mode = "ab"
            elif partial_size and response.status_code == 200:
                partial_size = 0
                resumed = False
                if status is not None:
                    status.emit(
                        "downloading",
                        artifact,
                        f"Downloading {artifact['label']}",
                        value=0,
                        maximum=expected_size,
                        overall_value=completed_bytes,
                        overall_max=total_bytes,
                    )
            else:
                response.raise_for_status()

            downloaded = partial_size
            progress.update_absolute(completed_bytes + downloaded, total_bytes)
            with temporary_path.open(mode) as file:
                for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                    if not chunk:
                        continue
                    comfy.model_management.throw_exception_if_processing_interrupted()
                    file.write(chunk)
                    downloaded += len(chunk)
                    if downloaded > expected_size:
                        raise RuntimeError(
                            f"MiniMax Music 3 download exceeded the expected size for {artifact['filename']}."
                        )
                    progress.update_absolute(completed_bytes + downloaded, total_bytes)
                    if status is not None:
                        status.emit(
                            "downloading",
                            artifact,
                            f"{'Resuming' if resumed else 'Downloading'} {artifact['label']}",
                            value=downloaded,
                            maximum=expected_size,
                            overall_value=completed_bytes + downloaded,
                            overall_max=total_bytes,
                            resumed=resumed,
                            force=False,
                        )
    except requests.RequestException as error:
        raise RuntimeError(
            f"MiniMax Music 3 download failed for {artifact['filename']}: {error}. "
            "Queue again to resume the partial download."
        ) from error
    except OSError as error:
        raise RuntimeError(
            f"MiniMax Music 3 could not write {target}: {error}."
        ) from error

    actual_size = temporary_path.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(
            f"MiniMax Music 3 downloaded {actual_size} bytes for {artifact['filename']}; "
            f"expected {expected_size}. Queue again to resume."
        )

    logger.info("Verifying %s", artifact["filename"])
    if status is not None:
        status.emit(
            "verifying",
            artifact,
            f"Verifying downloaded {artifact['label']}",
            value=expected_size,
            maximum=expected_size,
            overall_value=completed_bytes + expected_size,
            overall_max=total_bytes,
        )
    if not _verified_file(temporary_path, artifact):
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"MiniMax Music 3 checksum mismatch for {artifact['filename']}. "
            "The invalid download was removed."
        )

    temporary_path.replace(target)
    progress.update_absolute(completed_bytes + expected_size, total_bytes)
    if status is not None:
        status.emit(
            "verified",
            artifact,
            f"{artifact['label']} verified",
            value=expected_size,
            maximum=expected_size,
            overall_value=completed_bytes + expected_size,
            overall_max=total_bytes,
        )
    return target


def _ensure_artifacts(unique_id=None, status=None, artifacts=None):
    artifacts = ARTIFACTS if artifacts is None else artifacts
    total_bytes = sum(artifact["size"] for artifact in artifacts)
    progress = comfy.utils.ProgressBar(total_bytes, node_id=str(unique_id) if unique_id is not None else None)
    status = status or _LoaderStatus(unique_id)
    resolved = []
    completed_bytes = 0
    status.emit("checking", message="Checking MiniMax Music 3 model files")

    with _download_lock:
        for artifact in artifacts:
            comfy.model_management.throw_exception_if_processing_interrupted()
            path = _find_verified_artifact(artifact, status)
            if path is None:
                status.emit("missing", artifact, f"{artifact['label']} is not installed")
                path = _download_artifact(
                    artifact,
                    _target_path(artifact),
                    progress,
                    completed_bytes,
                    total_bytes,
                    status,
                )
            resolved.append(path)
            completed_bytes += artifact["size"]
            progress.update_absolute(completed_bytes, total_bytes)

    status.emit("files_ready", message="All MiniMax Music 3 files are verified")
    return tuple(resolved)


def _load_vae(path):
    state_dict, metadata = comfy.utils.load_torch_file(str(path), return_metadata=True)
    vae = comfy.sd.VAE(sd=state_dict, metadata=metadata)
    vae.throw_exception_if_invalid()
    vae.patcher.cached_patcher_init = (comfy.sd.load_vae_patcher, (str(path), metadata, None))
    return vae


class FL_MiniMaxMusic3Loader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "weight_dtype": (
                    ["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],
                    {
                        "default": "default",
                        "advanced": True,
                        "tooltip": "Controls the diffusion model's in-memory weight precision. Default follows ComfyUI's model policy.",
                    },
                ),
                "clip_device": (
                    ["default", "cpu"],
                    {
                        "default": "default",
                        "advanced": True,
                        "tooltip": "Runs the MiniMax text encoder using ComfyUI's default device policy or entirely on the CPU.",
                    },
                ),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    OUTPUT_TOOLTIPS = (
        "The MiniMax Music 3 diffusion model.",
        "The MiniMax Music 3 text encoder.",
        "The MiniMax Music 3 DAV VAE.",
    )
    FUNCTION = "load_models"
    CATEGORY = "🏵️Fill Nodes/Loaders"
    DESCRIPTION = (
        "Downloads the official ComfyUI MiniMax Music 3 model set when missing, verifies it, "
        "and loads the diffusion model, MiniMax CLIP, and DAV VAE. The first download is 13.34 GiB."
    )
    SEARCH_ALIASES = ["minimax music", "music 3", "music model loader"]

    def load_models(self, unique_id=None, weight_dtype="default", clip_device="default"):
        status = _LoaderStatus(unique_id)
        try:
            if not hasattr(comfy.sd.CLIPType, "MINIMAX"):
                raise RuntimeError(
                    "This ComfyUI version does not support MiniMax Music 3. Update ComfyUI before using this loader."
                )

            model_path, clip_path, vae_path = _ensure_artifacts(unique_id, status)
            model_options = {}
            if weight_dtype == "fp8_e4m3fn":
                model_options["dtype"] = torch.float8_e4m3fn
            elif weight_dtype == "fp8_e4m3fn_fast":
                model_options["dtype"] = torch.float8_e4m3fn
                model_options["fp8_optimizations"] = True
            elif weight_dtype == "fp8_e5m2":
                model_options["dtype"] = torch.float8_e5m2

            clip_options = {}
            if clip_device == "cpu":
                clip_options["load_device"] = clip_options["offload_device"] = torch.device("cpu")

            status.emit("loading", ARTIFACTS[0], "Loading diffusion model")
            model = comfy.sd.load_diffusion_model(str(model_path), model_options=model_options)
            status.emit("ready", ARTIFACTS[0], "Diffusion model loaded")

            status.emit("loading", ARTIFACTS[1], "Loading text encoder")
            clip = comfy.sd.load_clip(
                ckpt_paths=[str(clip_path)],
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
                clip_type=comfy.sd.CLIPType.MINIMAX,
                model_options=clip_options,
            )
            status.emit("ready", ARTIFACTS[1], "Text encoder loaded")

            status.emit("loading", ARTIFACTS[2], "Loading DAV VAE")
            vae = _load_vae(vae_path)
            status.emit("ready", ARTIFACTS[2], "DAV VAE loaded")
            status.emit("complete", message="MiniMax Music 3 is ready")
            return model, clip, vae
        except comfy.model_management.InterruptProcessingException:
            status.emit("interrupted", status.current_artifact, "MiniMax Music 3 loading interrupted")
            raise
        except Exception as error:
            status.emit("error", status.current_artifact, str(error))
            raise


class FL_MiniMaxMusic3AudioVAELoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae",)
    OUTPUT_TOOLTIPS = (
        "The full MiniMax Music 3 DAV with waveform encoding and decoding support.",
    )
    FUNCTION = "load_vae"
    CATEGORY = "ðŸµï¸Fill Nodes/Loaders"
    DESCRIPTION = (
        "Downloads and loads the full MiniMax Music 3 DAV audio VAE. Connect it to VAE Encode Audio "
        "to convert 44.1 kHz mono or stereo audio into native Music 3 latents."
    )
    SEARCH_ALIASES = ["minimax music encoder", "music 3 audio vae", "music audio to latent"]

    def load_vae(self, unique_id=None):
        status = _LoaderStatus(unique_id)
        try:
            (vae_path,) = _ensure_artifacts(unique_id, status, (FULL_DAV_ARTIFACT,))
            status.emit("loading", FULL_DAV_ARTIFACT, "Loading full DAV audio VAE")
            vae = _load_vae(vae_path)
            status.emit("ready", FULL_DAV_ARTIFACT, "Full DAV audio VAE loaded")
            status.emit("complete", message="MiniMax Music 3 audio encoding is ready")
            return (vae,)
        except comfy.model_management.InterruptProcessingException:
            status.emit("interrupted", status.current_artifact, "MiniMax Music 3 VAE loading interrupted")
            raise
        except Exception as error:
            status.emit("error", status.current_artifact, str(error))
            raise
