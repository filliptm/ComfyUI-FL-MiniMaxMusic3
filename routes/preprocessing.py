import asyncio
import json
import os
import threading
from pathlib import Path

from aiohttp import web
from server import PromptServer

from ..preprocessing.environment import environment_status, install_environment
from ..preprocessing.model import ensure_model, model_inventory
from ..preprocessing.run_store import list_runs, load_run, request_stop
from ..preprocessing.source import source_folders
from ..training.dataset import AUDIO_EXTENSIONS
from ..training.paths import contained_path, resolve_output_dataset, source_root


_model_job = {"running": False, "state": "idle", "artifact": None, "value": 0, "max": 0, "message": "", "error": None}
_backend_job = {"running": False, "message": "", "error": None}
_model_stop = threading.Event()


def _atomic_text(path, value):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _dataset_track(dataset, relative_audio):
    root = resolve_output_dataset(dataset)
    if not root.is_dir():
        raise ValueError(f"Dataset does not exist: {dataset}")
    audio = contained_path(root, relative_audio)
    if not audio.is_file() or audio.suffix.lower() not in AUDIO_EXTENSIONS:
        raise ValueError("Audio track does not exist in the selected dataset")
    return root, audio


@PromptServer.instance.routes.get("/fl/minimax-music3/preprocess/status")
async def minimax_music3_preprocess_status(_request):
    inventory, backend = await asyncio.gather(
        asyncio.to_thread(model_inventory),
        asyncio.to_thread(environment_status, False),
    )
    return web.json_response({"model": inventory, "model_job": dict(_model_job), "backend": backend, "backend_job": dict(_backend_job)})


@PromptServer.instance.routes.post("/fl/minimax-music3/preprocess/model/download")
async def minimax_music3_preprocess_model_download(_request):
    if _model_job["running"]:
        return web.json_response({"error": "MOSS-Music is already downloading"}, status=409)
    _model_stop.clear()
    _model_job.update({"running": True, "state": "starting", "artifact": None, "value": 0, "max": 0, "message": "Starting MOSS-Music download", "error": None})

    def callback(event):
        _model_job.update(event)

    async def download():
        try:
            await asyncio.to_thread(ensure_model, "download_if_missing", callback, _model_stop.is_set)
            _model_job.update({"state": "ready", "message": "MOSS-Music is verified"})
        except InterruptedError:
            _model_job.update({"state": "stopped", "message": "MOSS-Music download stopped; Download will resume it", "error": None})
        except (OSError, RuntimeError, ValueError) as error:
            _model_job.update({"state": "error", "error": str(error), "message": "MOSS-Music download failed"})
        finally:
            _model_job["running"] = False

    asyncio.create_task(download())
    return web.json_response({"started": True})


@PromptServer.instance.routes.post("/fl/minimax-music3/preprocess/model/stop")
async def minimax_music3_preprocess_model_stop(_request):
    if not _model_job["running"]:
        return web.json_response({"error": "MOSS-Music is not downloading"}, status=409)
    _model_stop.set()
    _model_job["message"] = "Stopping MOSS-Music download"
    return web.json_response({"stopping": True})


@PromptServer.instance.routes.post("/fl/minimax-music3/preprocess/backend/install")
async def minimax_music3_preprocess_backend_install(_request):
    if _backend_job["running"]:
        return web.json_response({"error": "The MOSS backend is already installing"}, status=409)
    _backend_job.update({"running": True, "message": "Starting MOSS backend installation", "error": None})

    def callback(message):
        _backend_job["message"] = str(message)[-500:]

    async def install():
        try:
            await asyncio.to_thread(install_environment, callback)
            _backend_job["message"] = "MOSS backend installed"
        except (OSError, RuntimeError, ValueError) as error:
            _backend_job["error"] = str(error)
        finally:
            _backend_job["running"] = False

    asyncio.create_task(install())
    return web.json_response({"started": True})


@PromptServer.instance.routes.get("/fl/minimax-music3/preprocess/sources")
async def minimax_music3_preprocess_sources(_request):
    return web.json_response({"root": str(source_root()), "folders": await asyncio.to_thread(source_folders)})


@PromptServer.instance.routes.get("/fl/minimax-music3/preprocess/runs")
async def minimax_music3_preprocess_runs(_request):
    return web.json_response({"runs": await asyncio.to_thread(list_runs)})


@PromptServer.instance.routes.get("/fl/minimax-music3/preprocess/runs/{run_id}")
async def minimax_music3_preprocess_run(request):
    try:
        _run_dir, spec, state = await asyncio.to_thread(load_run, request.match_info["run_id"])
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return web.json_response({"error": str(error)}, status=404)
    return web.json_response({"spec": spec, "state": state})


@PromptServer.instance.routes.post("/fl/minimax-music3/preprocess/runs/{run_id}/stop")
async def minimax_music3_preprocess_stop(request):
    try:
        run_dir, _spec, state = await asyncio.to_thread(load_run, request.match_info["run_id"])
        if state.get("status") not in {"running", "stop_requested"}:
            return web.json_response({"error": f"Run is {state.get('status')}, not running"}, status=409)
        await asyncio.to_thread(request_stop, run_dir)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return web.json_response({"error": str(error)}, status=404)
    return web.json_response({"stopping": True})


@PromptServer.instance.routes.get("/fl/minimax-music3/preprocess/dataset/{dataset}/tracks")
async def minimax_music3_preprocess_tracks(request):
    try:
        root = resolve_output_dataset(request.match_info["dataset"])
        if not root.is_dir():
            raise ValueError("Dataset does not exist")
        tracks = []
        for audio in sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS):
            metadata_path = audio.with_suffix(".music3.json")
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError):
                metadata = None
            tracks.append({
                "audio": audio.relative_to(root).as_posix(),
                "caption": audio.with_suffix(".txt").read_text(encoding="utf-8").strip() if audio.with_suffix(".txt").is_file() else "",
                "lyrics": audio.with_suffix(".lyrics").read_text(encoding="utf-8").strip() if audio.with_suffix(".lyrics").is_file() else "",
                "metadata": metadata,
            })
    except (OSError, ValueError) as error:
        return web.json_response({"error": str(error)}, status=404)
    return web.json_response({"dataset": request.match_info["dataset"], "root": str(root), "tracks": tracks})


@PromptServer.instance.routes.post("/fl/minimax-music3/preprocess/dataset/{dataset}/track")
async def minimax_music3_preprocess_track_update(request):
    try:
        payload = await request.json()
        _root, audio = _dataset_track(request.match_info["dataset"], payload.get("audio", ""))
        caption = str(payload.get("caption", "")).strip()
        lyrics = str(payload.get("lyrics", "")).strip()
        if not caption:
            raise ValueError("Caption cannot be empty")
        await asyncio.to_thread(_atomic_text, audio.with_suffix(".txt"), caption + "\n")
        if lyrics:
            await asyncio.to_thread(_atomic_text, audio.with_suffix(".lyrics"), lyrics + "\n")
        elif audio.with_suffix(".lyrics").exists():
            await asyncio.to_thread(_atomic_text, audio.with_suffix(".lyrics"), "")
        metadata_path = audio.with_suffix(".music3.json")
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata.setdefault("review", {})["status"] = "edited"
            await asyncio.to_thread(_atomic_text, metadata_path, json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        return web.json_response({"error": str(error)}, status=400)
    return web.json_response({"saved": True})


@PromptServer.instance.routes.post("/fl/minimax-music3/preprocess/dataset/{dataset}/track/approve")
async def minimax_music3_preprocess_track_approve(request):
    try:
        payload = await request.json()
        _root, audio = _dataset_track(request.match_info["dataset"], payload.get("audio", ""))
        metadata_path = audio.with_suffix(".music3.json")
        if not metadata_path.is_file():
            raise ValueError("This track has no MOSS metadata to approve")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.setdefault("review", {})["status"] = "approved"
        await asyncio.to_thread(_atomic_text, metadata_path, json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        return web.json_response({"error": str(error)}, status=400)
    return web.json_response({"approved": True})


@PromptServer.instance.routes.get("/fl/minimax-music3/preprocess/dataset/{dataset}/audio")
async def minimax_music3_preprocess_audio(request):
    try:
        _root, audio = _dataset_track(request.match_info["dataset"], request.query.get("path", ""))
    except (OSError, ValueError) as error:
        return web.json_response({"error": str(error)}, status=404)
    return web.FileResponse(audio)
