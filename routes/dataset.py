import asyncio

from aiohttp import web
from server import PromptServer

from ..nodes.dataset import dataset_folders
from ..training.dataset import scan_dataset
from ..training.paths import dataset_root


@PromptServer.instance.routes.get("/fl/minimax-music3/datasets")
async def minimax_music3_datasets(_request):
    return web.json_response({"root": str(dataset_root()), "folders": dataset_folders()})


@PromptServer.instance.routes.post("/fl/minimax-music3/datasets/scan")
async def minimax_music3_dataset_scan(request):
    try:
        payload = await request.json()
        dataset = await asyncio.to_thread(
            scan_dataset,
            payload.get("dataset_folder"),
            bool(payload.get("recursive", True)),
            payload.get("caption_extension", ".txt"),
            payload.get("lyrics_extension", ".lyrics"),
            payload.get("missing_lyrics", "instrumental"),
            float(payload.get("min_duration", 1.0)),
            float(payload.get("max_duration", 60.0)),
            float(payload.get("duration_interval", 3.0)),
            payload.get("audio_analysis", "metadata"),
            bool(payload.get("include_invalid", False)),
        )
    except (OSError, ValueError, TypeError) as error:
        return web.json_response({"error": str(error)}, status=400)
    return web.json_response(dataset["report"])
