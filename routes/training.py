import asyncio
from pathlib import Path

from aiohttp import web
from server import PromptServer

from ..training.environment import environment_status, install_environment
from ..training.paths import contained_path
from ..training.process import request_stop
from ..training.run_store import list_runs, load_run, recent_metrics


_backend_install = {"running": False, "message": "", "error": None}


def _validation_files(run_dir):
    result = []
    for path in Path(run_dir).rglob("*.wav"):
        relative = path.relative_to(run_dir).as_posix()
        result.append({"path": relative, "name": path.name, "size": path.stat().st_size, "modified": path.stat().st_mtime})
    return sorted(result, key=lambda item: item["modified"], reverse=True)


@PromptServer.instance.routes.get("/fl/minimax-music3/training/backend")
async def minimax_music3_backend_status(_request):
    status = await asyncio.to_thread(environment_status, False)
    return web.json_response({**status, "install": dict(_backend_install)})


@PromptServer.instance.routes.post("/fl/minimax-music3/training/backend/install")
async def minimax_music3_backend_install(_request):
    if _backend_install["running"]:
        return web.json_response({"error": "The training backend is already installing"}, status=409)
    _backend_install.update({"running": True, "message": "Starting installation", "error": None})

    def callback(message):
        _backend_install["message"] = message[-500:]

    async def install():
        try:
            await asyncio.to_thread(install_environment, callback)
            _backend_install["message"] = "Training backend installed"
        except (OSError, RuntimeError, ValueError) as error:
            _backend_install["error"] = str(error)
        finally:
            _backend_install["running"] = False

    asyncio.create_task(install())
    return web.json_response({"started": True})


@PromptServer.instance.routes.get("/fl/minimax-music3/training/runs")
async def minimax_music3_training_runs(_request):
    return web.json_response({"runs": await asyncio.to_thread(list_runs)})


@PromptServer.instance.routes.get("/fl/minimax-music3/training/runs/{run_id}")
async def minimax_music3_training_run(request):
    try:
        run_dir, spec, state = await asyncio.to_thread(load_run, request.match_info["run_id"])
    except (OSError, ValueError) as error:
        return web.json_response({"error": str(error)}, status=404)
    return web.json_response({
        "spec": spec,
        "state": state,
        "metrics": recent_metrics(run_dir),
        "validation": _validation_files(run_dir),
    })


@PromptServer.instance.routes.post("/fl/minimax-music3/training/runs/{run_id}/stop")
async def minimax_music3_training_stop(request):
    try:
        run_dir, _spec, state = await asyncio.to_thread(load_run, request.match_info["run_id"])
        if state["status"] not in {"running", "stop_requested"}:
            return web.json_response({"error": f"Run is {state['status']}, not running"}, status=409)
        await asyncio.to_thread(request_stop, run_dir)
    except (OSError, ValueError) as error:
        return web.json_response({"error": str(error)}, status=404)
    return web.json_response({"stopping": True, "run_id": request.match_info["run_id"]})


@PromptServer.instance.routes.get("/fl/minimax-music3/training/runs/{run_id}/artifact")
async def minimax_music3_training_artifact(request):
    try:
        run_dir, _spec, _state = await asyncio.to_thread(load_run, request.match_info["run_id"])
        path = contained_path(run_dir, request.query.get("path", ""))
        if not path.is_file() or path.suffix.lower() not in {".wav", ".json", ".log", ".txt"}:
            raise ValueError("Artifact is not an allowed training output")
    except (OSError, ValueError) as error:
        return web.json_response({"error": str(error)}, status=404)
    return web.FileResponse(path)
