import json
import logging
import subprocess
import time
from pathlib import Path

import comfy.model_management
import comfy.utils

from ..preprocessing.environment import require_environment
from ..preprocessing.materialize import materialize_dataset
from ..preprocessing.model import ensure_model
from ..preprocessing.process import close_worker, launch_worker
from ..preprocessing.run_store import create_run, request_stop, update_state
from ..preprocessing.settings import DEFAULT_SETTINGS, normalize_settings, settings_json
from ..preprocessing.source import discover_sources, source_change_token
from ..training.dataset import scan_dataset
from ..training.paths import preprocess_cache_root, resolve_output_dataset, source_root


logger = logging.getLogger("fl_minimax_music3.preprocessor")
STATUS_EVENT = "fl_minimax_music3_preprocess_status"


def _send_status(node_id, state):
    if node_id is None:
        return
    try:
        from server import PromptServer

        server = PromptServer.instance
        server.send_sync(STATUS_EVENT, {"node": str(node_id), **state}, server.client_id)
    except Exception:
        logger.debug("Could not emit MOSS preprocessing status", exc_info=True)


def _record(run_dir, node_id, **changes):
    state = update_state(run_dir, **changes)
    _send_status(node_id, state)
    return state


class FL_MiniMaxMusic3DatasetPreprocessor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_folder": ("STRING", {"default": str(source_root()), "tooltip": "Absolute path to a local folder containing source audio"}),
                "output_dataset": ("STRING", {"default": "moss_processed"}),
                "settings_json": ("STRING", {"default": settings_json(DEFAULT_SETTINGS), "multiline": True}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("FL_MINIMAX_MUSIC3_DATASET", "STRING")
    RETURN_NAMES = ("dataset", "report")
    FUNCTION = "preprocess"
    CATEGORY = "FL/MiniMax Music 3/Training"
    OUTPUT_NODE = True
    DESCRIPTION = "Uses an isolated, pinned MOSS-Music model to caption, transcribe, segment, and materialize a trainer-ready MiniMax Music 3 dataset."

    @classmethod
    def IS_CHANGED(cls, source_folder, output_dataset, settings_json, **_kwargs):
        try:
            return f"{source_change_token(source_folder)}|{output_dataset}|{settings_json}"
        except (OSError, ValueError):
            return float("NaN")

    def preprocess(self, source_folder, output_dataset, settings_json, unique_id=None):
        settings = normalize_settings(settings_json)
        source_root, sources = discover_sources(source_folder)
        destination = resolve_output_dataset(output_dataset)
        preprocess_cache_root().mkdir(parents=True, exist_ok=True)
        run_dir, spec, _state = create_run({
            "source_folder": source_folder,
            "source_root": str(source_root),
            "output_dataset": output_dataset,
            "output_path": str(destination),
            "settings": settings,
            "sources": sources,
            "cache_root": str(preprocess_cache_root()),
        })

        def install_message(message):
            _record(run_dir, unique_id, status="running", phase="installing_backend", message=str(message)[-500:], error=None)

        last_model_event = {"time": 0.0}

        def model_message(event):
            now = time.monotonic()
            if now - last_model_event["time"] < 0.15 and event.get("value") != event.get("max"):
                return
            last_model_event["time"] = now
            _record(
                run_dir,
                unique_id,
                status="running",
                phase="model_" + event.get("state", "checking"),
                message=event.get("message", "Checking MOSS-Music"),
                artifact=event.get("artifact"),
                bytes_current=int(event.get("value") or 0),
                bytes_total=int(event.get("max") or 0),
                error=None,
            )

        _record(run_dir, unique_id, status="running", phase="preflight", message="Verifying the isolated MOSS backend", error=None)
        try:
            backend = require_environment(settings["backend_policy"], install_message)
            model = ensure_model(
                settings["model_policy"],
                model_message,
                lambda: comfy.model_management.processing_interrupted(),
            )
        except InterruptedError:
            _record(run_dir, unique_id, status="interrupted", phase="interrupted", message="MOSS preprocessing was interrupted")
            comfy.model_management.throw_exception_if_processing_interrupted()
            raise
        except (OSError, RuntimeError, ValueError) as error:
            _record(run_dir, unique_id, status="failed", phase="failed", message="MOSS preflight failed", error=str(error))
            raise
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
        _record(run_dir, unique_id, status="running", phase="launching", message="Launching the isolated MOSS worker")
        worker = launch_worker(run_dir, backend, model["path"])
        progress = comfy.utils.ProgressBar(len(sources), node_id=str(unique_id) if unique_id is not None else None)
        last_signature = None
        try:
            while worker.poll() is None:
                comfy.model_management.throw_exception_if_processing_interrupted()
                try:
                    state = json.loads((Path(run_dir) / "state.json").read_text(encoding="utf-8"))
                except (FileNotFoundError, json.JSONDecodeError):
                    state = None
                if state:
                    signature = (state.get("updated_at"), state.get("phase"), state.get("current"), state.get("message"))
                    if signature != last_signature:
                        _send_status(unique_id, state)
                        progress.update_absolute(int(state.get("current") or 0), max(1, int(state.get("total") or len(sources))))
                        last_signature = signature
                time.sleep(0.25)
        except comfy.model_management.InterruptProcessingException:
            request_stop(run_dir)
            _record(run_dir, unique_id, status="stop_requested", phase="stop_requested", message="Stopping MOSS preprocessing")
            try:
                worker.wait(timeout=30)
            except subprocess.TimeoutExpired:
                worker.terminate()
                try:
                    worker.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    worker.kill()
            raise
        finally:
            close_worker(worker)

        try:
            final_state = json.loads((Path(run_dir) / "state.json").read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            final_state = {}
        if final_state.get("status") == "interrupted":
            _send_status(unique_id, final_state)
            raise RuntimeError("MiniMax Music 3 preprocessing was stopped")
        if worker.returncode != 0 or final_state.get("status") != "completed":
            message = final_state.get("error") or f"MOSS worker exited with code {worker.returncode}"
            _record(run_dir, unique_id, status="failed", phase="failed", message="MOSS preprocessing failed", error=message)
            raise RuntimeError(f"MiniMax Music 3 preprocessing failed: {message}. See {Path(run_dir) / 'logs' / 'worker.log'}")

        try:
            _record(run_dir, unique_id, status="running", phase="materializing", message="Materializing the trainer dataset")
            materialized = materialize_dataset(Path(run_dir) / "dataset", destination, settings["write_policy"])
            dataset = scan_dataset(
                destination.name,
                True,
                ".txt",
                ".lyrics",
                "instrumental",
                1.0,
                settings["max_segment_seconds"] + 0.25,
                3.0,
                "metadata",
                False,
            )
            if not dataset["tracks"]:
                raise RuntimeError("The materialized MOSS dataset contains no valid trainer tracks")
        except (OSError, RuntimeError, ValueError) as error:
            _record(run_dir, unique_id, status="failed", phase="failed", message="Dataset materialization failed", error=str(error))
            raise
        if settings["execution_mode"] == "require_review":
            pending = []
            for metadata_path in (Path(run_dir) / "dataset").glob("*.music3.json"):
                if metadata_path.name == "dataset.music3.json":
                    continue
                target = destination / metadata_path.name
                try:
                    review_status = json.loads(target.read_text(encoding="utf-8")).get("review", {}).get("status")
                except (FileNotFoundError, json.JSONDecodeError):
                    review_status = None
                if review_status not in {"edited", "approved"}:
                    pending.append(metadata_path.stem.replace(".music3", ""))
            if pending:
                _record(
                    run_dir,
                    unique_id,
                    status="awaiting_review",
                    phase="awaiting_review",
                    message=f"Review {len(pending)} generated segments in the preprocessor dashboard, then queue again",
                    pending_review=pending,
                    dataset_path=str(destination),
                )
                raise RuntimeError(f"MOSS dataset requires review before training: {len(pending)} segments are pending")
        dataset["preprocess_run_id"] = spec["run_id"]
        dataset["report"]["preprocessor"] = {
            "run_id": spec["run_id"],
            "source_tracks": len(sources),
            "written": len(materialized["written"]),
            "skipped": len(materialized["skipped"]),
            "worker_warnings": final_state.get("warnings", []),
        }
        state = _record(
            run_dir,
            unique_id,
            status="completed",
            phase="completed",
            message=f"Dataset ready with {len(dataset['tracks'])} training segments",
            current=len(sources),
            total=len(sources),
            dataset_path=str(destination),
            report=dataset["report"],
        )
        progress.update_absolute(len(sources), len(sources))
        return dataset, json.dumps({"state": state, "dataset": dataset["report"], "materialized": materialized}, indent=2, ensure_ascii=False)
