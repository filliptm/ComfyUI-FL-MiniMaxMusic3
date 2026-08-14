import logging
import subprocess
import time
from pathlib import Path

import comfy.model_management

from ..training.config_writer import export_adapter, write_simpletuner_config
from ..training.environment import require_environment
from ..training.process import launch_worker, request_stop, worker_state
from ..training.progress import LogProgress
from ..training.run_store import (
    append_event,
    append_metric,
    create_run,
    list_runs,
    load_run,
    update_state,
    validate_resume,
)


logger = logging.getLogger("fl_minimax_music3.trainer")
STATUS_EVENT = "fl_minimax_music3_training_status"


def _send_status(node_id, state):
    if node_id is None:
        return
    try:
        from server import PromptServer

        PromptServer.instance.send_sync(STATUS_EVENT, {"node": str(node_id), **state}, PromptServer.instance.client_id)
    except Exception:
        logger.debug("Could not emit MiniMax Music 3 training status", exc_info=True)


def _record(run_dir, node_id, **changes):
    state = update_state(run_dir, **changes)
    event = append_event(run_dir, state)
    _send_status(node_id, event)
    return state


def _latest_checkpoint(run_dir):
    output = Path(run_dir) / "backend_output"
    checkpoints = [path for path in output.glob("checkpoint-*") if path.is_dir()] if output.is_dir() else []
    if not checkpoints:
        return None
    return str(max(checkpoints, key=lambda path: path.stat().st_mtime_ns))


def training_run_choices():
    choices = [item["run_id"] for item in list_runs()]
    return choices or ["<no training runs found>"]


class FL_MiniMaxMusic3TrainingRun:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"run_id": (training_run_choices(),)}}

    RETURN_TYPES = ("FL_MINIMAX_MUSIC3_TRAINING_RUN", "STRING")
    RETURN_NAMES = ("training_run", "summary")
    FUNCTION = "load"
    CATEGORY = "FL/MiniMax Music 3/Training"
    DESCRIPTION = "Loads one persisted Music 3 training run for inspection or resume."

    def load(self, run_id):
        if run_id == "<no training runs found>":
            raise ValueError("No saved MiniMax Music 3 training runs were found")
        run_dir, spec, state = load_run(run_id)
        handle = {"run_id": run_id, "run_dir": str(run_dir), "spec": spec, "state": state}
        summary = (
            f"Run: {run_id}\nStatus: {state['status']}\n"
            f"Step: {state.get('current', 0)}/{state.get('total', 0)}\n"
            f"Checkpoint: {state.get('last_checkpoint') or 'none'}\n"
            f"Adapter: {state.get('adapter_path') or 'none'}"
        )
        return handle, summary


class FL_MiniMaxMusic3LoRATrainer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset": ("FL_MINIMAX_MUSIC3_DATASET",),
                "train_config": ("FL_MINIMAX_MUSIC3_TRAIN_CONFIG",),
                "output_name": ("STRING", {"default": "music3_lora"}),
                "backend_policy": (["require_installed", "install_pinned_if_missing"],),
            },
            "optional": {
                "validation_config": ("FL_MINIMAX_MUSIC3_VALIDATION_CONFIG",),
                "resume_run": ("FL_MINIMAX_MUSIC3_TRAINING_RUN",),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("FL_MINIMAX_MUSIC3_TRAINING_RUN", "STRING")
    RETURN_NAMES = ("training_run", "lora_path")
    FUNCTION = "train"
    CATEGORY = "FL/MiniMax Music 3/Training"
    OUTPUT_NODE = True
    DESCRIPTION = "Runs a pinned SimpleTuner Music 3 LoRA job in an isolated environment with persistent progress and resume state."

    def train(self, dataset, train_config, output_name, backend_policy, validation_config=None, resume_run=None, unique_id=None):
        if resume_run:
            run_dir, original_spec, previous_state = load_run(resume_run["run_id"])
            if previous_state["status"] == "completed":
                raise ValueError("Completed training runs cannot be resumed")
            validate_resume(original_spec, dataset, train_config, output_name)
            if _latest_checkpoint(run_dir) is None:
                raise ValueError("The selected training run has no full checkpoint to resume")
            spec = dict(original_spec)
            spec["train_config"] = train_config
            spec["validation_config"] = validation_config
            resume = True
        else:
            run_dir, spec = create_run(dataset, train_config, validation_config, output_name)
            resume = False

        def install_message(message):
            state = _record(
                run_dir,
                unique_id,
                status="running",
                phase="installing_backend",
                message=message[-500:],
                current=0,
                total=train_config["max_train_steps"],
            )
            logger.info("[%s] %s", state["run_id"], message)

        _record(run_dir, unique_id, status="running", phase="preflight", message="Verifying the isolated training backend", error=None)
        backend = require_environment(backend_policy, install_message)
        write_simpletuner_config(run_dir, spec, resume=resume)
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
        _record(run_dir, unique_id, status="running", phase="preflight", message="Launching the SimpleTuner worker")
        worker = launch_worker(run_dir, backend)
        progress = LogProgress(Path(run_dir) / "logs" / "trainer.log", train_config["max_train_steps"])
        last_step = -1
        try:
            while True:
                running = worker.poll() is None
                if running:
                    comfy.model_management.throw_exception_if_processing_interrupted()
                update = progress.poll()
                if update:
                    checkpoint = _latest_checkpoint(run_dir)
                    _record(
                        run_dir,
                        unique_id,
                        status="running",
                        phase=update["phase"],
                        message=update["message"],
                        current=update["current"],
                        total=update["total"],
                        metrics=update["metrics"],
                        last_checkpoint=checkpoint,
                    )
                    if (
                        update["phase"] == "training"
                        and update["metrics"]
                        and 0 < update["current"] <= train_config["max_train_steps"]
                        and update["current"] != last_step
                    ):
                        append_metric(run_dir, {
                            "timestamp": update["timestamp"],
                            "step": update["current"],
                            **update["metrics"],
                        })
                        last_step = update["current"]
                if not running:
                    break
                time.sleep(0.25)
        except comfy.model_management.InterruptProcessingException:
            request_stop(run_dir)
            _record(run_dir, unique_id, status="stop_requested", phase="stop_requested", message="Stopping training and preserving the latest checkpoint")
            try:
                worker.wait(timeout=30)
            except subprocess.TimeoutExpired:
                worker.terminate()
                try:
                    worker.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    worker.kill()
            _record(
                run_dir,
                unique_id,
                status="interrupted",
                phase="interrupted",
                message="Training was interrupted and can be resumed from the latest checkpoint",
                last_checkpoint=_latest_checkpoint(run_dir),
            )
            raise

        final_worker_state = worker_state(run_dir) or {}
        if worker.returncode != 0 or final_worker_state.get("status") != "completed":
            state = _record(
                run_dir,
                unique_id,
                status="failed",
                phase="failed",
                message="SimpleTuner training failed. Open the run log for details.",
                error=f"Worker exit code {worker.returncode}",
                last_checkpoint=_latest_checkpoint(run_dir),
            )
            raise RuntimeError(f"MiniMax Music 3 training failed for {state['run_id']}. See {Path(run_dir) / 'logs' / 'trainer.log'}")

        _record(run_dir, unique_id, status="running", phase="exporting", message="Exporting the ComfyUI LoRA")
        adapter = export_adapter(run_dir, spec["output_name"])
        state = _record(
            run_dir,
            unique_id,
            status="completed",
            phase="completed",
            message="MiniMax Music 3 LoRA training completed",
            current=train_config["max_train_steps"],
            total=train_config["max_train_steps"],
            adapter_path=str(adapter),
            artifact=str(adapter),
            last_checkpoint=_latest_checkpoint(run_dir),
        )
        handle = {"run_id": spec["run_id"], "run_dir": str(run_dir), "spec": spec, "state": state}
        return handle, str(adapter)
