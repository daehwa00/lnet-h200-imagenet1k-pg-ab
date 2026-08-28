#!/usr/bin/env python3
"""Run the failure-isolated H200 ImageNet-1K baseline campaign."""

from __future__ import annotations

# pyright: reportAny=false, reportExplicitAny=false
# ruff: noqa: FBT003, PLR0911
import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import signal
import statistics
import subprocess
import sys
import time
from collections import Counter, deque
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, cast

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "h200" / "baselines" / "campaign.json"
DEFAULT_WORKER = ROOT / "scripts" / "run_h200_baseline_worker.py"
STATUS_SCHEMA = "lnet.h200.imagenet1k.baseline-queue-status.v1"
RESULT_SUCCESS = {"completed", "done", "success", "pass"}
MODEL_KEYS = (
    "parc_net_xs",
    "parc_net_s",
    "mobilevitv2_050",
    "mobilevitv2_075",
    "mobilevitv2_100",
    "sret_tiny",
    "moganet_xt",
    "uniconvnet_a",
    "convnextv2_atto",
    "efficientmod_xxs",
    "emov2_1m",
    "emov2_2m",
    "mobileone_s0",
    "mobileone_s1",
    "efficientformerv2_s0",
    "swiftformer_xs",
    "fastvit_t8",
    "tinynext_t",
    "tinynext_s",
    "tinynext_m",
)


def _now() -> str:
    return datetime.now(UTC).astimezone().isoformat()


def _json_object(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read JSON object: {path}") from error
    if not isinstance(payload, dict):
        raise TypeError(f"JSON payload must be an object: {path}")
    return cast("dict[str, object]", payload)


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _manifest_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha256(payload: object) -> str:
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _external_source_root(repo: Path) -> Path:
    value = os.environ.get("H200_BASELINE_SOURCE_ROOT")
    return Path(value).expanduser().resolve() if value else repo


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be an object")
    return cast("dict[str, object]", value)


def _integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    return value


def _number(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


@dataclass(frozen=True)
class Model:
    key: str
    display_name: str


@dataclass(frozen=True)
class Campaign:
    campaign_id: str
    models: tuple[Model, ...]
    seeds: tuple[int, ...]
    calibration_seed: int
    calibration_epochs: int
    learning_rates: tuple[float, ...]
    selection_metric: str
    full_epochs: int
    batch_size: int
    dataloader_workers: int
    prefetch_factor: int
    gpu_memory_fraction: float
    default_max_parallel: int
    max_attempts: int
    mps_active_thread_percentage: int
    preflight_parallelism: tuple[int, ...]
    preflight_steps: int
    preflight_trials: int


def load_campaign(path: Path) -> Campaign:
    payload = _json_object(path)
    if payload.get("schema") != "lnet.h200.imagenet1k.baselines.v1":
        raise ValueError("unsupported baseline campaign schema")
    raw_models = payload.get("models")
    if not isinstance(raw_models, list):
        raise TypeError("models must be a list")
    models: list[Model] = []
    for index, raw_model in enumerate(raw_models):
        model = _object(raw_model, f"models[{index}]")
        key = model.get("key")
        display_name = model.get("display_name")
        if not isinstance(key, str) or not isinstance(display_name, str):
            raise TypeError("every model needs string key and display_name fields")
        models.append(Model(key=key, display_name=display_name))
    if tuple(model.key for model in models) != MODEL_KEYS:
        raise ValueError("campaign must contain the exact ordered 20-model baseline set")

    raw_seeds = payload.get("seeds")
    if not isinstance(raw_seeds, list):
        raise TypeError("seeds must be a list")
    seeds = tuple(_integer(seed, "seed") for seed in raw_seeds)
    if seeds != (501, 509, 521):
        raise ValueError("baseline seeds must be exactly 501, 509, and 521")

    calibration = _object(payload.get("calibration"), "calibration")
    raw_learning_rates = calibration.get("learning_rates")
    if not isinstance(raw_learning_rates, list):
        raise TypeError("calibration.learning_rates must be a list")
    learning_rates = tuple(
        _number(learning_rate, "learning rate") for learning_rate in raw_learning_rates
    )
    if learning_rates != (3e-4, 1e-3, 3e-3):
        raise ValueError("calibration LR grid must be 3e-4, 1e-3, and 3e-3")
    if calibration.get("wandb_mode") != "disabled":
        raise ValueError("calibration must disable W&B")

    full = _object(payload.get("full_training"), "full_training")
    worker = _object(payload.get("worker"), "worker")
    queue = _object(payload.get("queue"), "queue")
    raw_parallelism = queue.get("preflight_parallelism")
    if not isinstance(raw_parallelism, list):
        raise TypeError("queue.preflight_parallelism must be a list")
    preflight_parallelism = tuple(
        _integer(value, "preflight parallelism") for value in raw_parallelism
    )
    campaign_id = payload.get("campaign_id")
    selection_metric = calibration.get("selection_metric")
    if not isinstance(campaign_id, str) or not isinstance(selection_metric, str):
        raise TypeError("campaign_id and calibration.selection_metric must be strings")

    result = Campaign(
        campaign_id=campaign_id,
        models=tuple(models),
        seeds=seeds,
        calibration_seed=_integer(calibration.get("seed"), "calibration.seed"),
        calibration_epochs=_integer(calibration.get("epochs"), "calibration.epochs"),
        learning_rates=learning_rates,
        selection_metric=selection_metric,
        full_epochs=_integer(full.get("epochs"), "full_training.epochs"),
        batch_size=_integer(worker.get("batch_size"), "worker.batch_size"),
        dataloader_workers=_integer(
            worker.get("dataloader_workers"),
            "worker.dataloader_workers",
        ),
        prefetch_factor=_integer(worker.get("prefetch_factor"), "worker.prefetch_factor"),
        gpu_memory_fraction=_number(
            worker.get("gpu_memory_fraction"),
            "worker.gpu_memory_fraction",
        ),
        default_max_parallel=_integer(
            queue.get("default_max_parallel"),
            "queue.default_max_parallel",
        ),
        max_attempts=_integer(queue.get("max_attempts"), "queue.max_attempts"),
        mps_active_thread_percentage=_integer(
            queue.get("mps_active_thread_percentage"),
            "queue.mps_active_thread_percentage",
        ),
        preflight_parallelism=preflight_parallelism,
        preflight_steps=_integer(queue.get("preflight_steps"), "queue.preflight_steps"),
        preflight_trials=_integer(queue.get("preflight_trials"), "queue.preflight_trials"),
    )
    if result.calibration_seed != 501 or result.calibration_epochs != 3:
        raise ValueError("calibration must use seed 501 for exactly 3 epochs")
    if result.full_epochs != 100:
        raise ValueError("full training must run for exactly 100 epochs")
    if (
        result.batch_size != 256
        or result.dataloader_workers != 8
        or result.prefetch_factor != 1
        or result.gpu_memory_fraction != 0.22
    ):
        raise ValueError("worker defaults must be batch=256, workers=2, prefetch=1, memory=0.22")
    if result.default_max_parallel != 4 or result.mps_active_thread_percentage != 50:
        raise ValueError("queue defaults must be max_parallel=4 and MPS percentage=50")
    if result.preflight_parallelism != (1, 2, 4):
        raise ValueError("preflight parallelism candidates must be 1, 2, and 4")
    if result.preflight_trials != 3:
        raise ValueError("concurrency preflight must run exactly three trials")
    return result


def _lr_slug(learning_rate: float) -> str:
    return format(learning_rate, ".0e").replace("+", "")


@dataclass(frozen=True)
class Task:
    task_id: str
    phase: str
    model_key: str
    seed: int
    learning_rate: float | None
    epochs: int
    wandb_mode: str
    output_dir: Path
    result_path: Path
    checkpoint_path: Path
    max_steps: int | None = None


def calibration_tasks(campaign: Campaign, root: Path) -> list[Task]:
    tasks: list[Task] = []
    for model in campaign.models:
        for learning_rate in campaign.learning_rates:
            lr_slug = _lr_slug(learning_rate)
            output_dir = root / "calibration" / model.key / f"lr_{lr_slug}"
            tasks.append(
                Task(
                    task_id=f"calibration:{model.key}:{lr_slug}",
                    phase="calibration",
                    model_key=model.key,
                    seed=campaign.calibration_seed,
                    learning_rate=learning_rate,
                    epochs=campaign.calibration_epochs,
                    wandb_mode="disabled",
                    output_dir=output_dir,
                    result_path=output_dir / "result.json",
                    checkpoint_path=output_dir / "checkpoint.pt",
                )
            )
    return tasks


def full_tasks(
    campaign: Campaign,
    root: Path,
    selected_learning_rates: dict[str, float],
) -> list[Task]:
    """Build the fixed-LR tasks in seed-major order."""
    tasks: list[Task] = []
    for seed in campaign.seeds:
        for model in campaign.models:
            output_dir = root / "full" / model.key / f"seed_{seed}"
            tasks.append(
                Task(
                    task_id=f"full:seed{seed}:{model.key}",
                    phase="full",
                    model_key=model.key,
                    seed=seed,
                    learning_rate=selected_learning_rates.get(model.key),
                    epochs=campaign.full_epochs,
                    wandb_mode="online",
                    output_dir=output_dir,
                    result_path=output_dir / "result.json",
                    checkpoint_path=output_dir / "checkpoint.pt",
                )
            )
    return tasks


def preflight_tasks(
    campaign: Campaign,
    root: Path,
    parallelism: int,
    *,
    mps_active: bool = False,
    trial: int = 0,
) -> list[Task]:
    tasks: list[Task] = []
    # Use known-current timm models for concurrency calibration. External
    # compatibility failures must not force the whole campaign to concurrency=1.
    # Replicate one stable workload so 1/2/4-way aggregate throughput is an
    # apples-to-apples concurrency measurement.
    preflight_keys = ("convnextv2_atto",) * 4
    models_by_key = {model.key: model for model in campaign.models}
    percentage = min(100, 200 // parallelism)
    mode = f"mps_pct{percentage}" if mps_active else "cuda_default"
    driver = os.environ.get("H200_GPU_DRIVER_VERSION", "unknown").replace(".", "_")
    execution_mode = f"{mode}_driver{driver}_trial{trial}"
    for slot, model_key in enumerate(preflight_keys[:parallelism]):
        model = models_by_key[model_key]
        output_dir = (
            root / "preflight" / execution_mode / f"parallel_{parallelism}" / f"slot_{slot}"
        )
        tasks.append(
            Task(
                task_id=(f"preflight:{execution_mode}:p{parallelism}:slot{slot}:{model.key}"),
                phase="preflight",
                model_key=model.key,
                seed=campaign.calibration_seed,
                learning_rate=campaign.learning_rates[1],
                epochs=1,
                wandb_mode="disabled",
                output_dir=output_dir,
                result_path=output_dir / "result.json",
                checkpoint_path=output_dir / "checkpoint.pt",
                max_steps=campaign.preflight_steps,
            )
        )
    return tasks


def worker_command(
    task: Task,
    *,
    python: Path,
    worker: Path,
    data_root: Path,
    source_root: Path,
    batch_size: int,
    dataloader_workers: int,
) -> list[str]:
    if task.learning_rate is None:
        raise RuntimeError(f"task has no selected learning rate: {task.task_id}")
    command = [
        str(python),
        "-u",
        str(worker),
        "--phase",
        task.phase,
        "--model-key",
        task.model_key,
        "--seed",
        str(task.seed),
        "--learning-rate",
        str(task.learning_rate),
        "--epochs",
        str(task.epochs),
        "--data-root",
        str(data_root),
        "--source-root",
        str(source_root),
        "--batch-size",
        str(batch_size),
        "--workers",
        str(dataloader_workers),
        "--output-dir",
        str(task.output_dir),
        "--result-path",
        str(task.result_path),
        "--checkpoint-path",
        str(task.checkpoint_path),
        "--wandb-mode",
        task.wandb_mode,
    ]
    if task.checkpoint_path.is_file():
        command.append("--resume")
    if task.max_steps is not None:
        command.extend(("--max-steps", str(task.max_steps)))
    return command


def _result_payload(task: Task) -> dict[str, object] | None:
    if not task.result_path.is_file():
        return None
    try:
        payload = _json_object(task.result_path)
    except (RuntimeError, TypeError):
        return None
    status = payload.get("status")
    if not isinstance(status, str) or status.lower() not in RESULT_SUCCESS:
        return None
    if payload.get("phase") != task.phase or payload.get("model_key") != task.model_key:
        return None
    if payload.get("seed") != task.seed:
        return None
    try:
        learning_rate = _number(payload.get("learning_rate"), "result learning_rate")
    except (TypeError, ValueError):
        return None
    if task.learning_rate is None or not math.isclose(
        learning_rate,
        task.learning_rate,
        rel_tol=1e-12,
        abs_tol=0.0,
    ):
        return None
    if (
        payload.get("completed_epochs") != task.epochs
        or payload.get("requested_epochs") != task.epochs
    ):
        return None
    task_name = (
        f"{task.model_key}__{task.phase}__seed{task.seed}__lr{task.learning_rate:.8g}".replace(
            ".", "p"
        )
    )
    contract_path = task.output_dir / "contracts" / f"{task_name}.json"
    if not contract_path.is_file():
        return None
    try:
        contract = _json_object(contract_path)
    except (RuntimeError, TypeError):
        return None
    if payload.get("contract_sha256") != _json_sha256(contract):
        return None
    if payload.get("task_sha256") != contract.get("task_sha256"):
        return None
    if payload.get("source_digest_sha256") != contract.get("source_digest_sha256"):
        return None
    dataset = contract.get("dataset")
    if not isinstance(dataset, dict) or not dataset.get("identity_sha256"):
        return None
    active_identity = os.environ.get("LNET_DATASET_IDENTITY_SHA256")
    if active_identity and dataset.get("identity_sha256") != active_identity:
        return None
    active_manifest = os.environ.get("LNET_DATASET_MANIFEST_PATH")
    if active_manifest:
        manifest_path = Path(active_manifest)
        if not manifest_path.is_file():
            return None
        if dataset.get("manifest_sha256") != _manifest_sha256(manifest_path):
            return None
    return payload


def _result_metric(payload: dict[str, object], metric: str) -> float | None:
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        return None
    try:
        return _number(metrics.get(metric), f"result metrics.{metric}")
    except (TypeError, ValueError):
        return None


def _needs_telemetry_replay(task: Task) -> bool:
    if task.phase != "full" or _result_payload(task) is None:
        return False
    if task.learning_rate is None:
        return False
    lr_label = f"{task.learning_rate:.8g}".replace(".", "p")
    task_name = f"{task.model_key}__{task.phase}__seed{task.seed}__lr{lr_label}"
    marker = task.output_dir / "telemetry" / f"{task_name}.mirror-complete.json"
    return not marker.is_file()


def _new_job(task: Task) -> dict[str, object]:
    return {
        "task_id": task.task_id,
        "phase": task.phase,
        "model_key": task.model_key,
        "seed": task.seed,
        "learning_rate": task.learning_rate,
        "epochs": task.epochs,
        "status": "BLOCKED_CALIBRATION" if task.learning_rate is None else "QUEUED",
        "attempts": 0,
        "resume_available": task.checkpoint_path.is_file(),
        "result_path": str(task.result_path),
        "checkpoint_path": str(task.checkpoint_path),
        "log_path": str(task.output_dir / "worker.log"),
        "last_exit_code": None,
        "last_error": None,
        "started_at": None,
        "ended_at": None,
    }


def _new_status(
    campaign: Campaign,
    manifest_sha256: str,
    root: Path,
) -> dict[str, object]:
    tasks = [*calibration_tasks(campaign, root), *full_tasks(campaign, root, {})]
    return {
        "schema": STATUS_SCHEMA,
        "campaign_id": campaign.campaign_id,
        "manifest_sha256": manifest_sha256,
        "created_at": _now(),
        "updated_at": _now(),
        "mps": {"active": False, "started_by_queue": False, "reason": "not_started"},
        "selected_learning_rates": {},
        "calibration_selections": {},
        "jobs": {task.task_id: _new_job(task) for task in tasks},
    }


def _jobs(status: dict[str, object]) -> dict[str, dict[str, object]]:
    raw_jobs = status.get("jobs")
    if not isinstance(raw_jobs, dict):
        raise TypeError("queue status has no jobs object")
    for task_id, raw_job in raw_jobs.items():
        if not isinstance(task_id, str) or not isinstance(raw_job, dict):
            raise TypeError("queue status jobs are invalid")
    return cast("dict[str, dict[str, object]]", raw_jobs)


def _load_or_create_status(
    campaign: Campaign,
    manifest_sha256: str,
    root: Path,
) -> dict[str, object]:
    path = root / "queue-status.json"
    if not path.is_file():
        return _new_status(campaign, manifest_sha256, root)
    status = _json_object(path)
    if status.get("schema") != STATUS_SCHEMA:
        raise ValueError("existing queue status has an incompatible schema")
    if status.get("campaign_id") != campaign.campaign_id:
        raise ValueError("existing queue status belongs to another campaign")
    if status.get("manifest_sha256") != manifest_sha256:
        raise ValueError("campaign manifest changed after queue state was created")
    jobs = _jobs(status)
    expected_ids = {
        task.task_id
        for task in [*calibration_tasks(campaign, root), *full_tasks(campaign, root, {})]
    }
    extra_ids = set(jobs) - expected_ids
    if not expected_ids.issubset(jobs) or any(
        not task_id.startswith("preflight:") for task_id in extra_ids
    ):
        raise ValueError("existing queue status task set differs from campaign")
    for job in jobs.values():
        if job.get("status") == "RUNNING":
            job["status"] = "RESUMABLE" if Path(str(job["checkpoint_path"])).is_file() else "QUEUED"
            job["last_error"] = "orchestrator_restart"
        if job.get("status") != "COMPLETED":
            # Attempts are bounded per orchestrator invocation. A later launch
            # may recover after transient network, compiler, or GPU failures.
            job["attempts"] = 0
    return status


def _write_status(root: Path, status: dict[str, object]) -> None:
    status["updated_at"] = _now()
    _atomic_json(root / "queue-status.json", status)


def reconcile_tasks(tasks: list[Task], status: dict[str, object]) -> None:
    jobs = _jobs(status)
    for task in tasks:
        job = jobs[task.task_id]
        job["learning_rate"] = task.learning_rate
        job["resume_available"] = task.checkpoint_path.is_file()
        if task.learning_rate is None:
            job["status"] = "BLOCKED_CALIBRATION"
        elif _result_payload(task) is not None:
            job["status"] = "COMPLETED"
            job["last_exit_code"] = 0
            job["last_error"] = None
        elif job.get("status") == "COMPLETED":
            job["status"] = "RESUMABLE" if task.checkpoint_path.is_file() else "QUEUED"
            job["last_error"] = "completed_result_missing_or_invalid"
        elif task.checkpoint_path.is_file() and job.get("status") != "RUNNING":
            job["status"] = "RESUMABLE"


def select_learning_rates(
    campaign: Campaign,
    tasks: list[Task],
) -> tuple[dict[str, float], dict[str, dict[str, object]]]:
    selected: dict[str, float] = {}
    evidence: dict[str, dict[str, object]] = {}
    for model in campaign.models:
        candidates: list[tuple[float, float]] = []
        for task in tasks:
            if task.model_key != model.key:
                continue
            payload = _result_payload(task)
            if payload is None or task.learning_rate is None:
                continue
            score = _result_metric(payload, campaign.selection_metric)
            if score is not None:
                candidates.append((score, task.learning_rate))
        if not candidates:
            evidence[model.key] = {
                "status": "NO_SUCCESSFUL_CALIBRATION",
                "completed_candidates": 0,
                "expected_candidates": len(campaign.learning_rates),
            }
            continue
        score, learning_rate = max(candidates, key=lambda candidate: (candidate[0], -candidate[1]))
        if len(candidates) != len(campaign.learning_rates):
            evidence[model.key] = {
                "status": "INCOMPLETE_GRID",
                "best_observed_learning_rate": learning_rate,
                "best_observed_score": score,
                "metric": campaign.selection_metric,
                "completed_candidates": len(candidates),
                "expected_candidates": len(campaign.learning_rates),
            }
            continue
        selected[model.key] = learning_rate
        evidence[model.key] = {
            "status": "SELECTED",
            "learning_rate": learning_rate,
            "score": score,
            "metric": campaign.selection_metric,
            "completed_candidates": len(candidates),
            "expected_candidates": len(campaign.learning_rates),
        }
    return selected, evidence


@dataclass
class RunningProcess:
    task: Task
    process: subprocess.Popen[str]
    log_stream: IO[str]


_ACTIVE_PROCESSES: dict[int, RunningProcess] = {}


def _terminate_active_processes() -> None:
    for pid, record in list(_ACTIVE_PROCESSES.items()):
        if record.process.poll() is None:
            try:
                os.killpg(pid, signal.SIGTERM)
                record.process.wait(timeout=10)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                if record.process.poll() is None:
                    with suppress(ProcessLookupError):
                        os.killpg(pid, signal.SIGKILL)
                    with suppress(subprocess.TimeoutExpired):
                        record.process.wait(timeout=5)
        if not record.log_stream.closed:
            record.log_stream.close()
        _ACTIVE_PROCESSES.pop(pid, None)


def _child_environment(
    task: Task,
    *,
    mps_active: bool,
    mps_percentage: int,
    gpu_memory_fraction: float,
) -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment["WANDB_MODE"] = task.wandb_mode
    environment["H200_GPU_MEMORY_FRACTION"] = str(gpu_memory_fraction)
    environment["H200_BASELINE_MPS_ACTIVE"] = "1" if mps_active else "0"
    environment["H200_BASELINE_MPS_ACTIVE_THREAD_PERCENTAGE"] = (
        str(mps_percentage) if mps_active else "0"
    )
    if task.wandb_mode == "disabled":
        environment.pop("WANDB_API_KEY", None)
        environment.pop("H200_BASELINE_RUN_ID", None)
    else:
        runtime_value = environment.get("H200_BASELINE_WANDB_RUNTIME")
        if runtime_value:
            runtime = _json_object(Path(runtime_value))
            runs = _object(runtime.get("runs"), "baseline W&B runs")
            model = _object(runs.get(task.model_key), f"W&B model {task.model_key}")
            seeds = _object(model.get("seeds"), f"W&B seeds {task.model_key}")
            record = _object(seeds.get(str(task.seed)), f"W&B run {task.model_key}/{task.seed}")
            run_id = record.get("id")
            display_name = record.get("display_name")
            tags = record.get("tags")
            if (
                not isinstance(run_id, str)
                or not isinstance(display_name, str)
                or not isinstance(tags, list)
            ):
                raise RuntimeError(f"invalid W&B runtime record for {task.model_key}/{task.seed}")
            environment["H200_BASELINE_RUN_ID"] = run_id
            environment["H200_BASELINE_DISPLAY_NAME"] = display_name
            environment["H200_BASELINE_TAGS_JSON"] = json.dumps(tags, separators=(",", ":"))
    if mps_active:
        environment["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(mps_percentage)
    else:
        environment.pop("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", None)
    return environment


def _launch(
    task: Task,
    *,
    python: Path,
    worker: Path,
    data_root: Path,
    repo: Path,
    mps_active: bool,
    mps_percentage: int,
    batch_size: int,
    dataloader_workers: int,
    gpu_memory_fraction: float,
) -> RunningProcess:
    task.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = task.output_dir / "worker.log"
    log_stream = log_path.open("a")
    launch_event = {"event": "launch", "task_id": task.task_id, "time": _now()}
    log_stream.write(json.dumps(launch_event) + "\n")
    log_stream.flush()
    try:
        process = subprocess.Popen(
            worker_command(
                task,
                python=python,
                worker=worker,
                data_root=data_root,
                source_root=_external_source_root(repo),
                batch_size=batch_size,
                dataloader_workers=dataloader_workers,
            ),
            cwd=repo,
            env=_child_environment(
                task,
                mps_active=mps_active,
                mps_percentage=mps_percentage,
                gpu_memory_fraction=gpu_memory_fraction,
            ),
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    except BaseException:
        log_stream.close()
        raise
    return RunningProcess(task=task, process=process, log_stream=log_stream)


def run_task_pool(
    tasks: list[Task],
    *,
    status: dict[str, object],
    root: Path,
    repo: Path,
    python: Path,
    worker: Path,
    data_root: Path,
    max_parallel: int,
    max_attempts: int,
    mps_active: bool,
    mps_percentage: int,
    batch_size: int,
    dataloader_workers: int,
    gpu_memory_fraction: float,
    poll_seconds: float,
) -> None:
    if max_parallel < 1:
        raise ValueError("max_parallel must be positive")
    reconcile_tasks(tasks, status)
    jobs = _jobs(status)
    pending = deque(
        task
        for task in tasks
        if (jobs[task.task_id].get("status") != "COMPLETED" or _needs_telemetry_replay(task))
        and task.learning_rate is not None
        and (
            _integer(jobs[task.task_id].get("attempts", 0), "job attempts") < max_attempts
            or _needs_telemetry_replay(task)
        )
    )
    running: dict[int, RunningProcess] = {}
    _write_status(root, status)
    while pending or running:
        while pending and len(running) < max_parallel:
            task = pending.popleft()
            job = jobs[task.task_id]
            attempts = _integer(job.get("attempts", 0), "job attempts") + 1
            job.update(
                {
                    "status": "RUNNING",
                    "attempts": attempts,
                    "resume_available": task.checkpoint_path.is_file(),
                    "started_at": _now(),
                    "ended_at": None,
                    "last_exit_code": None,
                    "last_error": None,
                }
            )
            try:
                record = _launch(
                    task,
                    python=python,
                    worker=worker,
                    data_root=data_root,
                    repo=repo,
                    mps_active=mps_active,
                    mps_percentage=mps_percentage,
                    batch_size=batch_size,
                    dataloader_workers=dataloader_workers,
                    gpu_memory_fraction=gpu_memory_fraction,
                )
            except OSError:
                job.update(
                    {
                        "status": "FAILED",
                        "ended_at": _now(),
                        "last_error": "process_launch_failed",
                    }
                )
                if attempts < max_attempts:
                    pending.append(task)
            else:
                running[record.process.pid] = record
                _ACTIVE_PROCESSES[record.process.pid] = record
            _write_status(root, status)

        completed_pids: list[int] = []
        for pid, record in running.items():
            return_code = record.process.poll()
            if return_code is None:
                continue
            record.log_stream.write(
                json.dumps(
                    {
                        "event": "exit",
                        "task_id": record.task.task_id,
                        "return_code": return_code,
                        "time": _now(),
                    }
                )
                + "\n"
            )
            record.log_stream.close()
            job = jobs[record.task.task_id]
            result_valid = _result_payload(record.task) is not None
            if return_code == 0 and result_valid:
                job.update(
                    {
                        "status": "COMPLETED",
                        "last_exit_code": 0,
                        "last_error": None,
                        "ended_at": _now(),
                        "resume_available": record.task.checkpoint_path.is_file(),
                    }
                )
            else:
                error = "worker_nonzero_exit" if return_code != 0 else "result_missing_or_invalid"
                job.update(
                    {
                        "status": "FAILED",
                        "last_exit_code": return_code,
                        "last_error": error,
                        "ended_at": _now(),
                        "resume_available": record.task.checkpoint_path.is_file(),
                    }
                )
                attempts = _integer(job["attempts"], "job attempts")
                if attempts < max_attempts:
                    pending.append(record.task)
            completed_pids.append(pid)
        for pid in completed_pids:
            del running[pid]
            _ACTIVE_PROCESSES.pop(pid, None)
        if completed_pids:
            _write_status(root, status)
        elif running:
            time.sleep(poll_seconds)


@dataclass(frozen=True)
class MpsSession:
    active: bool
    started_by_queue: bool
    executable: str | None
    environment: dict[str, str]
    reason: str


def _mps_query(executable: str, environment: dict[str, str]) -> bool:
    try:
        result = subprocess.run(
            [executable],
            input="get_server_list\n",
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and bool(result.stdout.strip())


def start_mps(root: Path, mode: str) -> MpsSession:
    if mode == "off":
        return MpsSession(False, False, None, {}, "disabled_by_cli")
    executable = shutil.which("nvidia-cuda-mps-control")
    if executable is None:
        return MpsSession(False, False, None, {}, "control_binary_unavailable")
    mps_root = root / "mps"
    pipe_directory = mps_root / "pipe"
    log_directory = mps_root / "log"
    pipe_directory.mkdir(parents=True, exist_ok=True)
    log_directory.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_MPS_PIPE_DIRECTORY": str(pipe_directory),
            "CUDA_MPS_LOG_DIRECTORY": str(log_directory),
        }
    )
    if _mps_query(executable, environment):
        os.environ.update(
            {
                "CUDA_MPS_PIPE_DIRECTORY": str(pipe_directory),
                "CUDA_MPS_LOG_DIRECTORY": str(log_directory),
            }
        )
        return MpsSession(True, False, executable, environment, "already_running")
    try:
        result = subprocess.run(
            [executable, "-d"],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return MpsSession(False, False, executable, environment, "start_failed")
    if result.returncode != 0:
        return MpsSession(False, False, executable, environment, "start_failed")
    os.environ.update(
        {
            "CUDA_MPS_PIPE_DIRECTORY": str(pipe_directory),
            "CUDA_MPS_LOG_DIRECTORY": str(log_directory),
        }
    )
    return MpsSession(True, True, executable, environment, "started")


def stop_mps(session: MpsSession) -> None:
    if not session.active or not session.started_by_queue or session.executable is None:
        return
    try:
        subprocess.run(
            [session.executable],
            input="quit\n",
            cwd=ROOT,
            env=session.environment,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return


def _status_summary(status: dict[str, object]) -> dict[str, object]:
    jobs = _jobs(status)
    counts = Counter(str(job.get("status")) for job in jobs.values())
    return {
        "campaign_id": status.get("campaign_id"),
        "updated_at": status.get("updated_at"),
        "counts": dict(sorted(counts.items())),
        "selected_learning_rates": status.get("selected_learning_rates", {}),
        "mps": status.get("mps"),
    }


def _task_listing(tasks: list[Task]) -> list[dict[str, object]]:
    return [
        {
            "task_id": task.task_id,
            "phase": task.phase,
            "model_key": task.model_key,
            "seed": task.seed,
            "learning_rate": task.learning_rate,
            "epochs": task.epochs,
            "wandb_mode": task.wandb_mode,
            "result_path": str(task.result_path),
            "checkpoint_path": str(task.checkpoint_path),
        }
        for task in tasks
    ]


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("run", "auto-run", "dry-run", "list", "status", "preflight"),
        default="run",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--worker", type=Path, default=DEFAULT_WORKER)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--max-parallel", type=int)
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument("--mps", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--poll-seconds", type=float, default=0.25)
    return parser.parse_args()


def _require_runtime_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    if args.data_root is None:
        raise ValueError("--data-root is required for run, dry-run, and preflight modes")
    worker = args.worker.resolve()
    python = args.python.resolve()
    data_root = args.data_root.resolve()
    if args.mode in {"run", "auto-run", "preflight"}:
        if not worker.is_file():
            raise FileNotFoundError(f"baseline worker is missing: {worker}")
        if not python.is_file():
            raise FileNotFoundError(f"Python executable is missing: {python}")
        if not data_root.is_dir():
            raise FileNotFoundError(f"ImageNet data root is missing: {data_root}")
    return worker, python, data_root


def _dry_run(
    campaign: Campaign,
    root: Path,
    status: dict[str, object],
    *,
    repo: Path,
    worker: Path,
    python: Path,
    data_root: Path,
) -> int:
    calibration = calibration_tasks(campaign, root)
    selected, _evidence = select_learning_rates(campaign, calibration)
    full = full_tasks(campaign, root, selected)
    commands = []
    for task in [*calibration, *full]:
        if task.learning_rate is None:
            continue
        if _jobs(status)[task.task_id].get("status") == "COMPLETED":
            continue
        commands.append(
            {
                "task_id": task.task_id,
                "command": worker_command(
                    task,
                    python=python,
                    worker=worker,
                    data_root=data_root,
                    source_root=_external_source_root(repo),
                    batch_size=campaign.batch_size,
                    dataloader_workers=campaign.dataloader_workers,
                ),
            }
        )
    output = {"commands": commands, "blocked_full_tasks": 60 - len(selected) * 3}
    print(json.dumps(output, indent=2))
    return 0


def _run_preflight(
    campaign: Campaign,
    root: Path,
    status: dict[str, object],
    *,
    repo: Path,
    worker: Path,
    python: Path,
    data_root: Path,
    session: MpsSession,
    max_attempts: int,
    poll_seconds: float,
) -> int:
    results: dict[str, object] = {}
    # CUDA DEFAULT mode permits multiple processes even without MPS. Benchmark
    # the same candidates and let aggregate throughput decide the safe fallback.
    candidates = campaign.preflight_parallelism
    for parallelism in candidates:
        trial_aggregates: list[float] = []
        trial_rows: list[dict[str, object]] = []
        for trial in range(campaign.preflight_trials):
            tasks = preflight_tasks(
                campaign,
                root,
                parallelism,
                mps_active=session.active,
                trial=trial,
            )
            jobs = _jobs(status)
            for task in tasks:
                jobs.setdefault(task.task_id, _new_job(task))
            run_task_pool(
                tasks,
                status=status,
                root=root,
                repo=repo,
                python=python,
                worker=worker,
                data_root=data_root,
                max_parallel=parallelism,
                max_attempts=max_attempts,
                mps_active=session.active,
                mps_percentage=min(100, 200 // parallelism),
                batch_size=campaign.batch_size,
                dataloader_workers=campaign.dataloader_workers,
                gpu_memory_fraction=campaign.gpu_memory_fraction,
                poll_seconds=poll_seconds,
            )
            throughputs: list[float] = []
            for task in tasks:
                result = _result_payload(task)
                if result is None:
                    continue
                throughput = _result_metric(result, "images_per_second")
                if throughput is not None:
                    throughputs.append(throughput)
            aggregate = sum(throughputs) if len(throughputs) == parallelism else None
            if aggregate is not None:
                trial_aggregates.append(aggregate)
            trial_rows.append(
                {
                    "trial": trial,
                    "aggregate_images_per_second": aggregate,
                    "completed_workers": len(throughputs),
                    "expected_workers": parallelism,
                }
            )
        results[str(parallelism)] = {
            "aggregate_images_per_second": (
                statistics.median(trial_aggregates)
                if len(trial_aggregates) == campaign.preflight_trials
                else None
            ),
            "trials": trial_rows,
        }
    valid = [
        (cast("float", row["aggregate_images_per_second"]), int(parallelism))
        for parallelism, raw_row in results.items()
        if isinstance(raw_row, dict)
        and (row := cast("dict[str, object]", raw_row)).get("aggregate_images_per_second")
        is not None
    ]
    selected = max(valid)[1] if valid else 1
    payload = {
        "schema": "lnet.h200.imagenet1k.baseline-preflight.v1",
        "created_at": _now(),
        "mps_active": session.active,
        "candidates": results,
        "selected_max_parallel": selected,
        "skipped_without_mps": [],
    }
    _atomic_json(root / "preflight-result.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if valid else 1


def _run_campaign(
    campaign: Campaign,
    root: Path,
    status: dict[str, object],
    *,
    repo: Path,
    worker: Path,
    python: Path,
    data_root: Path,
    session: MpsSession,
    max_parallel: int,
    max_attempts: int,
    poll_seconds: float,
) -> int:
    active_thread_percentage = min(100, 200 // max_parallel)
    calibration = calibration_tasks(campaign, root)
    run_task_pool(
        calibration,
        status=status,
        root=root,
        repo=repo,
        python=python,
        worker=worker,
        data_root=data_root,
        max_parallel=max_parallel,
        max_attempts=max_attempts,
        mps_active=session.active,
        mps_percentage=active_thread_percentage,
        batch_size=campaign.batch_size,
        dataloader_workers=campaign.dataloader_workers,
        gpu_memory_fraction=campaign.gpu_memory_fraction,
        poll_seconds=poll_seconds,
    )
    selected, evidence = select_learning_rates(campaign, calibration)
    status["selected_learning_rates"] = selected
    status["calibration_selections"] = evidence
    full = full_tasks(campaign, root, selected)
    reconcile_tasks(full, status)
    _write_status(root, status)
    run_task_pool(
        full,
        status=status,
        root=root,
        repo=repo,
        python=python,
        worker=worker,
        data_root=data_root,
        max_parallel=max_parallel,
        max_attempts=max_attempts,
        mps_active=session.active,
        mps_percentage=active_thread_percentage,
        batch_size=campaign.batch_size,
        dataloader_workers=campaign.dataloader_workers,
        gpu_memory_fraction=campaign.gpu_memory_fraction,
        poll_seconds=poll_seconds,
    )
    summary = _status_summary(status)
    print(json.dumps(summary, indent=2, sort_keys=True))
    campaign_tasks = [*calibration, *full]
    jobs = _jobs(status)
    all_completed = all(jobs[task.task_id].get("status") == "COMPLETED" for task in campaign_tasks)
    return 0 if all_completed else 1


def main() -> int:
    def interrupt_queue(_signal_number: int, _frame: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt_queue)
    signal.signal(signal.SIGINT, interrupt_queue)
    args = _arguments()
    campaign = load_campaign(args.manifest.resolve())
    root = args.root.resolve()
    manifest_sha256 = _manifest_sha256(args.manifest.resolve())
    status_path = root / "queue-status.json"

    if args.mode == "list":
        tasks = [*calibration_tasks(campaign, root), *full_tasks(campaign, root, {})]
        print(json.dumps(_task_listing(tasks), indent=2))
        return 0
    if args.mode == "status":
        if not status_path.is_file():
            print(json.dumps({"campaign_id": campaign.campaign_id, "status": "NOT_STARTED"}))
            return 0
        status = _json_object(status_path)
        print(json.dumps(_status_summary(status), indent=2, sort_keys=True))
        return 0

    worker, python, data_root = _require_runtime_paths(args)
    status = _load_or_create_status(campaign, manifest_sha256, root)
    calibration = calibration_tasks(campaign, root)
    reconcile_tasks(calibration, status)
    selected, evidence = select_learning_rates(campaign, calibration)
    status["selected_learning_rates"] = selected
    status["calibration_selections"] = evidence
    reconcile_tasks(full_tasks(campaign, root, selected), status)
    if args.mode == "dry-run":
        return _dry_run(
            campaign,
            root,
            status,
            repo=args.repo.resolve(),
            worker=worker,
            python=python,
            data_root=data_root,
        )

    max_parallel = (
        args.max_parallel if args.max_parallel is not None else campaign.default_max_parallel
    )
    max_attempts = args.max_attempts if args.max_attempts is not None else campaign.max_attempts
    if max_parallel < 1 or max_attempts < 1 or args.poll_seconds <= 0:
        raise ValueError("parallelism, attempts, and poll interval must be positive")
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".queue.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another baseline queue owns this root") from error
        session = start_mps(root, args.mps)
        effective_parallel = (
            max_parallel
            if session.active or args.max_parallel is not None or args.mps == "off"
            else 1
        )
        status["mps"] = {
            "active": session.active,
            "started_by_queue": session.started_by_queue,
            "reason": session.reason,
            "requested_mode": args.mps,
            "requested_max_parallel": max_parallel,
            "effective_max_parallel": effective_parallel,
            "active_thread_percentage": (
                min(100, 200 // effective_parallel) if session.active else None
            ),
        }
        _write_status(root, status)
        try:
            if args.mode == "auto-run":
                preflight_status = _run_preflight(
                    campaign,
                    root,
                    status,
                    repo=args.repo.resolve(),
                    worker=worker,
                    python=python,
                    data_root=data_root,
                    session=session,
                    max_attempts=max_attempts,
                    poll_seconds=args.poll_seconds,
                )
                selected_parallel = 1
                preflight_path = root / "preflight-result.json"
                if preflight_status == 0 and preflight_path.is_file():
                    payload = _json_object(preflight_path)
                    candidate = _integer(
                        payload.get("selected_max_parallel"),
                        "selected max parallel",
                    )
                    if candidate in campaign.preflight_parallelism:
                        selected_parallel = candidate
                status["mps"] = {
                    **_object(status["mps"], "mps status"),
                    "effective_max_parallel": selected_parallel,
                    "active_thread_percentage": (
                        min(100, 200 // selected_parallel) if session.active else None
                    ),
                }
                _write_status(root, status)
                print(f"H200_BASELINE_SELECTED_PARALLELISM={selected_parallel}")
                return _run_campaign(
                    campaign,
                    root,
                    status,
                    repo=args.repo.resolve(),
                    worker=worker,
                    python=python,
                    data_root=data_root,
                    session=session,
                    max_parallel=selected_parallel,
                    max_attempts=max_attempts,
                    poll_seconds=args.poll_seconds,
                )
            if args.mode == "preflight":
                return _run_preflight(
                    campaign,
                    root,
                    status,
                    repo=args.repo.resolve(),
                    worker=worker,
                    python=python,
                    data_root=data_root,
                    session=session,
                    max_attempts=max_attempts,
                    poll_seconds=args.poll_seconds,
                )
            return _run_campaign(
                campaign,
                root,
                status,
                repo=args.repo.resolve(),
                worker=worker,
                python=python,
                data_root=data_root,
                session=session,
                max_parallel=effective_parallel,
                max_attempts=max_attempts,
                poll_seconds=args.poll_seconds,
            )
        finally:
            _terminate_active_processes()
            stop_mps(session)


if __name__ == "__main__":
    raise SystemExit(main())
