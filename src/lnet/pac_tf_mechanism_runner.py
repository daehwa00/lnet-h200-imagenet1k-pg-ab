from __future__ import annotations

import csv
import fcntl
import json
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, replace
from typing import TYPE_CHECKING

import torch

from .pac_overnight_io import append_csv_row, prepare_overnight_dirs
from .pac_tf_mechanism_eval import run_mechanism_job
from .pac_tf_mechanism_jobs import MECHANISM_TASKS, build_mechanism_jobs
from .pac_tf_mechanism_models import MECHANISM_MODELS, build_mechanism_model
from .pac_tf_mechanism_types import (
    MechanismEvent,
    MechanismJob,
    MechanismQueueConfig,
    MechanismStatus,
)
from .pac_types import PACExperimentConfig

if TYPE_CHECKING:
    from pathlib import Path

    from .tapped_prl_followup_schema import JsonRow

RESULT_FILE = "mechanism_recovery.csv"


def full_config(config: MechanismQueueConfig) -> PACExperimentConfig:
    return PACExperimentConfig(
        sample_count=1024,
        validation_count=256,
        test_count=256,
        sequence_length=64,
        model_dim=64,
        modes=16,
        epochs=60,
        batch_size=64,
        learning_rate=3.0e-3,
        weight_decay=1.0e-4,
        seeds=config.seeds,
        device=config.device,
        output_dir=config.output_root,
    )


def sanity_config(config: MechanismQueueConfig) -> PACExperimentConfig:
    return replace(
        full_config(config),
        sample_count=24,
        validation_count=12,
        test_count=12,
        sequence_length=24,
        model_dim=8,
        modes=2,
        epochs=1,
        batch_size=8,
        seeds=(7,),
    )


def enqueue_jobs(config: MechanismQueueConfig) -> None:
    prepare_overnight_dirs(config.output_root)
    (config.output_root / "logs").mkdir(parents=True, exist_ok=True)
    jobs = build_mechanism_jobs(config)
    (config.output_root / "queue_manifest.jsonl").write_text(
        "".join(json.dumps(asdict(job), sort_keys=True) + "\n" for job in jobs),
        encoding="utf-8",
    )
    experiment_payload = asdict(full_config(config))
    experiment_payload["output_dir"] = str(config.output_root)
    protocol = {
        "causality": "all models are forward-only causal sequence regressors",
        "job_count": len(jobs),
        "models": list(MECHANISM_MODELS),
        "tasks": list(MECHANISM_TASKS),
        "seeds": list(config.seeds),
        "predictive_metrics": ["test_mse", "test_nrmse"],
        "pac_tf_recovery_metrics": [
            "assignment_and_sign_invariant_frequency_mae",
            "damping_correlation",
            "damping_regime_auc",
            "impulse_response_nmse",
        ],
        "experiment_config": experiment_payload,
    }
    (config.output_root / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _event(config.output_root, MechanismEvent("enqueue", "done", f"jobs={len(jobs)}"))


def run_sanity(config: MechanismQueueConfig) -> None:
    smoke = replace(config, seeds=(7,), workers=1, total_slots=2)
    enqueue_jobs(smoke)
    run_workers(smoke, max_jobs=6, experiment_config=sanity_config(smoke))


def run_workers(
    config: MechanismQueueConfig,
    *,
    max_jobs: int | None = None,
    experiment_config: PACExperimentConfig | None = None,
) -> None:
    prepare_overnight_dirs(config.output_root)
    manifest = config.output_root / "queue_manifest.jsonl"
    if not manifest.exists():
        enqueue_jobs(config)
    run_config = experiment_config or full_config(config)
    pending = [job for job in _read_jobs(manifest) if job.key not in _done_keys(config.output_root)]
    _prewarm_cuda(run_config)
    active: dict[Future[tuple[MechanismJob, JsonRow | None]], int] = {}
    launched = 0
    with ThreadPoolExecutor(max_workers=config.workers) as pool:
        while pending or active:
            available = config.total_slots - sum(active.values())
            while pending and available > 0 and (max_jobs is None or launched < max_jobs):
                index = _next_fit_index(pending, available)
                if index is None:
                    break
                job = pending.pop(index)
                _event(config.output_root, MechanismEvent(job.key, "running"))
                active[pool.submit(_execute_job, config.output_root, run_config, job)] = job.slots
                available -= job.slots
                launched += 1
            if not active:
                break
            completed, _ = wait(tuple(active), return_when=FIRST_COMPLETED)
            for future in completed:
                active.pop(future)
                job, row = future.result()
                status: MechanismStatus = "done" if row is not None else "failed"
                _event(config.output_root, MechanismEvent(job.key, status))
            if max_jobs is not None and launched >= max_jobs and not active:
                break
    write_report(config.output_root)


def _execute_job(
    root: Path, config: PACExperimentConfig, job: MechanismJob
) -> tuple[MechanismJob, JsonRow | None]:
    stream = torch.cuda.Stream() if config.device != "cpu" and torch.cuda.is_available() else None
    try:
        if stream is None:
            row = run_mechanism_job(config, job)
        else:
            with torch.cuda.stream(stream):
                row = run_mechanism_job(config, job)
            stream.synchronize()
        append_csv_row(root / "results" / RESULT_FILE, row)
    except (ImportError, RuntimeError, ValueError, OSError, KeyError, TypeError) as error:
        failed: JsonRow = {
            "queue_key": job.key,
            "task": job.task,
            "model": job.model,
            "seed": job.seed,
            "status": "failed",
            "notes": f"{type(error).__name__}: {error}",
        }
        append_csv_row(root / "results" / RESULT_FILE, failed)
        return job, None
    return job, row


def _prewarm_cuda(config: PACExperimentConfig) -> None:
    if config.device == "cpu" or not torch.cuda.is_available():
        return
    model = build_mechanism_model("pac_tf", config).to(device="cuda")
    inputs = torch.zeros(1, min(config.sequence_length, 8), config.raw_input_dim, device="cuda")
    with torch.no_grad():
        model(inputs)
    torch.cuda.synchronize()
    del model, inputs
    torch.cuda.empty_cache()


def write_report(root: Path) -> None:
    result_path = root / "results" / RESULT_FILE
    rows: list[dict[str, str]] = []
    if result_path.exists():
        with result_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    done = sum(row.get("status") == "done" for row in rows)
    failed = sum(row.get("status") == "failed" for row in rows)
    manifest_count = len(_read_jobs(root / "queue_manifest.jsonl"))
    report = (
        "# PAC-TF synthetic mechanism-recovery queue\n\n"
        f"- Manifest jobs: {manifest_count}\n"
        f"- Completed result rows: {done}\n"
        f"- Failed result rows: {failed}\n"
        f"- Remaining: {max(manifest_count - done, 0)}\n"
    )
    (root / "reports" / "STATUS.md").write_text(report, encoding="utf-8")


def _read_jobs(path: Path) -> tuple[MechanismJob, ...]:
    if not path.exists():
        return ()
    return tuple(
        MechanismJob(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _done_keys(root: Path) -> set[str]:
    return {
        key
        for key, status in _latest_statuses(root / "queue_state.jsonl").items()
        if status == "done"
    }


def _latest_statuses(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    latest: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        latest[str(row["key"])] = str(row["status"])
    return latest


def _event(root: Path, event: MechanismEvent) -> None:
    path = root / "queue_state.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(json.dumps(asdict(event), sort_keys=True) + "\n")
        handle.flush()
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _next_fit_index(jobs: list[MechanismJob], available: int) -> int | None:
    for index, job in enumerate(jobs):
        if job.slots <= available:
            return index
    return None
