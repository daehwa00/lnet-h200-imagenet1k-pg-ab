from __future__ import annotations

import fcntl
import json
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, replace
from typing import TYPE_CHECKING

from .pac_interpretability_eval import InterpretabilityRows, run_interpretability_job
from .pac_interpretability_jobs import build_interpretability_jobs
from .pac_interpretability_report import RESULT_FILES, write_interpretability_report
from .pac_interpretability_types import (
    InterpretabilityEvent,
    InterpretabilityJob,
    InterpretabilityQueueConfig,
    InterpretabilityStatus,
)
from .pac_overnight_io import append_csv_row, prepare_overnight_dirs
from .pac_types import PACExperimentConfig

if TYPE_CHECKING:
    from pathlib import Path

    from .tapped_prl_followup_schema import JsonRow


def sanity_config(config: InterpretabilityQueueConfig) -> PACExperimentConfig:
    return PACExperimentConfig(
        24,
        12,
        12,
        24,
        raw_input_dim=2,
        output_dim=2,
        model_dim=4,
        modes=2,
        tap_kernel_size=5,
        fir_kernel_size=3,
        epochs=1,
        batch_size=8,
        seeds=(7,),
        device=config.device,
        output_dir=config.output_root,
    )


def full_config(config: InterpretabilityQueueConfig) -> PACExperimentConfig:
    return PACExperimentConfig(
        1024,
        256,
        256,
        64,
        device=config.device,
        output_dir=config.output_root,
    )


def run_sanity(config: InterpretabilityQueueConfig) -> None:
    prepare_overnight_dirs(config.output_root)
    enqueue_jobs(replace(config, preset="smoke", seeds=(7,)))
    run_workers(
        replace(config, preset="smoke", workers=1, total_slots=2),
        max_jobs=4,
        experiment_config=sanity_config(config),
    )
    _event(config.output_root, InterpretabilityEvent("sanity", "sanity", "done"))


def enqueue_jobs(config: InterpretabilityQueueConfig) -> None:
    prepare_overnight_dirs(config.output_root)
    jobs = build_interpretability_jobs(config)
    manifest = config.output_root / "queue_manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(asdict(job), sort_keys=True) + "\n" for job in jobs),
        encoding="utf-8",
    )
    _event(
        config.output_root,
        InterpretabilityEvent("enqueue", "enqueue", "done", f"jobs={len(jobs)}"),
    )


def run_workers(
    config: InterpretabilityQueueConfig,
    *,
    max_jobs: int | None = None,
    experiment_config: PACExperimentConfig | None = None,
) -> None:
    prepare_overnight_dirs(config.output_root)
    manifest = config.output_root / "queue_manifest.jsonl"
    if not manifest.exists():
        enqueue_jobs(config)
    pending = [job for job in _read_jobs(manifest) if job.key not in _done_keys(config.output_root)]
    active: dict[Future[tuple[InterpretabilityJob, InterpretabilityRows | None]], int] = {}
    launched = 0
    run_config = experiment_config or full_config(config)
    with ThreadPoolExecutor(max_workers=config.workers) as pool:
        while pending or active:
            launched = _launch_ready(config, run_config, pool, pending, active, launched, max_jobs)
            if not active:
                break
            completed, _ = wait(tuple(active), return_when=FIRST_COMPLETED)
            for future in completed:
                active.pop(future)
                job, rows = future.result()
                _event(
                    config.output_root,
                    InterpretabilityEvent(job.key, job.package, _status(rows)),
                )
            if max_jobs is not None and launched >= max_jobs and not active:
                break
    write_interpretability_report(config.output_root)


def _launch_ready(
    config: InterpretabilityQueueConfig,
    run_config: PACExperimentConfig,
    pool: ThreadPoolExecutor,
    pending: list[InterpretabilityJob],
    active: dict[Future[tuple[InterpretabilityJob, InterpretabilityRows | None]], int],
    launched: int,
    max_jobs: int | None,
) -> int:
    available = config.total_slots - sum(active.values())
    while pending and available > 0 and (max_jobs is None or launched < max_jobs):
        index = _next_fit_index(pending, available)
        if index is None:
            break
        job = pending.pop(index)
        _event(config.output_root, InterpretabilityEvent(job.key, job.package, "running"))
        active[pool.submit(_execute_job, config.output_root, run_config, job)] = job.slots
        available -= job.slots
        launched += 1
    return launched


def _execute_job(
    root: Path, config: PACExperimentConfig, job: InterpretabilityJob
) -> tuple[InterpretabilityJob, InterpretabilityRows | None]:
    try:
        rows = run_interpretability_job(config, job)
        _write_rows(root, rows)
    except (ImportError, RuntimeError, ValueError, OSError, KeyError, TypeError) as error:
        append_csv_row(root / "results" / _failure_file(job), _failed_row(job, error))
        return job, None
    return job, rows


def _write_rows(root: Path, rows: InterpretabilityRows) -> None:
    for name, filename in RESULT_FILES.items():
        for row in getattr(rows, name):
            append_csv_row(root / "results" / filename, row)


def _read_jobs(path: Path) -> tuple[InterpretabilityJob, ...]:
    return tuple(
        InterpretabilityJob(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _done_keys(root: Path) -> set[str]:
    return {
        key
        for key, status in _latest_statuses(root / "queue_state.jsonl").items()
        if status == "done"
    }


def _event(root: Path, event: InterpretabilityEvent) -> None:
    path = root / "queue_state.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(json.dumps(asdict(event), sort_keys=True) + "\n")
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _latest_statuses(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    latest: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = row.get("key")
        status = row.get("status")
        if isinstance(key, str) and isinstance(status, str):
            latest[key] = status
    return latest


def _next_fit_index(jobs: list[InterpretabilityJob], available: int) -> int | None:
    for index, job in enumerate(jobs):
        if job.slots <= available:
            return index
    return None


def _status(rows: InterpretabilityRows | None) -> InterpretabilityStatus:
    return "done" if rows is not None else "failed"


def _failure_file(job: InterpretabilityJob) -> str:
    match job.package:
        case "synthetic_mechanism":
            return RESULT_FILES["synthetic_performance"]
        case "real_modal":
            return RESULT_FILES["real_performance"]


def _failed_row(job: InterpretabilityJob, error: Exception) -> JsonRow:
    return {
        "package": job.package,
        "queue_key": job.key,
        "task": job.task,
        "dataset_or_task": job.task,
        "seed": job.seed,
        "model": job.model,
        "status": "failed",
        "notes": f"{type(error).__name__}: {error}",
    }
