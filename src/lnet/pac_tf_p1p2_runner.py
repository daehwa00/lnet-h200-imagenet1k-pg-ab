from __future__ import annotations

import fcntl
import json
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict
from typing import TYPE_CHECKING, cast

import torch

from .pac_overnight_io import append_csv_row, prepare_overnight_dirs, read_csv
from .pac_tf_p1p2_eval import run_p1p2_job
from .pac_tf_p1p2_jobs import build_p1p2_jobs, write_manifest
from .pac_tf_p1p2_types import EvidencePackage, P1P2Config, P1P2Job

if TYPE_CHECKING:
    from pathlib import Path

    from .tapped_prl_followup_schema import JsonRow

RESULT_FILES = {
    "low_data": "pac_tf_low_data.csv",
    "synthetic_ood": "pac_tf_synthetic_ood.csv",
    "real_diagnostics": "pac_tf_real_diagnostics.csv",
    "real_domain_ood": "pac_tf_real_domain_ood.csv",
    "efficiency": "pac_tf_final_efficiency.csv",
}


def enqueue(config: P1P2Config) -> Path:
    prepare_overnight_dirs(config.output_root)
    path = write_manifest(config)
    _event(config.output_root, "enqueue", "done", f"jobs={len(build_p1p2_jobs(config))}")
    return path


def run_workers(
    config: P1P2Config,
    *,
    max_jobs: int | None = None,
    package: EvidencePackage | None = None,
) -> None:
    expected_jobs = build_p1p2_jobs(config)
    prepare_overnight_dirs(config.output_root)
    manifest = config.output_root / "queue_manifest.jsonl"
    if not manifest.exists():
        enqueue(config)
    manifest_jobs = _read_jobs(manifest)
    if manifest_jobs != expected_jobs:
        message = "P1/P2 workers refused: manifest differs from locked protocol/tuning selection"
        raise ValueError(message)
    pending = [
        job
        for job in manifest_jobs
        if job.key not in _done_keys(config.output_root)
        and (package is None or job.package == package)
    ]
    active: dict[Future[tuple[P1P2Job, dict[str, object] | None]], int] = {}
    launched = 0
    with ThreadPoolExecutor(max_workers=config.workers) as pool:
        while pending or active:
            available = config.total_slots - sum(active.values())
            while pending and available > 0 and (max_jobs is None or launched < max_jobs):
                index = next(
                    (
                        idx
                        for idx, candidate in enumerate(pending)
                        if required_slots(config, candidate) <= available
                    ),
                    None,
                )
                if index is None:
                    break
                job = pending.pop(index)
                _event(config.output_root, job.key, "running")
                reserved_slots = required_slots(config, job)
                active[pool.submit(_execute, config, job)] = reserved_slots
                available -= reserved_slots
                launched += 1
            if not active:
                break
            complete, _ = wait(tuple(active), return_when=FIRST_COMPLETED)
            for future in complete:
                active.pop(future)
                job, row = future.result()
                _event(config.output_root, job.key, "done" if row is not None else "failed")
            if max_jobs is not None and launched >= max_jobs and not active:
                break
    write_report(config.output_root)


def write_report(root: Path) -> Path:
    manifest = root / "queue_manifest.jsonl"
    jobs = _read_jobs(manifest) if manifest.exists() else ()
    statuses = _latest_statuses(root / "queue_state.jsonl")
    reference_models = {job.reference_model for job in jobs if job.reference_model}
    selected_reference = (
        next(iter(reference_models)) if len(reference_models) == 1 else "not-yet-manifested"
    )
    counts = {package: sum(job.package == package for job in jobs) for package in RESULT_FILES}
    done = sum(statuses.get(job.key) == "done" for job in jobs)
    failed = sum(statuses.get(job.key) == "failed" for job in jobs)
    efficiency_rows = read_csv(root / "results" / RESULT_FILES["efficiency"])
    resource_limited = sum(
        row.get("status") == "done" and row.get("outcome_status") == "resource_limit"
        for row in efficiency_rows
    )
    compile_unsupported = sum(
        row.get("status") == "done" and row.get("outcome_status") == "compile_unsupported"
        for row in efficiency_rows
    )
    report = root / "reports" / "pac_tf_p1p2_queue.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "\n".join(
            (
                "# Canonical PAC-TF P1/P2 Evidence Queue",
                "",
                f"- selected PAC-TF reference model: `{selected_reference}`",
                "- protocol: `pac-tf-confirmatory-20260711-v1`",
                f"- total jobs: {len(jobs)}",
                f"- done: {done}",
                f"- failed: {failed}",
                f"- pending: {len(jobs) - done - failed}",
                f"- low-data-only jobs: {counts['low_data']}",
                f"- synthetic-OOD suite jobs: {counts['synthetic_ood']}",
                f"- ratio-1 calibration/real-corruption jobs: {counts['real_diagnostics']}",
                f"- patient-disjoint real-domain OOD jobs: {counts['real_domain_ood']}",
                f"- final-efficiency jobs: {counts['efficiency']}",
                f"- efficiency resource-limited (censored): {resource_limited}",
                f"- efficiency compile-unsupported (censored): {compile_unsupported}",
                "",
                (
                    "UCR diagnostic rows are corruption shifts; MIT-BIH rows are separately "
                    "identified patient-disjoint DS1-to-DS2 domain OOD."
                ),
                (
                    "Efficiency resource limits are terminal `done` observations with null "
                    "latency/throughput, an explicit reason, and estimated or observed memory; "
                    "they are not silently dropped or treated as successful timings."
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def _execute(config: P1P2Config, job: P1P2Job) -> tuple[P1P2Job, dict[str, object] | None]:
    stream = torch.cuda.Stream() if config.device != "cpu" and torch.cuda.is_available() else None
    try:
        if stream is None:
            row = run_p1p2_job(config, job)
        else:
            with torch.cuda.stream(stream):
                row = run_p1p2_job(config, job)
            stream.synchronize()
        _require_done_result(row)
        append_csv_row(
            config.output_root / "results" / RESULT_FILES[job.package],
            cast("JsonRow", row),
        )
    except Exception as error:  # noqa: BLE001 - one failure must not terminate the queue
        failed = asdict(job) | {
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
        }
        append_csv_row(config.output_root / "results" / RESULT_FILES[job.package], failed)
        return job, None
    else:
        return job, row


def _require_done_result(row: dict[str, object]) -> None:
    if row.get("status") != "done":
        message = f"required P1/P2 job returned non-terminal-success status: {row.get('status')}"
        raise RuntimeError(message)


def _read_jobs(path: Path) -> tuple[P1P2Job, ...]:
    return tuple(
        P1P2Job(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def required_slots(config: P1P2Config, job: P1P2Job) -> int:
    # Peak-memory counters are device-global. Every efficiency measurement must
    # therefore own the queue's complete slot budget regardless of runtime.
    return config.total_slots if job.package == "efficiency" else job.slots


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
        if line.strip():
            row = json.loads(line)
            latest[str(row["key"])] = str(row["status"])
    return latest


def _event(root: Path, key: str, status: str, notes: str = "") -> None:
    path = root / "queue_state.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(
            json.dumps({"key": key, "status": status, "notes": notes}, sort_keys=True) + "\n"
        )
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
