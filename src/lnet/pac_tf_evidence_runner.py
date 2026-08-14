from __future__ import annotations

import fcntl
import json
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import TYPE_CHECKING, cast

from .pac_overnight_io import append_csv_row
from .pac_tf_evidence_eval import run_evidence_job
from .pac_tf_evidence_queue import (
    DEFAULT_ROOT,
    EvidenceJob,
    EvidenceKind,
    full_evidence_config,
    mechanism_checkpoint_config,
    validate_selected_evidence_root,
)

if TYPE_CHECKING:
    from pathlib import Path

    from .pac_types import PACDevice, PACExperimentConfig
    from .tapped_prl_followup_schema import JsonRow


def run_workers(
    root: Path = DEFAULT_ROOT,
    *,
    kind: EvidenceKind,
    device: str = "auto",
    workers: int = 4,
    total_slots: int = 8,
    max_jobs: int | None = None,
) -> None:
    binding = validate_selected_evidence_root(root)
    manifest = root / f"{kind}_manifest.jsonl"
    jobs = _read_jobs(manifest)
    completed = _terminal_keys(root / "queue_state.jsonl")
    pending = [job for job in jobs if job.key not in completed]
    config_factory = (
        mechanism_checkpoint_config if kind == "mechanism_checkpoint" else full_evidence_config
    )
    config = config_factory(
        root,
        model_dim=binding.model_dim,
        modes=binding.modes,
        device=cast("PACDevice", device),
    )
    active: dict[Future[tuple[EvidenceJob, JsonRow | None]], int] = {}
    launched = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while pending or active:
            available = total_slots - sum(active.values())
            while pending and available > 0 and (max_jobs is None or launched < max_jobs):
                index = _next_fit(pending, available)
                if index is None:
                    break
                job = pending.pop(index)
                _event(root, job.key, "running")
                active[pool.submit(_execute, root, config, job)] = job.slots
                available -= job.slots
                launched += 1
            if not active:
                break
            done, _ = wait(tuple(active), return_when=FIRST_COMPLETED)
            for future in done:
                active.pop(future)
                job, row = future.result()
                _event(root, job.key, "done" if row is not None else "failed")
            if max_jobs is not None and launched >= max_jobs and not active:
                break
    write_status(root)


def _execute(
    root: Path, config: PACExperimentConfig, job: EvidenceJob
) -> tuple[EvidenceJob, JsonRow | None]:
    try:
        row = run_evidence_job(root, config, job)
        append_csv_row(root / "results" / f"{job.kind}.csv", row)
    except (
        FileNotFoundError,
        ImportError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        failed: JsonRow = {
            "queue_key": job.key,
            "protocol_id": job.protocol_id,
            "protocol_sha256": job.protocol_sha256,
            "capacity_artifact_sha256": job.capacity_artifact_sha256,
            "selected_model": job.selected_model,
            "selected_model_dim": job.selected_model_dim,
            "selected_modes": job.selected_modes,
            "experiment_group": job.kind,
            "dataset_or_task": job.scope,
            "seed": job.seed,
            "model": job.model,
            "intervention": job.intervention,
            "status": "failed",
            "notes": f"{type(error).__name__}: {error}",
        }
        append_csv_row(root / "results" / f"{job.kind}.csv", failed)
        return job, None
    return job, row


def write_status(root: Path = DEFAULT_ROOT) -> None:
    latest = _latest_statuses(root / "queue_state.jsonl")
    counts: dict[str, dict[str, int]] = {}
    for manifest in sorted(root.glob("*_manifest.jsonl")):
        kind = manifest.name.removesuffix("_manifest.jsonl")
        jobs = _read_jobs(manifest)
        counts[kind] = {
            "manifest": len(jobs),
            "done": sum(latest.get(job.key) == "done" for job in jobs),
            "failed": sum(latest.get(job.key) == "failed" for job in jobs),
            "remaining": sum(latest.get(job.key) not in {"done", "failed"} for job in jobs),
        }
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "STATUS.json").write_text(
        json.dumps(counts, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# PAC-TF evidence queue",
        "",
        "| Kind | Manifest | Done | Failed | Remaining |",
        "|---|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {kind} | {row['manifest']} | {row['done']} | {row['failed']} | {row['remaining']} |"
        for kind, row in counts.items()
    )
    (reports / "STATUS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_jobs(path: Path) -> tuple[EvidenceJob, ...]:
    if not path.exists():
        message = f"manifest does not exist: {path}"
        raise FileNotFoundError(message)
    return tuple(
        EvidenceJob(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _next_fit(jobs: list[EvidenceJob], available: int) -> int | None:
    return next((index for index, job in enumerate(jobs) if job.slots <= available), None)


def _terminal_keys(path: Path) -> set[str]:
    return {
        key for key, status in _latest_statuses(path).items() if status in {"done", "failed"}
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


def _event(root: Path, key: str, status: str) -> None:
    path = root / "queue_state.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(json.dumps({"key": key, "status": status}, sort_keys=True) + "\n")
        handle.flush()
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
