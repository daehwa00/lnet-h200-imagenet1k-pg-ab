from __future__ import annotations

import json
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, replace
from typing import TYPE_CHECKING, Final

from .pac_overnight_audit import run_param_audit
from .pac_overnight_io import append_csv_row, prepare_overnight_dirs
from .pac_paper_queue_eval import run_job_row
from .pac_paper_queue_jobs import build_jobs
from .pac_paper_queue_report import write_paper_queue_reports
from .pac_paper_queue_types import JobKind, PaperJob, PaperQueueConfig, PaperQueueEvent
from .pac_types import PACExperimentConfig

if TYPE_CHECKING:
    from pathlib import Path

    from .pac_types import PACDevice
    from .tapped_prl_followup_schema import JsonRow, JsonValue

RESULT_FILES: Final[dict[JobKind, str]] = {
    "param_audit": "param_count_audit.csv",
    "sampling_rate_ood": "sampling_rate_ood.csv",
    "irregular_time_ood": "irregular_time_ood.csv",
    "damping_counterfactual": "damping_counterfactual.csv",
    "expanded_ood": "expanded_ood.csv",
    "role_ablation": "role_ablation_knockouts.csv",
    "low_data_scaling": "low_data_scaling.csv",
    "strong_baselines_synthetic": "strong_baselines_synthetic.csv",
    "strong_baselines_real": "strong_baselines_real.csv",
    "speed_correctness": "speed_correctness.csv",
}


def sanity_config(root: Path, device: PACDevice) -> PACExperimentConfig:
    return PACExperimentConfig(
        sample_count=24,
        validation_count=12,
        test_count=12,
        sequence_length=24,
        model_dim=4,
        modes=2,
        tap_kernel_size=5,
        fir_kernel_size=3,
        epochs=1,
        batch_size=8,
        seeds=(7,),
        device=device,
        output_dir=root,
    )


def full_config(root: Path, device: PACDevice) -> PACExperimentConfig:
    return PACExperimentConfig(2048, 512, 512, 64, device=device, output_dir=root)


def run_sanity(config: PaperQueueConfig) -> None:
    prepare_overnight_dirs(config.output_root)
    tiny = sanity_config(config.output_root, config.device)
    enqueue_jobs(replace(config, preset="smoke", seeds=(7,)))
    run_workers(replace(config, workers=1, total_slots=4), experiment_config=tiny)
    write_paper_queue_reports(config.output_root)
    _event(config.output_root, PaperQueueEvent("sanity", "sanity", "done"))


def enqueue_jobs(config: PaperQueueConfig) -> None:
    prepare_overnight_dirs(config.output_root)
    jobs = build_jobs(config)
    manifest = config.output_root / "queue_manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(asdict(job), sort_keys=True) + "\n" for job in jobs),
        encoding="utf-8",
    )
    _event(config.output_root, PaperQueueEvent("enqueue", "enqueue", "done", f"jobs={len(jobs)}"))


def run_workers(
    config: PaperQueueConfig,
    *,
    max_jobs: int | None = None,
    experiment_config: PACExperimentConfig | None = None,
) -> None:
    prepare_overnight_dirs(config.output_root)
    manifest = config.output_root / "queue_manifest.jsonl"
    if not manifest.exists():
        enqueue_jobs(config)
    run_config = experiment_config or full_config(config.output_root, config.device)
    pending = [job for job in _read_jobs(manifest) if job.key not in _done_keys(config.output_root)]
    launched = 0
    active: dict[Future[tuple[PaperJob, JsonRow | None]], int] = {}
    with ThreadPoolExecutor(max_workers=config.workers) as pool:
        while pending or active:
            available = config.total_slots - sum(active.values())
            while pending and available > 0 and (max_jobs is None or launched < max_jobs):
                index = _next_fit_index(pending, available)
                if index is None:
                    break
                job = pending.pop(index)
                _event(config.output_root, PaperQueueEvent(job.key, job.kind, "running"))
                future = pool.submit(_execute_job, config.output_root, run_config, job)
                active[future] = job.slots
                available -= job.slots
                launched += 1
            if not active:
                break
            done, _ = wait(tuple(active), return_when=FIRST_COMPLETED)
            for future in done:
                active.pop(future)
                job, row = future.result()
                status = "done" if row is not None else "failed"
                _event(config.output_root, PaperQueueEvent(job.key, job.kind, status))
            if max_jobs is not None and launched >= max_jobs and not active:
                break
    write_paper_queue_reports(config.output_root)


def _execute_job(
    root: Path, config: PACExperimentConfig, job: PaperJob
) -> tuple[PaperJob, JsonRow | None]:
    try:
        if job.kind == "param_audit":
            run_param_audit(root, replace(config, raw_input_dim=1), _audit_models())
            return job, {"status": "done"}
        row = run_job_row(root, config, job)
        append_csv_row(_result_path(root, job.kind), row)
    except (RuntimeError, ValueError, OSError, KeyError, TypeError) as error:
        append_csv_row(_result_path(root, job.kind), _failed_row(job, error))
        return job, None
    else:
        return job, row


def _read_jobs(path: Path) -> tuple[PaperJob, ...]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return tuple(_parse_job(json.loads(line)) for line in lines)


def _parse_job(row: JsonRow) -> PaperJob:
    return PaperJob(
        key=str(row["key"]),
        kind=_job_kind(str(row["kind"])),
        seed=int(str(row["seed"])),
        model=str(row["model"]),
        task=str(row["task"]),
        slots=int(str(row["slots"])),
        ratio=_optional_float(row.get("ratio")),
        value=_optional_float(row.get("value")),
        dataset=str(row["dataset"]) if row.get("dataset") is not None else None,
    )


def _job_kind(value: str) -> JobKind:
    match value:
        case (
            "param_audit"
            | "sampling_rate_ood"
            | "irregular_time_ood"
            | "damping_counterfactual"
            | "expanded_ood"
            | "role_ablation"
            | "low_data_scaling"
            | "strong_baselines_synthetic"
            | "strong_baselines_real"
            | "speed_correctness"
        ):
            return value
        case _:
            message = f"unsupported paper job kind: {value}"
            raise KeyError(message)


def _optional_float(value: JsonValue | None) -> float | None:
    return None if value is None else float(str(value))


def _next_fit_index(jobs: list[PaperJob], available_slots: int) -> int | None:
    for index, job in enumerate(jobs):
        if job.slots <= available_slots:
            return index
    return None


def _done_keys(root: Path) -> set[str]:
    path = root / "queue_state.jsonl"
    if not path.exists():
        return set()
    keys: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("status") == "done":
            keys.add(str(row.get("key")))
    return keys


def _event(root: Path, event: PaperQueueEvent) -> None:
    path = root / "queue_state.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(event), sort_keys=True) + "\n")


def _result_path(root: Path, kind: JobKind) -> Path:
    return root / "results" / RESULT_FILES[kind]


def _failed_row(job: PaperJob, error: Exception) -> JsonRow:
    return {
        "experiment_group": job.kind,
        "task": job.task,
        "seed": job.seed,
        "model": job.model,
        "status": "failed",
        "notes": f"{type(error).__name__}: {error}",
    }


def _audit_models() -> tuple[str, ...]:
    return (
        "pac_lite",
        "pac_full",
        "controlled_tapped_prl_only",
        "fixed_prl",
        "gru",
        "lstm",
        "cnn1d",
        "tcn",
        "transformer_tiny",
        "fir_classifier",
    )
