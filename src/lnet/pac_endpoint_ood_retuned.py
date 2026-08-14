# ruff: noqa: EM101, EM102, T201, TRY003
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from threading import Lock
from typing import Final, cast

import torch

from .hybrid_experiment_types import resolve_device
from .pac_confirmatory_baselines import (
    CONFIRMATORY_FAMILIES,
    confirmatory_implementation_metadata,
    confirmatory_trial_spec,
)
from .pac_metrics import count_parameters, nrmse
from .pac_overnight_io import append_csv_row, prepare_overnight_dirs
from .pac_tf_p1p2_eval import (
    _build_synthetic_model,
    _endpoint_task,
    _SequenceEndpointRegressor,
    _trial_adjusted_experiment,
    run_p1p2_job,
    synthetic_ood_training_task,
)
from .pac_tf_p1p2_types import P1P2Config, P1P2Job
from .pac_training import train_regression_model
from .pac_types import PACDevice, PACExperimentConfig, PACRegressionTask

DEFAULT_ROOT: Final = Path(".omx/results/pac-endpoint-ood-retuned-pro6000-20260713")
TARGET_PARAMS: Final = 11_140
SEEDS: Final = (7, 11, 19, 23, 31)
FAMILIES: Final = tuple(family for family in CONFIRMATORY_FAMILIES if family != "pac_tf")
REFERENCE_MODEL: Final = "pac_stiefel_depth2_norm_autocorr_d64_m16"
PROTOCOL_PATH: Final = Path(".omx/protocols/pac_tf_confirmatory_20260711.json")
_MODEL_LOCK = Lock()


def tuning_jobs() -> tuple[P1P2Job, ...]:
    protocol_sha = hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()
    return tuple(
        P1P2Job(
            key=f"endpoint_retune:id_validation:{family}:trial{trial}:seed{seed}",
            package="synthetic_ood",
            seed=seed,
            model=family,
            reference_model=REFERENCE_MODEL,
            selection_trial=trial,
            architecture_metadata_json=_architecture_json(family, trial),
            refit_epochs=100,
            learning_rate=confirmatory_trial_spec(family, trial).learning_rate,
            weight_decay=confirmatory_trial_spec(family, trial).weight_decay,
            protocol_sha256=protocol_sha,
            synthetic_estimand="endpoint",
            synthetic_target_params=TARGET_PARAMS,
        )
        for family in FAMILIES
        for trial in range(1, 7)
        for seed in SEEDS
    )


def enqueue_tuning(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    jobs = tuning_jobs()
    stage = root / "tuning"
    prepare_overnight_dirs(stage)
    _write_locked_manifest(stage / "queue_manifest.jsonl", jobs)
    contract = {
        "schema": "pac_endpoint_ood_retuned.v1",
        "target_params": TARGET_PARAMS,
        "families": list(FAMILIES),
        "trials_per_family": 6,
        "seeds": list(SEEDS),
        "tuning_jobs": len(jobs),
        "selection_data": "ID synthetic train and validation endpoints only",
        "ood_observed_during_selection": False,
        "final_conditions": 19,
        "final_jobs": len(FAMILIES) * len(SEEDS),
        "restart_safe": True,
    }
    root.mkdir(parents=True, exist_ok=True)
    _write_locked_json(root / "contract.json", contract)
    _event(stage, "enqueue", "done", f"jobs={len(jobs)}")
    return contract


def run_tuning_workers(
    root: Path = DEFAULT_ROOT,
    *,
    device: PACDevice = "auto",
    workers: int = 4,
) -> dict[str, int]:
    if workers < 1 or workers > 4:
        raise ValueError("endpoint OOD tuning workers must be in [1, 4]")
    stage = root / "tuning"
    expected = tuning_jobs()
    manifest = _read_jobs(stage / "queue_manifest.jsonl")
    if manifest != expected:
        raise ValueError("tuning manifest differs from the frozen 6-trial contract")
    pending = [job for job in manifest if job.key not in _done_keys(stage)]
    _run_pool(stage, pending, device=device, workers=workers, tuning=True)
    return status(root)["tuning"]  # type: ignore[return-value]


def select_trials(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    stage = root / "tuning"
    expected = tuning_jobs()
    done = _latest_done_rows(stage / "results" / "id_validation.csv")
    missing = [job.key for job in expected if job.key not in done]
    if missing:
        raise ValueError(f"refusing selection before all ID-validation jobs finish: {missing[:3]}")
    selected: dict[str, object] = {}
    for family in FAMILIES:
        trials: dict[str, object] = {}
        ranked: list[tuple[float, int]] = []
        for trial in range(1, 7):
            rows = [
                done[f"endpoint_retune:id_validation:{family}:trial{trial}:seed{seed}"]
                for seed in SEEDS
            ]
            values = [float(row["validation_nrmse"]) for row in rows]
            aggregate = mean(values)
            ranked.append((aggregate, trial))
            trials[str(trial)] = {
                "mean_validation_nrmse": aggregate,
                "seed_validation_nrmse": {
                    str(seed): value for seed, value in zip(SEEDS, values, strict=True)
                },
                "params_trainable": int(rows[0]["params_trainable"]),
                "relative_param_error": float(rows[0]["relative_param_error"]),
                "architecture": json.loads(_architecture_json(family, trial)),
            }
        _, winner = min(ranked)
        spec = confirmatory_trial_spec(family, winner)
        selected[family] = {
            "trial": winner,
            "learning_rate": spec.learning_rate,
            "weight_decay": spec.weight_decay,
            "architecture": json.loads(_architecture_json(family, winner)),
            "trials": trials,
        }
    payload: dict[str, object] = {
        "schema": "pac_endpoint_ood_id_selection.v1",
        "status": "complete",
        "target_params": TARGET_PARAMS,
        "tuning_manifest_sha256": hashlib.sha256(
            (stage / "queue_manifest.jsonl").read_bytes()
        ).hexdigest(),
        "tuning_result_sha256": hashlib.sha256(
            (stage / "results" / "id_validation.csv").read_bytes()
        ).hexdigest(),
        "selection_metric": "mean endpoint ID validation NRMSE over five seeds",
        "ood_observed_during_selection": False,
        "selected_trials": selected,
    }
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    _write_locked_json(reports / "selection.json", payload)
    return payload


def enqueue_final(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    selection = _read_valid_selection(root)
    selected = cast("dict[str, dict[str, object]]", selection["selected_trials"])
    protocol_sha = hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()
    jobs = tuple(
        P1P2Job(
            key=f"endpoint_retuned:final_ood:{family}:seed{seed}",
            package="synthetic_ood",
            seed=seed,
            model=family,
            reference_model=REFERENCE_MODEL,
            selection_trial=int(selected[family]["trial"]),
            architecture_metadata_json=json.dumps(
                selected[family]["architecture"], sort_keys=True, separators=(",", ":")
            ),
            refit_epochs=100,
            learning_rate=float(selected[family]["learning_rate"]),
            weight_decay=float(selected[family]["weight_decay"]),
            protocol_sha256=protocol_sha,
            synthetic_estimand="endpoint",
            synthetic_target_params=TARGET_PARAMS,
        )
        for family in FAMILIES
        for seed in SEEDS
    )
    stage = root / "final"
    prepare_overnight_dirs(stage)
    _write_locked_manifest(stage / "queue_manifest.jsonl", jobs)
    lock = {
        "schema": "pac_endpoint_ood_retuned_final.v1",
        "selection_sha256": hashlib.sha256(
            (root / "reports" / "selection.json").read_bytes()
        ).hexdigest(),
        "selection_artifact": str((root / "reports" / "selection.json").resolve()),
        "target_params": TARGET_PARAMS,
        "jobs": len(jobs),
        "seeds": list(SEEDS),
        "conditions_per_job": 19,
        "test_policy": "selected once on ID validation, then frozen endpoint OOD evaluation",
    }
    _write_locked_json(stage / "lock.json", lock)
    _event(stage, "enqueue", "done", f"jobs={len(jobs)}")
    return lock


def run_final_workers(
    root: Path = DEFAULT_ROOT,
    *,
    device: PACDevice = "auto",
    workers: int = 4,
) -> dict[str, int]:
    if workers < 1 or workers > 4:
        raise ValueError("endpoint OOD final workers must be in [1, 4]")
    _read_valid_selection(root)
    stage = root / "final"
    jobs = _read_jobs(stage / "queue_manifest.jsonl")
    if len(jobs) != len(FAMILIES) * len(SEEDS):
        raise ValueError("final endpoint OOD manifest must contain exactly 40 jobs")
    pending = [job for job in jobs if job.key not in _done_keys(stage)]
    _run_pool(stage, pending, device=device, workers=workers, tuning=False)
    return status(root)["final"]  # type: ignore[return-value]


def status(root: Path = DEFAULT_ROOT) -> dict[str, dict[str, int]]:
    return {
        "tuning": _stage_status(root / "tuning", len(tuning_jobs())),
        "final": _stage_status(root / "final", len(FAMILIES) * len(SEEDS)),
    }


def write_report(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": "pac_endpoint_ood_retuned_report.v1",
        "status": status(root),
        "selection": (
            json.loads((root / "reports" / "selection.json").read_text(encoding="utf-8"))
            if (root / "reports" / "selection.json").exists()
            else None
        ),
        "partial_safe": True,
    }
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "RETUNED_ENDPOINT_OOD.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def _run_pool(
    stage: Path,
    jobs: list[P1P2Job],
    *,
    device: PACDevice,
    workers: int,
    tuning: bool,
) -> None:
    active: dict[Future[tuple[P1P2Job, dict[str, object] | None]], P1P2Job] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = list(jobs)
        while pending or active:
            while pending and len(active) < workers:
                job = pending.pop(0)
                _event(stage, job.key, "running")
                active[pool.submit(_execute, stage, job, device, tuning=tuning)] = job
            if not active:
                break
            complete, _ = wait(tuple(active), return_when=FIRST_COMPLETED)
            for future in complete:
                job = active.pop(future)
                _, row = future.result()
                _event(stage, job.key, "done" if row is not None else "failed")
    write_report(stage.parent)


def _execute(
    stage: Path,
    job: P1P2Job,
    device: PACDevice,
    *,
    tuning: bool,
) -> tuple[P1P2Job, dict[str, object] | None]:
    stream = torch.cuda.Stream() if device != "cpu" and torch.cuda.is_available() else None
    try:
        if stream is None:
            row = (
                _run_tuning_job(stage, job, device)
                if tuning
                else _run_final_job(stage, job, device)
            )
        else:
            with torch.cuda.stream(stream):
                row = (
                    _run_tuning_job(stage, job, device)
                    if tuning
                    else _run_final_job(stage, job, device)
                )
            stream.synchronize()
        append_csv_row(
            stage / "results" / ("id_validation.csv" if tuning else "synthetic_ood.csv"),
            row,
        )
    except Exception as error:  # noqa: BLE001 - durable queue records failures
        failed = asdict(job) | {
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
        }
        append_csv_row(
            stage / "results" / ("id_validation.csv" if tuning else "synthetic_ood.csv"),
            failed,
        )
        return job, None
    return job, row


def _run_tuning_job(stage: Path, job: P1P2Job, device: PACDevice) -> dict[str, object]:
    active_device = resolve_device(device)
    experiment = _trial_adjusted_experiment(
        job,
        PACExperimentConfig(
            2048,
            512,
            512,
            64,
            raw_input_dim=4,
            output_dim=2,
            model_dim=64,
            modes=16,
            epochs=100,
            batch_size=64,
            learning_rate=job.learning_rate,
            weight_decay=job.weight_decay,
            seeds=(job.seed,),
            device=cast("PACDevice", active_device),
            output_dir=stage,
        ),
    )
    original = _endpoint_task(synthetic_ood_training_task(experiment, job.seed))
    task = PACRegressionTask(
        original.label + "_validation_only",
        original.train_inputs,
        original.train_targets,
        original.validation_inputs,
        original.validation_targets,
        original.validation_inputs,
        original.validation_targets,
        true_delay=original.true_delay,
        true_frequency=original.true_frequency,
        true_frequencies=original.true_frequencies,
        true_dampings=original.true_dampings,
        mechanism_expectation=original.mechanism_expectation,
    )
    with _MODEL_LOCK:
        model, target, relative_error = _build_synthetic_model(
            job.model,
            job.reference_model,
            experiment,
            job.seed,
            job.selection_trial,
            max_relative_error=0.055,
            explicit_target_params=TARGET_PARAMS,
        )
    if model is None or target is None or relative_error is None:
        raise RuntimeError(f"no capacity match for {job.model}/trial{job.selection_trial}")
    endpoint_model = _SequenceEndpointRegressor(model)
    outcome = train_regression_model(endpoint_model, task, experiment, active_device, job.seed)
    return {
        "job_key": job.key,
        "stage": "id_validation_selection",
        "model": job.model,
        "trial": job.selection_trial,
        "seed": job.seed,
        "learning_rate": job.learning_rate,
        "weight_decay": job.weight_decay,
        "validation_loss": outcome.validation_loss,
        "validation_nrmse": nrmse(outcome.validation_loss, task.validation_targets),
        "params_trainable": count_parameters(endpoint_model),
        "target_params": target,
        "relative_param_error": relative_error,
        "ood_observed": False,
        "status": "done",
    }


def _run_final_job(stage: Path, job: P1P2Job, device: PACDevice) -> dict[str, object]:
    config = P1P2Config(
        output_root=stage,
        device=device,
        workers=1,
        total_slots=1,
        models=(job.model,),
        packages=("synthetic_ood",),
        synthetic_estimand="endpoint",
        synthetic_target_params=TARGET_PARAMS,
    )
    row = run_p1p2_job(config, job)
    row["selection_artifact"] = str((stage.parent / "reports" / "selection.json").resolve())
    row["retuned_on_id_validation_only"] = True
    return row


def _architecture_json(family: str, trial: int) -> str:
    return json.dumps(
        confirmatory_implementation_metadata(family, trial),  # type: ignore[arg-type]
        sort_keys=True,
        separators=(",", ":"),
    )


def _read_valid_selection(root: Path) -> dict[str, object]:
    path = root / "reports" / "selection.json"
    payload: dict[str, object] = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema") != "pac_endpoint_ood_id_selection.v1"
        or payload.get("status") != "complete"
        or payload.get("ood_observed_during_selection") is not False
        or payload.get("target_params") != TARGET_PARAMS
    ):
        raise ValueError("invalid or non-ID-only endpoint OOD selection artifact")
    manifest_sha = hashlib.sha256(
        (root / "tuning" / "queue_manifest.jsonl").read_bytes()
    ).hexdigest()
    if payload.get("tuning_manifest_sha256") != manifest_sha:
        raise ValueError("selection artifact does not match the tuning manifest")
    selected = payload.get("selected_trials")
    if not isinstance(selected, dict) or set(selected) != set(FAMILIES):
        raise ValueError("selection artifact does not cover all eight baselines")
    return payload


def _write_locked_manifest(path: Path, jobs: tuple[P1P2Job, ...]) -> None:
    encoded = "".join(json.dumps(asdict(job), sort_keys=True) + "\n" for job in jobs)
    if path.exists():
        current = path.read_text(encoding="utf-8")
        if current == encoded or _read_jobs(path) == jobs:
            return
        raise ValueError(f"refusing to overwrite a different manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded, encoding="utf-8")


def _write_locked_json(path: Path, payload: dict[str, object]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != encoded:
        raise ValueError(f"refusing to overwrite a different locked artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded, encoding="utf-8")


def _read_jobs(path: Path) -> tuple[P1P2Job, ...]:
    return tuple(
        P1P2Job(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _done_keys(stage: Path) -> set[str]:
    latest = _latest_statuses(stage / "queue_state.jsonl")
    return {key for key, state in latest.items() if state == "done"}


def _latest_statuses(path: Path) -> dict[str, str]:
    latest: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                latest[str(row["key"])] = str(row["status"])
    return latest


def _latest_done_rows(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    if path.exists():
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if row.get("status") == "done":
                    rows[str(row["job_key"])] = dict(row)
    return rows


def _stage_status(stage: Path, expected: int) -> dict[str, int]:
    manifest = stage / "queue_manifest.jsonl"
    jobs = _read_jobs(manifest) if manifest.exists() else ()
    job_keys = {job.key for job in jobs}
    manifested = len(jobs)
    counts = {"done": 0, "running": 0, "failed": 0}
    for key, value in _latest_statuses(stage / "queue_state.jsonl").items():
        if key not in job_keys:
            continue
        if value in counts:
            counts[value] += 1
    terminal_or_active = sum(counts.values())
    return {
        "expected": expected,
        "manifested": manifested,
        **counts,
        "pending": max(0, expected - terminal_or_active),
    }


def _event(stage: Path, key: str, status_value: str, notes: str = "") -> None:
    path = stage / "queue_state.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(
            json.dumps(
                {"key": key, "status": status_value, "notes": notes}, sort_keys=True
            )
            + "\n"
        )
        handle.flush()
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=(
            "enqueue-tuning",
            "tune-workers",
            "select",
            "enqueue-final",
            "final-workers",
            "status",
            "report",
        ),
        required=True,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.stage == "enqueue-tuning":
        payload = enqueue_tuning(args.output_root)
    elif args.stage == "tune-workers":
        payload = run_tuning_workers(
            args.output_root, device=cast("PACDevice", args.device), workers=args.workers
        )
    elif args.stage == "select":
        payload = select_trials(args.output_root)
    elif args.stage == "enqueue-final":
        payload = enqueue_final(args.output_root)
    elif args.stage == "final-workers":
        payload = run_final_workers(
            args.output_root, device=cast("PACDevice", args.device), workers=args.workers
        )
    elif args.stage == "report":
        payload = write_report(args.output_root)
    else:
        payload = status(args.output_root)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
