# ruff: noqa: EM101, EM102, TRY003
from __future__ import annotations

import csv
import gc
import json
import math
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING, Final, Literal, cast

import torch

from .pac_external_benchmarks import ExternalBenchmarkConfig, run_external_benchmarks
from .pac_matched_zoh_ood import matched_zoh_conditions, matched_zoh_training_task
from .pac_metrics import count_parameters, nrmse
from .pac_recommended_low_data_eval import run_low_data_job
from .pac_recommended_low_data_types import LowDataJob
from .pac_revised_final_queues import (
    _external_historical_seconds,  # pyright: ignore[reportPrivateUsage]
)
from .pac_training import evaluate_regression_loss, train_regression_model
from .pac_types import PACExperimentConfig
from .pac_unified_models import (
    PAC_UNIFIED_MODEL,
    CoordinateTimePACSequenceRegressor,
)

if TYPE_CHECKING:
    from .pac_external_tasks import ExternalDatasetName
    from .tapped_prl_followup_schema import JsonRow, JsonValue

DEFAULT_ROOT: Final = Path(".omx/results/pac-unified-global-20260712")
PROTOCOL_PATH: Final = Path(".omx/protocols/pac_tf_confirmatory_20260711.json")
CampaignKind = Literal["external", "ucr_validation", "ucr_test", "synthetic_ood"]
_EXTERNAL_BATCHES: Final = {
    "ptb-xl": 64,
    "mit-bih": 64,
    "cwru": 64,
    "speech-commands": 64,
    "ettm1": 64,
    "ettm2": 64,
    "electricity": 64,
    "weather": 64,
    "lra-listops": 32,
    "lra-text": 64,
    "lra-retrieval": 16,
    "lra-image": 64,
    "sequential-mnist": 64,
    "permuted-mnist": 64,
    "sequential-cifar": 64,
    "audioset-balanced": 64,
}
_FAST_WEAK_EXTERNAL: Final = {
    "lra-image",
    "sequential-cifar",
    "audioset-balanced",
}
_SLOW_WEAK_EXTERNAL: Final = {"lra-listops"}


@dataclass(frozen=True, slots=True)
class UnifiedCampaignJob:
    key: str
    kind: CampaignKind
    seed: int
    dataset: str = ""
    batch_size: int = 64
    estimated_seconds: float = 60.0
    priority: int = 2
    refit_epochs: int | None = None


def enqueue_phase1(root: Path = DEFAULT_ROOT, *, workers: int = 2) -> int:
    if workers < 1:
        raise ValueError("workers must be positive")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    ucr_datasets = tuple(
        dict.fromkeys(protocol["development_datasets"] + protocol["untouched_final_datasets"])
    )
    seeds = tuple(int(seed) for seed in protocol["seeds"])
    if seeds != (7, 11, 19, 23, 31):
        raise ValueError("unified campaign requires the locked five seeds")
    historical = _external_historical_seconds()
    jobs: list[UnifiedCampaignJob] = [
        UnifiedCampaignJob(
            key=f"external:{dataset}:seed{seed}",
            kind="external",
            dataset=dataset,
            seed=seed,
            batch_size=batch_size,
            estimated_seconds=historical.get((dataset, "pac"), 60.0) * 2.0,
            priority=(
                0
                if dataset in _FAST_WEAK_EXTERNAL
                else 1 if dataset in _SLOW_WEAK_EXTERNAL else 2
            ),
        )
        for dataset, batch_size in _EXTERNAL_BATCHES.items()
        for seed in (7, 11, 19)
    ]
    jobs.extend(
        UnifiedCampaignJob(
            key=f"ucr_validation:{dataset}:seed{seed}",
            kind="ucr_validation",
            dataset=str(dataset),
            seed=seed,
            estimated_seconds=120.0,
            priority=0 if dataset == "Phoneme" else 2,
        )
        for dataset in ucr_datasets
        for seed in seeds
    )
    jobs.extend(
        UnifiedCampaignJob(
            key=f"synthetic_ood:seed{seed}",
            kind="synthetic_ood",
            seed=seed,
            estimated_seconds=900.0,
            priority=-1,
        )
        for seed in seeds
    )
    _write_phase(root, "phase1", jobs, workers)
    contract = {
        "schema": "pac_unified_global_campaign.v1",
        "model": PAC_UNIFIED_MODEL,
        "model_dim": 64,
        "modes": 16,
        "coordinate_polynomial_degree": 2,
        "scales": 3,
        "evidence_slots": 4,
        "phase1_jobs": len(jobs),
        "phase1_breakdown": {
            "external": 48,
            "ucr_validation": 90,
            "synthetic_ood": 5,
        },
        "planned_ucr_test_jobs": 90,
        "workers": workers,
        "seeds": list(seeds),
        "synthetic_ood_protocol": "matched_exact_zoh_physical_time_v2",
        "ucr_datasets": list(ucr_datasets),
        "external_datasets": list(_EXTERNAL_BATCHES),
        "pathfinder_policy": "not scheduled; excluded by user because of runtime",
        "default_revised_model_unchanged": True,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return len(jobs)


def enqueue_ucr_test(root: Path = DEFAULT_ROOT, *, workers: int = 2) -> tuple[int, int]:
    validation_rows = _completed_rows(root, "ucr_validation")
    if len(validation_rows) != 90:
        message = f"expected 90 completed UCR validation jobs, found {len(validation_rows)}"
        raise RuntimeError(message)
    best_epochs = [
        _int_value(row["best_epoch"])
        for row in validation_rows
        if row.get("best_epoch") is not None
    ]
    if len(best_epochs) != 90:
        raise RuntimeError("all UCR validation jobs must report best_epoch")
    refit_epochs = max(1, math.floor(median(best_epochs) + 0.5))
    jobs = [
        UnifiedCampaignJob(
            key=f"ucr_test:{_str_value(row['dataset_or_task'])}:seed{_int_value(row['seed'])}",
            kind="ucr_test",
            dataset=_str_value(row["dataset_or_task"]),
            seed=_int_value(row["seed"]),
            estimated_seconds=_float_value(row.get("elapsed_train_time", 120.0)),
            refit_epochs=refit_epochs,
        )
        for row in validation_rows
    ]
    _write_phase(root, "phase2", jobs, workers)
    lock = {
        "schema": "pac_unified_ucr_refit_lock.v1",
        "validation_jobs": len(validation_rows),
        "median_best_epoch": median(best_epochs),
        "refit_epochs": refit_epochs,
        "official_test_observed_during_selection": False,
    }
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "ucr_refit_lock.json").write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return len(jobs), refit_epochs


def run_manifest(root: Path, manifest: Path, *, device: str = "cuda") -> None:
    jobs = [UnifiedCampaignJob(**json.loads(line)) for line in manifest.read_text().splitlines()]
    failures = 0
    for job in jobs:
        if _completed_path(root, job).exists():
            continue
        try:
            row = run_campaign_job(root, job, device=device)
        except Exception as error:  # noqa: BLE001 - durable queue records and continues
            failures += 1
            row: JsonRow = {
                "job_key": job.key,
                "kind": job.kind,
                "dataset_or_task": job.dataset,
                "seed": job.seed,
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
            _write_job_result(root, job, row, failed=True)
        else:
            _write_job_result(root, job, row, failed=False)
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    marker = root / "completion" / f"{manifest.stem}.COMPLETE"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(f"failures={failures}\n", encoding="utf-8")


def run_campaign_job(
    root: Path,
    job: UnifiedCampaignJob,
    *,
    device: str,
) -> JsonRow:
    match job.kind:
        case "external":
            return _run_external(root, job, device)
        case "ucr_validation" | "ucr_test":
            return _run_ucr(root, job, device)
        case "synthetic_ood":
            return _run_synthetic_ood(root, job, device)
        case unreachable:
            raise AssertionError(unreachable)


def campaign_status(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    contract = json.loads((root / "contract.json").read_text(encoding="utf-8"))
    counts: dict[str, int] = {}
    failures: dict[str, int] = {}
    for kind in ("external", "ucr_validation", "ucr_test", "synthetic_ood"):
        counts[kind] = len(_completed_rows(root, kind))
        failures[kind] = len(list((root / "failed" / kind).glob("*.json")))
    payload: dict[str, object] = {
        "model": contract["model"],
        "completed": counts,
        "failed": failures,
        "phase1_complete": sum(
            counts[kind] for kind in ("external", "ucr_validation", "synthetic_ood")
        )
        == int(contract["phase1_jobs"]),
        "phase2_complete": counts["ucr_test"] == int(contract["planned_ucr_test_jobs"]),
    }
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "STATUS.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _run_external(root: Path, job: UnifiedCampaignJob, device: str) -> JsonRow:
    output = root / "jobs" / "external" / _safe(job.key)
    config = ExternalBenchmarkConfig(
        data_root=Path("data/external"),
        output_root=output,
        datasets=(cast("ExternalDatasetName", job.dataset),),
        models=("pac",),
        model_dim=64,
        modes=16,
        max_baseline_width=8192,
        parameter_match_tolerance=0.05,
        epochs=60,
        batch_size=job.batch_size,
        learning_rate=1.0e-3,
        weight_decay=1.0e-4,
        grad_clip_norm=1.0,
        patience=12,
        seeds=(job.seed,),
        device=cast("Literal['auto', 'cpu', 'cuda']", device),
        pac_model=PAC_UNIFIED_MODEL,
    )
    run_external_benchmarks(config)
    rows = list(csv.DictReader((output / "results" / "external_comparisons.csv").open()))
    if len(rows) != 1 or rows[0].get("status") != "done":
        raise RuntimeError(f"external job did not complete: {rows}")
    return {"job_key": job.key, "kind": job.kind, **rows[0]}


def _run_ucr(root: Path, job: UnifiedCampaignJob, device: str) -> JsonRow:
    validation = job.kind == "ucr_validation"
    low_data = LowDataJob(
        key=job.key,
        seed=job.seed,
        model=PAC_UNIFIED_MODEL,
        dataset=job.dataset,
        ratio=1.0,
        evaluation_split="validation" if validation else "test",
        refit_full_train=not validation,
        data_protocol="clean_stratified",
        restore_best_validation=validation,
        evaluation_collection=(
            "pac_unified_train_validation" if validation else "pac_unified_official_test"
        ),
        reference_model=PAC_UNIFIED_MODEL,
        refit_epochs=job.refit_epochs,
        learning_rate=3.0e-3,
        weight_decay=1.0e-4,
    )
    config = PACExperimentConfig(
        2048,
        512,
        512,
        64,
        raw_input_dim=1,
        output_dim=2,
        model_dim=64,
        modes=16,
        epochs=100,
        batch_size=64,
        learning_rate=3.0e-3,
        weight_decay=1.0e-4,
        grad_clip_norm=1.0,
        seeds=(job.seed,),
        device=cast("Literal['auto', 'cpu', 'cuda']", device),
        output_dir=root / "jobs" / job.kind / _safe(job.key),
        optimizer_mode="fused" if device == "cuda" else "default",
    )
    return {"kind": job.kind, **run_low_data_job(config, low_data)}


def _run_synthetic_ood(root: Path, job: UnifiedCampaignJob, device: str) -> JsonRow:
    config = PACExperimentConfig(
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
        learning_rate=3.0e-3,
        weight_decay=1.0e-4,
        seeds=(job.seed,),
        device=cast("Literal['auto', 'cpu', 'cuda']", device),
        output_dir=root / "jobs" / "synthetic_ood" / _safe(job.key),
        optimizer_mode="fused" if device == "cuda" else "default",
    )
    task = matched_zoh_training_task(config, job.seed)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(job.seed)
        model = CoordinateTimePACSequenceRegressor(config)
    outcome = train_regression_model(model, task, config, device, job.seed)
    sweeps: list[dict[str, object]] = []
    for condition in matched_zoh_conditions(config, job.seed):
        variable_id_loss = evaluate_regression_loss(
            model,
            condition.id_inputs.to(device=device),
            condition.id_targets.to(device=device),
        )
        variable_ood_loss = evaluate_regression_loss(
            model,
            condition.ood_inputs.to(device=device),
            condition.ood_targets.to(device=device),
        )
        fixed_id_loss = evaluate_regression_loss(
            model,
            force_unit_time_delta(condition.id_inputs).to(device=device),
            condition.id_targets.to(device=device),
        )
        fixed_ood_loss = evaluate_regression_loss(
            model,
            force_unit_time_delta(condition.ood_inputs).to(device=device),
            condition.ood_targets.to(device=device),
        )
        variable_ood_nrmse = nrmse(variable_ood_loss, condition.ood_targets)
        fixed_ood_nrmse = nrmse(fixed_ood_loss, condition.ood_targets)
        sweeps.append(
            {
                "family": condition.family,
                "level": condition.level,
                "id_nrmse": nrmse(variable_id_loss, condition.id_targets),
                "ood_nrmse": variable_ood_nrmse,
                "variable_delta_id_nrmse": nrmse(
                    variable_id_loss, condition.id_targets
                ),
                "fixed_delta_id_nrmse": nrmse(fixed_id_loss, condition.id_targets),
                "variable_delta_ood_nrmse": variable_ood_nrmse,
                "fixed_delta_ood_nrmse": fixed_ood_nrmse,
                "variable_delta_gain_nrmse": fixed_ood_nrmse - variable_ood_nrmse,
            }
        )
    return {
        "job_key": job.key,
        "kind": job.kind,
        "dataset_or_task": "synthetic_ood",
        "seed": job.seed,
        "model": PAC_UNIFIED_MODEL,
        "delta_comparison": "paired_same_weights_variable_vs_unit",
        "synthetic_protocol": "matched_exact_zoh_physical_time_v2",
        "params_trainable": count_parameters(model),
        "id_test_loss": outcome.test_loss,
        "id_test_nrmse": nrmse(outcome.test_loss, task.test_targets),
        "ood_sweep_json": json.dumps(sweeps, separators=(",", ":")),
        "status": "done",
    }


def force_unit_time_delta(inputs: torch.Tensor) -> torch.Tensor:
    if inputs.ndim != 3 or inputs.shape[-1] < 4:
        raise ValueError("delta comparison inputs must be [B,N,C] with dt and mask channels")
    return torch.cat(
        (
            inputs[..., :-2],
            torch.ones_like(inputs[..., -2:-1]),
            inputs[..., -1:],
        ),
        dim=-1,
    )


def _write_phase(
    root: Path,
    phase: str,
    jobs: list[UnifiedCampaignJob],
    workers: int,
) -> None:
    shards: list[list[UnifiedCampaignJob]] = [[] for _ in range(workers)]
    loads = [0.0] * workers
    for job in sorted(jobs, key=lambda item: item.estimated_seconds, reverse=True):
        index = min(range(workers), key=loads.__getitem__)
        shards[index].append(job)
        loads[index] += job.estimated_seconds
    manifests = root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    for index, shard in enumerate(shards):
        ordered = sorted(shard, key=lambda item: (item.priority, item.estimated_seconds, item.key))
        path = manifests / f"{phase}-worker-{index}.jsonl"
        path.write_text(
            "".join(json.dumps(asdict(job), sort_keys=True) + "\n" for job in ordered),
            encoding="utf-8",
        )


def _write_job_result(
    root: Path,
    job: UnifiedCampaignJob,
    row: JsonRow,
    *,
    failed: bool,
) -> None:
    base = root / ("failed" if failed else "completed") / job.kind
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"{_safe(job.key)}.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _completed_path(root: Path, job: UnifiedCampaignJob) -> Path:
    return root / "completed" / job.kind / f"{_safe(job.key)}.json"


def _completed_rows(root: Path, kind: CampaignKind) -> list[JsonRow]:
    rows: list[JsonRow] = []
    for path in sorted((root / "completed" / kind).glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and payload.get("status") == "done":
            rows.append(payload)
    return rows


def _safe(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in value
    )


def _int_value(value: JsonValue) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError(f"expected integer-compatible value, got {type(value).__name__}")
    return int(value)


def _float_value(value: JsonValue) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError(f"expected float-compatible value, got {type(value).__name__}")
    return float(value)


def _str_value(value: JsonValue) -> str:
    if not isinstance(value, str):
        raise TypeError(f"expected string value, got {type(value).__name__}")
    return value
