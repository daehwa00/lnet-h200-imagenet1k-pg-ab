from __future__ import annotations

import gc
import hashlib
import json
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Final, cast

import torch

from .pac_eval_sections import clean_validation_classification_task
from .pac_final_validation import UCR_SECONDS
from .pac_metrics import count_parameters
from .pac_real_data import ensure_ucr_train_only
from .pac_retrained_ablation_models import (
    ABLATION_VARIANTS,
    AblationVariant,
    build_retrained_ablation_model,
)
from .pac_training import classification_metric_bundle, train_classifier
from .pac_types import PACDevice, PACExperimentConfig

DEFAULT_ROOT: Final = Path(".omx/results/pac-retrained-core-ablation-pro6000-20260713")
DATASETS: Final = (
    "ArrowHead",
    "CinCECGTorso",
    "CricketX",
    "ECG200",
    "ECG5000",
    "ECGFiveDays",
    "Earthquakes",
    "FordA",
    "FordB",
    "GunPoint",
    "ItalyPowerDemand",
    "MoteStrain",
    "Phoneme",
    "Plane",
    "StarLightCurves",
    "Trace",
    "TwoLeadECG",
    "Wafer",
)
SEEDS: Final = (7, 11, 19, 23, 31)


@dataclass(frozen=True, slots=True)
class RetrainedAblationJob:
    key: str
    dataset: str
    variant: AblationVariant
    seed: int
    epochs: int = 100
    batch_size: int = 64
    estimated_seconds: float = 60.0


def retrained_ablation_jobs() -> list[RetrainedAblationJob]:
    return [
        RetrainedAblationJob(
            key=f"ucr_validation:{dataset}:{variant}:seed{seed}",
            dataset=dataset,
            variant=variant,
            seed=seed,
            estimated_seconds=UCR_SECONDS[dataset] * _runtime_factor(variant),
        )
        for dataset in DATASETS
        for variant in ABLATION_VARIANTS
        for seed in SEEDS
    ]


def enqueue_retrained_ablation(root: Path = DEFAULT_ROOT, *, workers: int = 6) -> dict[str, object]:
    if not 1 <= workers <= 6:
        message = "workers must be between 1 and 6 on the shared pro6000"
        raise ValueError(message)
    jobs = retrained_ablation_jobs()
    completed = _result_keys(root / "completed")
    pending = [job for job in jobs if job.key not in completed]
    shards: list[list[RetrainedAblationJob]] = [[] for _ in range(workers)]
    loads = [0.0] * workers
    for job in sorted(pending, key=lambda item: item.estimated_seconds, reverse=True):
        worker = min(range(workers), key=loads.__getitem__)
        shards[worker].append(job)
        loads[worker] += job.estimated_seconds
    manifests = root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    for stale in manifests.glob("*.jsonl"):
        stale.unlink()
    for worker, shard in enumerate(shards):
        ordered = sorted(shard, key=lambda item: (item.estimated_seconds, item.key))
        (manifests / f"pro6000-gpu0-worker{worker}.jsonl").write_text(
            "".join(json.dumps(asdict(job), sort_keys=True) + "\n" for job in ordered),
            encoding="utf-8",
        )
    root.mkdir(parents=True, exist_ok=True)
    contract = {
        "schema": "pac_retrained_core_ablation.v1",
        "purpose": "submission-facing retrained ALPHABET component controls",
        "evaluation_split": "TRAIN-derived validation only",
        "official_test_accessed": False,
        "datasets": list(DATASETS),
        "seeds": list(SEEDS),
        "variants": list(ABLATION_VARIANTS),
        "jobs": len(jobs),
        "pending": len(pending),
        "workers": workers,
        "estimated_worker_seconds": loads,
        "shared_recipe": {
            "epochs": 100,
            "batch_size": 64,
            "optimizer": "AdamW",
            "learning_rate": 0.003,
            "weight_decay": 0.0001,
            "gradient_clip": 1.0,
            "checkpoint": "minimum TRAIN-derived validation loss",
            "normalization": "fit on optimization fold only",
        },
        "factorization": {
            "raw_shared": {"bands": "raw", "sharing": "single", "origin": "none"},
            "low_only_fixed": {"bands": "low", "sharing": "single", "origin": "fixed"},
            "detail_only_fixed": {
                "bands": "detail",
                "sharing": "single",
                "origin": "fixed",
            },
            "shared_two_band_fixed": {
                "bands": "low+detail",
                "sharing": "shared",
                "origin": "fixed",
            },
            "unshared_two_band_fixed": {
                "bands": "low+detail",
                "sharing": "unshared parameter-matched",
                "origin": "fixed",
            },
            "alphabet_dual": {
                "bands": "low+detail",
                "sharing": "shared",
                "origin": "random training / dual inference",
            },
        },
    }
    (root / "contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "jobs": len(jobs),
        "pending": len(pending),
        "workers": workers,
        "estimated_worker_seconds": loads,
    }


def run_manifest(root: Path, manifest: Path, *, device: str = "cuda") -> None:
    jobs = [
        RetrainedAblationJob(**json.loads(line))
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line
    ]
    for job in jobs:
        completed = _result_path(root, job, failed=False)
        if completed.exists():
            continue
        try:
            row = run_job(job, device=device, checkpoint_root=root / "checkpoints")
        except Exception as error:  # noqa: BLE001 - durable restart-safe queue
            row: dict[str, object] = {
                "job_key": job.key,
                **asdict(job),
                "status": "failed",
                "official_test_accessed": False,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
            _write_result(root, job, row, failed=True)
        else:
            _write_result(root, job, row, failed=False)
            failed_path = _result_path(root, job, failed=True)
            if failed_path.exists():
                failed_path.unlink()
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def run_job(
    job: RetrainedAblationJob,
    *,
    device: str,
    checkpoint_root: Path | None = None,
) -> dict[str, object]:
    dataset = ensure_ucr_train_only(job.dataset, Path(".omx/data/ucr"), allow_download=True)
    task = clean_validation_classification_task(dataset, job.seed)
    config = PACExperimentConfig(
        task.train_inputs.shape[0],
        task.validation_inputs.shape[0],
        0,
        task.train_inputs.shape[1],
        raw_input_dim=task.train_inputs.shape[-1],
        output_dim=task.class_count,
        model_dim=64,
        modes=16,
        epochs=job.epochs,
        batch_size=job.batch_size,
        learning_rate=3.0e-3,
        weight_decay=1.0e-4,
        grad_clip_norm=1.0,
        seeds=(job.seed,),
        device=cast("PACDevice", device),
    )
    torch.manual_seed(job.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(job.seed)
        torch.cuda.reset_peak_memory_stats()
    model, metadata = build_retrained_ablation_model(job.variant, config, task.class_count)
    started = perf_counter()
    outcome = train_classifier(
        model,
        task,
        config,
        device,
        job.seed,
        evaluate_test=False,
        restore_best_validation=True,
    )
    train_seconds = perf_counter() - started
    metrics = classification_metric_bundle(
        model,
        task.validation_inputs.to(device=device),
        task.validation_labels.to(device=device),
        batch_size=job.batch_size,
    )
    checkpoint: dict[str, object] = {}
    if job.variant == "alphabet_dual" and checkpoint_root is not None:
        checkpoint = _save_alphabet_checkpoint(
            checkpoint_root,
            job,
            model,
            config,
            class_count=task.class_count,
            best_epoch=outcome.best_epoch,
        )
    peak_memory_mb = float(torch.cuda.max_memory_allocated() / 1.0e6) if device == "cuda" else 0.0
    return {
        "job_key": job.key,
        **asdict(job),
        "status": "done",
        "evaluation_split": "validation",
        "official_test_accessed": False,
        "normalization_fit": "optimization fold only",
        "checkpoint_policy": "minimum validation loss",
        "best_epoch": outcome.best_epoch,
        "validation_loss": outcome.validation_loss,
        "validation_accuracy": metrics.accuracy,
        "validation_macro_f1": metrics.macro_f1,
        "validation_weighted_f1": metrics.weighted_f1,
        "validation_balanced_accuracy": metrics.balanced_accuracy,
        "train_seconds": train_seconds,
        "peak_memory_mb": peak_memory_mb,
        "params_trainable": count_parameters(model),
        "model_metadata": asdict(metadata),
        **checkpoint,
    }


def _save_alphabet_checkpoint(
    checkpoint_root: Path,
    job: RetrainedAblationJob,
    model: torch.nn.Module,
    config: PACExperimentConfig,
    *,
    class_count: int,
    best_epoch: int | None,
) -> dict[str, object]:
    path = checkpoint_root / job.dataset / f"alphabet_dual_seed{job.seed}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    payload = {
        "schema": "pac_alphabet_validation_checkpoint.v1",
        "job": asdict(job),
        "architecture": "PA2WP",
        "objective": "classification",
        "evaluation_split": "TRAIN-derived validation only",
        "official_test_accessed": False,
        "checkpoint_policy": "minimum TRAIN-derived validation loss",
        "best_epoch": best_epoch,
        "config": {
            "sample_count": config.sample_count,
            "validation_count": config.validation_count,
            "test_count": config.test_count,
            "sequence_length": config.sequence_length,
            "raw_input_dim": config.raw_input_dim,
            "output_dim": config.output_dim,
            "model_dim": config.model_dim,
            "modes": config.modes,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "grad_clip_norm": config.grad_clip_norm,
            "seed": job.seed,
            "class_count": class_count,
        },
        "model_state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
    }
    torch.save(payload, temporary)
    temporary.replace(path)
    return {
        "checkpoint_path": str(path.resolve()),
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def retrained_ablation_status(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    expected = {job.key for job in retrained_ablation_jobs()}
    completed = _result_keys(root / "completed")
    failed = _result_keys(root / "failed") - completed
    return {
        "expected": len(expected),
        "completed": len(expected & completed),
        "failed": len(expected & failed),
        "remaining": len(expected - completed - failed),
        "done": expected <= completed and not (expected & failed),
    }


def write_retrained_ablation_report(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    rows = _completed_rows(root / "completed")
    variants: dict[str, object] = {}
    for variant in ABLATION_VARIANTS:
        selected = [row for row in rows if row["variant"] == variant]
        datasets: dict[str, object] = {}
        for dataset in DATASETS:
            values = [
                float(row["validation_balanced_accuracy"])
                for row in selected
                if row["dataset"] == dataset
            ]
            datasets[dataset] = {"mean": mean(values) if values else None, "seeds": len(values)}
        variants[variant] = {
            "mean_balanced_accuracy": (
                mean(float(row["validation_balanced_accuracy"]) for row in selected)
                if selected
                else None
            ),
            "rows": len(selected),
            "datasets": datasets,
        }
    payload: dict[str, object] = {
        "status": retrained_ablation_status(root),
        "variants": variants,
    }
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "RETRAINED_CORE_ABLATION.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def _runtime_factor(variant: AblationVariant) -> float:
    return {
        "raw_shared": 1.0,
        "low_only_fixed": 0.65,
        "detail_only_fixed": 0.65,
        "shared_two_band_fixed": 1.0,
        "unshared_two_band_fixed": 1.1,
        "alphabet_dual": 1.2,
    }[variant]


def _safe(key: str) -> str:
    return key.replace(":", "_").replace("/", "_")


def _result_path(root: Path, job: RetrainedAblationJob, *, failed: bool) -> Path:
    return root / ("failed" if failed else "completed") / f"{_safe(job.key)}.json"


def _write_result(
    root: Path, job: RetrainedAblationJob, row: dict[str, object], *, failed: bool
) -> None:
    path = _result_path(root, job, failed=failed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _result_keys(directory: Path) -> set[str]:
    if not directory.exists():
        return set()
    return {
        str(json.loads(path.read_text(encoding="utf-8"))["job_key"])
        for path in directory.glob("*.json")
    }


def _completed_rows(directory: Path) -> list[dict[str, object]]:
    if not directory.exists():
        return []
    return [json.loads(path.read_text(encoding="utf-8")) for path in directory.glob("*.json")]
