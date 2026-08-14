# ruff: noqa: EM101, EM102, TRY003
from __future__ import annotations

import gc
import json
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Final, Literal, cast

import torch

from .pac_eval_sections import clean_validation_classification_task
from .pac_external_benchmarks import (
    ExternalBenchmarkConfig,
    _build_model,  # pyright: ignore[reportPrivateUsage]
    _loss,  # pyright: ignore[reportPrivateUsage]
    _predict,  # pyright: ignore[reportPrivateUsage]
    _seed_everything,  # pyright: ignore[reportPrivateUsage]
    _train_model,  # pyright: ignore[reportPrivateUsage]
    external_metric_bundle,
)
from .pac_external_tasks import ExternalDatasetName, load_external_task
from .pac_headroom_efficient_models import (
    DUAL_PHASE_WP_PAC_MODEL,
    FINAL_PAC_MODEL,
    LEARNED_PAIR_WP_PAC_MODEL,
    OVERLAPPING_ANTIALIASED_PAC_MODEL,
    PHASE_AUGMENTED_ENSEMBLE_WP_PAC_MODEL,
    PHASE_AUGMENTED_WP_PAC_MODEL,
    PHASE_COMPLETE_WP_PAC_MODEL,
    SPARSE_MULTISCALE_FBFB_PAC_MODEL,
    SPARSE_MULTISCALE_FF_PAC_MODEL,
    SPARSE_MULTISCALE_PAC_MODEL,
    UNDECIMATED_MODAL_DYADIC_PAC_MODEL,
    build_efficient_headroom_classifier,
)
from .pac_headroom_models import HEADROOM_SPECS, build_headroom_pac_classifier
from .pac_metrics import count_parameters
from .pac_real_data import ensure_ucr_train_only
from .pac_training import classification_metric_bundle, train_classifier
from .pac_types import PACDevice, PACExperimentConfig

if TYPE_CHECKING:
    from .pac_headroom_efficient_models import EfficientHeadroomSpec

DEFAULT_ROOT: Final = Path(".omx/results/pac-headroom-fast-3seed-20260712")
ScreenTaskKind = Literal["external", "ucr_validation"]
_SEEDS: Final = (7, 11, 19)
_PHASE_DIAGNOSTIC_SPECS: Final = {"WP", "PAWP", "PA2WP", "LPWP"}


@dataclass(frozen=True, slots=True)
class HeadroomScreenJob:
    key: str
    task_kind: ScreenTaskKind
    dataset: str
    spec: str
    seed: int
    epochs: int
    batch_size: int = 64
    estimated_seconds: float = 60.0
    patience: int = 4


def phase1_jobs() -> list[HeadroomScreenJob]:
    jobs: list[HeadroomScreenJob] = []
    jobs.extend(_external_jobs("sequential-mnist", ("B", "G"), 20, 160.0))
    jobs.extend(_external_jobs("audioset-balanced", ("B", "M", "S"), 20, 70.0))
    jobs.extend(_external_jobs("ettm2", ("B", "M"), 20, 70.0))
    jobs.extend(_external_jobs("cwru", ("B", "G", "M", "S"), 20, 15.0))
    jobs.extend(
        HeadroomScreenJob(
            key=f"ucr_validation:Phoneme:{spec}:seed{seed}",
            task_kind="ucr_validation",
            dataset="Phoneme",
            spec=spec,
            seed=seed,
            epochs=100,
            estimated_seconds=75.0,
        )
        for spec in ("B", "S")
        for seed in _SEEDS
    )
    return jobs


def enqueue_phase1(root: Path = DEFAULT_ROOT, *, workers: int = 2) -> int:
    if workers < 1:
        raise ValueError("workers must be positive")
    jobs = phase1_jobs()
    shards: list[list[HeadroomScreenJob]] = [[] for _ in range(workers)]
    loads = [0.0] * workers
    for job in sorted(jobs, key=lambda item: item.estimated_seconds, reverse=True):
        index = min(range(workers), key=loads.__getitem__)
        shards[index].append(job)
        loads[index] += job.estimated_seconds
    manifests = root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    for index, shard in enumerate(shards):
        ordered = sorted(shard, key=lambda item: (item.estimated_seconds, item.key))
        (manifests / f"phase1-worker-{index}.jsonl").write_text(
            "".join(json.dumps(asdict(job), sort_keys=True) + "\n" for job in ordered),
            encoding="utf-8",
        )
    contract = {
        "schema": "pac_headroom_fast_3seed.v1",
        "baseline": "Revised PAC + optional variable-step exact ZOH",
        "seeds": list(_SEEDS),
        "jobs": len(jobs),
        "workers": workers,
        "official_test_accessed": False,
        "excluded_long_tasks": [
            "pathfinder",
            "lra-image",
            "sequential-cifar",
            "lra-listops",
            "lra-text",
            "lra-retrieval",
        ],
        "specs": {name: asdict(spec) for name, spec in HEADROOM_SPECS.items()},
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return len(jobs)


def run_manifest(root: Path, manifest: Path, *, device: str = "cuda") -> None:
    jobs = [HeadroomScreenJob(**json.loads(line)) for line in manifest.read_text().splitlines()]
    for job in jobs:
        completed = _result_path(root, job, failed=False)
        if completed.exists():
            continue
        try:
            row = run_job(job, device=device)
        except Exception as error:  # noqa: BLE001 - durable queue records and continues
            row: dict[str, object] = {
                "job_key": job.key,
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                **asdict(job),
            }
            _write_result(root, job, row, failed=True)
        else:
            _write_result(root, job, row, failed=False)
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def run_job(job: HeadroomScreenJob, *, device: str) -> dict[str, object]:
    if job.spec not in HEADROOM_SPECS and job.spec not in {
        "WP",
        "PCWP",
        "DPWP",
        "PAWP",
        "PA2WP",
        "LPWP",
        "OA",
        "SMR",
        "SMRFF",
        "SMRFBFB",
        "UMD",
        "EFP8",
        "EFP16",
        "EFU8",
        "C2M8",
    }:
        raise ValueError(f"unknown headroom spec: {job.spec}")
    if job.task_kind == "external":
        return _run_external(job, device)
    return _run_ucr(job, device)


def status(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    completed = list((root / "completed").glob("*.json")) if (root / "completed").exists() else []
    failed = list((root / "failed").glob("*.json")) if (root / "failed").exists() else []
    expected_keys = {job.key for job in phase1_jobs()}
    completed_keys = {
        str(json.loads(path.read_text(encoding="utf-8"))["job_key"]) for path in completed
    }
    failed_keys = {str(json.loads(path.read_text(encoding="utf-8"))["job_key"]) for path in failed}
    payload: dict[str, object] = {
        "expected": len(expected_keys),
        "completed": len(expected_keys & completed_keys),
        "failed": len(expected_keys & failed_keys),
        "done": expected_keys <= completed_keys and not (expected_keys & failed_keys),
    }
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "STATUS.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _external_jobs(
    dataset: str,
    specs: tuple[str, ...],
    epochs: int,
    seconds: float,
) -> list[HeadroomScreenJob]:
    return [
        HeadroomScreenJob(
            key=f"external:{dataset}:{spec}:seed{seed}",
            task_kind="external",
            dataset=dataset,
            spec=spec,
            seed=seed,
            epochs=epochs,
            estimated_seconds=seconds * _spec_runtime_factor(spec),
        )
        for spec in specs
        for seed in _SEEDS
    ]


def _spec_runtime_factor(spec: str) -> float:
    if spec == "WP":
        return 1.1
    active = HEADROOM_SPECS[spec]
    factor = 1.0
    if active.geometry_compatible_stem:
        factor *= 1.6
    if active.geometry:
        factor *= 1.25
    if active.multiscale:
        factor *= 1.75
    if active.slots:
        factor *= 1.1
    return factor


def _run_external(job: HeadroomScreenJob, device: str) -> dict[str, object]:
    task = load_external_task(
        cast("ExternalDatasetName", job.dataset),
        Path("data/external"),
    )
    # The paired retrieval objective multiplies two encoded representations and
    # was the only final-validation task to diverge for every seed at 1e-3.
    # Keep the model and all other optimization settings fixed while using the
    # conservative rate already required by this numerically sharper objective.
    learning_rate = 3.0e-4 if job.dataset == "lra-retrieval" else 1.0e-3
    benchmark = ExternalBenchmarkConfig(
        data_root=Path("data/external"),
        output_root=DEFAULT_ROOT,
        datasets=(cast("ExternalDatasetName", job.dataset),),
        models=("pac",),
        model_dim=64,
        modes=16,
        epochs=job.epochs,
        batch_size=job.batch_size,
        learning_rate=learning_rate,
        weight_decay=1.0e-4,
        grad_clip_norm=1.0,
        patience=job.patience,
        seeds=(job.seed,),
        device=cast("Literal['auto', 'cpu', 'cuda']", device),
        pac_model=(
            SPARSE_MULTISCALE_PAC_MODEL
            if job.spec == "SMR"
            else OVERLAPPING_ANTIALIASED_PAC_MODEL
            if job.spec == "OA"
            else PHASE_COMPLETE_WP_PAC_MODEL
            if job.spec == "PCWP"
            else DUAL_PHASE_WP_PAC_MODEL
            if job.spec == "DPWP"
            else PHASE_AUGMENTED_WP_PAC_MODEL
            if job.spec == "PAWP"
            else PHASE_AUGMENTED_ENSEMBLE_WP_PAC_MODEL
            if job.spec == "PA2WP"
            else LEARNED_PAIR_WP_PAC_MODEL
            if job.spec == "LPWP"
            else UNDECIMATED_MODAL_DYADIC_PAC_MODEL
            if job.spec == "UMD"
            else SPARSE_MULTISCALE_FBFB_PAC_MODEL
            if job.spec == "SMRFBFB"
            else SPARSE_MULTISCALE_FF_PAC_MODEL
            if job.spec == "SMRFF"
            else FINAL_PAC_MODEL
        ),
    )
    config = PACExperimentConfig(
        task.train_inputs.shape[0],
        task.validation_inputs.shape[0],
        0,
        task.sequence_length,
        raw_input_dim=task.input_dim,
        output_dim=task.output_dim,
        model_dim=64,
        modes=16,
        epochs=job.epochs,
        batch_size=job.batch_size,
        learning_rate=benchmark.learning_rate,
        weight_decay=benchmark.weight_decay,
        grad_clip_norm=benchmark.grad_clip_norm,
        seeds=(job.seed,),
        device=cast("PACDevice", device),
    )
    _seed_everything(job.seed, device)
    if job.spec in {
        "WP",
        "PCWP",
        "DPWP",
        "PAWP",
        "PA2WP",
        "LPWP",
        "OA",
        "SMR",
        "SMRFF",
        "SMRFBFB",
        "UMD",
    }:
        model = _build_model("pac", 64, task, benchmark).to(device=device)
    else:
        if task.input_encoding != "continuous":
            message = "development headroom controls support continuous external tasks only"
            raise ValueError(message)
        coordinate_shape = (28, 28) if job.dataset == "sequential-mnist" else None
        objective = "regression" if task.objective == "forecasting" else "classification"
        model = build_headroom_pac_classifier(
            job.spec,
            config,
            task.output_dim,
            coordinate_shape=coordinate_shape,
            objective=objective,
        ).to(device=device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    best_epoch, validation_loss = _train_model(model, task, benchmark, device, job.seed)
    train_seconds = perf_counter() - started
    logits, targets = _predict(
        model,
        task.validation_inputs,
        task.validation_targets,
        job.batch_size,
        device,
    )
    metrics = external_metric_bundle(logits, targets, task.objective)
    measured_loss = float(_loss(logits, targets, task.objective).item())
    phase_js = None
    if (
        job.spec in _PHASE_DIAGNOSTIC_SPECS
        and task.objective == "multiclass"
        and task.input_encoding == "continuous"
        and task.validation_inputs.shape[1] > 1
    ):
        shifted_logits, _ = _predict(
            model,
            task.validation_inputs[:, 1:],
            task.validation_targets,
            job.batch_size,
            device,
        )
        phase_js = _phase_js_divergence(logits, shifted_logits)
    latency_ms = _validation_latency(model, task.validation_inputs, job.batch_size, device)
    peak_memory_mb = float(torch.cuda.max_memory_allocated() / 1.0e6) if device == "cuda" else 0.0
    return {
        "job_key": job.key,
        **asdict(job),
        "status": "done",
        "evaluation_split": "validation",
        "official_test_accessed": False,
        "params_trainable": count_parameters(model),
        "best_epoch": best_epoch,
        "validation_loss": validation_loss,
        "validation_loss_recomputed": measured_loss,
        "train_seconds": train_seconds,
        "latency_ms": latency_ms,
        "peak_memory_mb": peak_memory_mb,
        **({"phase_js_divergence": phase_js} if phase_js is not None else {}),
        **{f"validation_{name}": value for name, value in metrics.items()},
    }


def _run_ucr(job: HeadroomScreenJob, device: str) -> dict[str, object]:
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
    if job.spec in {
        "WP",
        "PCWP",
        "DPWP",
        "PAWP",
        "PA2WP",
        "LPWP",
        "OA",
        "SMR",
        "SMRFF",
        "SMRFBFB",
        "UMD",
        "EFP8",
        "EFP16",
        "EFU8",
        "C2M8",
    }:
        model = build_efficient_headroom_classifier(
            cast("EfficientHeadroomSpec", job.spec),
            config,
            task.class_count,
            objective="classification",
        ).to(device=device)
    else:
        model = build_headroom_pac_classifier(job.spec, config, task.class_count).to(device=device)
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
    phase_js = None
    if job.spec in _PHASE_DIAGNOSTIC_SPECS and task.validation_inputs.shape[1] > 1:
        ordinary_logits = _batched_logits(
            model,
            task.validation_inputs,
            batch_size=job.batch_size,
            device=device,
        )
        shifted_logits = _batched_logits(
            model,
            task.validation_inputs[:, 1:],
            batch_size=job.batch_size,
            device=device,
        )
        phase_js = _phase_js_divergence(ordinary_logits, shifted_logits)
    latency_ms = _validation_latency(model, task.validation_inputs, job.batch_size, device)
    peak_memory_mb = float(torch.cuda.max_memory_allocated() / 1.0e6) if device == "cuda" else 0.0
    return {
        "job_key": job.key,
        **asdict(job),
        "status": "done",
        "evaluation_split": "validation",
        "official_test_accessed": False,
        "params_trainable": count_parameters(model),
        "best_epoch": outcome.best_epoch,
        "validation_loss": outcome.validation_loss,
        "train_seconds": train_seconds,
        "latency_ms": latency_ms,
        "peak_memory_mb": peak_memory_mb,
        **({"phase_js_divergence": phase_js} if phase_js is not None else {}),
        "validation_accuracy": metrics.accuracy,
        "validation_macro_f1": metrics.macro_f1,
        "validation_weighted_f1": metrics.weighted_f1,
        "validation_balanced_accuracy": metrics.balanced_accuracy,
    }


def _batched_logits(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    *,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        outputs = [
            model(inputs[start : start + batch_size].to(device=device)).cpu()
            for start in range(0, inputs.shape[0], batch_size)
        ]
    model.train(was_training)
    return torch.cat(outputs)


def _phase_js_divergence(ordinary_logits: torch.Tensor, shifted_logits: torch.Tensor) -> float:
    ordinary = torch.softmax(ordinary_logits.float(), dim=-1).clamp_min(1.0e-12)
    shifted = torch.softmax(shifted_logits.float(), dim=-1).clamp_min(1.0e-12)
    midpoint = 0.5 * (ordinary + shifted)
    divergence = 0.5 * (
        (ordinary * (ordinary.log() - midpoint.log())).sum(dim=-1)
        + (shifted * (shifted.log() - midpoint.log())).sum(dim=-1)
    )
    return float(divergence.mean().item())


def _validation_latency(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    batch_size: int,
    device: str,
) -> float:
    batch = inputs[: min(batch_size, inputs.shape[0])].to(device=device)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for _ in range(3):
            model(batch)
        if device == "cuda":
            torch.cuda.synchronize()
        started = perf_counter()
        for _ in range(10):
            model(batch)
        if device == "cuda":
            torch.cuda.synchronize()
    model.train(was_training)
    return 100.0 * (perf_counter() - started)


def _safe(key: str) -> str:
    return key.replace(":", "_").replace("/", "_")


def _result_path(root: Path, job: HeadroomScreenJob, *, failed: bool) -> Path:
    return root / ("failed" if failed else "completed") / f"{_safe(job.key)}.json"


def _write_result(
    root: Path,
    job: HeadroomScreenJob,
    row: dict[str, object],
    *,
    failed: bool,
) -> None:
    path = _result_path(root, job, failed=failed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
