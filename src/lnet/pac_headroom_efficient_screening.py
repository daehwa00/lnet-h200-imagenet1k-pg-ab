# ruff: noqa: EM101, TRY003
from __future__ import annotations

import gc
import json
import math
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Final, Literal, cast

import torch
from torch import Tensor, nn
from torch.nn import functional

from .pac_external_benchmarks import (
    ExternalBenchmarkConfig,
    _loss,  # pyright: ignore[reportPrivateUsage]
    _predict,  # pyright: ignore[reportPrivateUsage]
    _seed_everything,  # pyright: ignore[reportPrivateUsage]
    _train_model,  # pyright: ignore[reportPrivateUsage]
    external_metric_bundle,
)
from .pac_external_tasks import (
    ExternalDatasetName,
    ExternalObjective,
    ExternalTask,
    load_external_task,
)
from .pac_headroom_efficient_models import (
    EFFICIENT_HEADROOM_SPECS,
    AuxiliaryDistilledPAC,
    EfficientHeadroomSpec,
    build_efficient_headroom_classifier,
)
from .pac_metrics import count_parameters
from .pac_types import PACDevice, PACExperimentConfig

DEFAULT_EFFICIENT_ROOT: Final = Path(".omx/results/pac-headroom-efficient-20260712")
_SCREEN_DATASETS: Final[tuple[ExternalDatasetName, ...]] = (
    "audioset-balanced",
    "ettm2",
    "cwru",
)


@dataclass(frozen=True, slots=True)
class EfficientScreenJob:
    key: str
    dataset: ExternalDatasetName
    spec: EfficientHeadroomSpec
    seed: int
    epochs: int = 20
    batch_size: int = 64
    patience: int = 4
    estimated_seconds: float = 90.0


def efficient_jobs(
    seeds: tuple[int, ...] = (7,),
    specs: tuple[EfficientHeadroomSpec, ...] = EFFICIENT_HEADROOM_SPECS,
) -> list[EfficientScreenJob]:
    seconds = {"cwru": 18.0, "audioset-balanced": 90.0, "ettm2": 90.0}
    factors = {"AD": 1.2}
    return [
        EfficientScreenJob(
            key=f"external:{dataset}:{spec}:seed{seed}",
            dataset=dataset,
            spec=spec,
            seed=seed,
            estimated_seconds=seconds[dataset] * factors.get(spec, 1.1),
        )
        for dataset in _SCREEN_DATASETS
        for spec in specs
        for seed in seeds
    ]


def enqueue_efficient(
    root: Path = DEFAULT_EFFICIENT_ROOT,
    *,
    seeds: tuple[int, ...] = (7,),
    specs: tuple[EfficientHeadroomSpec, ...] = EFFICIENT_HEADROOM_SPECS,
    workers: int = 3,
    prefix: str = "screen1",
) -> int:
    jobs = efficient_jobs(seeds, specs)
    shards: list[list[EfficientScreenJob]] = [[] for _ in range(workers)]
    loads = [0.0] * workers
    for job in sorted(jobs, key=lambda item: item.estimated_seconds, reverse=True):
        index = min(range(workers), key=loads.__getitem__)
        shards[index].append(job)
        loads[index] += job.estimated_seconds
    manifests = root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    for index, shard in enumerate(shards):
        ordered = sorted(shard, key=lambda item: (item.estimated_seconds, item.key))
        (manifests / f"{prefix}-worker-{index}.jsonl").write_text(
            "".join(json.dumps(asdict(job), sort_keys=True) + "\n" for job in ordered),
            encoding="utf-8",
        )
    contract = {
        "schema": "pac_headroom_efficient.v1",
        "baseline_root": ".omx/results/pac-headroom-fast-3seed-20260712",
        "specs": list(specs),
        "datasets": list(_SCREEN_DATASETS),
        "seeds": list(seeds),
        "jobs": len(jobs),
        "official_test_accessed": False,
        "ad_auxiliary_every_batches": 4,
        "ad_inference_path": "fine-only exact Revised PAC",
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / f"contract-{prefix}.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return len(jobs)


def run_efficient_manifest(root: Path, manifest: Path, *, device: str = "cuda") -> None:
    jobs = [EfficientScreenJob(**json.loads(line)) for line in manifest.read_text().splitlines()]
    for job in jobs:
        completed = _result_path(root, job, failed=False)
        if completed.exists():
            continue
        try:
            row = run_efficient_job(job, device=device)
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


def run_efficient_job(job: EfficientScreenJob, *, device: str) -> dict[str, object]:
    task = load_external_task(job.dataset, Path("data/external"))
    benchmark = ExternalBenchmarkConfig(
        data_root=Path("data/external"),
        output_root=DEFAULT_EFFICIENT_ROOT,
        datasets=(job.dataset,),
        models=("pac",),
        model_dim=64,
        modes=16,
        epochs=job.epochs,
        batch_size=job.batch_size,
        learning_rate=1.0e-3,
        weight_decay=1.0e-4,
        grad_clip_norm=1.0,
        patience=job.patience,
        seeds=(job.seed,),
        device=cast("Literal['auto', 'cpu', 'cuda']", device),
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
    model = build_efficient_headroom_classifier(
        job.spec,
        config,
        task.output_dim,
        objective="regression" if task.objective == "forecasting" else "classification",
    ).to(device=device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    if isinstance(model, AuxiliaryDistilledPAC):
        best_epoch, validation_loss, auxiliary_batches, total_batches = _train_auxiliary(
            model, task, benchmark, device
        )
    else:
        best_epoch, validation_loss = _train_model(model, task, benchmark, device, job.seed)
        auxiliary_batches = 0
        total_batches = 0
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
    latency_ms = _validation_latency(model, task.validation_inputs, job.batch_size, device)
    peak_memory_mb = float(torch.cuda.max_memory_allocated() / 1.0e6) if device == "cuda" else 0.0
    return {
        "job_key": job.key,
        **asdict(job),
        "status": "done",
        "evaluation_split": "validation",
        "official_test_accessed": False,
        "params_trainable": count_parameters(model),
        "params_inference": count_parameters(model) - int(job.spec == "AD"),
        "best_epoch": best_epoch,
        "validation_loss": validation_loss,
        "validation_loss_recomputed": measured_loss,
        "train_seconds": train_seconds,
        "latency_ms": latency_ms,
        "peak_memory_mb": peak_memory_mb,
        "auxiliary_batches": auxiliary_batches,
        "total_batches": total_batches,
        "auxiliary_fraction": auxiliary_batches / total_batches if total_batches else 0.0,
        **{f"validation_{name}": value for name, value in metrics.items()},
    }


def status_efficient(
    root: Path = DEFAULT_EFFICIENT_ROOT,
    *,
    seeds: tuple[int, ...] = (7,),
    specs: tuple[EfficientHeadroomSpec, ...] = EFFICIENT_HEADROOM_SPECS,
) -> dict[str, object]:
    expected = {job.key for job in efficient_jobs(seeds, specs)}
    completed = _keys(root / "completed")
    failed = _keys(root / "failed")
    return {
        "expected": len(expected),
        "completed": len(expected & completed),
        "failed": len(expected & failed),
        "done": expected <= completed and not (expected & failed),
    }


def _train_auxiliary(
    model: AuxiliaryDistilledPAC,
    task: ExternalTask,
    config: ExternalBenchmarkConfig,
    device: str,
) -> tuple[int, float, int, int]:
    train_inputs = task.train_inputs
    train_targets = task.train_targets
    validation_inputs = task.validation_inputs
    validation_targets = task.validation_targets
    objective = task.objective
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        fused=device == "cuda",
    )
    generator = torch.Generator().manual_seed(config.seeds[0])
    best_state: dict[str, Tensor] | None = None
    best_validation = math.inf
    best_epoch = 0
    stale_epochs = 0
    auxiliary_batches = 0
    total_batches = 0
    for epoch in range(config.epochs):
        model.train()
        order = torch.randperm(train_inputs.shape[0], generator=generator)
        for indices in order.split(config.batch_size):
            inputs = train_inputs[indices].to(device=device)
            targets = train_targets[indices].to(device=device)
            optimizer.zero_grad(set_to_none=True)
            use_auxiliary = total_batches % 4 == 0
            if use_auxiliary:
                fine, coarse = model.forward_with_auxiliary(inputs)
                fine_loss = _loss(fine, targets, objective)
                coarse_loss = _loss(coarse, targets, objective)
                distillation = _distillation_loss(fine, coarse.detach(), objective)
                loss = fine_loss + 0.5 * coarse_loss + 0.25 * distillation
                auxiliary_batches += 1
            else:
                loss = _loss(model(inputs), targets, objective)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()
            model.post_optimizer_step()
            total_batches += 1
        validation_logits, validation_targets = _predict(
            model,
            validation_inputs,
            validation_targets,
            config.batch_size,
            device,
        )
        validation_loss = float(_loss(validation_logits, validation_targets, objective).item())
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("auxiliary training produced no validation checkpoint")
    model.load_state_dict(best_state)
    model.finalize_constraints()
    return best_epoch, best_validation, auxiliary_batches, total_batches


def _distillation_loss(fine: Tensor, teacher: Tensor, objective: ExternalObjective) -> Tensor:
    temperature = 2.0
    if objective == "multiclass":
        return temperature**2 * functional.kl_div(
            functional.log_softmax(fine / temperature, dim=-1),
            functional.softmax(teacher / temperature, dim=-1),
            reduction="batchmean",
        )
    if objective == "multilabel":
        return temperature**2 * functional.binary_cross_entropy_with_logits(
            fine / temperature,
            torch.sigmoid(teacher / temperature),
        )
    return functional.mse_loss(fine, teacher)


def _validation_latency(
    model: nn.Module,
    inputs: Tensor,
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


def _result_path(root: Path, job: EfficientScreenJob, *, failed: bool) -> Path:
    safe = job.key.replace(":", "_").replace("/", "_")
    return root / ("failed" if failed else "completed") / f"{safe}.json"


def _write_result(
    root: Path,
    job: EfficientScreenJob,
    row: dict[str, object],
    *,
    failed: bool,
) -> None:
    path = _result_path(root, job, failed=failed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        str(json.loads(result.read_text(encoding="utf-8"))["job_key"])
        for result in path.glob("*.json")
    }
