"""Validation-only confirmatory campaign for ALPHABET's two-scan structure."""

from __future__ import annotations

import copy
import gc
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median, stdev
from time import perf_counter
from typing import TYPE_CHECKING, Any, Final, cast

import torch
from scipy.stats import t as student_t
from torch import Tensor

from .pac_campaign_utils import seed_everything
from .pac_confirmatory_baselines import confirmatory_trial_spec
from .pac_eval_sections import clean_validation_classification_task
from .pac_final_two_scan_ablation import (
    VARIANTS,
    FinalTwoScanAblation,
    FinalTwoScanVariant,
    trainable_parameter_count,
)
from .pac_metrics import count_parameters
from .pac_real_data import ensure_ucr_train_only
from .pac_training import classification_metric_bundle, train_classifier
from .pac_types import PACDevice, PACExperimentConfig

if TYPE_CHECKING:
    from collections.abc import Mapping


DEFAULT_ROOT: Final = Path(".omx/results/pac-final-two-scan-ablation-20260727")
DEFAULT_DATA_ROOT: Final = Path(".omx/data/ucr")
DEFAULT_DATASETS: Final = (
    "GunPoint",
    "ECG200",
    "FordA",
    "CricketX",
    "Wafer",
)
DEFAULT_SEEDS: Final = (23, 31, 43, 47, 59)
MODEL_DIM: Final = 64
MODES: Final = 16
TRIAL: Final = 4
EPOCHS: Final = 100


@dataclass(frozen=True, slots=True)
class TwoScanJob:
    dataset: str
    variant: FinalTwoScanVariant
    seed: int

    @property
    def key(self) -> str:
        return f"{self.dataset}:{self.variant}:seed{self.seed}"


def jobs(
    datasets: tuple[str, ...] = DEFAULT_DATASETS,
    variants: tuple[FinalTwoScanVariant, ...] = VARIANTS,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> list[TwoScanJob]:
    return [
        TwoScanJob(dataset, variant, seed)
        for dataset in datasets
        for variant in variants
        for seed in seeds
    ]


def prepare(
    root: Path = DEFAULT_ROOT,
    *,
    datasets: tuple[str, ...] = DEFAULT_DATASETS,
    variants: tuple[FinalTwoScanVariant, ...] = VARIANTS,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> dict[str, object]:
    unknown = set(variants) - set(VARIANTS)
    if unknown:
        message = f"unknown two-scan variants: {sorted(unknown)}"
        raise ValueError(message)
    active = jobs(datasets, variants, seeds)
    root.mkdir(parents=True, exist_ok=True)
    (root / "queue.jsonl").write_text(
        "".join(json.dumps(asdict(job), sort_keys=True) + "\n" for job in active),
        encoding="utf-8",
    )
    contract: dict[str, object] = {
        "schema": "pac_final_two_scan_ablation_contract.v1",
        "purpose": "isolate the contribution of ALPHABET's asymmetric two-scan hierarchy",
        "model": "final radial-log modal-only ALPHABET",
        "evaluation_split": "TRAIN-derived validation only",
        "official_test_accessed": False,
        "selection": "datasets, variants, seeds, and optimizer fixed before execution",
        "datasets": list(datasets),
        "variants": list(variants),
        "seeds": list(seeds),
        "capacity": {"model_dim": MODEL_DIM, "modes": MODES},
        "one_scan_control": {
            "modes": "M'=2M, so 7M'=14M modal descriptor coordinates",
            "feature_width": "D' chosen nearest the full trainable parameter count",
            "parameter_tolerance": "at most 1%",
        },
        "optimizer_trial": TRIAL,
        "epochs": EPOCHS,
        "jobs": len(active),
        "primary_comparison": (
            "full minus wider_one_scan across paired task-level five-seed means"
        ),
    }
    _write_json(root / "contract.json", contract)
    return status(root)


def run(
    root: Path = DEFAULT_ROOT,
    *,
    device: PACDevice = "cuda",
    data_root: Path = DEFAULT_DATA_ROOT,
    shard_index: int = 0,
    shard_count: int = 1,
) -> dict[str, object]:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        message = (
            "shard_index must satisfy 0 <= shard_index < shard_count; "
            f"received index={shard_index}, count={shard_count}"
        )
        raise ValueError(message)
    all_jobs = [
        TwoScanJob(**json.loads(line))
        for line in (root / "queue.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    queued = [
        job for position, job in enumerate(all_jobs) if position % shard_count == shard_index
    ]
    for job in queued:
        output = _result_path(root, job)
        if output.exists():
            continue
        started = perf_counter()
        try:
            row, checkpoint = run_job(job, device=device, data_root=data_root)
            row["elapsed_seconds"] = perf_counter() - started
            checkpoint_path = root / "checkpoints" / f"{_safe_key(job.key)}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, checkpoint_path)
            row["checkpoint"] = "<local-path>"
            _write_json(output, row)
        except Exception as error:
            failure: dict[str, object] = {
                "schema": "pac_final_two_scan_ablation_failure.v1",
                "job_key": job.key,
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "official_test_accessed": False,
            }
            _write_json(root / "failed" / f"{_safe_key(job.key)}.json", failure)
            raise
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    current = status(root)
    return report(root) if current["done"] else current


def run_job(
    job: TwoScanJob,
    *,
    device: PACDevice,
    data_root: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    runtime_device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
    seed_everything(job.seed)
    task = clean_validation_classification_task(
        ensure_ucr_train_only(job.dataset, data_root, allow_download=True),
        job.seed,
    )
    spec = confirmatory_trial_spec("pac_tf", TRIAL)
    config = PACExperimentConfig(
        task.train_inputs.shape[0],
        task.validation_inputs.shape[0],
        0,
        task.train_inputs.shape[1],
        raw_input_dim=task.train_inputs.shape[-1],
        output_dim=task.class_count,
        model_dim=MODEL_DIM,
        modes=MODES,
        epochs=EPOCHS,
        batch_size=spec.batch_size,
        learning_rate=spec.learning_rate,
        weight_decay=spec.weight_decay,
        grad_clip_norm=spec.grad_clip_norm,
        seeds=(job.seed,),
        device=cast("PACDevice", runtime_device),
        optimizer_mode="fused" if runtime_device == "cuda" else "default",
    )
    model = FinalTwoScanAblation(
        config,
        task.class_count,
        variant=job.variant,
        objective="classification",
    ).to(runtime_device)
    outcome = train_classifier(
        model,
        task,
        config,
        runtime_device,
        job.seed,
        evaluate_test=False,
        restore_best_validation=True,
    )
    metrics = classification_metric_bundle(
        model,
        task.validation_inputs.to(runtime_device),
        task.validation_labels.to(runtime_device),
        batch_size=spec.batch_size,
    )
    timing = measure_latency(
        model,
        task.train_inputs,
        task.train_labels,
        batch_size=min(32, task.train_inputs.shape[0]),
        grad_clip_norm=spec.grad_clip_norm,
        device=runtime_device,
    )
    audit = model.capacity_audit
    row: dict[str, object] = {
        "schema": "pac_final_two_scan_ablation_result.v1",
        "job_key": job.key,
        "status": "done",
        "dataset": job.dataset,
        "variant": job.variant,
        "seed": job.seed,
        "evaluation_split": "TRAIN-derived validation only",
        "official_test_accessed": False,
        "train_count": int(task.train_inputs.shape[0]),
        "validation_count": int(task.validation_inputs.shape[0]),
        "class_count": task.class_count,
        "sequence_length": int(task.train_inputs.shape[1]),
        "raw_input_dim": int(task.train_inputs.shape[-1]),
        "model_dim": model.model_dim,
        "modes": model.modes,
        "epochs": EPOCHS,
        "best_epoch": outcome.best_epoch,
        "params_trainable": count_parameters(model),
        "target_parameters": audit.target_parameters,
        "parameter_relative_error": audit.relative_error,
        "validation_accuracy": metrics.accuracy,
        "validation_balanced_accuracy": metrics.balanced_accuracy,
        **timing,
    }
    checkpoint: dict[str, Any] = {
        "schema": "pac_final_two_scan_ablation_checkpoint.v1",
        "job": asdict(job),
        "config": {
            "raw_input_dim": int(task.train_inputs.shape[-1]),
            "output_dim": task.class_count,
            "sequence_length": int(task.train_inputs.shape[1]),
            "model_dim": model.model_dim,
            "modes": model.modes,
        },
        "capacity_audit": asdict(audit),
        "best_epoch": outcome.best_epoch,
        "evaluation_split": "TRAIN-derived validation only",
        "official_test_accessed": False,
        "model_state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
    }
    return row, checkpoint


def audit_latency(
    root: Path = DEFAULT_ROOT,
    *,
    device: PACDevice = "cuda",
    data_root: Path = DEFAULT_DATA_ROOT,
) -> dict[str, object]:
    """Run a dedicated single-process latency audit after model fitting."""

    contract = json.loads((root / "contract.json").read_text(encoding="utf-8"))
    datasets = tuple(str(value) for value in contract["datasets"])
    variants = cast("tuple[FinalTwoScanVariant, ...]", tuple(contract["variants"]))
    seed = int(contract["seeds"][0])
    runtime_device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
    spec = confirmatory_trial_spec("pac_tf", TRIAL)
    rows: list[dict[str, object]] = []
    for dataset in datasets:
        task = clean_validation_classification_task(
            ensure_ucr_train_only(dataset, data_root, allow_download=True),
            seed,
        )
        config = PACExperimentConfig(
            task.train_inputs.shape[0],
            task.validation_inputs.shape[0],
            0,
            task.train_inputs.shape[1],
            raw_input_dim=task.train_inputs.shape[-1],
            output_dim=task.class_count,
            model_dim=MODEL_DIM,
            modes=MODES,
            epochs=EPOCHS,
            batch_size=spec.batch_size,
            learning_rate=spec.learning_rate,
            weight_decay=spec.weight_decay,
            grad_clip_norm=spec.grad_clip_norm,
            seeds=(seed,),
            device=cast("PACDevice", runtime_device),
            optimizer_mode="fused" if runtime_device == "cuda" else "default",
        )
        for variant in variants:
            seed_everything(seed)
            model = FinalTwoScanAblation(
                config,
                task.class_count,
                variant=variant,
                objective="classification",
            ).to(runtime_device)
            timing = measure_latency(
                model,
                task.train_inputs,
                task.train_labels,
                batch_size=min(32, task.train_inputs.shape[0]),
                grad_clip_norm=spec.grad_clip_norm,
                device=runtime_device,
            )
            rows.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "seed": seed,
                    "sequence_length": int(task.train_inputs.shape[1]),
                    "parameters": trainable_parameter_count(model),
                    **timing,
                }
            )
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    payload: dict[str, object] = {
        "schema": "pac_final_two_scan_sequential_latency.v1",
        "execution": "single process; variants measured sequentially",
        "official_test_accessed": False,
        "device": runtime_device,
        "seed": seed,
        "rows": rows,
    }
    _write_json(root / "reports" / "sequential_latency.json", payload)
    return payload


def measure_latency(
    model: FinalTwoScanAblation,
    inputs: Tensor,
    labels: Tensor,
    *,
    batch_size: int,
    grad_clip_norm: float,
    device: str,
    warmups: int = 3,
    repeats: int = 7,
) -> dict[str, float]:
    """Measure eager inference and complete-step latency on one fixed batch."""

    batch_inputs = inputs[:batch_size].to(device)
    batch_labels = labels[:batch_size].to(device)
    model.eval()

    def inference_call() -> None:
        with torch.no_grad():
            model(batch_inputs)

    inference = _timed_calls(inference_call, device, warmups, repeats)

    training_model = copy.deepcopy(model).train()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in training_model.parameters() if parameter.requires_grad),
        lr=1.0e-3,
        fused=device.startswith("cuda"),
    )

    def complete_step() -> None:
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(training_model(batch_inputs), batch_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(training_model.parameters(), grad_clip_norm)
        optimizer.step()
        training_model.post_optimizer_step()

    training = _timed_calls(complete_step, device, warmups, repeats)
    return {
        "latency_batch_size": batch_size,
        "inference_latency_ms": inference,
        "complete_step_latency_ms": training,
    }


def _timed_calls(
    call: object,
    device: str,
    warmups: int,
    repeats: int,
) -> float:
    function = cast("Any", call)
    for _ in range(warmups):
        function()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        samples: list[float] = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            function()
            end.record()
            end.synchronize()
            samples.append(float(start.elapsed_time(end)))
        return median(samples)
    samples = []
    for _ in range(repeats):
        started = perf_counter()
        function()
        samples.append(1.0e3 * (perf_counter() - started))
    return median(samples)


def report(root: Path = DEFAULT_ROOT) -> dict[str, object]:  # noqa: C901, PLR0912
    rows: list[dict[str, Any]] = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((root / "completed").glob("*.json"))
    ]
    contract = json.loads((root / "contract.json").read_text(encoding="utf-8"))
    datasets = tuple(str(value) for value in contract["datasets"])
    variants = tuple(str(value) for value in contract["variants"])
    latency_path = root / "reports" / "sequential_latency.json"
    latency_rows: list[dict[str, Any]] = []
    if latency_path.exists():
        latency_payload = json.loads(latency_path.read_text(encoding="utf-8"))
        latency_rows = list(latency_payload["rows"])
    latency_by_cell = {
        (str(row["dataset"]), str(row["variant"])): row for row in latency_rows
    }
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["dataset"]), str(row["variant"])), []).append(row)

    cells: dict[str, dict[str, Any]] = {}
    for dataset in datasets:
        cells[dataset] = {}
        for variant in variants:
            selected = grouped.get((dataset, variant), [])
            if not selected:
                continue
            cells[dataset][variant] = {
                "seeds": len(selected),
                "balanced_accuracy": _mean_sd(
                    [float(row["validation_balanced_accuracy"]) for row in selected]
                ),
                "parameters": int(selected[0]["params_trainable"]),
                "model_dim": int(selected[0]["model_dim"]),
                "modes": int(selected[0]["modes"]),
                "inference_latency_ms": (
                    float(latency_by_cell[(dataset, variant)]["inference_latency_ms"])
                    if (dataset, variant) in latency_by_cell
                    else None
                ),
                "complete_step_latency_ms": (
                    float(latency_by_cell[(dataset, variant)]["complete_step_latency_ms"])
                    if (dataset, variant) in latency_by_cell
                    else None
                ),
            }

    variant_summary: dict[str, Any] = {}
    dataset_means: dict[tuple[str, str], float] = {
        (dataset, variant): float(cells[dataset][variant]["balanced_accuracy"]["mean"])
        for dataset in cells
        for variant in cells[dataset]
    }
    ranks: dict[str, list[float]] = {variant: [] for variant in variants}
    for dataset in datasets:
        present = [
            (variant, dataset_means[(dataset, variant)])
            for variant in variants
            if (dataset, variant) in dataset_means
        ]
        for variant, value in present:
            ranks[variant].append(_average_rank(value, [other for _, other in present]))
    for variant in variants:
        selected = [row for row in rows if row["variant"] == variant]
        task_values = [
            dataset_means[(dataset, variant)]
            for dataset in datasets
            if (dataset, variant) in dataset_means
        ]
        variant_summary[variant] = {
            "tasks": len(task_values),
            "mean_task_balanced_accuracy": mean(task_values) if task_values else None,
            "mean_rank": mean(ranks[variant]) if ranks[variant] else None,
            "median_parameters": (
                median(int(row["params_trainable"]) for row in selected) if selected else None
            ),
            "mean_dataset_seed_sd": (
                mean(
                    float(cells[dataset][variant]["balanced_accuracy"]["sample_sd"])
                    for dataset in datasets
                    if variant in cells[dataset]
                )
                if task_values
                else None
            ),
            "inference_latency_geomean_ms": (
                _geometric_mean(
                    [
                        float(row["inference_latency_ms"])
                        for row in latency_rows
                        if row["variant"] == variant
                    ]
                )
                if any(row["variant"] == variant for row in latency_rows)
                else None
            ),
            "complete_step_latency_geomean_ms": (
                _geometric_mean(
                    [
                        float(row["complete_step_latency_ms"])
                        for row in latency_rows
                        if row["variant"] == variant
                    ]
                )
                if any(row["variant"] == variant for row in latency_rows)
                else None
            ),
        }

    comparisons: dict[str, Any] = {}
    seed_pair_comparisons: dict[str, Any] = {}
    full_by_key = {
        (str(row["dataset"]), int(row["seed"])): row
        for row in rows
        if row["variant"] == "full"
    }
    for variant in variants:
        if variant == "full":
            continue
        seed_differences: list[float] = []
        for row in rows:
            if row["variant"] != variant:
                continue
            full = full_by_key.get((str(row["dataset"]), int(row["seed"])))
            if full is None:
                continue
            seed_differences.append(
                float(full["validation_balanced_accuracy"])
                - float(row["validation_balanced_accuracy"])
            )
        task_differences = [
            dataset_means[(dataset, "full")] - dataset_means[(dataset, variant)]
            for dataset in datasets
            if (dataset, "full") in dataset_means and (dataset, variant) in dataset_means
        ]
        comparisons[variant] = _paired_summary(task_differences)
        seed_pair_comparisons[variant] = _paired_summary(seed_differences)

    payload: dict[str, object] = {
        "schema": "pac_final_two_scan_ablation_report.v1",
        "official_test_accessed": False,
        "rows": len(rows),
        "cells": cells,
        "variant_summary": variant_summary,
        "full_minus_ablation": comparisons,
        "descriptive_seed_pair_effects": seed_pair_comparisons,
        "primary_comparison": comparisons.get("wider_one_scan"),
        "latency_audit": (
            "single-process sequential audit" if latency_rows else "not yet run"
        ),
        "status": status(root),
    }
    _write_json(root / "reports" / "summary.json", payload)
    _write_markdown(root / "reports" / "SUMMARY.md", payload, datasets, variants)
    return payload


def status(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    queue = root / "queue.jsonl"
    expected = sum(1 for line in queue.read_text(encoding="utf-8").splitlines() if line)
    completed = len(list((root / "completed").glob("*.json")))
    failed = len(list((root / "failed").glob("*.json")))
    return {
        "expected": expected,
        "completed": completed,
        "failed": failed,
        "remaining": expected - completed,
        "done": completed == expected and failed == 0,
    }


def _paired_summary(values: list[float]) -> dict[str, object]:
    if not values:
        return {"pairs": 0, "mean": None, "sample_sd": None, "ci95": None}
    average = mean(values)
    sample_sd = stdev(values) if len(values) > 1 else 0.0
    half_width = (
        float(student_t.ppf(0.975, len(values) - 1)) * sample_sd / math.sqrt(len(values))
        if len(values) > 1
        else 0.0
    )
    return {
        "pairs": len(values),
        "mean": average,
        "sample_sd": sample_sd,
        "ci95": [average - half_width, average + half_width],
        "full_wins": sum(value > 0.0 for value in values),
        "ties": sum(value == 0.0 for value in values),
        "full_losses": sum(value < 0.0 for value in values),
    }


def _average_rank(value: float, values: list[float], tolerance: float = 1.0e-12) -> float:
    better = sum(other > value + tolerance for other in values)
    tied = sum(abs(other - value) <= tolerance for other in values)
    return 1.0 + better + 0.5 * (tied - 1)


def _mean_sd(values: list[float]) -> dict[str, float]:
    return {
        "mean": mean(values),
        "sample_sd": stdev(values) if len(values) > 1 else 0.0,
    }


def _geometric_mean(values: list[float]) -> float:
    return math.exp(mean(math.log(max(value, 1.0e-12)) for value in values))


def _write_markdown(
    path: Path,
    payload: Mapping[str, object],
    datasets: tuple[str, ...],
    variants: tuple[str, ...],
) -> None:
    cells = cast("dict[str, dict[str, Any]]", payload["cells"])
    summary = cast("dict[str, dict[str, Any]]", payload["variant_summary"])
    lines = [
        "# Final two-scan ALPHABET ablation",
        "",
        "TRAIN-derived validation only; official UCR TEST was never loaded.",
        "",
        "| Variant | Mean BAcc. | Mean rank | Median params | Seed SD | Infer. ms | Step ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in variants:
        row = summary[variant]
        inference_latency = row["inference_latency_geomean_ms"]
        step_latency = row["complete_step_latency_geomean_ms"]
        inference_text = (
            f"{float(inference_latency):.4f}" if inference_latency is not None else "--"
        )
        step_text = f"{float(step_latency):.4f}" if step_latency is not None else "--"
        lines.append(
            f"| {variant} | {row['mean_task_balanced_accuracy']:.4f} | "
            f"{row['mean_rank']:.3f} | {row['median_parameters']:.0f} | "
            f"{row['mean_dataset_seed_sd']:.4f} | "
            f"{inference_text} | {step_text} |"
        )
    lines.extend(("", "## Dataset means", ""))
    lines.append("| Dataset | " + " | ".join(variants) + " |")
    lines.append("|---|" + "|".join("---:" for _ in variants) + "|")
    for dataset in datasets:
        values = [
            f"{float(cells[dataset][variant]['balanced_accuracy']['mean']):.4f}"
            for variant in variants
        ]
        lines.append(f"| {dataset} | " + " | ".join(values) + " |")
    comparisons = cast("dict[str, dict[str, Any]]", payload["full_minus_ablation"])
    lines.extend(("", "## Paired full-model effects", ""))
    for variant, row in comparisons.items():
        ci = row["ci95"]
        if row["mean"] is None or ci is None:
            lines.append(f"- Full minus `{variant}`: unavailable (no paired full rows).")
            continue
        lines.append(
            f"- Full minus `{variant}`: {float(row['mean']):+.4f}, "
            f"95% CI [{float(ci[0]):+.4f}, {float(ci[1]):+.4f}], "
            f"{row['full_wins']}/{row['pairs']} task wins."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _safe_key(value: str) -> str:
    return value.replace(":", "__").replace("/", "_")


def _result_path(root: Path, job: TwoScanJob) -> Path:
    return root / "completed" / f"{_safe_key(job.key)}.json"


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "DEFAULT_DATASETS",
    "DEFAULT_DATA_ROOT",
    "DEFAULT_ROOT",
    "DEFAULT_SEEDS",
    "EPOCHS",
    "TwoScanJob",
    "audit_latency",
    "jobs",
    "measure_latency",
    "prepare",
    "report",
    "run",
    "run_job",
    "status",
]
