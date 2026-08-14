from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

import torch

from .pac_eval_sections import classification_task
from .pac_metrics import count_parameters
from .pac_overnight_io import append_csv_row, read_csv
from .pac_overnight_models import build_overnight_classifier
from .pac_real_data import ensure_ucr_dataset
from .pac_training import train_classifier

if TYPE_CHECKING:
    from .pac_types import PACClassificationTask, PACExperimentConfig
    from .tapped_prl_followup_schema import JsonValue


def run_real_baselines(
    output_root: Path,
    config: PACExperimentConfig,
    device: str,
    datasets: tuple[str, ...],
    models: tuple[str, ...],
    seeds: tuple[int, ...],
) -> None:
    path = output_root / "results" / "real_baselines_ecg5000_forda.csv"
    completed = _completed_real_rows(path)
    for dataset_name in datasets:
        dataset = ensure_ucr_dataset(dataset_name, Path(".omx/data/ucr"), allow_download=True)
        task = classification_task(dataset.name, dataset)
        for seed in seeds:
            for model_name in models:
                key = (task.label, seed, model_name)
                if key in completed:
                    continue
                row = _real_row(model_name, seed, config, task, device)
                append_csv_row(path, row)
                completed.add(key)
    _write_report(output_root)


def _completed_real_rows(path: Path) -> set[tuple[str, int, str]]:
    completed: set[tuple[str, int, str]] = set()
    for row in read_csv(path):
        dataset = row.get("dataset_or_task")
        seed = row.get("seed")
        model = row.get("model")
        if dataset is None or seed is None or model is None:
            continue
        try:
            completed.add((dataset, int(seed), model))
        except ValueError:
            continue
    return completed


def _real_row(
    model_name: str,
    seed: int,
    config: PACExperimentConfig,
    task: PACClassificationTask,
    device: str,
) -> dict[str, JsonValue]:
    run_config = replace(config, seeds=(seed,), raw_input_dim=1, output_dim=task.class_count)
    model = build_overnight_classifier(model_name, run_config, task.class_count)
    started_at = perf_counter()
    outcome = train_classifier(model, task, run_config, device, seed)
    eval_started = perf_counter()
    metrics = _classification_scores(
        model,
        task.test_inputs.to(device=device),
        task.test_labels.to(device=device),
        task.class_count,
    )
    return {
        "experiment_group": "real_baselines",
        "dataset_or_task": task.label,
        "seed": seed,
        "model": model_name,
        "variant": model_name,
        "params_trainable": count_parameters(model),
        "params_total": sum(parameter.numel() for parameter in model.parameters()),
        "sequence_length": task.test_inputs.shape[1],
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "best_val_loss": outcome.validation_loss,
        "test_loss": outcome.test_loss,
        "test_accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "elapsed_train_time": outcome.elapsed_time,
        "elapsed_eval_time": perf_counter() - eval_started,
        "elapsed_time": perf_counter() - started_at,
        "notes": "completed",
    }


def _classification_scores(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    class_count: int,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        predictions = torch.argmax(model(inputs), dim=-1).detach().cpu()
    model.train(was_training)
    labels_cpu = labels.detach().cpu()
    per_class_f1: list[float] = []
    recalls: list[float] = []
    weights: list[int] = []
    for class_index in range(class_count):
        actual = labels_cpu == class_index
        predicted = predictions == class_index
        true_positive = int((actual & predicted).sum().item())
        support = int(actual.sum().item())
        precision = true_positive / max(int(predicted.sum().item()), 1)
        recall = true_positive / max(support, 1)
        f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
        per_class_f1.append(f1)
        recalls.append(recall)
        weights.append(support)
    total = max(sum(weights), 1)
    return {
        "accuracy": float((predictions == labels_cpu).to(torch.float32).mean().item()),
        "macro_f1": sum(per_class_f1) / max(len(per_class_f1), 1),
        "weighted_f1": sum(
            score * weight for score, weight in zip(per_class_f1, weights, strict=True)
        )
        / total,
        "balanced_accuracy": sum(recalls) / max(len(recalls), 1),
    }


def _write_report(output_root: Path) -> None:
    path = output_root / "reports" / "overnight_real_baselines.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Real Benchmark Expanded Baselines\n\nreal_baseline_status: mixed\n",
        encoding="utf-8",
    )
