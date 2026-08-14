from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

import torch
from torch import nn

from .pac_builders import build_classifier_model, build_regression_model
from .pac_data_split import stratified_partition_indices
from .pac_metrics import count_parameters
from .pac_real_data import ensure_ucr_dataset, load_ucr_dataset, write_tiny_ucr_fixture
from .pac_training import classification_metrics, train_classifier
from .pac_types import PACClassificationTask, PACExperimentConfig, PACMode, PACModelName, UCRDataset

if TYPE_CHECKING:
    from .tapped_prl_followup_schema import JsonRow


def efficiency(config: PACExperimentConfig, device: str, mode: PACMode) -> list[JsonRow]:
    lengths = (32, 64) if mode == "smoke" else (128, 256, 512, 1024, 2048, 4096)
    rows: list[JsonRow] = []
    for model_name in ("pac_full", "pac_lite", "gru", "transformer"):
        rows.extend(
            efficiency_row(
                build_regression_model(model_name, config).to(device=device),
                model_name,
                config,
                device,
                length,
            )
            for length in lengths
        )
    return rows


def real_signal(config: PACExperimentConfig, device: str, mode: PACMode) -> list[JsonRow]:
    root = config.output_dir / "artifacts" / "ucr" if mode == "smoke" else Path(".omx/data/ucr")
    if mode == "smoke":
        write_tiny_ucr_fixture(root)
        datasets = (load_ucr_dataset("Tiny", root),)
    else:
        datasets = _real_datasets(root)
    rows: list[JsonRow] = []
    for dataset in datasets:
        task = classification_task(dataset.name, dataset)
        rows.extend(_real_row(model_name, config, task, device) for model_name in _real_models())
    return rows


def efficiency_row(
    model: nn.Module,
    model_name: str,
    config: PACExperimentConfig,
    device: str,
    length: int,
) -> JsonRow:
    inputs = torch.randn(4, length, config.raw_input_dim, device=device)
    targets = torch.randn(4, length, config.output_dim, device=device)
    model.train()
    for _ in range(2):
        model.zero_grad(set_to_none=True)
        torch.nn.functional.mse_loss(model(inputs), targets).backward()
    _sync_if_cuda(device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model.zero_grad(set_to_none=True)
    started_at = perf_counter()
    output = model(inputs)
    _sync_if_cuda(device)
    forward_time = perf_counter() - started_at
    started_at = perf_counter()
    torch.nn.functional.mse_loss(output, targets).backward()
    _sync_if_cuda(device)
    backward_time = perf_counter() - started_at
    memory = torch.cuda.max_memory_allocated() if device == "cuda" else None
    tokens = inputs.shape[0] * inputs.shape[1]
    return {
        "model": model_name,
        "implementation": "naive_loop",
        "sequence_length": length,
        "params": count_parameters(model),
        "forward_time": forward_time,
        "backward_time": backward_time,
        "tokens_per_second": tokens / max(forward_time + backward_time, 1.0e-12),
        "peak_memory": memory,
    }


def _sync_if_cuda(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def classification_task(name: str, dataset: UCRDataset) -> PACClassificationTask:
    train, test = _normalize(dataset.train_inputs, dataset.test_inputs)
    split = max(1, train.shape[0] // 5)
    return PACClassificationTask(
        name,
        train[split:],
        dataset.train_labels[split:],
        train[:split],
        dataset.train_labels[:split],
        test,
        dataset.test_labels,
        dataset.class_count,
    )


def clean_validation_classification_task(
    dataset: UCRDataset,
    seed: int,
    validation_ratio: float = 0.2,
) -> PACClassificationTask:
    """Split raw official TRAIN first, then fit preprocessing on optimization data only."""
    if dataset.test_inputs.shape[0] or dataset.test_labels.shape[0]:
        message = (
            "clean validation requires a TRAIN-only UCR dataset; refusing a dataset "
            "whose official TEST tensors have already been loaded"
        )
        raise ValueError(message)
    train_indices, validation_indices = stratified_partition_indices(
        dataset.train_labels,
        validation_ratio,
        seed,
    )
    raw_train = dataset.train_inputs.index_select(0, train_indices)
    raw_validation = dataset.train_inputs.index_select(0, validation_indices)
    train, validation, test = _normalize_from_fold(
        raw_train,
        raw_validation,
        dataset.test_inputs,
    )
    return PACClassificationTask(
        dataset.name,
        train,
        dataset.train_labels.index_select(0, train_indices),
        validation,
        dataset.train_labels.index_select(0, validation_indices),
        test,
        dataset.test_labels,
        dataset.class_count,
    )


def full_train_classification_task(dataset: UCRDataset) -> PACClassificationTask:
    """Fit preprocessing on all official TRAIN examples for a frozen final refit."""
    train, _, test = _normalize_from_fold(
        dataset.train_inputs,
        dataset.train_inputs[:0],
        dataset.test_inputs,
    )
    return PACClassificationTask(
        dataset.name,
        train,
        dataset.train_labels,
        train[:0],
        dataset.train_labels[:0],
        test,
        dataset.test_labels,
        dataset.class_count,
    )


def _real_row(
    model_name: PACModelName,
    config: PACExperimentConfig,
    task: PACClassificationTask,
    device: str,
) -> JsonRow:
    model = build_classifier_model(model_name, config, task.class_count)
    outcome = train_classifier(model, task, config, device, config.seeds[0])
    accuracy, macro_f1 = classification_metrics(
        model,
        task.test_inputs.to(device=device),
        task.test_labels.to(device=device),
    )
    return {
        "dataset": task.label,
        "model": model_name,
        "params": count_parameters(model),
        "test_loss": outcome.test_loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "elapsed_time": outcome.elapsed_time,
    }


def _real_datasets(root: Path) -> tuple[UCRDataset, ...]:
    return tuple(
        ensure_ucr_dataset(name, root, allow_download=True) for name in ("ECG5000", "FordA")
    )


def _real_models() -> tuple[PACModelName, ...]:
    return "pac_full", "pac_lite", "gru"


def _finite_train_statistics(train: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit scalar normalization statistics from finite TRAIN observations only."""
    finite = torch.isfinite(train)
    if bool(finite.all()):
        observed = train
    else:
        observed = train[finite]
        if observed.numel() == 0:
            message = "cannot normalize a TRAIN fold with no finite observations"
            raise ValueError(message)
    mean = observed.mean()
    std = observed.std(unbiased=False).clamp_min(1.0e-6)
    return mean, std


def _normalize_with_statistics(
    inputs: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    normalized = (inputs - mean) / std
    if bool(torch.isfinite(inputs).all()):
        return normalized
    return torch.where(torch.isfinite(inputs), normalized, torch.zeros_like(normalized))


def _normalize(train: torch.Tensor, test: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean, std = _finite_train_statistics(train)
    return (
        _normalize_with_statistics(train, mean, std),
        _normalize_with_statistics(test, mean, std),
    )


def _normalize_from_fold(
    train: torch.Tensor,
    validation: torch.Tensor,
    test: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean, std = _finite_train_statistics(train)
    return (
        _normalize_with_statistics(train, mean, std),
        _normalize_with_statistics(validation, mean, std),
        _normalize_with_statistics(test, mean, std),
    )
