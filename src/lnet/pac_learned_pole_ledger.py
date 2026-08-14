"""Exact pole-level logit accounting for final ALPHABET checkpoints."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import TYPE_CHECKING, Final, cast

os.environ.setdefault("MPLBACKEND", "Agg")

import torch
from matplotlib import pyplot as plt
from scipy.stats import t as student_t
from torch import Tensor, nn

from .pac_eval_sections import clean_validation_classification_task
from .pac_final_two_scan_ablation import FinalTwoScanAblation, FinalTwoScanVariant
from .pac_real_data import ensure_ucr_train_only
from .pac_types import PACClassificationTask, PACExperimentConfig

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from matplotlib.axes import Axes

JsonDict = dict[str, object]

DEFAULT_CHECKPOINT_ROOT: Final = Path(
    ".omx/results/pac-final-two-scan-ablation-20260727/checkpoints"
)
DEFAULT_DATA_ROOT: Final = Path(".omx/data/ucr")
DEFAULT_OUTPUT_ROOT: Final = Path(".omx/results/pac-learned-pole-ledger-20260727")
DEFAULT_K_VALUES: Final = (1, 2, 4, 8, 16, 32)
DEFAULT_RANDOM_DRAWS: Final = 64
DEFAULT_AUDIT_DATASETS: Final = ("CricketX", "ECG200", "FordA", "GunPoint", "Wafer")
POLE_GROUPS: Final = 7
DAMPING_MIN: Final = 1.0e-3
DAMPING_MAX: Final = 2.0
MAIN_REPRESENTATIVE_DATASETS: Final = ("ECG200", "FordA")
DISPLAY_BANK_NAMES: Final = ("direct", "cascaded")


def _as_mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        message = f"expected a string-keyed mapping, got {type(value).__name__}"
        raise TypeError(message)
    return cast("Mapping[str, object]", value)


def _as_mapping_sequence(value: object) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, (list, tuple)):
        message = f"expected a sequence of mappings, got {type(value).__name__}"
        raise TypeError(message)
    return tuple(_as_mapping(item) for item in value)


def _as_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        message = f"expected an integer-valued number, got {type(value).__name__}"
        raise TypeError(message)
    integer = int(value)
    if integer != value:
        message = f"expected an integer-valued number, got {value}"
        raise ValueError(message)
    return integer


def _as_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        message = f"expected a number, got {type(value).__name__}"
        raise TypeError(message)
    return float(value)


def _as_float_matrix(value: object) -> list[list[float]]:
    if not isinstance(value, (list, tuple)):
        message = f"expected a numeric matrix, got {type(value).__name__}"
        raise TypeError(message)
    matrix: list[list[float]] = []
    for row in value:
        if not isinstance(row, (list, tuple)):
            message = f"expected a numeric matrix row, got {type(row).__name__}"
            raise TypeError(message)
        matrix.append([_as_float(entry) for entry in row])
    return matrix


@dataclass(frozen=True, slots=True)
class CapturedBatch:
    descriptors: Tensor
    logits: Tensor


@dataclass(frozen=True, slots=True)
class PoleMarginDecomposition:
    contributions: Tensor
    base_margin: Tensor
    margin: Tensor
    predicted_class: Tensor
    runner_up_class: Tensor
    max_residual: float


def pole_coordinate_indices(
    modes: int,
    pole_ids: Sequence[int],
) -> tuple[int, ...]:
    """Map direct/cascaded pole ids to their seven group-major descriptor coordinates."""
    indices: list[int] = []
    for pole_id in pole_ids:
        if not 0 <= pole_id < 2 * modes:
            message = f"pole id {pole_id} is outside [0, {2 * modes})"
            raise ValueError(message)
        bank, mode = divmod(pole_id, modes)
        start = bank * POLE_GROUPS * modes
        indices.extend(start + group * modes + mode for group in range(POLE_GROUPS))
    return tuple(indices)


def neutralize_poles(
    descriptors: Tensor,
    reference: Tensor,
    modes: int,
    pole_ids: Sequence[int],
) -> Tensor:
    """Replace selected pole coordinates by their optimization-fold reference mean."""
    indices = pole_coordinate_indices(modes, pole_ids)
    columns = list(indices)
    result = descriptors.clone()
    result[:, columns] = reference[columns]
    return result


def retain_poles(
    descriptors: Tensor,
    reference: Tensor,
    modes: int,
    pole_ids: Sequence[int],
) -> Tensor:
    """Retain selected pole coordinates and neutralize every other pole."""
    result = reference.expand_as(descriptors).clone()
    indices = pole_coordinate_indices(modes, pole_ids)
    columns = list(indices)
    result[:, columns] = descriptors[:, columns]
    return result


def decompose_margin(
    descriptors: Tensor,
    logits: Tensor,
    weight: Tensor,
    bias: Tensor,
    reference: Tensor,
    modes: int,
) -> PoleMarginDecomposition:
    """Exactly decompose the predicted-versus-runner-up margin by bank and pole."""
    expected_dim = 2 * POLE_GROUPS * modes
    if descriptors.ndim != 2 or descriptors.shape[1] != expected_dim:
        message = f"expected descriptor shape [N,{expected_dim}], got {tuple(descriptors.shape)}"
        raise ValueError(message)
    if weight.shape[1] != expected_dim:
        message = f"classifier width {weight.shape[1]} does not match descriptor {expected_dim}"
        raise ValueError(message)
    top_two = torch.topk(logits, k=2, dim=-1).indices
    predicted = top_two[:, 0]
    runner_up = top_two[:, 1]
    delta_weight = weight.index_select(0, predicted) - weight.index_select(0, runner_up)
    centered = descriptors - reference
    coordinate_contributions = delta_weight * centered
    contributions = coordinate_contributions.reshape(
        descriptors.shape[0],
        2,
        POLE_GROUPS,
        modes,
    ).sum(dim=2)
    base_margin = (
        bias.index_select(0, predicted)
        - bias.index_select(0, runner_up)
        + (delta_weight * reference).sum(dim=-1)
    )
    margin = logits.gather(1, predicted[:, None]).squeeze(1) - logits.gather(
        1,
        runner_up[:, None],
    ).squeeze(1)
    reconstructed = base_margin + contributions.sum(dim=(1, 2))
    residual = float(torch.max(torch.abs(reconstructed - margin)).item())
    return PoleMarginDecomposition(
        contributions=contributions,
        base_margin=base_margin,
        margin=margin,
        predicted_class=predicted,
        runner_up_class=runner_up,
        max_residual=residual,
    )


def capture_descriptors(
    model: FinalTwoScanAblation,
    inputs: Tensor,
    *,
    device: str,
    batch_size: int,
) -> CapturedBatch:
    """Capture the exact two-bank tensor consumed by the final affine head."""
    descriptors: list[Tensor] = []
    logits: list[Tensor] = []
    captured: list[Tensor] = []

    def capture(_module: nn.Module, arguments: tuple[object, ...]) -> None:
        if len(arguments) != 2 or not all(isinstance(value, Tensor) for value in arguments):
            message = "ALPHABET head did not receive writer and reader tensors"
            raise RuntimeError(message)
        writer, reader = cast("tuple[Tensor, Tensor]", arguments)
        captured.append(torch.cat((writer, reader), dim=-1).detach().cpu())

    handle = model.head.register_forward_pre_hook(capture)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for batch in inputs.split(batch_size):
                captured.clear()
                output = model(batch.to(device))
                if len(captured) != 1:
                    message = f"expected one affine-head capture, observed {len(captured)}"
                    raise RuntimeError(message)
                descriptors.append(captured[0])
                logits.append(output.detach().cpu())
    finally:
        handle.remove()
        model.train(was_training)
    return CapturedBatch(
        descriptors=torch.cat(descriptors, dim=0),
        logits=torch.cat(logits, dim=0),
    )


def balanced_accuracy(logits: Tensor, labels: Tensor) -> float:
    predictions = torch.argmax(logits, dim=-1).cpu()
    labels_cpu = labels.cpu()
    recalls = []
    for class_index in torch.unique(labels_cpu).tolist():
        selected = labels_cpu == int(class_index)
        recalls.append(float((predictions[selected] == int(class_index)).to(torch.float32).mean()))
    return mean(recalls) if recalls else 0.0


def prediction_agreement(reference_logits: Tensor, counterfactual_logits: Tensor) -> float:
    reference = torch.argmax(reference_logits, dim=-1)
    counterfactual = torch.argmax(counterfactual_logits, dim=-1)
    return float((reference == counterfactual).to(torch.float32).mean().item())


def original_class_margin(logits: Tensor, original_class: Tensor) -> Tensor:
    selected = logits.gather(1, original_class[:, None]).squeeze(1)
    alternatives = logits.clone()
    alternatives.scatter_(1, original_class[:, None], -torch.inf)
    return selected - alternatives.max(dim=-1).values


def rank_correlation(left: Tensor, right: Tensor) -> float:
    """Spearman correlation for untied mode-importance vectors."""
    if left.numel() != right.numel() or left.numel() < 2:
        return 0.0
    left_rank = torch.argsort(torch.argsort(left)).to(torch.float64)
    right_rank = torch.argsort(torch.argsort(right)).to(torch.float64)
    left_rank -= left_rank.mean()
    right_rank -= right_rank.mean()
    denominator = torch.linalg.vector_norm(left_rank) * torch.linalg.vector_norm(right_rank)
    if float(denominator.item()) == 0.0:
        return 0.0
    return float(torch.dot(left_rank, right_rank).div(denominator).item())


def _model_from_checkpoint(
    checkpoint: Mapping[str, object],
    task: PACClassificationTask,
    *,
    device: str,
    variant: FinalTwoScanVariant,
) -> FinalTwoScanAblation:
    config_values = _as_mapping(checkpoint["config"])
    config = PACExperimentConfig(
        int(task.train_inputs.shape[0]),
        int(task.validation_inputs.shape[0]),
        0,
        _as_int(config_values["sequence_length"]),
        raw_input_dim=_as_int(config_values["raw_input_dim"]),
        output_dim=_as_int(config_values["output_dim"]),
        model_dim=_as_int(config_values["model_dim"]),
        modes=_as_int(config_values["modes"]),
        batch_size=128,
        device="cuda" if device.startswith("cuda") else "cpu",
    )
    model = FinalTwoScanAblation(
        config,
        _as_int(config_values["output_dim"]),
        variant=variant,
        objective="classification",
    )
    state_dict = cast("Mapping[str, Tensor]", checkpoint["model_state_dict"])
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def _checkpoint_job(
    checkpoint: Mapping[str, object],
    *,
    expected_variant: FinalTwoScanVariant,
) -> tuple[str, int]:
    job = _as_mapping(checkpoint["job"])
    if str(job["variant"]) != expected_variant:
        message = f"expected a {expected_variant} checkpoint, got {job['variant']}"
        raise ValueError(message)
    if checkpoint.get("official_test_accessed") is not False:
        message = "interpretability audit refuses checkpoints that accessed official TEST"
        raise ValueError(message)
    return str(job["dataset"]), _as_int(job["seed"])


def _pole_locations(model: FinalTwoScanAblation) -> tuple[JsonDict, ...]:
    rows: list[JsonDict] = []
    for bank_index, (bank, block) in enumerate(
        (("writer", model.forward_block), ("reader", model.backward_block))
    ):
        raw_decay = cast("Tensor", block.raw_decay)
        raw_frequency = cast("Tensor", block.raw_frequency)
        damping_min = float(getattr(block, "damping_min", DAMPING_MIN))
        damping_max = float(getattr(block, "damping_max", DAMPING_MAX))
        damping = damping_min + (damping_max - damping_min) * torch.sigmoid(
            raw_decay.detach().cpu()
        )
        frequency_bound = float(getattr(block, "frequency_bound", math.pi))
        omega = frequency_bound * torch.tanh(raw_frequency.detach().cpu())
        rows.extend(
            {
                "bank": bank,
                "bank_index": bank_index,
                "mode": mode,
                "pole_id": bank_index * model.modes + mode,
                "damping": float(damping[mode].item()),
                "memory_scale": float(damping[mode].reciprocal().item()),
                "omega": float(omega[mode].item()),
                "cycles_per_token": float(omega[mode].item() / (2.0 * math.pi)),
            }
            for mode in range(model.modes)
        )
    return tuple(rows)


def _linear_logits(descriptors: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    return torch.nn.functional.linear(descriptors, weight, bias)


def _counterfactual_metrics(
    descriptors: Tensor,
    reference: Tensor,
    weight: Tensor,
    bias: Tensor,
    labels: Tensor,
    original_logits: Tensor,
    modes: int,
    pole_ids: Sequence[int],
    *,
    retain: bool,
) -> dict[str, float]:
    counterfactual = (
        retain_poles(descriptors, reference, modes, pole_ids)
        if retain
        else neutralize_poles(descriptors, reference, modes, pole_ids)
    )
    logits = _linear_logits(counterfactual, weight, bias)
    original_class = torch.argmax(original_logits, dim=-1)
    original_margin = original_class_margin(original_logits, original_class)
    new_margin = original_class_margin(logits, original_class)
    return {
        "balanced_accuracy": balanced_accuracy(logits, labels),
        "prediction_agreement": prediction_agreement(original_logits, logits),
        "mean_original_class_margin": float(new_margin.mean().item()),
        "mean_margin_drop": float((original_margin - new_margin).mean().item()),
        "flip_rate": float(
            (torch.argmax(original_logits, dim=-1) != torch.argmax(logits, dim=-1))
            .to(torch.float32)
            .mean()
            .item()
        ),
    }


def _random_metrics(
    descriptors: Tensor,
    reference: Tensor,
    weight: Tensor,
    bias: Tensor,
    labels: Tensor,
    original_logits: Tensor,
    modes: int,
    k: int,
    *,
    retain: bool,
    draws: int,
    seed: int,
) -> dict[str, float]:
    generator = random.Random(seed)  # noqa: S311 - deterministic analysis control
    values: dict[str, list[float]] = {}
    population = list(range(2 * modes))
    for _ in range(draws):
        selected = generator.sample(population, k)
        metrics = _counterfactual_metrics(
            descriptors,
            reference,
            weight,
            bias,
            labels,
            original_logits,
            modes,
            selected,
            retain=retain,
        )
        for key, value in metrics.items():
            values.setdefault(key, []).append(value)
    result = {key: mean(observations) for key, observations in values.items()}
    result["draws"] = float(draws)
    return result


def _bank_intervention_rows(
    descriptors: Tensor,
    reference: Tensor,
    weight: Tensor,
    bias: Tensor,
    labels: Tensor,
    original_logits: Tensor,
    modes: int,
    *,
    dataset: str,
    seed: int,
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for strategy, selected in (
        ("direct_bank", tuple(range(modes))),
        ("cascaded_bank", tuple(range(modes, 2 * modes))),
    ):
        for intervention, retain in (("neutralize", False), ("retain", True)):
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "k": modes,
                    "strategy": strategy,
                    "intervention": intervention,
                    **_counterfactual_metrics(
                        descriptors,
                        reference,
                        weight,
                        bias,
                        labels,
                        original_logits,
                        modes,
                        selected,
                        retain=retain,
                    ),
                }
            )
    return rows


def _select_representative(
    labels: Tensor,
    decomposition: PoleMarginDecomposition,
) -> int:
    correct = decomposition.predicted_class == labels
    candidates = torch.nonzero(correct, as_tuple=False).flatten()
    if candidates.numel() == 0:
        candidates = torch.arange(labels.numel())
    margins = decomposition.margin.index_select(0, candidates)
    median_margin = torch.median(margins)
    local_index = torch.argmin(torch.abs(margins - median_margin))
    return int(candidates[local_index].item())


def analyze_checkpoint(
    checkpoint_path: Path,
    *,
    data_root: Path,
    device: str,
    batch_size: int,
    k_values: Sequence[int],
    random_draws: int,
    variant: FinalTwoScanVariant = "full",
) -> tuple[JsonDict, JsonDict]:
    checkpoint = cast(
        "Mapping[str, object]",
        torch.load(checkpoint_path, map_location="cpu", weights_only=True),
    )
    dataset, seed = _checkpoint_job(checkpoint, expected_variant=variant)
    task = clean_validation_classification_task(
        ensure_ucr_train_only(dataset, data_root, allow_download=False),
        seed,
    )
    model = _model_from_checkpoint(checkpoint, task, device=device, variant=variant)
    train = capture_descriptors(
        model,
        task.train_inputs,
        device=device,
        batch_size=batch_size,
    )
    validation = capture_descriptors(
        model,
        task.validation_inputs,
        device=device,
        batch_size=batch_size,
    )
    weight = model.head.classifier.weight.detach().cpu()
    classifier_bias = model.head.classifier.bias
    if classifier_bias is None:
        message = "final ALPHABET classifier unexpectedly lacks a bias"
        raise RuntimeError(message)
    bias = classifier_bias.detach().cpu()
    reference = train.descriptors.mean(dim=0)
    train_reconstruction = _linear_logits(train.descriptors, weight, bias)
    validation_reconstruction = _linear_logits(validation.descriptors, weight, bias)
    logit_residual = max(
        float(torch.max(torch.abs(train_reconstruction - train.logits)).item()),
        float(torch.max(torch.abs(validation_reconstruction - validation.logits)).item()),
    )
    train_decomposition = decompose_margin(
        train.descriptors,
        train.logits,
        weight,
        bias,
        reference,
        model.modes,
    )
    validation_decomposition = decompose_margin(
        validation.descriptors,
        validation.logits,
        weight,
        bias,
        reference,
        model.modes,
    )
    train_importance = train_decomposition.contributions.abs().mean(dim=0).flatten()
    validation_importance = validation_decomposition.contributions.abs().mean(dim=0).flatten()
    order = torch.argsort(train_importance, descending=True).tolist()
    bottom_order = list(reversed(order))
    full_balanced_accuracy = balanced_accuracy(validation.logits, task.validation_labels)
    rows: list[JsonDict] = []
    for k in k_values:
        if not 1 <= k <= 2 * model.modes:
            continue
        for strategy, selected in (("top", order[:k]), ("bottom", bottom_order[:k])):
            removed = _counterfactual_metrics(
                validation.descriptors,
                reference,
                weight,
                bias,
                task.validation_labels,
                validation.logits,
                model.modes,
                selected,
                retain=False,
            )
            retained = _counterfactual_metrics(
                validation.descriptors,
                reference,
                weight,
                bias,
                task.validation_labels,
                validation.logits,
                model.modes,
                selected,
                retain=True,
            )
            rows.extend(
                (
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "k": k,
                        "strategy": strategy,
                        "intervention": "neutralize",
                        **removed,
                    },
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "k": k,
                        "strategy": strategy,
                        "intervention": "retain",
                        **retained,
                    },
                )
            )
        random_removed = _random_metrics(
            validation.descriptors,
            reference,
            weight,
            bias,
            task.validation_labels,
            validation.logits,
            model.modes,
            k,
            retain=False,
            draws=random_draws,
            seed=seed * 1009 + k,
        )
        random_retained = _random_metrics(
            validation.descriptors,
            reference,
            weight,
            bias,
            task.validation_labels,
            validation.logits,
            model.modes,
            k,
            retain=True,
            draws=random_draws,
            seed=seed * 1013 + k,
        )
        rows.extend(
            (
                {
                    "dataset": dataset,
                    "seed": seed,
                    "k": k,
                    "strategy": "random",
                    "intervention": "neutralize",
                    **random_removed,
                },
                {
                    "dataset": dataset,
                    "seed": seed,
                    "k": k,
                    "strategy": "random",
                    "intervention": "retain",
                    **random_retained,
                },
            )
        )
    rows.extend(
        _bank_intervention_rows(
            validation.descriptors,
            reference,
            weight,
            bias,
            task.validation_labels,
            validation.logits,
            model.modes,
            dataset=dataset,
            seed=seed,
        )
    )
    locations, importance_matrix, validation_matrix, signed_matrix = (
        list(_pole_locations(model)),
        train_importance.reshape(2, model.modes),
        validation_importance.reshape(2, model.modes),
        train_decomposition.contributions.mean(dim=0),
    )
    for location in locations:
        bank_index = _as_int(location["bank_index"])
        mode = _as_int(location["mode"])
        location["train_mean_abs_margin_contribution"] = float(
            importance_matrix[bank_index, mode].item()
        )
        location["validation_mean_abs_margin_contribution"] = float(
            validation_matrix[bank_index, mode].item()
        )
        location["train_mean_signed_margin_contribution"] = float(
            signed_matrix[bank_index, mode].item()
        )
    cell: JsonDict = {
        "dataset": dataset,
        "seed": seed,
        "variant": variant,
        "checkpoint": "<local-path>",
        "evaluation_split": "TRAIN-derived validation only",
        "official_test_accessed": False,
        "train_count": int(task.train_inputs.shape[0]),
        "validation_count": int(task.validation_inputs.shape[0]),
        "class_count": task.class_count,
        "sequence_length": int(task.train_inputs.shape[1]),
        "modes_per_bank": model.modes,
        "descriptor_dimension": int(train.descriptors.shape[1]),
        "full_validation_balanced_accuracy": full_balanced_accuracy,
        "max_logit_reconstruction_residual": logit_residual,
        "max_margin_decomposition_residual": max(
            train_decomposition.max_residual,
            validation_decomposition.max_residual,
        ),
        "train_validation_importance_rank_correlation": rank_correlation(
            train_importance,
            validation_importance,
        ),
        "train_mean_cascaded_attribution_share": (
            _mean_cascaded_attribution_share(train_decomposition.contributions)
        ),
        "validation_mean_cascaded_attribution_share": (
            _mean_cascaded_attribution_share(validation_decomposition.contributions)
        ),
        "pole_order_by_train_importance": order,
        "poles": locations,
        "interventions": rows,
    }
    index = _select_representative(task.validation_labels, validation_decomposition)
    sample_contributions = validation_decomposition.contributions[index]
    correct_prediction = bool(
        validation_decomposition.predicted_class[index] == task.validation_labels[index]
    )
    representative: JsonDict = {
        "dataset": dataset,
        "seed": seed,
        "validation_index": index,
        "true_class": int(task.validation_labels[index].item()),
        "predicted_class": int(validation_decomposition.predicted_class[index].item()),
        "runner_up_class": int(validation_decomposition.runner_up_class[index].item()),
        "correct_prediction": correct_prediction,
        "full_validation_balanced_accuracy": full_balanced_accuracy,
        "base_margin": float(validation_decomposition.base_margin[index].item()),
        "margin": float(validation_decomposition.margin[index].item()),
        "contributions": sample_contributions.tolist(),
        "poles": locations,
        "within_checkpoint_selection": (
            "correct prediction nearest the median correct margin"
            if correct_prediction
            else "fallback prediction nearest the median margin because no prediction was correct"
        ),
        "max_logit_reconstruction_residual": logit_residual,
        "max_margin_decomposition_residual": validation_decomposition.max_residual,
    }
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return cell, representative


def _mean_ci(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    average = mean(values)
    if len(values) == 1:
        return average, 0.0
    sample_sd = stdev(values)
    half_width = float(student_t.ppf(0.975, len(values) - 1)) * sample_sd / math.sqrt(len(values))
    return average, half_width


def _mean_cascaded_attribution_share(contributions: Tensor) -> float:
    """Average the per-example share of absolute attribution in bank two."""
    absolute = contributions.abs()
    total = absolute.sum(dim=(1, 2))
    cascaded = absolute[:, 1, :].sum(dim=1)
    valid = total > 0.0
    if not bool(valid.any()):
        return 0.0
    return float((cascaded[valid] / total[valid]).mean().item())


def _aggregate_interventions(
    cells: Sequence[Mapping[str, object]],
) -> list[JsonDict]:
    grouped: dict[tuple[str, str, int], list[tuple[float, float, float]]] = {}
    grouped_by_dataset: dict[
        tuple[str, str, int],
        dict[str, list[tuple[float, float, float]]],
    ] = {}
    for cell in cells:
        full = _as_float(cell["full_validation_balanced_accuracy"])
        dataset = str(cell["dataset"])
        interventions = _as_mapping_sequence(cell["interventions"])
        for row in interventions:
            key = (str(row["intervention"]), str(row["strategy"]), _as_int(row["k"]))
            observation = (
                full - _as_float(row["balanced_accuracy"]),
                _as_float(row["prediction_agreement"]),
                _as_float(row["mean_margin_drop"]),
            )
            grouped.setdefault(key, []).append(observation)
            grouped_by_dataset.setdefault(key, {}).setdefault(dataset, []).append(observation)
    rows: list[JsonDict] = []
    for (intervention, strategy, k), observations in sorted(grouped.items()):
        bacc, bacc_ci = _mean_ci([value[0] for value in observations])
        agreement, agreement_ci = _mean_ci([value[1] for value in observations])
        margin, margin_ci = _mean_ci([value[2] for value in observations])
        dataset_observations = [
            (
                mean(value[0] for value in dataset_rows),
                mean(value[1] for value in dataset_rows),
                mean(value[2] for value in dataset_rows),
            )
            for dataset_rows in grouped_by_dataset[(intervention, strategy, k)].values()
        ]
        _, dataset_bacc_ci = _mean_ci([value[0] for value in dataset_observations])
        _, dataset_agreement_ci = _mean_ci([value[1] for value in dataset_observations])
        _, dataset_margin_ci = _mean_ci([value[2] for value in dataset_observations])
        rows.append(
            {
                "intervention": intervention,
                "strategy": strategy,
                "k": k,
                "cells": len(observations),
                "datasets": len(dataset_observations),
                "mean_balanced_accuracy_drop": bacc,
                "balanced_accuracy_drop_ci95_half_width": bacc_ci,
                "dataset_balanced_accuracy_drop_ci95_half_width": dataset_bacc_ci,
                "mean_prediction_agreement": agreement,
                "prediction_agreement_ci95_half_width": agreement_ci,
                "dataset_prediction_agreement_ci95_half_width": dataset_agreement_ci,
                "mean_margin_drop": margin,
                "margin_drop_ci95_half_width": margin_ci,
                "dataset_margin_drop_ci95_half_width": dataset_margin_ci,
            }
        )
    return rows


def _paired_strategy_contrasts(
    cells: Sequence[Mapping[str, object]],
    *,
    k: int,
    reference_strategy: str = "top",
    comparators: Sequence[str] = ("random", "bottom"),
) -> list[JsonDict]:
    """Compare a fixed pole selection with matched controls within each dataset."""
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for cell in cells:
        dataset = str(cell["dataset"])
        full = _as_float(cell["full_validation_balanced_accuracy"])
        for row in _as_mapping_sequence(cell["interventions"]):
            if _as_int(row["k"]) != k:
                continue
            intervention = str(row["intervention"])
            strategy = str(row["strategy"])
            if intervention == "neutralize":
                outcome = full - _as_float(row["balanced_accuracy"])
            elif intervention == "retain":
                outcome = _as_float(row["prediction_agreement"])
            else:
                continue
            grouped.setdefault((intervention, strategy, dataset), []).append(outcome)

    contrasts: list[JsonDict] = []
    for intervention in ("neutralize", "retain"):
        reference_datasets = {
            dataset
            for current_intervention, strategy, dataset in grouped
            if current_intervention == intervention and strategy == reference_strategy
        }
        for comparator in comparators:
            comparator_datasets = {
                dataset
                for current_intervention, strategy, dataset in grouped
                if current_intervention == intervention and strategy == comparator
            }
            if reference_datasets != comparator_datasets or not reference_datasets:
                message = (
                    f"unpaired {intervention} {reference_strategy}-versus-"
                    f"{comparator} datasets at k={k}"
                )
                raise RuntimeError(message)
            dataset_differences = {
                dataset: mean(grouped[(intervention, reference_strategy, dataset)])
                - mean(grouped[(intervention, comparator, dataset)])
                for dataset in sorted(reference_datasets)
            }
            differences = list(dataset_differences.values())
            average, half_width = _mean_ci(differences)
            if len(differences) > 1 and stdev(differences) > 0.0:
                standard_error = stdev(differences) / math.sqrt(len(differences))
                statistic = average / standard_error
                p_value = float(2.0 * student_t.sf(abs(statistic), len(differences) - 1))
            else:
                statistic = math.inf if average != 0.0 else 0.0
                p_value = 0.0 if average != 0.0 else 1.0
            contrasts.append(
                {
                    "intervention": intervention,
                    "reference_strategy": reference_strategy,
                    "comparator": comparator,
                    "k": k,
                    "datasets": len(differences),
                    "mean_difference": average,
                    "ci95_half_width": half_width,
                    "ci95_low": average - half_width,
                    "ci95_high": average + half_width,
                    "paired_t_statistic": statistic,
                    "paired_t_p_value": p_value,
                    "dataset_differences": dataset_differences,
                }
            )
    return contrasts


def _bank_attribution_summary(cells: Sequence[Mapping[str, object]]) -> JsonDict:
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for cell in cells:
        grouped.setdefault(str(cell["dataset"]), []).append(cell)

    datasets: list[JsonDict] = []
    checkpoints_with_cascaded_top_eight = 0
    for dataset, dataset_cells in sorted(grouped.items()):
        train_shares = [
            _as_float(cell["train_mean_cascaded_attribution_share"]) for cell in dataset_cells
        ]
        validation_shares = [
            _as_float(cell["validation_mean_cascaded_attribution_share"]) for cell in dataset_cells
        ]
        top_eight_counts: list[int] = []
        for cell in dataset_cells:
            modes = _as_int(cell["modes_per_bank"])
            order = cast("Sequence[object]", cell["pole_order_by_train_importance"])
            count = sum(_as_int(pole_id) >= modes for pole_id in order[:8])
            top_eight_counts.append(count)
            checkpoints_with_cascaded_top_eight += int(count > 0)
        datasets.append(
            {
                "dataset": dataset,
                "checkpoints": len(dataset_cells),
                "train_mean_cascaded_attribution_share": mean(train_shares),
                "validation_mean_cascaded_attribution_share": mean(validation_shares),
                "mean_cascaded_poles_in_top_eight": mean(top_eight_counts),
                "minimum_cascaded_poles_in_top_eight": min(top_eight_counts),
                "maximum_cascaded_poles_in_top_eight": max(top_eight_counts),
            }
        )

    def summarize(key: str) -> JsonDict:
        values = [_as_float(row[key]) for row in datasets]
        average, half_width = _mean_ci(values)
        return {
            "mean": average,
            "ci95_low": average - half_width,
            "ci95_high": average + half_width,
        }

    top_eight = summarize("mean_cascaded_poles_in_top_eight")
    return {
        "datasets": datasets,
        "train_cascaded_attribution_share": summarize("train_mean_cascaded_attribution_share"),
        "validation_cascaded_attribution_share": summarize(
            "validation_mean_cascaded_attribution_share"
        ),
        "cascaded_top_eight_fraction": {
            key: _as_float(value) / 8.0 for key, value in top_eight.items()
        },
        "checkpoints_with_cascaded_pole_in_top_eight": (checkpoints_with_cascaded_top_eight),
        "total_checkpoints": sum(len(dataset_cells) for dataset_cells in grouped.values()),
    }


def _select_dataset_representatives(
    candidates: Sequence[Mapping[str, object]],
) -> list[JsonDict]:
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for candidate in candidates:
        grouped.setdefault(str(candidate["dataset"]), []).append(candidate)
    selected: list[JsonDict] = []
    for dataset, rows in sorted(grouped.items()):
        ordered = sorted(
            rows,
            key=lambda row: (
                _as_float(row["full_validation_balanced_accuracy"]),
                _as_int(row["seed"]),
            ),
        )
        chosen = dict(ordered[len(ordered) // 2])
        chosen["dataset_checkpoint_count"] = len(ordered)
        chosen["checkpoint_selection"] = "median validation balanced accuracy"
        chosen["dataset"] = dataset
        selected.append(chosen)
    return selected


def _waterfall(
    axis: Axes,
    representative: Mapping[str, object],
    *,
    panel_label: str | None,
    compact: bool = False,
    show_ylabel: bool = True,
) -> None:
    contributions = torch.tensor(
        _as_float_matrix(representative["contributions"]),
        dtype=torch.float64,
    )
    modes = contributions.shape[1]
    flattened = contributions.flatten()
    order = torch.argsort(flattened.abs(), descending=True)
    shown_count = 5 if compact else 10
    shown = order[:shown_count].tolist()
    remaining = order[shown_count:].tolist()
    base = _as_float(representative["base_margin"])
    labels = [r"$\beta(\mu)$"]
    values = [base]
    colors = ["#7B8290"]
    for pole_id in shown:
        bank, mode = divmod(pole_id, modes)
        value = float(flattened[pole_id].item())
        labels.append(rf"$\mathcal{{B}}_{{{bank + 1},{mode + 1}}}$")
        values.append(value)
        colors.append("#C23B22" if value >= 0.0 else "#247BA0")
    if remaining:
        labels.append("rest")
        values.append(float(flattened[remaining].sum().item()))
        colors.append("#9AA0A6")
    running = 0.0
    for position, (value, color) in enumerate(zip(values, colors, strict=True)):
        if position > 0:
            axis.plot(
                (position - 0.6, position - 0.4),
                (running, running),
                color="#555555",
                linewidth=0.7,
            )
        bottom = min(running, running + value)
        axis.bar(
            position,
            abs(value),
            bottom=bottom,
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )
        running += value
    margin = _as_float(representative["margin"])
    net_pole_evidence = float(flattened.sum().item())
    axis.axhline(margin, color="#222222", linestyle="--", linewidth=1)
    axis.axhline(0.0, color="#555555", linestyle=":", linewidth=0.7)
    axis.set_xticks(
        range(len(labels)),
        labels,
        rotation=52 if compact else 45,
        ha="right",
        fontsize=7.3 if compact else 8,
    )
    if show_ylabel:
        axis.set_ylabel(
            "Predicted vs runner-up margin",
            fontsize=7.3 if compact else None,
        )
    dataset = str(representative["dataset"])
    if panel_label is not None:
        axis.set_title(
            f"({panel_label}) {dataset}: exact margin ledger",
            fontsize=7.3 if compact else None,
            pad=22 if compact else None,
        )
    annotation = (
        (
            rf"$\beta(\mu)={base:+.2f}$; $\sum c={net_pole_evidence:+.2f}$"
            "\n"
            rf"$\mathrm{{margin}}={margin:+.2f}$"
        )
        if compact
        else (
            rf"$\beta(\mu)={base:+.2f}$"
            "\n"
            rf"$\sum c={net_pole_evidence:+.2f}$"
            "\n"
            rf"$\mathrm{{margin}}={margin:+.2f}$"
        )
    )
    axis.text(
        0.5 if compact else 0.98,
        1.01 if compact else 0.04,
        annotation,
        transform=axis.transAxes,
        ha="center" if compact else "right",
        va="bottom",
        fontsize=7.3 if compact else 8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 1.0, "pad": 1.0},
        clip_on=False,
    )
    axis.margins(x=0.02)


def _pole_map(
    axis: Axes,
    representative: Mapping[str, object],
) -> None:
    locations = _as_mapping_sequence(representative["poles"])
    importance = torch.tensor(
        [_as_float(row["train_mean_abs_margin_contribution"]) for row in locations],
        dtype=torch.float64,
    )
    scale = 40.0 + 260.0 * importance / importance.max().clamp_min(1.0e-12)
    for bank_index, (bank, marker, color) in enumerate(
        (("writer", "o", "#16838E"), ("reader", "^", "#7754A5"))
    ):
        selected = [index for index, row in enumerate(locations) if row["bank"] == bank]
        axis.scatter(
            [_as_float(locations[index]["cycles_per_token"]) for index in selected],
            [_as_float(locations[index]["memory_scale"]) for index in selected],
            s=scale[selected].numpy(),
            color=color,
            marker=marker,
            edgecolor="#222222",
            linewidth=0.5,
            alpha=0.72,
            label=(
                f"{DISPLAY_BANK_NAMES[bank_index]} "
                rf"$\mathcal{{B}}_{bank_index + 1}$ ({len(selected)})"
            ),
        )
    axis.set_yscale("log")
    axis.set_xlabel("Oscillation (cycles/token)")
    axis.set_ylabel(r"Memory length $1/\alpha$")
    axis.set_title(f"(a) {representative['dataset']}: TRAIN pole importance")
    axis.legend(frameon=False, fontsize=8)


def _aggregate_lookup(  # pyright: ignore[reportUnusedFunction]
    aggregate: Sequence[Mapping[str, object]],
    intervention: str,
    strategy: str,
) -> list[Mapping[str, object]]:
    return sorted(
        (
            row
            for row in aggregate
            if row["intervention"] == intervention and row["strategy"] == strategy
        ),
        key=lambda row: _as_int(row["k"]),
    )


def _paired_curve_panel(
    axis: Axes,
    contrasts: Sequence[Mapping[str, object]],
    *,
    intervention: str,
    title: str,
    ylabel: str,
) -> None:
    styles = {
        "random": ("#6F4E9C", "o", "top \u2212 random"),
        "bottom": ("#2A9D8F", "s", "top \u2212 bottom"),
    }
    for comparator, (color, marker, label) in styles.items():
        rows = sorted(
            (
                row
                for row in contrasts
                if row["intervention"] == intervention and row["comparator"] == comparator
            ),
            key=lambda row: _as_int(row["k"]),
        )
        if not rows:
            continue
        x_values = [_as_int(row["k"]) for row in rows]
        y_values = [_as_float(row["mean_difference"]) for row in rows]
        errors = [_as_float(row["ci95_half_width"]) for row in rows]
        axis.errorbar(
            x_values,
            y_values,
            yerr=errors,
            color=color,
            marker=marker,
            linewidth=1.5,
            capsize=2,
            label=label,
        )
    axis.set_xscale("log", base=2)
    axis.set_xticks(DEFAULT_K_VALUES, [str(value) for value in DEFAULT_K_VALUES])
    axis.set_xlabel("Number of poles")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.legend(frameon=False, fontsize=8)
    axis.axhline(0.0, color="#222222", linewidth=0.7, linestyle=":")


def write_figure(
    path: Path,
    representatives: Sequence[Mapping[str, object]],
    paired_contrasts: Sequence[Mapping[str, object]],
) -> None:
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    first, second = _main_representatives(representatives)
    figure = plt.figure(figsize=(12.2, 7.2))
    grid = figure.add_gridspec(2, 6)
    pole_axis = figure.add_subplot(grid[0, 0:2])
    first_axis = figure.add_subplot(grid[0, 2:4])
    second_axis = figure.add_subplot(grid[0, 4:6])
    removal_axis = figure.add_subplot(grid[1, 0:3])
    retention_axis = figure.add_subplot(grid[1, 3:6])
    _pole_map(pole_axis, first)
    _waterfall(first_axis, first, panel_label="b")
    _waterfall(second_axis, second, panel_label="c", show_ylabel=False)
    _paired_curve_panel(
        removal_axis,
        paired_contrasts,
        intervention="neutralize",
        title="(d) Removal: paired advantage",
        ylabel="BAcc. loss difference",
    )
    _paired_curve_panel(
        retention_axis,
        paired_contrasts,
        intervention="retain",
        title="(e) Retention: paired advantage",
        ylabel="Prediction-agreement difference",
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _main_representatives(
    representatives: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    by_dataset = {str(row["dataset"]): row for row in representatives}
    chosen = [
        by_dataset[dataset] for dataset in MAIN_REPRESENTATIVE_DATASETS if dataset in by_dataset
    ]
    if len(chosen) < 2:
        chosen = sorted(representatives, key=lambda row: str(row["dataset"]))[:2]
    if len(chosen) < 2:
        message = "at least two dataset representatives are required"
        raise ValueError(message)
    return chosen[0], chosen[1]


def _compact_combined_contrasts_panel(
    axis: Axes,
    contrasts: Sequence[Mapping[str, object]],
    *,
    panel_label: str | None,
    k: int = 8,
) -> None:
    specifications = (
        ("neutralize", "random", "#6F4E9C"),
        ("neutralize", "bottom", "#2A9D8F"),
        ("retain", "random", "#6F4E9C"),
        ("retain", "bottom", "#2A9D8F"),
    )
    rows = [
        next(
            row
            for row in contrasts
            if row["intervention"] == intervention and row["comparator"] == comparator
            if _as_int(row["k"]) == k
        )
        for intervention, comparator, _color in specifications
    ]
    values = [_as_float(row["mean_difference"]) for row in rows]
    errors = [_as_float(row["ci95_half_width"]) for row in rows]
    positions = torch.arange(len(specifications), dtype=torch.float64).numpy()
    for position, row, value, error, specification in zip(
        positions,
        rows,
        values,
        errors,
        specifications,
        strict=True,
    ):
        color = specification[2]
        dataset_values = [
            _as_float(item) for item in _as_mapping(row["dataset_differences"]).values()
        ]
        offsets = torch.linspace(-0.06, 0.06, len(dataset_values)).numpy()
        axis.scatter(
            position + offsets,
            dataset_values,
            s=8,
            color=color,
            edgecolor="none",
            alpha=0.38,
            zorder=2,
        )
        axis.errorbar(
            position,
            value,
            yerr=error,
            color=color,
            marker="D",
            markersize=4.2,
            capsize=3,
            linewidth=1.2,
            zorder=3,
        )
        axis.text(
            position,
            value + error + 0.014,
            f"{100.0 * value:+.0f} pp",
            ha="left" if position == 0 else "center",
            va="bottom",
            fontsize=7.3,
            color=color,
        )
    axis.axhline(0.0, color="#333333", linestyle=":", linewidth=0.7)
    axis.axvline(1.5, color="#B9B9B9", linewidth=0.7)
    upper = max(value + error for value, error in zip(values, errors, strict=True))
    lower = min(
        (
            0.0,
            *(
                _as_float(item)
                for row in rows
                for item in _as_mapping(row["dataset_differences"]).values()
            ),
        ),
    )
    axis.set_ylim(lower - 0.025, max(0.52, upper + 0.055))
    axis.set_yticks((0.0, 0.2, 0.4))
    axis.set_xticks(positions, ("random", "bottom", "random", "bottom"))
    axis.tick_params(axis="both", labelsize=7.3)
    axis.tick_params(axis="x", pad=2)
    axis.set_ylabel("validation difference", fontsize=7.3)
    if panel_label is not None:
        axis.set_title(f"({panel_label}) Frozen ranking predicts reliance", fontsize=7.0)
    axis.text(
        0.5,
        -0.21,
        "removal",
        transform=axis.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=7.3,
        fontweight="bold",
    )
    axis.text(
        2.5,
        -0.21,
        "retention",
        transform=axis.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=7.3,
        fontweight="bold",
    )
    axis.spines[["top", "right"]].set_visible(False)


def write_compact_figure(
    path: Path,
    representatives: Sequence[Mapping[str, object]],
    paired_contrasts: Sequence[Mapping[str, object]],
) -> None:
    """Write the single-column explanatory-and-empirical main-paper figure."""
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    first, _second = _main_representatives(representatives)
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(3.45, 2.35),
        gridspec_kw={"width_ratios": (1.0, 1.55)},
    )
    axes[0].tick_params(axis="both", labelsize=7.3)
    _waterfall(axes[0], first, panel_label=None, compact=True)
    _compact_combined_contrasts_panel(
        axes[1],
        paired_contrasts,
        panel_label=None,
    )
    figure.subplots_adjust(left=0.03, right=0.97, top=0.98, bottom=0.30, wspace=0.31)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.02)
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)


def _write_summary(
    path: Path,
    payload: Mapping[str, object],
) -> None:
    aggregate = _as_mapping_sequence(payload["aggregate"])
    paired_contrasts = _as_mapping_sequence(payload["paired_contrasts"])
    bank_contrasts = _as_mapping_sequence(payload["bank_contrasts"])
    bank_attribution = _as_mapping(payload["bank_attribution"])
    cells = _as_mapping_sequence(payload["cells"])
    variant = str(payload["variant"])

    def row(intervention: str, strategy: str, k: int) -> Mapping[str, object] | None:
        return next(
            (
                value
                for value in aggregate
                if value["intervention"] == intervention
                and value["strategy"] == strategy
                and _as_int(value["k"]) == k
            ),
            None,
        )

    def cell_row(
        cell: Mapping[str, object],
        intervention: str,
        strategy: str,
        k: int,
    ) -> Mapping[str, object]:
        interventions = _as_mapping_sequence(cell["interventions"])
        match = next(
            (
                value
                for value in interventions
                if value["intervention"] == intervention
                and value["strategy"] == strategy
                and _as_int(value["k"]) == k
            ),
            None,
        )
        if match is None:
            message = f"missing {intervention}/{strategy}/k={k} intervention"
            raise RuntimeError(message)
        return match

    top_eight = row("neutralize", "top", 8)
    random_eight = row("neutralize", "random", 8)
    bottom_eight = row("neutralize", "bottom", 8)
    retain_eight = row("retain", "top", 8)
    retain_random_eight = row("retain", "random", 8)
    deletion_wins = sum(
        _as_float(cell_row(cell, "neutralize", "top", 8)["balanced_accuracy"])
        < _as_float(cell_row(cell, "neutralize", "random", 8)["balanced_accuracy"])
        for cell in cells
    )
    retention_wins = sum(
        _as_float(cell_row(cell, "retain", "top", 8)["prediction_agreement"])
        > _as_float(cell_row(cell, "retain", "random", 8)["prediction_agreement"])
        for cell in cells
    )
    lines = [
        f"# Exact pole ledger: {variant}",
        "",
        "TRAIN-derived validation only; official UCR TEST was never loaded.",
        "",
        "## Fidelity",
        "",
        (
            f"- {len(cells)} `{variant}` checkpoints analyzed over "
            f"{len({str(value['dataset']) for value in cells})} datasets."
        ),
        (
            "- Maximum affine-logit reconstruction residual: "
            f"`{_as_float(payload['max_logit_reconstruction_residual']):.3e}`."
        ),
        (
            "- Maximum predicted-margin decomposition residual: "
            f"`{_as_float(payload['max_margin_decomposition_residual']):.3e}`."
        ),
        (
            "- Mean TRAIN-to-validation pole-importance rank correlation: "
            f"`{_as_float(payload['mean_importance_rank_correlation']):.3f}`."
        ),
        "",
        "## Held-out interventions",
        "",
    ]
    if top_eight is not None and random_eight is not None and bottom_eight is not None:
        top_drop = _as_float(top_eight["mean_balanced_accuracy_drop"])
        random_drop = _as_float(random_eight["mean_balanced_accuracy_drop"])
        bottom_drop = _as_float(bottom_eight["mean_balanced_accuracy_drop"])
        lines.append(
            " ".join(
                (
                    "- Neutralizing eight of 32 TRAIN-ranked poles changes validation BAcc. by",
                    f"`{top_drop:+.3f}` on average versus `{random_drop:+.3f}` for random",
                    f"and `{bottom_drop:+.3f}` for bottom-ranked poles;",
                    f"top exceeds random in `{deletion_wins}/{len(cells)}` cells.",
                )
            )
        )
    if retain_eight is not None and retain_random_eight is not None:
        retained_agreement = 100.0 * _as_float(retain_eight["mean_prediction_agreement"])
        random_agreement = 100.0 * _as_float(retain_random_eight["mean_prediction_agreement"])
        lines.append(
            " ".join(
                (
                    "- Retaining only eight of 32 TRAIN-ranked poles preserves",
                    f"`{retained_agreement:.1f}%` of full-model validation predictions versus",
                    f"`{random_agreement:.1f}%` for random poles;",
                    f"top exceeds random in `{retention_wins}/{len(cells)}` cells.",
                )
            )
        )
    lines.extend(("", "## Dataset-paired contrasts", ""))
    for intervention, label in (
        ("neutralize", "BAcc. loss"),
        ("retain", "prediction agreement"),
    ):
        for comparator in ("random", "bottom"):
            contrast = next(
                row
                for row in paired_contrasts
                if row["intervention"] == intervention
                and row["comparator"] == comparator
                and _as_int(row["k"]) == 8
            )
            lines.append(
                " ".join(
                    (
                        f"- Top-minus-{comparator} {label}:",
                        f"`{_as_float(contrast['mean_difference']):+.3f}` with paired 95% CI",
                        f"`[{_as_float(contrast['ci95_low']):+.3f},",
                        f"{_as_float(contrast['ci95_high']):+.3f}]`",
                        f"(`p={_as_float(contrast['paired_t_p_value']):.4f}`).",
                    )
                )
            )
    train_share = _as_mapping(bank_attribution["train_cascaded_attribution_share"])
    validation_share = _as_mapping(bank_attribution["validation_cascaded_attribution_share"])
    top_eight_share = _as_mapping(bank_attribution["cascaded_top_eight_fraction"])
    cascaded_top_eight_checkpoints = _as_int(
        bank_attribution["checkpoints_with_cascaded_pole_in_top_eight"]
    )
    total_checkpoints = _as_int(bank_attribution["total_checkpoints"])
    lines.extend(
        (
            "",
            "## Bank attribution and interventions",
            "",
            (
                "- Cascaded-bank per-example absolute-attribution share: "
                f"TRAIN `{_as_float(train_share['mean']):.3f}` "
                f"(`{_as_float(train_share['ci95_low']):.3f}`, "
                f"`{_as_float(train_share['ci95_high']):.3f}`); "
                f"validation `{_as_float(validation_share['mean']):.3f}` "
                f"(`{_as_float(validation_share['ci95_low']):.3f}`, "
                f"`{_as_float(validation_share['ci95_high']):.3f}`)."
            ),
            (
                "- Cascaded-bank share of TRAIN-ranked top-eight positions: "
                f"`{_as_float(top_eight_share['mean']):.3f}` "
                f"(`{_as_float(top_eight_share['ci95_low']):.3f}`, "
                f"`{_as_float(top_eight_share['ci95_high']):.3f}`); "
                f"present in `{cascaded_top_eight_checkpoints}/"
                f"{total_checkpoints}` checkpoints."
            ),
        )
    )
    for contrast in bank_contrasts:
        intervention = str(contrast["intervention"])
        reference = str(contrast["reference_strategy"])
        comparator = str(contrast["comparator"])
        label = "BAcc. loss" if intervention == "neutralize" else "prediction agreement"
        lines.append(
            " ".join(
                (
                    f"- {reference}-minus-{comparator} {label}:",
                    f"`{_as_float(contrast['mean_difference']):+.3f}` with paired 95% CI",
                    f"`[{_as_float(contrast['ci95_low']):+.3f},",
                    f"{_as_float(contrast['ci95_high']):+.3f}]`",
                    f"(`p={_as_float(contrast['paired_t_p_value']):.4f}`).",
                )
            )
        )
    lines.extend(
        (
            "",
            "## Claim boundary",
            "",
            (
                "The affine ledger is exact for the final descriptor. Cascaded-bank pole "
                "neutralization removes a terminal readout path. Direct-bank descriptor "
                "neutralization measures direct-head reliance only because its modes also "
                "influence the cascaded bank through synthesis. Mean replacement can leave the "
                "joint descriptor distribution, so removal and retention are interface stress "
                "tests rather than generative causal interventions. Pole frequencies are "
                "feature-space scales, not direct estimates of a raw signal's physical frequency."
            ),
            "",
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_audit(
    *,
    checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    device: str = "cuda",
    batch_size: int = 128,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    random_draws: int = DEFAULT_RANDOM_DRAWS,
    variant: FinalTwoScanVariant = "full",
    datasets: Sequence[str] = DEFAULT_AUDIT_DATASETS,
) -> JsonDict:
    runtime_device = device if device != "cuda" or torch.cuda.is_available() else "cpu"
    checkpoints = sorted(checkpoint_root.glob(f"*__{variant}__seed*.pt"))
    selected_datasets = set(datasets)
    checkpoints = [
        checkpoint
        for checkpoint in checkpoints
        if checkpoint.name.split("__", maxsplit=1)[0] in selected_datasets
    ]
    if not checkpoints:
        message = (
            f"no {variant} ALPHABET checkpoints for {sorted(selected_datasets)} "
            f"found under {checkpoint_root}"
        )
        raise FileNotFoundError(message)
    cells: list[JsonDict] = []
    candidates: list[JsonDict] = []
    for checkpoint_path in checkpoints:
        cell, candidate = analyze_checkpoint(
            checkpoint_path,
            data_root=data_root,
            device=runtime_device,
            batch_size=batch_size,
            k_values=k_values,
            random_draws=random_draws,
            variant=variant,
        )
        cells.append(cell)
        candidates.append(candidate)
    representatives = _select_dataset_representatives(candidates)
    aggregate = _aggregate_interventions(cells)
    paired_contrasts = [
        contrast for k in k_values for contrast in _paired_strategy_contrasts(cells, k=k)
    ]
    bank_contrasts = [
        contrast
        for reference_strategy in ("direct_bank", "cascaded_bank")
        for contrast in _paired_strategy_contrasts(
            cells,
            k=16,
            reference_strategy=reference_strategy,
            comparators=("random",),
        )
    ]
    bank_contrasts.extend(
        _paired_strategy_contrasts(
            cells,
            k=16,
            reference_strategy="cascaded_bank",
            comparators=("direct_bank",),
        )
    )
    bank_attribution = _bank_attribution_summary(cells)
    payload: JsonDict = {
        "schema": "pac.learned_pole_ledger.v2",
        "status": "complete",
        "evaluation_split": "TRAIN-derived validation only",
        "official_test_accessed": False,
        "runtime_device": runtime_device,
        "variant": variant,
        "checkpoint_root": "<local-path>",
        "checkpoints": len(checkpoints),
        "datasets": sorted({str(cell["dataset"]) for cell in cells}),
        "seeds": sorted({_as_int(cell["seed"]) for cell in cells}),
        "random_draws_per_cell": random_draws,
        "uncertainty_unit": "dataset means after averaging five seeds within dataset",
        "confidence_interval": (
            f"two-sided 95% Student-t interval across "
            f"{len({str(cell['dataset']) for cell in cells})} dataset means"
        ),
        "paired_confidence_interval": (
            "two-sided 95% Student-t interval of within-dataset top-minus-control differences"
        ),
        "representative_selection": (
            "within each dataset, choose the checkpoint at median validation balanced accuracy; "
            "within it, choose the correct validation prediction nearest the median correct margin"
        ),
        "max_logit_reconstruction_residual": max(
            _as_float(cell["max_logit_reconstruction_residual"]) for cell in cells
        ),
        "max_margin_decomposition_residual": max(
            _as_float(cell["max_margin_decomposition_residual"]) for cell in cells
        ),
        "mean_importance_rank_correlation": mean(
            _as_float(cell["train_validation_importance_rank_correlation"]) for cell in cells
        ),
        "aggregate": aggregate,
        "paired_contrasts": paired_contrasts,
        "bank_contrasts": bank_contrasts,
        "bank_attribution": bank_attribution,
        "representatives": representatives,
        "cells": cells,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_summary(output_root / "SUMMARY.md", payload)
    write_figure(
        output_root / "figures" / "exact_pole_counterfactual_audit.png",
        representatives,
        paired_contrasts,
    )
    write_compact_figure(
        output_root / "figures" / "exact_pole_attribution_compact.png",
        representatives,
        paired_contrasts,
    )
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--random-draws", type=int, default=DEFAULT_RANDOM_DRAWS)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_AUDIT_DATASETS,
        help="dataset names included in the audit",
    )
    parser.add_argument(
        "--variant",
        choices=("full", "fixed_random_poles"),
        default="full",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    arguments = _parser().parse_args(list(argv) if argv is not None else None)
    payload = run_audit(
        checkpoint_root=arguments.checkpoint_root,
        data_root=arguments.data_root,
        output_root=arguments.output_root,
        device=arguments.device,
        batch_size=arguments.batch_size,
        random_draws=arguments.random_draws,
        variant=cast("FinalTwoScanVariant", arguments.variant),
        datasets=arguments.datasets,
    )
    print(json.dumps({key: payload[key] for key in ("status", "checkpoints", "datasets")}))  # noqa: T201


if __name__ == "__main__":
    main()
