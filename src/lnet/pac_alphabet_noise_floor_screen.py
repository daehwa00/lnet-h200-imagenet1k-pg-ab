"""Validation-only screen for noise-aware ALPHABET modal descriptors.

The screen preserves the final writer-reader backbone and affine head.  It
changes only how the same 7M per-stream modal statistics are represented.
Official TEST samples are deliberately not accessed by this module.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Final, Literal, cast

import torch
from torch import Tensor

from .alphabet import Alphabet
from .pac_campaign_utils import write_once
from .pac_classification_diagnostics import corruption_suite
from .pac_data_split import stratified_partition_indices
from .pac_device import resolve_device
from .pac_metrics import count_parameters
from .pac_real_data import ensure_ucr_train_only
from .pac_training import classification_metric_bundle, train_classifier
from .pac_types import (
    PACClassificationTask,
    PACDevice,
    PACExperimentConfig,
    UCRDataset,
)

DescriptorName = Literal[
    "radial_log_r",
    "q_plus_c",
    "c_only",
    "stationary_radial_g0",
    "stationary_radial_g05",
    "stationary_radial_g1",
]

SCREEN_DATASETS: Final[tuple[str, ...]] = (
    "CinCECGTorso",
    "CricketX",
    "ECG5000",
    "GunPoint",
    "StarLightCurves",
)
SCREEN_SEEDS: Final[tuple[int, ...]] = (23, 31, 43)
DESCRIPTORS: Final[tuple[DescriptorName, ...]] = (
    "radial_log_r",
    "q_plus_c",
    "c_only",
    "stationary_radial_g0",
    "stationary_radial_g05",
    "stationary_radial_g1",
)
PROMOTABLE_DESCRIPTORS: Final[tuple[DescriptorName, ...]] = (
    "stationary_radial_g0",
    "stationary_radial_g05",
    "stationary_radial_g1",
)
REFERENCE_DESCRIPTOR: Final[DescriptorName] = "radial_log_r"
DEFAULT_ROOT: Final = Path("results/noise-shrinkage/screen")
DEFAULT_SELECTION_PATH: Final = Path("selection/base.json")
UCR_DATA_ROOT: Final = Path(".omx/data/ucr")
MAXIMUM_MEAN_ID_LOSS: Final[float] = 0.01
INNER_VALIDATION_RATIO: Final[float] = 0.2
OUTER_SCREEN_RATIO: Final[float] = 0.2
SOURCE_FILES: Final = (
    "src/lnet/alphabet.py",
    "src/lnet/pac_tight_frame_models.py",
    "src/lnet/pac_alphabet_noise_floor_screen.py",
    "src/lnet/pac_training.py",
)


@dataclass(frozen=True, slots=True)
class SelectedConfig:
    dataset: str
    model_dim: int
    modes: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    grad_clip_norm: float
    config_key: str


@dataclass(frozen=True, slots=True)
class NoiseFloorJob:
    dataset: str
    descriptor: DescriptorName
    seed: int
    model_dim: int
    modes: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    grad_clip_norm: float
    selection_config_key: str

    @property
    def key(self) -> str:
        return f"alphabet_noise_floor__{self.dataset}__{self.descriptor}__seed{self.seed}"


def stationary_radial_from_qc(
    moments: Tensor,
    modes: int,
    *,
    coherence_exponent: float,
) -> Tensor:
    """Reconstruct scale-aware lag moments from global energy and coherence.

    For each lag, the effective radius is

        Q * |C_l| ** (1 + coherence_exponent)

    while the complex phase is inherited from C_l.  ``coherence_exponent=0``
    is the stationary approximation R_l ~= Q C_l; positive exponents shrink
    low-coherence modes without discarding global scale.
    """
    if moments.shape[-1] != 7 * modes:
        message = f"expected 7M q+C moments, got {moments.shape[-1]} for M={modes}"
        raise ValueError(message)
    if coherence_exponent < 0.0:
        message = "coherence_exponent must be non-negative"
        raise ValueError(message)
    q = moments[..., :modes]
    energy = torch.expm1(q).clamp_min(0.0)
    epsilon = torch.finfo(moments.dtype).tiny
    transformed = [q]
    for offset in (modes, 3 * modes, 5 * modes):
        real = moments[..., offset : offset + modes]
        imag = moments[..., offset + modes : offset + 2 * modes]
        radius = torch.sqrt((real.square() + imag.square()).clamp_min(epsilon))
        effective_radius = energy * radius.pow(1.0 + coherence_exponent)
        scale = torch.log1p(effective_radius) / radius
        transformed.extend((scale * real, scale * imag))
    return torch.cat(transformed, dim=-1)


class _QCAlphabet(Alphabet):
    """Final ALPHABET geometry returning q plus normalized coherence."""

    def __init__(self, config: PACExperimentConfig, output_dim: int) -> None:
        super().__init__(config, output_dim, objective="classification")
        for block in (self.forward_block, self.backward_block):
            block.log_energy = True
            block.normalize_autocorrelation = True
            block.fused_lag124_moments = True
            block.radial_log_lag124_moments = False
            block.parallel_static_radial_log_recurrence_moments_training = False
            block.parallel_static_radial_log_recurrence_moments_inference = False
            block.static_radial_log_lag124_recurrence_moments_inference = False

    def _represent_moments(
        self,
        moments: Tensor,
        block: object | None = None,
        *,
        metadata_free: bool = True,
    ) -> Tensor:
        del block, metadata_free
        return moments


class _COnlyAlphabet(_QCAlphabet):
    """Normalized lag shape with the q coordinates zeroed at identical fan-in."""

    def _represent_moments(
        self,
        moments: Tensor,
        block: object | None = None,
        *,
        metadata_free: bool = True,
    ) -> Tensor:
        del block, metadata_free
        return torch.cat(
            (
                torch.zeros_like(moments[..., : self.modes]),
                moments[..., self.modes :],
            ),
            dim=-1,
        )


class _StationaryRadialAlphabet(_QCAlphabet):
    """Scale-shape reconstruction with fixed coherence shrinkage."""

    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        *,
        coherence_exponent: float,
    ) -> None:
        super().__init__(config, output_dim)
        self.coherence_exponent = coherence_exponent

    def _represent_moments(
        self,
        moments: Tensor,
        block: object | None = None,
        *,
        metadata_free: bool = True,
    ) -> Tensor:
        del block, metadata_free
        return stationary_radial_from_qc(
            moments,
            self.modes,
            coherence_exponent=self.coherence_exponent,
        )


def build_model(
    descriptor: DescriptorName,
    config: PACExperimentConfig,
    class_count: int,
) -> Alphabet:
    if descriptor == "radial_log_r":
        return Alphabet(config, class_count, objective="classification")
    if descriptor == "q_plus_c":
        return _QCAlphabet(config, class_count)
    if descriptor == "c_only":
        return _COnlyAlphabet(config, class_count)
    exponent = {
        "stationary_radial_g0": 0.0,
        "stationary_radial_g05": 0.5,
        "stationary_radial_g1": 1.0,
    }[descriptor]
    return _StationaryRadialAlphabet(
        config,
        class_count,
        coherence_exponent=exponent,
    )


def nested_validation_task(dataset: UCRDataset, seed: int) -> PACClassificationTask:
    """Create inner checkpoint and outer architecture-screen folds from TRAIN."""
    outer_train, screen = stratified_partition_indices(
        dataset.train_labels,
        OUTER_SCREEN_RATIO,
        seed,
    )
    outer_labels = dataset.train_labels.index_select(0, outer_train)
    inner_train, checkpoint = stratified_partition_indices(
        outer_labels,
        INNER_VALIDATION_RATIO,
        seed + 10_009,
    )
    optimization = outer_train.index_select(0, inner_train)
    checkpoint = outer_train.index_select(0, checkpoint)
    raw_optimization = dataset.train_inputs.index_select(0, optimization)
    optimization_labels = dataset.train_labels.index_select(0, optimization)
    raw_checkpoint = dataset.train_inputs.index_select(0, checkpoint)
    checkpoint_labels = dataset.train_labels.index_select(0, checkpoint)
    raw_screen = dataset.train_inputs.index_select(0, screen)
    screen_labels = dataset.train_labels.index_select(0, screen)
    fit_mean = raw_optimization.mean()
    fit_std = raw_optimization.std(unbiased=False).clamp_min(1.0e-6)
    return PACClassificationTask(
        f"{dataset.name}:nested-validation",
        (raw_optimization - fit_mean) / fit_std,
        optimization_labels,
        (raw_checkpoint - fit_mean) / fit_std,
        checkpoint_labels,
        (raw_screen - fit_mean) / fit_std,
        screen_labels,
        dataset.class_count,
    )


def selected_configs(
    path: Path = DEFAULT_SELECTION_PATH,
    *,
    datasets: tuple[str, ...] = SCREEN_DATASETS,
) -> dict[str, SelectedConfig]:
    payload = cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))
    selected = payload.get("selected")
    if (
        payload.get("schema") != "pac.balanced_hpo_alphabet_27task_recovery_stage2_selection.v1"
        or payload.get("configuration_frozen_before_test") is not True
        or payload.get("official_test_accessed_during_selection") is not False
        or not isinstance(selected, dict)
    ):
        message = "noise-floor screen requires the TEST-free final selection"
        raise RuntimeError(message)
    output: dict[str, SelectedConfig] = {}
    for dataset in datasets:
        row = selected.get(f"ucr:{dataset}:alphabet")
        if not isinstance(row, dict) or row.get("architecture") != "radial-log-r-affine":
            message = f"missing final radial-log selection for {dataset}"
            raise RuntimeError(message)
        recipe = row.get("recipe")
        if not isinstance(recipe, dict):
            message = f"missing optimizer recipe for {dataset}"
            raise TypeError(message)
        output[dataset] = SelectedConfig(
            dataset=dataset,
            model_dim=_required_int(row, "width"),
            modes=_required_int(row, "modes"),
            epochs=_required_int(row, "final_epochs"),
            batch_size=_required_int(recipe, "batch_size"),
            learning_rate=_required_float(recipe, "learning_rate"),
            weight_decay=_required_float(recipe, "weight_decay"),
            grad_clip_norm=_required_float(recipe, "grad_clip_norm"),
            config_key=_required_str(row, "config_key"),
        )
    return output


def jobs(selection_path: Path = DEFAULT_SELECTION_PATH) -> tuple[NoiseFloorJob, ...]:
    selections = selected_configs(selection_path)
    return tuple(
        NoiseFloorJob(
            dataset=dataset,
            descriptor=descriptor,
            seed=seed,
            model_dim=selections[dataset].model_dim,
            modes=selections[dataset].modes,
            epochs=selections[dataset].epochs,
            batch_size=selections[dataset].batch_size,
            learning_rate=selections[dataset].learning_rate,
            weight_decay=selections[dataset].weight_decay,
            grad_clip_norm=selections[dataset].grad_clip_norm,
            selection_config_key=selections[dataset].config_key,
        )
        for dataset in SCREEN_DATASETS
        for descriptor in DESCRIPTORS
        for seed in SCREEN_SEEDS
    )


def enqueue(
    root: Path = DEFAULT_ROOT,
    *,
    shard_count: int = 2,
    selection_path: Path = DEFAULT_SELECTION_PATH,
) -> dict[str, object]:
    if shard_count < 1:
        message = "shard_count must be positive"
        raise ValueError(message)
    active = jobs(selection_path)
    shards: list[list[NoiseFloorJob]] = [[] for _ in range(shard_count)]
    loads = [0.0] * shard_count
    for job in sorted(active, key=_job_weight, reverse=True):
        index = min(range(shard_count), key=loads.__getitem__)
        shards[index].append(job)
        loads[index] += _job_weight(job)
    for index, shard in enumerate(shards):
        shard_root = root / "shards" / f"shard-{index:02d}"
        body = "".join(json.dumps(asdict(job), sort_keys=True) + "\n" for job in shard)
        write_once(shard_root / "manifest.jsonl", body)
    contract: dict[str, object] = {
        "schema": "alphabet.noise_floor_screen.contract.v1",
        "claim_status": "TRAIN-only development diagnostic; official TEST untouched",
        "datasets": list(SCREEN_DATASETS),
        "seeds": list(SCREEN_SEEDS),
        "descriptors": list(DESCRIPTORS),
        "promotable_descriptors": list(PROMOTABLE_DESCRIPTORS),
        "reference_descriptor": REFERENCE_DESCRIPTOR,
        "jobs": len(active),
        "selection_path": str(selection_path),
        "selection_sha256": _sha256(selection_path),
        "selection_uses_official_test": False,
        "evaluation_uses_official_test": False,
        "split": {
            "outer_screen_ratio": OUTER_SCREEN_RATIO,
            "inner_checkpoint_ratio_within_outer_train": INNER_VALIDATION_RATIO,
            "normalization_fit": "inner optimization fold only",
            "corruption_evaluation": "outer official-TRAIN screen fold",
        },
        "candidate_rule": {
            "eligibility": (
                f"mean ID balanced-accuracy delta >= -{MAXIMUM_MEAN_ID_LOSS} "
                "and mean noise delta > 0"
            ),
            "ranking": "largest mean noise balanced-accuracy delta, then largest mean ID delta",
            "controls_not_promotable": ["q_plus_c", "c_only"],
        },
        "descriptor_formula": ("L0=log1p(Q); L_l=log1p(Q*|C_l|^(1+gamma))*C_l/|C_l|"),
        "coherence_exponents": {
            "stationary_radial_g0": 0.0,
            "stationary_radial_g05": 0.5,
            "stationary_radial_g1": 1.0,
        },
        "same_backbone_head_and_parameter_count": True,
        "shards": shard_count,
        "estimated_shard_loads": loads,
        "source_sha256": _source_sha256(),
        "restart_safe": True,
    }
    write_once(root / "contract.json", json.dumps(contract, indent=2, sort_keys=True) + "\n")
    return contract


def run_manifest(
    shard_root: Path,
    *,
    selection_path: Path = DEFAULT_SELECTION_PATH,
    data_root: Path = UCR_DATA_ROOT,
    device: PACDevice = "auto",
) -> dict[str, object]:
    manifest = tuple(
        NoiseFloorJob(**json.loads(line))
        for line in (shard_root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    expected = {job.key: job for job in jobs(selection_path)}
    if any(expected.get(job.key) != job for job in manifest):
        message = "manifest differs from the sealed noise-floor screen"
        raise RuntimeError(message)
    runtime_device = cast("PACDevice", resolve_device(device))
    completed = _local_keys(shard_root, "completed")
    for job in manifest:
        if job.key in completed:
            continue
        try:
            result = run_job(job, data_root=data_root, device=runtime_device)
        except Exception as error:  # noqa: BLE001 - durable worker failure record
            result: dict[str, object] = {
                "schema": "alphabet.noise_floor_screen.failure.v1",
                **asdict(job),
                "job_key": job.key,
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
            _write_json(_result_path(shard_root, job.key, "failed"), result, replace=True)
            continue
        _write_json(_result_path(shard_root, job.key, "completed"), result)
        failed = _result_path(shard_root, job.key, "failed")
        if failed.exists():
            failed.unlink()
    return _shard_status(shard_root, manifest)


def run_job(
    job: NoiseFloorJob,
    *,
    data_root: Path = UCR_DATA_ROOT,
    device: PACDevice = "auto",
) -> dict[str, object]:
    runtime_device = cast("PACDevice", resolve_device(device))
    dataset = ensure_ucr_train_only(
        job.dataset,
        data_root,
        allow_download=True,
    )
    task = nested_validation_task(dataset, job.seed)
    config = PACExperimentConfig(
        sample_count=int(task.train_inputs.shape[0]),
        validation_count=int(task.validation_inputs.shape[0]),
        test_count=int(task.test_inputs.shape[0]),
        sequence_length=int(task.train_inputs.shape[1]),
        raw_input_dim=int(task.train_inputs.shape[-1]),
        output_dim=task.class_count,
        model_dim=job.model_dim,
        modes=job.modes,
        epochs=job.epochs,
        batch_size=job.batch_size,
        learning_rate=job.learning_rate,
        weight_decay=job.weight_decay,
        grad_clip_norm=job.grad_clip_norm,
        seeds=(job.seed,),
        device=runtime_device,
        output_dir=DEFAULT_ROOT,
        optimizer_mode="fused" if runtime_device == "cuda" else "default",
    )
    torch.manual_seed(job.seed)
    if runtime_device == "cuda":
        torch.cuda.manual_seed_all(job.seed)
    model = build_model(job.descriptor, config, task.class_count)
    started = perf_counter()
    outcome = train_classifier(
        model,
        task,
        config,
        runtime_device,
        job.seed,
        evaluate_test=True,
        restore_best_validation=True,
    )
    screen_inputs = task.test_inputs.to(runtime_device)
    screen_labels = task.test_labels.to(runtime_device)
    condition_scores = _condition_scores(
        model,
        screen_inputs,
        screen_labels,
        job.seed,
        batch_size=job.batch_size,
    )
    clean_score = condition_scores["id"]
    noise_score = mean((condition_scores["noise_std_0.1"], condition_scores["noise_std_0.2"]))
    return {
        "schema": "alphabet.noise_floor_screen.result.v1",
        **asdict(job),
        "job_key": job.key,
        "status": "done",
        "params_trainable": count_parameters(model),
        "train_loss": outcome.train_loss,
        "checkpoint_validation_loss": outcome.validation_loss,
        "outer_screen_loss": outcome.test_loss,
        "best_epoch": outcome.best_epoch,
        "condition_balanced_accuracy": condition_scores,
        "id_balanced_accuracy": clean_score,
        "noise_mean_balanced_accuracy": noise_score,
        "mean_noise_drop": clean_score - noise_score,
        "elapsed_seconds": perf_counter() - started,
        "official_test_accessed": False,
        "official_test_file_opened": False,
        "dataset_loader": "ensure_ucr_train_only",
        "claim_status": "train_only_development_diagnostic",
        "environment": _environment_metadata(runtime_device),
        "code_sha256": _sha256(Path(__file__)),
    }


def status(
    root: Path = DEFAULT_ROOT,
    *,
    selection_path: Path = DEFAULT_SELECTION_PATH,
) -> dict[str, object]:
    expected = {job.key for job in jobs(selection_path)}
    completed = _all_keys(root, "completed")
    failed = _all_keys(root, "failed") - completed
    return {
        "schema": "alphabet.noise_floor_screen.status.v1",
        "expected": len(expected),
        "completed": len(expected & completed),
        "failed_retryable": len(expected & failed),
        "remaining": len(expected - completed),
        "done": expected <= completed,
    }


def report(
    root: Path = DEFAULT_ROOT,
    *,
    selection_path: Path = DEFAULT_SELECTION_PATH,
    bootstrap_resamples: int = 20_000,
) -> dict[str, object]:
    campaign_status = status(root, selection_path=selection_path)
    if campaign_status["done"] is not True:
        message = f"cannot report incomplete noise-floor screen: {campaign_status}"
        raise RuntimeError(message)
    expected = {job.key for job in jobs(selection_path)}
    rows = [row for row in _all_rows(root, "completed") if row.get("job_key") in expected]
    indexed = {
        (str(row["descriptor"]), str(row["dataset"]), int(cast("int", row["seed"]))): row
        for row in rows
    }
    aggregates: dict[str, dict[str, object]] = {}
    paired: dict[str, dict[str, object]] = {}
    for descriptor in DESCRIPTORS:
        active = [row for row in rows if row.get("descriptor") == descriptor]
        aggregates[descriptor] = {
            "rows": len(active),
            "params_trainable": sorted(
                {int(cast("int", row["params_trainable"])) for row in active}
            ),
            "mean_id_balanced_accuracy": mean(
                _float_field(row, "id_balanced_accuracy") for row in active
            ),
            "mean_noise_balanced_accuracy": mean(
                _float_field(row, "noise_mean_balanced_accuracy") for row in active
            ),
            "mean_noise_drop": mean(_float_field(row, "mean_noise_drop") for row in active),
            "condition_means": _condition_means(active),
        }
        if descriptor == REFERENCE_DESCRIPTOR:
            continue
        id_values: list[tuple[str, int, float]] = []
        noise_values: list[tuple[str, int, float]] = []
        drop_values: list[tuple[str, int, float]] = []
        for dataset in SCREEN_DATASETS:
            for seed in SCREEN_SEEDS:
                row = indexed[(descriptor, dataset, seed)]
                reference = indexed[(REFERENCE_DESCRIPTOR, dataset, seed)]
                id_values.append(
                    (
                        dataset,
                        seed,
                        _float_field(row, "id_balanced_accuracy")
                        - _float_field(reference, "id_balanced_accuracy"),
                    )
                )
                noise_values.append(
                    (
                        dataset,
                        seed,
                        _float_field(row, "noise_mean_balanced_accuracy")
                        - _float_field(reference, "noise_mean_balanced_accuracy"),
                    )
                )
                drop_values.append(
                    (
                        dataset,
                        seed,
                        _float_field(reference, "mean_noise_drop")
                        - _float_field(row, "mean_noise_drop"),
                    )
                )
        paired[descriptor] = {
            "id_balanced_accuracy_delta": _paired_summary(
                id_values,
                bootstrap_resamples=bootstrap_resamples,
            ),
            "noise_balanced_accuracy_delta": _paired_summary(
                noise_values,
                bootstrap_resamples=bootstrap_resamples,
            ),
            "noise_drop_reduction": _paired_summary(
                drop_values,
                bootstrap_resamples=bootstrap_resamples,
            ),
        }
    eligible = [
        descriptor
        for descriptor in PROMOTABLE_DESCRIPTORS
        if _summary_mean(paired[descriptor], "id_balanced_accuracy_delta") >= -MAXIMUM_MEAN_ID_LOSS
        and _summary_mean(paired[descriptor], "noise_balanced_accuracy_delta") > 0.0
    ]
    candidate = (
        max(
            eligible,
            key=lambda descriptor: (
                _summary_mean(paired[descriptor], "noise_balanced_accuracy_delta"),
                _summary_mean(paired[descriptor], "id_balanced_accuracy_delta"),
            ),
        )
        if eligible
        else None
    )
    selection: dict[str, object] = {
        "schema": "alphabet.noise_floor_screen.selection.v1",
        "candidate": candidate,
        "eligible": eligible,
        "reference": REFERENCE_DESCRIPTOR,
        "maximum_mean_id_loss": MAXIMUM_MEAN_ID_LOSS,
        "official_test_accessed": False,
        "configuration_frozen_before_official_test": candidate is not None,
        "candidate_paired_effects": paired.get(candidate) if candidate is not None else None,
    }
    payload: dict[str, object] = {
        "schema": "alphabet.noise_floor_screen.report.v1",
        "status": campaign_status,
        "claim_status": "TRAIN-only development diagnostic",
        "official_test_accessed": False,
        "reference_descriptor": REFERENCE_DESCRIPTOR,
        "aggregates": aggregates,
        "paired_vs_reference": paired,
        "selection": selection,
        "rows": len(rows),
    }
    _write_json(root / "reports" / "summary.json", payload, replace=True)
    _write_json(root / "reports" / "selection.json", selection, replace=True)
    return payload


@torch.no_grad()
def _condition_scores(
    model: torch.nn.Module,
    inputs: Tensor,
    labels: Tensor,
    seed: int,
    *,
    batch_size: int,
) -> dict[str, float]:
    return {
        shift: classification_metric_bundle(
            model,
            shifted,
            labels,
            batch_size=batch_size,
        ).balanced_accuracy
        for shift, shifted in corruption_suite(inputs, seed)
    }


def _condition_means(rows: list[dict[str, object]]) -> dict[str, float]:
    if not rows:
        return {}
    first = rows[0].get("condition_balanced_accuracy")
    if not isinstance(first, dict):
        message = "result is missing condition scores"
        raise TypeError(message)
    return {
        condition: mean(_condition_score(row, condition) for row in rows)
        for condition in sorted(first)
    }


def _condition_score(row: dict[str, object], condition: str) -> float:
    values = row.get("condition_balanced_accuracy")
    if not isinstance(values, dict):
        message = "condition_balanced_accuracy must be an object"
        raise TypeError(message)
    value = values.get(condition)
    if not isinstance(value, int | float) or isinstance(value, bool):
        message = f"condition score {condition} must be numeric"
        raise TypeError(message)
    return float(value)


def _paired_summary(
    values: list[tuple[str, int, float]],
    *,
    bootstrap_resamples: int,
) -> dict[str, object]:
    deltas = [value for _, _, value in values]
    tolerance = 1.0e-12
    return {
        "mean": mean(deltas),
        "hierarchical_bootstrap_ci95": _hierarchical_bootstrap_ci(
            values,
            resamples=bootstrap_resamples,
        ),
        "wins_ties_losses": {
            "wins": sum(value > tolerance for value in deltas),
            "ties": sum(abs(value) <= tolerance for value in deltas),
            "losses": sum(value < -tolerance for value in deltas),
        },
        "pairs": len(deltas),
    }


def _hierarchical_bootstrap_ci(
    values: list[tuple[str, int, float]],
    *,
    resamples: int,
) -> list[float]:
    if resamples < 1:
        message = "bootstrap_resamples must be positive"
        raise ValueError(message)
    grouped: dict[str, list[float]] = {}
    for dataset, _, value in values:
        grouped.setdefault(dataset, []).append(value)
    datasets = sorted(grouped)
    generator = random.Random(20_260_727)  # noqa: S311 - deterministic bootstrap
    draws: list[float] = []
    for _ in range(resamples):
        sampled: list[float] = []
        for _ in datasets:
            dataset = datasets[generator.randrange(len(datasets))]
            active = grouped[dataset]
            sampled.extend(active[generator.randrange(len(active))] for _ in active)
        draws.append(mean(sampled))
    draws.sort()
    return [
        draws[int(0.025 * (resamples - 1))],
        draws[int(0.975 * (resamples - 1))],
    ]


def _summary_mean(values: dict[str, object], metric: str) -> float:
    summary = values.get(metric)
    if not isinstance(summary, dict):
        message = f"missing paired summary {metric}"
        raise TypeError(message)
    value = summary.get("mean")
    if not isinstance(value, int | float) or isinstance(value, bool):
        message = f"paired mean {metric} must be numeric"
        raise TypeError(message)
    return float(value)


def _job_weight(job: NoiseFloorJob) -> float:
    dataset_weight = {
        "CinCECGTorso": 2.0,
        "CricketX": 1.0,
        "ECG5000": 2.0,
        "GunPoint": 1.0,
        "StarLightCurves": 4.0,
    }[job.dataset]
    return dataset_weight * math.sqrt(job.model_dim / 32.0)


def _shard_status(
    root: Path,
    manifest: tuple[NoiseFloorJob, ...],
) -> dict[str, object]:
    expected = {job.key for job in manifest}
    completed = _local_keys(root, "completed")
    failed = _local_keys(root, "failed") - completed
    return {
        "expected": len(expected),
        "completed": len(expected & completed),
        "failed_retryable": len(expected & failed),
        "remaining": len(expected - completed),
        "done": expected <= completed,
    }


def _result_path(root: Path, key: str, bucket: Literal["completed", "failed"]) -> Path:
    return root / bucket / f"{key}.json"


def _local_rows(
    root: Path,
    bucket: Literal["completed", "failed"],
) -> list[dict[str, object]]:
    return [
        cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))
        for path in sorted((root / bucket).glob("*.json"))
    ]


def _local_keys(root: Path, bucket: Literal["completed", "failed"]) -> set[str]:
    return {str(row["job_key"]) for row in _local_rows(root, bucket)}


def _all_rows(
    root: Path,
    bucket: Literal["completed", "failed"],
) -> list[dict[str, object]]:
    return [
        cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(root.glob(f"shards/*/{bucket}/*.json"))
    ]


def _all_keys(root: Path, bucket: Literal["completed", "failed"]) -> set[str]:
    return {str(row["job_key"]) for row in _all_rows(root, bucket)}




def _write_json(path: Path, payload: dict[str, object], *, replace: bool = False) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if replace:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    else:
        write_once(path, text)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_sha256() -> dict[str, str]:
    project = Path(__file__).resolve().parents[2]
    return {name: _sha256(project / name) for name in SOURCE_FILES}


def _environment_metadata(device: str) -> dict[str, object]:
    return {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": device,
        "gpu_name": torch.cuda.get_device_name() if device == "cuda" else None,
        "precision": "fp32",
    }


def _float_field(values: dict[str, object], name: str) -> float:
    value = values.get(name)
    if not isinstance(value, int | float) or isinstance(value, bool):
        message = f"{name} must be numeric"
        raise TypeError(message)
    return float(value)


def _required_int(values: dict[str, object], name: str) -> int:
    value = values.get(name)
    if not isinstance(value, int) or isinstance(value, bool):
        message = f"{name} must be an integer"
        raise TypeError(message)
    return value


def _required_float(values: dict[str, object], name: str) -> float:
    value = values.get(name)
    if not isinstance(value, int | float) or isinstance(value, bool):
        message = f"{name} must be numeric"
        raise TypeError(message)
    return float(value)


def _required_str(values: dict[str, object], name: str) -> str:
    value = values.get(name)
    if not isinstance(value, str) or not value:
        message = f"{name} must be a non-empty string"
        raise TypeError(message)
    return value


__all__ = [
    "DEFAULT_ROOT",
    "DEFAULT_SELECTION_PATH",
    "DESCRIPTORS",
    "SCREEN_DATASETS",
    "SCREEN_SEEDS",
    "UCR_DATA_ROOT",
    "build_model",
    "enqueue",
    "jobs",
    "nested_validation_task",
    "report",
    "run_manifest",
    "stationary_radial_from_qc",
    "status",
]
