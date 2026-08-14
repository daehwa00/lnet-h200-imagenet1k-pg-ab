"""Resumable P0-1/P1-2 bridge campaign for the canonical BenchmarkAlphabetBackbone.

The protocol is deliberately split-sample and fail-closed:

1. load official UCR TRAIN only;
2. split it into representation-training, validation, and calibration pools;
3. train and validation-select the end-to-end CE model;
4. freeze the complete model and select every bank intervention using only the
   representation-training pool;
5. recover raw per-path energy with ``expm1`` and fit
   ``log1p(class-mean raw energy)`` affine nearest-prototype heads using
   calibration only;
6. materialize official TEST and compare the frozen CE and prototype heads.

The v1 pre-execution root is immutable and excluded. No function in this
module edits paper sources.
"""

# pyright: reportAny=false, reportExplicitAny=false, reportPrivateUsage=false
# pyright: reportArgumentType=false, reportIndexIssue=false, reportAttributeAccessIssue=false
# pyright: reportImplicitStringConcatenation=false, reportUnnecessaryCast=false
# ruff: noqa: PLR0915

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median, stdev
from time import perf_counter
from typing import TYPE_CHECKING, Any, Final, Literal, cast

import torch
from torch import Tensor

from .pac_campaign_utils import seed_everything, source_file_hashes
from .pac_confirmatory_baselines import confirmatory_trial_spec
from .pac_metrics import count_parameters
from .pac_real_data import ensure_ucr_dataset, ensure_ucr_train_only
from .pac_training import classification_metric_bundle, train_classifier
from .pac_types import PACClassificationTask, PACDevice, PACExperimentConfig

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping, Sequence
    from typing import Protocol

    from .alphabet_backbone import AlphabetBackbone

    class _BankBlock(Protocol):
        raw_decay: Tensor
        raw_frequency: Tensor

        def frame_matrix(self) -> Tensor: ...

SCHEMA: Final = "pac_theory_bridge_campaign.v2"
DEFAULT_ROOT: Final = Path(".omx/results/pac-theory-bridge-p0p1-v2-20260725")
SUPERSEDED_ROOT: Final = Path(".omx/results/pac-theory-bridge-p0p1-20260724")
DEFAULT_DATA_ROOT: Final = Path(".omx/data/ucr")
DEFAULT_DATASETS: Final = (
    "ECG200",
    "ECGFiveDays",
    "GunPoint",
    "ItalyPowerDemand",
    "Wafer",
    "FordA",
    "CricketX",
    "StarLightCurves",
)
DATASET_STRATA: Final[dict[str, str]] = {
    "ECG200": "short ECG; small binary TRAIN",
    "ECGFiveDays": "short sensor trace; very small binary TRAIN",
    "GunPoint": "medium-length motion trace; small binary TRAIN",
    "ItalyPowerDemand": "very short power trace; small binary TRAIN",
    "Wafer": "medium-length sensor trace; large imbalanced binary TRAIN",
    "FordA": "long automotive trace; large binary TRAIN",
    "CricketX": "long motion trace; 12 classes",
    "StarLightCurves": "very long astronomical trace; large TEST set",
}
DEFAULT_SEEDS: Final = (7, 11, 19, 23, 31)
MODE_PREFIXES: Final = (2, 4, 8, 16, 32)
TRAIN_MODES: Final = 32
MODEL_DIM: Final = 64
RANDOM_DRAWS: Final = 20
CLASS_BOOTSTRAP_REPLICATES: Final = 999
AGGREGATE_BOOTSTRAP_REPLICATES: Final = 999
CALIBRATION_FRACTIONS: Final = (0.25, 0.5, 0.75, 1.0)
VALIDATION_RATIO: Final = 0.2
CALIBRATION_RATIO: Final = 0.2
EPOCHS: Final = 100
TRIAL: Final = 4
ENERGY_VIEWS: Final = ("writer_energy", "reader_energy", "joint_energy")
PRIMARY_VIEW: Final = "writer_energy"
SECONDARY_VIEW: Final = "joint_energy"
PRIMARY_MODES: Final = 32
PRIMARY_FRACTION_KEY: Final = "1.00"
PRIMARY_COMPARATORS: Final = (
    "initial",
    "pole_randomized",
    "direction_randomized",
    "random_best",
    "random_median",
)
NEAR_COLLISION_NORMALIZED_THRESHOLD: Final = 1.0e-3
SOURCE_FILES: Final = (
    "src/lnet/alphabet_backbone.py",
    "src/lnet/pac_confirmatory_baselines.py",
    "src/lnet/pac_metrics.py",
    "src/lnet/pac_real_data.py",
    "src/lnet/pac_theory_bridge_campaign.py",
    "src/lnet/pac_training.py",
)

BankInterventionKind = Literal["pole_randomized", "direction_randomized", "fully_randomized"]
EnergyView = Literal["writer_energy", "reader_energy", "joint_energy"]


@dataclass(frozen=True, slots=True)
class TheoryBridgeJob:
    dataset: str
    seed: int

    @property
    def key(self) -> str:
        return f"theory_bridge:{self.dataset}:seed{self.seed}"


@dataclass(frozen=True, slots=True)
class ThreeWaySplit:
    representation_train: Tensor
    validation: Tensor
    calibration: Tensor

    def validate(self, sample_count: int) -> None:
        pools = (self.representation_train, self.validation, self.calibration)
        if any(pool.dtype != torch.long or pool.ndim != 1 for pool in pools):
            message = "split indices must be one-dimensional int64 tensors"
            raise TypeError(message)
        joined = torch.cat(pools)
        if joined.numel() != sample_count:
            message = "three-way split does not cover official TRAIN exactly once"
            raise ValueError(message)
        if joined.unique().numel() != sample_count:
            message = "three-way split pools overlap"
            raise ValueError(message)
        if joined.min().item() != 0 or joined.max().item() != sample_count - 1:
            message = "three-way split indices fall outside official TRAIN"
            raise ValueError(message)


@dataclass(frozen=True, slots=True)
class BlockBankState:
    raw_decay: Tensor
    raw_frequency: Tensor
    frame: Tensor


@dataclass(frozen=True, slots=True)
class BankState:
    writer: BlockBankState
    reader: BlockBankState


def jobs(
    datasets: tuple[str, ...] = DEFAULT_DATASETS,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> list[TheoryBridgeJob]:
    return [TheoryBridgeJob(dataset, seed) for dataset in datasets for seed in seeds]


def prepare(
    root: Path = DEFAULT_ROOT,
    *,
    datasets: tuple[str, ...] = DEFAULT_DATASETS,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> dict[str, object]:
    """Freeze a deterministic queue and protocol contract without loading UCR TEST."""
    if root.resolve() == SUPERSEDED_ROOT.resolve():
        message = (
            "the v1 theory-bridge root is superseded and immutable; "
            f"use the v2 root {DEFAULT_ROOT}"
        )
        raise ValueError(message)
    if len(datasets) < 8:
        message = "the theory bridge requires at least eight representative UCR tasks"
        raise ValueError(message)
    if len(seeds) < 5:
        message = "the theory bridge requires at least five fixed seeds"
        raise ValueError(message)
    active = jobs(datasets, seeds)
    if len({job.key for job in active}) != len(active):
        message = "the frozen queue must contain unique dataset/seed keys"
        raise ValueError(message)
    root.mkdir(parents=True, exist_ok=True)
    queue_text = "".join(json.dumps(asdict(job), sort_keys=True) + "\n" for job in active)
    queue_path = root / "queue_manifest.jsonl"
    _write_immutable_text(queue_path, queue_text)
    source_manifest = _source_manifest()
    source_path = root / "source_manifest.json"
    _write_immutable_json(source_path, source_manifest)
    source_snapshot_sha256 = _sha256_path(source_path)
    contract: dict[str, object] = {
        "schema": f"{SCHEMA}.contract",
        "purpose": "P0-1 split-calibrated theory bridge plus P1-2 learned/random bank audit",
        "campaign_root": str(root),
        "supersedes": {
            "root": str(SUPERSEDED_ROOT),
            "schema": "pac_theory_bridge_campaign.v1",
            "disposition": (
                "superseded pre-execution design; immutable and excluded from v2"
            ),
            "rows_reused": 0,
            "reason": (
                "v1 averaged log-energy instead of applying log1p after the "
                "class mean of raw per-path energy and lacked fail-closed provenance"
            ),
            "artifact_sha256_at_v2_freeze": _superseded_artifact_hashes(),
        },
        "execution_target": "local_gpu CUDA GPU",
        "official_test_accessed_at_prepare": False,
        "paper_edits_authorized": False,
        "data_order": [
            "official TRAIN only",
            "three-way disjoint split",
            "end-to-end CE training and validation selection",
            "freeze encoder, both pole banks, and CE head",
            "select random-bank best/median on representation-training only",
            "fit every prototype affine head on calibration only",
            "official TEST materialization and one-shot evaluation",
        ],
        "datasets": list(datasets),
        "dataset_selection_strata": {
            name: DATASET_STRATA.get(name, "user-specified UCR task") for name in datasets
        },
        "seeds": list(seeds),
        "jobs": len(active),
        "queue_manifest_sha256": _sha256_path(queue_path),
        "source_snapshot": str(source_path),
        "source_snapshot_sha256": source_snapshot_sha256,
        "source_sha256": source_manifest["source_sha256"],
        "model": {
            "name": "AlphabetBackbone",
            "model_dim": MODEL_DIM,
            "trained_modes": TRAIN_MODES,
            "mode_evaluation": list(MODE_PREFIXES),
            "mode_protocol": (
                "nested prefix/subbank energy evaluation of one trained M=32 representation; "
                "no retraining/capacity change across M"
            ),
            "epochs": EPOCHS,
            "optimizer_trial": TRIAL,
        },
        "split": {
            "validation_ratio": VALIDATION_RATIO,
            "calibration_ratio": CALIBRATION_RATIO,
            "minimum_per_class_per_pool": 1,
            "preprocessing_fit_pool": "representation_train",
        },
        "prototype": {
            "views": list(ENERGY_VIEWS),
            "primary_view": PRIMARY_VIEW,
            "primary_estimand": (
                "writer-only M=32 empirical class-separation margin"
            ),
            "primary_effective_dimension_at_m32": PRIMARY_MODES,
            "secondary_view": SECONDARY_VIEW,
            "secondary_effective_dimension_at_m32": 2 * PRIMARY_MODES,
            "calibration_fractions": list(CALIBRATION_FRACTIONS),
            "per_path_feature": (
                "log1p(raw pole energy); TEST retains the same per-path transform"
            ),
            "class_prototype_estimator": (
                "log1p(class mean of raw per-path energy), with raw energy "
                "recovered exactly as expm1(per-path log1p energy)"
            ),
            "jensen_order": (
                "log1p(mean_c(raw energy)); never mean_c(log1p(raw energy))"
            ),
            "margin_name": (
                "empirical hat_delta_M: minimum Euclidean distance between "
                "estimated class prototypes"
            ),
            "uncertainty": {
                "primary_and_secondary_full_calibration_m32": (
                    "class-stratified nonparametric bootstrap"
                ),
                "bootstrap_replicates": CLASS_BOOTSTRAP_REPLICATES,
                "resampling_independence": (
                    "examples resampled independently within every class"
                ),
                "other_diagnostics": (
                    "delta method for log1p of a raw-energy sample mean at the "
                    "empirical minimizing class pair"
                ),
            },
            "affine_weight": "2 * class prototype",
            "affine_bias": "-squared prototype norm",
        },
        "interventions": {
            "conditions": [
                "learned",
                "initial",
                "pole_randomized",
                "direction_randomized",
                "random_best",
                "random_median",
            ],
            "fully_randomized_draws": RANDOM_DRAWS,
            "draw_selection_pool": "representation_train",
            "draw_selection_metric": "writer-energy empirical hat_delta_32",
            "interpretation": {
                "learned": "trained bank in the frozen validation-selected model",
                "initial": (
                    "initial bank substituted into the otherwise trained "
                    "frozen model; not an independently trained initialization baseline"
                ),
                "pole_randomized": (
                    "fresh independent random pole locations for writer and reader; "
                    "learned directions and all non-bank parameters retained"
                ),
                "direction_randomized": (
                    "fresh independent Haar-style writer and reader frames; learned "
                    "poles and all non-bank parameters retained"
                ),
                "random_best": (
                    "best of 20 fully randomized banks selected with representation-"
                    "training labels; independent of calibration and TEST but not an "
                    "unselected iid random-bank draw"
                ),
                "random_median": (
                    "median-ranked of the same 20 representation-selected fully "
                    "randomized banks; independent of calibration and TEST but not an "
                    "unselected iid random-bank draw"
                ),
            },
            "draw_independence": (
                "each fully randomized draw has an independently derived seed; "
                "within a draw, writer/reader poles and frames consume non-overlapping "
                "segments of one deterministic pseudorandom stream"
            ),
            "random_pole_sampling_law": {
                "raw_decay": "independent Uniform[-3, 1]",
                "normalized_frequency": "independent Uniform[0, 0.75]",
                "scope": (
                    "counterfactual randomized locations; not claimed to reproduce "
                    "the model initializer or a Bayesian prior"
                ),
            },
        },
        "aggregate_inference": {
            "paired_unit": "dataset/seed",
            "bootstrap": (
                "paired dataset-cluster bootstrap over within-run differences"
            ),
            "bootstrap_replicates": AGGREGATE_BOOTSTRAP_REPLICATES,
            "p1_gate": (
                "lower endpoint of the learned-minus-random-median paired 95% "
                "cluster-bootstrap interval must exceed zero"
            ),
        },
    }
    _write_immutable_json(root / "contract.json", contract)
    return status(root)


def status(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    context = _validated_campaign_context(root)
    expected_keys = context["expected_keys"]
    completed_rows = _validated_result_rows(root, context, kind="completed")
    failed_rows = _validated_result_rows(root, context, kind="failed")
    completed_keys = {str(row["job_key"]) for row in completed_rows}
    failed_keys = {str(row["job_key"]) for row in failed_rows}
    _reject_overlapping_terminal_rows(completed_keys, failed_keys)
    return {
        "schema": f"{SCHEMA}.status",
        "integrity": "validated",
        "expected": len(expected_keys),
        "completed": len(completed_keys),
        "failed": len(failed_keys),
        "remaining": len(expected_keys - completed_keys),
        "done": bool(expected_keys) and completed_keys == expected_keys and not failed_keys,
        "queue_manifest_sha256": context["queue_manifest_sha256"],
        "contract_sha256": context["contract_sha256"],
        "source_snapshot_sha256": context["source_snapshot_sha256"],
    }


def stratified_three_way_indices(
    labels: Tensor,
    seed: int,
    *,
    validation_ratio: float = VALIDATION_RATIO,
    calibration_ratio: float = CALIBRATION_RATIO,
) -> ThreeWaySplit:
    """Partition every class across representation, validation, and calibration."""
    if not 0.0 < validation_ratio < 1.0 or not 0.0 < calibration_ratio < 1.0:
        message = "validation and calibration ratios must be strictly between zero and one"
        raise ValueError(message)
    if validation_ratio + calibration_ratio >= 1.0:
        message = "validation plus calibration ratios must leave representation-training data"
        raise ValueError(message)
    labels_cpu = labels.detach().cpu().to(torch.long)
    generator = torch.Generator().manual_seed(seed)
    representation: list[int] = []
    validation: list[int] = []
    calibration: list[int] = []
    for class_value in torch.unique(labels_cpu, sorted=True).tolist():
        indices = torch.nonzero(labels_cpu == int(class_value), as_tuple=False).flatten()
        if indices.numel() < 3:
            message = (
                f"class {class_value} has {indices.numel()} official-TRAIN examples; "
                "three disjoint pools require at least three"
            )
            raise ValueError(message)
        shuffled = indices[torch.randperm(indices.numel(), generator=generator)]
        validation_count = max(1, round(indices.numel() * validation_ratio))
        calibration_count = max(1, round(indices.numel() * calibration_ratio))
        while validation_count + calibration_count > indices.numel() - 1:
            if calibration_count >= validation_count and calibration_count > 1:
                calibration_count -= 1
            elif validation_count > 1:
                validation_count -= 1
            else:
                message = "unable to retain one example in every three-way split pool"
                raise ValueError(message)
        validation.extend(int(value) for value in shuffled[:validation_count].tolist())
        calibration.extend(
            int(value)
            for value in shuffled[
                validation_count : validation_count + calibration_count
            ].tolist()
        )
        representation.extend(
            int(value)
            for value in shuffled[validation_count + calibration_count :].tolist()
        )
    result = ThreeWaySplit(
        torch.tensor(sorted(representation), dtype=torch.long),
        torch.tensor(sorted(validation), dtype=torch.long),
        torch.tensor(sorted(calibration), dtype=torch.long),
    )
    result.validate(labels_cpu.numel())
    for pool in (
        result.representation_train,
        result.validation,
        result.calibration,
    ):
        observed = torch.unique(labels_cpu.index_select(0, pool), sorted=True)
        if not torch.equal(observed, torch.unique(labels_cpu, sorted=True)):
            message = "each split pool must preserve the complete TRAIN label space"
            raise ValueError(message)
    return result


def calibration_subset_indices(labels: Tensor, fraction: float, seed: int) -> Tensor:
    """Return deterministic, class-stratified, nested calibration prefixes."""
    if not 0.0 < fraction <= 1.0:
        message = "calibration fraction must lie in (0, 1]"
        raise ValueError(message)
    labels_cpu = labels.detach().cpu().to(torch.long)
    generator = torch.Generator().manual_seed(seed)
    selected: list[int] = []
    for class_value in torch.unique(labels_cpu, sorted=True).tolist():
        indices = torch.nonzero(labels_cpu == int(class_value), as_tuple=False).flatten()
        shuffled = indices[torch.randperm(indices.numel(), generator=generator)]
        count = max(1, math.ceil(indices.numel() * fraction))
        selected.extend(int(value) for value in shuffled[:count].tolist())
    return torch.tensor(sorted(selected), dtype=torch.long)


def affine_prototype_parameters(prototypes: Tensor) -> tuple[Tensor, Tensor]:
    """Return W,b where xW^T+b has nearest-squared-distance decisions."""
    active = prototypes.to(torch.float64)
    return 2.0 * active, -active.square().sum(dim=1)


def affine_prototype_logits(features: Tensor, prototypes: Tensor) -> Tensor:
    weight, bias = affine_prototype_parameters(prototypes)
    active = features.to(dtype=weight.dtype, device=weight.device)
    return active @ weight.transpose(0, 1) + bias


def negative_squared_distance_logits(features: Tensor, prototypes: Tensor) -> Tensor:
    active = features.to(dtype=torch.float64, device=prototypes.device)
    return -torch.cdist(active, prototypes.to(torch.float64)).square()


def raw_energy_class_prototypes(features: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
    """Estimate ``log1p(E[raw energy | class])`` from per-path log energies.

    The network exposes ``log1p(raw_energy)`` for each path and pole.  The
    theorem's class energy is an expectation in raw-energy space, so averaging
    the logged paths would be a Jensen-biased, different estimand.
    """
    values = features.detach().cpu().to(torch.float64)
    labels_cpu = labels.detach().cpu().to(torch.long)
    if values.ndim != 2 or labels_cpu.ndim != 1 or values.shape[0] != labels_cpu.numel():
        message = "prototype features must be [samples, poles] with one label per sample"
        raise ValueError(message)
    if not torch.isfinite(values).all():
        message = "prototype features must be finite"
        raise ValueError(message)
    if bool((values < -1.0e-10).any()):
        message = "per-path log1p energy cannot be negative"
        raise ValueError(message)
    raw_energy = torch.expm1(values).clamp_min(0.0)
    classes = torch.unique(labels_cpu, sorted=True)
    if classes.numel() < 2:
        message = "prototype fitting requires at least two classes"
        raise ValueError(message)
    prototypes = torch.stack(
        [
            torch.log1p(raw_energy[labels_cpu == class_value].mean(dim=0))
            for class_value in classes
        ]
    )
    return classes, prototypes


def fit_prototype_head(
    features: Tensor,
    labels: Tensor,
    *,
    bootstrap_seed: int = 0,
    bootstrap_replicates: int = CLASS_BOOTSTRAP_REPLICATES,
) -> dict[str, object]:
    """Fit theorem-aligned prototypes using only supplied calibration rows."""
    if bootstrap_replicates != 0 and bootstrap_replicates < 999:
        message = "class bootstrap must use at least 999 replicates"
        raise ValueError(message)
    values = features.detach().cpu().to(torch.float64)
    labels_cpu = labels.detach().cpu().to(torch.long)
    classes, prototypes = raw_energy_class_prototypes(values, labels_cpu)
    raw_energy = torch.expm1(values).clamp_min(0.0)
    counts = torch.tensor(
        [int((labels_cpu == class_value).sum()) for class_value in classes],
        dtype=torch.long,
    )
    delta_covariances = [
        _prototype_delta_covariance(
            raw_energy[labels_cpu == class_value],
            prototypes[index],
        )
        for index, class_value in enumerate(classes)
    ]
    bootstrap_prototypes = (
        _class_bootstrap_prototypes(
            raw_energy,
            labels_cpu,
            classes,
            seed=bootstrap_seed,
            replicates=bootstrap_replicates,
        )
        if bootstrap_replicates
        else None
    )
    pair_bootstrap_distances: list[Tensor] = []
    pairs: list[dict[str, object]] = []
    for left in range(classes.numel()):
        for right in range(left + 1, classes.numel()):
            difference = prototypes[left] - prototypes[right]
            distance = float(torch.linalg.vector_norm(difference))
            unit = difference / max(distance, 1.0e-12)
            delta_variance = float(
                unit @ (delta_covariances[left] + delta_covariances[right]) @ unit
            )
            delta_standard_error = math.sqrt(max(delta_variance, 0.0))
            if bootstrap_prototypes is not None:
                bootstrap_distance = torch.linalg.vector_norm(
                    bootstrap_prototypes[left] - bootstrap_prototypes[right],
                    dim=1,
                )
                pair_bootstrap_distances.append(bootstrap_distance)
                estimator_standard_error = float(bootstrap_distance.std(unbiased=True))
                interval = _tensor_percentile_interval(bootstrap_distance)
                uncertainty_method = "class_stratified_nonparametric_bootstrap"
            else:
                estimator_standard_error = delta_standard_error
                interval = [
                    max(0.0, distance - 1.96 * delta_standard_error),
                    distance + 1.96 * delta_standard_error,
                ]
                uncertainty_method = "delta_method"
            pairs.append(
                {
                    "left_class": int(classes[left]),
                    "right_class": int(classes[right]),
                    "empirical_pair_margin": distance,
                    "normalized_empirical_pair_margin": (
                        distance / math.sqrt(values.shape[1])
                    ),
                    "estimator_standard_error": estimator_standard_error,
                    "delta_method_standard_error": delta_standard_error,
                    "se_snr": distance / max(estimator_standard_error, 1.0e-12),
                    "uncertainty_method": uncertainty_method,
                    "uncertainty_interval_95": interval,
                }
            )
    minimum_pair_index = min(
        range(len(pairs)),
        key=lambda index: float(pairs[index]["empirical_pair_margin"]),
    )
    minimum_pair = pairs[minimum_pair_index]
    hat_delta_m = float(minimum_pair["empirical_pair_margin"])
    if bootstrap_prototypes is not None:
        all_pair_distances = torch.stack(pair_bootstrap_distances, dim=1)
        bootstrap_hat_delta = all_pair_distances.min(dim=1).values
        hat_delta_standard_error = float(bootstrap_hat_delta.std(unbiased=True))
        hat_delta_interval = _tensor_percentile_interval(bootstrap_hat_delta)
        hat_delta_bias = float(bootstrap_hat_delta.mean()) - hat_delta_m
        uncertainty_method = "class_stratified_nonparametric_bootstrap"
    else:
        hat_delta_standard_error = float(minimum_pair["delta_method_standard_error"])
        hat_delta_interval = [
            max(0.0, hat_delta_m - 1.96 * hat_delta_standard_error),
            hat_delta_m + 1.96 * hat_delta_standard_error,
        ]
        hat_delta_bias = None
        uncertainty_method = "delta_method_at_empirical_minimizing_pair"
    weight, bias = affine_prototype_parameters(prototypes)
    return {
        "classes": classes,
        "prototypes": prototypes,
        "weight": weight,
        "bias": bias,
        "calibration_counts": counts,
        "calibration_size": int(values.shape[0]),
        "feature_dimension": int(values.shape[1]),
        "prototype_estimator": "log1p(class_mean(expm1(per_path_log1p_energy)))",
        "empirical_hat_delta_m": hat_delta_m,
        "normalized_empirical_hat_delta_m": (
            hat_delta_m / math.sqrt(values.shape[1])
        ),
        "hat_delta_standard_error": hat_delta_standard_error,
        "hat_delta_se_snr": hat_delta_m / max(hat_delta_standard_error, 1.0e-12),
        "hat_delta_uncertainty_method": uncertainty_method,
        "hat_delta_uncertainty_interval_95": hat_delta_interval,
        "hat_delta_bootstrap_bias": hat_delta_bias,
        "class_bootstrap_replicates": bootstrap_replicates,
        "class_bootstrap_seed": bootstrap_seed if bootstrap_replicates else None,
        "minimum_class_pair": minimum_pair,
        "class_pair_margins": pairs,
        "singleton_calibration_classes": [
            int(classes[index]) for index in range(classes.numel()) if int(counts[index]) == 1
        ],
    }


def _prototype_delta_covariance(raw_class_energy: Tensor, prototype: Tensor) -> Tensor:
    """Delta-method covariance for log1p of a raw-energy sample mean."""
    sample_count = raw_class_energy.shape[0]
    dimension = raw_class_energy.shape[1]
    if sample_count <= 1:
        return torch.zeros((dimension, dimension), dtype=torch.float64)
    centered = raw_class_energy - raw_class_energy.mean(dim=0)
    raw_covariance = centered.transpose(0, 1) @ centered / (sample_count - 1)
    mean_covariance = raw_covariance / sample_count
    derivative = torch.exp(-prototype)
    return derivative[:, None] * mean_covariance * derivative[None, :]


def _class_bootstrap_prototypes(
    raw_energy: Tensor,
    labels: Tensor,
    classes: Tensor,
    *,
    seed: int,
    replicates: int,
) -> list[Tensor]:
    """Bootstrap class prototypes with independent within-class resampling."""
    output: list[Tensor] = []
    batch_replicates = 128
    for class_index, class_value in enumerate(classes.tolist()):
        active = raw_energy[labels == int(class_value)]
        generator = torch.Generator().manual_seed(
            _derived_seed(seed, f"class-bootstrap-{class_index}-{class_value}")
        )
        chunks: list[Tensor] = []
        for start in range(0, replicates, batch_replicates):
            count = min(batch_replicates, replicates - start)
            indices = torch.randint(
                active.shape[0],
                (count, active.shape[0]),
                generator=generator,
            )
            chunks.append(torch.log1p(active[indices].mean(dim=1)))
        output.append(torch.cat(chunks, dim=0))
    return output


def _tensor_percentile_interval(values: Tensor) -> list[float]:
    quantiles = torch.quantile(
        values.to(torch.float64),
        torch.tensor([0.025, 0.975], dtype=torch.float64),
    )
    return [float(quantiles[0]), float(quantiles[1])]


def evaluate_prototype_head(
    features: Tensor,
    labels: Tensor,
    head: Mapping[str, object],
) -> dict[str, float]:
    prototypes = cast("Tensor", head["prototypes"])
    classes = cast("Tensor", head["classes"])
    logits = affine_prototype_logits(features, prototypes)
    predictions = classes[logits.argmax(dim=1)].to(torch.long)
    labels_cpu = labels.detach().cpu().to(torch.long)
    negative_distance = negative_squared_distance_logits(features, prototypes)
    common_offset = features.detach().cpu().to(torch.float64).square().sum(dim=1, keepdim=True)
    equivalence_error = float(
        (logits.detach().cpu() - negative_distance - common_offset).abs().max()
    )
    return {
        **_prediction_metrics(predictions, labels_cpu),
        "affine_distance_equivalence_max_error": equivalence_error,
    }


def random_orthonormal_frame(rows: int, columns: int, generator: torch.Generator) -> Tensor:
    """Draw a deterministic Haar-style column-orthonormal frame on CPU."""
    if rows < columns:
        message = f"semi-orthogonal frame requires rows >= columns, got {rows} < {columns}"
        raise ValueError(message)
    matrix = torch.randn(rows, columns, generator=generator, dtype=torch.float64)
    frame, triangular = torch.linalg.qr(matrix, mode="reduced")
    signs = torch.sign(torch.diagonal(triangular))
    signs[signs == 0] = 1
    return (frame * signs).to(torch.float32)


def randomized_bank_state(
    base: BankState,
    kind: BankInterventionKind,
    seed: int,
) -> BankState:
    """Randomize poles and/or directions without changing any other model parameter."""
    generator = torch.Generator().manual_seed(seed)

    def randomized(block: BlockBankState) -> BlockBankState:
        randomize_poles = kind in {"pole_randomized", "fully_randomized"}
        randomize_directions = kind in {"direction_randomized", "fully_randomized"}
        if randomize_poles:
            raw_decay = (
                torch.rand(block.raw_decay.shape, generator=generator, dtype=torch.float64) * 4.0
                - 3.0
            ).to(torch.float32)
            normalized_frequency = (
                0.75
                * torch.rand(
                    block.raw_frequency.shape,
                    generator=generator,
                    dtype=torch.float64,
                )
            ).clamp(max=0.999)
            raw_frequency = torch.atanh(normalized_frequency).to(torch.float32)
        else:
            raw_decay = block.raw_decay.clone()
            raw_frequency = block.raw_frequency.clone()
        frame = (
            random_orthonormal_frame(block.frame.shape[0], block.frame.shape[1], generator)
            if randomize_directions
            else block.frame.clone()
        )
        return BlockBankState(raw_decay, raw_frequency, frame)

    return BankState(randomized(base.writer), randomized(base.reader))


def capture_bank_state(model: AlphabetBackbone) -> BankState:
    return BankState(
        _capture_block(model.forward_block),
        _capture_block(model.backward_block),
    )


@contextmanager
def temporary_bank_state(
    model: AlphabetBackbone,
    state: BankState,
) -> Generator[None]:
    """Install a counterfactual bank and restore the learned bank on exit."""
    blocks = (model.forward_block, model.backward_block)
    saved: list[tuple[Tensor, Tensor, Tensor | None]] = []
    for block in blocks:
        frame = block.intervention_frame()
        saved.append(
            (
                block.raw_decay.detach().clone(),
                block.raw_frequency.detach().clone(),
                None if frame is None else frame.detach().clone(),
            )
        )
    try:
        for block, active in zip(blocks, (state.writer, state.reader), strict=True):
            with torch.no_grad():
                block.raw_decay.copy_(
                    active.raw_decay.to(block.raw_decay.device, block.raw_decay.dtype)
                )
                block.raw_frequency.copy_(
                    active.raw_frequency.to(block.raw_frequency.device, block.raw_frequency.dtype)
                )
            block.set_intervention_frame(
                active.frame.to(block.raw_decay.device, block.raw_decay.dtype)
            )
        yield
    finally:
        for block, (raw_decay, raw_frequency, frame) in zip(blocks, saved, strict=True):
            with torch.no_grad():
                block.raw_decay.copy_(raw_decay)
                block.raw_frequency.copy_(raw_frequency)
            block.set_intervention_frame(frame)


@torch.no_grad()
def extract_energy_views(
    model: AlphabetBackbone,
    inputs: Tensor,
    *,
    batch_size: int,
    device: str,
) -> dict[EnergyView, Tensor]:
    """Extract per-path ``log1p(raw energy)`` vectors consumed by the model head."""
    output: dict[EnergyView, list[Tensor]] = {
        "writer_energy": [],
        "reader_energy": [],
        "joint_energy": [],
    }
    was_training = model.training
    model.eval()
    for batch in inputs.split(batch_size):
        active = batch.to(device=device)
        first_local, delta, observation, valid = model._edge_stem(  # noqa: SLF001
            active, None, None, None
        )
        first_stream, writer = model._writer(  # noqa: SLF001
            first_local, delta, observation, valid
        )
        _, reader = model._terminal_reader(  # noqa: SLF001
            first_stream, delta, None, valid
        )
        writer_energy = writer[:, : model.modes]
        reader_energy = reader[:, : model.modes]
        values = {
            "writer_energy": writer_energy,
            "reader_energy": reader_energy,
            "joint_energy": torch.cat((writer_energy, reader_energy), dim=-1),
        }
        for name, value in values.items():
            if not torch.isfinite(value).all() or bool((value < -1.0e-6).any()):
                message = f"{name} is not a valid per-path log1p-energy tensor"
                raise RuntimeError(message)
            output[cast("EnergyView", name)].append(value.detach().cpu().to(torch.float64))
    model.train(was_training)
    return {name: torch.cat(chunks) for name, chunks in output.items()}


def prefix_energy_view(
    features: Mapping[EnergyView, Tensor],
    view: EnergyView,
    modes: int,
    *,
    trained_modes: int = TRAIN_MODES,
) -> Tensor:
    if modes not in MODE_PREFIXES or modes > trained_modes:
        message = f"unsupported nested mode prefix M={modes}"
        raise ValueError(message)
    if view == "writer_energy":
        return features[view][:, :modes]
    if view == "reader_energy":
        return features[view][:, :modes]
    joint = features[view]
    return torch.cat((joint[:, :modes], joint[:, trained_modes : trained_modes + modes]), dim=1)


def run(
    root: Path = DEFAULT_ROOT,
    *,
    device: PACDevice = "cuda",
    data_root: Path = DEFAULT_DATA_ROOT,
    job_key: str | None = None,
    limit: int | None = None,
) -> dict[str, object]:
    """Resume incomplete jobs; completed rows are never overwritten."""
    context = _validated_campaign_context(root)
    completed_rows = _validated_result_rows(root, context, kind="completed")
    failed_rows = _validated_result_rows(root, context, kind="failed")
    _reject_overlapping_terminal_rows(
        {str(row["job_key"]) for row in completed_rows},
        {str(row["job_key"]) for row in failed_rows},
    )
    if device != "cuda":
        message = "the frozen campaign contract requires the local_gpu CUDA GPU"
        raise ValueError(message)
    if not torch.cuda.is_available():
        message = "CUDA is unavailable; refusing to silently run the local_gpu campaign on CPU"
        raise RuntimeError(message)
    queued = cast("list[TheoryBridgeJob]", context["jobs"])
    if job_key is not None:
        queued = [job for job in queued if job.key == job_key]
        if not queued:
            message = f"job key is not present in the frozen queue: {job_key}"
            raise KeyError(message)
    executed = 0
    for job in queued:
        output = _result_path(root, job)
        if output.exists():
            continue
        if limit is not None and executed >= limit:
            break
        started = perf_counter()
        try:
            row = run_job(job, root=root, device=device, data_root=data_root)
            row["elapsed_seconds"] = perf_counter() - started
            _write_json(output, row)
            failure = _failure_path(root, job)
            if failure.exists():
                failure.unlink()
        except Exception as error:
            _write_json(
                _failure_path(root, job),
                {
                    "schema": f"{SCHEMA}.failure",
                    "source_sha256": context["source_sha256"],
                    "source_snapshot_sha256": context["source_snapshot_sha256"],
                    "contract_sha256": context["contract_sha256"],
                    "queue_manifest_sha256": context["queue_manifest_sha256"],
                    "job": asdict(job),
                    "job_key": job.key,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                },
            )
            raise
        finally:
            executed += 1
            gc.collect()
            torch.cuda.empty_cache()
    return report(root)


def run_job(
    job: TheoryBridgeJob,
    *,
    root: Path,
    device: PACDevice,
    data_root: Path,
) -> dict[str, object]:
    """Execute one train/freeze/calibrate/TEST job and preserve all sidecars."""
    context = _validated_campaign_context(root)
    if job.key not in cast("set[str]", context["expected_keys"]):
        message = f"job is not present in the frozen v2 queue: {job.key}"
        raise KeyError(message)
    if device != "cuda" or not torch.cuda.is_available():
        message = "a theory-bridge job requires an available CUDA device"
        raise RuntimeError(message)
    seed_everything(job.seed)
    protocol_events: list[str] = ["official_train_only_loaded"]
    train_only = ensure_ucr_train_only(job.dataset, data_root, allow_download=True)
    split = stratified_three_way_indices(train_only.train_labels, job.seed)
    protocol_events.append("official_train_three_way_split")
    split_payload = _split_payload(job, train_only.train_labels, split)
    split_path = root / "manifests" / "splits" / f"{_safe_key(job.key)}.json"
    _write_json(split_path, split_payload)

    raw_representation = train_only.train_inputs.index_select(0, split.representation_train)
    raw_validation = train_only.train_inputs.index_select(0, split.validation)
    raw_calibration = train_only.train_inputs.index_select(0, split.calibration)
    normalization_mean = raw_representation.mean()
    normalization_std = raw_representation.std(unbiased=False).clamp_min(1.0e-6)
    representation = (raw_representation - normalization_mean) / normalization_std
    validation = (raw_validation - normalization_mean) / normalization_std
    calibration = (raw_calibration - normalization_mean) / normalization_std
    representation_labels = train_only.train_labels.index_select(0, split.representation_train)
    validation_labels = train_only.train_labels.index_select(0, split.validation)
    calibration_labels = train_only.train_labels.index_select(0, split.calibration)
    protocol_events.append("preprocessing_fit_on_representation_train")

    spec = confirmatory_trial_spec("pac_tf", TRIAL)
    config = PACExperimentConfig(
        representation.shape[0],
        validation.shape[0],
        0,
        representation.shape[1],
        raw_input_dim=representation.shape[-1],
        output_dim=train_only.class_count,
        model_dim=MODEL_DIM,
        modes=TRAIN_MODES,
        epochs=EPOCHS,
        batch_size=spec.batch_size,
        learning_rate=spec.learning_rate,
        weight_decay=spec.weight_decay,
        grad_clip_norm=spec.grad_clip_norm,
        seeds=(job.seed,),
        device="cuda",
        optimizer_mode="fused",
    )
    from .alphabet_backbone import AlphabetBackbone  # noqa: PLC0415

    model = AlphabetBackbone(config, train_only.class_count, objective="classification").cuda()
    trainable_parameter_count = count_parameters(model)
    initial_bank = capture_bank_state(model)
    training_task = PACClassificationTask(
        job.dataset,
        representation,
        representation_labels,
        validation,
        validation_labels,
        representation[:0],
        representation_labels[:0],
        train_only.class_count,
    )
    outcome = train_classifier(
        model,
        training_task,
        config,
        "cuda",
        job.seed,
        evaluate_test=False,
        restore_best_validation=True,
    )
    protocol_events.append("end_to_end_ce_model_validation_selected")
    model.eval()
    model.requires_grad_(requires_grad=False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        message = "model freeze failed before calibration"
        raise RuntimeError(message)
    learned_bank = capture_bank_state(model)
    protocol_events.append("encoder_banks_and_ce_head_frozen")

    intervention_seed = _derived_seed(job.seed, "fixed-interventions")
    pole_randomized = randomized_bank_state(
        learned_bank, "pole_randomized", intervention_seed
    )
    direction_randomized = randomized_bank_state(
        learned_bank, "direction_randomized", intervention_seed + 1
    )
    random_draw_rows: list[dict[str, object]] = []
    random_states: list[BankState] = []
    for draw in range(RANDOM_DRAWS):
        draw_seed = _derived_seed(job.seed, f"fully-randomized-{draw}")
        state = randomized_bank_state(learned_bank, "fully_randomized", draw_seed)
        with temporary_bank_state(model, state):
            draw_features = extract_energy_views(
                model,
                representation,
                batch_size=config.batch_size,
                device="cuda",
            )
        summaries = _bank_selection_summaries(draw_features, representation_labels)
        random_draw_rows.append(
            {
                "draw": draw,
                "seed": draw_seed,
                "selection_pool": "representation_train",
                "views": summaries,
                "primary_empirical_hat_delta_32": summaries[PRIMARY_VIEW][
                    str(PRIMARY_MODES)
                ]["empirical_hat_delta_m"],
            }
        )
        random_states.append(state)
    ranked = sorted(
        range(RANDOM_DRAWS),
        key=lambda index: (
            float(random_draw_rows[index]["primary_empirical_hat_delta_32"]),
            -index,
        ),
    )
    median_draw = ranked[(len(ranked) - 1) // 2]
    best_draw = ranked[-1]
    condition_states = {
        "learned": learned_bank,
        "initial": initial_bank,
        "pole_randomized": pole_randomized,
        "direction_randomized": direction_randomized,
        "random_best": random_states[best_draw],
        "random_median": random_states[median_draw],
    }
    protocol_events.append("all_intervention_banks_fixed_without_calibration_or_test")

    frozen_checkpoint_path = root / "checkpoints" / f"{_safe_key(job.key)}.pt"
    frozen_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": f"{SCHEMA}.frozen_checkpoint",
            "job": asdict(job),
            "config": asdict(config),
            "best_epoch": outcome.best_epoch,
            "model_state_dict": {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            },
            "initial_bank": _bank_payload(initial_bank),
            "learned_bank": _bank_payload(learned_bank),
            "condition_states": {
                name: _bank_payload(state) for name, state in condition_states.items()
            },
            "random_draw_selection": {
                "pool": "representation_train",
                "best_draw": best_draw,
                "median_draw": median_draw,
                "draws": random_draw_rows,
            },
            "split": split_payload,
            "normalization": {
                "mean": normalization_mean.detach().cpu(),
                "std": normalization_std.detach().cpu(),
            },
            "official_test_accessed": False,
        },
        frozen_checkpoint_path,
    )
    protocol_events.append("frozen_checkpoint_saved_before_test")

    calibration_features: dict[str, dict[EnergyView, Tensor]] = {}
    for condition, state in condition_states.items():
        with temporary_bank_state(model, state):
            calibration_features[condition] = extract_energy_views(
                model,
                calibration,
                batch_size=config.batch_size,
                device="cuda",
            )
    calibration_bundle = _fit_calibration_bundle(
        calibration_features,
        calibration_labels,
        seed=_derived_seed(job.seed, "calibration-curves"),
    )
    calibration_bundle["condition_states"] = {
        name: _bank_payload(state) for name, state in condition_states.items()
    }
    calibration_bundle["random_draw_selection"] = {
        "best_draw": best_draw,
        "median_draw": median_draw,
        "draws": random_draw_rows,
    }
    calibration_bundle["official_test_accessed"] = False
    calibration_path = root / "calibration_heads" / f"{_safe_key(job.key)}.pt"
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(calibration_bundle, calibration_path)
    protocol_events.append("all_prototype_heads_fit_from_calibration_only")

    full_dataset = ensure_ucr_dataset(
        job.dataset,
        data_root,
        allow_download=True,
        require_train_label_space=True,
    )
    _assert_same_official_train(train_only.train_inputs, full_dataset.train_inputs)
    _assert_same_official_train(train_only.train_labels, full_dataset.train_labels)
    test_inputs = (full_dataset.test_inputs - normalization_mean) / normalization_std
    test_labels = full_dataset.test_labels
    protocol_events.append("official_test_loaded_after_freeze_and_calibration")

    with temporary_bank_state(model, learned_bank):
        ce_metrics = classification_metric_bundle(
            model,
            test_inputs.cuda(),
            test_labels.cuda(),
            batch_size=config.batch_size,
        )
    conditions: dict[str, Any] = {}
    for condition, state in condition_states.items():
        with temporary_bank_state(model, state):
            test_features = extract_energy_views(
                model,
                test_inputs,
                batch_size=config.batch_size,
                device="cuda",
            )
        conditions[condition] = _evaluate_condition(
            calibration_bundle["conditions"][condition],
            test_features,
            test_labels,
        )
    protocol_events.append("ce_and_prototype_heads_evaluated_on_same_official_test")

    primary = cast(
        "dict[str, Any]",
        conditions["learned"]["views"][PRIMARY_VIEW][str(PRIMARY_MODES)]["full_calibration"],
    )
    secondary = cast(
        "dict[str, Any]",
        conditions["learned"]["views"][SECONDARY_VIEW][str(PRIMARY_MODES)][
            "full_calibration"
        ],
    )
    row: dict[str, Any] = {
        "schema": f"{SCHEMA}.result",
        "status": "done",
        "source_sha256": context["source_sha256"],
        "source_snapshot_sha256": context["source_snapshot_sha256"],
        "contract_sha256": context["contract_sha256"],
        "queue_manifest_sha256": context["queue_manifest_sha256"],
        "job_key": job.key,
        "dataset": job.dataset,
        "seed": job.seed,
        "protocol_events": protocol_events,
        "official_test_accessed": True,
        "split_counts": {
            "representation_train": int(representation.shape[0]),
            "validation": int(validation.shape[0]),
            "calibration": int(calibration.shape[0]),
            "test": int(test_inputs.shape[0]),
        },
        "class_count": train_only.class_count,
        "sequence_length": int(representation.shape[1]),
        "model_dim": MODEL_DIM,
        "trained_modes": TRAIN_MODES,
        "mode_prefixes": list(MODE_PREFIXES),
        "params_trainable_before_freeze": trainable_parameter_count,
        "best_epoch": outcome.best_epoch,
        "training_elapsed_seconds": outcome.elapsed_time,
        "ce_test_metrics": {
            "accuracy": ce_metrics.accuracy,
            "balanced_accuracy": ce_metrics.balanced_accuracy,
            "macro_f1": ce_metrics.macro_f1,
            "weighted_f1": ce_metrics.weighted_f1,
        },
        "conditions": conditions,
        "random_draw_selection": {
            "selection_pool": "representation_train",
            "selection_uses_labels": True,
            "calibration_independent": True,
            "official_test_independent": True,
            "selected_conditions_are_not_unselected_iid_draws": True,
            "best_draw": best_draw,
            "median_draw": median_draw,
            "draws": random_draw_rows,
        },
        "primary_bridge": {
            "view": PRIMARY_VIEW,
            "modes_per_bank": PRIMARY_MODES,
            "effective_dimension": PRIMARY_MODES,
            "prototype_estimator": primary["prototype_estimator"],
            "empirical_hat_delta_m": primary["empirical_hat_delta_m"],
            "hat_delta_standard_error": primary["hat_delta_standard_error"],
            "hat_delta_se_snr": primary["hat_delta_se_snr"],
            "hat_delta_uncertainty_method": primary[
                "hat_delta_uncertainty_method"
            ],
            "hat_delta_uncertainty_interval_95": primary[
                "hat_delta_uncertainty_interval_95"
            ],
            "prototype_test_balanced_accuracy": primary["test_metrics"]["balanced_accuracy"],
            "ce_test_balanced_accuracy": ce_metrics.balanced_accuracy,
            "prototype_minus_ce_balanced_accuracy": (
                float(primary["test_metrics"]["balanced_accuracy"])
                - ce_metrics.balanced_accuracy
            ),
        },
        "secondary_joint_bridge": {
            "view": SECONDARY_VIEW,
            "modes_per_bank": PRIMARY_MODES,
            "effective_dimension": 2 * PRIMARY_MODES,
            "prototype_estimator": secondary["prototype_estimator"],
            "empirical_hat_delta_m": secondary["empirical_hat_delta_m"],
            "hat_delta_standard_error": secondary["hat_delta_standard_error"],
            "hat_delta_se_snr": secondary["hat_delta_se_snr"],
            "hat_delta_uncertainty_method": secondary[
                "hat_delta_uncertainty_method"
            ],
            "hat_delta_uncertainty_interval_95": secondary[
                "hat_delta_uncertainty_interval_95"
            ],
            "prototype_test_balanced_accuracy": secondary["test_metrics"][
                "balanced_accuracy"
            ],
            "ce_test_balanced_accuracy": ce_metrics.balanced_accuracy,
            "prototype_minus_ce_balanced_accuracy": (
                float(secondary["test_metrics"]["balanced_accuracy"])
                - ce_metrics.balanced_accuracy
            ),
        },
        "artifacts": {
            "split_manifest": str(split_path.resolve()),
            "split_manifest_sha256": _sha256_path(split_path),
            "frozen_checkpoint": str(frozen_checkpoint_path.resolve()),
            "frozen_checkpoint_sha256": _sha256_path(frozen_checkpoint_path),
            "calibration_heads": str(calibration_path.resolve()),
            "calibration_heads_sha256": _sha256_path(calibration_path),
        },
    }
    raw_path = root / "raw_rows" / f"{_safe_key(job.key)}.json"
    _write_json(raw_path, row)
    row["artifacts"]["raw_row"] = str(raw_path.resolve())
    row["artifacts"]["raw_row_sha256"] = _sha256_path(raw_path)
    return row


def report(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    context = _validated_campaign_context(root)
    rows = _validated_result_rows(root, context, kind="completed")
    failed_rows = _validated_result_rows(root, context, kind="failed")
    _reject_overlapping_terminal_rows(
        {str(row["job_key"]) for row in rows},
        {str(row["job_key"]) for row in failed_rows},
    )
    primary = [cast("dict[str, Any]", row["primary_bridge"]) for row in rows]
    secondary = [cast("dict[str, Any]", row["secondary_joint_bridge"]) for row in rows]
    hat_delta = [float(value["empirical_hat_delta_m"]) for value in primary]
    prototype_ba = [float(value["prototype_test_balanced_accuracy"]) for value in primary]
    ce_ba = [float(value["ce_test_balanced_accuracy"]) for value in primary]
    datasets = [str(row["dataset"]) for row in rows]
    conditions = _condition_aggregates(rows)
    paired_contrasts = _paired_primary_contrasts(rows)
    mode_trends = _mode_trends(rows)
    sample_efficiency = _sample_efficiency_contrast(rows)
    gaps = [
        float(value["prototype_minus_ce_balanced_accuracy"])
        for value in primary
    ]
    gap_paired_interval = _dataset_cluster_bootstrap_interval(
        gaps,
        datasets,
        seed=_derived_seed(0, "primary-prototype-minus-ce"),
    )
    se_snr = [float(value["hat_delta_se_snr"]) for value in primary]
    current_status = status(root)
    prototype_correlations = {
        "overall": _correlations(hat_delta, prototype_ba),
        "task_centered": _correlations(
            _task_center(hat_delta, datasets),
            _task_center(prototype_ba, datasets),
        ),
    }
    ce_correlations = {
        "overall": _correlations(hat_delta, ce_ba),
        "task_centered": _correlations(
            _task_center(hat_delta, datasets),
            _task_center(ce_ba, datasets),
        ),
    }
    decision_gates = _decision_gates(
        current_status,
        gaps=gaps,
        gap_paired_interval=gap_paired_interval,
        se_snr=se_snr,
        prototype_correlations=prototype_correlations,
        paired_contrasts=paired_contrasts,
        sample_efficiency=sample_efficiency,
    )
    summary: dict[str, object] = {
        "schema": f"{SCHEMA}.report",
        "rows": len(rows),
        "status": current_status,
        "evaluation_state": decision_gates["evaluation_state"],
        "primary_estimand": (
            "learned writer-energy prototype log1p(class mean raw energy), "
            "M=32, full independent calibration pool"
        ),
        "secondary_estimand": (
            "learned joint writer+reader prototype, M=32 per bank (2M dimensions), "
            "full independent calibration pool"
        ),
        "provenance": {
            "queue_manifest_sha256": context["queue_manifest_sha256"],
            "contract_sha256": context["contract_sha256"],
            "source_snapshot_sha256": context["source_snapshot_sha256"],
            "validated_result_rows_only": True,
        },
        "nested_prefix_protocol_limitation": (
            "M=2,4,8,16 are descriptor-level prefixes of a representation trained once at M=32. "
            "They isolate nested observed energy coordinates without retraining capacity, but the "
            "reader trajectory still depends on the full learned M=32 writer synthesis; these "
            "curves are not equivalent to independently trained smaller-M architectures."
        ),
        "primary": {
            "empirical_hat_delta_m": _summary(hat_delta),
            "prototype_test_balanced_accuracy": _summary(prototype_ba),
            "ce_test_balanced_accuracy": _summary(ce_ba),
            "prototype_minus_ce_balanced_accuracy": {
                **_summary(gaps),
                "paired_dataset_cluster_bootstrap_interval_95": (
                    gap_paired_interval
                ),
                "bootstrap_replicates": AGGREGATE_BOOTSTRAP_REPLICATES,
            },
            "hat_delta_se_snr": _summary(se_snr),
            "se_snr_gt_3": {
                "count": sum(value > 3.0 for value in se_snr),
                "total": len(se_snr),
            },
            "prototype_absolute_within_2pp_of_ce": {
                "count": sum(abs(value) <= 0.02 for value in gaps),
                "total": len(gaps),
            },
            "prototype_no_more_than_2pp_below_ce": {
                "count": sum(value >= -0.02 for value in gaps),
                "total": len(gaps),
            },
        },
        "secondary_joint": {
            "empirical_hat_delta_m": _summary(
                [float(value["empirical_hat_delta_m"]) for value in secondary]
            ),
            "hat_delta_se_snr": _summary(
                [float(value["hat_delta_se_snr"]) for value in secondary]
            ),
            "prototype_test_balanced_accuracy": _summary(
                [
                    float(value["prototype_test_balanced_accuracy"])
                    for value in secondary
                ]
            ),
            "prototype_minus_ce_balanced_accuracy": _summary(
                [
                    float(value["prototype_minus_ce_balanced_accuracy"])
                    for value in secondary
                ]
            ),
        },
        "correlations": {
            "hat_delta_vs_prototype_test_ba": prototype_correlations,
            "hat_delta_vs_ce_test_ba": ce_correlations,
        },
        "condition_aggregates": conditions,
        "paired_primary_contrasts": paired_contrasts,
        "mode_trends": mode_trends,
        "sample_efficiency": sample_efficiency,
        "decision_gates": decision_gates,
        "raw_result_files": [
            str((root / "completed" / f"{_safe_key(str(row['job_key']))}.json").resolve())
            for row in rows
        ],
    }
    report_dir = root / "reports"
    _write_json(report_dir / "summary.json", summary)
    _write_report_markdown(report_dir / "SUMMARY.md", summary)
    _write_per_run_csv(report_dir / "per_run.csv", rows)
    return summary


def _fit_calibration_bundle(
    condition_features: Mapping[str, Mapping[EnergyView, Tensor]],
    labels: Tensor,
    *,
    seed: int,
) -> dict[str, Any]:
    subsets = {
        f"{fraction:.2f}": calibration_subset_indices(labels, fraction, seed)
        for fraction in CALIBRATION_FRACTIONS
    }
    conditions: dict[str, Any] = {}
    for condition, features in condition_features.items():
        view_payload: dict[str, Any] = {}
        for view in ENERGY_VIEWS:
            mode_payload: dict[str, Any] = {}
            for modes in MODE_PREFIXES:
                values = prefix_energy_view(features, view, modes)
                curve: dict[str, Any] = {}
                for fraction_key, indices in subsets.items():
                    use_class_bootstrap = (
                        fraction_key == PRIMARY_FRACTION_KEY
                        and modes == PRIMARY_MODES
                        and view in {PRIMARY_VIEW, SECONDARY_VIEW}
                    )
                    curve[fraction_key] = {
                        "requested_fraction": float(fraction_key),
                        "indices": indices,
                        "head": fit_prototype_head(
                            values.index_select(0, indices),
                            labels.index_select(0, indices),
                            bootstrap_seed=_derived_seed(
                                seed,
                                f"{condition}-{view}-m{modes}-{fraction_key}",
                            ),
                            bootstrap_replicates=(
                                CLASS_BOOTSTRAP_REPLICATES
                                if use_class_bootstrap
                                else 0
                            ),
                        ),
                    }
                mode_payload[str(modes)] = curve
            view_payload[view] = mode_payload
        conditions[condition] = view_payload
    return {
        "schema": f"{SCHEMA}.calibration_heads",
        "calibration_labels": labels.detach().cpu(),
        "calibration_subsets": subsets,
        "conditions": conditions,
    }


def _evaluate_condition(
    calibrated: Mapping[str, Any],
    test_features: Mapping[EnergyView, Tensor],
    test_labels: Tensor,
) -> dict[str, object]:
    views: dict[str, Any] = {}
    for view in ENERGY_VIEWS:
        mode_rows: dict[str, Any] = {}
        for modes in MODE_PREFIXES:
            active_test = prefix_energy_view(
                test_features,
                view,
                modes,
            )
            curve_rows: list[dict[str, object]] = []
            for fraction_key in (f"{value:.2f}" for value in CALIBRATION_FRACTIONS):
                fit = calibrated[view][str(modes)][fraction_key]
                head = fit["head"]
                test_metrics = evaluate_prototype_head(active_test, test_labels, head)
                curve_rows.append(
                    {
                        "requested_fraction": fit["requested_fraction"],
                        "effective_calibration_size": head["calibration_size"],
                        "empirical_hat_delta_m": head["empirical_hat_delta_m"],
                        "normalized_empirical_hat_delta_m": head[
                            "normalized_empirical_hat_delta_m"
                        ],
                        "hat_delta_se_snr": head["hat_delta_se_snr"],
                        "hat_delta_uncertainty_method": head[
                            "hat_delta_uncertainty_method"
                        ],
                        "test_metrics": test_metrics,
                    }
                )
            full_fit = calibrated[view][str(modes)][PRIMARY_FRACTION_KEY]["head"]
            full_metrics = evaluate_prototype_head(active_test, test_labels, full_fit)
            mode_rows[str(modes)] = {
                "calibration_curve": curve_rows,
                "full_calibration": {
                    **_json_head_diagnostics(full_fit),
                    "test_metrics": full_metrics,
                },
            }
        views[view] = mode_rows
    return {"views": views}


def _bank_selection_summaries(
    features: Mapping[EnergyView, Tensor],
    labels: Tensor,
) -> dict[str, dict[str, dict[str, object]]]:
    result: dict[str, dict[str, dict[str, object]]] = {}
    for view in ENERGY_VIEWS:
        result[view] = {}
        for modes in MODE_PREFIXES:
            diagnostics = fit_prototype_head(
                prefix_energy_view(features, view, modes),
                labels,
                bootstrap_replicates=0,
            )
            result[view][str(modes)] = {
                "empirical_hat_delta_m": diagnostics["empirical_hat_delta_m"],
                "normalized_empirical_hat_delta_m": diagnostics[
                    "normalized_empirical_hat_delta_m"
                ],
                "hat_delta_se_snr": diagnostics["hat_delta_se_snr"],
                "hat_delta_uncertainty_method": diagnostics[
                    "hat_delta_uncertainty_method"
                ],
                "minimum_class_pair": diagnostics["minimum_class_pair"],
            }
    return result


def _condition_aggregates(rows: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    if not rows:
        return {}
    output: dict[str, Any] = {}
    condition_names = tuple(cast("Mapping[str, Any]", rows[0]["conditions"]))
    for condition in condition_names:
        output[condition] = {}
        for view in ENERGY_VIEWS:
            view_output: dict[str, object] = {}
            for modes in MODE_PREFIXES:
                selected = [
                    row["conditions"][condition]["views"][view][str(modes)][
                        "full_calibration"
                    ]
                    for row in rows
                ]
                view_output[str(modes)] = {
                    "empirical_hat_delta_m": _summary(
                        [float(value["empirical_hat_delta_m"]) for value in selected]
                    ),
                    "hat_delta_se_snr": _summary(
                        [float(value["hat_delta_se_snr"]) for value in selected]
                    ),
                    "test_balanced_accuracy": _summary(
                        [float(value["test_metrics"]["balanced_accuracy"]) for value in selected]
                    ),
                }
            output[condition][view] = view_output
    return output


def _primary_condition_record(
    row: Mapping[str, Any],
    condition: str,
) -> Mapping[str, Any]:
    return row["conditions"][condition]["views"][PRIMARY_VIEW][str(PRIMARY_MODES)][
        "full_calibration"
    ]


def _paired_primary_contrasts(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    """Aggregate within-run learned-minus-counterfactual primary contrasts."""
    output: dict[str, object] = {}
    datasets = [str(row["dataset"]) for row in rows]
    for comparator in PRIMARY_COMPARATORS:
        metric_differences: dict[str, list[float]] = {
            "empirical_hat_delta_m": [],
            "hat_delta_se_snr": [],
            "test_balanced_accuracy": [],
        }
        for row in rows:
            learned = _primary_condition_record(row, "learned")
            counterfactual = _primary_condition_record(row, comparator)
            metric_differences["empirical_hat_delta_m"].append(
                float(learned["empirical_hat_delta_m"])
                - float(counterfactual["empirical_hat_delta_m"])
            )
            metric_differences["hat_delta_se_snr"].append(
                float(learned["hat_delta_se_snr"])
                - float(counterfactual["hat_delta_se_snr"])
            )
            metric_differences["test_balanced_accuracy"].append(
                float(learned["test_metrics"]["balanced_accuracy"])
                - float(counterfactual["test_metrics"]["balanced_accuracy"])
            )
        output[comparator] = {
            "contrast": "learned minus comparator, paired within dataset/seed",
            "empirical_hat_delta_m": {
                **_paired_metric_summary(
                    metric_differences["empirical_hat_delta_m"],
                    datasets,
                    seed_label=f"{comparator}-hat-delta",
                ),
                "learned_wins": sum(
                    value > 0.0
                    for value in metric_differences["empirical_hat_delta_m"]
                ),
            },
            "hat_delta_se_snr": {
                **_paired_metric_summary(
                    metric_differences["hat_delta_se_snr"],
                    datasets,
                    seed_label=f"{comparator}-hat-delta-se-snr",
                ),
                "learned_wins": sum(
                    value > 0.0
                    for value in metric_differences["hat_delta_se_snr"]
                ),
            },
            "test_balanced_accuracy": {
                **_paired_metric_summary(
                    metric_differences["test_balanced_accuracy"],
                    datasets,
                    seed_label=f"{comparator}-test-ba",
                ),
                "learned_wins": sum(
                    value > 0.0 for value in metric_differences["test_balanced_accuracy"]
                ),
            },
        }
    return output


def _mode_trends(rows: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    """Report M-wise margins, performance, and a scale-normalized collision count."""
    if not rows:
        return {}
    condition_names = tuple(cast("Mapping[str, Any]", rows[0]["conditions"]))
    output: dict[str, object] = {}
    for condition in condition_names:
        by_mode: dict[str, object] = {}
        per_run_delta: dict[int, list[float]] = {modes: [] for modes in MODE_PREFIXES}
        per_run_ba: dict[int, list[float]] = {modes: [] for modes in MODE_PREFIXES}
        for modes in MODE_PREFIXES:
            selected = [
                row["conditions"][condition]["views"][PRIMARY_VIEW][str(modes)][
                    "full_calibration"
                ]
                for row in rows
            ]
            delta_values = [float(value["empirical_hat_delta_m"]) for value in selected]
            normalized_values = [
                float(value["normalized_empirical_hat_delta_m"]) for value in selected
            ]
            se_values = [
                float(value["hat_delta_se_snr"]) for value in selected
            ]
            ba_values = [
                float(value["test_metrics"]["balanced_accuracy"]) for value in selected
            ]
            per_run_delta[modes] = delta_values
            per_run_ba[modes] = ba_values
            near_collisions = sum(
                value <= NEAR_COLLISION_NORMALIZED_THRESHOLD
                for value in normalized_values
            )
            by_mode[str(modes)] = {
                "empirical_hat_delta_m": _summary(delta_values),
                "normalized_empirical_hat_delta_m": _summary(normalized_values),
                "hat_delta_se_snr": _summary(se_values),
                "test_balanced_accuracy": _summary(ba_values),
                "near_collision": {
                    "definition": (
                        "normalized_empirical_hat_delta_m <= "
                        f"{NEAR_COLLISION_NORMALIZED_THRESHOLD:g}"
                    ),
                    "count": near_collisions,
                    "total": len(selected),
                    "rate": near_collisions / len(selected) if selected else None,
                },
            }
        delta_change = [
            final - initial
            for initial, final in zip(
                per_run_delta[MODE_PREFIXES[0]],
                per_run_delta[MODE_PREFIXES[-1]],
                strict=True,
            )
        ]
        ba_change = [
            final - initial
            for initial, final in zip(
                per_run_ba[MODE_PREFIXES[0]],
                per_run_ba[MODE_PREFIXES[-1]],
                strict=True,
            )
        ]
        output[condition] = {
            "by_mode": by_mode,
            "m2_to_m32": {
                "empirical_hat_delta_m_change": {
                    **_summary(delta_change),
                    "positive_runs": sum(value > 0.0 for value in delta_change),
                },
                "test_balanced_accuracy_change": {
                    **_summary(ba_change),
                    "positive_runs": sum(value > 0.0 for value in ba_change),
                },
            },
        }
    return output


def _curve_mean_balanced_accuracy(record: Mapping[str, Any]) -> float:
    curve = cast("Sequence[Mapping[str, Any]]", record["calibration_curve"])
    return mean(float(point["test_metrics"]["balanced_accuracy"]) for point in curve)


def _sample_efficiency_contrast(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    learned_values: list[float] = []
    random_values: list[float] = []
    differences: list[float] = []
    datasets: list[str] = []
    for row in rows:
        learned = row["conditions"]["learned"]["views"][PRIMARY_VIEW][
            str(PRIMARY_MODES)
        ]
        random_median = row["conditions"]["random_median"]["views"][PRIMARY_VIEW][
            str(PRIMARY_MODES)
        ]
        learned_curve_mean = _curve_mean_balanced_accuracy(learned)
        random_curve_mean = _curve_mean_balanced_accuracy(random_median)
        learned_values.append(learned_curve_mean)
        random_values.append(random_curve_mean)
        differences.append(learned_curve_mean - random_curve_mean)
        datasets.append(str(row["dataset"]))
    return {
        "metric": (
            "mean TEST balanced accuracy over pre-registered calibration fractions "
            "0.25,0.50,0.75,1.00"
        ),
        "learned": _summary(learned_values),
        "random_median": _summary(random_values),
        "paired_learned_minus_random_median": {
            **_paired_metric_summary(
                differences,
                datasets,
                seed_label="sample-efficiency-random-median",
            ),
            "learned_wins": sum(value > 0.0 for value in differences),
        },
    }


def _decision_gates(
    campaign_status: Mapping[str, object],
    *,
    gaps: Sequence[float],
    gap_paired_interval: Sequence[float | None],
    se_snr: Sequence[float],
    prototype_correlations: Mapping[str, Any],
    paired_contrasts: Mapping[str, Any],
    sample_efficiency: Mapping[str, Any],
) -> dict[str, object]:
    """Evaluate pre-registered gates only after the complete 8x5 grid exists."""
    complete = bool(campaign_status["done"])
    mean_gap = mean(gaps) if gaps else None
    mean_absolute_gap = mean(abs(value) for value in gaps) if gaps else None
    se_majority = (
        sum(value > 3.0 for value in se_snr) > len(se_snr) / 2
        if se_snr
        else None
    )
    centered = prototype_correlations["task_centered"]
    centered_positive = (
        float(centered["pearson_r"]) > 0.0 and float(centered["spearman_rho"]) > 0.0
        if centered["pearson_r"] is not None and centered["spearman_rho"] is not None
        else None
    )
    random_contrast = paired_contrasts.get("random_median", {})
    delta_contrast = cast(
        "Mapping[str, object]",
        random_contrast.get("empirical_hat_delta_m", {}),
    )
    ba_contrast = cast(
        "Mapping[str, object]",
        random_contrast.get("test_balanced_accuracy", {}),
    )
    efficiency_contrast = cast(
        "Mapping[str, object]",
        sample_efficiency.get("paired_learned_minus_random_median", {}),
    )
    delta_interval = cast(
        "Sequence[float | None]",
        delta_contrast.get("paired_dataset_cluster_bootstrap_interval_95", []),
    )
    efficiency_interval = cast(
        "Sequence[float | None]",
        efficiency_contrast.get("paired_dataset_cluster_bootstrap_interval_95", []),
    )
    learned_delta_better = _interval_lower_exceeds(delta_interval, 0.0)
    learned_ba_better = (
        float(ba_contrast["mean"]) > 0.0
        if ba_contrast.get("mean") is not None
        else None
    )
    learned_efficiency_better = _interval_lower_exceeds(efficiency_interval, 0.0)
    primary_gap_noninferior = _interval_lower_at_least(gap_paired_interval, -0.02)
    p0_values = (
        primary_gap_noninferior,
        se_majority,
        centered_positive,
    )
    p1_values = (learned_delta_better, learned_efficiency_better)
    p0_status = _gate_status(complete=complete, values=p0_values)
    p1_status = _gate_status(complete=complete, values=p1_values)
    overall = (
        "incomplete"
        if not complete
        else ("pass" if p0_status == "pass" and p1_status == "pass" else "fail")
    )
    return {
        "evaluation_state": overall,
        "complete_grid_required_before_pass_fail": True,
        "completed_rows": campaign_status["completed"],
        "expected_rows": campaign_status["expected"],
        "p0_split_calibration_bridge": {
            "status": p0_status,
            "criteria": {
                "paired_prototype_minus_ce_ba_noninferior_at_minus_0_02": {
                    "value": mean_gap,
                    "paired_dataset_cluster_bootstrap_interval_95": list(
                        gap_paired_interval
                    ),
                    "threshold": -0.02,
                    "satisfied_if_complete": primary_gap_noninferior,
                },
                "strict_mean_absolute_gap_at_most_0_02_diagnostic": {
                    "value": mean_absolute_gap,
                    "threshold": 0.02,
                    "satisfied_if_complete": (
                        None
                        if mean_absolute_gap is None
                        else mean_absolute_gap <= 0.02
                    ),
                },
                "empirical_hat_delta_se_snr_gt_3_in_majority": {
                    "count": sum(value > 3.0 for value in se_snr),
                    "total": len(se_snr),
                    "satisfied_if_complete": se_majority,
                },
                "positive_task_centered_margin_test_correlation": {
                    "pearson_r": centered["pearson_r"],
                    "spearman_rho": centered["spearman_rho"],
                    "satisfied_if_complete": centered_positive,
                },
            },
        },
        "p1_2_learned_vs_random_bank": {
            "status": p1_status,
            "criteria": {
                "learned_empirical_hat_delta_m_gt_random_median": {
                    "paired_mean_difference": delta_contrast.get("mean"),
                    "paired_dataset_cluster_bootstrap_interval_95": list(
                        delta_interval
                    ),
                    "satisfied_if_complete": learned_delta_better,
                },
                "learned_calibration_efficiency_gt_random_median": {
                    "paired_mean_difference": efficiency_contrast.get("mean"),
                    "paired_dataset_cluster_bootstrap_interval_95": list(
                        efficiency_interval
                    ),
                    "satisfied_if_complete": learned_efficiency_better,
                },
                "learned_test_ba_gt_random_median_diagnostic": {
                    "paired_mean_difference": ba_contrast.get("mean"),
                    "satisfied_if_complete": learned_ba_better,
                },
            },
        },
    }


def _gate_status(*, complete: bool, values: Sequence[bool | None]) -> str:
    if not complete:
        return "incomplete"
    return "pass" if all(value is True for value in values) else "fail"


def _correlations(left: Sequence[float], right: Sequence[float]) -> dict[str, object]:
    if len(left) < 3 or len(set(left)) < 2 or len(set(right)) < 2:
        return {
            "n": len(left),
            "pearson_r": None,
            "pearson_p": None,
            "spearman_rho": None,
            "spearman_p": None,
        }
    from scipy.stats import pearsonr, spearmanr  # noqa: PLC0415

    pearson_r, pearson_p = pearsonr(left, right)
    spearman_r, spearman_p = spearmanr(left, right)
    return {
        "n": len(left),
        "pearson_r": _finite_or_none(float(pearson_r)),
        "pearson_p": _finite_or_none(float(pearson_p)),
        "spearman_rho": _finite_or_none(float(spearman_r)),
        "spearman_p": _finite_or_none(float(spearman_p)),
    }


def _task_center(values: Sequence[float], datasets: Sequence[str]) -> list[float]:
    means = {
        dataset: mean(
            value
            for value, active in zip(values, datasets, strict=True)
            if active == dataset
        )
        for dataset in set(datasets)
    }
    return [
        value - means[dataset]
        for value, dataset in zip(values, datasets, strict=True)
    ]


def _summary(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "sample_sd": None}
    return {
        "n": len(values),
        "mean": mean(values),
        "median": median(values),
        "sample_sd": stdev(values) if len(values) > 1 else 0.0,
    }


def _paired_metric_summary(
    values: Sequence[float],
    datasets: Sequence[str],
    *,
    seed_label: str,
) -> dict[str, object]:
    return {
        **_summary(values),
        "paired_dataset_cluster_bootstrap_interval_95": (
            _dataset_cluster_bootstrap_interval(
                values,
                datasets,
                seed=_derived_seed(0, seed_label),
            )
        ),
        "bootstrap_replicates": AGGREGATE_BOOTSTRAP_REPLICATES,
    }


def _dataset_cluster_bootstrap_interval(
    values: Sequence[float],
    datasets: Sequence[str],
    *,
    seed: int,
) -> list[float | None]:
    """Bootstrap paired differences by resampling predeclared task clusters."""
    if len(values) != len(datasets):
        message = "paired values and dataset cluster labels must have equal length"
        raise ValueError(message)
    if not values:
        return [None, None]
    grouped = {
        dataset: [
            float(value)
            for value, active_dataset in zip(values, datasets, strict=True)
            if active_dataset == dataset
        ]
        for dataset in sorted(set(datasets))
    }
    cluster_means = torch.tensor(
        [mean(grouped[dataset]) for dataset in sorted(grouped)],
        dtype=torch.float64,
    )
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(
        cluster_means.numel(),
        (AGGREGATE_BOOTSTRAP_REPLICATES, cluster_means.numel()),
        generator=generator,
    )
    bootstrap_means = cluster_means[indices].mean(dim=1)
    low, high = _tensor_percentile_interval(bootstrap_means)
    return [low, high]


def _interval_lower_exceeds(
    interval: Sequence[float | None],
    threshold: float,
) -> bool | None:
    if len(interval) != 2 or interval[0] is None:
        return None
    return float(interval[0]) > threshold


def _interval_lower_at_least(
    interval: Sequence[float | None],
    threshold: float,
) -> bool | None:
    if len(interval) != 2 or interval[0] is None:
        return None
    return float(interval[0]) >= threshold


def _prediction_metrics(predictions: Tensor, labels: Tensor) -> dict[str, float]:
    classes = torch.unique(labels, sorted=True)
    recalls: list[float] = []
    f1_scores: list[float] = []
    weighted_f1 = 0.0
    for class_value in classes.tolist():
        actual = labels == int(class_value)
        predicted = predictions == int(class_value)
        true_positive = int((actual & predicted).sum())
        support = int(actual.sum())
        predicted_count = int(predicted.sum())
        recall = true_positive / max(support, 1)
        precision = true_positive / max(predicted_count, 1)
        f1 = (
            0.0
            if precision + recall == 0.0
            else 2.0 * precision * recall / (precision + recall)
        )
        recalls.append(recall)
        f1_scores.append(f1)
        weighted_f1 += f1 * support
    return {
        "accuracy": float((predictions == labels).to(torch.float64).mean()),
        "balanced_accuracy": mean(recalls),
        "macro_f1": mean(f1_scores),
        "weighted_f1": weighted_f1 / max(labels.numel(), 1),
    }


def _capture_block(block: _BankBlock) -> BlockBankState:
    return BlockBankState(
        block.raw_decay.detach().cpu().clone(),
        block.raw_frequency.detach().cpu().clone(),
        block.frame_matrix().detach().cpu().clone(),
    )


def _bank_payload(state: BankState) -> dict[str, dict[str, Tensor]]:
    return {
        "writer": {
            "raw_decay": state.writer.raw_decay,
            "raw_frequency": state.writer.raw_frequency,
            "frame": state.writer.frame,
        },
        "reader": {
            "raw_decay": state.reader.raw_decay,
            "raw_frequency": state.reader.raw_frequency,
            "frame": state.reader.frame,
        },
    }


def _json_head_diagnostics(head: Mapping[str, object]) -> dict[str, object]:
    return {
        "calibration_size": head["calibration_size"],
        "calibration_counts": cast("Tensor", head["calibration_counts"]).tolist(),
        "feature_dimension": head["feature_dimension"],
        "prototype_estimator": head["prototype_estimator"],
        "empirical_hat_delta_m": head["empirical_hat_delta_m"],
        "normalized_empirical_hat_delta_m": head[
            "normalized_empirical_hat_delta_m"
        ],
        "hat_delta_standard_error": head["hat_delta_standard_error"],
        "hat_delta_se_snr": head["hat_delta_se_snr"],
        "hat_delta_uncertainty_method": head["hat_delta_uncertainty_method"],
        "hat_delta_uncertainty_interval_95": head[
            "hat_delta_uncertainty_interval_95"
        ],
        "hat_delta_bootstrap_bias": head["hat_delta_bootstrap_bias"],
        "class_bootstrap_replicates": head["class_bootstrap_replicates"],
        "class_bootstrap_seed": head["class_bootstrap_seed"],
        "minimum_class_pair": head["minimum_class_pair"],
        "class_pair_margins": head["class_pair_margins"],
        "singleton_calibration_classes": head["singleton_calibration_classes"],
        "affine_head": {
            "weight_rule": "2 * prototype",
            "bias_rule": "-squared prototype norm",
            "equivalence": "affine logits = negative squared distance + ||x||^2",
        },
    }


def _split_payload(
    job: TheoryBridgeJob,
    labels: Tensor,
    split: ThreeWaySplit,
) -> dict[str, object]:
    return {
        "schema": f"{SCHEMA}.split_manifest",
        "job": asdict(job),
        "official_train_labels_sha256": _tensor_sha256(labels),
        "representation_train_indices": split.representation_train.tolist(),
        "validation_indices": split.validation.tolist(),
        "calibration_indices": split.calibration.tolist(),
        "disjoint": True,
        "complete_partition": True,
        "official_test_accessed": False,
    }


def _assert_same_official_train(expected: Tensor, observed: Tensor) -> None:
    if expected.shape != observed.shape or not torch.equal(expected, observed):
        message = "official TRAIN changed between TRAIN-only and TEST materialization"
        raise RuntimeError(message)


def _tensor_sha256(value: Tensor) -> str:
    active = value.detach().cpu().contiguous()
    header = json.dumps(
        {"shape": list(active.shape), "dtype": str(active.dtype)},
        sort_keys=True,
    ).encode()
    return hashlib.sha256(header + active.numpy().tobytes()).hexdigest()


def _derived_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def _load_jobs(root: Path) -> list[TheoryBridgeJob]:
    path = root / "queue_manifest.jsonl"
    active: list[TheoryBridgeJob] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            message = f"invalid queue JSON at line {line_number}: {error}"
            raise ValueError(message) from error
        if (
            not isinstance(payload, dict)
            or set(payload) != {"dataset", "seed"}
            or not isinstance(payload["dataset"], str)
            or type(payload["seed"]) is not int
        ):
            message = f"invalid queue job schema at line {line_number}"
            raise ValueError(message)
        active.append(TheoryBridgeJob(payload["dataset"], payload["seed"]))
    if not active:
        message = "the frozen v2 queue is empty"
        raise ValueError(message)
    keys = [job.key for job in active]
    if len(set(keys)) != len(keys):
        message = "the frozen v2 queue contains duplicate job keys"
        raise ValueError(message)
    return active


def _source_hashes() -> dict[str, str]:
    return source_file_hashes(
        SOURCE_FILES,
        project_root=Path(__file__).resolve().parents[2],
    )


def _source_manifest() -> dict[str, object]:
    source_sha256 = _source_hashes()
    canonical = json.dumps(
        source_sha256,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return {
        "schema": f"{SCHEMA}.source_manifest",
        "captured_before_first_gpu_job": True,
        "source_sha256": source_sha256,
        "snapshot_payload_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def _superseded_artifact_hashes() -> dict[str, str]:
    return {
        str(path): (
            _sha256_path(path)
            if path.is_file()
            else "unavailable"
        )
        for path in (
            SUPERSEDED_ROOT / "contract.json",
            SUPERSEDED_ROOT / "queue_manifest.jsonl",
            SUPERSEDED_ROOT / "source_manifest.json",
        )
    }


def _validated_campaign_context(root: Path) -> dict[str, Any]:  # noqa: C901, PLR0912
    """Validate the complete frozen campaign envelope before any counting/use."""
    if root.resolve() == SUPERSEDED_ROOT.resolve():
        message = (
            "the v1 theory-bridge root is superseded, immutable, and excluded "
            "from v2 aggregation"
        )
        raise ValueError(message)
    queue_path = root / "queue_manifest.jsonl"
    contract_path = root / "contract.json"
    source_path = root / "source_manifest.json"
    for path in (queue_path, contract_path, source_path):
        if not path.is_file():
            message = f"missing frozen v2 campaign artifact: {path}"
            raise FileNotFoundError(message)
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        source_manifest = json.loads(source_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        message = f"invalid frozen campaign JSON: {error}"
        raise ValueError(message) from error
    if not isinstance(contract, dict) or contract.get("schema") != f"{SCHEMA}.contract":
        message = "contract schema is not the frozen theory-bridge v2 schema"
        raise ValueError(message)
    if (
        not isinstance(source_manifest, dict)
        or source_manifest.get("schema") != f"{SCHEMA}.source_manifest"
        or source_manifest.get("captured_before_first_gpu_job") is not True
    ):
        message = "source manifest schema/state is invalid"
        raise ValueError(message)
    frozen_source_hashes = source_manifest.get("source_sha256")
    if (
        not isinstance(frozen_source_hashes, dict)
        or set(frozen_source_hashes) != set(SOURCE_FILES)
        or any(
            not isinstance(value, str) or len(value) != 64
            for value in frozen_source_hashes.values()
        )
    ):
        message = "source manifest does not seal the exact v2 source set"
        raise ValueError(message)
    source_snapshot_sha256 = _sha256_path(source_path)
    queue_manifest_sha256 = _sha256_path(queue_path)
    if contract.get("source_snapshot_sha256") != source_snapshot_sha256:
        message = "contract/source-manifest hash mismatch"
        raise ValueError(message)
    if contract.get("queue_manifest_sha256") != queue_manifest_sha256:
        message = "contract/queue-manifest hash mismatch"
        raise ValueError(message)
    if contract.get("source_sha256") != frozen_source_hashes:
        message = "contract/source hash mapping mismatch"
        raise ValueError(message)
    canonical = json.dumps(
        frozen_source_hashes,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    if source_manifest.get("snapshot_payload_sha256") != hashlib.sha256(canonical).hexdigest():
        message = "source snapshot payload digest is invalid"
        raise ValueError(message)
    current_hashes = _source_hashes()
    if current_hashes != frozen_source_hashes:
        message = "current source differs from the frozen pre-GPU source snapshot"
        raise ValueError(message)
    active = _load_jobs(root)
    datasets = contract.get("datasets")
    seeds = contract.get("seeds")
    if (
        not isinstance(datasets, list)
        or not all(isinstance(value, str) for value in datasets)
        or not isinstance(seeds, list)
        or not all(type(value) is int for value in seeds)
    ):
        message = "contract dataset/seed grid is invalid"
        raise ValueError(message)
    expected_from_contract = jobs(tuple(datasets), tuple(seeds))
    if [job.key for job in active] != [job.key for job in expected_from_contract]:
        message = "queue keys/order do not equal the contract dataset-by-seed grid"
        raise ValueError(message)
    if contract.get("jobs") != len(active):
        message = "contract job count does not match the frozen queue"
        raise ValueError(message)
    return {
        "contract": contract,
        "jobs": active,
        "expected_keys": {job.key for job in active},
        "job_by_key": {job.key: job for job in active},
        "queue_manifest_sha256": queue_manifest_sha256,
        "contract_sha256": _sha256_path(contract_path),
        "source_snapshot_sha256": source_snapshot_sha256,
        "source_sha256": frozen_source_hashes,
    }


def _validated_result_rows(  # noqa: C901, PLR0912
    root: Path,
    context: Mapping[str, Any],
    *,
    kind: Literal["completed", "failed"],
) -> list[dict[str, Any]]:
    """Load terminal rows only after queue/schema/status/provenance validation."""
    expected_keys = cast("set[str]", context["expected_keys"])
    job_by_key = cast("Mapping[str, TheoryBridgeJob]", context["job_by_key"])
    expected_schema = f"{SCHEMA}.{'result' if kind == 'completed' else 'failure'}"
    expected_status = "done" if kind == "completed" else "failed"
    directory = root / kind
    if not directory.exists():
        return []
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix != ".json":
            message = f"unrecognized terminal artifact in {kind}: {path.name}"
            raise ValueError(message)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            message = f"invalid terminal row JSON: {path}"
            raise ValueError(message) from error
        if not isinstance(payload, dict):
            message = f"terminal row is not an object: {path}"
            raise TypeError(message)
        job_key = payload.get("job_key")
        if not isinstance(job_key, str) or job_key not in expected_keys:
            message = f"terminal row is unrelated to the frozen queue: {path}"
            raise ValueError(message)
        if job_key in seen:
            message = f"duplicate terminal row for frozen queue key: {job_key}"
            raise ValueError(message)
        if path.name != f"{_safe_key(job_key)}.json":
            message = f"terminal row filename/key mismatch: {path}"
            raise ValueError(message)
        job = job_by_key[job_key]
        if payload.get("schema") != expected_schema or payload.get("status") != expected_status:
            message = f"terminal row schema/status mismatch: {path}"
            raise ValueError(message)
        if payload.get("job") not in (None, asdict(job)):
            message = f"terminal row job payload mismatch: {path}"
            raise ValueError(message)
        if kind == "completed" and (
            payload.get("dataset") != job.dataset
            or payload.get("seed") != job.seed
            or payload.get("official_test_accessed") is not True
            or not isinstance(payload.get("primary_bridge"), dict)
            or not isinstance(payload.get("secondary_joint_bridge"), dict)
            or not isinstance(payload.get("conditions"), dict)
        ):
            message = f"completed row payload does not match its frozen job: {path}"
            raise ValueError(message)
        if (
            payload.get("queue_manifest_sha256") != context["queue_manifest_sha256"]
            or payload.get("contract_sha256") != context["contract_sha256"]
            or payload.get("source_snapshot_sha256")
            != context["source_snapshot_sha256"]
            or payload.get("source_sha256") != context["source_sha256"]
        ):
            message = f"terminal row provenance does not match the frozen campaign: {path}"
            raise ValueError(message)
        seen.add(job_key)
        rows.append(payload)
    return rows


def _reject_overlapping_terminal_rows(
    completed_keys: set[str],
    failed_keys: set[str],
) -> None:
    overlap = completed_keys & failed_keys
    if overlap:
        message = (
            "the same frozen queue keys appear in completed and failed: "
            + ", ".join(sorted(overlap))
        )
        raise ValueError(message)


def _safe_key(key: str) -> str:
    return key.replace(":", "__").replace("/", "_")


def _result_path(root: Path, job: TheoryBridgeJob) -> Path:
    return root / "completed" / f"{_safe_key(job.key)}.json"


def _failure_path(root: Path, job: TheoryBridgeJob) -> Path:
    return root / "failed" / f"{_safe_key(job.key)}.json"


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_immutable_json(path: Path, payload: Mapping[str, object]) -> None:
    _write_immutable_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def _write_immutable_text(path: Path, value: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != value:
            message = f"refusing to mutate frozen campaign artifact: {path}"
            raise ValueError(message)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _write_per_run_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "dataset,seed,empirical_hat_delta_m,hat_delta_se_snr,"
        "prototype_test_ba,ce_test_ba,"
        "prototype_minus_ce_ba"
    ]
    for row in rows:
        primary = row["primary_bridge"]
        lines.append(
            ",".join(
                (
                    str(row["dataset"]),
                    str(row["seed"]),
                    str(primary["empirical_hat_delta_m"]),
                    str(primary["hat_delta_se_snr"]),
                    str(primary["prototype_test_balanced_accuracy"]),
                    str(primary["ce_test_balanced_accuracy"]),
                    str(primary["prototype_minus_ce_balanced_accuracy"]),
                )
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    primary = payload["primary"]
    correlations = payload["correlations"]
    random_contrast = payload["paired_primary_contrasts"]["random_median"][
        "empirical_hat_delta_m"
    ]
    gap_interval = primary["prototype_minus_ce_balanced_accuracy"][
        "paired_dataset_cluster_bootstrap_interval_95"
    ]
    lines = [
        "# P0-1 / P1-2 theory bridge campaign",
        "",
        f"Completed raw rows: {payload['rows']}.",
        f"Decision state: {payload['evaluation_state']}. Pass/fail is withheld until the "
        "complete frozen queue is present.",
        "",
        "Primary estimand: theorem-aligned learned writer-energy prototype with "
        "M=32 and the full independent calibration pool. Each prototype is "
        "log1p(class mean raw energy), not the mean of logged path energies.",
        "",
        "The joint 2M writer+reader descriptor is secondary.",
        "",
        f"- Mean empirical hat-delta: {primary['empirical_hat_delta_m']['mean']}",
        f"- Mean prototype-minus-CE balanced accuracy: "
        f"{primary['prototype_minus_ce_balanced_accuracy']['mean']}",
        f"- Paired prototype-minus-CE 95% dataset-cluster bootstrap interval: "
        f"{gap_interval}",
        f"- Minimum-SE-SNR > 3: {primary['se_snr_gt_3']['count']}/"
        f"{primary['se_snr_gt_3']['total']}",
        f"- Prototype no more than 2 pp below CE: "
        f"{primary['prototype_no_more_than_2pp_below_ce']['count']}/"
        f"{primary['prototype_no_more_than_2pp_below_ce']['total']}",
        f"- Empirical hat-delta/prototype-TEST Pearson: "
        f"{correlations['hat_delta_vs_prototype_test_ba']['overall']['pearson_r']}",
        f"- Empirical hat-delta/prototype-TEST Spearman: "
        f"{correlations['hat_delta_vs_prototype_test_ba']['overall']['spearman_rho']}",
        f"- Learned-minus-random-median empirical hat-delta paired 95% interval: "
        f"{random_contrast['paired_dataset_cluster_bootstrap_interval_95']}",
        "",
        "Primary and secondary full-calibration M=32 hat-delta uncertainty uses "
        f"{CLASS_BOOTSTRAP_REPLICATES} independent within-class bootstrap resamples. "
        "Random-best and random-median are representation-training-label-selected "
        "conditions, not unselected iid draws.",
        "",
        f"Nested-prefix limitation: {payload['nested_prefix_protocol_limitation']}",
        "",
        "This report is an experiment artifact only; it does not modify or pre-apply claims "
        "to the paper.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


__all__ = [
    "CLASS_BOOTSTRAP_REPLICATES",
    "DEFAULT_DATASETS",
    "DEFAULT_ROOT",
    "DEFAULT_SEEDS",
    "MODE_PREFIXES",
    "RANDOM_DRAWS",
    "SCHEMA",
    "SUPERSEDED_ROOT",
    "BankState",
    "BlockBankState",
    "TheoryBridgeJob",
    "ThreeWaySplit",
    "affine_prototype_logits",
    "affine_prototype_parameters",
    "calibration_subset_indices",
    "capture_bank_state",
    "evaluate_prototype_head",
    "fit_prototype_head",
    "jobs",
    "negative_squared_distance_logits",
    "prefix_energy_view",
    "prepare",
    "random_orthonormal_frame",
    "randomized_bank_state",
    "raw_energy_class_prototypes",
    "report",
    "run",
    "run_job",
    "status",
    "stratified_three_way_indices",
    "temporary_bank_state",
]
