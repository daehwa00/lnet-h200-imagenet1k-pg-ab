"""Resumable synthetic controls for temporal second- and higher-order claims.

This campaign is deliberately separate from the paper sources.  It contains:

* P1-1: matched-marginal AR(1) controls evaluated with a raw fixed bank, a
  frozen orthogonal encoder, and a learned BenchmarkAlphabetBackbone representation.
* P1-3: equal-second-order Gaussian- versus Rademacher-innovation processes
  evaluated with a controlled linear writer and six reader/readout variants.

Every job writes one immutable raw JSON row.  Reports aggregate those rows
without hiding failed criteria, and manifests make multi-GPU execution
restart-safe.
"""

# pyright: reportExplicitAny=false, reportImplicitStringConcatenation=false
# pyright: reportPrivateUsage=false

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, stdev
from time import perf_counter
from typing import TYPE_CHECKING, Any, Final, Literal, Protocol, cast

import torch
from torch import Tensor, nn
from torch.nn import functional

from .pac_campaign_utils import seed_everything, source_file_hashes, write_once

if TYPE_CHECKING:
    from collections.abc import Mapping

DEFAULT_ROOT: Final = Path(".omx/results/pac-temporal-controls-local_gpu-20260724")
DEFAULT_SEEDS: Final = (7, 11, 19, 23, 31, 43, 47, 59, 61, 67)
AR1_CONTROLS: Final = (
    "original",
    "global_time_shuffle",
    "timestamp_rearrangement",
    "marginal_autocorrelation_destroyed",
)
AR1_VIEWS: Final = ("raw_fixed_bank", "frozen_orthogonal", "learned_identity")
HIGHER_ORDER_VARIANTS: Final = (
    "writer_energy_only",
    "writer_energy_lag",
    "one_scan_full",
    "full_writer_reader",
    "mlp_reader",
    "conv_reader",
)
NONLINEAR_READER_VARIANTS: Final = (
    "full_writer_reader",
    "mlp_reader",
    "conv_reader",
)

AR1_PHI: Final = 0.8
AR1_LENGTH: Final = 128
AR1_TRAIN_COUNT: Final = 512
AR1_VALIDATION_COUNT: Final = 256
AR1_TEST_COUNT: Final = 256
AR1_EPOCHS: Final = 60

HIGHER_ORDER_PHI: Final = 0.6
HIGHER_ORDER_LENGTH: Final = 192
HIGHER_ORDER_TRAIN_COUNT: Final = 768
HIGHER_ORDER_VALIDATION_COUNT: Final = 256
HIGHER_ORDER_TEST_COUNT: Final = 512
HIGHER_ORDER_EPOCHS: Final = 60

FIXED_MODES: Final = 8
ORTHOGONAL_DIM: Final = 8
IDENTITY_DIM: Final = 32
HIGHER_ORDER_PARAMETER_TARGET: Final = 8192
MOMENT_LAGS: Final = (1, 2, 4)
SPECTRUM_PERMUTATIONS: Final = 199
SECOND_ORDER_BOOTSTRAPS: Final = 399
SECOND_ORDER_CONFIDENCE_LEVEL: Final = 0.95
AUTOCOVARIANCE_EQUIVALENCE_TOLERANCE: Final = 0.05
SPECTRUM_EQUIVALENCE_TOLERANCE: Final = 0.10

Family = Literal["p1_1_ar1", "p1_3_higher_order"]
AR1Control = Literal[
    "original",
    "global_time_shuffle",
    "timestamp_rearrangement",
    "marginal_autocorrelation_destroyed",
]
AR1View = Literal["raw_fixed_bank", "frozen_orthogonal", "learned_identity"]
HigherOrderVariant = Literal[
    "writer_energy_only",
    "writer_energy_lag",
    "one_scan_full",
    "full_writer_reader",
    "mlp_reader",
    "conv_reader",
]


class _IdentityFeatureModel(Protocol):
    modes: int

    def _edge_stem(
        self,
        inputs: Tensor,
        time_delta: Tensor | None,
        observation_mask: Tensor | None,
        valid_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor | None]: ...

    def _writer(
        self,
        first_local: Tensor,
        active_delta: Tensor | None,
        active_observation: Tensor | None,
        active_valid: Tensor | None,
    ) -> tuple[Tensor, Tensor]: ...


@dataclass(frozen=True, slots=True)
class TemporalControlJob:
    family: Family
    seed: int

    @property
    def key(self) -> str:
        return f"{self.family}__seed{self.seed}"


def campaign_jobs(seeds: tuple[int, ...] = DEFAULT_SEEDS) -> list[TemporalControlJob]:
    """Return the complete prospectively fixed two-family grid."""
    return [
        TemporalControlJob(family, seed)
        for family in ("p1_1_ar1", "p1_3_higher_order")
        for seed in seeds
    ]


def prepare_campaign(
    root: Path = DEFAULT_ROOT,
    *,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    lane_count: int = 2,
) -> dict[str, object]:
    """Freeze the contract and write disjoint restart-safe worker manifests."""
    if not seeds:
        message = "at least one seed is required"
        raise ValueError(message)
    if lane_count < 1:
        message = "lane_count must be positive"
        raise ValueError(message)
    active = campaign_jobs(seeds)
    root.mkdir(parents=True, exist_ok=True)
    for name in ("completed", "failed", "attempts", "manifests", "reports"):
        (root / name).mkdir(exist_ok=True)

    queue_text = "".join(json.dumps(asdict(job), sort_keys=True) + "\n" for job in active)
    write_once(root / "queue.jsonl", queue_text)
    for lane in range(lane_count):
        assigned = active[lane::lane_count]
        manifest = "".join(json.dumps(asdict(job), sort_keys=True) + "\n" for job in assigned)
        write_once(root / "manifests" / f"worker-{lane:02d}.jsonl", manifest)

    contract: dict[str, object] = {
        "schema": "pac_temporal_controls_contract.v1",
        "paper_sources_modified": False,
        "hardware_target": "local_gpu CUDA GPU",
        "seeds": list(seeds),
        "seed_count": len(seeds),
        "jobs": len(active),
        "lane_count": lane_count,
        "p1_1": {
            "process": (
                "stationary Gaussian AR(1), phi in {+0.8,-0.8}, "
                "equal population mean zero and variance one"
            ),
            "counts": {
                "train": AR1_TRAIN_COUNT,
                "validation": AR1_VALIDATION_COUNT,
                "test": AR1_TEST_COUNT,
                "length": AR1_LENGTH,
            },
            "views": list(AR1_VIEWS),
            "controls": list(AR1_CONTROLS),
            "estimands": [
                "pole-input class mean",
                "pole-input Gamma(0)",
                "pole-input nonzero-lag covariance",
                "standardized mean and covariance differences",
                "energy nearest-prototype balanced accuracy",
            ],
            "criteria": {
                "original_np_ba_strictly_above": 0.95,
                "destroyed_np_ba_interval": [0.47, 0.53],
                "per_view_original_standardized_pole_input_mean_difference_max": (0.10),
                "per_view_original_standardized_pole_input_gamma0_difference_max": (0.10),
                "nonzero_lag_difference": (
                    "reported as the intended signal; no small-difference gate"
                ),
            },
        },
        "p1_3": {
            "process": (
                "common AR(1) filter with phi=0.6; class 0 unit Gaussian "
                "innovations and class 1 unit Rademacher innovations"
            ),
            "counts": {
                "train": HIGHER_ORDER_TRAIN_COUNT,
                "validation": HIGHER_ORDER_VALIDATION_COUNT,
                "test": HIGHER_ORDER_TEST_COUNT,
                "length": HIGHER_ORDER_LENGTH,
            },
            "controlled_writer": "fixed linear complex pole writer",
            "writer_log_energy": "log1p(mean_t(real_state^2 + imag_state^2))",
            "variants": list(HIGHER_ORDER_VARIANTS),
            "parameter_target": HIGHER_ORDER_PARAMETER_TARGET,
            "parameter_tolerance": 0.03,
            "criteria": {
                "second_order_equivalence": {
                    "bootstrap_count": SECOND_ORDER_BOOTSTRAPS,
                    "confidence_level": SECOND_ORDER_CONFIDENCE_LEVEL,
                    "autocovariance_absolute_tolerance": (AUTOCOVARIANCE_EQUIVALENCE_TOLERANCE),
                    "relative_mean_spectrum_tolerance": (SPECTRUM_EQUIVALENCE_TOLERANCE),
                    "decision_rule": (
                        "every lagwise signed-autocovariance CI must be "
                        "contained in +/- the autocovariance tolerance and "
                        "the relative mean-spectrum-difference CI upper bound "
                        "must not exceed its tolerance"
                    ),
                },
                "exploratory_spectrum_difference_test_alpha": 0.05,
                "difference_test_non_rejection_is_not_equivalence": True,
                "one_scan_full_balanced_accuracy_interval": [0.47, 0.53],
                "full_minus_one_scan_minimum": 0.10,
                "full_minus_best_matched_nonlinear_strictly_above": 0.0,
            },
        },
        "source_sha256": _source_hashes(),
    }
    write_once(
        root / "contract.json",
        json.dumps(contract, indent=2, sort_keys=True) + "\n",
    )
    return contract


def campaign_status(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    queue = _read_jobs(root / "queue.jsonl")
    expected_keys = {job.key for job in queue}
    completed_keys = {
        path.stem for path in (root / "completed").glob("*.json") if path.stem in expected_keys
    }
    failed_keys = {
        path.stem for path in (root / "failed").glob("*.json") if path.stem in expected_keys
    }
    return {
        "expected": len(expected_keys),
        "completed": len(completed_keys),
        "failed": len(failed_keys),
        "remaining": len(expected_keys - completed_keys),
        "done": completed_keys == expected_keys and not failed_keys,
    }


def run_manifest(
    root: Path,
    manifest: Path,
    *,
    device: str = "cuda",
) -> dict[str, object]:
    """Run one disjoint manifest, preserving failures and skipping completed rows."""
    if device == "cuda" and not torch.cuda.is_available():
        message = "CUDA was requested but is unavailable"
        raise RuntimeError(message)
    jobs = _read_jobs(manifest)
    for job in jobs:
        output = root / "completed" / f"{job.key}.json"
        if output.exists():
            continue
        attempt_dir = root / "attempts" / job.key
        attempt_dir.mkdir(parents=True, exist_ok=True)
        attempt_id = f"{os.getpid()}-{int(perf_counter() * 1_000_000)}"
        _write_json(
            attempt_dir / f"{attempt_id}.started.json",
            {
                "schema": "pac_temporal_controls_attempt.v1",
                "job": asdict(job),
                "status": "started",
                "pid": os.getpid(),
                "device": device,
            },
        )
        started = perf_counter()
        try:
            row = run_job(job, device=device)
            row["elapsed_seconds"] = perf_counter() - started
            _write_json(output, row)
            _write_json(
                attempt_dir / f"{attempt_id}.succeeded.json",
                {
                    "schema": "pac_temporal_controls_attempt.v1",
                    "job_key": job.key,
                    "status": "succeeded",
                    "elapsed_seconds": row["elapsed_seconds"],
                },
            )
            failure_path = root / "failed" / f"{job.key}.json"
            if failure_path.exists():
                failure_path.unlink()
        except Exception as error:  # noqa: BLE001
            failure = {
                "schema": "pac_temporal_controls_failure.v1",
                "job": asdict(job),
                "job_key": job.key,
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "elapsed_seconds": perf_counter() - started,
            }
            _write_json(root / "failed" / f"{job.key}.json", failure)
            _write_json(
                attempt_dir / f"{attempt_id}.failed.json",
                failure,
            )
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return campaign_status(root)


def run_job(job: TemporalControlJob, *, device: str) -> dict[str, object]:
    seed_everything(job.seed)
    if job.family == "p1_1_ar1":
        row = run_ar1_control_job(job.seed, device=device)
    elif job.family == "p1_3_higher_order":
        row = run_higher_order_job(job.seed, device=device)
    else:
        message = f"unsupported temporal-control family: {job.family}"
        raise ValueError(message)
    row.update(
        {
            "job_key": job.key,
            "family": job.family,
            "seed": job.seed,
            "status": "done",
            "device": device,
            "torch_version": torch.__version__,
            "cuda_device": (
                torch.cuda.get_device_name(torch.cuda.current_device())
                if device == "cuda"
                else None
            ),
        }
    )
    return row


def matched_ar1_samples(
    count: int,
    length: int,
    *,
    seed: int,
) -> tuple[Tensor, Tensor]:
    """Draw balanced stationary Gaussian AR(1) paths with opposite correlation."""
    if count % 2:
        message = "matched AR(1) count must be even"
        raise ValueError(message)
    generator = torch.Generator().manual_seed(seed)
    labels = torch.arange(count, dtype=torch.long) % 2
    labels = labels[torch.randperm(count, generator=generator)]
    values = torch.empty(count, length)
    values[:, 0] = torch.randn(count, generator=generator)
    innovations = torch.randn(count, length - 1, generator=generator)
    phi = torch.where(labels == 0, AR1_PHI, -AR1_PHI).to(values.dtype)
    innovation_scale = math.sqrt(1.0 - AR1_PHI**2)
    for index in range(1, length):
        values[:, index] = phi * values[:, index - 1] + innovation_scale * innovations[:, index - 1]
    return values.unsqueeze(-1), labels


def higher_order_samples(
    count: int,
    length: int,
    *,
    seed: int,
    phi: float = HIGHER_ORDER_PHI,
    burn_in: int = 192,
) -> tuple[Tensor, Tensor]:
    """Draw equal-second-order AR paths with Gaussian/Rademacher innovations."""
    if count % 2:
        message = "higher-order sample count must be even"
        raise ValueError(message)
    if not 0.0 <= abs(phi) < 1.0:
        message = "phi must define a stable AR(1) process"
        raise ValueError(message)
    generator = torch.Generator().manual_seed(seed)
    labels = torch.arange(count, dtype=torch.long) % 2
    labels = labels[torch.randperm(count, generator=generator)]
    total = length + burn_in
    gaussian = torch.randn(count, total, generator=generator)
    rademacher = (
        torch.randint(0, 2, (count, total), generator=generator, dtype=torch.int64)
        .mul(2)
        .sub(1)
        .to(torch.float32)
    )
    innovations = torch.where(labels[:, None] == 0, gaussian, rademacher)
    values = torch.zeros(count, total)
    scale = math.sqrt(1.0 - phi**2)
    for index in range(1, total):
        values[:, index] = phi * values[:, index - 1] + scale * innovations[:, index]
    values = values[:, burn_in:]
    # Enforce identical empirical first two marginal moments per split.  This
    # leaves the class-specific innovation kurtosis and temporal higher-order law.
    for label in (0, 1):
        selected = labels == label
        active = values[selected]
        values[selected] = (active - active.mean()) / active.std(unbiased=False).clamp_min(1e-8)
    return values.unsqueeze(-1), labels


def apply_ar1_control(
    values: Tensor,
    labels: Tensor,
    control: AR1Control,
    *,
    seed: int,
) -> Tensor:
    """Apply a label-blind or label-stratified marginal-preserving control."""
    if control == "original":
        return values.clone()
    generator = torch.Generator().manual_seed(seed)
    output = values.clone()
    count, length, _ = values.shape
    if control == "global_time_shuffle":
        permutation = torch.randperm(length, generator=generator)
        return values[:, permutation].clone()
    if control == "timestamp_rearrangement":
        for label in torch.unique(labels).tolist():
            indices = torch.nonzero(labels == int(label), as_tuple=False).flatten()
            for timestamp in range(length):
                permutation = torch.randperm(indices.numel(), generator=generator)
                output[indices, timestamp] = values[indices[permutation], timestamp]
        return output
    if control == "marginal_autocorrelation_destroyed":
        for sample in range(count):
            permutation = torch.randperm(length, generator=generator)
            output[sample] = values[sample, permutation]
        return output
    message = f"unknown AR(1) control: {control}"
    raise ValueError(message)


class FixedComplexPoleBank(nn.Module):
    """A data-independent stable complex bank with fixed mode directions."""

    def __init__(self, directions: Tensor) -> None:
        super().__init__()
        if directions.ndim != 2:
            message = "directions must have shape [M,D]"
            raise ValueError(message)
        modes = directions.shape[0]
        normalized = directions / directions.norm(dim=1, keepdim=True).clamp_min(1e-8)
        damping = torch.logspace(
            math.log10(0.04),
            math.log10(1.6),
            modes,
        )
        frequency = torch.linspace(0.0, math.pi, modes)
        radius = torch.exp(-damping)
        self.directions: Tensor
        self.decay_real: Tensor
        self.decay_imag: Tensor
        self.input_gain: Tensor
        self.register_buffer("directions", normalized)
        self.register_buffer("decay_real", radius * torch.cos(frequency))
        self.register_buffer("decay_imag", radius * torch.sin(frequency))
        self.register_buffer("input_gain", torch.sqrt((1.0 - radius.square()).clamp_min(1e-6)))

    @property
    def modes(self) -> int:
        return int(self.directions.shape[0])

    def forward(self, values: Tensor) -> tuple[Tensor, Tensor]:
        if values.shape[-1] != self.directions.shape[-1]:
            message = "pole input dimension does not match fixed directions"
            raise ValueError(message)
        drive = torch.einsum("btd,md->btm", values, self.directions)
        decay_real = self.decay_real.to(dtype=values.dtype)
        decay_imag = self.decay_imag.to(dtype=values.dtype)
        gain = self.input_gain.to(dtype=values.dtype)
        state_real = values.new_zeros(values.shape[0], self.modes)
        state_imag = values.new_zeros(values.shape[0], self.modes)
        real_path: list[Tensor] = []
        imag_path: list[Tensor] = []
        for timestamp in range(values.shape[1]):
            previous_real = state_real
            previous_imag = state_imag
            state_real = (
                decay_real * previous_real - decay_imag * previous_imag + gain * drive[:, timestamp]
            )
            state_imag = decay_imag * previous_real + decay_real * previous_imag
            real_path.append(state_real)
            imag_path.append(state_imag)
        return torch.stack(real_path, dim=1), torch.stack(imag_path, dim=1)


def complex_modal_moments(
    real: Tensor,
    imag: Tensor,
    *,
    lags: tuple[int, ...] = MOMENT_LAGS,
) -> Tensor:
    """Return M log1p-energies plus normalized complex correlations per lag."""
    energy = (real.square() + imag.square()).mean(dim=1)
    chunks = [torch.log1p(energy)]
    for lag in lags:
        if real.shape[1] <= lag:
            chunks.extend((torch.zeros_like(energy), torch.zeros_like(energy)))
            continue
        current_real = real[:, lag:]
        current_imag = imag[:, lag:]
        prior_real = real[:, :-lag]
        prior_imag = imag[:, :-lag]
        numerator_real = (current_real * prior_real + current_imag * prior_imag).mean(dim=1)
        numerator_imag = (current_imag * prior_real - current_real * prior_imag).mean(dim=1)
        current_energy = (current_real.square() + current_imag.square()).mean(dim=1)
        prior_energy = (prior_real.square() + prior_imag.square()).mean(dim=1)
        denominator = torch.sqrt((current_energy * prior_energy).clamp_min(1e-8))
        chunks.extend((numerator_real / denominator, numerator_imag / denominator))
    return torch.cat(chunks, dim=-1)


def temporal_statistics(
    values: Tensor,
    labels: Tensor,
    *,
    lags: tuple[int, ...] = MOMENT_LAGS,
) -> dict[str, object]:
    """Measure class means, Gamma(0), nonzero-lag covariance, and effects."""
    classes = sorted(int(value) for value in torch.unique(labels).tolist())
    if classes != [0, 1]:
        message = "temporal diagnostics require binary labels {0,1}"
        raise ValueError(message)
    means: list[Tensor] = []
    gamma_zero: list[Tensor] = []
    gamma_lags: dict[int, list[Tensor]] = {lag: [] for lag in lags}
    for label in classes:
        active = values[labels == label].to(torch.float64)
        active_mean = active.mean(dim=(0, 1))
        centered = active - active_mean
        means.append(active_mean)
        gamma_zero.append(
            torch.einsum("btd,bte->de", centered, centered) / centered.numel() * centered.shape[-1]
        )
        for lag in lags:
            covariance = torch.einsum(
                "btd,bte->de",
                centered[:, lag:],
                centered[:, :-lag],
            ) / max(centered.shape[0] * (centered.shape[1] - lag), 1)
            gamma_lags[lag].append(covariance)

    pooled_scale = math.sqrt(
        0.5 * float(torch.trace(gamma_zero[0]) + torch.trace(gamma_zero[1])) + 1e-12
    )
    mean_standardized = float(torch.linalg.vector_norm(means[0] - means[1])) / pooled_scale
    gamma_zero_scale = (
        0.5
        * float(torch.linalg.matrix_norm(gamma_zero[0]) + torch.linalg.matrix_norm(gamma_zero[1]))
        + 1e-12
    )
    gamma_zero_standardized = (
        float(torch.linalg.matrix_norm(gamma_zero[0] - gamma_zero[1])) / gamma_zero_scale
    )
    lag_effects = {
        str(lag): (
            float(torch.linalg.matrix_norm(gamma_lags[lag][0] - gamma_lags[lag][1]))
            / (
                0.5
                * float(
                    torch.linalg.matrix_norm(gamma_lags[lag][0])
                    + torch.linalg.matrix_norm(gamma_lags[lag][1])
                )
                + 1e-12
            )
        )
        for lag in lags
    }
    return {
        "class_mean": {str(label): means[index].tolist() for index, label in enumerate(classes)},
        "gamma0": {str(label): gamma_zero[index].tolist() for index, label in enumerate(classes)},
        "gamma_nonzero": {
            str(lag): {
                str(label): gamma_lags[lag][index].tolist() for index, label in enumerate(classes)
            }
            for lag in lags
        },
        "standardized_mean_difference": mean_standardized,
        "standardized_gamma0_difference": gamma_zero_standardized,
        "standardized_nonzero_lag_difference": lag_effects,
    }


def prototype_metrics(
    calibration_features: Tensor,
    calibration_labels: Tensor,
    test_features: Tensor,
    test_labels: Tensor,
) -> dict[str, float]:
    classes = sorted(int(value) for value in torch.unique(calibration_labels).tolist())
    prototypes = torch.stack(
        [calibration_features[calibration_labels == label].mean(dim=0) for label in classes]
    ).to(torch.float64)
    distances = torch.cdist(test_features.to(torch.float64), prototypes)
    predicted = torch.tensor(classes, dtype=test_labels.dtype)[distances.argmin(dim=1)]
    recalls = [
        float((predicted[test_labels == label] == label).to(torch.float64).mean())
        for label in classes
    ]
    margin = min(
        float(torch.linalg.vector_norm(prototypes[left] - prototypes[right]))
        for left in range(len(classes))
        for right in range(left + 1, len(classes))
    )
    return {
        "accuracy": float((predicted == test_labels).to(torch.float64).mean()),
        "balanced_accuracy": mean(recalls),
        "minimum_prototype_distance": margin,
    }


def run_ar1_control_job(seed: int, *, device: str) -> dict[str, object]:
    train_inputs, train_labels = matched_ar1_samples(
        AR1_TRAIN_COUNT,
        AR1_LENGTH,
        seed=seed * 101 + 1,
    )
    validation_inputs, validation_labels = matched_ar1_samples(
        AR1_VALIDATION_COUNT,
        AR1_LENGTH,
        seed=seed * 101 + 2,
    )
    test_inputs, test_labels = matched_ar1_samples(
        AR1_TEST_COUNT,
        AR1_LENGTH,
        seed=seed * 101 + 3,
    )

    identity_model, training = _train_identity_model(
        train_inputs,
        train_labels,
        validation_inputs,
        validation_labels,
        seed=seed,
        device=device,
    )
    evaluations: dict[str, dict[str, object]] = {view: {} for view in AR1_VIEWS}
    for control_index, control in enumerate(AR1_CONTROLS):
        transformed_train = apply_ar1_control(
            train_inputs,
            train_labels,
            control,
            seed=seed * 1009 + control_index * 17 + 1,
        )
        transformed_test = apply_ar1_control(
            test_inputs,
            test_labels,
            control,
            seed=seed * 1009 + control_index * 17 + 2,
        )
        for view_index, view in enumerate(AR1_VIEWS):
            if view == "learned_identity":
                train_pole_input, train_energy = _identity_pole_features(
                    identity_model,
                    transformed_train,
                    device=device,
                )
                _, test_energy = _identity_pole_features(
                    identity_model,
                    transformed_test,
                    device=device,
                )
            else:
                train_pole_input, train_energy = _fixed_ar1_view(
                    transformed_train,
                    view=view,
                    seed=seed * 409 + view_index,
                    device=device,
                )
                _, test_energy = _fixed_ar1_view(
                    transformed_test,
                    view=view,
                    seed=seed * 409 + view_index,
                    device=device,
                )
            evaluations[view][control] = {
                "pole_input": temporal_statistics(
                    train_pole_input,
                    train_labels,
                ),
                "energy_nearest_prototype": prototype_metrics(
                    train_energy,
                    train_labels,
                    test_energy,
                    test_labels,
                ),
            }
    return {
        "schema": "pac_temporal_controls_ar1_result.v1",
        "process": {
            "phi": [AR1_PHI, -AR1_PHI],
            "population_mean": [0.0, 0.0],
            "population_variance": [1.0, 1.0],
            "length": AR1_LENGTH,
            "train_count": AR1_TRAIN_COUNT,
            "validation_count": AR1_VALIDATION_COUNT,
            "test_count": AR1_TEST_COUNT,
        },
        "identity_training": training,
        "evaluations": evaluations,
    }


def _fixed_ar1_view(
    inputs: Tensor,
    *,
    view: AR1View,
    seed: int,
    device: str,
) -> tuple[Tensor, Tensor]:
    if view == "raw_fixed_bank":
        pole_input = inputs
        directions = torch.ones(FIXED_MODES, 1)
    elif view == "frozen_orthogonal":
        generator = torch.Generator().manual_seed(seed)
        encoder = torch.randn(ORTHOGONAL_DIM, 1, generator=generator)
        encoder = torch.linalg.qr(encoder, mode="reduced").Q
        pole_input = torch.einsum("btc,dc->btd", inputs, encoder)
        directions = torch.linalg.qr(
            torch.randn(ORTHOGONAL_DIM, ORTHOGONAL_DIM, generator=generator)
        ).Q.T[:FIXED_MODES]
    else:
        message = f"fixed AR view does not support {view}"
        raise ValueError(message)
    bank = FixedComplexPoleBank(directions).to(device)
    with torch.no_grad():
        real, imag = bank(pole_input.to(device))
        energy = complex_modal_moments(real, imag)[:, :FIXED_MODES]
    return pole_input.cpu(), energy.cpu()


def _train_identity_model(
    train_inputs: Tensor,
    train_labels: Tensor,
    validation_inputs: Tensor,
    validation_labels: Tensor,
    *,
    seed: int,
    device: str,
) -> tuple[nn.Module, dict[str, object]]:
    from .alphabet_backbone import AlphabetBackbone  # noqa: PLC0415
    from .pac_types import PACDevice, PACExperimentConfig  # noqa: PLC0415

    config = PACExperimentConfig(
        train_inputs.shape[0],
        validation_inputs.shape[0],
        0,
        train_inputs.shape[1],
        raw_input_dim=1,
        output_dim=2,
        model_dim=IDENTITY_DIM,
        modes=FIXED_MODES,
        epochs=AR1_EPOCHS,
        batch_size=64,
        learning_rate=3e-3,
        weight_decay=1e-4,
        grad_clip_norm=1.0,
        seeds=(seed,),
        device=cast("PACDevice", device),
        optimizer_mode="default",
    )
    model = AlphabetBackbone(config, 2, objective="classification").to(device)
    for name in (
        "use_efp16_exact_split_training",
        "require_external_exact_split_training",
        "use_fused_efp16_inference_readout",
        "use_fused_rmsnorm_mean_training",
        "use_fused_rmsnorm_mean_backward_training",
        "use_d32_rmsnorm_backward_training",
        "use_fused_terminal_reader_local_training",
        "use_fused_terminal_reader_scan_training",
        "use_fused_writer_reader_local_training",
        "use_fused_writer_modal_reader_local_training",
    ):
        if hasattr(model, name):
            setattr(model, name, False)
    training = fit_classifier(
        model,
        train_inputs,
        train_labels,
        validation_inputs,
        validation_labels,
        seed=seed,
        device=device,
        epochs=AR1_EPOCHS,
        batch_size=64,
        learning_rate=3e-3,
        weight_decay=1e-4,
    )
    return model, training


@torch.no_grad()
def _identity_pole_features(
    model: nn.Module,
    inputs: Tensor,
    *,
    device: str,
    batch_size: int = 128,
) -> tuple[Tensor, Tensor]:
    pole_inputs: list[Tensor] = []
    energies: list[Tensor] = []
    model.eval()
    active_model = cast("_IdentityFeatureModel", cast("object", model))
    for batch in inputs.split(batch_size):
        first_local, delta, observation, valid = active_model._edge_stem(  # noqa: SLF001
            batch.to(device),
            None,
            None,
            None,
        )
        _, moments = active_model._writer(  # noqa: SLF001
            first_local,
            delta,
            observation,
            valid,
        )
        pole_inputs.append(first_local.cpu())
        energies.append(moments[:, : active_model.modes].cpu())
    return torch.cat(pole_inputs), torch.cat(energies)


def spectrum_equality_diagnostics(  # noqa: PLR0915
    values: Tensor,
    labels: Tensor,
    *,
    seed: int,
    permutations: int = SPECTRUM_PERMUTATIONS,
    bootstraps: int = SECOND_ORDER_BOOTSTRAPS,
    max_lag: int = 16,
    confidence_level: float = SECOND_ORDER_CONFIDENCE_LEVEL,
    autocovariance_tolerance: float = AUTOCOVARIANCE_EQUIVALENCE_TOLERANCE,
    spectrum_tolerance: float = SPECTRUM_EQUIVALENCE_TOLERANCE,
) -> dict[str, object]:
    """Test differences and independently establish second-order equivalence.

    The permutation p-value is retained as a difference diagnostic only.
    Equivalence requires bootstrap confidence intervals to fit inside the
    prospectively declared autocovariance and relative-spectrum margins.
    """
    if permutations < 1:
        message = "permutations must be positive"
        raise ValueError(message)
    if bootstraps < 2:
        message = "bootstraps must be at least two"
        raise ValueError(message)
    if not 0.0 < confidence_level < 1.0:
        message = "confidence_level must lie strictly between zero and one"
        raise ValueError(message)
    if autocovariance_tolerance <= 0.0 or spectrum_tolerance <= 0.0:
        message = "equivalence tolerances must be positive"
        raise ValueError(message)

    flattened = values.squeeze(-1).detach().cpu().to(torch.float64)
    labels = labels.detach().cpu()
    class_values = [flattened[labels == label] for label in (0, 1)]
    if any(active.shape[0] < 2 for active in class_values):
        message = "each class needs at least two paths"
        raise ValueError(message)
    class_means = [float(active.mean()) for active in class_values]
    sequence_autocovariances: list[Tensor] = []
    for active, active_mean in zip(class_values, class_means, strict=True):
        centered = active - active_mean
        sequence_autocovariances.append(
            torch.stack(
                [
                    (centered[:, lag:] * centered[:, : centered.shape[1] - lag]).mean(dim=1)
                    if lag
                    else centered.square().mean(dim=1)
                    for lag in range(max_lag + 1)
                ],
                dim=1,
            )
        )
    autocovariance_tensors = [active.mean(dim=0) for active in sequence_autocovariances]
    signed_autocovariance_difference = autocovariance_tensors[0] - autocovariance_tensors[1]
    absolute_autocovariance_difference = signed_autocovariance_difference.abs()

    centered = flattened - flattened.mean(dim=1, keepdim=True)
    periodograms = torch.fft.rfft(centered, dim=1).abs().square() / centered.shape[1]
    group_zero = periodograms[labels == 0]
    group_one = periodograms[labels == 1]
    mean_zero = group_zero.mean(dim=0)
    mean_one = group_one.mean(dim=0)

    def relative_spectrum_difference(left: Tensor, right: Tensor) -> Tensor:
        return torch.linalg.vector_norm(left - right) / (
            0.5 * (torch.linalg.vector_norm(left) + torch.linalg.vector_norm(right)) + 1e-12
        )

    observed = float(relative_spectrum_difference(mean_zero, mean_one))
    permutation_generator = torch.Generator().manual_seed(seed)
    exceedances = 0
    zero_count = int((labels == 0).sum())
    for _ in range(permutations):
        permutation = torch.randperm(
            labels.numel(),
            generator=permutation_generator,
        )
        permuted_zero = periodograms[permutation[:zero_count]].mean(dim=0)
        permuted_one = periodograms[permutation[zero_count:]].mean(dim=0)
        statistic = float(relative_spectrum_difference(permuted_zero, permuted_one))
        exceedances += statistic >= observed

    bootstrap_generator = torch.Generator().manual_seed(seed + 104_729)
    autocovariance_bootstrap = torch.empty(
        bootstraps,
        max_lag + 1,
        dtype=torch.float64,
    )
    spectrum_bootstrap = torch.empty(bootstraps, dtype=torch.float64)
    for index in range(bootstraps):
        zero_indices = torch.randint(
            sequence_autocovariances[0].shape[0],
            (sequence_autocovariances[0].shape[0],),
            generator=bootstrap_generator,
        )
        one_indices = torch.randint(
            sequence_autocovariances[1].shape[0],
            (sequence_autocovariances[1].shape[0],),
            generator=bootstrap_generator,
        )
        autocovariance_bootstrap[index] = sequence_autocovariances[0][zero_indices].mean(
            dim=0
        ) - sequence_autocovariances[1][one_indices].mean(dim=0)
        spectrum_bootstrap[index] = relative_spectrum_difference(
            group_zero[zero_indices].mean(dim=0),
            group_one[one_indices].mean(dim=0),
        )

    tail = 0.5 * (1.0 - confidence_level)
    autocovariance_lower = torch.quantile(
        autocovariance_bootstrap,
        tail,
        dim=0,
    )
    autocovariance_upper = torch.quantile(
        autocovariance_bootstrap,
        1.0 - tail,
        dim=0,
    )
    spectrum_lower = float(torch.quantile(spectrum_bootstrap, tail))
    spectrum_upper = float(torch.quantile(spectrum_bootstrap, 1.0 - tail))
    lag_intervals = [
        {
            "lag": lag,
            "point_signed_difference": float(signed_autocovariance_difference[lag]),
            "ci_low": float(autocovariance_lower[lag]),
            "ci_high": float(autocovariance_upper[lag]),
            "contained_within_tolerance": bool(
                autocovariance_lower[lag] >= -autocovariance_tolerance
                and autocovariance_upper[lag] <= autocovariance_tolerance
            ),
        }
        for lag in range(max_lag + 1)
    ]
    autocovariance_equivalent = all(
        bool(interval["contained_within_tolerance"]) for interval in lag_intervals
    )
    spectrum_equivalent = spectrum_upper <= spectrum_tolerance
    second_order_equivalent = autocovariance_equivalent and spectrum_equivalent
    return {
        "class_mean": class_means,
        "class_autocovariance_lag_0_to_max": [active.tolist() for active in autocovariance_tensors],
        "absolute_autocovariance_difference": (absolute_autocovariance_difference.tolist()),
        "maximum_nonzero_lag_autocovariance_difference": max(
            absolute_autocovariance_difference[1:].tolist(),
            default=0.0,
        ),
        "mean_periodogram": [mean_zero.tolist(), mean_one.tolist()],
        "relative_mean_spectrum_difference": observed,
        "spectrum_permutation_count": permutations,
        "spectrum_permutation_p_value": (1.0 + exceedances) / (permutations + 1.0),
        "difference_test_interpretation": ("non-rejection does not establish equality"),
        "equivalence": {
            "bootstrap_count": bootstraps,
            "confidence_level": confidence_level,
            "autocovariance": {
                "absolute_tolerance": autocovariance_tolerance,
                "lagwise_signed_difference_intervals": lag_intervals,
                "all_intervals_contained": autocovariance_equivalent,
            },
            "relative_mean_spectrum": {
                "tolerance": spectrum_tolerance,
                "point_difference": observed,
                "ci_low": spectrum_lower,
                "ci_high": spectrum_upper,
                "interval_contained": spectrum_equivalent,
            },
            "second_order_equivalence_established": second_order_equivalent,
        },
    }


class ActiveBudgetHead(nn.Module):
    """A logit-producing head that uses exactly a requested parameter budget."""

    def __init__(self, input_dim: int, output_dim: int, budget: int) -> None:
        super().__init__()
        base = input_dim * output_dim + output_dim
        if budget < base:
            message = f"head budget {budget} is below the linear minimum {base}"
            raise ValueError(message)
        self.linear = nn.Linear(input_dim, output_dim)
        per_hidden = input_dim + 1 + output_dim
        hidden = (budget - base) // per_hidden
        self.adapter_in = nn.Linear(input_dim, hidden) if hidden else None
        self.adapter_out = nn.Linear(hidden, output_dim, bias=False) if hidden else None
        used = base + hidden * per_hidden
        remainder = budget - used
        self.remainder: nn.Parameter | None = (
            nn.Parameter(torch.zeros(remainder)) if remainder else None
        )
        self.remainder_indices: Tensor
        self.remainder_signs: Tensor
        if remainder:
            indices = torch.arange(remainder, dtype=torch.long) % input_dim
            signs = torch.linspace(-1.0, 1.0, output_dim)
            self.register_buffer("remainder_indices", indices)
            self.register_buffer("remainder_signs", signs)
        else:
            self.register_buffer(
                "remainder_indices",
                torch.empty(0, dtype=torch.long),
            )
            self.register_buffer("remainder_signs", torch.empty(0))

    def forward(self, features: Tensor) -> Tensor:
        logits = self.linear(features)
        if self.adapter_in is not None and self.adapter_out is not None:
            logits = logits + self.adapter_out(functional.silu(self.adapter_in(features)))
        if self.remainder is not None:
            selected = features[:, self.remainder_indices]
            scalar = (selected * self.remainder).sum(dim=-1, keepdim=True) / math.sqrt(
                self.remainder.numel()
            )
            logits = logits + scalar * self.remainder_signs
        return logits


class HigherOrderClassifier(nn.Module):
    """Controlled fixed-linear writer with one of six readout/reader variants."""

    def __init__(
        self,
        variant: HigherOrderVariant,
        *,
        seed: int,
        modes: int = FIXED_MODES,
        parameter_target: int = HIGHER_ORDER_PARAMETER_TARGET,
    ) -> None:
        super().__init__()
        if variant not in HIGHER_ORDER_VARIANTS:
            message = f"unknown higher-order variant: {variant}"
            raise ValueError(message)
        self.variant = variant
        self.modes = modes
        self.state_dim = 2 * modes
        self.writer = FixedComplexPoleBank(torch.ones(modes, 1))
        generator = torch.Generator().manual_seed(seed)
        reader_directions = torch.randn(
            modes,
            self.state_dim,
            generator=generator,
        )
        self.reader_bank = FixedComplexPoleBank(reader_directions)
        self.reader_map: nn.Module | None
        if variant == "full_writer_reader":
            self.reader_map = nn.Linear(self.state_dim, self.state_dim)
        elif variant == "mlp_reader":
            self.reader_map = nn.Sequential(
                nn.Linear(self.state_dim, self.state_dim),
                nn.SiLU(),
                nn.Linear(self.state_dim, self.state_dim),
            )
        elif variant == "conv_reader":
            self.reader_map = nn.Conv1d(
                self.state_dim,
                self.state_dim,
                kernel_size=3,
                padding=1,
            )
        else:
            self.reader_map = None

        descriptor_dim = self._descriptor_dim()
        if variant in NONLINEAR_READER_VARIANTS:
            reader_parameters = _parameter_count(self)
            head_budget = parameter_target - reader_parameters
            self.head = ActiveBudgetHead(descriptor_dim, 2, head_budget)
            self.parameter_target = parameter_target
        else:
            self.head = nn.Linear(descriptor_dim, 2)
            self.parameter_target = _parameter_count(self)

    def _descriptor_dim(self) -> int:
        writer_moments = 7 * self.modes
        if self.variant == "writer_energy_only":
            return self.modes
        if self.variant == "writer_energy_lag":
            return writer_moments
        if self.variant == "one_scan_full":
            return writer_moments + 2 * self.state_dim
        if self.variant == "full_writer_reader":
            return 2 * writer_moments
        return writer_moments + 5 * self.state_dim

    def _descriptor(self, inputs: Tensor) -> Tensor:
        writer_real, writer_imag = self.writer(inputs)
        writer_moments = complex_modal_moments(writer_real, writer_imag)
        if self.variant == "writer_energy_only":
            return writer_moments[:, : self.modes]
        if self.variant == "writer_energy_lag":
            return writer_moments
        writer_path = torch.cat((writer_real, writer_imag), dim=-1)
        if self.variant == "one_scan_full":
            return torch.cat(
                (
                    writer_moments,
                    writer_path.mean(dim=1),
                    writer_path[:, -1],
                ),
                dim=-1,
            )
        if self.reader_map is None:
            message = "nonlinear reader variant lost its reader map"
            raise RuntimeError(message)
        if self.variant == "conv_reader":
            refined = functional.silu(self.reader_map(writer_path.transpose(1, 2)).transpose(1, 2))
        else:
            refined = functional.silu(self.reader_map(writer_path))
        if self.variant == "full_writer_reader":
            reader_real, reader_imag = self.reader_bank(refined)
            reader_moments = complex_modal_moments(reader_real, reader_imag)
            return torch.cat((writer_moments, reader_moments), dim=-1)
        return torch.cat((writer_moments, real_temporal_moments(refined)), dim=-1)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.head(self._descriptor(inputs))

    def capacity_match(self) -> dict[str, float | int | bool]:
        actual = _parameter_count(self)
        relative_error = (actual - self.parameter_target) / self.parameter_target
        return {
            "target_parameters": self.parameter_target,
            "actual_parameters": actual,
            "relative_error": relative_error,
            "within_three_percent": abs(relative_error) <= 0.03,
        }


def real_temporal_moments(values: Tensor) -> Tensor:
    chunks = [values.mean(dim=1), values.square().mean(dim=1)]
    chunks.extend((values[:, lag:] * values[:, :-lag]).mean(dim=1) for lag in MOMENT_LAGS)
    return torch.cat(chunks, dim=-1)


def fit_classifier(
    model: nn.Module,
    train_inputs: Tensor,
    train_labels: Tensor,
    validation_inputs: Tensor,
    validation_labels: Tensor,
    *,
    seed: int,
    device: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
) -> dict[str, object]:
    """Train without consulting TEST and restore the best validation checkpoint."""
    seed_everything(seed)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    best_score = -math.inf
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    generator = torch.Generator().manual_seed(seed * 8191 + 17)
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(
            train_inputs.shape[0],
            generator=generator,
        )
        for indices in permutation.split(batch_size):
            batch_inputs = train_inputs[indices].to(device)
            batch_labels = train_labels[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = functional.cross_entropy(model(batch_inputs), batch_labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            post_step = getattr(model, "post_optimizer_step", None)
            if callable(post_step):
                post_step()
        score = evaluate_balanced_accuracy(
            model,
            validation_inputs,
            validation_labels,
            device=device,
            batch_size=batch_size,
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
    if best_state is None:
        message = "training did not produce a validation checkpoint"
        raise RuntimeError(message)
    model.load_state_dict(best_state)
    model.to(device)
    return {
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_validation_balanced_accuracy": best_score,
    }


@torch.no_grad()
def evaluate_balanced_accuracy(
    model: nn.Module,
    inputs: Tensor,
    labels: Tensor,
    *,
    device: str,
    batch_size: int,
) -> float:
    model.eval()
    predictions = torch.cat(
        [model(batch.to(device)).argmax(dim=-1).cpu() for batch in inputs.split(batch_size)]
    )
    recalls = [
        float((predictions[labels == label] == label).to(torch.float64).mean())
        for label in sorted(int(value) for value in torch.unique(labels).tolist())
    ]
    return mean(recalls)


def run_higher_order_job(seed: int, *, device: str) -> dict[str, object]:
    train_inputs, train_labels = higher_order_samples(
        HIGHER_ORDER_TRAIN_COUNT,
        HIGHER_ORDER_LENGTH,
        seed=seed * 211 + 1,
    )
    validation_inputs, validation_labels = higher_order_samples(
        HIGHER_ORDER_VALIDATION_COUNT,
        HIGHER_ORDER_LENGTH,
        seed=seed * 211 + 2,
    )
    test_inputs, test_labels = higher_order_samples(
        HIGHER_ORDER_TEST_COUNT,
        HIGHER_ORDER_LENGTH,
        seed=seed * 211 + 3,
    )
    second_order = spectrum_equality_diagnostics(
        torch.cat((train_inputs, validation_inputs)),
        torch.cat((train_labels, validation_labels)),
        seed=seed * 211 + 4,
    )
    models: dict[str, object] = {}
    for variant_index, variant in enumerate(HIGHER_ORDER_VARIANTS):
        model = HigherOrderClassifier(
            variant,
            seed=seed,
        )
        training = fit_classifier(
            model,
            train_inputs,
            train_labels,
            validation_inputs,
            validation_labels,
            seed=seed * 31 + variant_index,
            device=device,
            epochs=HIGHER_ORDER_EPOCHS,
            batch_size=64,
            learning_rate=3e-3,
            weight_decay=1e-4,
        )
        test_score = evaluate_balanced_accuracy(
            model,
            test_inputs,
            test_labels,
            device=device,
            batch_size=128,
        )
        models[variant] = {
            **training,
            "test_balanced_accuracy": test_score,
            "parameters": _parameter_count(model),
            "capacity_match": model.capacity_match(),
        }
    return {
        "schema": "pac_temporal_controls_higher_order_result.v1",
        "process": {
            "common_phi": HIGHER_ORDER_PHI,
            "class_0_innovations": "unit Gaussian",
            "class_1_innovations": "unit Rademacher",
            "population_innovation_mean": [0.0, 0.0],
            "population_innovation_variance": [1.0, 1.0],
            "population_autocovariance": "Gamma(k)=phi^|k| for both classes",
            "length": HIGHER_ORDER_LENGTH,
            "train_count": HIGHER_ORDER_TRAIN_COUNT,
            "validation_count": HIGHER_ORDER_VALIDATION_COUNT,
            "test_count": HIGHER_ORDER_TEST_COUNT,
        },
        "second_order_equality": second_order,
        "models": models,
    }


def report_campaign(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((root / "completed").glob("*.json"))
    ]
    status = campaign_status(root)
    ar1_rows = [row for row in rows if row.get("family") == "p1_1_ar1"]
    higher_rows = [row for row in rows if row.get("family") == "p1_3_higher_order"]
    execution_done = bool(status["done"])
    p1_1 = _aggregate_ar1(
        ar1_rows,
        campaign_execution_done=execution_done,
    )
    p1_3 = _aggregate_higher_order(
        higher_rows,
        campaign_execution_done=execution_done,
    )
    complete_claims_authorized = bool(p1_1["complete_evidence"] and p1_3["complete_evidence"])
    payload: dict[str, object] = {
        "schema": "pac_temporal_controls_report.v1",
        "status": status,
        "report_state": ("complete" if complete_claims_authorized else "partial"),
        "complete_claims_authorized": complete_claims_authorized,
        "p1_1": p1_1,
        "p1_3": p1_3,
        "interpretation_policy": (
            "criteria are evaluated mechanically and failed criteria remain "
            "visible; incomplete execution never authorizes a complete claim"
        ),
    }
    _write_json(root / "reports" / "TEMPORAL_CONTROLS_REPORT.json", payload)
    _write_markdown_report(
        root / "reports" / "TEMPORAL_CONTROLS_REPORT.md",
        payload,
    )
    return payload


def _aggregate_ar1(
    rows: list[dict[str, Any]],
    *,
    campaign_execution_done: bool,
) -> dict[str, object]:
    if not rows:
        return {"rows": 0, "complete_evidence": False}
    table: dict[str, dict[str, object]] = {}
    all_accuracy_pass = True
    all_pass = True
    for view in AR1_VIEWS:
        controls: dict[str, object] = {}
        for control in AR1_CONTROLS:
            values = [
                float(
                    row["evaluations"][view][control]["energy_nearest_prototype"][
                        "balanced_accuracy"
                    ]
                )
                for row in rows
            ]
            summary = _mean_sd_ci(values)
            criterion = (
                summary["mean"] > 0.95 if control == "original" else 0.47 <= summary["mean"] <= 0.53
            )
            all_accuracy_pass = all_accuracy_pass and bool(criterion)
            all_pass = all_pass and bool(criterion)
            controls[control] = {
                **summary,
                "criterion_passed": criterion,
            }
        original_diagnostics = [row["evaluations"][view]["original"]["pole_input"] for row in rows]
        mean_difference = _mean_sd_ci(
            [float(value["standardized_mean_difference"]) for value in original_diagnostics]
        )
        gamma0_difference = _mean_sd_ci(
            [float(value["standardized_gamma0_difference"]) for value in original_diagnostics]
        )
        mean_gate = mean_difference["mean"] <= 0.10
        gamma0_gate = gamma0_difference["mean"] <= 0.10
        all_pass = all_pass and mean_gate and gamma0_gate
        table[view] = {
            "controls": controls,
            "pole_input_original": {
                "standardized_mean_difference": mean_difference,
                "standardized_mean_difference_at_most_0_10": mean_gate,
                "standardized_gamma0_difference": gamma0_difference,
                "standardized_gamma0_difference_at_most_0_10": gamma0_gate,
                "standardized_nonzero_lag_difference": {
                    str(lag): _mean_sd_ci(
                        [
                            float(value["standardized_nonzero_lag_difference"][str(lag)])
                            for value in original_diagnostics
                        ]
                    )
                    for lag in MOMENT_LAGS
                },
            },
        }
    return {
        "rows": len(rows),
        "views": table,
        "all_prespecified_accuracy_criteria_passed": all_accuracy_pass,
        "all_prespecified_criteria_passed": all_pass,
        "complete_evidence": (campaign_execution_done and len(rows) >= 10),
    }


def _aggregate_higher_order(
    rows: list[dict[str, Any]],
    *,
    campaign_execution_done: bool,
) -> dict[str, object]:
    if not rows:
        return {"rows": 0, "complete_evidence": False}
    model_scores = {
        variant: [float(row["models"][variant]["test_balanced_accuracy"]) for row in rows]
        for variant in HIGHER_ORDER_VARIANTS
    }
    summaries = {variant: _mean_sd_ci(values) for variant, values in model_scores.items()}
    full_minus_one = [
        full - one
        for full, one in zip(
            model_scores["full_writer_reader"],
            model_scores["one_scan_full"],
            strict=True,
        )
    ]
    full_minus_mlp = [
        full - control
        for full, control in zip(
            model_scores["full_writer_reader"],
            model_scores["mlp_reader"],
            strict=True,
        )
    ]
    full_minus_conv = [
        full - control
        for full, control in zip(
            model_scores["full_writer_reader"],
            model_scores["conv_reader"],
            strict=True,
        )
    ]
    full_minus_best_matched = [
        full - max(mlp, conv)
        for full, mlp, conv in zip(
            model_scores["full_writer_reader"],
            model_scores["mlp_reader"],
            model_scores["conv_reader"],
            strict=True,
        )
    ]
    p_values = [float(row["second_order_equality"]["spectrum_permutation_p_value"]) for row in rows]
    equivalence_rows = [row["second_order_equality"]["equivalence"] for row in rows]
    autocovariance_equivalence = [
        bool(active["autocovariance"]["all_intervals_contained"]) for active in equivalence_rows
    ]
    spectrum_equivalence = [
        bool(active["relative_mean_spectrum"]["interval_contained"]) for active in equivalence_rows
    ]
    second_order_equivalence = [
        bool(active["second_order_equivalence_established"]) for active in equivalence_rows
    ]
    autocovariance_interval_maxima = [
        max(
            max(abs(float(interval["ci_low"])), abs(float(interval["ci_high"])))
            for interval in active["autocovariance"]["lagwise_signed_difference_intervals"]
        )
        for active in equivalence_rows
    ]
    spectrum_interval_upper = [
        float(active["relative_mean_spectrum"]["ci_high"]) for active in equivalence_rows
    ]
    max_parameter_error = max(
        abs(float(row["models"][variant]["capacity_match"]["relative_error"]))
        for row in rows
        for variant in NONLINEAR_READER_VARIANTS
    )
    full_one_summary = _mean_sd_ci(full_minus_one)
    full_mlp_summary = _mean_sd_ci(full_minus_mlp)
    full_conv_summary = _mean_sd_ci(full_minus_conv)
    full_best_summary = _mean_sd_ci(full_minus_best_matched)
    one_scan_near_chance = 0.47 <= summaries["one_scan_full"]["mean"] <= 0.53
    full_one_gate = full_one_summary["mean"] >= 0.10
    full_best_gate = full_best_summary["mean"] > 0.0
    capacity_gate = max_parameter_error <= 0.03
    equivalence_gate = all(second_order_equivalence)
    return {
        "rows": len(rows),
        "model_test_balanced_accuracy": summaries,
        "one_scan_full_near_chance": {
            "prespecified_interval": [0.47, 0.53],
            "observed_mean": summaries["one_scan_full"]["mean"],
            "criterion_passed": one_scan_near_chance,
        },
        "second_order_equality": {
            "prespecified_equivalence_rule": {
                "confidence_level": SECOND_ORDER_CONFIDENCE_LEVEL,
                "autocovariance_absolute_tolerance": (AUTOCOVARIANCE_EQUIVALENCE_TOLERANCE),
                "relative_mean_spectrum_tolerance": (SPECTRUM_EQUIVALENCE_TOLERANCE),
                "non_rejection_is_not_equivalence": True,
            },
            "equivalence_established_fraction": (
                sum(second_order_equivalence) / len(second_order_equivalence)
            ),
            "all_seed_rows_established_equivalence": equivalence_gate,
            "autocovariance_interval_contained_fraction": (
                sum(autocovariance_equivalence) / len(autocovariance_equivalence)
            ),
            "relative_spectrum_interval_contained_fraction": (
                sum(spectrum_equivalence) / len(spectrum_equivalence)
            ),
            "maximum_absolute_autocovariance_ci_bound": _mean_sd_ci(autocovariance_interval_maxima),
            "relative_spectrum_ci_upper_bound": _mean_sd_ci(spectrum_interval_upper),
            "exploratory_difference_test": {
                "non_rejection_fraction_at_0.05": sum(value >= 0.05 for value in p_values)
                / len(p_values),
                "spectrum_permutation_p_value": _mean_sd_ci(p_values),
                "non_rejection_establishes_equality": False,
            },
            "maximum_nonzero_lag_autocovariance_difference": _mean_sd_ci(
                [
                    float(
                        row["second_order_equality"][
                            "maximum_nonzero_lag_autocovariance_difference"
                        ]
                    )
                    for row in rows
                ]
            ),
        },
        "paired_contrasts": {
            "full_minus_one_scan": {
                **full_one_summary,
                "ten_percentage_point_criterion_passed": full_one_gate,
            },
            "full_minus_mlp_reader": {
                **full_mlp_summary,
                "strict_improvement_criterion_passed": (full_mlp_summary["mean"] > 0.0),
            },
            "full_minus_conv_reader": {
                **full_conv_summary,
                "strict_improvement_criterion_passed": (full_conv_summary["mean"] > 0.0),
            },
            "full_minus_best_matched_nonlinear_reader": {
                **full_best_summary,
                "strict_improvement_criterion_passed": full_best_gate,
            },
        },
        "capacity_match": {
            "maximum_absolute_relative_error": max_parameter_error,
            "within_three_percent": capacity_gate,
        },
        "all_prespecified_criteria_passed": (
            equivalence_gate
            and one_scan_near_chance
            and full_one_gate
            and full_best_gate
            and capacity_gate
        ),
        "complete_evidence": (campaign_execution_done and len(rows) >= 10),
    }


def _write_markdown_report(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    p1_1 = cast("dict[str, Any]", payload["p1_1"])
    p1_3 = cast("dict[str, Any]", payload["p1_3"])
    status = cast("dict[str, Any]", payload["status"])
    report_state = str(payload["report_state"])
    lines = [
        "# Temporal controls report",
        "",
        f"Report state: **{report_state.upper()}**.",
        f"Complete claims authorized: **{payload['complete_claims_authorized']}**.",
        (
            "This is an explicitly partial report; no complete or confirmatory "
            "claim may be made from it."
            if report_state == "partial"
            else "All scheduled rows required for complete evidence are present."
        ),
        "",
        f"Campaign status: {status['completed']}/{status['expected']} completed, "
        f"{status['failed']} failed.",
        "",
        "## P1-1: matched-marginal AR(1)",
        "",
    ]
    if p1_1.get("rows"):
        lines.extend(
            (
                "| View | Original NP-BA | Global shuffle | Timestamp rearrangement | "
                "Marginal-preserving destruction |",
                "|---|---:|---:|---:|---:|",
            )
        )
        for view in AR1_VIEWS:
            controls = p1_1["views"][view]["controls"]
            lines.append(
                f"| {view} | {controls['original']['mean']:.3f} | "
                f"{controls['global_time_shuffle']['mean']:.3f} | "
                f"{controls['timestamp_rearrangement']['mean']:.3f} | "
                f"{controls['marginal_autocorrelation_destroyed']['mean']:.3f} |"
            )
        lines.extend(
            (
                "",
                "| View | Std. pole-input mean diff. | <=0.10 | Std. Gamma(0) diff. | <=0.10 |",
                "|---|---:|:---:|---:|:---:|",
            )
        )
        for view in AR1_VIEWS:
            diagnostics = p1_1["views"][view]["pole_input_original"]
            lines.append(
                f"| {view} | "
                f"{diagnostics['standardized_mean_difference']['mean']:.3f} | "
                f"{diagnostics['standardized_mean_difference_at_most_0_10']} | "
                f"{diagnostics['standardized_gamma0_difference']['mean']:.3f} | "
                f"{diagnostics['standardized_gamma0_difference_at_most_0_10']} |"
            )
        lines.extend(
            (
                "",
                "Current-row prespecified P1-1 gates passed "
                "(not a complete claim unless the report state is COMPLETE): "
                f"**{p1_1['all_prespecified_criteria_passed']}**.",
            )
        )
    else:
        lines.append("No P1-1 rows are available.")

    lines.extend(("", "## P1-3: candidate equal-second-order higher-order law", ""))
    if p1_3.get("rows"):
        lines.extend(
            (
                "| Variant | TEST balanced accuracy |",
                "|---|---:|",
            )
        )
        for variant in HIGHER_ORDER_VARIANTS:
            summary = p1_3["model_test_balanced_accuracy"][variant]
            lines.append(f"| {variant} | {summary['mean']:.3f} +/- {summary['sample_sd']:.3f} |")
        contrasts = p1_3["paired_contrasts"]
        second_order = p1_3["second_order_equality"]
        difference_test = second_order["exploratory_difference_test"]
        lines.extend(
            (
                "",
                "One-scan TEST BA inside the prespecified [0.47, 0.53] "
                "near-chance interval: "
                f"**{p1_3['one_scan_full_near_chance']['criterion_passed']}** "
                f"(mean {p1_3['one_scan_full_near_chance']['observed_mean']:.3f}).",
                "",
                f"Full minus one-scan: {contrasts['full_minus_one_scan']['mean']:+.3f}; "
                "10-point criterion passed: "
                f"**{contrasts['full_minus_one_scan']['ten_percentage_point_criterion_passed']}**.",
                "",
                f"Full minus MLP reader: {contrasts['full_minus_mlp_reader']['mean']:+.3f}; "
                "strict-improvement criterion passed: "
                f"**{contrasts['full_minus_mlp_reader']['strict_improvement_criterion_passed']}**.",
                "",
                f"Full minus convolution reader: "
                f"{contrasts['full_minus_conv_reader']['mean']:+.3f}; "
                "strict-improvement criterion passed: "
                f"**{contrasts['full_minus_conv_reader']['strict_improvement_criterion_passed']}**.",
                "",
                "Full minus the better parameter-matched nonlinear reader: "
                f"{contrasts['full_minus_best_matched_nonlinear_reader']['mean']:+.3f}; "
                "strict-improvement criterion passed: "
                f"**{contrasts['full_minus_best_matched_nonlinear_reader']['strict_improvement_criterion_passed']}**.",
                "",
                "Bootstrap second-order equivalence established fraction: "
                f"{second_order['equivalence_established_fraction']:.3f}; "
                "established for every seed row: "
                f"**{second_order['all_seed_rows_established_equivalence']}**.",
                (
                    "Equivalence requires every lag-0..16 autocovariance "
                    f"interval inside +/-{AUTOCOVARIANCE_EQUIVALENCE_TOLERANCE:.2f} "
                    "and the relative mean-spectrum interval upper bound <= "
                    f"{SPECTRUM_EQUIVALENCE_TOLERANCE:.2f}."
                ),
                "",
                "Exploratory spectrum difference-test non-rejection fraction: "
                f"{difference_test['non_rejection_fraction_at_0.05']:.3f}. "
                "Non-rejection is not used as evidence of equality.",
                "",
                "All nonlinear readers within 3% parameter tolerance: "
                f"**{p1_3['capacity_match']['within_three_percent']}**.",
                "",
                "Current-row prespecified P1-3 gates passed "
                "(not a complete claim unless the report state is COMPLETE): "
                f"**{p1_3['all_prespecified_criteria_passed']}**.",
            )
        )
    else:
        lines.append("No P1-3 rows are available.")
    lines.extend(
        (
            "",
            "This report records failed gates without rewriting or promoting paper claims.",
            "",
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def _mean_sd_ci(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "mean": math.nan,
            "sample_sd": math.nan,
            "ci95_low": math.nan,
            "ci95_high": math.nan,
        }
    average = mean(values)
    deviation = stdev(values) if len(values) > 1 else 0.0
    half_width = 1.96 * deviation / math.sqrt(len(values))
    return {
        "mean": average,
        "sample_sd": deviation,
        "ci95_low": average - half_width,
        "ci95_high": average + half_width,
    }


def _read_jobs(path: Path) -> list[TemporalControlJob]:
    return [
        TemporalControlJob(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _source_hashes() -> dict[str, str]:
    return source_file_hashes(
        ("src/lnet/pac_temporal_controls_campaign.py",),
        project_root=Path(__file__).resolve().parents[2],
        missing="placeholder",
    )


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


__all__ = [
    "AR1_CONTROLS",
    "AR1_VIEWS",
    "AUTOCOVARIANCE_EQUIVALENCE_TOLERANCE",
    "DEFAULT_ROOT",
    "DEFAULT_SEEDS",
    "HIGHER_ORDER_VARIANTS",
    "SECOND_ORDER_BOOTSTRAPS",
    "SPECTRUM_EQUIVALENCE_TOLERANCE",
    "HigherOrderClassifier",
    "TemporalControlJob",
    "apply_ar1_control",
    "campaign_jobs",
    "campaign_status",
    "complex_modal_moments",
    "higher_order_samples",
    "matched_ar1_samples",
    "prepare_campaign",
    "prototype_metrics",
    "report_campaign",
    "run_ar1_control_job",
    "run_higher_order_job",
    "run_job",
    "run_manifest",
    "spectrum_equality_diagnostics",
    "temporal_statistics",
]
