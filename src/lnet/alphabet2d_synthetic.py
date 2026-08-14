# ruff: noqa: EM101, EM102, TRY003
"""Synthetic controls for diagnosing an ALPHABET-2D implementation.

The two tasks deliberately separate spatial spectrum from spatial phase:

* ``off_axis`` is a Gaussian field whose class spectra differ near an off-axis
  frequency tile.  The difference is projected to have exactly zero covariance
  at every requested small vector lag.
* ``equal_power_phase`` uses paired examples with identical Fourier magnitude
  and different, fixed spatial phase arrangements.

All generators are CPU deterministic for a fixed seed.  The returned tensors
can subsequently be moved to an accelerator by the caller.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

Alphabet2DTask = Literal["off_axis", "equal_power_phase"]


@dataclass(frozen=True, slots=True)
class OffAxisSpectralConfig:
    """Configuration of the moment-matched Gaussian random-field task."""

    height: int = 32
    width: int = 32
    channels: int = 1
    omega_x: float = math.pi / 4.0
    omega_y: float = math.pi / 3.0
    bandwidth: float = 0.24
    contrast: float = 0.8
    matched_lag_radius: int = 2

    def validate(self) -> None:
        if self.height < 8 or self.width < 8:
            raise ValueError("off-axis fields require height and width >= 8")
        if self.channels < 1:
            raise ValueError("channels must be positive")
        if self.bandwidth <= 0.0:
            raise ValueError("bandwidth must be positive")
        if self.contrast not in {0.2, 0.4, 0.6, 0.8}:
            raise ValueError("contrast must be one of the preregistered epsilon values")
        if self.matched_lag_radius < 0:
            raise ValueError("matched_lag_radius must be non-negative")
        if 2 * self.matched_lag_radius + 1 >= min(self.height, self.width):
            raise ValueError("matched lag neighborhood is too large for the field")


@dataclass(frozen=True, slots=True)
class EqualPowerPhaseConfig:
    """Configuration of the equal-power, different-arrangement task."""

    height: int = 32
    width: int = 32
    channels: int = 1
    blob_scale: float = 0.10
    spectral_jitter: float = 0.20
    scramble_seed: int = 91_337

    def validate(self) -> None:
        if self.height < 8 or self.width < 8:
            raise ValueError("phase fields require height and width >= 8")
        if self.channels < 1:
            raise ValueError("channels must be positive")
        if not 0.02 <= self.blob_scale <= 0.30:
            raise ValueError("blob_scale must be in [0.02, 0.30]")
        if self.spectral_jitter < 0.0:
            raise ValueError("spectral_jitter must be non-negative")


@dataclass(frozen=True, slots=True)
class TensorClassificationSplit:
    """One immutable tensor split."""

    inputs: Tensor
    targets: Tensor


@dataclass(frozen=True, slots=True)
class Alphabet2DSplits:
    """Train/validation/test tensors for a synthetic classification task."""

    train: TensorClassificationSplit
    validation: TensorClassificationSplit
    test: TensorClassificationSplit


def _frequency_grid(height: int, width: int, *, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    omega_y = torch.fft.fftfreq(height, dtype=dtype) * (2.0 * math.pi)
    omega_x = torch.fft.fftfreq(width, dtype=dtype) * (2.0 * math.pi)
    grid_y, grid_x = torch.meshgrid(omega_y, omega_x, indexing="ij")
    return grid_x, grid_y


def _wrapped_distance(values: Tensor, center: float) -> Tensor:
    return torch.atan2(torch.sin(values - center), torch.cos(values - center))


def _matched_lags(radius: int) -> tuple[tuple[int, int], ...]:
    # cos(k.lag) is invariant to negating the lag, so retain one from each pair.
    lags = [(dx, 0) for dx in range(radius + 1)]
    lags.extend(
        (dx, dy)
        for dy in range(1, radius + 1)
        for dx in range(-radius, radius + 1)
    )
    return tuple(lags)


def off_axis_spectra(config: OffAxisSpectralConfig) -> tuple[Tensor, Tensor, Tensor]:
    """Return class spectra and their normalized discriminating direction.

    The outputs have shape ``[height, width]`` and dtype ``float64``.  For all
    ``|dx|, |dy| <= matched_lag_radius``, the inverse Fourier transforms of the
    two spectra agree at lag ``(dx, dy)`` up to numerical precision.
    """

    config.validate()
    grid_x, grid_y = _frequency_grid(config.height, config.width, dtype=torch.float64)
    scale = 2.0 * config.bandwidth**2

    def bump(center_x: float, center_y: float) -> Tensor:
        distance = _wrapped_distance(grid_x, center_x).square()
        distance = distance + _wrapped_distance(grid_y, center_y).square()
        return torch.exp(-distance / scale)

    # The paired bumps make the spectral density even, hence a valid spectrum
    # for a real-valued stationary field.
    raw_direction = bump(config.omega_x, config.omega_y)
    raw_direction = raw_direction + bump(-config.omega_x, -config.omega_y)

    rows = [
        torch.cos(grid_x * dx + grid_y * dy).reshape(-1)
        for dx, dy in _matched_lags(config.matched_lag_radius)
    ]
    # Match every horizontal and vertical spectral marginal as well.  This
    # prevents an axial-only model from reading the joint off-axis signal from
    # a changed one-dimensional projection of the spectrum.
    for column in range(config.width):
        marginal = torch.zeros(config.height, config.width, dtype=torch.float64)
        marginal[:, column] = 1.0
        rows.append(marginal.reshape(-1))
    for row in range(config.height):
        marginal = torch.zeros(config.height, config.width, dtype=torch.float64)
        marginal[row, :] = 1.0
        rows.append(marginal.reshape(-1))
    constraint = torch.stack(rows)
    raw_flat = raw_direction.reshape(-1)
    _, singular_values, right_vectors = torch.linalg.svd(
        constraint,
        full_matrices=False,
    )
    tolerance = (
        max(constraint.shape)
        * torch.finfo(constraint.dtype).eps
        * singular_values.amax()
    )
    row_basis = right_vectors[singular_values > tolerance]
    projected = row_basis.T @ (row_basis @ raw_flat)
    direction = (raw_flat - projected).reshape(config.height, config.width)
    direction = 0.5 * (direction + _negate_frequencies(direction))
    direction = direction / direction.abs().amax().clamp_min(torch.finfo(direction.dtype).eps)

    radial_frequency = torch.sqrt(grid_x.square() + grid_y.square())
    class_zero = (0.15 + radial_frequency.square()).rsqrt()
    class_zero = class_zero / class_zero.mean()
    negative = direction < 0
    safe_scale = (
        0.9
        * (class_zero[negative] / (-0.8 * direction[negative])).amin()
    )
    class_one = class_zero + config.contrast * safe_scale * direction
    if float(torch.minimum(class_zero.amin(), class_one.amin())) <= 0.0:
        raise RuntimeError("constructed spectra are not strictly positive")
    return class_zero, class_one, direction


def _negate_frequencies(values: Tensor) -> Tensor:
    height, width = values.shape[-2:]
    indices_y = torch.remainder(-torch.arange(height), height)
    indices_x = torch.remainder(-torch.arange(width), width)
    return values.index_select(-2, indices_y).index_select(-1, indices_x)


def _balanced_shuffle(
    inputs: Tensor,
    targets: Tensor,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    order = torch.randperm(targets.numel(), generator=generator)
    return inputs[order], targets[order]


def make_off_axis_spectral_dataset(
    samples_per_class: int,
    *,
    seed: int,
    config: OffAxisSpectralConfig | None = None,
) -> tuple[Tensor, Tensor]:
    """Sample a balanced, moment-matched Gaussian spectral-field dataset."""

    active = config or OffAxisSpectralConfig()
    active.validate()
    if samples_per_class < 1:
        raise ValueError("samples_per_class must be positive")
    class_zero, class_one, _ = off_axis_spectra(active)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    fields: list[Tensor] = []
    for spectrum in (class_zero, class_one):
        white = torch.randn(
            samples_per_class,
            active.channels,
            active.height,
            active.width,
            generator=generator,
            dtype=torch.float64,
        )
        coefficients = torch.fft.fft2(white, norm="ortho")
        field = torch.fft.ifft2(coefficients * spectrum.sqrt(), norm="ortho").real
        fields.append(field.to(torch.float32))
    inputs = torch.cat(fields)
    targets = torch.arange(2, dtype=torch.long).repeat_interleave(samples_per_class)
    return _balanced_shuffle(inputs, targets, generator)


def off_axis_oracle_score(
    inputs: Tensor,
    config: OffAxisSpectralConfig | None = None,
) -> Tensor:
    """Return the exact Gaussian log-likelihood-ratio score (class 1 vs 0)."""

    active = config or OffAxisSpectralConfig()
    class_zero, class_one, _ = off_axis_spectra(active)
    if inputs.ndim != 4 or inputs.shape[-2:] != (active.height, active.width):
        raise ValueError("inputs must have shape [batch, channels, configured height, width]")
    power = torch.fft.fft2(inputs.to(torch.float64), norm="ortho").abs().square()
    log_ratio = torch.log(class_one / class_zero)
    precision_difference = class_one.reciprocal() - class_zero.reciprocal()
    # Full FFT bins double-count conjugate pairs, which only scales the LLR and
    # therefore does not alter its sign or ranking.
    return -0.5 * (log_ratio + power * precision_difference).sum(dim=(-3, -2, -1))


def _phase_references(config: EqualPowerPhaseConfig) -> tuple[Tensor, Tensor, Tensor]:
    config.validate()
    dtype = torch.float64
    y = torch.arange(config.height, dtype=dtype) / config.height - 0.5
    x = torch.arange(config.width, dtype=dtype) / config.width - 0.5
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")

    def blob(center_x: float, center_y: float, weight: float) -> Tensor:
        radius = (grid_x - center_x).square() + (grid_y - center_y).square()
        return weight * torch.exp(-radius / (2.0 * config.blob_scale**2))

    arrangement = blob(-0.22, -0.10, 1.0)
    arrangement = arrangement + blob(0.18, 0.11, 0.72) + blob(0.03, 0.25, 0.43)
    arrangement = arrangement - arrangement.mean()
    base_coefficients = torch.fft.fft2(arrangement, norm="ortho")
    magnitude = base_coefficients.abs()

    generator = torch.Generator(device="cpu").manual_seed(config.scramble_seed)
    scramble_source = torch.randn(config.height, config.width, generator=generator, dtype=dtype)
    scrambled_coefficients = torch.fft.fft2(scramble_source, norm="ortho")
    epsilon = torch.finfo(dtype).eps
    phase_zero = base_coefficients / magnitude.clamp_min(epsilon)
    phase_one = scrambled_coefficients / scrambled_coefficients.abs().clamp_min(epsilon)
    phase_zero = torch.where(magnitude > epsilon, phase_zero, torch.zeros_like(phase_zero))
    phase_one = torch.where(magnitude > epsilon, phase_one, torch.zeros_like(phase_one))
    return magnitude, phase_zero, phase_one


def make_equal_power_phase_dataset(
    samples_per_class: int,
    *,
    seed: int,
    config: EqualPowerPhaseConfig | None = None,
) -> tuple[Tensor, Tensor]:
    """Create paired examples with equal Fourier power and different phase."""

    active = config or EqualPowerPhaseConfig()
    active.validate()
    if samples_per_class < 1:
        raise ValueError("samples_per_class must be positive")
    magnitude, phase_zero, phase_one = _phase_references(active)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    grid_x, grid_y = _frequency_grid(active.height, active.width, dtype=torch.float64)
    radial_coordinate = (grid_x.square() + grid_y.square()).sqrt() / (math.pi * math.sqrt(2.0))

    examples = [[], []]
    for _ in range(samples_per_class):
        global_gain = torch.randn((), generator=generator, dtype=torch.float64)
        radial_gain = torch.randn((), generator=generator, dtype=torch.float64)
        gain = torch.exp(
            active.spectral_jitter * (0.5 * global_gain + radial_gain * radial_coordinate)
        )
        shift_y = int(torch.randint(active.height, (), generator=generator).item())
        shift_x = int(torch.randint(active.width, (), generator=generator).item())
        for label, phase in enumerate((phase_zero, phase_one)):
            image = torch.fft.ifft2(magnitude * gain * phase, norm="ortho").real
            image = torch.roll(image, shifts=(shift_y, shift_x), dims=(-2, -1))
            examples[label].append(image.expand(active.channels, -1, -1).clone())

    inputs = torch.stack(examples[0] + examples[1]).to(torch.float32)
    targets = torch.arange(2, dtype=torch.long).repeat_interleave(samples_per_class)
    return _balanced_shuffle(inputs, targets, generator)


def phase_arrangement_oracle_score(
    inputs: Tensor,
    config: EqualPowerPhaseConfig | None = None,
) -> Tensor:
    """Score phase coherence with each template, invariant to circular shift."""

    active = config or EqualPowerPhaseConfig()
    _, phase_zero, phase_one = _phase_references(active)
    if inputs.ndim != 4 or inputs.shape[-2:] != (active.height, active.width):
        raise ValueError("inputs must have shape [batch, channels, configured height, width]")
    coefficients = torch.fft.fft2(inputs.to(torch.float64), norm="ortho")
    epsilon = torch.finfo(torch.float64).eps
    unit_phase = coefficients / coefficients.abs().clamp_min(epsilon)

    def coherence(reference: Tensor) -> Tensor:
        correlation = torch.fft.ifft2(unit_phase * reference.conj(), norm="ortho")
        return correlation.abs().amax(dim=(-2, -1)).mean(dim=-1)

    return coherence(phase_one) - coherence(phase_zero)


def make_alphabet2d_splits(
    task: Alphabet2DTask,
    *,
    train_per_class: int,
    validation_per_class: int,
    test_per_class: int,
    seed: int,
    off_axis_config: OffAxisSpectralConfig | None = None,
    phase_config: EqualPowerPhaseConfig | None = None,
) -> Alphabet2DSplits:
    """Build independent deterministic train/validation/test splits."""

    if min(train_per_class, validation_per_class, test_per_class) < 1:
        raise ValueError("all split sizes must be positive")
    if task not in {"off_axis", "equal_power_phase"}:
        raise ValueError(f"unknown ALPHABET-2D task: {task}")

    def generate(count: int, split_seed: int) -> tuple[Tensor, Tensor]:
        if task == "off_axis":
            return make_off_axis_spectral_dataset(
                count,
                seed=split_seed,
                config=off_axis_config,
            )
        return make_equal_power_phase_dataset(count, seed=split_seed, config=phase_config)

    train = generate(train_per_class, seed)
    validation = generate(validation_per_class, seed + 10_007)
    test = generate(test_per_class, seed + 20_011)
    return Alphabet2DSplits(
        train=TensorClassificationSplit(*train),
        validation=TensorClassificationSplit(*validation),
        test=TensorClassificationSplit(*test),
    )


__all__ = [
    "Alphabet2DSplits",
    "Alphabet2DTask",
    "EqualPowerPhaseConfig",
    "OffAxisSpectralConfig",
    "TensorClassificationSplit",
    "make_alphabet2d_splits",
    "make_equal_power_phase_dataset",
    "make_off_axis_spectral_dataset",
    "off_axis_oracle_score",
    "off_axis_spectra",
    "phase_arrangement_oracle_score",
]
