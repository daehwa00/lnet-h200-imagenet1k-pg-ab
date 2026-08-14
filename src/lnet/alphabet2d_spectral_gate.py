"""Mechanistic Phase S-A controls for the ALPHABET-2D product-pole hypothesis."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional

from .alphabet2d import product_pole_scan_2d
from .pac_real2d_math import discrete_pole_real2d
from .pac_recurrence import recurrence_real2d_directional

SpectralGateVariant = Literal[
    "local_covariance",
    "raster1d",
    "axial2d",
    "product_single",
    "product_four",
]
SPECTRAL_GATE_VARIANTS: tuple[SpectralGateVariant, ...] = (
    "local_covariance",
    "raster1d",
    "axial2d",
    "product_single",
    "product_four",
)


@dataclass(frozen=True, slots=True)
class SpectralGateConfig:
    """Frozen feature and affine-head contract shared by all S-A controls."""

    modes: int = 16
    pole_radius: float = 0.94
    matched_lag_radius: int = 2
    head_epochs: int = 300
    head_patience: int = 30
    head_learning_rate: float = 3.0e-2
    head_weight_decay: float = 1.0e-4

    def validate(self) -> None:
        if self.modes < 4 or self.modes % 4:
            message = "modes must be a positive multiple of four"
            raise ValueError(message)
        if not 0.0 < self.pole_radius < 1.0:
            message = "pole_radius must lie in (0, 1)"
            raise ValueError(message)
        if self.matched_lag_radius < 0:
            message = "matched_lag_radius must be nonnegative"
            raise ValueError(message)


def _complex_drive(
    excitation: Tensor,
    damping: Tensor,
    frequency: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    decay_real, decay_imag, gamma_real, gamma_imag = discrete_pole_real2d(
        damping,
        frequency,
        1.0,
    )
    return decay_real, decay_imag, excitation * gamma_real, excitation * gamma_imag


def _scan_1d(
    excitation: Tensor,
    frequency: Tensor,
    pole_radius: float,
) -> tuple[Tensor, Tensor]:
    modes = frequency.numel()
    damping = torch.full_like(frequency, -math.log(pole_radius))
    decay_real, decay_imag, drive_real, drive_imag = _complex_drive(
        excitation,
        damping,
        frequency,
    )
    return recurrence_real2d_directional(
        decay_real.view(1, 1, modes).expand_as(drive_real),
        decay_imag.view(1, 1, modes).expand_as(drive_imag),
        drive_real,
        drive_imag,
        "real2d_loop",
        "forward",
    )


def _digital_frequency_pairs(modes: int, device: torch.device) -> tuple[Tensor, Tensor]:
    side = round(math.sqrt(modes))
    if side * side != modes:
        message = "product modes must form a square frequency atlas"
        raise ValueError(message)
    values = torch.linspace(math.pi / 8.0, math.pi / 2.0, side, device=device)
    grid_y, grid_x = torch.meshgrid(values, values, indexing="ij")
    return grid_x.flatten(), grid_y.flatten()


def product_energy_features(
    inputs: Tensor,
    *,
    modes: int,
    four_scan: bool,
    pole_radius: float = 0.94,
) -> Tensor:
    """Return global product-Poisson energy using one or four causal scans."""
    if inputs.ndim != 4 or inputs.shape[1] != 1:
        message = "inputs must have shape [B,1,H,W]"
        raise ValueError(message)
    frequency_x, frequency_y = _digital_frequency_pairs(modes, inputs.device)
    damping = torch.full_like(frequency_x, -math.log(pole_radius))
    excitation = inputs[:, 0, :, :, None].expand(-1, -1, -1, modes)
    zeros = torch.zeros_like(excitation)
    directions = (
        ((1, 1), (-1, 1), (1, -1), (-1, -1))
        if four_scan
        else ((1, 1),)
    )
    energies = []
    for direction_x, direction_y in directions:
        real, imag = product_pole_scan_2d(
            excitation,
            zeros,
            damping_x=damping,
            damping_y=damping,
            frequency_x=frequency_x,
            frequency_y=frequency_y,
            spacing_x=1.0,
            spacing_y=1.0,
            direction_x=direction_x,
            direction_y=direction_y,
            recurrence_backend="real2d_loop",
        )
        energies.append((real.square() + imag.square()).mean(dim=(1, 2)))
    return torch.stack(energies).mean(dim=0).log1p()


def axial_energy_features(
    inputs: Tensor,
    *,
    modes: int,
    pole_radius: float = 0.94,
) -> Tensor:
    """Return equally sized horizontal/vertical one-dimensional pole energies."""
    if modes % 2:
        message = "axial modes must be even"
        raise ValueError(message)
    per_axis = modes // 2
    frequency = torch.linspace(
        math.pi / 8.0,
        math.pi,
        per_axis,
        device=inputs.device,
    )
    batch, _, height, width = inputs.shape
    horizontal = inputs[:, 0, :, :, None].expand(-1, -1, -1, per_axis)
    horizontal = horizontal.reshape(batch * height, width, per_axis)
    horizontal_real, horizontal_imag = _scan_1d(
        horizontal,
        frequency,
        pole_radius,
    )
    horizontal_energy = (
        horizontal_real.square() + horizontal_imag.square()
    ).reshape(batch, height, width, per_axis).mean(dim=(1, 2))
    vertical = inputs[:, 0].transpose(1, 2)[..., None].expand(
        -1,
        -1,
        -1,
        per_axis,
    )
    vertical = vertical.reshape(batch * width, height, per_axis)
    vertical_real, vertical_imag = _scan_1d(vertical, frequency, pole_radius)
    vertical_energy = (
        vertical_real.square() + vertical_imag.square()
    ).reshape(batch, width, height, per_axis).mean(dim=(1, 2))
    return torch.cat((horizontal_energy, vertical_energy), dim=-1).log1p()


def raster_energy_features(
    inputs: Tensor,
    *,
    modes: int,
    pole_radius: float = 0.94,
) -> Tensor:
    """Return forward raster-scan pole energies."""
    batch = inputs.shape[0]
    frequency = torch.linspace(
        math.pi / 16.0,
        math.pi,
        modes,
        device=inputs.device,
    )
    flat = inputs[:, 0].flatten(1)[..., None].expand(-1, -1, modes)
    real, imag = _scan_1d(flat, frequency, pole_radius)
    return (real.square() + imag.square()).mean(dim=1).reshape(batch, modes).log1p()


def local_covariance_features(inputs: Tensor, *, radius: int) -> Tensor:
    """Return the exact raw coordinates used by the affine covariance control."""
    lags = tuple(
        (dx, dy)
        for dy in range(radius + 1)
        for dx in range(-radius, radius + 1)
        if dy > 0 or dx > 0
    )
    centered = inputs - inputs.mean(dim=(-2, -1), keepdim=True)
    coordinates = [
        inputs.mean(dim=(-3, -2, -1)),
        centered.square().mean(dim=(-3, -2, -1)),
    ]
    height, width = inputs.shape[-2:]
    for delta_x, delta_y in lags:
        current_y = slice(delta_y, height)
        previous_y = slice(0, height - delta_y)
        if delta_x >= 0:
            current_x = slice(delta_x, width)
            previous_x = slice(0, width - delta_x)
        else:
            current_x = slice(0, width + delta_x)
            previous_x = slice(-delta_x, width)
        covariance = (
            centered[..., current_y, current_x]
            * centered[..., previous_y, previous_x]
        ).mean(dim=(-3, -2, -1))
        coordinates.append(covariance)
    return torch.stack(coordinates, dim=-1)


def extract_spectral_features(
    inputs: Tensor,
    variant: SpectralGateVariant,
    config: SpectralGateConfig,
) -> Tensor:
    """Extract one fixed, label-free S-A representation."""
    config.validate()
    if variant == "local_covariance":
        return local_covariance_features(inputs, radius=config.matched_lag_radius)
    if variant == "raster1d":
        return raster_energy_features(
            inputs,
            modes=config.modes,
            pole_radius=config.pole_radius,
        )
    if variant == "axial2d":
        return axial_energy_features(
            inputs,
            modes=config.modes,
            pole_radius=config.pole_radius,
        )
    if variant in {"product_single", "product_four"}:
        return product_energy_features(
            inputs,
            modes=config.modes,
            four_scan=variant == "product_four",
            pole_radius=config.pole_radius,
        )
    message = f"unknown spectral gate variant: {variant}"
    raise ValueError(message)


def fit_affine_head(
    train_features: Tensor,
    train_targets: Tensor,
    validation_features: Tensor,
    validation_targets: Tensor,
    *,
    seed: int,
    config: SpectralGateConfig,
) -> tuple[nn.Linear, dict[str, Tensor | float | int]]:
    """Fit the identical standardized affine head for every representation."""
    mean = train_features.mean(dim=0)
    scale = train_features.std(dim=0).clamp_min(1.0e-6)
    train = (train_features - mean) / scale
    validation = (validation_features - mean) / scale
    torch.manual_seed(seed)
    head = nn.Linear(train.shape[1], 2, device=train.device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=config.head_learning_rate,
        weight_decay=config.head_weight_decay,
    )
    best_accuracy = -1.0
    best_loss = math.inf
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    stale = 0
    for epoch in range(1, config.head_epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = functional.cross_entropy(head(train), train_targets)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            validation_logits = head(validation)
            accuracy = float(
                (validation_logits.argmax(dim=-1) == validation_targets).float().mean()
            )
            validation_loss = float(
                functional.cross_entropy(validation_logits, validation_targets)
            )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_accuracy = accuracy
            best_epoch = epoch
            best_state = copy.deepcopy(head.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= config.head_patience:
                break
    if best_state is None:
        message = "affine head did not produce a validation checkpoint"
        raise RuntimeError(message)
    head.load_state_dict(best_state)
    return head, {
        "mean": mean,
        "scale": scale,
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_accuracy,
        "best_validation_loss": best_loss,
    }


__all__ = [
    "SPECTRAL_GATE_VARIANTS",
    "SpectralGateConfig",
    "SpectralGateVariant",
    "axial_energy_features",
    "extract_spectral_features",
    "fit_affine_head",
    "local_covariance_features",
    "product_energy_features",
    "raster_energy_features",
]
