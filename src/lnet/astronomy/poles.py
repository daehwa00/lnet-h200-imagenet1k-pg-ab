"""Physical pole ranges for light curves measured in days."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from lnet.alphabet import Alphabet


@dataclass(frozen=True, slots=True)
class AstronomyPoleRange:
    damping_min_per_day: float = 1.0 / 3000.0
    damping_max_per_day: float = 1.0 / 0.5
    frequency_max_rad_per_day: float = 2.0 * math.pi / 0.05


def configure_astronomy_poles(
    model: Alphabet,
    pole_range: AstronomyPoleRange | None = None,
) -> Alphabet:
    """Apply physical-day pole bounds and log-spaced damping initialization."""
    if pole_range is None:
        pole_range = AstronomyPoleRange()
    if pole_range.damping_min_per_day <= 0.0:
        message = "minimum damping must be positive"
        raise ValueError(message)
    if pole_range.damping_max_per_day <= pole_range.damping_min_per_day:
        message = "maximum damping must exceed minimum damping"
        raise ValueError(message)
    if pole_range.frequency_max_rad_per_day <= 0.0:
        message = "frequency bound must be positive"
        raise ValueError(message)
    for block in (model.forward_block, model.backward_block):
        block.damping_min = pole_range.damping_min_per_day
        block.damping_max = pole_range.damping_max_per_day
        block.frequency_bound = pole_range.frequency_max_rad_per_day
        target = torch.logspace(
            math.log10(pole_range.damping_min_per_day * 1.01),
            math.log10(pole_range.damping_max_per_day * 0.99),
            block.raw_decay.numel(),
            device=block.raw_decay.device,
            dtype=block.raw_decay.dtype,
        )
        unit = (target - block.damping_min) / (block.damping_max - block.damping_min)
        with torch.no_grad():
            block.raw_decay.copy_(torch.logit(unit.clamp(1.0e-6, 1.0 - 1.0e-6)))
    return model


def configure_astronomy_impulse_poles(
    model: Alphabet,
    pole_range: AstronomyPoleRange | None = None,
    *,
    near_undamped_modes: int = 0,
    near_undamped_alpha_per_day: float = 1.0e-6,
    point_sample_local_convolution: bool = False,
) -> Alphabet:
    """Configure point-sample impulse injection and an optional low-damping sub-bank."""
    model = configure_astronomy_poles(model, pole_range)
    if not 0 <= near_undamped_modes <= model.modes:
        message = "near_undamped_modes must lie between zero and the mode count"
        raise ValueError(message)
    if near_undamped_alpha_per_day < 0.0:
        message = "near-undamped alpha must be non-negative"
        raise ValueError(message)
    model.point_sample_local_convolution = point_sample_local_convolution
    for block in (model.forward_block, model.backward_block):
        block.impulse_injection = True
    model.forward_block.set_fixed_damping_prefix(
        near_undamped_modes,
        near_undamped_alpha_per_day,
    )
    model.backward_block.set_fixed_damping_prefix(0, near_undamped_alpha_per_day)
    if near_undamped_modes:
        maximum_frequency = model.forward_block.frequency_bound
        frequencies = torch.logspace(
            math.log10(2.0 * math.pi / 10.0),
            math.log10(maximum_frequency * 0.999),
            near_undamped_modes,
            device=model.forward_block.raw_frequency.device,
            dtype=model.forward_block.raw_frequency.dtype,
        )
        with torch.no_grad():
            model.forward_block.raw_frequency[:near_undamped_modes].copy_(
                torch.atanh(frequencies / maximum_frequency)
            )
    return model
