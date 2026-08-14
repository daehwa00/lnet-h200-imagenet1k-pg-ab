"""Analytic bias diagnostics for the physical-time moment approximation."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GapBias:
    damping_per_day: float
    gap_days: float
    lag_days: float
    endpoint_energy_relative_error: float
    interpolated_previous_log10_ratio: float


def exponential_gap_bias(
    damping_per_day: float,
    gap_days: float,
    lag_days: float,
) -> GapBias:
    """Compare endpoint quadrature and linear interpolation to exp(-alpha*t)."""
    if damping_per_day <= 0.0:
        message = "damping_per_day must be positive"
        raise ValueError(message)
    if gap_days <= 0.0:
        message = "gap_days must be positive"
        raise ValueError(message)
    if not 0.0 <= lag_days <= gap_days:
        message = "lag_days must lie within the gap"
        raise ValueError(message)

    exponent = damping_per_day * gap_days
    endpoint_energy = math.exp(-2.0 * exponent)
    exact_mean_energy = -math.expm1(-2.0 * exponent) / (2.0 * exponent)
    endpoint_energy_relative_error = endpoint_energy / exact_mean_energy - 1.0

    fraction_from_end = lag_days / gap_days
    endpoint_state = math.exp(-exponent)
    interpolated_previous = (
        fraction_from_end + (1.0 - fraction_from_end) * endpoint_state
    )
    exact_previous_log = -damping_per_day * (gap_days - lag_days)
    interpolated_previous_log10_ratio = (
        math.log(interpolated_previous) - exact_previous_log
    ) / math.log(10.0)
    return GapBias(
        damping_per_day=damping_per_day,
        gap_days=gap_days,
        lag_days=lag_days,
        endpoint_energy_relative_error=endpoint_energy_relative_error,
        interpolated_previous_log10_ratio=interpolated_previous_log10_ratio,
    )
