"""Classical folded-light-curve Fourier shape parameters."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from lnet.astronomy.plasticc import LightCurve
from lnet.astronomy.pole_audit import lomb_scargle_period_days

if TYPE_CHECKING:
    from numpy.typing import NDArray

@dataclass(frozen=True, slots=True)
class FourierShape:
    """Third-order Fourier shape fitted to robustly normalized passbands."""

    period_days: float
    fundamental_amplitude: float
    r21: float
    r31: float
    phi21: float
    phi31: float
    explained_variance: float
    observation_count: int

    def audit_target(self) -> NDArray[np.float64]:
        """Return a Euclidean embedding of ratios and circular phase differences."""
        return np.asarray(
            [
                math.log(self.r21),
                math.log(self.r31),
                math.cos(self.phi21),
                math.sin(self.phi21),
                math.cos(self.phi31),
                math.sin(self.phi31),
            ],
            dtype=np.float64,
        )


def _wrap_phase(value: float) -> float:
    return float((value + math.pi) % (2.0 * math.pi) - math.pi)


def alternating_observation_views(
    curve: LightCurve,
) -> tuple[LightCurve, LightCurve]:
    """Split each passband chronologically into deterministic interleaved views."""
    times = np.cumsum(curve.time_delta, dtype=np.float64)
    view_masks = [
        np.zeros_like(curve.observation_mask, dtype=np.bool_),
        np.zeros_like(curve.observation_mask, dtype=np.bool_),
    ]
    for band in range(curve.flux.shape[1]):
        indices = np.flatnonzero(curve.observation_mask[:, band])
        start = (curve.object_id * 1315423911 + band * 2654435761) & 1
        for offset, index in enumerate(indices):
            view_masks[(start + offset) & 1][index, band] = True
    views: list[LightCurve] = []
    for mask in view_masks:
        retained = mask.any(axis=1)
        retained_times = times[retained]
        flux = np.where(mask[retained], curve.flux[retained], 0.0).astype(np.float32)
        flux_error = np.where(
            mask[retained],
            curve.flux_error[retained],
            0.0,
        ).astype(np.float32)
        views.append(
            LightCurve(
                object_id=curve.object_id,
                time_delta=np.diff(
                    retained_times,
                    prepend=retained_times[0],
                ).astype(np.float32),
                flux=flux,
                flux_error=flux_error,
                observation_mask=mask[retained],
            )
        )
    return views[0], views[1]


def _normalized_observations(
    curve: LightCurve,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    times = np.cumsum(curve.time_delta, dtype=np.float64)
    selected_times: list[NDArray[np.float64]] = []
    selected_flux: list[NDArray[np.float64]] = []
    selected_bands: list[NDArray[np.int64]] = []
    for band in range(curve.flux.shape[1]):
        observed = curve.observation_mask[:, band]
        if int(observed.sum()) < 7:
            continue
        values = curve.flux[observed, band].astype(np.float64)
        centered = values - np.median(values)
        scale = float(np.sqrt(np.mean(np.square(centered))))
        if not math.isfinite(scale) or scale <= np.finfo(np.float64).eps:
            continue
        selected_times.append(times[observed])
        selected_flux.append(centered / scale)
        selected_bands.append(np.full(int(observed.sum()), band, dtype=np.int64))
    if not selected_times:
        empty_float = np.empty(0, dtype=np.float64)
        return empty_float, empty_float, np.empty(0, dtype=np.int64)
    return (
        np.concatenate(selected_times),
        np.concatenate(selected_flux),
        np.concatenate(selected_bands),
    )


def fit_fourier_shape(
    curve: LightCurve,
    period_days: float,
    *,
    harmonic_count: int = 3,
) -> FourierShape | None:
    """Fit a shared multiband Fourier shape with per-band floating means."""
    if not math.isfinite(period_days) or period_days <= 0.0:
        return None
    times, values, bands = _normalized_observations(curve)
    if values.size < 2 * harmonic_count + 7:
        return None
    phase = 2.0 * math.pi * times / period_days
    band_columns = np.eye(curve.flux.shape[1], dtype=np.float64)[bands]
    harmonic_columns = [
        function(harmonic * phase)
        for harmonic in range(1, harmonic_count + 1)
        for function in (np.cos, np.sin)
    ]
    design = np.column_stack((band_columns, *harmonic_columns))
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    fitted = design @ coefficients
    total_sum_squares = float(np.sum(np.square(values - values.mean())))
    residual_sum_squares = float(np.sum(np.square(values - fitted)))
    explained_variance = (
        1.0 - residual_sum_squares / total_sum_squares
        if total_sum_squares > np.finfo(np.float64).eps
        else float("nan")
    )
    harmonic = coefficients[curve.flux.shape[1] :].reshape(harmonic_count, 2)
    amplitudes = np.linalg.norm(harmonic, axis=1)
    if amplitudes[0] <= 1.0e-6 or np.any(amplitudes[:3] <= 0.0):
        return None
    phases = np.arctan2(-harmonic[:, 1], harmonic[:, 0])
    return FourierShape(
        period_days=period_days,
        fundamental_amplitude=float(amplitudes[0]),
        r21=float(amplitudes[1] / amplitudes[0]),
        r31=float(amplitudes[2] / amplitudes[0]),
        phi21=_wrap_phase(float(phases[1] - 2.0 * phases[0])),
        phi31=_wrap_phase(float(phases[2] - 3.0 * phases[0])),
        explained_variance=explained_variance,
        observation_count=int(values.size),
    )


def estimate_fourier_shape(
    curve: LightCurve,
    *,
    minimum_period_days: float = 0.05,
    maximum_period_days: float = 10.0,
) -> FourierShape | None:
    """Estimate a label-free period and choose between its one- and two-cycle fits."""
    period = lomb_scargle_period_days(
        curve,
        minimum_period_days=minimum_period_days,
        maximum_period_days=maximum_period_days,
    )
    candidates = [
        fit_fourier_shape(curve, candidate)
        for candidate in (0.5 * period, period, 2.0 * period)
        if minimum_period_days <= candidate <= 2.0 * maximum_period_days
    ]
    valid = [candidate for candidate in candidates if candidate is not None]
    return max(valid, key=lambda candidate: candidate.explained_variance, default=None)


def is_reliable_fourier_shape(shape: FourierShape | None) -> bool:
    """Apply the preregistered quality filter used by the descriptor audit."""
    return (
        shape is not None
        and shape.observation_count >= 30
        and shape.explained_variance >= 0.10
        and 1.0e-3 <= shape.r21 <= 1.0e3
        and 1.0e-3 <= shape.r31 <= 1.0e3
    )
