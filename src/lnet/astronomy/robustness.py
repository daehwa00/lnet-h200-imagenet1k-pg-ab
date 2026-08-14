"""Evaluation-only light-curve perturbations for Phase-2 experiments."""

from __future__ import annotations

import numpy as np

from lnet.astronomy.plasticc import LightCurve


def truncate_after_days(curve: LightCurve, days: float) -> LightCurve:
    """Keep epochs no later than ``days`` after an object's first observation."""
    if days < 0.0:
        message = "days must be nonnegative"
        raise ValueError(message)
    elapsed = np.cumsum(curve.time_delta, dtype=np.float64)
    length = max(1, int(np.searchsorted(elapsed, days, side="right")))
    return LightCurve(
        object_id=curve.object_id,
        time_delta=curve.time_delta[:length].copy(),
        flux=curve.flux[:length].copy(),
        flux_error=curve.flux_error[:length].copy(),
        observation_mask=curve.observation_mask[:length].copy(),
    )


def insert_seasonal_gap(curve: LightCurve, gap_days: float) -> LightCurve:
    """Insert elapsed time before the middle observed epoch without deleting data."""
    if gap_days < 0.0:
        message = "gap_days must be nonnegative"
        raise ValueError(message)
    time_delta = curve.time_delta.copy()
    if time_delta.size > 1:
        time_delta[time_delta.size // 2] += gap_days
    return LightCurve(
        object_id=curve.object_id,
        time_delta=time_delta,
        flux=curve.flux.copy(),
        flux_error=curve.flux_error.copy(),
        observation_mask=curve.observation_mask.copy(),
    )


def replace_with_unit_intervals(curve: LightCurve) -> LightCurve:
    """Remove physical interval information by setting every event interval to one."""
    return LightCurve(
        object_id=curve.object_id,
        time_delta=np.ones_like(curve.time_delta),
        flux=curve.flux.copy(),
        flux_error=curve.flux_error.copy(),
        observation_mask=curve.observation_mask.copy(),
    )


def interpolate_uniform_grid(
    curve: LightCurve,
    *,
    step_days: float = 1.0,
) -> LightCurve:
    """Linearly interpolate each band on a regular grid within its support."""
    if step_days <= 0.0:
        message = "step_days must be positive"
        raise ValueError(message)
    timestamps = np.cumsum(curve.time_delta, dtype=np.float64)
    stop = np.floor(timestamps[-1] / step_days) * step_days
    grid = np.arange(0.0, stop + 0.5 * step_days, step_days, dtype=np.float64)
    flux = np.zeros((len(grid), curve.flux.shape[1]), dtype=np.float32)
    flux_error = np.zeros_like(flux)
    observation_mask = np.zeros_like(flux, dtype=np.bool_)
    for band in range(curve.flux.shape[1]):
        observed = curve.observation_mask[:, band]
        if observed.sum() < 2:
            continue
        band_time = timestamps[observed]
        supported = (grid >= band_time[0]) & (grid <= band_time[-1])
        flux[supported, band] = np.interp(
            grid[supported],
            band_time,
            curve.flux[observed, band],
        )
        flux_error[supported, band] = np.interp(
            grid[supported],
            band_time,
            curve.flux_error[observed, band],
        )
        observation_mask[supported, band] = True
    return LightCurve(
        object_id=curve.object_id,
        time_delta=np.diff(grid, prepend=grid[0]).astype(np.float32),
        flux=flux,
        flux_error=flux_error,
        observation_mask=observation_mask,
    )
