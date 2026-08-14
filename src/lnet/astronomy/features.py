"""Compact broker-style statistical features for a feature-RF incumbent."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from lnet.astronomy.plasticc import PASSBAND_COUNT

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from lnet.astronomy.plasticc import LightCurve

FEATURES_PER_BAND = 8


def broker_features(curve: LightCurve) -> NDArray[np.float32]:
    """Extract fixed-size signed-flux and cadence summaries without fitting state."""
    values: list[float] = []
    elapsed = np.cumsum(curve.time_delta, dtype=np.float64)
    for band in range(PASSBAND_COUNT):
        observed = curve.observation_mask[:, band]
        flux = curve.flux[observed, band].astype(np.float64)
        if flux.size == 0:
            values.extend((0.0,) * FEATURES_PER_BAND)
            continue
        median = float(np.median(flux))
        values.extend(
            (
                float(flux.size),
                float(np.mean(flux)),
                float(np.std(flux)),
                float(np.min(flux)),
                float(np.max(flux)),
                median,
                float(np.median(np.abs(flux - median))),
                float(np.max(flux) - np.min(flux)),
            )
        )
    positive_delta = curve.time_delta[curve.time_delta > 0]
    values.extend(
        (
            float(curve.flux.shape[0]),
            float(elapsed[-1]) if elapsed.size else 0.0,
            float(np.mean(positive_delta)) if positive_delta.size else 0.0,
            float(np.std(positive_delta)) if positive_delta.size else 0.0,
            float(np.max(positive_delta)) if positive_delta.size else 0.0,
        )
    )
    return np.nan_to_num(
        np.asarray(values, dtype=np.float32),
        nan=0.0,
        posinf=float(np.finfo(np.float32).max),
        neginf=float(np.finfo(np.float32).min),
    )
