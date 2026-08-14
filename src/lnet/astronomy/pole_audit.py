"""Astronomy pole attribution and independent Lomb-Scargle diagnostics."""
# pyright: reportPrivateUsage=false
# ruff: noqa: SLF001

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from scipy.signal import lombscargle
from scipy.stats import spearmanr
from torch import Tensor

if TYPE_CHECKING:
    from lnet.alphabet import Alphabet
    from lnet.astronomy.plasticc import LightCurve, LightCurveBatch


@dataclass(frozen=True, slots=True)
class ObjectPoleAudit:
    object_id: int
    target: int
    attributed_bank: str
    attributed_mode: int
    attributed_period_days: float
    lomb_scargle_period_days: float


def finite_period_spearman(
    attributed_periods: list[float],
    reference_periods: list[float],
) -> tuple[int, float, float]:
    """Return a finite-pair Spearman statistic and the effective sample count."""
    pairs = [
        (attributed, reference)
        for attributed, reference in zip(
            attributed_periods,
            reference_periods,
            strict=True,
        )
        if np.isfinite(attributed) and np.isfinite(reference)
    ]
    if len(pairs) < 2:
        return len(pairs), float("nan"), float("nan")
    result = spearmanr(
        [pair[0] for pair in pairs],
        [pair[1] for pair in pairs],
    )
    return (
        len(pairs),
        float(np.asarray(result[0], dtype=np.float64).item()),
        float(np.asarray(result[1], dtype=np.float64).item()),
    )


def modal_representations(
    model: Alphabet,
    batch: LightCurveBatch,
) -> tuple[Tensor, Tensor]:
    """Return the exact writer/reader features consumed by the affine head."""
    first_local, active_delta, active_observation, active_valid = model._edge_stem(
        batch.flux,
        batch.time_delta,
        batch.observation_mask,
        batch.valid_mask,
    )
    first_stream, first_moments = model._writer(
        first_local,
        active_delta,
        active_observation,
        active_valid,
    )
    second_moments = model._terminal_reader_moments(
        first_stream,
        active_delta,
        None,
        active_valid,
    )
    writer = model._represent_moments(
        first_moments,
        model.forward_block,
        metadata_free=False,
    )
    reader = model._represent_moments(
        second_moments,
        model.backward_block,
        metadata_free=False,
    )
    return writer, reader


def mode_attribution(
    model: Alphabet,
    writer: Tensor,
    reader: Tensor,
    targets: Tensor,
) -> tuple[Tensor, Tensor]:
    """Rank each bank/mode by absolute additive true-class logit contribution."""
    features = torch.stack((writer, reader), dim=1)
    selected_weights = model.head.classifier.weight[targets]
    weights = selected_weights.reshape(targets.shape[0], 2, 7, model.modes)
    grouped_features = features.reshape(targets.shape[0], 2, 7, model.modes)
    score = (weights * grouped_features).abs().sum(dim=2)
    flat_index = score.reshape(targets.shape[0], -1).argmax(dim=-1)
    return flat_index // model.modes, flat_index % model.modes


def bank_mode_attribution(
    model: Alphabet,
    representation: Tensor,
    targets: Tensor,
    *,
    bank: int,
) -> Tensor:
    """Rank modes within one writer/reader bank by true-class logit contribution."""
    if bank not in (0, 1):
        message = "bank must be 0 (writer) or 1 (reader)"
        raise ValueError(message)
    selected_weights = model.head.classifier.weight[targets]
    weights = selected_weights.reshape(targets.shape[0], 2, 7, model.modes)[:, bank]
    grouped_features = representation.reshape(targets.shape[0], 7, model.modes)
    return (weights * grouped_features).abs().sum(dim=1).argmax(dim=-1)


def pole_period_days(model: Alphabet, banks: Tensor, modes: Tensor) -> Tensor:
    frequencies = torch.stack(
        (model.forward_block.frequency_values(), model.backward_block.frequency_values())
    )
    omega = frequencies[banks, modes].abs()
    return torch.where(
        omega > torch.finfo(omega.dtype).eps,
        2.0 * math.pi / omega,
        torch.full_like(omega, torch.inf),
    )


def lomb_scargle_period_days(
    curve: LightCurve,
    *,
    minimum_period_days: float = 0.05,
    maximum_period_days: float = 10.0,
    frequency_count: int = 4096,
) -> float:
    """Estimate period from the mean normalized per-band periodogram."""
    times = np.cumsum(curve.time_delta, dtype=np.float64)
    angular_frequency = np.geomspace(
        2.0 * math.pi / maximum_period_days,
        2.0 * math.pi / minimum_period_days,
        frequency_count,
    )
    band_powers: list[np.ndarray] = []
    for band in range(curve.flux.shape[1]):
        observed = curve.observation_mask[:, band]
        if observed.sum() < 3:
            continue
        band_flux = curve.flux[observed, band].astype(np.float64)
        centered = band_flux - np.median(band_flux)
        scale = np.median(np.abs(centered))
        if scale > 0.0:
            centered /= scale
        power = lombscargle(
            times[observed],
            centered,
            angular_frequency,
            precenter=False,
            normalize=True,
        )
        maximum = power.max(initial=0.0)
        if maximum > 0.0:
            band_powers.append(power / maximum)
    if not band_powers:
        return float("nan")
    power = np.mean(band_powers, axis=0)
    return float(2.0 * math.pi / angular_frequency[int(np.argmax(power))])
