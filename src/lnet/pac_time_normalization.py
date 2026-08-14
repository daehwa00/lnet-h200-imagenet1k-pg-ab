"""Leakage-safe nondimensionalization for irregular event times."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def _event_mask(mask: Tensor, shape: torch.Size) -> Tensor:
    if mask.shape[:2] != shape[:2]:
        message = (
            "valid_mask must share the sample and time axes with time_delta: "
            f"{tuple(mask.shape)} versus {tuple(shape)}"
        )
        raise ValueError(message)
    active = mask.bool()
    while active.ndim > 2:
        active = active.any(dim=-1)
    return active


def fit_characteristic_time_scale(
    time_delta: Tensor,
    valid_mask: Tensor | None = None,
    *,
    exclude_first: bool = True,
) -> float:
    """Fit the median positive TRAIN transition without reading held-out data.

    ``time_delta`` may be ``[S,T]`` or ``[S,T,1]``.  With a validity mask, a
    transition is eligible only when both of its endpoints are valid.  The
    initial offset is excluded by default because it is not an
    observation-to-observation interval.
    """
    if time_delta.ndim not in (2, 3) or (
        time_delta.ndim == 3 and time_delta.shape[-1] != 1
    ):
        message = "time_delta must have shape [S,T] or [S,T,1]"
        raise ValueError(message)
    delta = time_delta.squeeze(-1).to(dtype=torch.float64, device="cpu")
    eligible = torch.isfinite(delta) & delta.gt(0)
    if valid_mask is not None:
        event_valid = _event_mask(valid_mask.to(device="cpu"), delta.shape)
        transition_valid = event_valid.clone()
        if delta.shape[1] > 1:
            transition_valid[:, 1:] &= event_valid[:, :-1]
        eligible &= transition_valid
    if exclude_first and delta.shape[1]:
        eligible[:, 0] = False
    selected = delta[eligible]
    if selected.numel() == 0:
        message = "TRAIN split contains no positive valid time transition"
        raise ValueError(message)
    scale = float(torch.quantile(selected, 0.5))
    if not math.isfinite(scale) or scale <= 0:
        message = f"invalid characteristic time scale: {scale}"
        raise ValueError(message)
    return scale


def normalize_time_delta(
    time_delta: Tensor,
    characteristic_time_scale: float,
    valid_mask: Tensor | None = None,
) -> Tensor:
    """Express elapsed time in units of one fitted characteristic interval."""
    scale = float(characteristic_time_scale)
    if not math.isfinite(scale) or scale <= 0:
        message = "characteristic_time_scale must be finite and positive"
        raise ValueError(message)
    normalized = time_delta / scale
    if valid_mask is None:
        return normalized
    event_valid = _event_mask(valid_mask.to(device=time_delta.device), time_delta.shape)
    if time_delta.ndim == 3:
        event_valid = event_valid.unsqueeze(-1)
    return normalized * event_valid.to(dtype=normalized.dtype)


__all__ = ["fit_characteristic_time_scale", "normalize_time_delta"]
