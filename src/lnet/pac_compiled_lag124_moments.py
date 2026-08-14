from __future__ import annotations

from typing import Final, Literal, assert_never

import torch
from torch import Tensor

Direction = Literal["forward", "backward"]

_EPSILON: Final[float] = 1.0e-8
_LAGS: Final[tuple[int, int, int]] = (1, 2, 4)
_LAST_DISPATCH = "uninitialized"


def _physical_lag124_moments(
    states_real: Tensor,
    states_imag: Tensor,
    epsilon: float,
) -> Tensor:
    energy = (states_real.square() + states_imag.square()).mean(dim=1)
    moments = [torch.log1p(energy)]
    for lag in _LAGS:
        current_real = states_real[:, lag:]
        current_imag = states_imag[:, lag:]
        previous_real = states_real[:, :-lag]
        previous_imag = states_imag[:, :-lag]
        correlation_real = (current_real * previous_real + current_imag * previous_imag).mean(dim=1)
        correlation_imag = (current_imag * previous_real - current_real * previous_imag).mean(dim=1)
        current_energy = (current_real.square() + current_imag.square()).mean(dim=1)
        previous_energy = (previous_real.square() + previous_imag.square()).mean(dim=1)
        denominator = torch.sqrt(
            (current_energy * previous_energy).clamp_min(epsilon * epsilon)
        )
        moments.extend((correlation_real / denominator, correlation_imag / denominator))
    return torch.cat(moments, dim=-1)


_compiled_physical_lag124_moments = torch.compile(
    _physical_lag124_moments,
    fullgraph=True,
    dynamic=True,
    options={"triton.cudagraphs": False},
)


def reference_lag124_modal_moments(
    states_real: Tensor,
    states_imag: Tensor,
    *,
    physical_direction: Direction = "forward",
    epsilon: float = _EPSILON,
) -> Tensor:
    _validate_inputs(states_real, states_imag, epsilon)
    if states_real.shape[1] <= max(_LAGS):
        return _short_sequence_reference(
            states_real,
            states_imag,
            physical_direction=physical_direction,
            epsilon=epsilon,
        )
    real, imag = _orient(states_real, states_imag, physical_direction)
    return _physical_lag124_moments(real, imag, epsilon)


def lag124_modal_moments(
    states_real: Tensor,
    states_imag: Tensor,
    *,
    physical_direction: Direction = "forward",
    epsilon: float = _EPSILON,
    force_reference: bool = False,
) -> Tensor:
    global _LAST_DISPATCH  # noqa: PLW0603
    _validate_inputs(states_real, states_imag, epsilon)
    supported = (
        states_real.is_cuda
        and states_real.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and states_real.shape[1] > max(_LAGS)
    )
    if force_reference or not supported:
        _LAST_DISPATCH = "reference"
        return reference_lag124_modal_moments(
            states_real,
            states_imag,
            physical_direction=physical_direction,
            epsilon=epsilon,
        )
    real, imag = _orient(states_real, states_imag, physical_direction)
    _LAST_DISPATCH = "inductor_triton_fused_forward_backward"
    return _compiled_physical_lag124_moments(real, imag, epsilon)


def last_lag124_moments_dispatch() -> str:
    return _LAST_DISPATCH


def _short_sequence_reference(
    states_real: Tensor,
    states_imag: Tensor,
    *,
    physical_direction: Direction,
    epsilon: float,
) -> Tensor:
    real, imag = _orient(states_real, states_imag, physical_direction)
    energy = (real.square() + imag.square()).mean(dim=1)
    moments = [torch.log1p(energy)]
    for lag in _LAGS:
        if real.shape[1] <= lag:
            zeros = real.new_zeros((real.shape[0], real.shape[2]))
            moments.extend((zeros, zeros))
            continue
        current_real = real[:, lag:]
        current_imag = imag[:, lag:]
        previous_real = real[:, :-lag]
        previous_imag = imag[:, :-lag]
        correlation_real = (current_real * previous_real + current_imag * previous_imag).mean(dim=1)
        correlation_imag = (current_imag * previous_real - current_real * previous_imag).mean(dim=1)
        current_energy = (current_real.square() + current_imag.square()).mean(dim=1)
        previous_energy = (previous_real.square() + previous_imag.square()).mean(dim=1)
        denominator = torch.sqrt(
            (current_energy * previous_energy).clamp_min(epsilon * epsilon)
        )
        moments.extend((correlation_real / denominator, correlation_imag / denominator))
    return torch.cat(moments, dim=-1)


def _orient(
    states_real: Tensor,
    states_imag: Tensor,
    direction: Direction,
) -> tuple[Tensor, Tensor]:
    match direction:
        case "forward":
            return states_real, states_imag
        case "backward":
            return torch.flip(states_real, (1,)), torch.flip(states_imag, (1,))
        case unreachable:
            assert_never(unreachable)


def _validate_inputs(states_real: Tensor, states_imag: Tensor, epsilon: float) -> None:
    if states_real.ndim != 3 or states_real.shape != states_imag.shape:
        message = "real and imaginary modal states must share [batch, time, modes] shape"
        raise ValueError(message)
    if states_real.device != states_imag.device or states_real.dtype != states_imag.dtype:
        message = "real and imaginary modal states must share device and dtype"
        raise ValueError(message)
    if (
        not states_real.is_floating_point()
        or states_real.shape[1] == 0
        or states_real.shape[2] == 0
    ):
        message = "modal states must be non-empty floating-point tensors"
        raise ValueError(message)
    if epsilon <= 0.0:
        message = "epsilon must be positive"
        raise ValueError(message)


__all__ = [
    "lag124_modal_moments",
    "last_lag124_moments_dispatch",
    "reference_lag124_modal_moments",
]
