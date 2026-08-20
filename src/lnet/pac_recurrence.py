from __future__ import annotations

from collections.abc import Callable
from typing import Literal, assert_never

import torch
from torch import Tensor

from .pac_triton_recurrence_op import (
    pac_triton_recurrence_op,
    pac_triton_recurrence_opaque_op,
    pac_triton_state_variance_recurrence_op,
)
from .pac_triton_scan import triton_scan_blocks_recurrence

_TRITON_FUSED_AVAILABLE = True

RecurrenceBackend = Literal[
    "complex_loop",
    "real2d_loop",
    "compiled_real2d",
    "triton_fused",
    "triton_fused_opaque",
    "triton_scan",
    "real2d_e2e",
    "triton_scan_blocks",
    "triton_modal_fused",
    "triton_modal_reduce",
    "triton_modal_reduce_recompute",
    "pac_lite_fast",
    "fixed_real2d_fast",
    "fused_pole_gamma",
    "auto",
]
RecurrenceDirection = Literal["forward", "backward"]
CompiledRecurrence = Callable[[Tensor, Tensor, Tensor, Tensor], tuple[Tensor, Tensor]]
_compiled_real2d: CompiledRecurrence | None = None


def recurrence_states(decay: Tensor, input_term: Tensor, backend: RecurrenceBackend) -> Tensor:
    real, imag = recurrence_real2d(
        decay.real,
        decay.imag,
        input_term.real,
        input_term.imag,
        backend,
    )
    return torch.complex(real, imag)


def recurrence_real2d(
    decay_real: Tensor,
    decay_imag: Tensor,
    input_real: Tensor,
    input_imag: Tensor,
    backend: RecurrenceBackend,
) -> tuple[Tensor, Tensor]:
    selected = _select_backend(decay_real, backend)
    match selected:
        case "complex_loop":
            states = _complex_loop_recurrence(
                torch.complex(decay_real, decay_imag),
                torch.complex(input_real, input_imag),
            )
            return states.real, states.imag
        case "real2d_loop" | "real2d_e2e":
            return real2d_loop_recurrence(decay_real, decay_imag, input_real, input_imag)
        case "compiled_real2d":
            return compiled_real2d_recurrence(decay_real, decay_imag, input_real, input_imag)
        case (
            "triton_fused"
            | "triton_fused_opaque"
            | "pac_lite_fast"
            | "triton_modal_fused"
            | "triton_modal_reduce"
            | "triton_modal_reduce_recompute"
            | "fixed_real2d_fast"
            | "fused_pole_gamma"
        ):
            return _triton_fused_recurrence(
                decay_real,
                decay_imag,
                input_real,
                input_imag,
                opaque=selected == "triton_fused_opaque",
            )
        case "triton_scan":
            return associative_scan_recurrence(
                decay_real,
                decay_imag,
                input_real,
                input_imag,
            )
        case "triton_scan_blocks":
            return _triton_scan_blocks_recurrence(
                decay_real,
                decay_imag,
                input_real,
                input_imag,
            )
        case "auto":
            message = "auto backend must be resolved before dispatch"
            raise RuntimeError(message)
        case unreachable:
            assert_never(unreachable)


def recurrence_real2d_directional(
    decay_real: Tensor,
    decay_imag: Tensor,
    input_real: Tensor,
    input_imag: Tensor,
    backend: RecurrenceBackend,
    direction: RecurrenceDirection,
) -> tuple[Tensor, Tensor]:
    selected = _select_backend(decay_real, backend)
    if direction == "forward":
        return recurrence_real2d(decay_real, decay_imag, input_real, input_imag, selected)
    if (
        selected in {"triton_fused", "triton_fused_opaque"}
        and decay_real.is_cuda
        and triton_fused_available()
    ):
        operation = (
            pac_triton_recurrence_opaque_op
            if selected == "triton_fused_opaque"
            else pac_triton_recurrence_op
        )
        return operation(
            decay_real,
            decay_imag,
            input_real,
            input_imag,
            reverse=True,
        )
    oriented = (
        torch.flip(decay_real, dims=(1,)),
        torch.flip(decay_imag, dims=(1,)),
        torch.flip(input_real, dims=(1,)),
        torch.flip(input_imag, dims=(1,)),
    )
    states_real, states_imag = recurrence_real2d(*oriented, selected)
    return torch.flip(states_real, dims=(1,)), torch.flip(states_imag, dims=(1,))


def recurrence_real2d_state_variance_directional(
    decay_real: Tensor,
    decay_imag: Tensor,
    input_real: Tensor,
    input_imag: Tensor,
    variance_decay: Tensor,
    variance_input: Tensor,
    backend: RecurrenceBackend,
    direction: RecurrenceDirection,
) -> tuple[Tensor, Tensor, Tensor]:
    """Fuse state and detached variance transport when Triton is available."""
    selected = _select_backend(decay_real, backend)
    if (
        selected in {"triton_fused", "triton_fused_opaque"}
        and decay_real.is_cuda
        and triton_fused_available()
    ):
        return pac_triton_state_variance_recurrence_op(
            decay_real,
            decay_imag,
            input_real,
            input_imag,
            variance_decay.detach(),
            variance_input.detach(),
            reverse=direction == "backward",
        )
    state_real, state_imag = recurrence_real2d_directional(
        decay_real,
        decay_imag,
        input_real,
        input_imag,
        selected,
        direction,
    )
    variance, _ = recurrence_real2d_directional(
        variance_decay,
        torch.zeros_like(variance_decay),
        variance_input,
        torch.zeros_like(variance_input),
        selected,
        direction,
    )
    return state_real, state_imag, variance


def real2d_loop_recurrence(
    decay_real: Tensor,
    decay_imag: Tensor,
    input_real: Tensor,
    input_imag: Tensor,
) -> tuple[Tensor, Tensor]:
    states_real = torch.empty_like(input_real)
    states_imag = torch.empty_like(input_imag)
    state_real = torch.zeros(
        input_real.shape[0], input_real.shape[2], dtype=input_real.dtype, device=input_real.device
    )
    state_imag = torch.zeros_like(state_real)
    for time_index in range(input_real.shape[1]):
        current_real = state_real
        current_imag = state_imag
        ar = decay_real[:, time_index, :]
        ai = decay_imag[:, time_index, :]
        state_real = ar * current_real - ai * current_imag + input_real[:, time_index, :]
        state_imag = ai * current_real + ar * current_imag + input_imag[:, time_index, :]
        states_real[:, time_index, :] = state_real
        states_imag[:, time_index, :] = state_imag
    return states_real, states_imag


def compiled_real2d_recurrence(
    decay_real: Tensor,
    decay_imag: Tensor,
    input_real: Tensor,
    input_imag: Tensor,
) -> tuple[Tensor, Tensor]:
    global _compiled_real2d
    if _compiled_real2d is None:
        _compiled_real2d = torch.compile(real2d_loop_recurrence)
    return _compiled_real2d(decay_real, decay_imag, input_real, input_imag)


def associative_scan_recurrence(
    decay_real: Tensor,
    decay_imag: Tensor,
    input_real: Tensor,
    input_imag: Tensor,
) -> tuple[Tensor, Tensor]:
    transform = torch.complex(decay_real, decay_imag)
    shift = torch.complex(input_real, input_imag)
    step = 1
    while step < transform.shape[1]:
        next_transform = transform.clone()
        next_shift = shift.clone()
        current_transform = transform[:, step:, :]
        next_transform[:, step:, :] = current_transform * transform[:, :-step, :]
        next_shift[:, step:, :] = current_transform * shift[:, :-step, :] + shift[:, step:, :]
        transform = next_transform
        shift = next_shift
        step *= 2
    return shift.real, shift.imag


def _complex_loop_recurrence(decay: Tensor, input_term: Tensor) -> Tensor:
    states = torch.empty_like(input_term)
    state = torch.zeros(
        input_term.shape[0],
        input_term.shape[2],
        dtype=input_term.dtype,
        device=input_term.device,
    )
    for time_index in range(input_term.shape[1]):
        state = decay[:, time_index, :] * state + input_term[:, time_index, :]
        states[:, time_index, :] = state
    return states


def _select_backend(decay: Tensor, backend: RecurrenceBackend) -> RecurrenceBackend:
    if backend != "auto":
        return backend
    if decay.is_cuda and triton_fused_available():
        return "triton_fused"
    return "real2d_loop"


def _triton_fused_recurrence(
    decay_real: Tensor,
    decay_imag: Tensor,
    input_real: Tensor,
    input_imag: Tensor,
    *,
    opaque: bool = False,
) -> tuple[Tensor, Tensor]:
    if not decay_real.is_cuda or not triton_fused_available():
        return real2d_loop_recurrence(decay_real, decay_imag, input_real, input_imag)
    operation = pac_triton_recurrence_opaque_op if opaque else pac_triton_recurrence_op
    return operation(decay_real, decay_imag, input_real, input_imag)


def _triton_scan_blocks_recurrence(
    decay_real: Tensor,
    decay_imag: Tensor,
    input_real: Tensor,
    input_imag: Tensor,
) -> tuple[Tensor, Tensor]:
    if not decay_real.is_cuda or not triton_fused_available():
        return real2d_loop_recurrence(decay_real, decay_imag, input_real, input_imag)
    return triton_scan_blocks_recurrence(decay_real, decay_imag, input_real, input_imag)


def triton_fused_available() -> bool:
    # Triton-backed modules are imported above, so reaching this module already proves
    # availability. Keep dispatch static so torch.compile can trace the auto path.
    return _TRITON_FUSED_AVAILABLE
