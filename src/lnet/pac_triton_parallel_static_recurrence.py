"""Time-parallel static complex recurrence for throughput-oriented training.

The exact training backend assigns one program to several modes and evaluates
time serially.  This module exposes the complementary high-occupancy lane: one
program owns one ``(batch, mode)`` pair and evaluates the affine recurrence with
``tl.associative_scan`` across a power-of-two time tile.

Parallel affine composition changes floating-point association.  The API is
therefore deliberately opt-in and accepts only a mode-static FP32 pole.  Its
custom backward uses the same parallel scan for the reverse adjoint and a
separate deterministic batch reduction for the static-pole gradient.
"""

from __future__ import annotations

# pyright: reportCallIssue=false, reportMissingParameterType=false
# ruff: noqa: ANN001, ANN202, FBT001, N803
from typing import Final, Protocol

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton

_MAX_STEPS: Final[int] = 1024
_VALID_WARPS: Final[tuple[int, ...]] = (4, 8)


class _AutogradContext(Protocol):
    saved_tensors: tuple[Tensor, ...]
    reverse: bool
    num_warps: int

    def save_for_backward(self, *tensors: Tensor) -> None: ...


@triton.jit
def _compose_complex_affine(
    left_ar,
    left_ai,
    left_br,
    left_bi,
    right_ar,
    right_ai,
    right_br,
    right_bi,
):
    """Compose adjacent transforms as ``right(left(state))``."""
    product_ar = right_ar * left_ar - right_ai * left_ai
    product_ai = right_ai * left_ar + right_ar * left_ai
    shift_br = right_ar * left_br - right_ai * left_bi + right_br
    shift_bi = right_ai * left_br + right_ar * left_bi + right_bi
    return product_ar, product_ai, shift_br, shift_bi


@triton.jit
def _parallel_static_forward_kernel(
    decay_real,
    decay_imag,
    packed_input,
    packed_states,
    n_steps: tl.constexpr,
    modes: tl.constexpr,
    reverse: tl.constexpr,
    BLOCK_T: tl.constexpr,
) -> None:
    lane = tl.program_id(0)
    batch = lane // modes
    mode = lane - batch * modes
    traversal = tl.arange(0, BLOCK_T)
    active = traversal < n_steps
    time_index = n_steps - 1 - traversal if reverse else traversal
    offset = (batch * n_steps + time_index) * 2 * modes + mode

    fixed_ar = tl.load(decay_real + mode).to(tl.float32)
    fixed_ai = tl.load(decay_imag + mode).to(tl.float32)
    # The inactive power-of-two tail is an identity transform.  It is never
    # stored and cannot perturb any active prefix.
    ar = tl.where(active, fixed_ar, 1.0)
    ai = tl.where(active, fixed_ai, 0.0)
    br = tl.load(packed_input + offset, mask=active, other=0.0).to(tl.float32)
    bi = tl.load(
        packed_input + offset + modes,
        mask=active,
        other=0.0,
    ).to(tl.float32)
    _prefix_ar, _prefix_ai, state_real, state_imag = tl.associative_scan(
        (ar, ai, br, bi),
        axis=0,
        combine_fn=_compose_complex_affine,
    )
    tl.store(packed_states + offset, state_real, mask=active)
    tl.store(packed_states + offset + modes, state_imag, mask=active)


@triton.jit
def _parallel_static_backward_kernel(
    decay_real,
    decay_imag,
    packed_states,
    grad_packed_states,
    grad_packed_input,
    grad_decay_per_batch,
    n_steps: tl.constexpr,
    modes: tl.constexpr,
    reverse: tl.constexpr,
    BLOCK_T: tl.constexpr,
) -> None:
    lane = tl.program_id(0)
    batch = lane // modes
    mode = lane - batch * modes
    traversal = tl.arange(0, BLOCK_T)
    active = traversal < n_steps
    # The recurrence adjoint traverses the opposite physical direction.
    time_index = traversal if reverse else n_steps - 1 - traversal
    offset = (batch * n_steps + time_index) * 2 * modes + mode

    fixed_ar = tl.load(decay_real + mode).to(tl.float32)
    fixed_ai = tl.load(decay_imag + mode).to(tl.float32)
    ar = tl.where(active, fixed_ar, 1.0)
    ai = tl.where(active, -fixed_ai, 0.0)
    gradient_real = tl.load(
        grad_packed_states + offset,
        mask=active,
        other=0.0,
    ).to(tl.float32)
    gradient_imag = tl.load(
        grad_packed_states + offset + modes,
        mask=active,
        other=0.0,
    ).to(tl.float32)
    _prefix_ar, _prefix_ai, lambda_real, lambda_imag = tl.associative_scan(
        (ar, ai, gradient_real, gradient_imag),
        axis=0,
        combine_fn=_compose_complex_affine,
    )
    tl.store(grad_packed_input + offset, lambda_real, mask=active)
    tl.store(grad_packed_input + offset + modes, lambda_imag, mask=active)

    previous_index = time_index + 1 if reverse else time_index - 1
    has_previous = active & (time_index < n_steps - 1 if reverse else time_index > 0)
    previous_offset = (batch * n_steps + previous_index) * 2 * modes + mode
    previous_real = tl.load(
        packed_states + previous_offset,
        mask=has_previous,
        other=0.0,
    ).to(tl.float32)
    previous_imag = tl.load(
        packed_states + previous_offset + modes,
        mask=has_previous,
        other=0.0,
    ).to(tl.float32)
    decay_gradient_real = tl.where(
        active,
        lambda_real * previous_real + lambda_imag * previous_imag,
        0.0,
    )
    decay_gradient_imag = tl.where(
        active,
        -lambda_real * previous_imag + lambda_imag * previous_real,
        0.0,
    )
    summary_offset = batch * 2 * modes + mode
    tl.store(grad_decay_per_batch + summary_offset, tl.sum(decay_gradient_real))
    tl.store(
        grad_decay_per_batch + summary_offset + modes,
        tl.sum(decay_gradient_imag),
    )


@triton.jit
def _reduce_batch_gradient_kernel(
    grad_decay_per_batch,
    grad_decay_real,
    grad_decay_imag,
    batch_size: tl.constexpr,
    modes: tl.constexpr,
    BLOCK_M: tl.constexpr,
) -> None:
    mode = tl.arange(0, BLOCK_M)
    valid = mode < modes
    total_real = tl.zeros((BLOCK_M,), tl.float32)
    total_imag = tl.zeros((BLOCK_M,), tl.float32)
    for batch in range(batch_size):
        offset = batch * 2 * modes + mode
        total_real += tl.load(
            grad_decay_per_batch + offset,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        total_imag += tl.load(
            grad_decay_per_batch + offset + modes,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
    tl.store(grad_decay_real + mode, total_real, mask=valid)
    tl.store(grad_decay_imag + mode, total_imag, mask=valid)


def _validate_inputs(decay_real: Tensor, decay_imag: Tensor, packed_input: Tensor) -> None:
    if decay_real.ndim != 1 or decay_real.numel() == 0:
        message = "parallel static recurrence decay must have shape [modes]"
        raise ValueError(message)
    if decay_imag.shape != decay_real.shape:
        message = "parallel static recurrence decay tensors must match"
        raise ValueError(message)
    if (
        packed_input.ndim != 3
        or packed_input.shape[1] == 0
        or packed_input.shape[-1] != 2 * decay_real.numel()
    ):
        message = "packed input must have shape [batch, steps, 2*modes]"
        raise ValueError(message)
    if packed_input.shape[1] > _MAX_STEPS:
        message = f"parallel static recurrence supports at most {_MAX_STEPS} steps"
        raise ValueError(message)
    tensors = (decay_real, decay_imag, packed_input)
    if any(tensor.dtype != torch.float32 for tensor in tensors):
        message = "parallel static recurrence supports exact FP32 tensors only"
        raise TypeError(message)
    if any(tensor.device != packed_input.device for tensor in tensors):
        message = "parallel static recurrence tensors must share one device"
        raise ValueError(message)


def _reference(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    reverse: bool,
) -> Tensor:
    modes = decay_real.numel()
    input_real, input_imag = packed_input.split(modes, dim=-1)
    state_real = torch.zeros_like(input_real[:, 0])
    state_imag = torch.zeros_like(input_imag[:, 0])
    states_real: list[Tensor | None] = [None] * packed_input.shape[1]
    states_imag: list[Tensor | None] = [None] * packed_input.shape[1]
    traversal = (
        range(packed_input.shape[1] - 1, -1, -1) if reverse else range(packed_input.shape[1])
    )
    for time_index in traversal:
        previous_real = state_real
        previous_imag = state_imag
        state_real = (
            decay_real * previous_real - decay_imag * previous_imag + input_real[:, time_index]
        )
        state_imag = (
            decay_imag * previous_real + decay_real * previous_imag + input_imag[:, time_index]
        )
        states_real[time_index] = state_real
        states_imag[time_index] = state_imag
    real = torch.stack([value for value in states_real if value is not None], dim=1)
    imag = torch.stack([value for value in states_imag if value is not None], dim=1)
    return torch.cat((real, imag), dim=-1)


def _reference_backward(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_states: Tensor,
    grad_packed_states: Tensor,
    reverse: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    modes = decay_real.numel()
    states_real, states_imag = packed_states.split(modes, dim=-1)
    grad_states_real, grad_states_imag = grad_packed_states.split(modes, dim=-1)
    lambda_real = torch.zeros_like(states_real[:, 0])
    lambda_imag = torch.zeros_like(states_imag[:, 0])
    grad_input_real: list[Tensor | None] = [None] * packed_states.shape[1]
    grad_input_imag: list[Tensor | None] = [None] * packed_states.shape[1]
    grad_decay_real = torch.zeros_like(decay_real)
    grad_decay_imag = torch.zeros_like(decay_imag)
    traversal = (
        range(packed_states.shape[1]) if reverse else range(packed_states.shape[1] - 1, -1, -1)
    )
    for time_index in traversal:
        lambda_real = lambda_real + grad_states_real[:, time_index]
        lambda_imag = lambda_imag + grad_states_imag[:, time_index]
        previous_index = time_index + 1 if reverse else time_index - 1
        has_previous = time_index < packed_states.shape[1] - 1 if reverse else time_index > 0
        if has_previous:
            previous_real = states_real[:, previous_index]
            previous_imag = states_imag[:, previous_index]
        else:
            previous_real = torch.zeros_like(lambda_real)
            previous_imag = torch.zeros_like(lambda_imag)
        grad_input_real[time_index] = lambda_real
        grad_input_imag[time_index] = lambda_imag
        grad_decay_real = grad_decay_real + (
            lambda_real * previous_real + lambda_imag * previous_imag
        ).sum(dim=0)
        grad_decay_imag = grad_decay_imag + (
            -lambda_real * previous_imag + lambda_imag * previous_real
        ).sum(dim=0)
        next_lambda_real = decay_real * lambda_real + decay_imag * lambda_imag
        next_lambda_imag = -decay_imag * lambda_real + decay_real * lambda_imag
        lambda_real = next_lambda_real
        lambda_imag = next_lambda_imag
    packed_input_gradient = torch.cat(
        (
            torch.stack(
                [value for value in grad_input_real if value is not None],
                dim=1,
            ),
            torch.stack(
                [value for value in grad_input_imag if value is not None],
                dim=1,
            ),
        ),
        dim=-1,
    )
    return grad_decay_real, grad_decay_imag, packed_input_gradient


@triton_op("lnet::pac_parallel_static_recurrence_impl", mutates_args={})
def _parallel_forward_impl(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    reverse: bool,
    num_warps: int,
) -> Tensor:
    _validate_inputs(decay_real, decay_imag, packed_input)
    if num_warps not in _VALID_WARPS:
        message = f"num_warps must be one of {_VALID_WARPS}"
        raise ValueError(message)
    if not packed_input.is_cuda:
        return _reference(decay_real, decay_imag, packed_input, reverse)
    drive = packed_input.contiguous()
    states = torch.empty_like(drive)
    batch, n_steps, packed_modes = drive.shape
    modes = packed_modes // 2
    block_t = triton.next_power_of_2(n_steps)
    wrap_triton(_parallel_static_forward_kernel)[(batch * modes,)](
        decay_real.contiguous(),
        decay_imag.contiguous(),
        drive,
        states,
        n_steps,
        modes,
        reverse,
        BLOCK_T=block_t,
        num_warps=num_warps,
    )
    return states


@triton_op("lnet::pac_parallel_static_recurrence_backward_impl", mutates_args={})
def _parallel_backward_impl(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_states: Tensor,
    grad_packed_states: Tensor,
    reverse: bool,
    num_warps: int,
) -> tuple[Tensor, Tensor, Tensor]:
    _validate_inputs(decay_real, decay_imag, packed_states)
    if grad_packed_states.shape != packed_states.shape:
        message = "parallel recurrence output gradient must match states"
        raise ValueError(message)
    if num_warps not in _VALID_WARPS:
        message = f"num_warps must be one of {_VALID_WARPS}"
        raise ValueError(message)
    if not packed_states.is_cuda:
        return _reference_backward(
            decay_real,
            decay_imag,
            packed_states,
            grad_packed_states,
            reverse,
        )
    states = packed_states.contiguous()
    output_gradient = grad_packed_states.contiguous()
    batch, n_steps, packed_modes = states.shape
    modes = packed_modes // 2
    block_t = triton.next_power_of_2(n_steps)
    grad_input = torch.empty_like(states)
    per_batch = torch.empty(
        (batch, 2 * modes),
        dtype=torch.float32,
        device=states.device,
    )
    grad_decay_real = torch.empty_like(decay_real)
    grad_decay_imag = torch.empty_like(decay_imag)
    wrap_triton(_parallel_static_backward_kernel)[(batch * modes,)](
        decay_real.contiguous(),
        decay_imag.contiguous(),
        states,
        output_gradient,
        grad_input,
        per_batch,
        n_steps,
        modes,
        reverse,
        BLOCK_T=block_t,
        num_warps=num_warps,
    )
    block_m = triton.next_power_of_2(modes)
    wrap_triton(_reduce_batch_gradient_kernel)[(1,)](
        per_batch,
        grad_decay_real,
        grad_decay_imag,
        batch,
        modes,
        BLOCK_M=block_m,
        num_warps=1,
    )
    return grad_decay_real, grad_decay_imag, grad_input


@torch.library.custom_op(
    "lnet::pac_parallel_static_recurrence_backward",
    mutates_args=(),
)
def _parallel_backward_opaque(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_states: Tensor,
    grad_packed_states: Tensor,
    reverse: bool,
    num_warps: int,
) -> tuple[Tensor, Tensor, Tensor]:
    return _parallel_backward_impl(
        decay_real,
        decay_imag,
        packed_states,
        grad_packed_states,
        reverse,
        num_warps,
    )


@_parallel_backward_opaque.register_fake
def _parallel_backward_fake(  # pyright: ignore[reportUnusedFunction]
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_states: Tensor,
    grad_packed_states: Tensor,
    reverse: bool,
    num_warps: int,
) -> tuple[Tensor, Tensor, Tensor]:
    del packed_states, reverse, num_warps
    return (
        torch.empty_like(decay_real),
        torch.empty_like(decay_imag),
        torch.empty_like(grad_packed_states),
    )


@torch.library.custom_op("lnet::pac_parallel_static_recurrence", mutates_args=())
def _parallel_forward_opaque(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    reverse: bool,
    num_warps: int,
) -> Tensor:
    return _parallel_forward_impl(
        decay_real,
        decay_imag,
        packed_input,
        reverse,
        num_warps,
    )


@_parallel_forward_opaque.register_fake
def _parallel_forward_fake(  # pyright: ignore[reportUnusedFunction]
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    reverse: bool,
    num_warps: int,
) -> Tensor:
    del decay_real, decay_imag, reverse, num_warps
    return torch.empty_like(packed_input)


def _setup_context(
    ctx: _AutogradContext,
    inputs: tuple[Tensor, Tensor, Tensor, bool, int],
    output: Tensor,
) -> None:
    decay_real, decay_imag, _packed_input, reverse, num_warps = inputs
    ctx.reverse = reverse
    ctx.num_warps = num_warps
    ctx.save_for_backward(decay_real, decay_imag, output)


def _backward(
    ctx: _AutogradContext,
    grad_packed_states: Tensor,
) -> tuple[Tensor, Tensor, Tensor, None, None]:
    decay_real, decay_imag, packed_states = ctx.saved_tensors
    gradients = _parallel_backward_opaque(
        decay_real,
        decay_imag,
        packed_states,
        grad_packed_states,
        ctx.reverse,
        ctx.num_warps,
    )
    return *gradients, None, None


torch.library.register_autograd(
    "lnet::pac_parallel_static_recurrence",
    _backward,
    setup_context=_setup_context,
)


def parallel_static_recurrence_packed(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    *,
    reverse: bool = False,
    num_warps: int = 4,
) -> Tensor:
    """Return packed states from the opt-in time-parallel static recurrence."""
    return _parallel_forward_opaque(
        decay_real,
        decay_imag,
        packed_input,
        reverse,
        num_warps,
    )


__all__ = ["parallel_static_recurrence_packed"]
