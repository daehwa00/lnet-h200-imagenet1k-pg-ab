"""Exact-FP32 static recurrence fused with lag-(1, 2, 4) training moments.

The writer keeps synthesis-ready packed states and reuses them in backward.
The memory-efficient reader writes only the seven modal moments in forward,
then recomputes the states in backward.  The speed reader keeps the forward
state buffer privately across autograd and feeds it directly to the fused
adjoint without materializing a zero direct-state gradient.  All paths expose
an opaque custom-autograd boundary while their implementations remain
``torch.library.triton_op`` compatible.

The CUDA backward deliberately uses two small phases: a deterministic moment
statistics pass and a recurrence adjoint pass.  The adjoint consumes moment
VJPs directly, so no sequence-sized moment-gradient tensor is materialized.
Static-pole gradients are accumulated per batch and reduced in a fixed order;
no atomics are used.
"""

from __future__ import annotations

# pyright: reportCallIssue=false, reportMissingParameterType=false, reportPrivateUsage=false
# ruff: noqa: ANN001, ANN202, C901, FBT001, N803, PLR0915
import os
from typing import Final, Protocol

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton

from .pac_triton_recurrence_lag124 import (
    _launch_cuda,
    _validate_static_packed_inputs,
    reference_static_recurrence_lag124_moments_only,
    reference_static_recurrence_lag124_moments_packed_io,
)
from .pac_triton_recurrence_op import _mode_grid, _select_block_modes

_DEFAULT_EPSILON: Final[float] = 1.0e-8
_WEIGHT_SLOTS: Final[int] = 13
_DISPATCH_READER_RECOMPUTE: Final[str] = "reader_recompute_states"
_DISPATCH_READER_SAVED: Final[str] = "reader_saved_states_no_direct_gradient"
_DISPATCH_WRITER_SAVED: Final[str] = "writer_saved_states_with_direct_gradient"
_last_training_dispatch = "uninitialized"


class _WriterContext(Protocol):
    saved_tensors: tuple[Tensor, ...]
    reverse: bool
    epsilon: float
    single_warp: bool

    def save_for_backward(self, *tensors: Tensor) -> None: ...


class _ReaderContext(Protocol):
    saved_tensors: tuple[Tensor, ...]
    reverse: bool
    epsilon: float
    single_warp: bool

    def save_for_backward(self, *tensors: Tensor) -> None: ...


class _SavedReaderContext(Protocol):
    saved_tensors: tuple[Tensor, ...]
    reverse: bool
    epsilon: float
    single_warp: bool

    def save_for_backward(self, *tensors: Tensor) -> None: ...

    def set_materialize_grads(self, value: bool) -> None: ...


def _validate_exact_inputs(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    epsilon: float,
) -> None:
    _validate_static_packed_inputs(decay_real, decay_imag, packed_input, epsilon)
    if packed_input.dtype != torch.float32:
        message = "lag124 training recurrence supports exact FP32 only"
        raise TypeError(message)


def _validate_backward_inputs(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_states: Tensor,
    grad_moments: Tensor,
    epsilon: float,
) -> None:
    _validate_exact_inputs(decay_real, decay_imag, packed_states, epsilon)
    modes = decay_real.numel()
    expected = (packed_states.shape[0], 7 * modes)
    if grad_moments.shape != expected:
        message = f"lag124 moment gradient must have shape {expected}"
        raise ValueError(message)
    if grad_moments.device != packed_states.device or grad_moments.dtype != torch.float32:
        message = "lag124 state and moment gradients must be FP32 tensors on one device"
        raise TypeError(message)


def _static_decay_view(decay: Tensor, packed_input: Tensor, *, name: str) -> Tensor:
    """Accept [M], [1,1,M], or a [B,T,M] zero-stride static expansion."""
    modes = packed_input.shape[-1] // 2
    if decay.ndim == 1:
        if decay.shape[0] != modes:
            message = f"{name} must contain {modes} static modes"
            raise ValueError(message)
        return decay
    if decay.ndim != 3 or decay.shape[-1] != modes:
        message = f"{name} must have [modes] or static [batch,time,modes] shape"
        raise ValueError(message)
    batch, n_steps = packed_input.shape[:2]
    for axis, expected in ((0, batch), (1, n_steps)):
        if decay.shape[axis] not in (1, expected):
            message = f"{name} is not broadcast-compatible with packed input"
            raise ValueError(message)
        if decay.shape[axis] > 1 and decay.stride(axis) != 0:
            message = f"{name} must be zero-stride along batch and time"
            raise ValueError(message)
    return decay[0, 0]


def _normalize_static_decay(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
) -> tuple[Tensor, Tensor]:
    if packed_input.ndim != 3 or packed_input.shape[-1] % 2 != 0:
        message = "packed recurrence input must have [batch,time,2*modes] shape"
        raise ValueError(message)
    real = _static_decay_view(decay_real, packed_input, name="decay_real")
    imag = _static_decay_view(decay_imag, packed_input, name="decay_imag")
    return real, imag


def lag124_training_backward_fusion_enabled() -> bool:
    """Return whether weights and recurrence adjoint share one CUDA kernel.

    The private environment override exists for paired hardware screens.  The
    measured production default is the fused two-pass-in-one-CTA kernel.
    """
    value = os.environ.get("LNET_PAC_FUSE_LAG124_BACKWARD", "1")
    if value not in {"0", "1"}:
        message = "LNET_PAC_FUSE_LAG124_BACKWARD must be 0 or 1"
        raise ValueError(message)
    return value == "1"


def _reference_moment_state_gradient(
    packed_states: Tensor,
    grad_moments: Tensor,
    *,
    reverse: bool,
    epsilon: float,
) -> Tensor:
    """Reference seven-moment VJP with forward-identical reduction order."""
    batch, n_steps, packed_modes = packed_states.shape
    modes = packed_modes // 2
    states_real, states_imag = packed_states.split(modes, dim=-1)
    zero = torch.zeros((batch, modes), dtype=torch.float32, device=packed_states.device)
    energy_sum = zero
    correlation = {1: (zero, zero), 2: (zero, zero), 4: (zero, zero)}
    overlap_energy = {1: (zero, zero), 2: (zero, zero), 4: (zero, zero)}
    history_real = [zero, zero, zero, zero]
    history_imag = [zero, zero, zero, zero]
    traversal = range(n_steps - 1, -1, -1) if reverse else range(n_steps)

    for step, time_index in enumerate(traversal):
        current_real = states_real[:, time_index].float()
        current_imag = states_imag[:, time_index].float()
        current_energy = current_real.square() + current_imag.square()
        energy_sum = energy_sum + current_energy
        for lag in (1, 2, 4):
            if step < lag:
                continue
            previous_real = history_real[lag - 1]
            previous_imag = history_imag[lag - 1]
            previous_energy = previous_real.square() + previous_imag.square()
            real_sum, imag_sum = correlation[lag]
            current_sum, previous_sum = overlap_energy[lag]
            real_sum = real_sum + current_real * previous_real + current_imag * previous_imag
            if reverse:
                imag_sum = imag_sum + previous_imag * current_real - previous_real * current_imag
                current_sum = current_sum + previous_energy
                previous_sum = previous_sum + current_energy
            else:
                imag_sum = imag_sum + current_imag * previous_real - current_real * previous_imag
                current_sum = current_sum + current_energy
                previous_sum = previous_sum + previous_energy
            correlation[lag] = real_sum, imag_sum
            overlap_energy[lag] = current_sum, previous_sum
        history_real = [current_real, *history_real[:3]]
        history_imag = [current_imag, *history_imag[:3]]

    energy = energy_sum / n_steps
    energy_scale = (2.0 / n_steps) * grad_moments[:, :modes].float() / (1.0 + energy)
    weights: dict[int, tuple[float, Tensor, Tensor, Tensor, Tensor]] = {}
    for lag, output_offset in ((1, modes), (2, 3 * modes), (4, 5 * modes)):
        if n_steps <= lag:
            weights[lag] = (1.0, zero, zero, zero, zero)
            continue
        count = n_steps - lag
        inverse_count = 1.0 / count
        real_sum, imag_sum = correlation[lag]
        current_sum, previous_sum = overlap_energy[lag]
        correlation_real = real_sum * inverse_count
        correlation_imag = imag_sum * inverse_count
        current_energy = current_sum * inverse_count
        previous_energy = previous_sum * inverse_count
        root = torch.sqrt(current_energy * previous_energy)
        denominator = root.clamp_min(epsilon)
        grad_real = grad_moments[:, output_offset : output_offset + modes].float()
        grad_imag = grad_moments[:, output_offset + modes : output_offset + 2 * modes].float()
        real_weight = grad_real / denominator
        imag_weight = grad_imag / denominator
        weighted = grad_real * correlation_real + grad_imag * correlation_imag
        root_gradient = torch.where(
            root > epsilon,
            -weighted / denominator.square(),
            torch.zeros_like(weighted),
        )
        safe_root = root.clamp_min(epsilon)
        current_gradient = 0.5 * root_gradient * previous_energy / safe_root
        previous_gradient = 0.5 * root_gradient * current_energy / safe_root
        weights[lag] = (
            inverse_count,
            real_weight,
            imag_weight,
            current_gradient,
            previous_gradient,
        )

    state_gradients: list[Tensor] = []
    for time_index in range(n_steps):
        current_real = states_real[:, time_index].float()
        current_imag = states_imag[:, time_index].float()
        grad_real = energy_scale * current_real
        grad_imag = energy_scale * current_imag
        for lag in (1, 2, 4):
            inverse_count, real_weight, imag_weight, current_weight, previous_weight = weights[lag]
            if time_index >= lag:
                previous_real = states_real[:, time_index - lag].float()
                previous_imag = states_imag[:, time_index - lag].float()
                grad_real = (
                    grad_real
                    + inverse_count * (real_weight * previous_real - imag_weight * previous_imag)
                    + 2.0 * inverse_count * current_weight * current_real
                )
                grad_imag = (
                    grad_imag
                    + inverse_count * (real_weight * previous_imag + imag_weight * previous_real)
                    + 2.0 * inverse_count * current_weight * current_imag
                )
            if time_index < n_steps - lag:
                next_real = states_real[:, time_index + lag].float()
                next_imag = states_imag[:, time_index + lag].float()
                grad_real = (
                    grad_real
                    + inverse_count * (real_weight * next_real + imag_weight * next_imag)
                    + 2.0 * inverse_count * previous_weight * current_real
                )
                grad_imag = (
                    grad_imag
                    + inverse_count * (real_weight * next_imag - imag_weight * next_real)
                    + 2.0 * inverse_count * previous_weight * current_imag
                )
        state_gradients.append(torch.cat((grad_real, grad_imag), dim=-1))
    return torch.stack(state_gradients, dim=1)


def _reference_backward(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_states: Tensor,
    direct_grad_packed_states: Tensor | None,
    grad_moments: Tensor,
    *,
    reverse: bool,
    epsilon: float,
) -> tuple[Tensor, Tensor, Tensor]:
    modes = decay_real.numel()
    states_real, states_imag = packed_states.split(modes, dim=-1)
    moment_gradient = _reference_moment_state_gradient(
        packed_states,
        grad_moments,
        reverse=reverse,
        epsilon=epsilon,
    )
    if direct_grad_packed_states is None:
        combined = moment_gradient
    else:
        combined = direct_grad_packed_states.float() + moment_gradient
    combined_real, combined_imag = combined.split(modes, dim=-1)
    batch, n_steps, _ = states_real.shape
    lambda_real = torch.zeros((batch, modes), dtype=torch.float32, device=packed_states.device)
    lambda_imag = torch.zeros_like(lambda_real)
    grad_decay_real = torch.zeros_like(decay_real)
    grad_decay_imag = torch.zeros_like(decay_imag)
    grad_inputs: list[Tensor | None] = [None] * n_steps
    adjoint_steps = range(n_steps) if reverse else range(n_steps - 1, -1, -1)
    for time_index in adjoint_steps:
        lambda_real = lambda_real + combined_real[:, time_index]
        lambda_imag = lambda_imag + combined_imag[:, time_index]
        has_previous = time_index < n_steps - 1 if reverse else time_index > 0
        if has_previous:
            previous_index = time_index + 1 if reverse else time_index - 1
            previous_real = states_real[:, previous_index].float()
            previous_imag = states_imag[:, previous_index].float()
        else:
            previous_real = torch.zeros_like(lambda_real)
            previous_imag = torch.zeros_like(lambda_imag)
        grad_decay_real = grad_decay_real + (
            lambda_real * previous_real + lambda_imag * previous_imag
        ).sum(dim=0)
        grad_decay_imag = grad_decay_imag + (
            -lambda_real * previous_imag + lambda_imag * previous_real
        ).sum(dim=0)
        grad_inputs[time_index] = torch.cat((lambda_real, lambda_imag), dim=-1)
        next_lambda_real = decay_real * lambda_real + decay_imag * lambda_imag
        next_lambda_imag = -decay_imag * lambda_real + decay_real * lambda_imag
        lambda_real, lambda_imag = next_lambda_real, next_lambda_imag
    if any(value is None for value in grad_inputs):
        message = "reference recurrence adjoint did not populate every input gradient"
        raise RuntimeError(message)
    grad_packed_input = torch.stack(
        [value for value in grad_inputs if value is not None],
        dim=1,
    )
    return grad_decay_real, grad_decay_imag, grad_packed_input


@triton.jit
def _materialize_fp32(value):
    return tl.inline_asm_elementwise(
        "mov.b32 $0, $1;",
        "=r,r",
        [value],
        dtype=tl.float32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _correlation_vjp_weights(
    correlation_real_sum,
    correlation_imag_sum,
    current_energy_sum,
    previous_energy_sum,
    output_real_gradient,
    output_imag_gradient,
    n_steps: int,
    epsilon: float,
    LAG: tl.constexpr,
):
    count = tl.maximum(n_steps - LAG, 1)
    inverse_count = 1.0 / count
    correlation_real = correlation_real_sum * inverse_count
    correlation_imag = correlation_imag_sum * inverse_count
    current_energy = current_energy_sum * inverse_count
    previous_energy = previous_energy_sum * inverse_count
    root = tl.sqrt(current_energy * previous_energy)
    denominator = tl.maximum(root, epsilon)
    real_weight = output_real_gradient / denominator
    imag_weight = output_imag_gradient / denominator
    weighted = output_real_gradient * correlation_real + output_imag_gradient * correlation_imag
    root_gradient = tl.where(
        root > epsilon,
        -weighted / (denominator * denominator),
        0.0,
    )
    safe_root = tl.maximum(root, epsilon)
    current_gradient = 0.5 * root_gradient * previous_energy / safe_root
    previous_gradient = 0.5 * root_gradient * current_energy / safe_root
    valid = n_steps > LAG
    return (
        inverse_count,
        tl.where(valid, real_weight, 0.0),
        tl.where(valid, imag_weight, 0.0),
        tl.where(valid, current_gradient, 0.0),
        tl.where(valid, previous_gradient, 0.0),
    )


@triton.jit
def _lag_state_vjp(
    packed_states,
    packed_offset,
    valid_mode,
    current_real,
    current_imag,
    real_weight,
    imag_weight,
    current_energy_gradient,
    previous_energy_gradient,
    inverse_count,
    time_index: int,
    n_steps: tl.constexpr,
    modes: tl.constexpr,
    LAG: tl.constexpr,
):
    has_previous = time_index >= LAG
    previous_offset = packed_offset - 2 * LAG * modes
    previous_real = tl.load(
        packed_states + previous_offset,
        mask=valid_mode & has_previous,
        other=0.0,
    ).to(tl.float32)
    previous_imag = tl.load(
        packed_states + previous_offset + modes,
        mask=valid_mode & has_previous,
        other=0.0,
    ).to(tl.float32)
    grad_real = tl.where(
        has_previous,
        inverse_count * (real_weight * previous_real - imag_weight * previous_imag)
        + 2.0 * inverse_count * current_energy_gradient * current_real,
        0.0,
    )
    grad_imag = tl.where(
        has_previous,
        inverse_count * (real_weight * previous_imag + imag_weight * previous_real)
        + 2.0 * inverse_count * current_energy_gradient * current_imag,
        0.0,
    )

    has_next = time_index < n_steps - LAG
    next_offset = packed_offset + 2 * LAG * modes
    next_real = tl.load(
        packed_states + next_offset,
        mask=valid_mode & has_next,
        other=0.0,
    ).to(tl.float32)
    next_imag = tl.load(
        packed_states + next_offset + modes,
        mask=valid_mode & has_next,
        other=0.0,
    ).to(tl.float32)
    grad_real += tl.where(
        has_next,
        inverse_count * (real_weight * next_real + imag_weight * next_imag)
        + 2.0 * inverse_count * previous_energy_gradient * current_real,
        0.0,
    )
    grad_imag += tl.where(
        has_next,
        inverse_count * (real_weight * next_imag - imag_weight * next_real)
        + 2.0 * inverse_count * previous_energy_gradient * current_imag,
        0.0,
    )
    return grad_real, grad_imag


@triton.jit
def _recompute_static_states_kernel(
    decay_real,
    decay_imag,
    packed_input,
    packed_states,
    n_steps: int,
    modes: int,
    reverse: tl.constexpr,
    BLOCK_MODES: tl.constexpr,
) -> None:
    program = tl.program_id(0)
    mode_blocks = tl.cdiv(modes, BLOCK_MODES)
    batch = program // mode_blocks
    mode_block = program - batch * mode_blocks
    mode = mode_block * BLOCK_MODES + tl.arange(0, BLOCK_MODES)
    valid_mode = mode < modes
    active_decay_real = tl.load(decay_real + mode, mask=valid_mode, other=0.0).to(tl.float32)
    active_decay_imag = tl.load(decay_imag + mode, mask=valid_mode, other=0.0).to(tl.float32)
    state_real = tl.zeros((BLOCK_MODES,), tl.float32)
    state_imag = tl.zeros((BLOCK_MODES,), tl.float32)
    for step in tl.range(
        0,
        n_steps,
        loop_unroll_factor=1,
        disable_licm=True,
    ):
        time_index = n_steps - 1 - step if reverse else step
        packed_offset = (batch * n_steps + time_index) * 2 * modes + mode
        drive_real = tl.load(packed_input + packed_offset, mask=valid_mode, other=0.0).to(
            tl.float32
        )
        drive_imag = tl.load(
            packed_input + packed_offset + modes,
            mask=valid_mode,
            other=0.0,
        ).to(tl.float32)
        previous_real = state_real
        previous_imag = state_imag
        state_real = (
            active_decay_real * previous_real - active_decay_imag * previous_imag + drive_real
        )
        state_imag = (
            active_decay_imag * previous_real + active_decay_real * previous_imag + drive_imag
        )
        tl.store(packed_states + packed_offset, state_real, mask=valid_mode)
        tl.store(packed_states + packed_offset + modes, state_imag, mask=valid_mode)


@triton.jit
def _compute_lag124_moment_weights(
    packed_states,
    grad_moments,
    batch,
    mode,
    valid_mode,
    n_steps: tl.constexpr,
    modes: tl.constexpr,
    epsilon: float,
    reverse: tl.constexpr,
    BLOCK_MODES: tl.constexpr,
):
    moment_base = batch * 7 * modes + mode

    energy_sum = tl.zeros((BLOCK_MODES,), tl.float32)
    correlation1_real = tl.zeros((BLOCK_MODES,), tl.float32)
    correlation1_imag = tl.zeros((BLOCK_MODES,), tl.float32)
    correlation2_real = tl.zeros((BLOCK_MODES,), tl.float32)
    correlation2_imag = tl.zeros((BLOCK_MODES,), tl.float32)
    correlation4_real = tl.zeros((BLOCK_MODES,), tl.float32)
    correlation4_imag = tl.zeros((BLOCK_MODES,), tl.float32)
    current1_energy = tl.zeros((BLOCK_MODES,), tl.float32)
    previous1_energy = tl.zeros((BLOCK_MODES,), tl.float32)
    current2_energy = tl.zeros((BLOCK_MODES,), tl.float32)
    previous2_energy = tl.zeros((BLOCK_MODES,), tl.float32)
    current4_energy = tl.zeros((BLOCK_MODES,), tl.float32)
    previous4_energy = tl.zeros((BLOCK_MODES,), tl.float32)
    history1_real = tl.zeros((BLOCK_MODES,), tl.float32)
    history1_imag = tl.zeros((BLOCK_MODES,), tl.float32)
    history2_real = tl.zeros((BLOCK_MODES,), tl.float32)
    history2_imag = tl.zeros((BLOCK_MODES,), tl.float32)
    history3_real = tl.zeros((BLOCK_MODES,), tl.float32)
    history3_imag = tl.zeros((BLOCK_MODES,), tl.float32)
    history4_real = tl.zeros((BLOCK_MODES,), tl.float32)
    history4_imag = tl.zeros((BLOCK_MODES,), tl.float32)

    for step in tl.range(
        0,
        n_steps,
        loop_unroll_factor=1,
        disable_licm=True,
    ):
        time_index = n_steps - 1 - step if reverse else step
        packed_offset = (batch * n_steps + time_index) * 2 * modes + mode
        current_real = tl.load(
            packed_states + packed_offset,
            mask=valid_mode,
            other=0.0,
        ).to(tl.float32)
        current_imag = tl.load(
            packed_states + packed_offset + modes,
            mask=valid_mode,
            other=0.0,
        ).to(tl.float32)
        active_energy = current_real * current_real + current_imag * current_imag
        energy_sum += active_energy

        valid1 = step >= 1
        valid2 = step >= 2
        valid4 = step >= 4
        history1_energy = history1_real * history1_real + history1_imag * history1_imag
        history2_energy = history2_real * history2_real + history2_imag * history2_imag
        history4_energy = history4_real * history4_real + history4_imag * history4_imag
        if reverse:
            corr1_imag = history1_imag * current_real - history1_real * current_imag
            corr2_imag = history2_imag * current_real - history2_real * current_imag
            corr4_imag = history4_imag * current_real - history4_real * current_imag
            active_current1_energy = history1_energy
            active_previous1_energy = active_energy
            active_current2_energy = history2_energy
            active_previous2_energy = active_energy
            active_current4_energy = history4_energy
            active_previous4_energy = active_energy
        else:
            corr1_imag = current_imag * history1_real - current_real * history1_imag
            corr2_imag = current_imag * history2_real - current_real * history2_imag
            corr4_imag = current_imag * history4_real - current_real * history4_imag
            active_current1_energy = active_energy
            active_previous1_energy = history1_energy
            active_current2_energy = active_energy
            active_previous2_energy = history2_energy
            active_current4_energy = active_energy
            active_previous4_energy = history4_energy
        correlation1_real += tl.where(
            valid1,
            current_real * history1_real + current_imag * history1_imag,
            0.0,
        )
        correlation1_imag += tl.where(valid1, corr1_imag, 0.0)
        current1_energy += tl.where(valid1, active_current1_energy, 0.0)
        previous1_energy += tl.where(valid1, active_previous1_energy, 0.0)

        correlation2_real += tl.where(
            valid2,
            current_real * history2_real + current_imag * history2_imag,
            0.0,
        )
        correlation2_imag += tl.where(valid2, corr2_imag, 0.0)
        current2_energy += tl.where(valid2, active_current2_energy, 0.0)
        previous2_energy += tl.where(valid2, active_previous2_energy, 0.0)

        correlation4_real += tl.where(
            valid4,
            current_real * history4_real + current_imag * history4_imag,
            0.0,
        )
        correlation4_imag += tl.where(valid4, corr4_imag, 0.0)
        current4_energy += tl.where(valid4, active_current4_energy, 0.0)
        previous4_energy += tl.where(valid4, active_previous4_energy, 0.0)

        history4_real = history3_real
        history4_imag = history3_imag
        history3_real = history2_real
        history3_imag = history2_imag
        history2_real = history1_real
        history2_imag = history1_imag
        history1_real = current_real
        history1_imag = current_imag

    energy = energy_sum / n_steps
    energy_output_gradient = tl.load(
        grad_moments + moment_base,
        mask=valid_mode,
        other=0.0,
    ).to(tl.float32)
    energy_scale = (2.0 / n_steps) * energy_output_gradient / (1.0 + energy)

    grad1_real = tl.load(
        grad_moments + moment_base + modes,
        mask=valid_mode,
        other=0.0,
    ).to(tl.float32)
    grad1_imag = tl.load(
        grad_moments + moment_base + 2 * modes,
        mask=valid_mode,
        other=0.0,
    ).to(tl.float32)
    inverse1, real1, imag1, current1, previous1 = _correlation_vjp_weights(
        correlation1_real,
        correlation1_imag,
        current1_energy,
        previous1_energy,
        grad1_real,
        grad1_imag,
        n_steps,
        epsilon,
        LAG=1,
    )
    grad2_real = tl.load(
        grad_moments + moment_base + 3 * modes,
        mask=valid_mode,
        other=0.0,
    ).to(tl.float32)
    grad2_imag = tl.load(
        grad_moments + moment_base + 4 * modes,
        mask=valid_mode,
        other=0.0,
    ).to(tl.float32)
    inverse2, real2, imag2, current2, previous2 = _correlation_vjp_weights(
        correlation2_real,
        correlation2_imag,
        current2_energy,
        previous2_energy,
        grad2_real,
        grad2_imag,
        n_steps,
        epsilon,
        LAG=2,
    )
    grad4_real = tl.load(
        grad_moments + moment_base + 5 * modes,
        mask=valid_mode,
        other=0.0,
    ).to(tl.float32)
    grad4_imag = tl.load(
        grad_moments + moment_base + 6 * modes,
        mask=valid_mode,
        other=0.0,
    ).to(tl.float32)
    inverse4, real4, imag4, current4, previous4 = _correlation_vjp_weights(
        correlation4_real,
        correlation4_imag,
        current4_energy,
        previous4_energy,
        grad4_real,
        grad4_imag,
        n_steps,
        epsilon,
        LAG=4,
    )

    # Counts are shape scalars and are intentionally recomputed in the adjoint.
    _ = inverse1, inverse2, inverse4
    return (
        energy_scale,
        real1,
        imag1,
        current1,
        previous1,
        real2,
        imag2,
        current2,
        previous2,
        real4,
        imag4,
        current4,
        previous4,
    )


@triton.jit
def _lag124_moment_weights_kernel(
    packed_states,
    grad_moments,
    weights,
    n_steps: tl.constexpr,
    modes: tl.constexpr,
    epsilon: float,
    reverse: tl.constexpr,
    BLOCK_MODES: tl.constexpr,
) -> None:
    program = tl.program_id(0)
    mode_blocks = tl.cdiv(modes, BLOCK_MODES)
    batch = program // mode_blocks
    mode_block = program - batch * mode_blocks
    mode = mode_block * BLOCK_MODES + tl.arange(0, BLOCK_MODES)
    valid_mode = mode < modes
    values = _compute_lag124_moment_weights(
        packed_states,
        grad_moments,
        batch,
        mode,
        valid_mode,
        n_steps,
        modes,
        epsilon,
        reverse,
        BLOCK_MODES,
    )
    weight_base = batch * 13 * modes + mode
    tl.store(weights + weight_base, values[0], mask=valid_mode)
    tl.store(weights + weight_base + modes, values[1], mask=valid_mode)
    tl.store(weights + weight_base + 2 * modes, values[2], mask=valid_mode)
    tl.store(weights + weight_base + 3 * modes, values[3], mask=valid_mode)
    tl.store(weights + weight_base + 4 * modes, values[4], mask=valid_mode)
    tl.store(weights + weight_base + 5 * modes, values[5], mask=valid_mode)
    tl.store(weights + weight_base + 6 * modes, values[6], mask=valid_mode)
    tl.store(weights + weight_base + 7 * modes, values[7], mask=valid_mode)
    tl.store(weights + weight_base + 8 * modes, values[8], mask=valid_mode)
    tl.store(weights + weight_base + 9 * modes, values[9], mask=valid_mode)
    tl.store(weights + weight_base + 10 * modes, values[10], mask=valid_mode)
    tl.store(weights + weight_base + 11 * modes, values[11], mask=valid_mode)
    tl.store(weights + weight_base + 12 * modes, values[12], mask=valid_mode)


@triton.jit
def _run_lag124_recurrence_adjoint(
    decay_real,
    decay_imag,
    packed_states,
    direct_grad_packed_states,
    energy_scale,
    real1,
    imag1,
    current1,
    previous1,
    real2,
    imag2,
    current2,
    previous2,
    real4,
    imag4,
    current4,
    previous4,
    grad_decay_per_batch,
    grad_packed_input,
    batch,
    mode,
    valid_mode,
    n_steps: tl.constexpr,
    modes: tl.constexpr,
    reverse: tl.constexpr,
    has_direct_state_grad: tl.constexpr,
    BLOCK_MODES: tl.constexpr,
) -> None:
    inverse1 = 1.0 / tl.maximum(n_steps - 1, 1)
    inverse2 = 1.0 / tl.maximum(n_steps - 2, 1)
    inverse4 = 1.0 / tl.maximum(n_steps - 4, 1)
    active_decay_real = tl.load(decay_real + mode, mask=valid_mode, other=0.0).to(tl.float32)
    active_decay_imag = tl.load(decay_imag + mode, mask=valid_mode, other=0.0).to(tl.float32)

    lambda_real = tl.zeros((BLOCK_MODES,), tl.float32)
    lambda_imag = tl.zeros((BLOCK_MODES,), tl.float32)
    decay_gradient_real = tl.zeros((BLOCK_MODES,), tl.float32)
    decay_gradient_imag = tl.zeros((BLOCK_MODES,), tl.float32)
    for adjoint_step in tl.range(
        0,
        n_steps,
        loop_unroll_factor=1,
        disable_licm=True,
    ):
        time_index = adjoint_step if reverse else n_steps - 1 - adjoint_step
        packed_offset = (batch * n_steps + time_index) * 2 * modes + mode
        current_real = tl.load(
            packed_states + packed_offset,
            mask=valid_mode,
            other=0.0,
        ).to(tl.float32)
        current_imag = tl.load(
            packed_states + packed_offset + modes,
            mask=valid_mode,
            other=0.0,
        ).to(tl.float32)
        moment_grad_real = energy_scale * current_real
        moment_grad_imag = energy_scale * current_imag
        lag_real, lag_imag = _lag_state_vjp(
            packed_states,
            packed_offset,
            valid_mode,
            current_real,
            current_imag,
            real1,
            imag1,
            current1,
            previous1,
            inverse1,
            time_index,
            n_steps,
            modes,
            LAG=1,
        )
        moment_grad_real += lag_real
        moment_grad_imag += lag_imag
        lag_real, lag_imag = _lag_state_vjp(
            packed_states,
            packed_offset,
            valid_mode,
            current_real,
            current_imag,
            real2,
            imag2,
            current2,
            previous2,
            inverse2,
            time_index,
            n_steps,
            modes,
            LAG=2,
        )
        moment_grad_real += lag_real
        moment_grad_imag += lag_imag
        lag_real, lag_imag = _lag_state_vjp(
            packed_states,
            packed_offset,
            valid_mode,
            current_real,
            current_imag,
            real4,
            imag4,
            current4,
            previous4,
            inverse4,
            time_index,
            n_steps,
            modes,
            LAG=4,
        )
        moment_grad_real += lag_real
        moment_grad_imag += lag_imag

        direct_real = tl.zeros((BLOCK_MODES,), tl.float32)
        direct_imag = tl.zeros((BLOCK_MODES,), tl.float32)
        if has_direct_state_grad:
            direct_real = tl.load(
                direct_grad_packed_states + packed_offset,
                mask=valid_mode,
                other=0.0,
            ).to(tl.float32)
            direct_imag = tl.load(
                direct_grad_packed_states + packed_offset + modes,
                mask=valid_mode,
                other=0.0,
            ).to(tl.float32)
        combined_real = _materialize_fp32(direct_real + moment_grad_real)
        combined_imag = _materialize_fp32(direct_imag + moment_grad_imag)
        lambda_real += combined_real
        lambda_imag += combined_imag

        previous_index = time_index + 1 if reverse else time_index - 1
        has_previous = time_index < n_steps - 1 if reverse else time_index > 0
        previous_offset = (batch * n_steps + previous_index) * 2 * modes + mode
        recurrence_previous_real = tl.load(
            packed_states + previous_offset,
            mask=valid_mode & has_previous,
            other=0.0,
        ).to(tl.float32)
        recurrence_previous_imag = tl.load(
            packed_states + previous_offset + modes,
            mask=valid_mode & has_previous,
            other=0.0,
        ).to(tl.float32)
        decay_gradient_real += (
            lambda_real * recurrence_previous_real + lambda_imag * recurrence_previous_imag
        )
        decay_gradient_imag += (
            -lambda_real * recurrence_previous_imag + lambda_imag * recurrence_previous_real
        )
        tl.store(grad_packed_input + packed_offset, lambda_real, mask=valid_mode)
        tl.store(grad_packed_input + packed_offset + modes, lambda_imag, mask=valid_mode)
        next_lambda_real = active_decay_real * lambda_real + active_decay_imag * lambda_imag
        next_lambda_imag = -active_decay_imag * lambda_real + active_decay_real * lambda_imag
        lambda_real = next_lambda_real
        lambda_imag = next_lambda_imag

    decay_offset = batch * 2 * modes + mode
    tl.store(grad_decay_per_batch + decay_offset, decay_gradient_real, mask=valid_mode)
    tl.store(
        grad_decay_per_batch + decay_offset + modes,
        decay_gradient_imag,
        mask=valid_mode,
    )


@triton.jit
def _lag124_recurrence_adjoint_kernel(
    decay_real,
    decay_imag,
    packed_states,
    direct_grad_packed_states,
    weights,
    grad_decay_per_batch,
    grad_packed_input,
    n_steps: tl.constexpr,
    modes: tl.constexpr,
    reverse: tl.constexpr,
    has_direct_state_grad: tl.constexpr,
    BLOCK_MODES: tl.constexpr,
) -> None:
    program = tl.program_id(0)
    mode_blocks = tl.cdiv(modes, BLOCK_MODES)
    batch = program // mode_blocks
    mode_block = program - batch * mode_blocks
    mode = mode_block * BLOCK_MODES + tl.arange(0, BLOCK_MODES)
    valid_mode = mode < modes
    weight_base = batch * 13 * modes + mode
    energy_scale = tl.load(weights + weight_base, mask=valid_mode, other=0.0)
    real1 = tl.load(weights + weight_base + modes, mask=valid_mode, other=0.0)
    imag1 = tl.load(weights + weight_base + 2 * modes, mask=valid_mode, other=0.0)
    current1 = tl.load(weights + weight_base + 3 * modes, mask=valid_mode, other=0.0)
    previous1 = tl.load(weights + weight_base + 4 * modes, mask=valid_mode, other=0.0)
    real2 = tl.load(weights + weight_base + 5 * modes, mask=valid_mode, other=0.0)
    imag2 = tl.load(weights + weight_base + 6 * modes, mask=valid_mode, other=0.0)
    current2 = tl.load(weights + weight_base + 7 * modes, mask=valid_mode, other=0.0)
    previous2 = tl.load(weights + weight_base + 8 * modes, mask=valid_mode, other=0.0)
    real4 = tl.load(weights + weight_base + 9 * modes, mask=valid_mode, other=0.0)
    imag4 = tl.load(weights + weight_base + 10 * modes, mask=valid_mode, other=0.0)
    current4 = tl.load(weights + weight_base + 11 * modes, mask=valid_mode, other=0.0)
    previous4 = tl.load(weights + weight_base + 12 * modes, mask=valid_mode, other=0.0)
    _run_lag124_recurrence_adjoint(
        decay_real,
        decay_imag,
        packed_states,
        direct_grad_packed_states,
        energy_scale,
        real1,
        imag1,
        current1,
        previous1,
        real2,
        imag2,
        current2,
        previous2,
        real4,
        imag4,
        current4,
        previous4,
        grad_decay_per_batch,
        grad_packed_input,
        batch,
        mode,
        valid_mode,
        n_steps,
        modes,
        reverse,
        has_direct_state_grad,
        BLOCK_MODES,
    )


@triton.jit
def _lag124_weights_recurrence_adjoint_kernel(
    decay_real,
    decay_imag,
    packed_states,
    direct_grad_packed_states,
    grad_moments,
    grad_decay_per_batch,
    grad_packed_input,
    n_steps: tl.constexpr,
    modes: tl.constexpr,
    epsilon: float,
    reverse: tl.constexpr,
    has_direct_state_grad: tl.constexpr,
    BLOCK_MODES: tl.constexpr,
) -> None:
    program = tl.program_id(0)
    mode_blocks = tl.cdiv(modes, BLOCK_MODES)
    batch = program // mode_blocks
    mode_block = program - batch * mode_blocks
    mode = mode_block * BLOCK_MODES + tl.arange(0, BLOCK_MODES)
    valid_mode = mode < modes
    values = _compute_lag124_moment_weights(
        packed_states,
        grad_moments,
        batch,
        mode,
        valid_mode,
        n_steps,
        modes,
        epsilon,
        reverse,
        BLOCK_MODES,
    )
    _run_lag124_recurrence_adjoint(
        decay_real,
        decay_imag,
        packed_states,
        direct_grad_packed_states,
        values[0],
        values[1],
        values[2],
        values[3],
        values[4],
        values[5],
        values[6],
        values[7],
        values[8],
        values[9],
        values[10],
        values[11],
        values[12],
        grad_decay_per_batch,
        grad_packed_input,
        batch,
        mode,
        valid_mode,
        n_steps,
        modes,
        reverse,
        has_direct_state_grad,
        BLOCK_MODES,
    )


@triton.jit
def _reduce_static_decay_gradient_kernel(
    grad_decay_per_batch,
    grad_decay_real,
    grad_decay_imag,
    batch_size: int,
    modes: int,
    BLOCK_MODES: tl.constexpr,
) -> None:
    mode = tl.program_id(0) * BLOCK_MODES + tl.arange(0, BLOCK_MODES)
    valid_mode = mode < modes
    total_real = tl.zeros((BLOCK_MODES,), tl.float32)
    total_imag = tl.zeros((BLOCK_MODES,), tl.float32)
    batch = 0
    while batch < batch_size:
        offset = batch * 2 * modes + mode
        total_real += tl.load(
            grad_decay_per_batch + offset,
            mask=valid_mode,
            other=0.0,
        ).to(tl.float32)
        total_imag += tl.load(
            grad_decay_per_batch + offset + modes,
            mask=valid_mode,
            other=0.0,
        ).to(tl.float32)
        batch += 1
    tl.store(grad_decay_real + mode, total_real, mask=valid_mode)
    tl.store(grad_decay_imag + mode, total_imag, mask=valid_mode)


def _launch_backward_cuda(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_states: Tensor,
    direct_grad_packed_states: Tensor | None,
    grad_moments: Tensor,
    *,
    reverse: bool,
    epsilon: float,
    single_warp: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    real = decay_real.contiguous()
    imag = decay_imag.contiguous()
    states = packed_states.contiguous()
    direct = states if direct_grad_packed_states is None else direct_grad_packed_states.contiguous()
    moment_gradient = grad_moments.contiguous()
    batch, n_steps, packed_modes = states.shape
    modes = packed_modes // 2
    fuse_weights_adjoint = lag124_training_backward_fusion_enabled()
    grad_decay_per_batch = torch.empty(
        (batch, 2 * modes),
        dtype=torch.float32,
        device=states.device,
    )
    # The real/imag input gradients share one packed destination allocation.
    grad_packed_input = torch.empty_like(states)
    grad_decay_real = torch.empty_like(real)
    grad_decay_imag = torch.empty_like(imag)
    block_modes = _select_block_modes(modes, batch=batch, n_steps=n_steps)
    grid = _mode_grid(batch, modes, block_modes)
    num_warps = 1 if single_warp or n_steps > 128 else 4
    if fuse_weights_adjoint:
        wrap_triton(_lag124_weights_recurrence_adjoint_kernel)[grid](
            real,
            imag,
            states,
            direct,
            moment_gradient,
            grad_decay_per_batch,
            grad_packed_input,
            n_steps,
            modes,
            epsilon,
            reverse=reverse,
            has_direct_state_grad=direct_grad_packed_states is not None,
            BLOCK_MODES=block_modes,
            num_warps=num_warps,
        )
    else:
        weights = torch.empty(
            (batch, _WEIGHT_SLOTS, modes),
            dtype=torch.float32,
            device=states.device,
        )
        wrap_triton(_lag124_moment_weights_kernel)[grid](
            states,
            moment_gradient,
            weights,
            n_steps,
            modes,
            epsilon,
            reverse=reverse,
            BLOCK_MODES=block_modes,
            num_warps=num_warps,
        )
        wrap_triton(_lag124_recurrence_adjoint_kernel)[grid](
            real,
            imag,
            states,
            direct,
            weights,
            grad_decay_per_batch,
            grad_packed_input,
            n_steps,
            modes,
            reverse=reverse,
            has_direct_state_grad=direct_grad_packed_states is not None,
            BLOCK_MODES=block_modes,
            num_warps=num_warps,
        )
    wrap_triton(_reduce_static_decay_gradient_kernel)[(triton.cdiv(modes, block_modes),)](
        grad_decay_per_batch,
        grad_decay_real,
        grad_decay_imag,
        batch,
        modes,
        BLOCK_MODES=block_modes,
        num_warps=1,
    )
    return grad_decay_real, grad_decay_imag, grad_packed_input


@triton_op("lnet::pac_static_real2d_recurrence_lag124_writer_training_impl", mutates_args={})
def _writer_forward_impl(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    reverse: bool,
    epsilon: float,
    single_warp: bool,
) -> tuple[Tensor, Tensor]:
    _validate_exact_inputs(decay_real, decay_imag, packed_input, epsilon)
    if not packed_input.is_cuda:
        return reference_static_recurrence_lag124_moments_packed_io(
            decay_real,
            decay_imag,
            packed_input,
            reverse=reverse,
            epsilon=epsilon,
        )
    packed_states, moments = _launch_cuda(
        decay_real,
        decay_imag,
        packed_input,
        reverse=reverse,
        epsilon=epsilon,
        single_warp=single_warp,
        store_states=True,
        wrapped=True,
    )
    if packed_states is None:
        message = "lag124 writer launch did not return packed states"
        raise RuntimeError(message)
    return packed_states, moments


@triton_op("lnet::pac_static_real2d_recurrence_lag124_reader_training_impl", mutates_args={})
def _reader_forward_impl(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    reverse: bool,
    epsilon: float,
    single_warp: bool,
) -> Tensor:
    _validate_exact_inputs(decay_real, decay_imag, packed_input, epsilon)
    if not packed_input.is_cuda:
        return reference_static_recurrence_lag124_moments_only(
            decay_real,
            decay_imag,
            packed_input,
            reverse=reverse,
            epsilon=epsilon,
        )
    _, moments = _launch_cuda(
        decay_real,
        decay_imag,
        packed_input,
        reverse=reverse,
        epsilon=epsilon,
        single_warp=single_warp,
        store_states=False,
        wrapped=True,
    )
    return moments


@triton_op(
    "lnet::pac_static_real2d_recurrence_lag124_reader_saved_training_impl",
    mutates_args={},
)
def _saved_reader_forward_impl(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    reverse: bool,
    epsilon: float,
    single_warp: bool,
) -> tuple[Tensor, Tensor]:
    """Return private states plus public moments for the speed reader.

    The packed states are an implementation output rather than a model output.
    Keeping them on the custom-op boundary lets autograd save the existing
    forward allocation.  Its output gradient is therefore ``None`` when the
    public wrapper returns only moments.
    """
    return _writer_forward_impl(
        decay_real,
        decay_imag,
        packed_input,
        reverse,
        epsilon,
        single_warp,
    )


@triton_op(
    "lnet::pac_static_real2d_recurrence_lag124_writer_training_backward_impl",
    mutates_args={},
)
def _writer_backward_impl(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_states: Tensor,
    direct_grad_packed_states: Tensor,
    grad_moments: Tensor,
    reverse: bool,
    epsilon: float,
    single_warp: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    _validate_backward_inputs(decay_real, decay_imag, packed_states, grad_moments, epsilon)
    if direct_grad_packed_states.shape != packed_states.shape:
        message = "direct packed-state gradient must match writer state shape"
        raise ValueError(message)
    if not packed_states.is_cuda:
        return _reference_backward(
            decay_real,
            decay_imag,
            packed_states,
            direct_grad_packed_states,
            grad_moments,
            reverse=reverse,
            epsilon=epsilon,
        )
    return _launch_backward_cuda(
        decay_real,
        decay_imag,
        packed_states,
        direct_grad_packed_states,
        grad_moments,
        reverse=reverse,
        epsilon=epsilon,
        single_warp=single_warp,
    )


@triton_op(
    "lnet::pac_static_real2d_recurrence_lag124_reader_training_backward_impl",
    mutates_args={},
)
def _reader_backward_impl(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    grad_moments: Tensor,
    reverse: bool,
    epsilon: float,
    single_warp: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    _validate_backward_inputs(decay_real, decay_imag, packed_input, grad_moments, epsilon)
    if not packed_input.is_cuda:
        packed_states, _ = reference_static_recurrence_lag124_moments_packed_io(
            decay_real,
            decay_imag,
            packed_input,
            reverse=reverse,
            epsilon=epsilon,
        )
        return _reference_backward(
            decay_real,
            decay_imag,
            packed_states,
            None,
            grad_moments,
            reverse=reverse,
            epsilon=epsilon,
        )
    drive = packed_input.contiguous()
    batch, n_steps, packed_modes = drive.shape
    modes = packed_modes // 2
    packed_states = torch.empty_like(drive)
    block_modes = _select_block_modes(modes, batch=batch, n_steps=n_steps)
    wrap_triton(_recompute_static_states_kernel)[_mode_grid(batch, modes, block_modes)](
        decay_real.contiguous(),
        decay_imag.contiguous(),
        drive,
        packed_states,
        n_steps,
        modes,
        reverse=reverse,
        BLOCK_MODES=block_modes,
        num_warps=1 if single_warp or n_steps > 128 else 4,
    )
    return _launch_backward_cuda(
        decay_real,
        decay_imag,
        packed_states,
        None,
        grad_moments,
        reverse=reverse,
        epsilon=epsilon,
        single_warp=single_warp,
    )


@triton_op(
    "lnet::pac_static_real2d_recurrence_lag124_reader_saved_training_backward_impl",
    mutates_args={},
)
def _saved_reader_backward_impl(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_states: Tensor,
    grad_moments: Tensor,
    reverse: bool,
    epsilon: float,
    single_warp: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    """Run the moment/recurrence adjoint against states saved in forward."""
    _validate_backward_inputs(decay_real, decay_imag, packed_states, grad_moments, epsilon)
    if not packed_states.is_cuda:
        return _reference_backward(
            decay_real,
            decay_imag,
            packed_states,
            None,
            grad_moments,
            reverse=reverse,
            epsilon=epsilon,
        )
    return _launch_backward_cuda(
        decay_real,
        decay_imag,
        packed_states,
        None,
        grad_moments,
        reverse=reverse,
        epsilon=epsilon,
        single_warp=single_warp,
    )


@torch.library.custom_op(
    "lnet::pac_static_real2d_recurrence_lag124_writer_training_backward",
    mutates_args=(),
)
def _writer_backward_opaque(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_states: Tensor,
    direct_grad_packed_states: Tensor,
    grad_moments: Tensor,
    reverse: bool,
    epsilon: float,
    single_warp: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    return _writer_backward_impl(
        decay_real,
        decay_imag,
        packed_states,
        direct_grad_packed_states,
        grad_moments,
        reverse,
        epsilon,
        single_warp,
    )


@_writer_backward_opaque.register_fake
def _writer_backward_fake(  # pyright: ignore[reportUnusedFunction]
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_states: Tensor,
    direct_grad_packed_states: Tensor,
    grad_moments: Tensor,
    reverse: bool,
    epsilon: float,
    single_warp: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    del packed_states, grad_moments, reverse, epsilon, single_warp
    return (
        torch.empty_like(decay_real),
        torch.empty_like(decay_imag),
        torch.empty_like(direct_grad_packed_states),
    )


@torch.library.custom_op(
    "lnet::pac_static_real2d_recurrence_lag124_reader_training_backward",
    mutates_args=(),
)
def _reader_backward_opaque(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    grad_moments: Tensor,
    reverse: bool,
    epsilon: float,
    single_warp: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    return _reader_backward_impl(
        decay_real,
        decay_imag,
        packed_input,
        grad_moments,
        reverse,
        epsilon,
        single_warp,
    )


@_reader_backward_opaque.register_fake
def _reader_backward_fake(  # pyright: ignore[reportUnusedFunction]
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    grad_moments: Tensor,
    reverse: bool,
    epsilon: float,
    single_warp: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    del grad_moments, reverse, epsilon, single_warp
    return (
        torch.empty_like(decay_real),
        torch.empty_like(decay_imag),
        torch.empty_like(packed_input),
    )


@torch.library.custom_op(
    "lnet::pac_static_real2d_recurrence_lag124_reader_saved_training_backward",
    mutates_args=(),
)
def _saved_reader_backward_opaque(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_states: Tensor,
    grad_moments: Tensor,
    reverse: bool,
    epsilon: float,
    single_warp: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    return _saved_reader_backward_impl(
        decay_real,
        decay_imag,
        packed_states,
        grad_moments,
        reverse,
        epsilon,
        single_warp,
    )


@_saved_reader_backward_opaque.register_fake
def _saved_reader_backward_fake(  # pyright: ignore[reportUnusedFunction]
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_states: Tensor,
    grad_moments: Tensor,
    reverse: bool,
    epsilon: float,
    single_warp: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    del grad_moments, reverse, epsilon, single_warp
    return (
        torch.empty_like(decay_real),
        torch.empty_like(decay_imag),
        torch.empty_like(packed_states),
    )


@torch.library.custom_op(
    "lnet::pac_static_real2d_recurrence_lag124_writer_training",
    mutates_args=(),
)
def _writer_opaque(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    reverse: bool,
    epsilon: float,
    single_warp: bool,
) -> tuple[Tensor, Tensor]:
    return _writer_forward_impl(
        decay_real,
        decay_imag,
        packed_input,
        reverse,
        epsilon,
        single_warp,
    )


@_writer_opaque.register_fake
def _writer_forward_fake(  # pyright: ignore[reportUnusedFunction]
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    reverse: bool,
    epsilon: float,
    single_warp: bool,
) -> tuple[Tensor, Tensor]:
    del decay_real, decay_imag, reverse, epsilon, single_warp
    modes = packed_input.shape[-1] // 2
    moments = packed_input.new_empty((packed_input.shape[0], 7 * modes))
    return torch.empty_like(packed_input), moments


@torch.library.custom_op(
    "lnet::pac_static_real2d_recurrence_lag124_reader_training",
    mutates_args=(),
)
def _reader_opaque(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    reverse: bool,
    epsilon: float,
    single_warp: bool,
) -> Tensor:
    return _reader_forward_impl(
        decay_real,
        decay_imag,
        packed_input,
        reverse,
        epsilon,
        single_warp,
    )


@_reader_opaque.register_fake
def _reader_forward_fake(  # pyright: ignore[reportUnusedFunction]
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    reverse: bool,
    epsilon: float,
    single_warp: bool,
) -> Tensor:
    del decay_real, decay_imag, reverse, epsilon, single_warp
    modes = packed_input.shape[-1] // 2
    return packed_input.new_empty((packed_input.shape[0], 7 * modes))


@torch.library.custom_op(
    "lnet::pac_static_real2d_recurrence_lag124_reader_saved_training",
    mutates_args=(),
)
def _saved_reader_opaque(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    reverse: bool,
    epsilon: float,
    single_warp: bool,
) -> tuple[Tensor, Tensor]:
    return _saved_reader_forward_impl(
        decay_real,
        decay_imag,
        packed_input,
        reverse,
        epsilon,
        single_warp,
    )


@_saved_reader_opaque.register_fake
def _saved_reader_forward_fake(  # pyright: ignore[reportUnusedFunction]
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    reverse: bool,
    epsilon: float,
    single_warp: bool,
) -> tuple[Tensor, Tensor]:
    del decay_real, decay_imag, reverse, epsilon, single_warp
    modes = packed_input.shape[-1] // 2
    moments = packed_input.new_empty((packed_input.shape[0], 7 * modes))
    return torch.empty_like(packed_input), moments


def _setup_writer_context(
    ctx: _WriterContext,
    inputs: tuple[Tensor, Tensor, Tensor, bool, float, bool],
    output: tuple[Tensor, Tensor],
) -> None:
    decay_real, decay_imag, _packed_input, reverse, epsilon, single_warp = inputs
    packed_states, _moments = output
    ctx.reverse = reverse
    ctx.epsilon = epsilon
    ctx.single_warp = single_warp
    ctx.save_for_backward(decay_real, decay_imag, packed_states)


def _writer_backward(
    ctx: _WriterContext,
    grad_packed_states: Tensor | None,
    grad_moments: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor, None, None, None]:
    decay_real, decay_imag, packed_states = ctx.saved_tensors
    modes = decay_real.numel()
    if grad_packed_states is None:
        grad_packed_states = torch.zeros_like(packed_states)
    if grad_moments is None:
        grad_moments = packed_states.new_zeros((packed_states.shape[0], 7 * modes))
    gradients = _writer_backward_opaque(
        decay_real,
        decay_imag,
        packed_states,
        grad_packed_states,
        grad_moments,
        ctx.reverse,
        ctx.epsilon,
        ctx.single_warp,
    )
    return *gradients, None, None, None


torch.library.register_autograd(
    "lnet::pac_static_real2d_recurrence_lag124_writer_training",
    _writer_backward,
    setup_context=_setup_writer_context,
)


def _setup_reader_context(
    ctx: _ReaderContext,
    inputs: tuple[Tensor, Tensor, Tensor, bool, float, bool],
    output: Tensor,
) -> None:
    decay_real, decay_imag, packed_input, reverse, epsilon, single_warp = inputs
    del output
    ctx.reverse = reverse
    ctx.epsilon = epsilon
    ctx.single_warp = single_warp
    # Reader forward keeps no sequence state; backward reconstructs it.
    ctx.save_for_backward(decay_real, decay_imag, packed_input)


def _reader_backward(
    ctx: _ReaderContext,
    grad_moments: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor, None, None, None]:
    decay_real, decay_imag, packed_input = ctx.saved_tensors
    if grad_moments is None:
        modes = decay_real.numel()
        grad_moments = packed_input.new_zeros((packed_input.shape[0], 7 * modes))
    gradients = _reader_backward_opaque(
        decay_real,
        decay_imag,
        packed_input,
        grad_moments,
        ctx.reverse,
        ctx.epsilon,
        ctx.single_warp,
    )
    return *gradients, None, None, None


torch.library.register_autograd(
    "lnet::pac_static_real2d_recurrence_lag124_reader_training",
    _reader_backward,
    setup_context=_setup_reader_context,
)


def _setup_saved_reader_context(
    ctx: _SavedReaderContext,
    inputs: tuple[Tensor, Tensor, Tensor, bool, float, bool],
    output: tuple[Tensor, Tensor],
) -> None:
    decay_real, decay_imag, _packed_input, reverse, epsilon, single_warp = inputs
    packed_states, _moments = output
    ctx.reverse = reverse
    ctx.epsilon = epsilon
    ctx.single_warp = single_warp
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(decay_real, decay_imag, packed_states)


def _saved_reader_backward(
    ctx: _SavedReaderContext,
    grad_packed_states: Tensor | None,
    grad_moments: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor, None, None, None]:
    decay_real, decay_imag, packed_states = ctx.saved_tensors
    modes = decay_real.numel()
    if grad_moments is None:
        grad_moments = packed_states.new_zeros((packed_states.shape[0], 7 * modes))
    if grad_packed_states is None:
        gradients = _saved_reader_backward_opaque(
            decay_real,
            decay_imag,
            packed_states,
            grad_moments,
            ctx.reverse,
            ctx.epsilon,
            ctx.single_warp,
        )
    else:
        # Preserve correctness when the low-level two-output op is called
        # directly.  The public moments-only API always takes the no-direct
        # branch and therefore never creates a sequence-sized zero tensor.
        gradients = _writer_backward_opaque(
            decay_real,
            decay_imag,
            packed_states,
            grad_packed_states,
            grad_moments,
            ctx.reverse,
            ctx.epsilon,
            ctx.single_warp,
        )
    return *gradients, None, None, None


torch.library.register_autograd(
    "lnet::pac_static_real2d_recurrence_lag124_reader_saved_training",
    _saved_reader_backward,
    setup_context=_setup_saved_reader_context,
)


def static_recurrence_lag124_moments_packed_io_training(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    *,
    reverse: bool = False,
    epsilon: float = _DEFAULT_EPSILON,
    single_warp: bool = False,
) -> tuple[Tensor, Tensor]:
    """Writer API: return packed states and seven moments with fused backward."""
    global _last_training_dispatch  # noqa: PLW0603
    backward = "fused_weights_adjoint" if lag124_training_backward_fusion_enabled() else "split"
    _last_training_dispatch = f"{_DISPATCH_WRITER_SAVED}:{backward}"
    static_real, static_imag = _normalize_static_decay(decay_real, decay_imag, packed_input)
    _validate_exact_inputs(static_real, static_imag, packed_input, epsilon)
    return _writer_opaque(
        static_real,
        static_imag,
        packed_input,
        reverse,
        epsilon,
        single_warp,
    )


def static_recurrence_lag124_moments_only_training(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    *,
    reverse: bool = False,
    epsilon: float = _DEFAULT_EPSILON,
    single_warp: bool = False,
) -> Tensor:
    """Reader API: return moments only and recompute recurrence states in backward."""
    global _last_training_dispatch  # noqa: PLW0603
    backward = "fused_weights_adjoint" if lag124_training_backward_fusion_enabled() else "split"
    _last_training_dispatch = f"{_DISPATCH_READER_RECOMPUTE}:{backward}"
    static_real, static_imag = _normalize_static_decay(decay_real, decay_imag, packed_input)
    _validate_exact_inputs(static_real, static_imag, packed_input, epsilon)
    return _reader_opaque(
        static_real,
        static_imag,
        packed_input,
        reverse,
        epsilon,
        single_warp,
    )


def static_recurrence_lag124_moments_only_saved_states_training(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    *,
    reverse: bool = False,
    epsilon: float = _DEFAULT_EPSILON,
    single_warp: bool = False,
) -> Tensor:
    """Speed reader: save forward states and fuse their lag124 recurrence VJP.

    The sequence state is a private custom-op output.  It is reused by backward
    with ``has_direct_state_grad=False``; no recurrence recomputation and no
    sequence-sized all-zero state gradient are emitted.
    """
    global _last_training_dispatch  # noqa: PLW0603
    backward = "fused_weights_adjoint" if lag124_training_backward_fusion_enabled() else "split"
    _last_training_dispatch = f"{_DISPATCH_READER_SAVED}:{backward}"
    static_real, static_imag = _normalize_static_decay(decay_real, decay_imag, packed_input)
    _validate_exact_inputs(static_real, static_imag, packed_input, epsilon)
    _packed_states, moments = _saved_reader_opaque(
        static_real,
        static_imag,
        packed_input,
        reverse,
        epsilon,
        single_warp,
    )
    return moments


def last_lag124_training_dispatch() -> str:
    """Return the most recently selected eager training storage policy."""
    return _last_training_dispatch


# Short role-oriented aliases used by integration code.
static_recurrence_lag124_writer_training = static_recurrence_lag124_moments_packed_io_training
static_recurrence_lag124_reader_training = static_recurrence_lag124_moments_only_training
static_recurrence_lag124_reader_saved_training = (
    static_recurrence_lag124_moments_only_saved_states_training
)


__all__ = [
    "lag124_training_backward_fusion_enabled",
    "last_lag124_training_dispatch",
    "static_recurrence_lag124_moments_only_saved_states_training",
    "static_recurrence_lag124_moments_only_training",
    "static_recurrence_lag124_moments_packed_io_training",
    "static_recurrence_lag124_reader_saved_training",
    "static_recurrence_lag124_reader_training",
    "static_recurrence_lag124_writer_training",
]
