"""Static packed-input recurrence fused with lag-(1,2,4) modal moments.

The writer entry point emits synthesis-ready packed states and seven canonical
moments per mode.  The reader entry point emits only the moments and therefore
does not allocate or write a sequence-sized state tensor.  CUDA FP32 inference
uses a dedicated Triton kernel; every other device, dtype, or autograd context
uses a differentiable PyTorch reference implementation.
"""

from __future__ import annotations

# pyright: reportCallIssue=false, reportMissingParameterType=false, reportPrivateUsage=false
# ruff: noqa: ANN001, N803, PLR0915
import os
from typing import Final

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton
from triton.language.extra import libdevice

from .pac_triton_recurrence_op import (
    _VALID_BLOCK_MODES,
    _mode_grid,
    _select_block_modes,
)

_EPSILON: Final[float] = 1.0e-8
_SUPPORTED_DTYPES: Final[tuple[torch.dtype, ...]] = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
)


def _select_lag124_block_modes(modes: int, *, batch: int, n_steps: int) -> int:
    """Select the production block width with a lag124-specific screen override."""
    override_name = "LNET_PAC_LAG124_BLOCK_MODES"
    override = os.environ.get(override_name)
    if override is not None:
        block_modes = int(override)
        if block_modes not in _VALID_BLOCK_MODES:
            message = f"{override_name} must be one of {_VALID_BLOCK_MODES}"
            raise ValueError(message)
        return block_modes

    return _select_block_modes(modes, batch=batch, n_steps=n_steps)


def _empty_state_slots(n_steps: int) -> list[Tensor | None]:
    return [None for _ in range(n_steps)]


def _validate_static_packed_inputs(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    epsilon: float,
) -> None:
    if decay_real.ndim != 1 or decay_real.numel() == 0:
        message = "static decay tensors must have non-empty [modes] shape"
        raise ValueError(message)
    if decay_imag.shape != decay_real.shape:
        message = "static real and imaginary decay tensors must have matching shapes"
        raise ValueError(message)
    if (
        packed_input.ndim != 3
        or packed_input.shape[0] == 0
        or packed_input.shape[1] == 0
        or packed_input.shape[2] != 2 * decay_real.numel()
    ):
        message = (
            "packed static recurrence input must have non-empty [batch, time, 2 * modes] shape"
        )
        raise ValueError(message)
    for tensor in (decay_imag, packed_input):
        if tensor.device != decay_real.device:
            message = "static recurrence tensors must share one device"
            raise ValueError(message)
        if tensor.dtype != decay_real.dtype:
            message = "static recurrence tensors must share one dtype"
            raise TypeError(message)
    if decay_real.dtype not in _SUPPORTED_DTYPES:
        message = "static lag124 recurrence supports fp16, bf16, fp32, and fp64"
        raise TypeError(message)
    if epsilon <= 0.0:
        message = "epsilon must be positive"
        raise ValueError(message)


def _reference_static_recurrence_lag124(  # noqa: C901, PLR0912
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    *,
    reverse: bool,
    epsilon: float,
    store_states: bool,
) -> tuple[Tensor | None, Tensor]:
    """Reference recurrence retaining only four moment-history states."""
    _validate_static_packed_inputs(decay_real, decay_imag, packed_input, epsilon)
    modes = decay_real.numel()
    drive_real, drive_imag = packed_input.split(modes, dim=-1)
    accumulator_dtype = torch.float64 if packed_input.dtype == torch.float64 else torch.float32
    active_decay_real = decay_real.to(dtype=accumulator_dtype)
    active_decay_imag = decay_imag.to(dtype=accumulator_dtype)
    state_real = torch.zeros_like(drive_real[:, 0], dtype=accumulator_dtype)
    state_imag = torch.zeros_like(drive_imag[:, 0], dtype=accumulator_dtype)
    history1_real = torch.zeros_like(state_real)
    history1_imag = torch.zeros_like(state_imag)
    history2_real = torch.zeros_like(state_real)
    history2_imag = torch.zeros_like(state_imag)
    history3_real = torch.zeros_like(state_real)
    history3_imag = torch.zeros_like(state_imag)
    history4_real = torch.zeros_like(state_real)
    history4_imag = torch.zeros_like(state_imag)
    energy_sum = torch.zeros_like(state_real)
    correlation1_real = torch.zeros_like(state_real)
    correlation1_imag = torch.zeros_like(state_real)
    correlation2_real = torch.zeros_like(state_real)
    correlation2_imag = torch.zeros_like(state_real)
    correlation4_real = torch.zeros_like(state_real)
    correlation4_imag = torch.zeros_like(state_real)
    current1_energy = torch.zeros_like(state_real)
    previous1_energy = torch.zeros_like(state_real)
    current2_energy = torch.zeros_like(state_real)
    previous2_energy = torch.zeros_like(state_real)
    current4_energy = torch.zeros_like(state_real)
    previous4_energy = torch.zeros_like(state_real)
    n_steps = packed_input.shape[1]
    stored_states = _empty_state_slots(n_steps) if store_states else None
    traversal = range(n_steps - 1, -1, -1) if reverse else range(n_steps)

    for step, time_index in enumerate(traversal):
        previous_state_real = state_real
        previous_state_imag = state_imag
        current_drive_real = drive_real[:, time_index].to(dtype=accumulator_dtype)
        current_drive_imag = drive_imag[:, time_index].to(dtype=accumulator_dtype)
        state_real = (
            active_decay_real * previous_state_real
            - active_decay_imag * previous_state_imag
            + current_drive_real
        )
        state_imag = (
            active_decay_imag * previous_state_real
            + active_decay_real * previous_state_imag
            + current_drive_imag
        )

        # CUDA recurrence stores states in the input dtype while retaining FP32
        # recurrence registers.  Moment reductions over materialized low-precision
        # states therefore observe this round trip; model it in the fallback too.
        observed_real = state_real.to(dtype=packed_input.dtype).to(dtype=accumulator_dtype)
        observed_imag = state_imag.to(dtype=packed_input.dtype).to(dtype=accumulator_dtype)
        if stored_states is not None:
            stored_states[time_index] = torch.cat((observed_real, observed_imag), dim=-1).to(
                dtype=packed_input.dtype
            )

        active_energy = observed_real.square() + observed_imag.square()
        energy_sum = energy_sum + active_energy
        if step >= 1:
            history_energy = history1_real.square() + history1_imag.square()
            correlation1_real = correlation1_real + (
                observed_real * history1_real + observed_imag * history1_imag
            )
            if reverse:
                correlation1_imag = correlation1_imag + (
                    history1_imag * observed_real - history1_real * observed_imag
                )
                current1_energy = current1_energy + history_energy
                previous1_energy = previous1_energy + active_energy
            else:
                correlation1_imag = correlation1_imag + (
                    observed_imag * history1_real - observed_real * history1_imag
                )
                current1_energy = current1_energy + active_energy
                previous1_energy = previous1_energy + history_energy
        if step >= 2:
            history_energy = history2_real.square() + history2_imag.square()
            correlation2_real = correlation2_real + (
                observed_real * history2_real + observed_imag * history2_imag
            )
            if reverse:
                correlation2_imag = correlation2_imag + (
                    history2_imag * observed_real - history2_real * observed_imag
                )
                current2_energy = current2_energy + history_energy
                previous2_energy = previous2_energy + active_energy
            else:
                correlation2_imag = correlation2_imag + (
                    observed_imag * history2_real - observed_real * history2_imag
                )
                current2_energy = current2_energy + active_energy
                previous2_energy = previous2_energy + history_energy
        if step >= 4:
            history_energy = history4_real.square() + history4_imag.square()
            correlation4_real = correlation4_real + (
                observed_real * history4_real + observed_imag * history4_imag
            )
            if reverse:
                correlation4_imag = correlation4_imag + (
                    history4_imag * observed_real - history4_real * observed_imag
                )
                current4_energy = current4_energy + history_energy
                previous4_energy = previous4_energy + active_energy
            else:
                correlation4_imag = correlation4_imag + (
                    observed_imag * history4_real - observed_real * history4_imag
                )
                current4_energy = current4_energy + active_energy
                previous4_energy = previous4_energy + history_energy

        history4_real, history4_imag = history3_real, history3_imag
        history3_real, history3_imag = history2_real, history2_imag
        history2_real, history2_imag = history1_real, history1_imag
        history1_real, history1_imag = observed_real, observed_imag

    moments = [torch.log1p(energy_sum / n_steps)]
    for lag, real_sum, imag_sum, current_sum, previous_sum in (
        (1, correlation1_real, correlation1_imag, current1_energy, previous1_energy),
        (2, correlation2_real, correlation2_imag, current2_energy, previous2_energy),
        (4, correlation4_real, correlation4_imag, current4_energy, previous4_energy),
    ):
        if n_steps <= lag:
            moments.extend((torch.zeros_like(state_real), torch.zeros_like(state_real)))
            continue
        count = n_steps - lag
        denominator = torch.sqrt(
            ((current_sum / count) * (previous_sum / count)).clamp_min(epsilon * epsilon)
        )
        moments.extend(((real_sum / count) / denominator, (imag_sum / count) / denominator))

    packed_states = None
    if stored_states is not None:
        if any(state is None for state in stored_states):
            message = "reference recurrence did not populate every state"
            raise RuntimeError(message)
        packed_states = torch.stack(
            [state for state in stored_states if state is not None],
            dim=1,
        )
    return packed_states, torch.cat(moments, dim=-1).to(dtype=packed_input.dtype)


def reference_static_recurrence_lag124_moments_only(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    *,
    reverse: bool = False,
    epsilon: float = _EPSILON,
) -> Tensor:
    """Compute seven moments without constructing a sequence state tensor."""
    _, moments = _reference_static_recurrence_lag124(
        decay_real,
        decay_imag,
        packed_input,
        reverse=reverse,
        epsilon=epsilon,
        store_states=False,
    )
    return moments


def reference_static_recurrence_lag124_moments_packed_io(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    *,
    reverse: bool = False,
    epsilon: float = _EPSILON,
) -> tuple[Tensor, Tensor]:
    """Materialize packed recurrence states and their seven canonical moments."""
    packed_states, moments = _reference_static_recurrence_lag124(
        decay_real,
        decay_imag,
        packed_input,
        reverse=reverse,
        epsilon=epsilon,
        store_states=True,
    )
    if packed_states is None:
        message = "packed-state reference did not produce states"
        raise RuntimeError(message)
    return packed_states, moments


@triton.jit
def _static_recurrence_lag124_kernel(
    decay_real,
    decay_imag,
    packed_input,
    packed_states,
    moment_output,
    n_steps: int,
    modes: int,
    epsilon: float,
    reverse: tl.constexpr,
    store_states: tl.constexpr,
    BLOCK_MODES: tl.constexpr,
) -> None:
    program = tl.program_id(0)
    mode_blocks = tl.cdiv(modes, BLOCK_MODES)
    batch = program // mode_blocks
    mode_block = program - batch * mode_blocks
    mode = mode_block * BLOCK_MODES + tl.arange(0, BLOCK_MODES)
    valid_mode = mode < modes
    state_real = tl.zeros((BLOCK_MODES,), tl.float32)
    state_imag = tl.zeros((BLOCK_MODES,), tl.float32)
    history1_real = tl.zeros((BLOCK_MODES,), tl.float32)
    history1_imag = tl.zeros((BLOCK_MODES,), tl.float32)
    history2_real = tl.zeros((BLOCK_MODES,), tl.float32)
    history2_imag = tl.zeros((BLOCK_MODES,), tl.float32)
    history3_real = tl.zeros((BLOCK_MODES,), tl.float32)
    history3_imag = tl.zeros((BLOCK_MODES,), tl.float32)
    history4_real = tl.zeros((BLOCK_MODES,), tl.float32)
    history4_imag = tl.zeros((BLOCK_MODES,), tl.float32)
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
    active_decay_real = tl.load(decay_real + mode, mask=valid_mode, other=0.0).to(tl.float32)
    active_decay_imag = tl.load(decay_imag + mode, mask=valid_mode, other=0.0).to(tl.float32)

    step = 0
    while step < n_steps:
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
        previous_state_real = state_real
        previous_state_imag = state_imag
        state_real = (
            active_decay_real * previous_state_real
            - active_decay_imag * previous_state_imag
            + drive_real
        )
        state_imag = (
            active_decay_imag * previous_state_real
            + active_decay_real * previous_state_imag
            + drive_imag
        )
        if store_states:
            tl.store(packed_states + packed_offset, state_real, mask=valid_mode)
            tl.store(packed_states + packed_offset + modes, state_imag, mask=valid_mode)

        active_energy = state_real * state_real + state_imag * state_imag
        energy_sum += active_energy
        valid1 = step >= 1
        valid2 = step >= 2
        valid4 = step >= 4
        history1_energy = history1_real * history1_real + history1_imag * history1_imag
        history2_energy = history2_real * history2_real + history2_imag * history2_imag
        history4_energy = history4_real * history4_real + history4_imag * history4_imag
        if reverse:
            corr1_imag = history1_imag * state_real - history1_real * state_imag
            corr2_imag = history2_imag * state_real - history2_real * state_imag
            corr4_imag = history4_imag * state_real - history4_real * state_imag
            active_current1_energy = history1_energy
            active_previous1_energy = active_energy
            active_current2_energy = history2_energy
            active_previous2_energy = active_energy
            active_current4_energy = history4_energy
            active_previous4_energy = active_energy
        else:
            corr1_imag = state_imag * history1_real - state_real * history1_imag
            corr2_imag = state_imag * history2_real - state_real * history2_imag
            corr4_imag = state_imag * history4_real - state_real * history4_imag
            active_current1_energy = active_energy
            active_previous1_energy = history1_energy
            active_current2_energy = active_energy
            active_previous2_energy = history2_energy
            active_current4_energy = active_energy
            active_previous4_energy = history4_energy

        correlation1_real += tl.where(
            valid1,
            state_real * history1_real + state_imag * history1_imag,
            0.0,
        )
        correlation1_imag += tl.where(valid1, corr1_imag, 0.0)
        current1_energy += tl.where(valid1, active_current1_energy, 0.0)
        previous1_energy += tl.where(valid1, active_previous1_energy, 0.0)
        correlation2_real += tl.where(
            valid2,
            state_real * history2_real + state_imag * history2_imag,
            0.0,
        )
        correlation2_imag += tl.where(valid2, corr2_imag, 0.0)
        current2_energy += tl.where(valid2, active_current2_energy, 0.0)
        previous2_energy += tl.where(valid2, active_previous2_energy, 0.0)
        correlation4_real += tl.where(
            valid4,
            state_real * history4_real + state_imag * history4_imag,
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
        history1_real = state_real
        history1_imag = state_imag
        step += 1

    count1 = tl.maximum(n_steps - 1, 1)
    count2 = tl.maximum(n_steps - 2, 1)
    count4 = tl.maximum(n_steps - 4, 1)
    denominator1 = tl.maximum(
        tl.sqrt((current1_energy / count1) * (previous1_energy / count1)),
        epsilon,
    )
    denominator2 = tl.maximum(
        tl.sqrt((current2_energy / count2) * (previous2_energy / count2)),
        epsilon,
    )
    denominator4 = tl.maximum(
        tl.sqrt((current4_energy / count4) * (previous4_energy / count4)),
        epsilon,
    )
    moment_base = batch * 7 * modes + mode
    tl.store(moment_output + moment_base, libdevice.log1p(energy_sum / n_steps), mask=valid_mode)
    tl.store(
        moment_output + moment_base + modes,
        tl.where(n_steps > 1, (correlation1_real / count1) / denominator1, 0.0),
        mask=valid_mode,
    )
    tl.store(
        moment_output + moment_base + 2 * modes,
        tl.where(n_steps > 1, (correlation1_imag / count1) / denominator1, 0.0),
        mask=valid_mode,
    )
    tl.store(
        moment_output + moment_base + 3 * modes,
        tl.where(n_steps > 2, (correlation2_real / count2) / denominator2, 0.0),
        mask=valid_mode,
    )
    tl.store(
        moment_output + moment_base + 4 * modes,
        tl.where(n_steps > 2, (correlation2_imag / count2) / denominator2, 0.0),
        mask=valid_mode,
    )
    tl.store(
        moment_output + moment_base + 5 * modes,
        tl.where(n_steps > 4, (correlation4_real / count4) / denominator4, 0.0),
        mask=valid_mode,
    )
    tl.store(
        moment_output + moment_base + 6 * modes,
        tl.where(n_steps > 4, (correlation4_imag / count4) / denominator4, 0.0),
        mask=valid_mode,
    )


def _launch_cuda(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    *,
    reverse: bool,
    epsilon: float,
    single_warp: bool,
    store_states: bool,
    wrapped: bool,
) -> tuple[Tensor | None, Tensor]:
    real = decay_real.contiguous()
    imag = decay_imag.contiguous()
    drive = packed_input.contiguous()
    batch, n_steps, packed_modes = drive.shape
    modes = packed_modes // 2
    moments = torch.empty((batch, 7 * modes), dtype=drive.dtype, device=drive.device)
    packed_states = torch.empty_like(drive) if store_states else None
    state_output = packed_states if packed_states is not None else moments
    block_modes = _select_lag124_block_modes(modes, batch=batch, n_steps=n_steps)
    grid = _mode_grid(batch, modes, block_modes)
    if wrapped:
        wrap_triton(_static_recurrence_lag124_kernel)[grid](
            real,
            imag,
            drive,
            state_output,
            moments,
            n_steps,
            modes,
            epsilon,
            reverse=reverse,
            store_states=store_states,
            BLOCK_MODES=block_modes,
            num_warps=1 if single_warp else 4,
        )
    else:
        _static_recurrence_lag124_kernel[grid](
            real,
            imag,
            drive,
            state_output,
            moments,
            n_steps,
            modes,
            epsilon,
            reverse=reverse,
            store_states=store_states,
            BLOCK_MODES=block_modes,
            num_warps=1 if single_warp else 4,
        )
    return packed_states, moments


@triton_op("lnet::pac_static_real2d_recurrence_lag124_moments_only", mutates_args={})
def _static_recurrence_lag124_moments_only_op(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    *,
    reverse: bool,
    epsilon: float,
    single_warp: bool,
) -> Tensor:
    _validate_static_packed_inputs(decay_real, decay_imag, packed_input, epsilon)
    if not packed_input.is_cuda or packed_input.dtype != torch.float32:
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


@triton_op("lnet::pac_static_real2d_recurrence_lag124_moments_packed_io", mutates_args={})
def _static_recurrence_lag124_moments_packed_io_op(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    *,
    reverse: bool,
    epsilon: float,
    single_warp: bool,
) -> tuple[Tensor, Tensor]:
    _validate_static_packed_inputs(decay_real, decay_imag, packed_input, epsilon)
    if not packed_input.is_cuda or packed_input.dtype != torch.float32:
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
        message = "packed-state CUDA launch did not produce states"
        raise RuntimeError(message)
    return packed_states, moments


def _needs_gradients(*tensors: Tensor) -> bool:
    return torch.is_grad_enabled() and any(tensor.requires_grad for tensor in tensors)


def static_recurrence_lag124_moments_only_inference(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    *,
    reverse: bool = False,
    epsilon: float = _EPSILON,
    single_warp: bool = False,
) -> Tensor:
    """Return seven fixed-pole moments without materializing recurrence states."""
    _validate_static_packed_inputs(decay_real, decay_imag, packed_input, epsilon)
    use_cuda_kernel = (
        packed_input.is_cuda
        and packed_input.dtype == torch.float32
        and not _needs_gradients(decay_real, decay_imag, packed_input)
    )
    if not use_cuda_kernel:
        return reference_static_recurrence_lag124_moments_only(
            decay_real,
            decay_imag,
            packed_input,
            reverse=reverse,
            epsilon=epsilon,
        )
    if torch.compiler.is_compiling():
        return _static_recurrence_lag124_moments_only_op(
            decay_real,
            decay_imag,
            packed_input,
            reverse=reverse,
            epsilon=epsilon,
            single_warp=single_warp,
        )
    _, moments = _launch_cuda(
        decay_real,
        decay_imag,
        packed_input,
        reverse=reverse,
        epsilon=epsilon,
        single_warp=single_warp,
        store_states=False,
        wrapped=False,
    )
    return moments


def static_recurrence_lag124_moments_packed_io_inference(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    *,
    reverse: bool = False,
    epsilon: float = _EPSILON,
    single_warp: bool = False,
) -> tuple[Tensor, Tensor]:
    """Return packed fixed-pole states and seven canonical moments per mode."""
    _validate_static_packed_inputs(decay_real, decay_imag, packed_input, epsilon)
    use_cuda_kernel = (
        packed_input.is_cuda
        and packed_input.dtype == torch.float32
        and not _needs_gradients(decay_real, decay_imag, packed_input)
    )
    if not use_cuda_kernel:
        return reference_static_recurrence_lag124_moments_packed_io(
            decay_real,
            decay_imag,
            packed_input,
            reverse=reverse,
            epsilon=epsilon,
        )
    if torch.compiler.is_compiling():
        return _static_recurrence_lag124_moments_packed_io_op(
            decay_real,
            decay_imag,
            packed_input,
            reverse=reverse,
            epsilon=epsilon,
            single_warp=single_warp,
        )
    packed_states, moments = _launch_cuda(
        decay_real,
        decay_imag,
        packed_input,
        reverse=reverse,
        epsilon=epsilon,
        single_warp=single_warp,
        store_states=True,
        wrapped=False,
    )
    if packed_states is None:
        message = "packed-state CUDA launch did not produce states"
        raise RuntimeError(message)
    return packed_states, moments


__all__ = [
    "reference_static_recurrence_lag124_moments_only",
    "reference_static_recurrence_lag124_moments_packed_io",
    "static_recurrence_lag124_moments_only_inference",
    "static_recurrence_lag124_moments_packed_io_inference",
]
