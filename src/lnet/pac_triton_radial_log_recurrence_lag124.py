"""Static recurrence fused with raw lag-(1,2,4) radial-log moments.

This is the absolute-autocorrelation counterpart of the normalized q+C kernel.
It deliberately keeps only the seven accumulators required by R0/R1/R2/R4 and
applies the radial-log map in the store epilogue.
"""

from __future__ import annotations

# pyright: reportCallIssue=false, reportMissingParameterType=false
# ruff: noqa: ANN001, N803, PLR0915
from typing import Final

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton
from triton.language.extra import libdevice

from .pac_triton_recurrence_lag124 import (
    _select_lag124_block_modes,
    _validate_static_packed_inputs,
)
from .pac_triton_recurrence_op import _mode_grid

_VALIDATION_EPSILON: Final[float] = 1.0e-8


def _radial_log_reference(raw_moments: Tensor, modes: int) -> Tensor:
    transformed = [torch.log1p(raw_moments[..., :modes].clamp_min(0.0))]
    tiny = torch.finfo(raw_moments.dtype).tiny
    for offset in (modes, 3 * modes, 5 * modes):
        real = raw_moments[..., offset : offset + modes]
        imag = raw_moments[..., offset + modes : offset + 2 * modes]
        radius = torch.sqrt((real.square() + imag.square()).clamp_min(tiny))
        scale = torch.log1p(radius) / radius
        transformed.extend((scale * real, scale * imag))
    return torch.cat(transformed, dim=-1)


def _reference_static_radial_log_recurrence_lag124(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    *,
    reverse: bool,
    store_states: bool,
) -> tuple[Tensor | None, Tensor]:
    _validate_static_packed_inputs(
        decay_real,
        decay_imag,
        packed_input,
        _VALIDATION_EPSILON,
    )
    modes = decay_real.numel()
    drive_real, drive_imag = packed_input.split(modes, dim=-1)
    accumulator_dtype = (
        torch.float64 if packed_input.dtype == torch.float64 else torch.float32
    )
    active_decay_real = decay_real.to(dtype=accumulator_dtype)
    active_decay_imag = decay_imag.to(dtype=accumulator_dtype)
    state_real = torch.zeros_like(drive_real[:, 0], dtype=accumulator_dtype)
    state_imag = torch.zeros_like(drive_imag[:, 0], dtype=accumulator_dtype)
    history_real = [torch.zeros_like(state_real) for _ in range(4)]
    history_imag = [torch.zeros_like(state_imag) for _ in range(4)]
    energy_sum = torch.zeros_like(state_real)
    correlations = [
        [torch.zeros_like(state_real), torch.zeros_like(state_real)]
        for _ in range(3)
    ]
    n_steps = packed_input.shape[1]
    states: list[Tensor | None] | None = (
        [None for _ in range(n_steps)] if store_states else None
    )
    traversal = range(n_steps - 1, -1, -1) if reverse else range(n_steps)

    for step, time_index in enumerate(traversal):
        previous_real = state_real
        previous_imag = state_imag
        current_drive_real = drive_real[:, time_index].to(dtype=accumulator_dtype)
        current_drive_imag = drive_imag[:, time_index].to(dtype=accumulator_dtype)
        state_real = (
            active_decay_real * previous_real
            - active_decay_imag * previous_imag
            + current_drive_real
        )
        state_imag = (
            active_decay_imag * previous_real
            + active_decay_real * previous_imag
            + current_drive_imag
        )
        observed_real = state_real.to(packed_input.dtype).to(accumulator_dtype)
        observed_imag = state_imag.to(packed_input.dtype).to(accumulator_dtype)
        if states is not None:
            states[time_index] = torch.cat((observed_real, observed_imag), dim=-1).to(
                packed_input.dtype
            )
        energy_sum = energy_sum + observed_real.square() + observed_imag.square()
        for index, lag in enumerate((1, 2, 4)):
            if step < lag:
                continue
            lagged_real = history_real[lag - 1]
            lagged_imag = history_imag[lag - 1]
            correlations[index][0] = correlations[index][0] + (
                observed_real * lagged_real + observed_imag * lagged_imag
            )
            if reverse:
                correlations[index][1] = correlations[index][1] + (
                    lagged_imag * observed_real - lagged_real * observed_imag
                )
            else:
                correlations[index][1] = correlations[index][1] + (
                    observed_imag * lagged_real - observed_real * lagged_imag
                )
        history_real = [observed_real, *history_real[:3]]
        history_imag = [observed_imag, *history_imag[:3]]

    raw = [energy_sum / n_steps]
    for index, lag in enumerate((1, 2, 4)):
        count = n_steps - lag
        if count < 1:
            raw.extend((torch.zeros_like(state_real), torch.zeros_like(state_real)))
        else:
            raw.extend(
                (
                    correlations[index][0] / count,
                    correlations[index][1] / count,
                )
            )
    moments = _radial_log_reference(torch.cat(raw, dim=-1), modes).to(
        packed_input.dtype
    )
    packed_states = None
    if states is not None:
        if any(state is None for state in states):
            raise RuntimeError("radial-log recurrence did not populate every state")
        packed_states = torch.stack(
            [state for state in states if state is not None],
            dim=1,
        )
    return packed_states, moments


def reference_static_radial_log_recurrence_lag124_moments_only(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    *,
    reverse: bool = False,
) -> Tensor:
    """Reference radial-log moments without materializing states."""
    return _reference_static_radial_log_recurrence_lag124(
        decay_real,
        decay_imag,
        packed_input,
        reverse=reverse,
        store_states=False,
    )[1]


def reference_static_radial_log_recurrence_lag124_moments_packed_io(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    *,
    reverse: bool = False,
) -> tuple[Tensor, Tensor]:
    """Reference packed states plus radial-log moments."""
    states, moments = _reference_static_radial_log_recurrence_lag124(
        decay_real,
        decay_imag,
        packed_input,
        reverse=reverse,
        store_states=True,
    )
    if states is None:
        raise RuntimeError("radial-log packed-state reference produced no states")
    return states, moments


@triton.jit
def _static_radial_log_recurrence_lag124_kernel(
    decay_real,
    decay_imag,
    packed_input,
    packed_states,
    moment_output,
    n_steps: int,
    modes: int,
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
    active_decay_real = tl.load(
        decay_real + mode,
        mask=valid_mode,
        other=0.0,
    ).to(tl.float32)
    active_decay_imag = tl.load(
        decay_imag + mode,
        mask=valid_mode,
        other=0.0,
    ).to(tl.float32)

    step = 0
    while step < n_steps:
        time_index = n_steps - 1 - step if reverse else step
        packed_offset = (batch * n_steps + time_index) * 2 * modes + mode
        drive_real = tl.load(
            packed_input + packed_offset,
            mask=valid_mode,
            other=0.0,
        ).to(tl.float32)
        drive_imag = tl.load(
            packed_input + packed_offset + modes,
            mask=valid_mode,
            other=0.0,
        ).to(tl.float32)
        previous_real = state_real
        previous_imag = state_imag
        state_real = (
            active_decay_real * previous_real
            - active_decay_imag * previous_imag
            + drive_real
        )
        state_imag = (
            active_decay_imag * previous_real
            + active_decay_real * previous_imag
            + drive_imag
        )
        if store_states:
            tl.store(packed_states + packed_offset, state_real, mask=valid_mode)
            tl.store(
                packed_states + packed_offset + modes,
                state_imag,
                mask=valid_mode,
            )

        energy_sum += state_real * state_real + state_imag * state_imag
        valid1 = step >= 1
        valid2 = step >= 2
        valid4 = step >= 4
        if reverse:
            corr1_imag = history1_imag * state_real - history1_real * state_imag
            corr2_imag = history2_imag * state_real - history2_real * state_imag
            corr4_imag = history4_imag * state_real - history4_real * state_imag
        else:
            corr1_imag = state_imag * history1_real - state_real * history1_imag
            corr2_imag = state_imag * history2_real - state_real * history2_imag
            corr4_imag = state_imag * history4_real - state_real * history4_imag
        correlation1_real += tl.where(
            valid1,
            state_real * history1_real + state_imag * history1_imag,
            0.0,
        )
        correlation1_imag += tl.where(valid1, corr1_imag, 0.0)
        correlation2_real += tl.where(
            valid2,
            state_real * history2_real + state_imag * history2_imag,
            0.0,
        )
        correlation2_imag += tl.where(valid2, corr2_imag, 0.0)
        correlation4_real += tl.where(
            valid4,
            state_real * history4_real + state_imag * history4_imag,
            0.0,
        )
        correlation4_imag += tl.where(valid4, corr4_imag, 0.0)
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
    raw1_real = tl.where(n_steps > 1, correlation1_real / count1, 0.0)
    raw1_imag = tl.where(n_steps > 1, correlation1_imag / count1, 0.0)
    raw2_real = tl.where(n_steps > 2, correlation2_real / count2, 0.0)
    raw2_imag = tl.where(n_steps > 2, correlation2_imag / count2, 0.0)
    raw4_real = tl.where(n_steps > 4, correlation4_real / count4, 0.0)
    raw4_imag = tl.where(n_steps > 4, correlation4_imag / count4, 0.0)
    radius1 = tl.sqrt(
        tl.maximum(
            raw1_real * raw1_real + raw1_imag * raw1_imag,
            1.1754943508222875e-38,
        )
    )
    radius2 = tl.sqrt(
        tl.maximum(
            raw2_real * raw2_real + raw2_imag * raw2_imag,
            1.1754943508222875e-38,
        )
    )
    radius4 = tl.sqrt(
        tl.maximum(
            raw4_real * raw4_real + raw4_imag * raw4_imag,
            1.1754943508222875e-38,
        )
    )
    scale1 = libdevice.log1p(radius1) / radius1
    scale2 = libdevice.log1p(radius2) / radius2
    scale4 = libdevice.log1p(radius4) / radius4
    moment_base = batch * 7 * modes + mode
    tl.store(
        moment_output + moment_base,
        libdevice.log1p(energy_sum / n_steps),
        mask=valid_mode,
    )
    tl.store(
        moment_output + moment_base + modes,
        scale1 * raw1_real,
        mask=valid_mode,
    )
    tl.store(
        moment_output + moment_base + 2 * modes,
        scale1 * raw1_imag,
        mask=valid_mode,
    )
    tl.store(
        moment_output + moment_base + 3 * modes,
        scale2 * raw2_real,
        mask=valid_mode,
    )
    tl.store(
        moment_output + moment_base + 4 * modes,
        scale2 * raw2_imag,
        mask=valid_mode,
    )
    tl.store(
        moment_output + moment_base + 5 * modes,
        scale4 * raw4_real,
        mask=valid_mode,
    )
    tl.store(
        moment_output + moment_base + 6 * modes,
        scale4 * raw4_imag,
        mask=valid_mode,
    )


def _launch_cuda(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    *,
    reverse: bool,
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
        wrap_triton(_static_radial_log_recurrence_lag124_kernel)[grid](
            real,
            imag,
            drive,
            state_output,
            moments,
            n_steps,
            modes,
            reverse=reverse,
            store_states=store_states,
            BLOCK_MODES=block_modes,
            num_warps=1 if single_warp else 4,
        )
    else:
        _static_radial_log_recurrence_lag124_kernel[grid](
            real,
            imag,
            drive,
            state_output,
            moments,
            n_steps,
            modes,
            reverse=reverse,
            store_states=store_states,
            BLOCK_MODES=block_modes,
            num_warps=1 if single_warp else 4,
        )
    return packed_states, moments


@triton_op(
    "lnet::pac_static_radial_log_recurrence_lag124_moments_only",
    mutates_args={},
)
def _moments_only_op(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    *,
    reverse: bool,
    single_warp: bool,
) -> Tensor:
    _validate_static_packed_inputs(
        decay_real,
        decay_imag,
        packed_input,
        _VALIDATION_EPSILON,
    )
    if not packed_input.is_cuda or packed_input.dtype != torch.float32:
        return reference_static_radial_log_recurrence_lag124_moments_only(
            decay_real,
            decay_imag,
            packed_input,
            reverse=reverse,
        )
    _, moments = _launch_cuda(
        decay_real,
        decay_imag,
        packed_input,
        reverse=reverse,
        single_warp=single_warp,
        store_states=False,
        wrapped=True,
    )
    return moments


@triton_op(
    "lnet::pac_static_radial_log_recurrence_lag124_moments_packed_io",
    mutates_args={},
)
def _packed_io_op(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    *,
    reverse: bool,
    single_warp: bool,
) -> tuple[Tensor, Tensor]:
    _validate_static_packed_inputs(
        decay_real,
        decay_imag,
        packed_input,
        _VALIDATION_EPSILON,
    )
    if not packed_input.is_cuda or packed_input.dtype != torch.float32:
        return reference_static_radial_log_recurrence_lag124_moments_packed_io(
            decay_real,
            decay_imag,
            packed_input,
            reverse=reverse,
        )
    states, moments = _launch_cuda(
        decay_real,
        decay_imag,
        packed_input,
        reverse=reverse,
        single_warp=single_warp,
        store_states=True,
        wrapped=True,
    )
    if states is None:
        raise RuntimeError("radial-log CUDA launch produced no states")
    return states, moments


@torch.library.custom_op(
    "lnet::pac_static_radial_log_recurrence_lag124_moments_only_opaque",
    mutates_args=(),
)
def _moments_only_opaque(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    reverse: bool,
    single_warp: bool,
) -> Tensor:
    return _moments_only_op(
        decay_real,
        decay_imag,
        packed_input,
        reverse=reverse,
        single_warp=single_warp,
    )


@_moments_only_opaque.register_fake
def _moments_only_opaque_fake(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    reverse: bool,
    single_warp: bool,
) -> Tensor:
    del decay_imag, reverse, single_warp
    return packed_input.new_empty(
        (packed_input.shape[0], 7 * decay_real.numel())
    )


@torch.library.custom_op(
    "lnet::pac_static_radial_log_recurrence_lag124_moments_packed_io_opaque",
    mutates_args=(),
)
def _packed_io_opaque(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    reverse: bool,
    single_warp: bool,
) -> tuple[Tensor, Tensor]:
    return _packed_io_op(
        decay_real,
        decay_imag,
        packed_input,
        reverse=reverse,
        single_warp=single_warp,
    )


@_packed_io_opaque.register_fake
def _packed_io_opaque_fake(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    reverse: bool,
    single_warp: bool,
) -> tuple[Tensor, Tensor]:
    del decay_imag, reverse, single_warp
    return (
        torch.empty_like(packed_input),
        packed_input.new_empty(
            (packed_input.shape[0], 7 * decay_real.numel())
        ),
    )


def _needs_gradients(*tensors: Tensor) -> bool:
    return torch.is_grad_enabled() and any(tensor.requires_grad for tensor in tensors)


def static_radial_log_recurrence_lag124_moments_only_inference(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    *,
    reverse: bool = False,
    single_warp: bool = False,
) -> Tensor:
    """Return radial-log R0/R1/R2/R4 without constructing recurrence states."""
    _validate_static_packed_inputs(
        decay_real,
        decay_imag,
        packed_input,
        _VALIDATION_EPSILON,
    )
    use_kernel = (
        packed_input.is_cuda
        and packed_input.dtype == torch.float32
        and not _needs_gradients(decay_real, decay_imag, packed_input)
    )
    if not use_kernel:
        return reference_static_radial_log_recurrence_lag124_moments_only(
            decay_real,
            decay_imag,
            packed_input,
            reverse=reverse,
        )
    if torch.compiler.is_compiling():
        return _moments_only_opaque(
            decay_real,
            decay_imag,
            packed_input,
            reverse,
            single_warp,
        )
    _, moments = _launch_cuda(
        decay_real,
        decay_imag,
        packed_input,
        reverse=reverse,
        single_warp=single_warp,
        store_states=False,
        wrapped=False,
    )
    return moments


def static_radial_log_recurrence_lag124_moments_packed_io_inference(
    decay_real: Tensor,
    decay_imag: Tensor,
    packed_input: Tensor,
    *,
    reverse: bool = False,
    single_warp: bool = False,
) -> tuple[Tensor, Tensor]:
    """Return packed states and radial-log R0/R1/R2/R4."""
    _validate_static_packed_inputs(
        decay_real,
        decay_imag,
        packed_input,
        _VALIDATION_EPSILON,
    )
    use_kernel = (
        packed_input.is_cuda
        and packed_input.dtype == torch.float32
        and not _needs_gradients(decay_real, decay_imag, packed_input)
    )
    if not use_kernel:
        return reference_static_radial_log_recurrence_lag124_moments_packed_io(
            decay_real,
            decay_imag,
            packed_input,
            reverse=reverse,
        )
    if torch.compiler.is_compiling():
        return _packed_io_opaque(
            decay_real,
            decay_imag,
            packed_input,
            reverse,
            single_warp,
        )
    states, moments = _launch_cuda(
        decay_real,
        decay_imag,
        packed_input,
        reverse=reverse,
        single_warp=single_warp,
        store_states=True,
        wrapped=False,
    )
    if states is None:
        raise RuntimeError("radial-log CUDA launch produced no states")
    return states, moments


__all__ = [
    "reference_static_radial_log_recurrence_lag124_moments_only",
    "reference_static_radial_log_recurrence_lag124_moments_packed_io",
    "static_radial_log_recurrence_lag124_moments_only_inference",
    "static_radial_log_recurrence_lag124_moments_packed_io_inference",
]
