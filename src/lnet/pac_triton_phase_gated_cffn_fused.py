"""Materialization-free narrow Phase-Gated complex FFN."""

from __future__ import annotations

# pyright: reportArgumentType=false, reportCallIssue=false, reportGeneralTypeIssues=false, reportIncompatibleMethodOverride=false, reportMissingParameterType=false, reportPrivateUsage=false
# ruff: noqa: ANN001, ANN202, EM101, N803, PLR0915, SLF001, TRY003
from typing import Protocol

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton
from triton.language.extra import libdevice

from . import pac_triton_complex_rmsnorm as triton_rmsnorm
from . import pac_triton_phase_gate as triton_phase_gate
from .pac_kernel_launch_config import (
    LaunchGeometry,
    autotuned,
    make_launch_scope,
    register_default,
)
from .pac_reduction_tiling import device_parameter_reduction_rows
from .pac_triton_hardware import (
    device_supports_single_warp_dot_tiles,
    diagnostic_sample_rows,
)
from .pac_triton_phase_gate_residual_fused import (
    BACKWARD_REDUCE_LAUNCH_NAME as RESIDUAL_BACKWARD_REDUCE_LAUNCH_NAME,
)
from .pac_triton_phase_gate_residual_fused import (
    _phase_gate_output_residual_backward_reduce_kernel,
)
from .pac_triton_rmsnorm_linear_fused import (
    _normalized_coordinates,
    _row_layout,
    _source_offset,
    _supports_row_layout,
)

FORWARD_LAUNCH_NAME = "phase_gated_cffn_fused_forward"
BACKWARD_LAUNCH_NAME = "phase_gated_cffn_fused_backward"

_FORWARD_CANDIDATES = tuple(
    LaunchGeometry.build(num_warps=warps, blocks={"BLOCK_ROWS": block_rows})
    for block_rows, warps in (
        (16, 4),
        (16, 8),
        (32, 4),
        (32, 8),
        (64, 4),
        (64, 8),
    )
)
_BACKWARD_CANDIDATES = tuple(
    LaunchGeometry.build(num_warps=warps, blocks={"BLOCK_ROWS": block_rows})
    for block_rows, warps in (
        (16, 4),
        (16, 8),
        (32, 4),
        (32, 8),
        (64, 4),
        (64, 8),
        (128, 8),
        (256, 4),
        (256, 8),
    )
)
register_default(
    FORWARD_LAUNCH_NAME,
    LaunchGeometry.build(num_warps=4, blocks={"BLOCK_ROWS": 32}),
    candidates=_FORWARD_CANDIDATES,
)
register_default(
    BACKWARD_LAUNCH_NAME,
    LaunchGeometry.build(num_warps=4, blocks={"BLOCK_ROWS": 64}),
    candidates=_BACKWARD_CANDIDATES,
)


class _AutogradContext(Protocol):
    saved_tensors: tuple[Tensor, ...]
    epsilon: float
    redistribution: float
    self_gated: bool

    def save_for_backward(self, *tensors: Tensor) -> None: ...

    def mark_non_differentiable(self, *tensors: Tensor) -> None: ...


def supports_fused_phase_gated_cffn(
    real: Tensor,
    imag: Tensor,
    norm_weight: Tensor,
    input_weight: Tensor,
    alpha: Tensor,
    output_weight: Tensor,
    gamma: Tensor,
    *,
    epsilon: float,
    redistribution: float,
    self_gated: bool,
) -> bool:
    """Return whether one hardware-bounded tile owns the complete FFN."""
    modes = real.shape[-1] if real.ndim >= 1 else 0
    hidden = alpha.numel() if alpha.ndim == 1 else 0
    projected_width = 2 * hidden if self_gated else 4 * hidden
    return (
        real.is_cuda
        and real.dtype in (torch.float32, torch.bfloat16)
        and real.shape == imag.shape
        and real.numel() > 0
        and real.stride() == imag.stride()
        and _supports_row_layout(real)
        and norm_weight.device == real.device
        and norm_weight.dtype is torch.float32
        and norm_weight.shape == (modes,)
        and norm_weight.is_contiguous()
        and input_weight.device == real.device
        and input_weight.dtype is torch.bfloat16
        and input_weight.shape == (projected_width, 2 * modes)
        and input_weight.is_contiguous()
        and alpha.device == real.device
        and alpha.dtype is torch.float32
        and alpha.shape == (hidden,)
        and alpha.is_contiguous()
        and output_weight.device == real.device
        and output_weight.dtype is torch.bfloat16
        and output_weight.shape == (2 * modes, 2 * hidden)
        and output_weight.is_contiguous()
        and gamma.device == real.device
        and gamma.dtype is torch.float32
        and gamma.shape == ()
        and gamma.is_contiguous()
        and device_supports_single_warp_dot_tiles(
            real,
            2 * modes,
            projected_width,
            2 * hidden,
        )
        and epsilon > 0.0
        and 0.0 < redistribution < 1.0
        and not torch.are_deterministic_algorithms_enabled()
    )


@triton.jit
def _project_component(
    input_weight,
    normalized_real,
    normalized_imag,
    output_coordinate,
    source_mode,
    output_mask,
    MODES: tl.constexpr,
):
    weight_offset = output_coordinate[:, None] * (2 * MODES) + source_mode[None, :]
    active_mask = output_mask[:, None] & (source_mode[None, :] < MODES)
    weight_real = tl.load(input_weight + weight_offset, mask=active_mask, other=0.0).to(tl.bfloat16)
    weight_imag = tl.load(
        input_weight + weight_offset + MODES,
        mask=active_mask,
        other=0.0,
    ).to(tl.bfloat16)
    projected = tl.dot(normalized_real, tl.trans(weight_real))
    projected += tl.dot(normalized_imag, tl.trans(weight_imag))
    return projected.to(tl.bfloat16)


@triton.jit
def _input_projection_backward_component(
    input_weight,
    grad_input_weight,
    grad_component,
    normalized_real,
    normalized_imag,
    output_coordinate,
    source_mode,
    output_mask,
    source_mask,
    MODES: tl.constexpr,
):
    weight_offset = output_coordinate[:, None] * (2 * MODES) + source_mode[None, :]
    weight_mask = output_mask[:, None] & (source_mode[None, :] < MODES)
    weight_real = tl.load(input_weight + weight_offset, mask=weight_mask, other=0.0).to(tl.bfloat16)
    weight_imag = tl.load(
        input_weight + weight_offset + MODES,
        mask=weight_mask,
        other=0.0,
    ).to(tl.bfloat16)
    grad_normalized_real = tl.dot(grad_component, weight_real).to(tl.bfloat16)
    grad_normalized_imag = tl.dot(grad_component, weight_imag).to(tl.bfloat16)
    grad_weight_real = tl.dot(tl.trans(grad_component), normalized_real)
    grad_weight_imag = tl.dot(tl.trans(grad_component), normalized_imag)
    tl.atomic_add(
        grad_input_weight + weight_offset,
        grad_weight_real,
        mask=weight_mask,
    )
    tl.atomic_add(
        grad_input_weight + weight_offset + MODES,
        grad_weight_imag,
        mask=weight_mask,
    )
    return (
        tl.where(source_mask, grad_normalized_real, 0.0),
        tl.where(source_mask, grad_normalized_imag, 0.0),
    )


@triton.jit
def _projected_coordinates(
    input_weight,
    normalized_real,
    normalized_imag,
    hidden_mode,
    source_mode,
    hidden_mask,
    MODES: tl.constexpr,
    HIDDEN: tl.constexpr,
    SELF_GATED: tl.constexpr,
):
    value_real = _project_component(
        input_weight,
        normalized_real,
        normalized_imag,
        hidden_mode,
        source_mode,
        hidden_mode < HIDDEN,
        MODES,
    )
    value_imag_offset = HIDDEN if SELF_GATED else 2 * HIDDEN
    value_imag = _project_component(
        input_weight,
        normalized_real,
        normalized_imag,
        hidden_mode + value_imag_offset,
        source_mode,
        hidden_mode < HIDDEN,
        MODES,
    )
    if SELF_GATED:
        gate_real = value_real
        gate_imag = value_imag
    else:
        gate_real = _project_component(
            input_weight,
            normalized_real,
            normalized_imag,
            hidden_mode + HIDDEN,
            source_mode,
            hidden_mode < HIDDEN,
            MODES,
        )
        gate_imag = _project_component(
            input_weight,
            normalized_real,
            normalized_imag,
            hidden_mode + 3 * HIDDEN,
            source_mode,
            hidden_mode < HIDDEN,
            MODES,
        )
    return (
        tl.where(hidden_mask, value_real, 0.0),
        tl.where(hidden_mask, value_imag, 0.0),
        tl.where(hidden_mask, gate_real, 0.0),
        tl.where(hidden_mask, gate_imag, 0.0),
    )


@triton.jit
def _gate_from_coordinates(
    value_real,
    value_imag,
    gate_real,
    gate_imag,
    alpha,
    hidden_mode,
    hidden_mask,
    HIDDEN: tl.constexpr,
    RHO: tl.constexpr,
):
    gate_real_fp32 = gate_real.to(tl.float32)
    gate_imag_fp32 = gate_imag.to(tl.float32)
    denominator = 1.0 + gate_real_fp32 * gate_real_fp32 + gate_imag_fp32 * gate_imag_fp32
    magnitude = tl.where(hidden_mask, tl.log(denominator), 0.0)
    centered = magnitude - tl.sum(magnitude, axis=1)[:, None] / HIDDEN
    active_alpha = tl.load(alpha + hidden_mode, mask=hidden_mode < HIDDEN, other=0.0)
    tangent = libdevice.tanh(active_alpha * centered)
    relative = tl.where(hidden_mask, 1.0 + RHO * tangent, 0.0)
    mean_relative = tl.maximum(tl.sum(relative, axis=1) / HIDDEN, 1.0e-6)
    gate_fp32 = tl.where(hidden_mask, relative / mean_relative[:, None], 0.0)
    gate = gate_fp32.to(tl.bfloat16)
    return (
        denominator,
        centered,
        active_alpha,
        tangent,
        mean_relative,
        gate_fp32,
        gate,
        (value_real * gate).to(tl.bfloat16),
        (value_imag * gate).to(tl.bfloat16),
    )


@triton.jit
def _phase_gated_cffn_forward_kernel(
    real,
    imag,
    norm_weight,
    input_weight,
    alpha,
    output_weight,
    gamma,
    output_real,
    output_imag,
    diagnostic_projected,
    diagnostic_delta,
    rows: int,
    diagnostic_rows: int,
    inner_rows: int,
    outer_row_stride: int,
    inner_row_stride: int,
    mode_stride: int,
    MODES: tl.constexpr,
    HIDDEN: tl.constexpr,
    PROJECTED_WIDTH: tl.constexpr,
    OUTPUT_WIDTH: tl.constexpr,
    EPSILON: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_MODES: tl.constexpr,
    BLOCK_HIDDEN: tl.constexpr,
    BLOCK_OUTPUT: tl.constexpr,
    RHO: tl.constexpr,
    SELF_GATED: tl.constexpr,
    CONTIGUOUS_SOURCE: tl.constexpr,
    SOURCE_IS_BF16: tl.constexpr,
) -> None:
    row_id = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    row = row_id[:, None]
    source_mode = tl.arange(0, BLOCK_MODES)
    source_mask = (row < rows) & (source_mode[None, :] < MODES)
    (
        _,
        _,
        _,
        _,
        _,
        normalized_real,
        normalized_imag,
    ) = _normalized_coordinates(
        real,
        imag,
        norm_weight,
        row,
        source_mode[None, :],
        source_mask,
        MODES,
        EPSILON,
        inner_rows,
        outer_row_stride,
        inner_row_stride,
        mode_stride,
        CONTIGUOUS_SOURCE,
        SOURCE_IS_BF16,
    )
    hidden_mode = tl.arange(0, BLOCK_HIDDEN)
    hidden_mask = (row < rows) & (hidden_mode[None, :] < HIDDEN)
    value_real, value_imag, gate_real, gate_imag = _projected_coordinates(
        input_weight,
        normalized_real,
        normalized_imag,
        hidden_mode,
        source_mode,
        hidden_mask,
        MODES,
        HIDDEN,
        SELF_GATED,
    )
    (
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        hidden_real,
        hidden_imag,
    ) = _gate_from_coordinates(
        value_real,
        value_imag,
        gate_real,
        gate_imag,
        alpha,
        hidden_mode[None, :],
        hidden_mask,
        HIDDEN,
        RHO,
    )

    diagnostic_mask = hidden_mask & (row < diagnostic_rows)
    diagnostic_base = row * PROJECTED_WIDTH + hidden_mode[None, :]
    tl.store(diagnostic_projected + diagnostic_base, value_real, mask=diagnostic_mask)
    value_imag_offset = HIDDEN if SELF_GATED else 2 * HIDDEN
    tl.store(
        diagnostic_projected + diagnostic_base + value_imag_offset,
        value_imag,
        mask=diagnostic_mask,
    )
    if not SELF_GATED:
        tl.store(
            diagnostic_projected + diagnostic_base + HIDDEN,
            gate_real,
            mask=diagnostic_mask,
        )
        tl.store(
            diagnostic_projected + diagnostic_base + 3 * HIDDEN,
            gate_imag,
            mask=diagnostic_mask,
        )

    output_coordinate = tl.arange(0, BLOCK_OUTPUT)
    output_mask = output_coordinate < OUTPUT_WIDTH
    weight_offset = output_coordinate[:, None] * (2 * HIDDEN) + hidden_mode[None, :]
    weight_mask = output_mask[:, None] & (hidden_mode[None, :] < HIDDEN)
    weight_real = tl.load(output_weight + weight_offset, mask=weight_mask, other=0.0).to(
        tl.bfloat16
    )
    weight_imag = tl.load(
        output_weight + weight_offset + HIDDEN,
        mask=weight_mask,
        other=0.0,
    ).to(tl.bfloat16)
    delta = tl.dot(hidden_real, tl.trans(weight_real))
    delta += tl.dot(hidden_imag, tl.trans(weight_imag))
    delta = delta.to(tl.bfloat16)
    packed_offset = row * OUTPUT_WIDTH + output_coordinate[None, :]
    active_output_mask = (row < rows) & output_mask[None, :]
    tl.store(
        diagnostic_delta + packed_offset,
        delta,
        mask=active_output_mask & (row < diagnostic_rows),
    )

    residual_mode = output_coordinate % MODES
    residual_offset = _source_offset(
        row,
        residual_mode[None, :],
        MODES,
        inner_rows,
        outer_row_stride,
        inner_row_stride,
        mode_stride,
        CONTIGUOUS_SOURCE,
    )
    source_real = tl.load(real + residual_offset, mask=active_output_mask, other=0.0)
    source_imag = tl.load(imag + residual_offset, mask=active_output_mask, other=0.0)
    source = tl.where(output_coordinate[None, :] < MODES, source_real, source_imag)
    active_gamma = tl.load(gamma).to(tl.bfloat16)
    output = source + (delta * active_gamma).to(tl.bfloat16)
    tl.store(
        output_real + residual_offset,
        output,
        mask=active_output_mask & (output_coordinate[None, :] < MODES),
    )
    tl.store(
        output_imag + residual_offset,
        output,
        mask=active_output_mask & (output_coordinate[None, :] >= MODES),
    )


@triton.jit
def _phase_gated_cffn_backward_kernel(
    real,
    imag,
    norm_weight,
    input_weight,
    alpha,
    output_weight,
    gamma,
    output_grad_real,
    output_grad_imag,
    grad_real,
    grad_imag,
    partial_grad_norm_weight,
    grad_input_weight,
    partial_grad_alpha,
    grad_output_weight,
    partial_grad_gamma,
    rows: int,
    source_inner_rows: int,
    source_outer_row_stride: int,
    source_inner_row_stride: int,
    source_mode_stride: int,
    grad_inner_rows: int,
    grad_outer_row_stride: int,
    grad_inner_row_stride: int,
    grad_mode_stride: int,
    MODES: tl.constexpr,
    HIDDEN: tl.constexpr,
    OUTPUT_WIDTH: tl.constexpr,
    EPSILON: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_MODES: tl.constexpr,
    BLOCK_HIDDEN: tl.constexpr,
    BLOCK_OUTPUT: tl.constexpr,
    PARTIAL_ROWS: tl.constexpr,
    RHO: tl.constexpr,
    SELF_GATED: tl.constexpr,
    CONTIGUOUS_SOURCE: tl.constexpr,
    CONTIGUOUS_GRADIENT: tl.constexpr,
    SOURCE_IS_BF16: tl.constexpr,
) -> None:
    row_block = tl.program_id(0)
    row_id = row_block * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    row = row_id[:, None]
    source_mode = tl.arange(0, BLOCK_MODES)
    source_mask = (row < rows) & (source_mode[None, :] < MODES)
    (
        _,
        active_real,
        active_imag,
        inverse_rms,
        active_norm_weight,
        normalized_real,
        normalized_imag,
    ) = _normalized_coordinates(
        real,
        imag,
        norm_weight,
        row,
        source_mode[None, :],
        source_mask,
        MODES,
        EPSILON,
        source_inner_rows,
        source_outer_row_stride,
        source_inner_row_stride,
        source_mode_stride,
        CONTIGUOUS_SOURCE,
        SOURCE_IS_BF16,
    )
    hidden_mode = tl.arange(0, BLOCK_HIDDEN)
    hidden_mask = (row < rows) & (hidden_mode[None, :] < HIDDEN)
    value_real, value_imag, gate_real, gate_imag = _projected_coordinates(
        input_weight,
        normalized_real,
        normalized_imag,
        hidden_mode,
        source_mode,
        hidden_mask,
        MODES,
        HIDDEN,
        SELF_GATED,
    )
    (
        denominator,
        centered,
        active_alpha,
        tangent,
        mean_relative,
        gate_fp32,
        gate,
        hidden_real,
        hidden_imag,
    ) = _gate_from_coordinates(
        value_real,
        value_imag,
        gate_real,
        gate_imag,
        alpha,
        hidden_mode[None, :],
        hidden_mask,
        HIDDEN,
        RHO,
    )

    output_coordinate = tl.arange(0, BLOCK_OUTPUT)
    output_mask = output_coordinate < OUTPUT_WIDTH
    residual_mode = output_coordinate % MODES
    gradient_offset = _source_offset(
        row,
        residual_mode[None, :],
        MODES,
        grad_inner_rows,
        grad_outer_row_stride,
        grad_inner_row_stride,
        grad_mode_stride,
        CONTIGUOUS_GRADIENT,
    )
    active_output_mask = (row < rows) & output_mask[None, :]
    incoming_real = tl.load(
        output_grad_real + gradient_offset,
        mask=active_output_mask,
        other=0.0,
    )
    incoming_imag = tl.load(
        output_grad_imag + gradient_offset,
        mask=active_output_mask,
        other=0.0,
    )
    raw_grad_output = tl.where(
        output_coordinate[None, :] < MODES,
        incoming_real,
        incoming_imag,
    ).to(tl.bfloat16)
    active_gamma = tl.load(gamma).to(tl.bfloat16)
    scaled_grad_output = (raw_grad_output * active_gamma).to(tl.bfloat16)

    output_weight_offset = output_coordinate[:, None] * (2 * HIDDEN) + hidden_mode[None, :]
    output_weight_mask = output_mask[:, None] & (hidden_mode[None, :] < HIDDEN)
    output_weight_real = tl.load(
        output_weight + output_weight_offset,
        mask=output_weight_mask,
        other=0.0,
    ).to(tl.bfloat16)
    output_weight_imag = tl.load(
        output_weight + output_weight_offset + HIDDEN,
        mask=output_weight_mask,
        other=0.0,
    ).to(tl.bfloat16)
    unscaled_grad_value_real = tl.dot(raw_grad_output, output_weight_real).to(tl.bfloat16)
    unscaled_grad_value_imag = tl.dot(raw_grad_output, output_weight_imag).to(tl.bfloat16)
    grad_value_real = (unscaled_grad_value_real * active_gamma).to(tl.bfloat16)
    grad_value_imag = (unscaled_grad_value_imag * active_gamma).to(tl.bfloat16)

    direct_real = (grad_value_real * gate).to(tl.bfloat16)
    direct_imag = (grad_value_imag * gate).to(tl.bfloat16)
    grad_gate = (
        (
            (grad_value_real * value_real).to(tl.bfloat16)
            + (grad_value_imag * value_imag).to(tl.bfloat16)
        )
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    weighted_mean = tl.sum(tl.where(hidden_mask, grad_gate * gate_fp32, 0.0), axis=1) / HIDDEN
    grad_relative = (grad_gate - weighted_mean[:, None]) / mean_relative[:, None]
    grad_logits = grad_relative * RHO * (1.0 - tangent * tangent)
    grad_centered = grad_logits * active_alpha
    grad_magnitude = grad_centered - (
        tl.sum(tl.where(hidden_mask, grad_centered, 0.0), axis=1)[:, None] / HIDDEN
    )
    magnitude_real = grad_magnitude * 2.0 * gate_real.to(tl.float32) / denominator
    magnitude_imag = grad_magnitude * 2.0 * gate_imag.to(tl.float32) / denominator
    direct_real = tl.where(hidden_mask, direct_real, 0.0)
    direct_imag = tl.where(hidden_mask, direct_imag, 0.0)
    magnitude_real = tl.where(hidden_mask, magnitude_real, 0.0)
    magnitude_imag = tl.where(hidden_mask, magnitude_imag, 0.0)
    if SELF_GATED:
        grad_projected_real = (direct_real + magnitude_real.to(tl.bfloat16)).to(tl.bfloat16)
        grad_projected_imag = (direct_imag + magnitude_imag.to(tl.bfloat16)).to(tl.bfloat16)
        grad_normalized_real, grad_normalized_imag = _input_projection_backward_component(
            input_weight,
            grad_input_weight,
            grad_projected_real,
            normalized_real,
            normalized_imag,
            hidden_mode,
            source_mode,
            hidden_mode < HIDDEN,
            source_mask,
            MODES,
        )
        component_real, component_imag = _input_projection_backward_component(
            input_weight,
            grad_input_weight,
            grad_projected_imag,
            normalized_real,
            normalized_imag,
            hidden_mode + HIDDEN,
            source_mode,
            hidden_mode < HIDDEN,
            source_mask,
            MODES,
        )
        grad_normalized_real += component_real
        grad_normalized_imag += component_imag
    else:
        grad_normalized_real, grad_normalized_imag = _input_projection_backward_component(
            input_weight,
            grad_input_weight,
            direct_real,
            normalized_real,
            normalized_imag,
            hidden_mode,
            source_mode,
            hidden_mode < HIDDEN,
            source_mask,
            MODES,
        )
        component_real, component_imag = _input_projection_backward_component(
            input_weight,
            grad_input_weight,
            magnitude_real.to(tl.bfloat16),
            normalized_real,
            normalized_imag,
            hidden_mode + HIDDEN,
            source_mode,
            hidden_mode < HIDDEN,
            source_mask,
            MODES,
        )
        grad_normalized_real += component_real
        grad_normalized_imag += component_imag
        component_real, component_imag = _input_projection_backward_component(
            input_weight,
            grad_input_weight,
            direct_imag,
            normalized_real,
            normalized_imag,
            hidden_mode + 2 * HIDDEN,
            source_mode,
            hidden_mode < HIDDEN,
            source_mask,
            MODES,
        )
        grad_normalized_real += component_real
        grad_normalized_imag += component_imag
        component_real, component_imag = _input_projection_backward_component(
            input_weight,
            grad_input_weight,
            magnitude_imag.to(tl.bfloat16),
            normalized_real,
            normalized_imag,
            hidden_mode + 3 * HIDDEN,
            source_mode,
            hidden_mode < HIDDEN,
            source_mask,
            MODES,
        )
        grad_normalized_real += component_real
        grad_normalized_imag += component_imag

    if SOURCE_IS_BF16:
        active_inverse_rms = inverse_rms.to(tl.bfloat16).to(tl.float32)
        active_scale = active_norm_weight.to(tl.bfloat16).to(tl.float32)
        normalized_base_real = (
            (active_real.to(tl.bfloat16) * inverse_rms[:, None].to(tl.bfloat16))
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        normalized_base_imag = (
            (active_imag.to(tl.bfloat16) * inverse_rms[:, None].to(tl.bfloat16))
            .to(tl.bfloat16)
            .to(tl.float32)
        )
    else:
        active_inverse_rms = inverse_rms
        active_scale = active_norm_weight
        normalized_base_real = active_real * inverse_rms[:, None]
        normalized_base_imag = active_imag * inverse_rms[:, None]
    scaled_grad_real = grad_normalized_real.to(tl.float32) * active_scale
    scaled_grad_imag = grad_normalized_imag.to(tl.float32) * active_scale
    radial = tl.sum(
        scaled_grad_real * active_real + scaled_grad_imag * active_imag,
        axis=1,
    )
    correction = inverse_rms * inverse_rms * inverse_rms * radial / MODES
    branch_grad_real = (
        active_inverse_rms[:, None] * scaled_grad_real - active_real * correction[:, None]
    )
    branch_grad_imag = (
        active_inverse_rms[:, None] * scaled_grad_imag - active_imag * correction[:, None]
    )
    direct_gradient_offset = _source_offset(
        row,
        source_mode[None, :],
        MODES,
        grad_inner_rows,
        grad_outer_row_stride,
        grad_inner_row_stride,
        grad_mode_stride,
        CONTIGUOUS_GRADIENT,
    )
    direct_source_real = tl.load(
        output_grad_real + direct_gradient_offset,
        mask=source_mask,
        other=0.0,
    ).to(tl.float32)
    direct_source_imag = tl.load(
        output_grad_imag + direct_gradient_offset,
        mask=source_mask,
        other=0.0,
    ).to(tl.float32)
    contiguous_source_offset = row * MODES + source_mode[None, :]
    tl.store(
        grad_real + contiguous_source_offset,
        branch_grad_real + direct_source_real,
        mask=source_mask,
    )
    tl.store(
        grad_imag + contiguous_source_offset,
        branch_grad_imag + direct_source_imag,
        mask=source_mask,
    )

    partial_block = (row_block * BLOCK_ROWS) // PARTIAL_ROWS
    active_source_mode = tl.arange(0, BLOCK_MODES)
    grad_weight_contribution = (
        grad_normalized_real.to(tl.float32) * normalized_base_real
        + grad_normalized_imag.to(tl.float32) * normalized_base_imag
    )
    partial_scale = tl.sum(tl.where(source_mask, grad_weight_contribution, 0.0), axis=0)
    partial_scale_offset = partial_block * MODES + active_source_mode
    partial_alpha_offset = partial_block * HIDDEN + hidden_mode
    partial_alpha = tl.sum(tl.where(hidden_mask, grad_logits * centered, 0.0), axis=0)
    gamma_contribution = unscaled_grad_value_real.to(tl.float32) * hidden_real.to(
        tl.float32
    ) + unscaled_grad_value_imag.to(tl.float32) * hidden_imag.to(tl.float32)
    partial_gamma = tl.sum(
        tl.sum(tl.where(hidden_mask, gamma_contribution, 0.0), axis=1),
        axis=0,
    )
    if BLOCK_ROWS >= PARTIAL_ROWS:
        tl.store(
            partial_grad_norm_weight + partial_scale_offset,
            partial_scale,
            mask=active_source_mode < MODES,
        )
        tl.store(
            partial_grad_alpha + partial_alpha_offset,
            partial_alpha,
            mask=hidden_mode < HIDDEN,
        )
        tl.store(partial_grad_gamma + partial_block, partial_gamma)
    else:
        tl.atomic_add(
            partial_grad_norm_weight + partial_scale_offset,
            partial_scale,
            mask=active_source_mode < MODES,
        )
        tl.atomic_add(
            partial_grad_alpha + partial_alpha_offset,
            partial_alpha,
            mask=hidden_mode < HIDDEN,
        )
        tl.atomic_add(partial_grad_gamma + partial_block, partial_gamma)

    grad_output_weight_real = tl.dot(tl.trans(scaled_grad_output), hidden_real)
    grad_output_weight_imag = tl.dot(tl.trans(scaled_grad_output), hidden_imag)
    tl.atomic_add(
        grad_output_weight + output_weight_offset,
        grad_output_weight_real,
        mask=output_weight_mask,
    )
    tl.atomic_add(
        grad_output_weight + output_weight_offset + HIDDEN,
        grad_output_weight_imag,
        mask=output_weight_mask,
    )


def _scope(kernel: object, real: Tensor, hidden: int):
    inner_rows, outer_stride, inner_stride, mode_stride = _row_layout(real)
    return make_launch_scope(
        kernel,
        real,
        shape={
            "rows": real.numel() // real.shape[-1],
            "modes": real.shape[-1],
            "hidden": hidden,
            "inner_rows": inner_rows,
            "outer_stride": outer_stride,
            "inner_stride": inner_stride,
            "mode_stride": mode_stride,
        },
    )


@triton_op("lnet::phase_gated_cffn_fused", mutates_args={})
def _forward_op(
    real: Tensor,
    imag: Tensor,
    norm_weight: Tensor,
    input_weight: Tensor,
    alpha: Tensor,
    output_weight: Tensor,
    gamma: Tensor,
    epsilon: float,
    redistribution: float,
    self_gated: bool,  # noqa: FBT001
    collect_diagnostics: bool,  # noqa: FBT001
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if not supports_fused_phase_gated_cffn(
        real,
        imag,
        norm_weight,
        input_weight,
        alpha,
        output_weight,
        gamma,
        epsilon=epsilon,
        redistribution=redistribution,
        self_gated=self_gated,
    ):
        raise RuntimeError("unsupported complete Phase-Gated CFFN fusion contract")
    modes = real.shape[-1]
    hidden = alpha.numel()
    projected_width = input_weight.shape[0]
    output_width = output_weight.shape[0]
    rows = real.numel() // modes
    diagnostic_rows = diagnostic_sample_rows(real) if collect_diagnostics else 0
    inner_rows, outer_stride, inner_stride, mode_stride = _row_layout(real)
    output_real = torch.empty_strided(
        real.shape,
        real.stride(),
        device=real.device,
        dtype=real.dtype,
    )
    output_imag = torch.empty_strided(
        imag.shape,
        imag.stride(),
        device=imag.device,
        dtype=imag.dtype,
    )
    diagnostic_projected = torch.empty(
        (diagnostic_rows, projected_width),
        device=real.device,
        dtype=torch.bfloat16,
    )
    diagnostic_delta = torch.empty(
        (diagnostic_rows, output_width),
        device=real.device,
        dtype=torch.bfloat16,
    )
    kernel = autotuned(
        _phase_gated_cffn_forward_kernel,
        FORWARD_LAUNCH_NAME,
        key=(
            "rows",
            "MODES",
            "HIDDEN",
            "PROJECTED_WIDTH",
            "OUTPUT_WIDTH",
            "EPSILON",
            "RHO",
            "SELF_GATED",
            "inner_rows",
            "outer_row_stride",
            "inner_row_stride",
            "mode_stride",
            "SOURCE_IS_BF16",
        ),
        scope=_scope(_phase_gated_cffn_forward_kernel, real, hidden),
    )

    def grid(metadata: dict[str, int]) -> tuple[int]:
        return (int(triton.cdiv(rows, metadata["BLOCK_ROWS"])),)

    wrap_triton(kernel)[grid](
        real,
        imag,
        norm_weight,
        input_weight,
        alpha,
        output_weight,
        gamma,
        output_real,
        output_imag,
        diagnostic_projected,
        diagnostic_delta,
        rows,
        diagnostic_rows,
        inner_rows,
        outer_stride,
        inner_stride,
        mode_stride,
        MODES=modes,
        HIDDEN=hidden,
        PROJECTED_WIDTH=projected_width,
        OUTPUT_WIDTH=output_width,
        EPSILON=epsilon,
        BLOCK_MODES=max(16, triton.next_power_of_2(modes)),
        BLOCK_HIDDEN=max(16, triton.next_power_of_2(hidden)),
        BLOCK_OUTPUT=max(16, triton.next_power_of_2(output_width)),
        RHO=redistribution,
        SELF_GATED=self_gated,
        CONTIGUOUS_SOURCE=real.is_contiguous(),
        SOURCE_IS_BF16=real.dtype is torch.bfloat16,
    )
    return output_real, output_imag, diagnostic_projected, diagnostic_delta


@triton_op("lnet::phase_gated_cffn_fused_backward", mutates_args={})
def _backward_op(
    real: Tensor,
    imag: Tensor,
    norm_weight: Tensor,
    input_weight: Tensor,
    alpha: Tensor,
    output_weight: Tensor,
    gamma: Tensor,
    output_grad_real: Tensor,
    output_grad_imag: Tensor,
    epsilon: float,
    redistribution: float,
    self_gated: bool,  # noqa: FBT001
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    if (
        output_grad_real.shape != output_grad_imag.shape
        or output_grad_real.stride() != output_grad_imag.stride()
        or not _supports_row_layout(output_grad_real)
    ):
        output_grad_real = output_grad_real.contiguous()
        output_grad_imag = output_grad_imag.contiguous()
    modes = real.shape[-1]
    hidden = alpha.numel()
    output_width = output_weight.shape[0]
    rows = real.numel() // modes
    source_inner_rows, source_outer_stride, source_inner_stride, source_mode_stride = _row_layout(
        real
    )
    grad_inner_rows, grad_outer_stride, grad_inner_stride, grad_mode_stride = _row_layout(
        output_grad_real
    )
    grad_real = torch.empty(real.shape, device=real.device, dtype=real.dtype)
    grad_imag = torch.empty(imag.shape, device=imag.device, dtype=imag.dtype)
    partial_rows = device_parameter_reduction_rows(real, rows)
    partial_count = int(triton.cdiv(rows, partial_rows))
    partial_grad_norm_weight = torch.zeros(
        (partial_count, modes),
        device=real.device,
        dtype=torch.float32,
    )
    partial_grad_alpha = torch.zeros(
        (partial_count, hidden),
        device=real.device,
        dtype=torch.float32,
    )
    partial_grad_gamma = torch.zeros(partial_count, device=real.device, dtype=torch.float32)
    grad_input_weight_fp32 = torch.zeros_like(input_weight, dtype=torch.float32)
    grad_output_weight_fp32 = torch.zeros_like(output_weight, dtype=torch.float32)
    kernel = autotuned(
        _phase_gated_cffn_backward_kernel,
        BACKWARD_LAUNCH_NAME,
        key=(
            "rows",
            "MODES",
            "HIDDEN",
            "OUTPUT_WIDTH",
            "EPSILON",
            "RHO",
            "SELF_GATED",
            "source_inner_rows",
            "source_outer_row_stride",
            "source_inner_row_stride",
            "source_mode_stride",
            "grad_inner_rows",
            "grad_outer_row_stride",
            "grad_inner_row_stride",
            "grad_mode_stride",
            "SOURCE_IS_BF16",
        ),
        scope=_scope(_phase_gated_cffn_backward_kernel, real, hidden),
        reset_to_zero=(
            "partial_grad_norm_weight",
            "grad_input_weight",
            "partial_grad_alpha",
            "grad_output_weight",
            "partial_grad_gamma",
        ),
    )

    def grid(metadata: dict[str, int]) -> tuple[int]:
        return (int(triton.cdiv(rows, metadata["BLOCK_ROWS"])),)

    wrap_triton(kernel)[grid](
        real,
        imag,
        norm_weight,
        input_weight,
        alpha,
        output_weight,
        gamma,
        output_grad_real,
        output_grad_imag,
        grad_real,
        grad_imag,
        partial_grad_norm_weight,
        grad_input_weight_fp32,
        partial_grad_alpha,
        grad_output_weight_fp32,
        partial_grad_gamma,
        rows,
        source_inner_rows,
        source_outer_stride,
        source_inner_stride,
        source_mode_stride,
        grad_inner_rows,
        grad_outer_stride,
        grad_inner_stride,
        grad_mode_stride,
        MODES=modes,
        HIDDEN=hidden,
        OUTPUT_WIDTH=output_width,
        EPSILON=epsilon,
        BLOCK_MODES=max(16, triton.next_power_of_2(modes)),
        BLOCK_HIDDEN=max(16, triton.next_power_of_2(hidden)),
        BLOCK_OUTPUT=max(16, triton.next_power_of_2(output_width)),
        PARTIAL_ROWS=partial_rows,
        RHO=redistribution,
        SELF_GATED=self_gated,
        CONTIGUOUS_SOURCE=real.is_contiguous(),
        CONTIGUOUS_GRADIENT=output_grad_real.is_contiguous(),
        SOURCE_IS_BF16=real.dtype is torch.bfloat16,
    )

    grad_norm_weight = torch.empty_like(norm_weight)
    scale_reduce_kernel = autotuned(
        triton_rmsnorm._packed_complex_rmsnorm_backward_reduce_kernel,
        triton_rmsnorm.BACKWARD_REDUCE_LAUNCH_NAME,
        key=("partial_count", "modes"),
        scope=make_launch_scope(
            triton_rmsnorm._packed_complex_rmsnorm_backward_reduce_kernel,
            partial_grad_norm_weight,
            shape={"partial_count": partial_count, "modes": modes},
        ),
    )

    def scale_grid(metadata: dict[str, int]) -> tuple[int]:
        return (int(triton.cdiv(modes, metadata["BLOCK_MODES"])),)

    wrap_triton(scale_reduce_kernel)[scale_grid](
        partial_grad_norm_weight,
        grad_norm_weight,
        partial_count,
        modes,
    )
    grad_alpha = torch.empty_like(alpha)
    alpha_reduce_kernel = autotuned(
        triton_phase_gate._phase_gate_backward_reduce_kernel,
        triton_phase_gate.BACKWARD_REDUCE_LAUNCH_NAME,
        key=("partial_count", "hidden"),
        scope=make_launch_scope(
            triton_phase_gate._phase_gate_backward_reduce_kernel,
            partial_grad_alpha,
            shape={"partial_count": partial_count, "hidden": hidden},
        ),
    )

    def alpha_grid(metadata: dict[str, int]) -> tuple[int]:
        return (int(triton.cdiv(hidden, metadata["BLOCK_HIDDEN"])),)

    wrap_triton(alpha_reduce_kernel)[alpha_grid](
        partial_grad_alpha,
        grad_alpha,
        partial_count,
        hidden,
    )
    grad_gamma = torch.empty_like(gamma)
    gamma_reduce_kernel = autotuned(
        _phase_gate_output_residual_backward_reduce_kernel,
        RESIDUAL_BACKWARD_REDUCE_LAUNCH_NAME,
        key=("partial_count",),
        scope=make_launch_scope(
            _phase_gate_output_residual_backward_reduce_kernel,
            partial_grad_gamma,
            shape={"partial_count": partial_count},
        ),
    )
    wrap_triton(gamma_reduce_kernel)[(1,)](
        partial_grad_gamma,
        grad_gamma,
        partial_count,
    )
    return (
        grad_real,
        grad_imag,
        grad_norm_weight,
        grad_input_weight_fp32.to(input_weight.dtype),
        grad_alpha,
        grad_output_weight_fp32.to(output_weight.dtype),
        grad_gamma,
    )


def _setup_context(
    ctx: _AutogradContext,
    inputs: tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        float,
        float,
        bool,
        bool,
    ],
    output: tuple[Tensor, Tensor, Tensor, Tensor],
) -> None:
    (
        real,
        imag,
        norm_weight,
        input_weight,
        alpha,
        output_weight,
        gamma,
        epsilon,
        rho,
        self_gated,
        _,
    ) = inputs
    _, _, diagnostic_projected, diagnostic_delta = output
    ctx.save_for_backward(real, imag, norm_weight, input_weight, alpha, output_weight, gamma)
    ctx.mark_non_differentiable(diagnostic_projected, diagnostic_delta)
    ctx.epsilon = epsilon
    ctx.redistribution = rho
    ctx.self_gated = self_gated


def _backward(
    ctx: _AutogradContext,
    output_grad_real: Tensor | None,
    output_grad_imag: Tensor | None,
    diagnostic_projected_grad: Tensor | None,
    diagnostic_delta_grad: Tensor | None,
) -> tuple[
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    None,
    None,
    None,
    None,
]:
    del diagnostic_projected_grad, diagnostic_delta_grad
    real, imag, norm_weight, input_weight, alpha, output_weight, gamma = ctx.saved_tensors
    if output_grad_real is None:
        output_grad_real = torch.zeros_like(real)
    if output_grad_imag is None:
        output_grad_imag = torch.zeros_like(imag)
    gradients = _backward_op(
        real,
        imag,
        norm_weight,
        input_weight,
        alpha,
        output_weight,
        gamma,
        output_grad_real,
        output_grad_imag,
        ctx.epsilon,
        ctx.redistribution,
        ctx.self_gated,
    )
    return *gradients, None, None, None, None


_forward_op.register_autograd(  # pyright: ignore[reportFunctionMemberAccess]
    _backward,
    setup_context=_setup_context,
)


def fused_phase_gated_cffn(
    real: Tensor,
    imag: Tensor,
    norm_weight: Tensor,
    input_weight: Tensor,
    alpha: Tensor,
    output_weight: Tensor,
    gamma: Tensor,
    *,
    epsilon: float,
    redistribution: float,
    self_gated: bool,
    collect_diagnostics: bool = True,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Execute the complete narrow residual FFN without projected activations."""
    return _forward_op(
        real,
        imag,
        norm_weight,
        input_weight,
        alpha + 0.0,
        output_weight,
        gamma,
        epsilon,
        redistribution,
        self_gated,
        collect_diagnostics,
    )


__all__ = [
    "BACKWARD_LAUNCH_NAME",
    "FORWARD_LAUNCH_NAME",
    "fused_phase_gated_cffn",
    "supports_fused_phase_gated_cffn",
]
