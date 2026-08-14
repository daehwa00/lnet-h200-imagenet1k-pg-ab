"""Materialization-free complex RMSNorm and narrow input projection."""

from __future__ import annotations

# pyright: reportArgumentType=false, reportCallIssue=false, reportGeneralTypeIssues=false, reportMissingParameterType=false
# ruff: noqa: ANN001, ANN202, EM101, N803, PLR0915, TRY003
from typing import Protocol

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton

from . import pac_triton_complex_rmsnorm as triton_rmsnorm
from .pac_kernel_launch_config import (
    LaunchGeometry,
    autotuned,
    make_launch_scope,
    register_default,
)
from .pac_reduction_tiling import device_parameter_reduction_rows
from .pac_triton_hardware import device_supports_single_warp_dot_tiles

FORWARD_LAUNCH_NAME = "rmsnorm_input_linear_fused_forward"
BACKWARD_LAUNCH_NAME = "rmsnorm_input_linear_fused_backward"
_FORWARD_CANDIDATES = tuple(
    LaunchGeometry.build(num_warps=warps, blocks={"BLOCK_ROWS": block_rows})
    for block_rows, warps in ((16, 4), (32, 4), (64, 4), (64, 8), (128, 8))
)
_BACKWARD_CANDIDATES = tuple(
    LaunchGeometry.build(num_warps=warps, blocks={"BLOCK_ROWS": block_rows})
    for block_rows, warps in ((32, 4), (64, 4), (64, 8), (128, 4), (128, 8))
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

    def save_for_backward(self, *tensors: Tensor) -> None: ...


def _supports_row_layout(tensor: Tensor) -> bool:
    return tensor.is_contiguous() or (tensor.ndim >= 2 and tensor.transpose(-2, -1).is_contiguous())


def _row_layout(tensor: Tensor) -> tuple[int, int, int, int]:
    if tensor.ndim == 1:
        return 1, 0, 0, tensor.stride(-1)
    inner_rows = tensor.shape[-2]
    outer_row_stride = tensor.stride(-3) if tensor.ndim >= 3 else 0
    return inner_rows, outer_row_stride, tensor.stride(-2), tensor.stride(-1)


def supports_fused_rmsnorm_input_linear(
    real: Tensor,
    imag: Tensor,
    norm_weight: Tensor,
    input_weight: Tensor,
    *,
    epsilon: float,
) -> bool:
    """Return whether one hardware-bounded dot tile represents the operation."""
    modes = real.shape[-1] if real.ndim >= 1 else 0
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
        and input_weight.ndim == 2
        and input_weight.shape[1] == 2 * modes
        and input_weight.is_contiguous()
        and device_supports_single_warp_dot_tiles(
            real,
            2 * modes,
            input_weight.shape[0],
        )
        and epsilon > 0.0
        and not torch.are_deterministic_algorithms_enabled()
    )


def _scope(kernel: object, real: Tensor, input_weight: Tensor):
    modes = real.shape[-1]
    rows = real.numel() // modes
    inner_rows, outer_stride, inner_stride, mode_stride = _row_layout(real)
    return make_launch_scope(
        kernel,
        real,
        shape={
            "rows": rows,
            "modes": modes,
            "output_width": input_weight.shape[0],
            "inner_rows": inner_rows,
            "outer_stride": outer_stride,
            "inner_stride": inner_stride,
            "mode_stride": mode_stride,
        },
    )


@triton.jit
def _source_offset(
    row,
    mode,
    modes: tl.constexpr,
    inner_rows,
    outer_row_stride,
    inner_row_stride,
    mode_stride,
    CONTIGUOUS_SOURCE: tl.constexpr,
):
    if CONTIGUOUS_SOURCE:
        return row * modes + mode
    outer_row = row // inner_rows
    inner_row = row % inner_rows
    return outer_row * outer_row_stride + inner_row * inner_row_stride + mode * mode_stride


@triton.jit
def _normalized_coordinates(
    real,
    imag,
    norm_weight,
    row,
    mode,
    mask,
    modes: tl.constexpr,
    epsilon: tl.constexpr,
    inner_rows,
    outer_row_stride,
    inner_row_stride,
    mode_stride,
    CONTIGUOUS_SOURCE: tl.constexpr,
    SOURCE_IS_BF16: tl.constexpr,
):
    offset = _source_offset(
        row,
        mode,
        modes,
        inner_rows,
        outer_row_stride,
        inner_row_stride,
        mode_stride,
        CONTIGUOUS_SOURCE,
    )
    active_real = tl.load(real + offset, mask=mask, other=0.0).to(tl.float32)
    active_imag = tl.load(imag + offset, mask=mask, other=0.0).to(tl.float32)
    energy = tl.sum(active_real * active_real + active_imag * active_imag, axis=1)
    inverse_rms = tl.rsqrt(energy / modes + epsilon)
    active_weight = tl.load(norm_weight + mode, mask=mode < modes, other=0.0)
    if SOURCE_IS_BF16:
        active_inverse_rms = inverse_rms.to(tl.bfloat16)
        scale = active_weight.to(tl.bfloat16)
        normalized_real = (
            (active_real.to(tl.bfloat16) * active_inverse_rms[:, None]).to(tl.bfloat16)
            * scale
        ).to(tl.bfloat16)
        normalized_imag = (
            (active_imag.to(tl.bfloat16) * active_inverse_rms[:, None]).to(tl.bfloat16)
            * scale
        ).to(tl.bfloat16)
    else:
        normalized_real = (active_real * inverse_rms[:, None] * active_weight).to(tl.bfloat16)
        normalized_imag = (active_imag * inverse_rms[:, None] * active_weight).to(tl.bfloat16)
    return (
        offset,
        active_real,
        active_imag,
        inverse_rms,
        active_weight,
        normalized_real,
        normalized_imag,
    )


@triton.jit
def _fused_forward_kernel(
    real,
    imag,
    norm_weight,
    input_weight,
    projected,
    row_inverse_rms,
    rows: int,
    inner_rows: int,
    outer_row_stride: int,
    inner_row_stride: int,
    mode_stride: int,
    MODES: tl.constexpr,
    OUTPUT_WIDTH: tl.constexpr,
    EPSILON: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_MODES: tl.constexpr,
    BLOCK_OUTPUT: tl.constexpr,
    CONTIGUOUS_SOURCE: tl.constexpr,
    SOURCE_IS_BF16: tl.constexpr,
) -> None:
    row_id = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    row = row_id[:, None]
    mode = tl.arange(0, BLOCK_MODES)[None, :]
    mask = (row < rows) & (mode < MODES)
    (
        _,
        _,
        _,
        inverse_rms,
        _,
        normalized_real,
        normalized_imag,
    ) = _normalized_coordinates(
        real,
        imag,
        norm_weight,
        row,
        mode,
        mask,
        MODES,
        EPSILON,
        inner_rows,
        outer_row_stride,
        inner_row_stride,
        mode_stride,
        CONTIGUOUS_SOURCE,
        SOURCE_IS_BF16,
    )
    output_coordinate = tl.arange(0, BLOCK_OUTPUT)
    output_mask = output_coordinate < OUTPUT_WIDTH
    weight_offset = output_coordinate[:, None] * (2 * MODES) + mode
    weight_real = tl.load(
        input_weight + weight_offset,
        mask=output_mask[:, None] & (mode < MODES),
        other=0.0,
    ).to(tl.bfloat16)
    weight_imag = tl.load(
        input_weight + weight_offset + MODES,
        mask=output_mask[:, None] & (mode < MODES),
        other=0.0,
    ).to(tl.bfloat16)
    active_projected = tl.dot(normalized_real, tl.trans(weight_real))
    active_projected += tl.dot(normalized_imag, tl.trans(weight_imag))
    projected_offset = row * OUTPUT_WIDTH + output_coordinate[None, :]
    tl.store(
        projected + projected_offset,
        active_projected.to(tl.bfloat16),
        mask=(row < rows) & output_mask[None, :],
    )
    tl.store(row_inverse_rms + row_id, inverse_rms, mask=row_id < rows)


@triton.jit
def _fused_backward_kernel(
    real,
    imag,
    norm_weight,
    input_weight,
    row_inverse_rms,
    grad_projected,
    grad_real,
    grad_imag,
    partial_grad_norm_weight,
    grad_input_weight,
    rows: int,
    inner_rows: int,
    outer_row_stride: int,
    inner_row_stride: int,
    mode_stride: int,
    MODES: tl.constexpr,
    OUTPUT_WIDTH: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_MODES: tl.constexpr,
    BLOCK_OUTPUT: tl.constexpr,
    PARTIAL_ROWS: tl.constexpr,
    CONTIGUOUS_SOURCE: tl.constexpr,
    SOURCE_IS_BF16: tl.constexpr,
) -> None:
    row_block = tl.program_id(0)
    row_id = row_block * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    row = row_id[:, None]
    active_mode = tl.arange(0, BLOCK_MODES)
    mode = active_mode[None, :]
    mask = (row < rows) & (mode < MODES)
    source_offset = _source_offset(
        row,
        mode,
        MODES,
        inner_rows,
        outer_row_stride,
        inner_row_stride,
        mode_stride,
        CONTIGUOUS_SOURCE,
    )
    active_real = tl.load(real + source_offset, mask=mask, other=0.0).to(tl.float32)
    active_imag = tl.load(imag + source_offset, mask=mask, other=0.0).to(tl.float32)
    inverse_rms = tl.load(row_inverse_rms + row_id, mask=row_id < rows, other=0.0).to(
        tl.float32
    )
    active_weight = tl.load(norm_weight + mode, mask=mode < MODES, other=0.0)
    if SOURCE_IS_BF16:
        active_inverse_rms = inverse_rms.to(tl.bfloat16).to(tl.float32)
        active_scale = active_weight.to(tl.bfloat16).to(tl.float32)
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
        active_scale = active_weight
        normalized_base_real = active_real * inverse_rms[:, None]
        normalized_base_imag = active_imag * inverse_rms[:, None]
    normalized_real = (normalized_base_real * active_scale).to(tl.bfloat16)
    normalized_imag = (normalized_base_imag * active_scale).to(tl.bfloat16)

    output_coordinate = tl.arange(0, BLOCK_OUTPUT)
    output_mask = output_coordinate < OUTPUT_WIDTH
    projected_offset = row * OUTPUT_WIDTH + output_coordinate[None, :]
    active_grad_projected = tl.load(
        grad_projected + projected_offset,
        mask=(row < rows) & output_mask[None, :],
        other=0.0,
    ).to(tl.bfloat16)
    weight_offset = output_coordinate[:, None] * (2 * MODES) + mode
    weight_real = tl.load(
        input_weight + weight_offset,
        mask=output_mask[:, None] & (mode < MODES),
        other=0.0,
    ).to(tl.bfloat16)
    weight_imag = tl.load(
        input_weight + weight_offset + MODES,
        mask=output_mask[:, None] & (mode < MODES),
        other=0.0,
    ).to(tl.bfloat16)
    grad_normalized_real = tl.dot(active_grad_projected, weight_real).to(tl.bfloat16)
    grad_normalized_imag = tl.dot(active_grad_projected, weight_imag).to(tl.bfloat16)

    scaled_grad_real = grad_normalized_real.to(tl.float32) * active_scale
    scaled_grad_imag = grad_normalized_imag.to(tl.float32) * active_scale
    radial = tl.sum(
        scaled_grad_real * active_real + scaled_grad_imag * active_imag,
        axis=1,
    )
    correction = inverse_rms * inverse_rms * inverse_rms * radial / MODES
    active_grad_real = (
        active_inverse_rms[:, None] * scaled_grad_real - active_real * correction[:, None]
    )
    active_grad_imag = (
        active_inverse_rms[:, None] * scaled_grad_imag - active_imag * correction[:, None]
    )
    contiguous_offset = row * MODES + mode
    tl.store(grad_real + contiguous_offset, active_grad_real, mask=mask)
    tl.store(grad_imag + contiguous_offset, active_grad_imag, mask=mask)

    grad_weight_contribution = (
        grad_normalized_real.to(tl.float32) * normalized_base_real
        + grad_normalized_imag.to(tl.float32) * normalized_base_imag
    )
    partial = tl.sum(tl.where(mask, grad_weight_contribution, 0.0), axis=0)
    partial_block = (row_block * BLOCK_ROWS) // PARTIAL_ROWS
    partial_offset = partial_block * MODES + active_mode
    if BLOCK_ROWS >= PARTIAL_ROWS:
        tl.store(
            partial_grad_norm_weight + partial_offset,
            partial,
            mask=active_mode < MODES,
        )
    else:
        tl.atomic_add(
            partial_grad_norm_weight + partial_offset,
            partial,
            mask=active_mode < MODES,
        )

    grad_weight_real = tl.dot(tl.trans(active_grad_projected), normalized_real)
    grad_weight_imag = tl.dot(tl.trans(active_grad_projected), normalized_imag)
    tl.atomic_add(
        grad_input_weight + weight_offset,
        grad_weight_real,
        mask=output_mask[:, None] & (mode < MODES),
    )
    tl.atomic_add(
        grad_input_weight + weight_offset + MODES,
        grad_weight_imag,
        mask=output_mask[:, None] & (mode < MODES),
    )


@triton_op("lnet::rmsnorm_input_linear_fused", mutates_args={})
def _forward_op(
    real: Tensor,
    imag: Tensor,
    norm_weight: Tensor,
    input_weight: Tensor,
    epsilon: float,
) -> tuple[Tensor, Tensor]:
    if not supports_fused_rmsnorm_input_linear(
        real,
        imag,
        norm_weight,
        input_weight,
        epsilon=epsilon,
    ):
        raise RuntimeError("unsupported fused RMSNorm input projection contract")
    modes = real.shape[-1]
    rows = real.numel() // modes
    output_width = input_weight.shape[0]
    inner_rows, outer_stride, inner_stride, mode_stride = _row_layout(real)
    projected = torch.empty(
        (*real.shape[:-1], output_width),
        device=real.device,
        dtype=torch.bfloat16,
    )
    row_inverse_rms = torch.empty(real.shape[:-1], device=real.device, dtype=torch.float32)
    kernel = autotuned(
        _fused_forward_kernel,
        FORWARD_LAUNCH_NAME,
        key=(
            "rows",
            "MODES",
            "OUTPUT_WIDTH",
            "inner_rows",
            "outer_row_stride",
            "inner_row_stride",
            "mode_stride",
            "SOURCE_IS_BF16",
        ),
        scope=_scope(_fused_forward_kernel, real, input_weight),
    )

    def grid(metadata: dict[str, int]) -> tuple[int]:
        return (int(triton.cdiv(rows, metadata["BLOCK_ROWS"])),)

    wrap_triton(kernel)[grid](
        real,
        imag,
        norm_weight,
        input_weight,
        projected,
        row_inverse_rms,
        rows,
        inner_rows,
        outer_stride,
        inner_stride,
        mode_stride,
        MODES=modes,
        OUTPUT_WIDTH=output_width,
        EPSILON=epsilon,
        BLOCK_MODES=max(16, triton.next_power_of_2(modes)),
        BLOCK_OUTPUT=max(16, triton.next_power_of_2(output_width)),
        CONTIGUOUS_SOURCE=real.is_contiguous(),
        SOURCE_IS_BF16=real.dtype is torch.bfloat16,
    )
    return projected, row_inverse_rms


@triton_op("lnet::rmsnorm_input_linear_fused_backward", mutates_args={})
def _backward_op(
    real: Tensor,
    imag: Tensor,
    norm_weight: Tensor,
    input_weight: Tensor,
    row_inverse_rms: Tensor,
    grad_projected: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    modes = real.shape[-1]
    rows = real.numel() // modes
    output_width = input_weight.shape[0]
    inner_rows, outer_stride, inner_stride, mode_stride = _row_layout(real)
    grad_real = torch.empty_like(real, memory_format=torch.contiguous_format)
    grad_imag = torch.empty_like(imag, memory_format=torch.contiguous_format)
    partial_rows = device_parameter_reduction_rows(real, rows)
    partial_count = int(triton.cdiv(rows, partial_rows))
    partial_grad_norm_weight = torch.zeros(
        (partial_count, modes),
        device=real.device,
        dtype=torch.float32,
    )
    grad_input_weight_fp32 = torch.zeros_like(input_weight, dtype=torch.float32)
    kernel = autotuned(
        _fused_backward_kernel,
        BACKWARD_LAUNCH_NAME,
        key=(
            "rows",
            "MODES",
            "OUTPUT_WIDTH",
            "inner_rows",
            "outer_row_stride",
            "inner_row_stride",
            "mode_stride",
            "SOURCE_IS_BF16",
        ),
        scope=_scope(_fused_backward_kernel, real, input_weight),
        reset_to_zero=("partial_grad_norm_weight", "grad_input_weight"),
    )

    def grid(metadata: dict[str, int]) -> tuple[int]:
        return (int(triton.cdiv(rows, metadata["BLOCK_ROWS"])),)

    wrap_triton(kernel)[grid](
        real,
        imag,
        norm_weight,
        input_weight,
        row_inverse_rms,
        grad_projected,
        grad_real,
        grad_imag,
        partial_grad_norm_weight,
        grad_input_weight_fp32,
        rows,
        inner_rows,
        outer_stride,
        inner_stride,
        mode_stride,
        MODES=modes,
        OUTPUT_WIDTH=output_width,
        BLOCK_MODES=max(16, triton.next_power_of_2(modes)),
        BLOCK_OUTPUT=max(16, triton.next_power_of_2(output_width)),
        PARTIAL_ROWS=partial_rows,
        CONTIGUOUS_SOURCE=real.is_contiguous(),
        SOURCE_IS_BF16=real.dtype is torch.bfloat16,
    )
    grad_norm_weight = torch.empty_like(norm_weight)
    reduce_kernel = autotuned(
        triton_rmsnorm._packed_complex_rmsnorm_backward_reduce_kernel,  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
        triton_rmsnorm.BACKWARD_REDUCE_LAUNCH_NAME,
        key=("partial_count", "modes"),
        scope=make_launch_scope(
            triton_rmsnorm._packed_complex_rmsnorm_backward_reduce_kernel,  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
            partial_grad_norm_weight,
            shape={"partial_count": partial_count, "modes": modes},
        ),
    )

    def reduce_grid(metadata: dict[str, int]) -> tuple[int]:
        return (int(triton.cdiv(modes, metadata["BLOCK_MODES"])),)

    wrap_triton(reduce_kernel)[reduce_grid](
        partial_grad_norm_weight,
        grad_norm_weight,
        partial_count,
        modes,
    )
    return (
        grad_real,
        grad_imag,
        grad_norm_weight,
        grad_input_weight_fp32.to(input_weight.dtype),
    )


def _setup_context(
    ctx: _AutogradContext,
    inputs: tuple[Tensor, Tensor, Tensor, Tensor, float],
    output: tuple[Tensor, Tensor],
) -> None:
    real, imag, norm_weight, input_weight, _ = inputs
    _, row_inverse_rms = output
    ctx.save_for_backward(real, imag, norm_weight, input_weight, row_inverse_rms)


def _backward(
    ctx: _AutogradContext,
    grad_projected: Tensor | None,
    grad_row_inverse_rms: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, None]:
    del grad_row_inverse_rms
    real, imag, norm_weight, input_weight, row_inverse_rms = ctx.saved_tensors
    if grad_projected is None:
        grad_projected = torch.zeros(
            (*real.shape[:-1], input_weight.shape[0]),
            device=real.device,
            dtype=torch.bfloat16,
        )
    gradients = _backward_op(
        real,
        imag,
        norm_weight,
        input_weight,
        row_inverse_rms,
        grad_projected.contiguous(),
    )
    return *gradients, None


_forward_op.register_autograd(  # pyright: ignore[reportFunctionMemberAccess]
    _backward,
    setup_context=_setup_context,
)


def fused_rmsnorm_input_linear(
    real: Tensor,
    imag: Tensor,
    norm_weight: Tensor,
    input_weight: Tensor,
    *,
    epsilon: float,
) -> Tensor:
    """Normalize and project without materializing packed normalized rows."""
    projected, _ = _forward_op(real, imag, norm_weight, input_weight, epsilon)
    return projected


__all__ = [
    "BACKWARD_LAUNCH_NAME",
    "FORWARD_LAUNCH_NAME",
    "fused_rmsnorm_input_linear",
    "supports_fused_rmsnorm_input_linear",
]
