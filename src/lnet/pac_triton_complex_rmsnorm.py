"""Packed complex RMS normalization for BF16-autocast GEMM inputs."""

from __future__ import annotations

# pyright: reportArgumentType=false, reportCallIssue=false, reportMissingParameterType=false
# ruff: noqa: ANN001, EM101, N803, TRY003
from typing import Protocol

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton

from .pac_kernel_launch_config import (
    LaunchGeometry,
    LaunchScope,
    autotuned,
    make_launch_scope,
    register_default,
)
from .pac_reduction_tiling import device_parameter_reduction_rows

FORWARD_LAUNCH_NAME = "packed_complex_rmsnorm_forward"
BACKWARD_LAUNCH_NAME = "packed_complex_rmsnorm_backward"
BACKWARD_REDUCE_LAUNCH_NAME = "packed_complex_rmsnorm_backward_reduce"
_LAUNCH_CANDIDATES = tuple(
    LaunchGeometry.build(num_warps=warps, blocks={"BLOCK_ROWS": block_rows})
    for block_rows, warps in ((1, 4), (2, 4), (4, 4), (4, 8), (8, 8))
)
_DEFAULT_LAUNCH = LaunchGeometry.build(num_warps=4, blocks={"BLOCK_ROWS": 4})
register_default(FORWARD_LAUNCH_NAME, _DEFAULT_LAUNCH, candidates=_LAUNCH_CANDIDATES)

_BACKWARD_LAUNCH_CANDIDATES = tuple(
    LaunchGeometry.build(num_warps=warps, blocks={"BLOCK_ROWS": block_rows})
    for block_rows in (4, 8, 16, 32, 64)
    for warps in (4, 8)
)
_BACKWARD_DEFAULT_LAUNCH = LaunchGeometry.build(
    num_warps=4,
    blocks={"BLOCK_ROWS": 4},
)
register_default(
    BACKWARD_LAUNCH_NAME,
    _BACKWARD_DEFAULT_LAUNCH,
    candidates=_BACKWARD_LAUNCH_CANDIDATES,
)


_REDUCE_LAUNCH_CANDIDATES = tuple(
    LaunchGeometry.build(
        num_warps=warps,
        blocks={"BLOCK_MODES": block_modes, "BLOCK_PARTIALS": block_partials},
    )
    for block_modes, block_partials, warps in (
        (8, 128, 4),
        (16, 128, 4),
        (16, 256, 8),
        (32, 128, 8),
    )
)
_REDUCE_DEFAULT_LAUNCH = LaunchGeometry.build(
    num_warps=4,
    blocks={"BLOCK_MODES": 16, "BLOCK_PARTIALS": 128},
)
register_default(
    BACKWARD_REDUCE_LAUNCH_NAME,
    _REDUCE_DEFAULT_LAUNCH,
    candidates=_REDUCE_LAUNCH_CANDIDATES,
)


class _AutogradContext(Protocol):
    saved_tensors: tuple[Tensor, ...]

    def save_for_backward(self, *tensors: Tensor) -> None: ...


def packed_complex_rms_norm_reference(
    real: Tensor,
    imag: Tensor,
    norm_weight: Tensor,
    epsilon: float,
) -> Tensor:
    """Return ``[normalized_real | normalized_imag]`` using the model equation."""
    energy = (real.float().square() + imag.float().square()).mean(dim=-1, keepdim=True)
    inverse_rms = torch.rsqrt(energy + epsilon).to(dtype=real.dtype)
    weight = norm_weight.to(dtype=real.dtype)
    return torch.cat((real * inverse_rms * weight, imag * inverse_rms * weight), dim=-1)


def _cuda_bf16_autocast_enabled() -> bool:
    return torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") is torch.bfloat16


def _validate(real: Tensor, imag: Tensor, norm_weight: Tensor, epsilon: float) -> None:
    if real.shape != imag.shape or real.ndim < 1:
        raise ValueError("complex RMSNorm requires matching coordinate tensors")
    if real.shape[-1] <= 0 or norm_weight.shape != (real.shape[-1],):
        raise ValueError("complex RMSNorm scale has incompatible dimensions")
    if epsilon <= 0.0:
        raise ValueError("complex RMSNorm epsilon must be positive")
    if imag.device != real.device or norm_weight.device != real.device:
        raise ValueError("complex RMSNorm tensors must share one device")
    if imag.dtype != real.dtype:
        raise TypeError("complex RMSNorm coordinates must share one dtype")


def supports_packed_complex_rms_norm(real: Tensor, imag: Tensor, norm_weight: Tensor) -> bool:
    try:
        _validate(real, imag, norm_weight, 1.0)
    except (TypeError, ValueError):
        return False
    return (
        real.is_cuda
        and (
            real.dtype is torch.bfloat16
            or (real.dtype is torch.float32 and _cuda_bf16_autocast_enabled())
        )
        and real.numel() > 0
        and real.stride() == imag.stride()
        and _supports_row_layout(real)
        and norm_weight.dtype is torch.float32
        and norm_weight.is_contiguous()
    )


def _supports_forward_op(real: Tensor, imag: Tensor, norm_weight: Tensor) -> bool:
    """Check the static custom-op contract; autocast was checked by the caller."""
    return (
        real.is_cuda
        and real.dtype in (torch.float32, torch.bfloat16)
        and real.numel() > 0
        and real.stride() == imag.stride()
        and _supports_row_layout(real)
        and norm_weight.dtype is torch.float32
        and norm_weight.is_contiguous()
    )


def _supports_row_layout(tensor: Tensor) -> bool:
    return tensor.is_contiguous() or (tensor.ndim >= 2 and tensor.transpose(-2, -1).is_contiguous())


def _row_layout(tensor: Tensor) -> tuple[int, int, int, int]:
    if tensor.ndim == 1:
        return 1, 0, 0, tensor.stride(-1)
    inner_rows = tensor.shape[-2]
    outer_row_stride = tensor.stride(-3) if tensor.ndim >= 3 else 0
    return inner_rows, outer_row_stride, tensor.stride(-2), tensor.stride(-1)


def _scope(kernel: object, real: Tensor) -> LaunchScope:
    modes = real.shape[-1]
    rows = real.numel() // modes
    inner_rows, outer_stride, inner_stride, mode_stride = _row_layout(real)
    return make_launch_scope(
        kernel,
        real,
        shape={
            "rows": rows,
            "modes": modes,
            "inner_rows": inner_rows,
            "outer_stride": outer_stride,
            "inner_stride": inner_stride,
            "mode_stride": mode_stride,
        },
    )


@triton.jit
def _packed_complex_rmsnorm_forward_kernel(
    real,
    imag,
    norm_weight,
    output,
    row_inverse_rms,
    rows: int,
    modes: int,
    epsilon: float,
    inner_rows: int,
    outer_row_stride: int,
    inner_row_stride: int,
    mode_stride: int,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_MODES: tl.constexpr,
    CONTIGUOUS_SOURCE: tl.constexpr,
    SOURCE_IS_BF16: tl.constexpr,
) -> None:
    row_id = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    row = row_id[:, None]
    mode = tl.arange(0, BLOCK_MODES)[None, :]
    mask = (row < rows) & (mode < modes)
    if CONTIGUOUS_SOURCE:
        offset = row * modes + mode
    else:
        outer_row = row // inner_rows
        inner_row = row % inner_rows
        offset = outer_row * outer_row_stride + inner_row * inner_row_stride + mode * mode_stride
    active_real = tl.load(real + offset, mask=mask, other=0.0).to(tl.float32)
    active_imag = tl.load(imag + offset, mask=mask, other=0.0).to(tl.float32)
    energy = tl.sum(active_real * active_real + active_imag * active_imag, axis=1)
    inverse_rms = tl.rsqrt(energy / modes + epsilon)
    active_weight = tl.load(norm_weight + mode, mask=mode < modes, other=0.0)
    if SOURCE_IS_BF16:
        active_inverse_rms = inverse_rms.to(tl.bfloat16)
        scale = active_weight.to(tl.bfloat16)
        normalized_real = (active_real.to(tl.bfloat16) * active_inverse_rms[:, None]).to(
            tl.bfloat16
        ) * scale
        normalized_imag = (active_imag.to(tl.bfloat16) * active_inverse_rms[:, None]).to(
            tl.bfloat16
        ) * scale
    else:
        normalized_real = active_real * inverse_rms[:, None] * active_weight
        normalized_imag = active_imag * inverse_rms[:, None] * active_weight
    output_offset = row * (2 * modes) + mode
    tl.store(output + output_offset, normalized_real, mask=mask)
    tl.store(output + output_offset + modes, normalized_imag, mask=mask)
    tl.store(row_inverse_rms + row_id, inverse_rms, mask=row_id < rows)


@triton.jit
def _packed_complex_rmsnorm_backward_kernel(
    real,
    imag,
    norm_weight,
    row_inverse_rms,
    grad_output,
    grad_real,
    grad_imag,
    partial_grad_norm_weight,
    rows: int,
    modes: int,
    inner_rows: int,
    outer_row_stride: int,
    inner_row_stride: int,
    mode_stride: int,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_MODES: tl.constexpr,
    PARTIAL_ROWS: tl.constexpr,
    CONTIGUOUS_SOURCE: tl.constexpr,
    SOURCE_IS_BF16: tl.constexpr,
) -> None:
    row_block = tl.program_id(0)
    row_id = row_block * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    row = row_id[:, None]
    mode = tl.arange(0, BLOCK_MODES)[None, :]
    mask = (row < rows) & (mode < modes)
    if CONTIGUOUS_SOURCE:
        source_offset = row * modes + mode
    else:
        outer_row = row // inner_rows
        inner_row = row % inner_rows
        source_offset = (
            outer_row * outer_row_stride + inner_row * inner_row_stride + mode * mode_stride
        )
    output_offset = row * modes + mode
    active_real = tl.load(real + source_offset, mask=mask, other=0.0).to(tl.float32)
    active_imag = tl.load(imag + source_offset, mask=mask, other=0.0).to(tl.float32)
    grad_normalized_real = tl.load(grad_output + row * (2 * modes) + mode, mask=mask, other=0.0)
    grad_normalized_imag = tl.load(
        grad_output + row * (2 * modes) + modes + mode,
        mask=mask,
        other=0.0,
    )
    inverse_rms = tl.load(row_inverse_rms + row_id, mask=row_id < rows, other=0.0).to(tl.float32)
    active_weight = tl.load(norm_weight + mode, mask=mode < modes, other=0.0)

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

    scaled_grad_real = grad_normalized_real.to(tl.float32) * active_scale
    scaled_grad_imag = grad_normalized_imag.to(tl.float32) * active_scale
    radial = tl.sum(
        scaled_grad_real * active_real + scaled_grad_imag * active_imag,
        axis=1,
    )
    correction = inverse_rms * inverse_rms * inverse_rms * radial / modes
    active_grad_real = (
        active_inverse_rms[:, None] * scaled_grad_real - active_real * correction[:, None]
    )
    active_grad_imag = (
        active_inverse_rms[:, None] * scaled_grad_imag - active_imag * correction[:, None]
    )
    tl.store(grad_real + output_offset, active_grad_real, mask=mask)
    tl.store(grad_imag + output_offset, active_grad_imag, mask=mask)

    grad_weight_contribution = (
        grad_normalized_real.to(tl.float32) * normalized_base_real
        + grad_normalized_imag.to(tl.float32) * normalized_base_imag
    )
    partial = tl.sum(tl.where(mask, grad_weight_contribution, 0.0), axis=0)
    partial_block = (row_block * BLOCK_ROWS) // PARTIAL_ROWS
    partial_offset = partial_block * modes + tl.arange(0, BLOCK_MODES)
    if BLOCK_ROWS >= PARTIAL_ROWS:
        tl.store(
            partial_grad_norm_weight + partial_offset,
            partial,
            mask=tl.arange(0, BLOCK_MODES) < modes,
        )
    else:
        tl.atomic_add(
            partial_grad_norm_weight + partial_offset,
            partial,
            mask=tl.arange(0, BLOCK_MODES) < modes,
        )


@triton.jit
def _packed_complex_rmsnorm_backward_reduce_kernel(
    partial_grad_norm_weight,
    grad_norm_weight,
    partial_count: int,
    modes: int,
    BLOCK_MODES: tl.constexpr,
    BLOCK_PARTIALS: tl.constexpr,
) -> None:
    mode = tl.program_id(0) * BLOCK_MODES + tl.arange(0, BLOCK_MODES)
    valid_mode = mode < modes
    accumulator = tl.zeros((BLOCK_MODES,), tl.float32)
    for partial_start in tl.range(
        0,
        partial_count,
        BLOCK_PARTIALS,
        loop_unroll_factor=1,
        disable_licm=True,
    ):
        partial = partial_start + tl.arange(0, BLOCK_PARTIALS)
        values = tl.load(
            partial_grad_norm_weight + partial[:, None] * modes + mode[None, :],
            mask=(partial[:, None] < partial_count) & valid_mode[None, :],
            other=0.0,
        )
        accumulator += tl.sum(values, axis=0)
    tl.store(grad_norm_weight + mode, accumulator, mask=valid_mode)


@triton_op("lnet::packed_complex_rmsnorm", mutates_args={})
def _forward_op(
    real: Tensor,
    imag: Tensor,
    norm_weight: Tensor,
    epsilon: float,
) -> tuple[Tensor, Tensor]:
    _validate(real, imag, norm_weight, epsilon)
    if not _supports_forward_op(real, imag, norm_weight):
        message = "CUDA packed RMSNorm requires BF16 or FP32 inside BF16 autocast"
        raise RuntimeError(message)
    modes = real.shape[-1]
    rows = real.numel() // modes
    inner_rows, outer_stride, inner_stride, mode_stride = _row_layout(real)
    output = torch.empty(
        (*real.shape[:-1], 2 * modes),
        device=real.device,
        dtype=torch.bfloat16,
    )
    row_inverse_rms = torch.empty(real.shape[:-1], device=real.device, dtype=torch.float32)
    kernel = autotuned(
        _packed_complex_rmsnorm_forward_kernel,
        FORWARD_LAUNCH_NAME,
        key=(
            "rows",
            "modes",
            "inner_rows",
            "outer_row_stride",
            "inner_row_stride",
            "mode_stride",
        ),
        scope=_scope(_packed_complex_rmsnorm_forward_kernel, real),
    )

    def grid(metadata: dict[str, int]) -> tuple[int]:
        return (int(triton.cdiv(rows, metadata["BLOCK_ROWS"])),)

    wrap_triton(kernel)[grid](
        real,
        imag,
        norm_weight,
        output,
        row_inverse_rms,
        rows,
        modes,
        epsilon,
        inner_rows,
        outer_stride,
        inner_stride,
        mode_stride,
        BLOCK_MODES=max(16, triton.next_power_of_2(modes)),
        CONTIGUOUS_SOURCE=real.is_contiguous(),
        SOURCE_IS_BF16=real.dtype is torch.bfloat16,
    )
    return output, row_inverse_rms


def _setup_context(
    ctx: _AutogradContext,
    inputs: tuple[Tensor, Tensor, Tensor, float],
    output: tuple[Tensor, Tensor],
) -> None:
    real, imag, norm_weight, _ = inputs
    _, row_inverse_rms = output
    ctx.save_for_backward(real, imag, norm_weight, row_inverse_rms)


@triton_op("lnet::packed_complex_rmsnorm_backward", mutates_args={})
def _backward_op(
    real: Tensor,
    imag: Tensor,
    norm_weight: Tensor,
    row_inverse_rms: Tensor,
    grad_output: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    modes = real.shape[-1]
    rows = real.numel() // modes
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
    grad_norm_weight = torch.empty_like(norm_weight, memory_format=torch.contiguous_format)
    backward_kernel = autotuned(
        _packed_complex_rmsnorm_backward_kernel,
        BACKWARD_LAUNCH_NAME,
        key=(
            "rows",
            "modes",
            "inner_rows",
            "outer_row_stride",
            "inner_row_stride",
            "mode_stride",
        ),
        scope=_scope(_packed_complex_rmsnorm_backward_kernel, real),
        reset_to_zero=("partial_grad_norm_weight",),
    )

    def backward_grid(metadata: dict[str, int]) -> tuple[int]:
        return (int(triton.cdiv(rows, metadata["BLOCK_ROWS"])),)

    wrap_triton(backward_kernel)[backward_grid](
        real,
        imag,
        norm_weight,
        row_inverse_rms,
        grad_output,
        grad_real,
        grad_imag,
        partial_grad_norm_weight,
        rows,
        modes,
        inner_rows,
        outer_stride,
        inner_stride,
        mode_stride,
        BLOCK_MODES=max(16, triton.next_power_of_2(modes)),
        PARTIAL_ROWS=partial_rows,
        CONTIGUOUS_SOURCE=real.is_contiguous(),
        SOURCE_IS_BF16=real.dtype is torch.bfloat16,
    )

    reduce_kernel = autotuned(
        _packed_complex_rmsnorm_backward_reduce_kernel,
        BACKWARD_REDUCE_LAUNCH_NAME,
        key=("partial_count", "modes"),
        scope=make_launch_scope(
            _packed_complex_rmsnorm_backward_reduce_kernel,
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
    return grad_real, grad_imag, grad_norm_weight


def _backward(
    ctx: _AutogradContext,
    grad_output: Tensor | None,
    grad_row_inverse_rms: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor, None]:
    del grad_row_inverse_rms
    real, imag, norm_weight, row_inverse_rms = ctx.saved_tensors
    if grad_output is None:
        grad_output = torch.zeros(
            (*real.shape[:-1], 2 * real.shape[-1]),
            device=real.device,
            dtype=torch.bfloat16,
        )
    gradients = _backward_op(
        real,
        imag,
        norm_weight,
        row_inverse_rms,
        grad_output.contiguous(),
    )
    return *gradients, None


torch.library.register_autograd(
    "lnet::packed_complex_rmsnorm",
    _backward,
    setup_context=_setup_context,
)


def packed_complex_rms_norm(
    real: Tensor,
    imag: Tensor,
    norm_weight: Tensor,
    epsilon: float,
) -> Tensor:
    """Normalize and pack coordinates for BF16 GEMMs without changing residual dtype."""
    if not supports_packed_complex_rms_norm(real, imag, norm_weight):
        if real.is_cuda:
            message = "CUDA packed RMSNorm requires BF16 or FP32 inside BF16 autocast"
            raise RuntimeError(message)
        return packed_complex_rms_norm_reference(real, imag, norm_weight, epsilon)
    output, _ = _forward_op(real, imag, norm_weight, epsilon)
    return output


__all__ = [
    "BACKWARD_LAUNCH_NAME",
    "BACKWARD_REDUCE_LAUNCH_NAME",
    "FORWARD_LAUNCH_NAME",
    "packed_complex_rms_norm",
    "packed_complex_rms_norm_reference",
    "supports_packed_complex_rms_norm",
]
