"""Shape-generic fused per-mode D4 path collapse with recompute backward."""

from __future__ import annotations

# pyright: reportArgumentType=false, reportCallIssue=false, reportMissingParameterType=false
# ruff: noqa: ANN001, EM101, N803, TRY003
from typing import Protocol, cast

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton
from torch.nn import functional

from .pac_kernel_launch_config import (
    LaunchGeometry,
    LaunchScope,
    autotuned,
    make_launch_scope,
    register_default,
)

ComplexField = tuple[Tensor, Tensor]
_PATH_COUNT = 4
_PACKED_INPUTS = 2 * _PATH_COUNT
_PACKED_OUTPUTS = 2
_PATHS = tl.constexpr(_PATH_COUNT)
_PACKED_INPUTS_TL = tl.constexpr(_PACKED_INPUTS)
_PACKED_OUTPUTS_TL = tl.constexpr(_PACKED_OUTPUTS)
_DOT_INPUTS = tl.constexpr(16)
_DOT_OUTPUTS = tl.constexpr(16)
_FUSED_SOURCE_DTYPE = torch.bfloat16

FORWARD_LAUNCH_NAME = "d4_grouped_path_collapse_forward"
BACKWARD_LAUNCH_NAME = "d4_grouped_path_collapse_backward"
_LAUNCH_CANDIDATES = tuple(
    LaunchGeometry.build(
        num_warps=warps,
        blocks={"BLOCK_TOKENS": block_tokens, "BLOCK_MODES": block_modes},
    )
    for block_tokens, block_modes, warps in (
        (16, 1, 4),
        (16, 2, 4),
        (16, 4, 4),
        (32, 1, 4),
        (32, 2, 4),
        (32, 4, 8),
        (64, 1, 4),
        (64, 2, 8),
    )
)
_DEFAULT_LAUNCH_GEOMETRY = LaunchGeometry.build(
    num_warps=4,
    blocks={"BLOCK_TOKENS": 32, "BLOCK_MODES": 2},
)

register_default(
    FORWARD_LAUNCH_NAME,
    _DEFAULT_LAUNCH_GEOMETRY,
    candidates=_LAUNCH_CANDIDATES,
)
register_default(
    BACKWARD_LAUNCH_NAME,
    _DEFAULT_LAUNCH_GEOMETRY,
    candidates=_LAUNCH_CANDIDATES,
)


class _AutogradContext(Protocol):
    saved_tensors: tuple[Tensor, ...]

    def save_for_backward(self, *tensors: Tensor) -> None: ...


def _execution_dtype(reference: Tensor) -> torch.dtype:
    if (
        reference.is_cuda
        and reference.dtype is torch.float32
        and torch.is_autocast_enabled("cuda")
        and torch.get_autocast_dtype("cuda") is torch.bfloat16
    ):
        return torch.bfloat16
    return reference.dtype


def _typed(value: Tensor, reference: Tensor) -> Tensor:
    return value.to(dtype=_execution_dtype(reference))


def d4_grouped_path_collapse_reference(
    source_real: Tensor,
    source_imag: Tensor,
    input_weight: Tensor,
    input_bias: Tensor,
    output_weight: Tensor,
    output_bias: Tensor,
) -> ComplexField:
    """Evaluate the per-mode packed Cartesian 4-to-hidden-to-1 CFFN."""
    _validate(
        source_real,
        source_imag,
        input_weight,
        input_bias,
        output_weight,
        output_bias,
    )
    source = torch.cat(
        (source_real.transpose(-2, -1), source_imag.transpose(-2, -1)),
        dim=-1,
    ).to(dtype=_execution_dtype(source_real))
    hidden = functional.silu(
        torch.einsum("...mi,mhi->...mh", source, _typed(input_weight, source))
        + _typed(input_bias, source)
    )
    output = torch.einsum("...mh,moh->...mo", hidden, _typed(output_weight, source)) + _typed(
        output_bias, source
    )
    return (
        output[..., 0].unsqueeze(-2).contiguous(),
        output[..., 1].unsqueeze(-2).contiguous(),
    )


def _validate(
    source_real: Tensor,
    source_imag: Tensor,
    input_weight: Tensor,
    input_bias: Tensor,
    output_weight: Tensor,
    output_bias: Tensor,
) -> None:
    if (
        source_real.shape != source_imag.shape
        or source_real.ndim < 2
        or source_real.shape[-2] != _PATH_COUNT
        or source_real.shape[-1] <= 0
    ):
        raise ValueError("grouped D4 path collapse requires matching (..., 4, modes) sources")
    modes = source_real.shape[-1]
    if (
        input_weight.ndim != 3
        or input_weight.shape[0] != modes
        or input_weight.shape[2] != _PACKED_INPUTS
        or input_weight.shape[1] <= 0
        or input_weight.shape[1] % 2 != 0
    ):
        raise ValueError("grouped D4 input projection has incompatible dimensions")
    packed_hidden = input_weight.shape[1]
    if (
        input_bias.shape != (modes, packed_hidden)
        or output_weight.shape != (modes, _PACKED_OUTPUTS, packed_hidden)
        or output_bias.shape != (modes, _PACKED_OUTPUTS)
    ):
        raise ValueError("grouped D4 output projection has incompatible dimensions")
    tensors = (source_imag, input_weight, input_bias, output_weight, output_bias)
    if any(value.device != source_real.device for value in tensors):
        raise ValueError("grouped D4 path-collapse tensors must share one device")
    if source_imag.dtype != source_real.dtype:
        raise TypeError("grouped D4 path-collapse sources must share one dtype")


def supports_d4_grouped_path_collapse(
    source_real: Tensor,
    source_imag: Tensor,
    input_weight: Tensor,
    input_bias: Tensor,
    output_weight: Tensor,
    output_bias: Tensor,
) -> bool:
    """Return whether the fused training kernel preserves the active contract."""
    try:
        _validate(
            source_real,
            source_imag,
            input_weight,
            input_bias,
            output_weight,
            output_bias,
        )
    except (TypeError, ValueError):
        return False
    bf16_source = source_real.dtype is _FUSED_SOURCE_DTYPE
    fp32_autocast_source = (
        source_real.dtype is torch.float32
        and torch.is_autocast_enabled("cuda")
        and torch.get_autocast_dtype("cuda") is torch.bfloat16
    )
    if (
        not source_real.is_cuda
        or not (bf16_source or fp32_autocast_source)
        or source_real.numel() == 0
        or not source_real.is_contiguous()
        or not source_imag.is_contiguous()
        or torch.are_deterministic_algorithms_enabled()
    ):
        return False
    return all(
        value.device == source_real.device
        and value.dtype == torch.float32
        and value.is_contiguous()
        for value in (input_weight, input_bias, output_weight, output_bias)
    )


def _supports_forward_op(
    source_real: Tensor,
    source_imag: Tensor,
    input_weight: Tensor,
    input_bias: Tensor,
    output_weight: Tensor,
    output_bias: Tensor,
) -> bool:
    """Check the static custom-op contract after public autocast validation."""
    return (
        source_real.is_cuda
        and source_real.dtype in (torch.float32, torch.bfloat16)
        and source_real.numel() > 0
        and source_real.is_contiguous()
        and source_imag.is_contiguous()
        and not torch.are_deterministic_algorithms_enabled()
        and all(
            value.device == source_real.device
            and value.dtype == torch.float32
            and value.is_contiguous()
            for value in (input_weight, input_bias, output_weight, output_bias)
        )
    )


def _launch_scope(kernel: object, source: Tensor, path_hidden: int) -> LaunchScope:
    modes = source.shape[-1]
    tokens = source.numel() // (_PATH_COUNT * modes)
    return make_launch_scope(
        kernel,
        source,
        shape={"tokens": tokens, "modes": modes, "path_hidden": path_hidden},
    )


@triton.jit
def _d4_grouped_path_collapse_forward_kernel(
    source_real,
    source_imag,
    input_weight,
    input_bias,
    output_weight,
    output_bias,
    output_real,
    output_imag,
    token_count: int,
    modes: int,
    packed_hidden: int,
    BLOCK_TOKENS: tl.constexpr,
    BLOCK_MODES: tl.constexpr,
    BLOCK_HIDDEN: tl.constexpr,
) -> None:
    mode = tl.program_id(1) * BLOCK_MODES + tl.arange(0, BLOCK_MODES)
    token = tl.program_id(0) * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
    valid_row = (mode[:, None] < modes) & (token[None, :] < token_count)

    coordinate = tl.arange(0, _DOT_INPUTS)
    path = coordinate % _PATHS
    source_offset = (token[None, :, None] * _PATHS + path[None, None, :]) * modes + mode[
        :, None, None
    ]
    source_mask = valid_row[:, :, None] & (coordinate[None, None, :] < _PACKED_INPUTS_TL)
    source = tl.where(
        coordinate[None, None, :] < _PATHS,
        tl.load(
            source_real + source_offset,
            mask=source_mask & (coordinate[None, None, :] < _PATHS),
            other=0.0,
        ),
        tl.where(
            coordinate[None, None, :] < _PACKED_INPUTS_TL,
            tl.load(
                source_imag + source_offset,
                mask=source_mask & (coordinate[None, None, :] >= _PATHS),
                other=0.0,
            ),
            0.0,
        ),
    ).to(tl.bfloat16)

    hidden_coordinate = tl.arange(0, BLOCK_HIDDEN)
    input_weight_offset = (
        mode[:, None, None] * packed_hidden + hidden_coordinate[None, None, :]
    ) * _PACKED_INPUTS_TL + coordinate[None, :, None]
    active_input_weight = tl.load(
        input_weight + input_weight_offset,
        mask=(mode[:, None, None] < modes)
        & (hidden_coordinate[None, None, :] < packed_hidden)
        & (coordinate[None, :, None] < _PACKED_INPUTS_TL),
        other=0.0,
    ).to(tl.bfloat16)
    hidden = tl.dot(source, active_input_weight)
    active_input_bias = tl.load(
        input_bias + mode[:, None] * packed_hidden + hidden_coordinate[None, :],
        mask=(mode[:, None] < modes) & (hidden_coordinate[None, :] < packed_hidden),
        other=0.0,
    ).to(tl.bfloat16)
    hidden = (hidden + active_input_bias[:, None, :]).to(tl.bfloat16).to(tl.float32)
    hidden = (hidden * tl.sigmoid(hidden)).to(tl.bfloat16)

    output_coordinate = tl.arange(0, _DOT_OUTPUTS)
    output_weight_offset = (
        mode[:, None, None] * _PACKED_OUTPUTS_TL + output_coordinate[None, None, :]
    ) * packed_hidden + hidden_coordinate[None, :, None]
    active_output_weight_transposed = tl.load(
        output_weight + output_weight_offset,
        mask=(mode[:, None, None] < modes)
        & (hidden_coordinate[None, :, None] < packed_hidden)
        & (output_coordinate[None, None, :] < _PACKED_OUTPUTS_TL),
        other=0.0,
    ).to(tl.bfloat16)
    active_output = tl.dot(hidden, active_output_weight_transposed)
    active_output_bias = tl.load(
        output_bias + mode[:, None] * _PACKED_OUTPUTS_TL + output_coordinate[None, :],
        mask=(mode[:, None] < modes) & (output_coordinate[None, :] < _PACKED_OUTPUTS_TL),
        other=0.0,
    ).to(tl.bfloat16)
    active_output = (active_output + active_output_bias[:, None, :]).to(tl.bfloat16)
    active_output_real = tl.sum(
        tl.where(output_coordinate[None, None, :] == 0, active_output, 0.0),
        axis=2,
    ).to(tl.bfloat16)
    active_output_imag = tl.sum(
        tl.where(output_coordinate[None, None, :] == 1, active_output, 0.0),
        axis=2,
    ).to(tl.bfloat16)
    output_offset = token[None, :] * modes + mode[:, None]
    tl.store(output_real + output_offset, active_output_real, mask=valid_row)
    tl.store(output_imag + output_offset, active_output_imag, mask=valid_row)


@triton.jit
def _d4_grouped_path_collapse_backward_kernel(
    source_real,
    source_imag,
    input_weight,
    input_bias,
    output_weight,
    grad_output_real,
    grad_output_imag,
    grad_source_real,
    grad_source_imag,
    grad_input_weight,
    grad_input_bias,
    grad_output_weight,
    grad_output_bias,
    token_count: int,
    modes: int,
    packed_hidden: int,
    BLOCK_TOKENS: tl.constexpr,
    BLOCK_MODES: tl.constexpr,
    BLOCK_HIDDEN: tl.constexpr,
) -> None:
    mode = tl.program_id(1) * BLOCK_MODES + tl.arange(0, BLOCK_MODES)
    token = tl.program_id(0) * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
    valid_row = (mode[:, None] < modes) & (token[None, :] < token_count)

    coordinate = tl.arange(0, _DOT_INPUTS)
    path = coordinate % _PATHS
    source_offset = (token[None, :, None] * _PATHS + path[None, None, :]) * modes + mode[
        :, None, None
    ]
    source_mask = valid_row[:, :, None] & (coordinate[None, None, :] < _PACKED_INPUTS_TL)
    source = tl.where(
        coordinate[None, None, :] < _PATHS,
        tl.load(
            source_real + source_offset,
            mask=source_mask & (coordinate[None, None, :] < _PATHS),
            other=0.0,
        ),
        tl.where(
            coordinate[None, None, :] < _PACKED_INPUTS_TL,
            tl.load(
                source_imag + source_offset,
                mask=source_mask & (coordinate[None, None, :] >= _PATHS),
                other=0.0,
            ),
            0.0,
        ),
    ).to(tl.bfloat16)

    hidden_coordinate = tl.arange(0, BLOCK_HIDDEN)
    input_weight_offset = (
        mode[:, None, None] * packed_hidden + hidden_coordinate[None, None, :]
    ) * _PACKED_INPUTS_TL + coordinate[None, :, None]
    active_input_weight = tl.load(
        input_weight + input_weight_offset,
        mask=(mode[:, None, None] < modes)
        & (hidden_coordinate[None, None, :] < packed_hidden)
        & (coordinate[None, :, None] < _PACKED_INPUTS_TL),
        other=0.0,
    ).to(tl.bfloat16)
    preactivation = tl.dot(source, active_input_weight)
    active_input_bias = tl.load(
        input_bias + mode[:, None] * packed_hidden + hidden_coordinate[None, :],
        mask=(mode[:, None] < modes) & (hidden_coordinate[None, :] < packed_hidden),
        other=0.0,
    ).to(tl.bfloat16)
    preactivation = (preactivation + active_input_bias[:, None, :]).to(tl.bfloat16)
    preactivation_float = preactivation.to(tl.float32)
    sigmoid = tl.sigmoid(preactivation_float)
    hidden = (preactivation_float * sigmoid).to(tl.bfloat16)

    output_coordinate = tl.arange(0, _DOT_OUTPUTS)
    output_weight_offset = (
        mode[:, None, None] * _PACKED_OUTPUTS_TL + output_coordinate[None, :, None]
    ) * packed_hidden + hidden_coordinate[None, None, :]
    active_output_weight = tl.load(
        output_weight + output_weight_offset,
        mask=(mode[:, None, None] < modes)
        & (output_coordinate[None, :, None] < _PACKED_OUTPUTS_TL)
        & (hidden_coordinate[None, None, :] < packed_hidden),
        other=0.0,
    ).to(tl.bfloat16)
    output_offset = token[None, :] * modes + mode[:, None]
    active_grad_output_real = tl.load(
        grad_output_real + output_offset,
        mask=valid_row,
        other=0.0,
    ).to(tl.bfloat16)
    active_grad_output_imag = tl.load(
        grad_output_imag + output_offset,
        mask=valid_row,
        other=0.0,
    ).to(tl.bfloat16)

    active_grad_output = tl.where(
        output_coordinate[None, None, :] == 0,
        active_grad_output_real[:, :, None],
        tl.where(
            output_coordinate[None, None, :] == 1,
            active_grad_output_imag[:, :, None],
            0.0,
        ),
    ).to(tl.bfloat16)
    grad_hidden = tl.dot(active_grad_output, active_output_weight).to(tl.float32)
    grad_preactivation = (grad_hidden * sigmoid * (1.0 + preactivation_float * (1.0 - sigmoid))).to(
        tl.bfloat16
    )

    grad_output_weight_tile = tl.dot(
        tl.permute(active_grad_output, (0, 2, 1)),
        hidden,
    )
    output_weight_grad_offset = (
        mode[:, None, None] * _PACKED_OUTPUTS_TL + output_coordinate[None, :, None]
    ) * packed_hidden + hidden_coordinate[None, None, :]
    tl.atomic_add(
        grad_output_weight + output_weight_grad_offset,
        grad_output_weight_tile,
        mask=(mode[:, None, None] < modes)
        & (output_coordinate[None, :, None] < _PACKED_OUTPUTS_TL)
        & (hidden_coordinate[None, None, :] < packed_hidden),
    )
    grad_output_bias_tile = tl.sum(active_grad_output.to(tl.float32), axis=1)
    tl.atomic_add(
        grad_output_bias + mode[:, None] * _PACKED_OUTPUTS_TL + output_coordinate[None, :],
        grad_output_bias_tile,
        mask=(mode[:, None] < modes) & (output_coordinate[None, :] < _PACKED_OUTPUTS_TL),
    )

    grad_input_weight_tile = tl.dot(
        tl.permute(grad_preactivation, (0, 2, 1)),
        source,
    )
    input_weight_grad_offset = (
        mode[:, None, None] * packed_hidden + hidden_coordinate[None, :, None]
    ) * _PACKED_INPUTS_TL + coordinate[None, None, :]
    tl.atomic_add(
        grad_input_weight + input_weight_grad_offset,
        grad_input_weight_tile,
        mask=(mode[:, None, None] < modes)
        & (hidden_coordinate[None, :, None] < packed_hidden)
        & (coordinate[None, None, :] < _PACKED_INPUTS_TL),
    )
    tl.atomic_add(
        grad_input_bias + mode[:, None] * packed_hidden + hidden_coordinate[None, :],
        tl.sum(grad_preactivation.to(tl.float32), axis=1),
        mask=(mode[:, None] < modes) & (hidden_coordinate[None, :] < packed_hidden),
    )

    grad_source = tl.dot(
        grad_preactivation,
        tl.permute(active_input_weight, (0, 2, 1)),
    ).to(tl.bfloat16)
    path_coordinate = tl.arange(0, _PATHS)
    grad_source_real_values = tl.gather(
        grad_source,
        tl.broadcast_to(
            path_coordinate[None, None, :],
            (BLOCK_MODES, BLOCK_TOKENS, _PATHS),
        ),
        axis=2,
    )
    grad_source_imag_values = tl.gather(
        grad_source,
        tl.broadcast_to(
            (path_coordinate + _PATHS)[None, None, :],
            (BLOCK_MODES, BLOCK_TOKENS, _PATHS),
        ),
        axis=2,
    )
    grad_source_offset = (
        token[None, :, None] * _PATHS + path_coordinate[None, None, :]
    ) * modes + mode[:, None, None]
    grad_source_mask = valid_row[:, :, None] & (path_coordinate[None, None, :] < _PATHS)
    tl.store(
        grad_source_real + grad_source_offset,
        grad_source_real_values,
        mask=grad_source_mask,
    )
    tl.store(
        grad_source_imag + grad_source_offset,
        grad_source_imag_values,
        mask=grad_source_mask,
    )


@triton_op("lnet::pac_d4_grouped_path_collapse", mutates_args={})
def _forward_op(
    source_real: Tensor,
    source_imag: Tensor,
    input_weight: Tensor,
    input_bias: Tensor,
    output_weight: Tensor,
    output_bias: Tensor,
) -> ComplexField:
    _validate(
        source_real,
        source_imag,
        input_weight,
        input_bias,
        output_weight,
        output_bias,
    )
    if not source_real.is_cuda:
        return d4_grouped_path_collapse_reference(
            source_real,
            source_imag,
            input_weight,
            input_bias,
            output_weight,
            output_bias,
        )
    if not _supports_forward_op(
        source_real,
        source_imag,
        input_weight,
        input_bias,
        output_weight,
        output_bias,
    ):
        raise RuntimeError("unsupported CUDA contract reached grouped D4 path collapse")
    modes = source_real.shape[-1]
    token_count = source_real.numel() // (_PATH_COUNT * modes)
    packed_hidden = input_weight.shape[1]
    output_shape = (*source_real.shape[:-2], 1, modes)
    outputs = (
        torch.empty(output_shape, device=source_real.device, dtype=torch.bfloat16),
        torch.empty(output_shape, device=source_imag.device, dtype=torch.bfloat16),
    )
    forward_kernel = autotuned(
        _d4_grouped_path_collapse_forward_kernel,
        FORWARD_LAUNCH_NAME,
        key=("token_count", "modes", "packed_hidden"),
        scope=_launch_scope(
            _d4_grouped_path_collapse_forward_kernel,
            source_real,
            packed_hidden // 2,
        ),
    )

    def grid(metadata: dict[str, int]) -> tuple[int, int]:
        return (
            int(triton.cdiv(token_count, metadata["BLOCK_TOKENS"])),
            int(triton.cdiv(modes, metadata["BLOCK_MODES"])),
        )

    wrap_triton(forward_kernel)[grid](
        source_real,
        source_imag,
        input_weight,
        input_bias,
        output_weight,
        output_bias,
        *outputs,
        token_count,
        modes,
        packed_hidden,
        BLOCK_HIDDEN=max(16, triton.next_power_of_2(packed_hidden)),
    )
    return outputs


@triton_op("lnet::pac_d4_grouped_path_collapse_backward", mutates_args={})
def _backward_op(
    source_real: Tensor,
    source_imag: Tensor,
    input_weight: Tensor,
    input_bias: Tensor,
    output_weight: Tensor,
    output_bias: Tensor,
    grad_output_real: Tensor,
    grad_output_imag: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    modes = source_real.shape[-1]
    token_count = source_real.numel() // (_PATH_COUNT * modes)
    packed_hidden = input_weight.shape[1]
    grad_source_real = torch.empty_like(source_real)
    grad_source_imag = torch.empty_like(source_imag)
    grad_input_weight, grad_input_bias, grad_output_weight, grad_output_bias = tuple(
        torch.zeros_like(value, memory_format=torch.contiguous_format)
        for value in (input_weight, input_bias, output_weight, output_bias)
    )
    backward_kernel = autotuned(
        _d4_grouped_path_collapse_backward_kernel,
        BACKWARD_LAUNCH_NAME,
        key=("token_count", "modes", "packed_hidden"),
        restore_value=(
            "grad_input_weight",
            "grad_input_bias",
            "grad_output_weight",
            "grad_output_bias",
        ),
        scope=_launch_scope(
            _d4_grouped_path_collapse_backward_kernel,
            source_real,
            packed_hidden // 2,
        ),
    )

    def grid(metadata: dict[str, int]) -> tuple[int, int]:
        return (
            int(triton.cdiv(token_count, metadata["BLOCK_TOKENS"])),
            int(triton.cdiv(modes, metadata["BLOCK_MODES"])),
        )

    wrap_triton(backward_kernel)[grid](
        source_real,
        source_imag,
        input_weight,
        input_bias,
        output_weight,
        grad_output_real,
        grad_output_imag,
        grad_source_real,
        grad_source_imag,
        grad_input_weight,
        grad_input_bias,
        grad_output_weight,
        grad_output_bias,
        token_count,
        modes,
        packed_hidden,
        BLOCK_HIDDEN=max(16, triton.next_power_of_2(packed_hidden)),
    )
    return (
        grad_source_real,
        grad_source_imag,
        grad_input_weight,
        grad_input_bias,
        grad_output_weight,
        grad_output_bias,
    )


def _setup_context(
    ctx: _AutogradContext,
    inputs: tuple[Tensor, ...],
    output: ComplexField,
) -> None:
    del output
    ctx.save_for_backward(*inputs)


def _backward(
    ctx: _AutogradContext,
    grad_output_real: Tensor | None,
    grad_output_imag: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    inputs = ctx.saved_tensors
    source_real, source_imag = inputs[:2]
    output_shape = (*source_real.shape[:-2], 1, source_real.shape[-1])
    gradients = (
        torch.zeros(output_shape, device=source_real.device, dtype=source_real.dtype)
        if grad_output_real is None
        else grad_output_real.contiguous(),
        torch.zeros(output_shape, device=source_imag.device, dtype=source_imag.dtype)
        if grad_output_imag is None
        else grad_output_imag.contiguous(),
    )
    if not source_real.is_cuda:
        differentiable = tuple(value.detach().requires_grad_() for value in inputs)
        with torch.enable_grad():
            outputs = d4_grouped_path_collapse_reference(*differentiable)
        return cast(
            "tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]",
            torch.autograd.grad(outputs, differentiable, gradients),
        )
    return _backward_op(*inputs, *gradients)


torch.library.register_autograd(
    "lnet::pac_d4_grouped_path_collapse",
    _backward,
    setup_context=_setup_context,
)


def d4_grouped_path_collapse(
    source_real: Tensor,
    source_imag: Tensor,
    input_weight: Tensor,
    input_bias: Tensor,
    output_weight: Tensor,
    output_bias: Tensor,
) -> ComplexField:
    """Collapse four D4 paths independently for every mode."""
    if not supports_d4_grouped_path_collapse(
        source_real,
        source_imag,
        input_weight,
        input_bias,
        output_weight,
        output_bias,
    ):
        if source_real.is_cuda:
            raise RuntimeError("unsupported CUDA contract reached grouped D4 path collapse")
        return d4_grouped_path_collapse_reference(
            source_real,
            source_imag,
            input_weight,
            input_bias,
            output_weight,
            output_bias,
        )
    return _forward_op(
        source_real,
        source_imag,
        input_weight,
        input_bias,
        output_weight,
        output_bias,
    )


__all__ = [
    "BACKWARD_LAUNCH_NAME",
    "FORWARD_LAUNCH_NAME",
    "d4_grouped_path_collapse",
    "d4_grouped_path_collapse_reference",
    "supports_d4_grouped_path_collapse",
]
