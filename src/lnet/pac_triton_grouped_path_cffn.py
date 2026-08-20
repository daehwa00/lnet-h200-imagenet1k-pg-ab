"""Shape-generic fused per-mode D4 path collapse with recompute backward."""

from __future__ import annotations

# pyright: reportArgumentType=false, reportCallIssue=false, reportMissingParameterType=false
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
_PHYSICAL_FROM_RELATIVE = (
    (0, 2, 1, 3),
    (2, 0, 3, 1),
    (1, 3, 0, 2),
    (3, 1, 2, 0),
)

FORWARD_LAUNCH_NAME = "d4_grouped_path_collapse_forward"
BACKWARD_LAUNCH_NAME = "d4_grouped_path_collapse_backward"
CELL_FORWARD_LAUNCH_NAME = "d4_grouped_cell_path_collapse_forward"
CELL_BACKWARD_LAUNCH_NAME = "d4_grouped_cell_path_collapse_backward"
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
register_default(
    CELL_FORWARD_LAUNCH_NAME,
    _DEFAULT_LAUNCH_GEOMETRY,
    candidates=_LAUNCH_CANDIDATES,
)
register_default(
    CELL_BACKWARD_LAUNCH_NAME,
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


def d4_grouped_path_swiglu_reference(
    source_real: Tensor,
    source_imag: Tensor,
    input_weight: Tensor,
    input_bias: Tensor,
    output_weight: Tensor,
    output_bias: Tensor,
) -> ComplexField:
    """Evaluate a pure complex-value, real-gated path projection."""
    hidden_modes = _validate_swiglu(
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
    joint = (
        torch.einsum("...mi,mhi->...mh", source, _typed(input_weight, source))
        + _typed(input_bias, source)
    )
    value = joint[..., : 2 * hidden_modes]
    gate_logits = joint[..., 2 * hidden_modes :]
    gate = functional.silu(gate_logits)
    gated_value = value * torch.cat((gate, gate), dim=-1)
    correction = torch.einsum(
        "...mh,moh->...mo",
        gated_value,
        _typed(output_weight, source),
    ) + _typed(output_bias, source)
    output = correction
    return (
        output[..., 0].unsqueeze(-2).contiguous(),
        output[..., 1].unsqueeze(-2).contiguous(),
    )


def d4_grouped_cell_path_collapse_reference(
    source_real: Tensor,
    source_imag: Tensor,
    input_weight: Tensor,
    input_bias: Tensor,
    output_weight: Tensor,
    output_bias: Tensor,
) -> ComplexField:
    """Collapse direction-relative 2x2 cells without changing their semantics."""
    _validate_cells(
        source_real,
        source_imag,
        input_weight,
        input_bias,
        output_weight,
        output_bias,
    )
    batch, coarse_height, coarse_width, _, _, modes = source_real.shape

    def restore(cells: Tensor) -> Tensor:
        directions = []
        for direction, permutation in enumerate(_PHYSICAL_FROM_RELATIVE):
            relative = cells[..., direction, :, :]
            physical = torch.stack(
                tuple(relative[..., index, :] for index in permutation),
                dim=-2,
            )
            directions.append(
                physical.reshape(batch, coarse_height, coarse_width, 2, 2, modes)
                .permute(0, 1, 3, 2, 4, 5)
                .reshape(batch, 2 * coarse_height, 2 * coarse_width, modes)
            )
        return torch.stack(directions, dim=-2)

    full_real = restore(source_real)
    full_imag = restore(source_imag)
    _, height, width, _, _ = full_real.shape
    source = torch.cat(
        (
            full_real.permute(0, 4, 3, 1, 2),
            full_imag.permute(0, 4, 3, 1, 2),
        ),
        dim=2,
    ).reshape(batch, 2 * modes * _PATH_COUNT, height, width)
    packed_hidden = input_weight.shape[1]
    hidden = functional.silu(
        functional.conv2d(
            source,
            _typed(input_weight, source).reshape(modes * packed_hidden, 8, 1, 1),
            _typed(input_bias, source).reshape(-1),
            groups=modes,
        )
    )
    output = functional.conv2d(
        hidden,
        _typed(output_weight, source).reshape(2 * modes, packed_hidden, 1, 1),
        _typed(output_bias, source).reshape(-1),
        groups=modes,
    ).reshape(batch, modes, 2, height, width)
    output_real, output_imag = output.split(1, dim=2)
    return (
        output_real.permute(0, 3, 4, 2, 1).contiguous(),
        output_imag.permute(0, 3, 4, 2, 1).contiguous(),
    )


def d4_grouped_cell_path_swiglu_reference(
    source_real: Tensor,
    source_imag: Tensor,
    input_weight: Tensor,
    input_bias: Tensor,
    output_weight: Tensor,
    output_bias: Tensor,
) -> ComplexField:
    """Restore direction-relative cells and apply the Path-SwiGLU reference."""
    _validate_swiglu_cells(
        source_real,
        source_imag,
        input_weight,
        input_bias,
        output_weight,
        output_bias,
    )
    batch, coarse_height, coarse_width, _, _, modes = source_real.shape

    def restore(cells: Tensor) -> Tensor:
        directions = []
        for direction, permutation in enumerate(_PHYSICAL_FROM_RELATIVE):
            relative = cells[..., direction, :, :]
            physical = torch.stack(
                tuple(relative[..., index, :] for index in permutation),
                dim=-2,
            )
            directions.append(
                physical.reshape(batch, coarse_height, coarse_width, 2, 2, modes)
                .permute(0, 1, 3, 2, 4, 5)
                .reshape(batch, 2 * coarse_height, 2 * coarse_width, modes)
            )
        return torch.stack(directions, dim=-2)

    return d4_grouped_path_swiglu_reference(
        restore(source_real),
        restore(source_imag),
        input_weight,
        input_bias,
        output_weight,
        output_bias,
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


def _validate_swiglu(
    source_real: Tensor,
    source_imag: Tensor,
    input_weight: Tensor,
    input_bias: Tensor,
    output_weight: Tensor,
    output_bias: Tensor,
) -> int:
    if (
        source_real.shape != source_imag.shape
        or source_real.ndim < 2
        or source_real.shape[-2] != _PATH_COUNT
        or source_real.shape[-1] <= 0
    ):
        raise ValueError("Path-SwiGLU requires matching (..., 4, modes) sources")
    modes = source_real.shape[-1]
    if (
        input_weight.ndim != 3
        or input_weight.shape[0] != modes
        or input_weight.shape[2] != _PACKED_INPUTS
        or input_weight.shape[1] < 3
        or input_weight.shape[1] % 3
    ):
        raise ValueError("Path-SwiGLU input projection has incompatible dimensions")
    hidden_modes = input_weight.shape[1] // 3
    if (
        input_bias.shape != input_weight.shape[:2]
        or output_weight.shape != (modes, _PACKED_OUTPUTS, 2 * hidden_modes)
        or output_bias.shape != (modes, _PACKED_OUTPUTS)
    ):
        raise ValueError("Path-SwiGLU output projection has incompatible dimensions")
    tensors = (source_imag, input_weight, input_bias, output_weight, output_bias)
    if any(value.device != source_real.device for value in tensors):
        raise ValueError("Path-SwiGLU tensors must share one device")
    if source_imag.dtype != source_real.dtype:
        raise TypeError("Path-SwiGLU sources must share one dtype")
    return hidden_modes


def _validate_swiglu_cells(
    source_real: Tensor,
    source_imag: Tensor,
    input_weight: Tensor,
    input_bias: Tensor,
    output_weight: Tensor,
    output_bias: Tensor,
) -> int:
    hidden_modes = _validate_swiglu(
        source_real,
        source_imag,
        input_weight,
        input_bias,
        output_weight,
        output_bias,
    )
    if source_real.ndim != 6 or tuple(source_real.shape[-3:-1]) != (
        _PATH_COUNT,
        _PATH_COUNT,
    ):
        raise ValueError("Path-SwiGLU cells require BHW-direction-local-mode sources")
    return hidden_modes


def _validate_cells(
    source_real: Tensor,
    source_imag: Tensor,
    input_weight: Tensor,
    input_bias: Tensor,
    output_weight: Tensor,
    output_bias: Tensor,
) -> None:
    _validate(
        source_real,
        source_imag,
        input_weight,
        input_bias,
        output_weight,
        output_bias,
    )
    if source_real.ndim != 6 or tuple(source_real.shape[-3:-1]) != (
        _PATH_COUNT,
        _PATH_COUNT,
    ):
        raise ValueError(
            "grouped D4 cell collapse requires BHW-direction-local-mode sources"
        )


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
    return (
        source_real.is_cuda
        and (bf16_source or fp32_autocast_source)
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
def supports_d4_grouped_cell_path_collapse(
    source_real: Tensor,
    source_imag: Tensor,
    input_weight: Tensor,
    input_bias: Tensor,
    output_weight: Tensor,
    output_bias: Tensor,
) -> bool:
    """Return whether direction-relative cells can use the fused collapse."""
    try:
        _validate_cells(
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
    return (
        source_real.is_cuda
        and (bf16_source or fp32_autocast_source)
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


def supports_d4_grouped_path_swiglu(
    source_real: Tensor,
    source_imag: Tensor,
    input_weight: Tensor,
    input_bias: Tensor,
    output_weight: Tensor,
    output_bias: Tensor,
) -> bool:
    """Return whether Path-SwiGLU can use its fused CUDA kernel."""
    try:
        _validate_swiglu(
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
    return (
        source_real.is_cuda
        and (bf16_source or fp32_autocast_source)
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


def _launch_scope(
    kernel: object,
    source: Tensor,
    path_hidden: int,
    *,
    cell_layout: bool = False,
) -> LaunchScope:
    modes = source.shape[-1]
    paths_per_token = _PATH_COUNT * (_PATH_COUNT if cell_layout else 1)
    tokens = source.numel() // (paths_per_token * modes)
    if cell_layout:
        tokens *= _PATH_COUNT
    return make_launch_scope(
        kernel,
        source,
        shape={
            "tokens": tokens,
            "modes": modes,
            "path_hidden": path_hidden,
            "cell_layout": cell_layout,
        },
    )


@triton.jit
def _swap_local_bits(value):
    return (value % 2) * 2 + value // 2


@triton.jit
def _source_offset(
    token,
    path,
    mode,
    modes: int,
    coarse_height: int,
    coarse_width: int,
    CELL_LAYOUT: tl.constexpr,
):
    if CELL_LAYOUT:
        full_height = 2 * coarse_height
        full_width = 2 * coarse_width
        image_tokens = full_height * full_width
        batch = token // image_tokens
        spatial = token % image_tokens
        physical_y = spatial // full_width
        physical_x = spatial % full_width
        cell = (batch * coarse_height + physical_y // 2) * coarse_width + physical_x // 2
        physical_local = (physical_y % 2) * 2 + physical_x % 2
        relative_local = _swap_local_bits(physical_local) ^ _swap_local_bits(path)
        return (((cell * _PATHS + path) * _PATHS + relative_local) * modes) + mode
    return (token * _PATHS + path) * modes + mode


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
    coarse_height: int,
    coarse_width: int,
    CELL_LAYOUT: tl.constexpr,
    PATH_SWIGLU: tl.constexpr,
    SWIGLU_HIDDEN: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    BLOCK_MODES: tl.constexpr,
    BLOCK_HIDDEN: tl.constexpr,
    BLOCK_SWIGLU_VALUE: tl.constexpr,
) -> None:
    mode = tl.program_id(1) * BLOCK_MODES + tl.arange(0, BLOCK_MODES)
    token = tl.program_id(0) * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
    valid_row = (mode[:, None] < modes) & (token[None, :] < token_count)

    coordinate = tl.arange(0, _DOT_INPUTS)
    path = coordinate % _PATHS
    source_offset = _source_offset(
        token[None, :, None],
        path[None, None, :],
        mode[:, None, None],
        modes,
        coarse_height,
        coarse_width,
        CELL_LAYOUT,
    )
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
    output_coordinate = tl.arange(0, _DOT_OUTPUTS)
    active_output_bias = tl.load(
        output_bias + mode[:, None] * _PACKED_OUTPUTS_TL + output_coordinate[None, :],
        mask=(mode[:, None] < modes) & (output_coordinate[None, :] < _PACKED_OUTPUTS_TL),
        other=0.0,
    ).to(tl.bfloat16)
    if PATH_SWIGLU:
        value_coordinate = tl.arange(0, BLOCK_SWIGLU_VALUE)
        value_indices = tl.broadcast_to(
            value_coordinate[None, None, :],
            (BLOCK_MODES, BLOCK_TOKENS, BLOCK_SWIGLU_VALUE),
        )
        gate_indices = tl.broadcast_to(
            (2 * SWIGLU_HIDDEN + value_coordinate % SWIGLU_HIDDEN)[None, None, :],
            (BLOCK_MODES, BLOCK_TOKENS, BLOCK_SWIGLU_VALUE),
        )
        value = tl.gather(preactivation, value_indices, axis=2).to(tl.bfloat16)
        gate_logits = tl.gather(preactivation, gate_indices, axis=2).to(tl.float32)
        gate = (gate_logits * tl.sigmoid(gate_logits)).to(tl.bfloat16)
        gated_value = tl.where(
            value_coordinate[None, None, :] < 2 * SWIGLU_HIDDEN,
            value * gate,
            0.0,
        ).to(tl.bfloat16)
        output_weight_offset = (
            mode[:, None, None] * _PACKED_OUTPUTS_TL + output_coordinate[None, None, :]
        ) * (2 * SWIGLU_HIDDEN) + value_coordinate[None, :, None]
        active_output_weight_transposed = tl.load(
            output_weight + output_weight_offset,
            mask=(mode[:, None, None] < modes)
            & (value_coordinate[None, :, None] < 2 * SWIGLU_HIDDEN)
            & (output_coordinate[None, None, :] < _PACKED_OUTPUTS_TL),
            other=0.0,
        ).to(tl.bfloat16)
        correction = tl.dot(gated_value, active_output_weight_transposed)
        active_output = (correction + active_output_bias[:, None, :]).to(tl.bfloat16)
    else:
        hidden_float = preactivation.to(tl.float32)
        hidden = (hidden_float * tl.sigmoid(hidden_float)).to(tl.bfloat16)
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
    coarse_height: int,
    coarse_width: int,
    CELL_LAYOUT: tl.constexpr,
    PATH_SWIGLU: tl.constexpr,
    SWIGLU_HIDDEN: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    BLOCK_MODES: tl.constexpr,
    BLOCK_HIDDEN: tl.constexpr,
    BLOCK_SWIGLU_VALUE: tl.constexpr,
) -> None:
    mode = tl.program_id(1) * BLOCK_MODES + tl.arange(0, BLOCK_MODES)
    token = tl.program_id(0) * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
    valid_row = (mode[:, None] < modes) & (token[None, :] < token_count)

    coordinate = tl.arange(0, _DOT_INPUTS)
    path = coordinate % _PATHS
    source_offset = _source_offset(
        token[None, :, None],
        path[None, None, :],
        mode[:, None, None],
        modes,
        coarse_height,
        coarse_width,
        CELL_LAYOUT,
    )
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
    output_coordinate = tl.arange(0, _DOT_OUTPUTS)
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
    if PATH_SWIGLU:
        value_coordinate = tl.arange(0, BLOCK_SWIGLU_VALUE)
        value_indices = tl.broadcast_to(
            value_coordinate[None, None, :],
            (BLOCK_MODES, BLOCK_TOKENS, BLOCK_SWIGLU_VALUE),
        )
        gate_indices = tl.broadcast_to(
            (2 * SWIGLU_HIDDEN + value_coordinate % SWIGLU_HIDDEN)[None, None, :],
            (BLOCK_MODES, BLOCK_TOKENS, BLOCK_SWIGLU_VALUE),
        )
        value = tl.gather(preactivation, value_indices, axis=2).to(tl.bfloat16)
        gate_logits = tl.gather(preactivation, gate_indices, axis=2).to(tl.float32)
        gate_sigmoid = tl.sigmoid(gate_logits)
        gate = (gate_logits * gate_sigmoid).to(tl.bfloat16)
        gated_value = tl.where(
            value_coordinate[None, None, :] < 2 * SWIGLU_HIDDEN,
            value * gate,
            0.0,
        ).to(tl.bfloat16)
        output_weight_offset = (
            mode[:, None, None] * _PACKED_OUTPUTS_TL + output_coordinate[None, :, None]
        ) * (2 * SWIGLU_HIDDEN) + value_coordinate[None, None, :]
        active_output_weight = tl.load(
            output_weight + output_weight_offset,
            mask=(mode[:, None, None] < modes)
            & (output_coordinate[None, :, None] < _PACKED_OUTPUTS_TL)
            & (value_coordinate[None, None, :] < 2 * SWIGLU_HIDDEN),
            other=0.0,
        ).to(tl.bfloat16)
        grad_gated_value = tl.dot(active_grad_output, active_output_weight).to(tl.float32)
        active_value = value.to(tl.float32)
        active_gate = gate.to(tl.float32)
        grad_value = grad_gated_value * active_gate
        gate_derivative = gate_sigmoid * (1.0 + gate_logits * (1.0 - gate_sigmoid))
        grad_gate_contribution = grad_gated_value * active_value * gate_derivative
        value_match = (
            hidden_coordinate[None, None, None, :]
            == value_coordinate[None, None, :, None]
        )
        value_joint_gradient = tl.sum(
            tl.where(value_match, grad_value[:, :, :, None], 0.0),
            axis=2,
        )
        gate_match = (
            hidden_coordinate[None, None, None, :]
            == (
                    2 * SWIGLU_HIDDEN + value_coordinate % SWIGLU_HIDDEN
            )[None, None, :, None]
        )
        gate_joint_gradient = tl.sum(
            tl.where(gate_match, grad_gate_contribution[:, :, :, None], 0.0),
            axis=2,
        )
        grad_preactivation = (value_joint_gradient + gate_joint_gradient).to(tl.bfloat16)
        grad_output_weight_tile = tl.dot(
            tl.permute(active_grad_output, (0, 2, 1)),
            gated_value,
        )
        tl.atomic_add(
            grad_output_weight + output_weight_offset,
            grad_output_weight_tile,
            mask=(mode[:, None, None] < modes)
            & (output_coordinate[None, :, None] < _PACKED_OUTPUTS_TL)
            & (value_coordinate[None, None, :] < 2 * SWIGLU_HIDDEN),
        )
    else:
        preactivation_float = preactivation.to(tl.float32)
        sigmoid = tl.sigmoid(preactivation_float)
        hidden = (preactivation_float * sigmoid).to(tl.bfloat16)
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
        grad_hidden = tl.dot(active_grad_output, active_output_weight).to(tl.float32)
        grad_preactivation = (
            grad_hidden * sigmoid * (1.0 + preactivation_float * (1.0 - sigmoid))
        ).to(tl.bfloat16)
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
    grad_source_offset = _source_offset(
        token[None, :, None],
        path_coordinate[None, None, :],
        mode[:, None, None],
        modes,
        coarse_height,
        coarse_width,
        CELL_LAYOUT,
    )
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
        0,
        0,
        CELL_LAYOUT=False,
        PATH_SWIGLU=False,
        SWIGLU_HIDDEN=0,
        BLOCK_HIDDEN=max(16, triton.next_power_of_2(packed_hidden)),
        BLOCK_SWIGLU_VALUE=16,
    )
    return outputs


@triton_op("lnet::pac_d4_grouped_path_swiglu", mutates_args={})
def _swiglu_forward_op(
    source_real: Tensor,
    source_imag: Tensor,
    input_weight: Tensor,
    input_bias: Tensor,
    output_weight: Tensor,
    output_bias: Tensor,
) -> ComplexField:
    hidden_modes = _validate_swiglu(
        source_real,
        source_imag,
        input_weight,
        input_bias,
        output_weight,
        output_bias,
    )
    if not source_real.is_cuda:
        return d4_grouped_path_swiglu_reference(
            source_real,
            source_imag,
            input_weight,
            input_bias,
            output_weight,
            output_bias,
        )
    if not supports_d4_grouped_path_swiglu(
        source_real,
        source_imag,
        input_weight,
        input_bias,
        output_weight,
        output_bias,
    ):
        raise RuntimeError("unsupported CUDA contract reached Path-SwiGLU")
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
            hidden_modes,
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
        0,
        0,
        CELL_LAYOUT=False,
        PATH_SWIGLU=True,
        SWIGLU_HIDDEN=hidden_modes,
        BLOCK_HIDDEN=max(16, triton.next_power_of_2(packed_hidden)),
        BLOCK_SWIGLU_VALUE=max(16, triton.next_power_of_2(2 * hidden_modes)),
    )
    return outputs


@triton_op("lnet::pac_d4_grouped_cell_path_collapse", mutates_args={})
def _cell_forward_op(
    source_real: Tensor,
    source_imag: Tensor,
    input_weight: Tensor,
    input_bias: Tensor,
    output_weight: Tensor,
    output_bias: Tensor,
) -> ComplexField:
    _validate_cells(
        source_real,
        source_imag,
        input_weight,
        input_bias,
        output_weight,
        output_bias,
    )
    if not source_real.is_cuda:
        return d4_grouped_cell_path_collapse_reference(
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
        raise RuntimeError("unsupported CUDA contract reached grouped D4 cell collapse")
    batch, coarse_height, coarse_width, _, _, modes = source_real.shape
    token_count = batch * 2 * coarse_height * 2 * coarse_width
    packed_hidden = input_weight.shape[1]
    output_shape = (batch, 2 * coarse_height, 2 * coarse_width, 1, modes)
    outputs = (
        torch.empty(output_shape, device=source_real.device, dtype=torch.bfloat16),
        torch.empty(output_shape, device=source_imag.device, dtype=torch.bfloat16),
    )
    forward_kernel = autotuned(
        _d4_grouped_path_collapse_forward_kernel,
        CELL_FORWARD_LAUNCH_NAME,
        key=("token_count", "modes", "packed_hidden"),
        scope=_launch_scope(
            _d4_grouped_path_collapse_forward_kernel,
            source_real,
            packed_hidden // 2,
            cell_layout=True,
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
        coarse_height,
        coarse_width,
        CELL_LAYOUT=True,
        PATH_SWIGLU=False,
        SWIGLU_HIDDEN=0,
        BLOCK_HIDDEN=max(16, triton.next_power_of_2(packed_hidden)),
        BLOCK_SWIGLU_VALUE=16,
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
    return _atomic_backward(
        source_real,
        source_imag,
        input_weight,
        input_bias,
        output_weight,
        output_bias,
        grad_output_real,
        grad_output_imag,
        cell_layout=False,
    )


@triton_op("lnet::pac_d4_grouped_path_swiglu_backward", mutates_args={})
def _swiglu_backward_op(
    source_real: Tensor,
    source_imag: Tensor,
    input_weight: Tensor,
    input_bias: Tensor,
    output_weight: Tensor,
    output_bias: Tensor,
    grad_output_real: Tensor,
    grad_output_imag: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    return _atomic_backward(
        source_real,
        source_imag,
        input_weight,
        input_bias,
        output_weight,
        output_bias,
        grad_output_real,
        grad_output_imag,
        cell_layout=False,
        path_swiglu=True,
    )


def _atomic_backward(
    source_real: Tensor,
    source_imag: Tensor,
    input_weight: Tensor,
    input_bias: Tensor,
    output_weight: Tensor,
    output_bias: Tensor,
    grad_output_real: Tensor,
    grad_output_imag: Tensor,
    *,
    cell_layout: bool,
    path_swiglu: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    modes = source_real.shape[-1]
    if cell_layout:
        _, coarse_height, coarse_width, _, _, _ = source_real.shape
        token_count = source_real.shape[0] * 4 * coarse_height * coarse_width
        launch_name = CELL_BACKWARD_LAUNCH_NAME
    else:
        coarse_height = coarse_width = 0
        token_count = source_real.numel() // (_PATH_COUNT * modes)
        launch_name = BACKWARD_LAUNCH_NAME
    packed_hidden = input_weight.shape[1]
    swiglu_hidden = packed_hidden // 3 if path_swiglu else 0
    grad_source_real = torch.empty_like(source_real)
    grad_source_imag = torch.empty_like(source_imag)
    grad_input_weight, grad_input_bias, grad_output_weight, grad_output_bias = tuple(
        torch.zeros_like(value, memory_format=torch.contiguous_format)
        for value in (input_weight, input_bias, output_weight, output_bias)
    )
    gradients = (
        grad_input_weight,
        grad_input_bias,
        grad_output_weight,
        grad_output_bias,
    )
    backward_kernel = autotuned(
        _d4_grouped_path_collapse_backward_kernel,
        launch_name,
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
            swiglu_hidden if path_swiglu else packed_hidden // 2,
            cell_layout=cell_layout,
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
        *gradients,
        token_count,
        modes,
        packed_hidden,
        coarse_height,
        coarse_width,
        CELL_LAYOUT=cell_layout,
        PATH_SWIGLU=path_swiglu,
        SWIGLU_HIDDEN=swiglu_hidden,
        BLOCK_HIDDEN=max(16, triton.next_power_of_2(packed_hidden)),
        BLOCK_SWIGLU_VALUE=(
            max(16, triton.next_power_of_2(2 * swiglu_hidden))
            if path_swiglu
            else 16
        ),
    )
    return (
        grad_source_real,
        grad_source_imag,
        grad_input_weight,
        grad_input_bias,
        grad_output_weight,
        grad_output_bias,
    )


@triton_op("lnet::pac_d4_grouped_cell_path_collapse_backward", mutates_args={})
def _cell_backward_op(
    source_real: Tensor,
    source_imag: Tensor,
    input_weight: Tensor,
    input_bias: Tensor,
    output_weight: Tensor,
    output_bias: Tensor,
    grad_output_real: Tensor,
    grad_output_imag: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    return _atomic_backward(
        source_real,
        source_imag,
        input_weight,
        input_bias,
        output_weight,
        output_bias,
        grad_output_real,
        grad_output_imag,
        cell_layout=True,
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


def _swiglu_backward(
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
            outputs = d4_grouped_path_swiglu_reference(*differentiable)
        return cast(
            "tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]",
            torch.autograd.grad(outputs, differentiable, gradients),
        )
    return _swiglu_backward_op(*inputs, *gradients)


torch.library.register_autograd(
    "lnet::pac_d4_grouped_path_swiglu",
    _swiglu_backward,
    setup_context=_setup_context,
)


def _cell_backward(
    ctx: _AutogradContext,
    grad_output_real: Tensor | None,
    grad_output_imag: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    inputs = ctx.saved_tensors
    source_real, source_imag = inputs[:2]
    batch, coarse_height, coarse_width, _, _, modes = source_real.shape
    output_shape = (batch, 2 * coarse_height, 2 * coarse_width, 1, modes)
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
            outputs = d4_grouped_cell_path_collapse_reference(*differentiable)
        return cast(
            "tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]",
            torch.autograd.grad(outputs, differentiable, gradients),
        )
    return _cell_backward_op(*inputs, *gradients)


torch.library.register_autograd(
    "lnet::pac_d4_grouped_cell_path_collapse",
    _cell_backward,
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


def d4_grouped_path_swiglu(
    source_real: Tensor,
    source_imag: Tensor,
    input_weight: Tensor,
    input_bias: Tensor,
    output_weight: Tensor,
    output_bias: Tensor,
) -> ComplexField:
    """Apply residual Path-SwiGLU independently for every mode."""
    if not supports_d4_grouped_path_swiglu(
        source_real,
        source_imag,
        input_weight,
        input_bias,
        output_weight,
        output_bias,
    ):
        if source_real.is_cuda:
            raise RuntimeError("unsupported CUDA contract reached Path-SwiGLU")
        return d4_grouped_path_swiglu_reference(
            source_real,
            source_imag,
            input_weight,
            input_bias,
            output_weight,
            output_bias,
        )
    return _swiglu_forward_op(
        source_real,
        source_imag,
        input_weight,
        input_bias,
        output_weight,
        output_bias,
    )


def d4_grouped_cell_path_collapse(
    source_real: Tensor,
    source_imag: Tensor,
    input_weight: Tensor,
    input_bias: Tensor,
    output_weight: Tensor,
    output_bias: Tensor,
) -> ComplexField:
    """Collapse direction-relative 2x2 cells directly to full-resolution output."""
    if not supports_d4_grouped_cell_path_collapse(
        source_real,
        source_imag,
        input_weight,
        input_bias,
        output_weight,
        output_bias,
    ):
        if source_real.is_cuda:
            raise RuntimeError("unsupported CUDA contract reached grouped D4 cell collapse")
        return d4_grouped_cell_path_collapse_reference(
            source_real,
            source_imag,
            input_weight,
            input_bias,
            output_weight,
            output_bias,
        )
    return _cell_forward_op(
        source_real,
        source_imag,
        input_weight,
        input_bias,
        output_weight,
        output_bias,
    )


__all__ = [
    "BACKWARD_LAUNCH_NAME",
    "CELL_BACKWARD_LAUNCH_NAME",
    "CELL_FORWARD_LAUNCH_NAME",
    "FORWARD_LAUNCH_NAME",
    "d4_grouped_cell_path_collapse",
    "d4_grouped_cell_path_collapse_reference",
    "d4_grouped_cell_path_swiglu_reference",
    "d4_grouped_path_collapse",
    "d4_grouped_path_collapse_reference",
    "d4_grouped_path_swiglu",
    "d4_grouped_path_swiglu_reference",
    "supports_d4_grouped_cell_path_collapse",
    "supports_d4_grouped_path_collapse",
    "supports_d4_grouped_path_swiglu",
]
