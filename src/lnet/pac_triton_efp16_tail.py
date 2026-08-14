from __future__ import annotations

# pyright: reportMissingParameterType=false
# ruff: noqa: ANN001, N803
from typing import Final

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton
from torch.nn import functional

_WIDTH: Final[int] = 32
_MODES: Final[int] = 16
_MOMENT_WIDTH: Final[int] = 5 * _MODES
_CLASS_COUNT: Final[int] = 5
_CLASSIFIER_WIDTH: Final[int] = _WIDTH + 2 * _MOMENT_WIDTH
_SUPPORTED_BLOCK_TIMES: Final = frozenset((16, 32, 64, 128, 256))
_SUPPORTED_NUM_WARPS: Final = frozenset((4, 8))


@triton.jit
def _efp16_tail_partials_kernel(
    block_inputs,
    local,
    packed_modal_coordinates,
    synthesis_frame,
    direct_scale,
    layer_scale,
    final_norm_weight,
    partials,
    n_steps: int,
    group_count: int,
    eps: float,
    BLOCK_TIME: tl.constexpr,
    WIDTH: tl.constexpr,
) -> None:
    """Fuse final synthesis/residual/RMSNorm and emit one sum per time tile."""
    program = tl.program_id(0)
    batch = program // group_count
    group = program - batch * group_count
    time = group * BLOCK_TIME + tl.arange(0, BLOCK_TIME)
    feature = tl.arange(0, WIDTH)
    time_mask = time < n_steps
    row_offsets = (batch * n_steps + time[:, None]) * WIDTH + feature[None, :]
    row_mask = time_mask[:, None]

    coordinates = tl.load(
        packed_modal_coordinates + row_offsets,
        mask=row_mask,
        other=0.0,
    ).to(tl.float32)
    synthesis = tl.load(
        synthesis_frame + feature[:, None] + feature[None, :] * WIDTH,
    ).to(tl.float32)
    modal = tl.dot(coordinates, synthesis, input_precision="ieee")

    active_local = tl.load(local + row_offsets, mask=row_mask, other=0.0).to(tl.float32)
    residual = tl.load(block_inputs + row_offsets, mask=row_mask, other=0.0).to(tl.float32)
    direct = tl.load(direct_scale + feature).to(tl.float32)
    layer = tl.load(layer_scale + feature).to(tl.float32)
    output = residual + layer[None, :] * (modal + direct[None, :] * active_local)

    square_sum = tl.sum(output * output, axis=1)
    inverse_rms = tl.rsqrt(square_sum / WIDTH + eps)
    gamma = tl.load(final_norm_weight + feature).to(tl.float32)
    normalized = output * inverse_rms[:, None] * gamma[None, :]
    partial = tl.sum(tl.where(row_mask, normalized, 0.0), axis=0)
    tl.store(partials + program * WIDTH + feature, partial)


@triton.jit
def _efp16_tail_reduce_classify_kernel(
    partials,
    forward_moments,
    backward_moments,
    classifier_weight,
    classifier_bias,
    output,
    n_steps: int,
    group_count: int,
    BLOCK_GROUPS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_MOMENTS: tl.constexpr,
    MOMENT_WIDTH: tl.constexpr,
    CLASSIFIER_WIDTH: tl.constexpr,
    CLASS_COUNT: tl.constexpr,
) -> None:
    """Reduce time-tile sums and fuse both moment branches with the classifier."""
    batch = tl.program_id(0)
    group = tl.arange(0, BLOCK_GROUPS)
    feature = tl.arange(0, WIDTH)
    group_mask = group < group_count
    partial_offsets = (
        (batch * group_count + group[:, None]) * WIDTH + feature[None, :]
    )
    pooled_sum = tl.sum(
        tl.load(
            partials + partial_offsets,
            mask=group_mask[:, None],
            other=0.0,
        ).to(tl.float32),
        axis=0,
    )
    pooled = pooled_sum / n_steps

    moment = tl.arange(0, BLOCK_MOMENTS)
    moment_mask = moment < MOMENT_WIDTH
    moment_offsets = batch * MOMENT_WIDTH + moment
    forward = tl.load(
        forward_moments + moment_offsets,
        mask=moment_mask,
        other=0.0,
    ).to(tl.float32)
    backward = tl.load(
        backward_moments + moment_offsets,
        mask=moment_mask,
        other=0.0,
    ).to(tl.float32)

    for class_index in tl.static_range(0, CLASS_COUNT):
        weight_base = class_index * CLASSIFIER_WIDTH
        pooled_weight = tl.load(classifier_weight + weight_base + feature).to(tl.float32)
        forward_weight = tl.load(
            classifier_weight + weight_base + WIDTH + moment,
            mask=moment_mask,
            other=0.0,
        ).to(tl.float32)
        backward_weight = tl.load(
            classifier_weight + weight_base + WIDTH + MOMENT_WIDTH + moment,
            mask=moment_mask,
            other=0.0,
        ).to(tl.float32)
        logit = tl.load(classifier_bias + class_index).to(tl.float32)
        logit += tl.sum(pooled * pooled_weight, axis=0)
        logit += tl.sum(forward * forward_weight, axis=0)
        logit += tl.sum(backward * backward_weight, axis=0)
        tl.store(output + batch * CLASS_COUNT + class_index, logit)


def reference_efp16_fused_tail(
    block_inputs: Tensor,
    local: Tensor,
    packed_modal_coordinates: Tensor,
    synthesis_frame: Tensor,
    direct_scale: Tensor,
    layer_scale: Tensor,
    final_norm_weight: Tensor,
    forward_moments: Tensor,
    backward_moments: Tensor,
    classifier_weight: Tensor,
    classifier_bias: Tensor,
    *,
    eps: float | None = None,
) -> Tensor:
    """Exact native expression replaced by the inference-only fused tail."""
    _validate_inputs(
        block_inputs,
        local,
        packed_modal_coordinates,
        synthesis_frame,
        direct_scale,
        layer_scale,
        final_norm_weight,
        forward_moments,
        backward_moments,
        classifier_weight,
        classifier_bias,
    )
    resolved_eps = _resolve_eps(block_inputs, eps)
    modal = torch.matmul(packed_modal_coordinates, synthesis_frame.transpose(0, 1))
    output = block_inputs + layer_scale.view(1, 1, -1) * (
        modal + direct_scale.view(1, 1, -1) * local
    )
    normalized = functional.rms_norm(
        output,
        (_WIDTH,),
        final_norm_weight,
        resolved_eps,
    )
    features = torch.cat(
        (normalized.mean(dim=1), forward_moments, backward_moments),
        dim=-1,
    )
    return functional.linear(features, classifier_weight, classifier_bias)


@triton_op("lnet::efp16_fused_tail", mutates_args={})
def _efp16_fused_tail_op(
    block_inputs: Tensor,
    local: Tensor,
    packed_modal_coordinates: Tensor,
    synthesis_frame: Tensor,
    direct_scale: Tensor,
    layer_scale: Tensor,
    final_norm_weight: Tensor,
    forward_moments: Tensor,
    backward_moments: Tensor,
    classifier_weight: Tensor,
    classifier_bias: Tensor,
    *,
    eps: float,
    block_time: int,
    num_warps: int,
) -> Tensor:
    _validate_inputs(
        block_inputs,
        local,
        packed_modal_coordinates,
        synthesis_frame,
        direct_scale,
        layer_scale,
        final_norm_weight,
        forward_moments,
        backward_moments,
        classifier_weight,
        classifier_bias,
    )
    _validate_options(eps, block_time, num_warps)
    if not block_inputs.is_cuda:
        return reference_efp16_fused_tail(
            block_inputs,
            local,
            packed_modal_coordinates,
            synthesis_frame,
            direct_scale,
            layer_scale,
            final_norm_weight,
            forward_moments,
            backward_moments,
            classifier_weight,
            classifier_bias,
            eps=eps,
        )
    batch_size, n_steps, _ = block_inputs.shape
    group_count = (n_steps + block_time - 1) // block_time
    partials = torch.empty(
        (batch_size, group_count, _WIDTH),
        dtype=torch.float32,
        device=block_inputs.device,
    )
    output = torch.empty(
        (batch_size, _CLASS_COUNT),
        dtype=torch.float32,
        device=block_inputs.device,
    )
    wrap_triton(_efp16_tail_partials_kernel)[(batch_size * group_count,)](
        block_inputs,
        local,
        packed_modal_coordinates,
        synthesis_frame,
        direct_scale,
        layer_scale,
        final_norm_weight,
        partials,
        n_steps,
        group_count,
        eps,
        BLOCK_TIME=block_time,
        WIDTH=_WIDTH,
        num_warps=num_warps,
    )
    wrap_triton(_efp16_tail_reduce_classify_kernel)[(batch_size,)](
        partials,
        forward_moments,
        backward_moments,
        classifier_weight,
        classifier_bias,
        output,
        n_steps,
        group_count,
        BLOCK_GROUPS=triton.next_power_of_2(group_count),
        WIDTH=_WIDTH,
        BLOCK_MOMENTS=triton.next_power_of_2(_MOMENT_WIDTH),
        MOMENT_WIDTH=_MOMENT_WIDTH,
        CLASSIFIER_WIDTH=_CLASSIFIER_WIDTH,
        CLASS_COUNT=_CLASS_COUNT,
        num_warps=4,
    )
    return output


def fused_efp16_tail_inference(
    block_inputs: Tensor,
    local: Tensor,
    packed_modal_coordinates: Tensor,
    synthesis_frame: Tensor,
    direct_scale: Tensor,
    layer_scale: Tensor,
    final_norm_weight: Tensor,
    forward_moments: Tensor,
    backward_moments: Tensor,
    classifier_weight: Tensor,
    classifier_bias: Tensor,
    *,
    eps: float | None = None,
    block_time: int = 32,
    num_warps: int = 4,
) -> Tensor:
    """Apply the opt-in canonical EFP16 final-block/readout fusion."""
    return _efp16_fused_tail_op(
        block_inputs,
        local,
        packed_modal_coordinates,
        synthesis_frame,
        direct_scale,
        layer_scale,
        final_norm_weight,
        forward_moments,
        backward_moments,
        classifier_weight,
        classifier_bias,
        eps=_resolve_eps(block_inputs, eps),
        block_time=block_time,
        num_warps=num_warps,
    )


def _resolve_eps(inputs: Tensor, eps: float | None) -> float:
    resolved = torch.finfo(inputs.dtype).eps if eps is None else eps
    if resolved <= 0.0:
        message = "EFP16 fused tail eps must be positive"
        raise ValueError(message)
    return float(resolved)


def _validate_options(eps: float, block_time: int, num_warps: int) -> None:
    if eps <= 0.0:
        message = "EFP16 fused tail eps must be positive"
        raise ValueError(message)
    if block_time not in _SUPPORTED_BLOCK_TIMES:
        message = "EFP16 fused tail block_time must be one of {16, 32, 64, 128, 256}"
        raise ValueError(message)
    if num_warps not in _SUPPORTED_NUM_WARPS:
        message = "EFP16 fused tail num_warps must be one of {4, 8}"
        raise ValueError(message)


def _validate_inputs(
    block_inputs: Tensor,
    local: Tensor,
    packed_modal_coordinates: Tensor,
    synthesis_frame: Tensor,
    direct_scale: Tensor,
    layer_scale: Tensor,
    final_norm_weight: Tensor,
    forward_moments: Tensor,
    backward_moments: Tensor,
    classifier_weight: Tensor,
    classifier_bias: Tensor,
) -> None:
    batch_size = block_inputs.shape[0] if block_inputs.ndim == 3 else -1
    n_steps = block_inputs.shape[1] if block_inputs.ndim == 3 else -1
    expected_shapes = (
        block_inputs.ndim == 3
        and n_steps >= 1
        and block_inputs.shape[2] == _WIDTH,
        local.shape == (batch_size, n_steps, _WIDTH),
        packed_modal_coordinates.shape == (batch_size, n_steps, _WIDTH),
        synthesis_frame.shape == (_WIDTH, _WIDTH),
        direct_scale.shape == (_WIDTH,),
        layer_scale.shape == (_WIDTH,),
        final_norm_weight.shape == (_WIDTH,),
        forward_moments.shape == (batch_size, _MOMENT_WIDTH),
        backward_moments.shape == (batch_size, _MOMENT_WIDTH),
        classifier_weight.shape == (_CLASS_COUNT, _CLASSIFIER_WIDTH),
        classifier_bias.shape == (_CLASS_COUNT,),
    )
    if not all(expected_shapes):
        message = (
            "canonical EFP16 fused tail requires sequence tensors [B,T,32], "
            "scales/norm [32], moments [B,80], synthesis [32,32], "
            "classifier [5,192], and bias [5]"
        )
        raise ValueError(message)
    tensors = (
        block_inputs,
        local,
        packed_modal_coordinates,
        synthesis_frame,
        direct_scale,
        layer_scale,
        final_norm_weight,
        forward_moments,
        backward_moments,
        classifier_weight,
        classifier_bias,
    )
    if any(tensor.dtype != torch.float32 for tensor in tensors):
        message = "canonical EFP16 fused tail requires exact FP32 tensors"
        raise TypeError(message)
    if any(tensor.device != block_inputs.device for tensor in tensors):
        message = "canonical EFP16 fused tail tensors must share one device"
        raise ValueError(message)
    if any(not tensor.is_contiguous() for tensor in tensors):
        message = "canonical EFP16 fused tail tensors must be contiguous"
        raise ValueError(message)
