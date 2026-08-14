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
_MOMENT_COMPONENTS: Final[int] = 5
_MOMENT_WIDTH: Final[int] = _MODES * _MOMENT_COMPONENTS
_CLASS_COUNT: Final[int] = 5
_CLASSIFIER_WIDTH: Final[int] = _WIDTH + 2 * _MOMENT_WIDTH
_BLOCK_TIME: Final[int] = 8


@triton.jit
def _efp16_fused_readout_kernel(
    inputs,
    norm_weight,
    forward_moments,
    backward_moments,
    classifier_weight,
    classifier_bias,
    output,
    n_steps: int,
    eps: float,
    BLOCK_TIME: tl.constexpr,
    BLOCK_WIDTH: tl.constexpr,
    BLOCK_MOMENTS: tl.constexpr,
    MOMENT_WIDTH: tl.constexpr,
    CLASSIFIER_WIDTH: tl.constexpr,
    CLASS_COUNT: tl.constexpr,
) -> None:
    """Own one batch row and emit all logits without rereading the sequence."""
    batch = tl.program_id(0)
    time_lane = tl.arange(0, BLOCK_TIME)
    feature = tl.arange(0, BLOCK_WIDTH)
    feature_mask = feature < BLOCK_WIDTH
    gamma = tl.load(norm_weight + feature, mask=feature_mask, other=0.0).to(tl.float32)
    pooled_sum = tl.zeros((BLOCK_WIDTH,), tl.float32)

    time_start = 0
    while time_start < n_steps:
        time = time_start + time_lane
        time_mask = time < n_steps
        offsets = (batch * n_steps + time[:, None]) * BLOCK_WIDTH + feature[None, :]
        mask = time_mask[:, None] & feature_mask[None, :]
        values = tl.load(inputs + offsets, mask=mask, other=0.0).to(tl.float32)
        square_sum = tl.sum(values * values, axis=1)
        inverse_rms = tl.rsqrt(square_sum / BLOCK_WIDTH + eps)
        normalized = values * inverse_rms[:, None] * gamma[None, :]
        pooled_sum += tl.sum(
            tl.where(time_mask[:, None], normalized, 0.0),
            axis=0,
        ).to(tl.float32)
        time_start += BLOCK_TIME
    pooled = pooled_sum / n_steps

    moment = tl.arange(0, BLOCK_MOMENTS)
    moment_mask = moment < MOMENT_WIDTH
    moment_base = batch * MOMENT_WIDTH + moment
    forward = tl.load(
        forward_moments + moment_base,
        mask=moment_mask,
        other=0.0,
    ).to(tl.float32)
    backward = tl.load(
        backward_moments + moment_base,
        mask=moment_mask,
        other=0.0,
    ).to(tl.float32)

    for class_index in tl.static_range(0, CLASS_COUNT):
        weight_base = class_index * CLASSIFIER_WIDTH
        pooled_weight = tl.load(
            classifier_weight + weight_base + feature,
            mask=feature_mask,
            other=0.0,
        ).to(tl.float32)
        forward_weight = tl.load(
            classifier_weight + weight_base + BLOCK_WIDTH + moment,
            mask=moment_mask,
            other=0.0,
        ).to(tl.float32)
        backward_weight = tl.load(
            classifier_weight + weight_base + BLOCK_WIDTH + MOMENT_WIDTH + moment,
            mask=moment_mask,
            other=0.0,
        ).to(tl.float32)
        logit = tl.load(classifier_bias + class_index).to(tl.float32)
        logit += tl.sum(pooled * pooled_weight, axis=0)
        logit += tl.sum(forward * forward_weight, axis=0)
        logit += tl.sum(backward * backward_weight, axis=0)
        tl.store(output + batch * CLASS_COUNT + class_index, logit)


def reference_efp16_fused_readout(
    inputs: Tensor,
    norm_weight: Tensor,
    forward_moments: Tensor,
    backward_moments: Tensor,
    classifier_weight: Tensor,
    classifier_bias: Tensor,
    *,
    eps: float | None = None,
) -> Tensor:
    """Canonical FP32 EFP16 RMSNorm/mean/invariant-head reference."""
    _validate_inputs(
        inputs,
        norm_weight,
        forward_moments,
        backward_moments,
        classifier_weight,
        classifier_bias,
    )
    resolved_eps = _resolve_eps(inputs, eps)
    normalized = functional.rms_norm(
        inputs,
        (_WIDTH,),
        norm_weight,
        resolved_eps,
    )
    features = torch.cat(
        (normalized.mean(dim=1), forward_moments, backward_moments),
        dim=-1,
    )
    return functional.linear(features, classifier_weight, classifier_bias)


@triton_op("lnet::efp16_fused_readout", mutates_args={})
def _efp16_fused_readout_op(
    inputs: Tensor,
    norm_weight: Tensor,
    forward_moments: Tensor,
    backward_moments: Tensor,
    classifier_weight: Tensor,
    classifier_bias: Tensor,
    *,
    eps: float,
) -> Tensor:
    _validate_inputs(
        inputs,
        norm_weight,
        forward_moments,
        backward_moments,
        classifier_weight,
        classifier_bias,
    )
    if eps <= 0.0:
        message = "EFP16 fused readout eps must be positive"
        raise ValueError(message)
    if not inputs.is_cuda:
        return reference_efp16_fused_readout(
            inputs,
            norm_weight,
            forward_moments,
            backward_moments,
            classifier_weight,
            classifier_bias,
            eps=eps,
        )
    batch_size, n_steps, _ = inputs.shape
    output = torch.empty(
        (batch_size, _CLASS_COUNT),
        dtype=torch.float32,
        device=inputs.device,
    )
    wrap_triton(_efp16_fused_readout_kernel)[(batch_size,)](
        inputs,
        norm_weight,
        forward_moments,
        backward_moments,
        classifier_weight,
        classifier_bias,
        output,
        n_steps,
        eps,
        BLOCK_TIME=_BLOCK_TIME,
        BLOCK_WIDTH=_WIDTH,
        BLOCK_MOMENTS=triton.next_power_of_2(_MOMENT_WIDTH),
        MOMENT_WIDTH=_MOMENT_WIDTH,
        CLASSIFIER_WIDTH=_CLASSIFIER_WIDTH,
        CLASS_COUNT=_CLASS_COUNT,
        num_warps=4,
    )
    return output


def fused_efp16_readout_inference(
    inputs: Tensor,
    norm_weight: Tensor,
    forward_moments: Tensor,
    backward_moments: Tensor,
    classifier_weight: Tensor,
    classifier_bias: Tensor,
    *,
    eps: float | None = None,
) -> Tensor:
    """Apply the opt-in canonical EFP16 one-kernel inference readout."""
    resolved_eps = _resolve_eps(inputs, eps)
    return _efp16_fused_readout_op(
        inputs,
        norm_weight,
        forward_moments,
        backward_moments,
        classifier_weight,
        classifier_bias,
        eps=resolved_eps,
    )


def _resolve_eps(inputs: Tensor, eps: float | None) -> float:
    resolved = torch.finfo(inputs.dtype).eps if eps is None else eps
    if resolved <= 0.0:
        message = "EFP16 fused readout eps must be positive"
        raise ValueError(message)
    return float(resolved)


def _validate_inputs(
    inputs: Tensor,
    norm_weight: Tensor,
    forward_moments: Tensor,
    backward_moments: Tensor,
    classifier_weight: Tensor,
    classifier_bias: Tensor,
) -> None:
    batch_size = inputs.shape[0] if inputs.ndim == 3 else -1
    expected_shapes = (
        inputs.ndim == 3 and inputs.shape[1] >= 1 and inputs.shape[2] == _WIDTH,
        norm_weight.shape == (_WIDTH,),
        forward_moments.shape == (batch_size, _MOMENT_WIDTH),
        backward_moments.shape == (batch_size, _MOMENT_WIDTH),
        classifier_weight.shape == (_CLASS_COUNT, _CLASSIFIER_WIDTH),
        classifier_bias.shape == (_CLASS_COUNT,),
    )
    if not all(expected_shapes):
        message = (
            "canonical EFP16 fused readout requires inputs [B,T,32], norm [32], "
            "moments [B,80], classifier [5,192], and bias [5]"
        )
        raise ValueError(message)
    tensors = (
        inputs,
        norm_weight,
        forward_moments,
        backward_moments,
        classifier_weight,
        classifier_bias,
    )
    if any(tensor.dtype != torch.float32 for tensor in tensors):
        message = "canonical EFP16 fused readout requires exact FP32 tensors"
        raise TypeError(message)
    if any(tensor.device != inputs.device for tensor in tensors):
        message = "canonical EFP16 fused readout tensors must share one device"
        raise ValueError(message)
    if any(not tensor.is_contiguous() for tensor in tensors):
        message = "canonical EFP16 fused readout tensors must be contiguous"
        raise ValueError(message)
