"""Packed-layout runtime for the A2D post-CFFN carry-main transition.

The eager transition alternates between split ``(real, imag)`` coordinates and
the packed ``[..., 2M]`` layout that every one of its GEMMs actually wants.  On
the production 8x8 and 4x4 stages that round trip costs far more than the
arithmetic: the eager forward dispatches 74 CUDA kernels, 21 of them
concatenations, and 33 of them recompute input-independent packed weights that
the previous call already built.

This module keeps one packed activation from the direction mixer through the
output projection. Eager inference caches input-independent parameter blocks;
compiled training rebuilds those differentiable blocks inside the captured
graph so gradients still reach every source parameter.
"""

from __future__ import annotations

# ``WidelyLinear`` satisfies ``ComplexAffine`` structurally, but its blocks are
# ``Parameter``, which the invariant protocol attributes reject; the calling
# modules already carry the same suppression.  The transition's fused-weight
# cache is private to the pair of modules that own it.
# The transition is consumed through a structural protocol below, so this
# low-level runtime never imports the high-level multiscale backbone module.
# pyright: reportArgumentType=false, reportPrivateUsage=false
# The cache key reads ``Tensor._version``, the documented mutation counter.
# ruff: noqa: SLF001
import os
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Protocol

import torch

from .pac_complex_ffn import module_calls_are_transparent
from .pac_complex_layers import (
    WidelyLinear,
    packed_widely_linear_bias,
    packed_widely_linear_weight,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from torch import Tensor

    from .pac_complex_ffn import ComplexFFNActivation
    from .pac_complex_layers import ComplexLinear

ComplexField = tuple["Tensor", "Tensor"]

_DISABLE_VARIABLE = "LNET_DISABLE_PACKED_POSTCARRY_INFERENCE"
_DISABLE_TRAINING_VARIABLE = "LNET_DISABLE_PACKED_POSTFUSION_TRAINING"
_CARRY_POSITIONS = 4


def _same_tensor_contract(reference: Tensor, candidate: Tensor) -> bool:
    return (
        reference.shape == candidate.shape
        and reference.stride() == candidate.stride()
        and reference.dtype == candidate.dtype
        and reference.device == candidate.device
    )


def _packed_projection_weight(projection: WidelyLinear) -> Tensor:
    return packed_widely_linear_weight(
        projection.weight_real,
        projection.weight_imag,
        projection.conjugate_real,
        projection.conjugate_imag,
    )


def _packed_projection_bias(projection: WidelyLinear) -> Tensor:
    bias = packed_widely_linear_bias(projection.bias_real, projection.bias_imag)
    if bias is None:
        message = "packed transition requires affine biases"
        raise RuntimeError(message)
    return bias


def _supports_packed_residual_cffn(
    real: Tensor,
    imag: Tensor,
    *,
    input_projection: WidelyLinear,
    output_projection: WidelyLinear,
    activation: ComplexFFNActivation,
    activation_scale: Tensor | float | None,
    residual_scale: Tensor,
    residual_source: ComplexField,
) -> bool:
    modes = real.shape[-1]
    return (
        type(input_projection) is WidelyLinear
        and type(output_projection) is WidelyLinear
        and _same_tensor_contract(real, imag)
        and _same_tensor_contract(real, residual_source[0])
        and _same_tensor_contract(real, residual_source[1])
        and activation == "cartesian_silu"
        and activation_scale is None
        and tuple(residual_scale.shape) in {(), (modes,)}
        and input_projection.input_modes == modes
        and output_projection.input_modes == input_projection.output_modes
        and output_projection.output_modes == modes
        and input_projection.bias_real is not None
        and input_projection.bias_imag is not None
        and output_projection.bias_real is not None
        and output_projection.bias_imag is not None
        and module_calls_are_transparent(input_projection, output_projection)
    )


@dataclass(frozen=True, slots=True)
class PackedRMSNormSpec:
    """Live parameters needed to reproduce one complex RMS norm."""

    weight: Tensor
    epsilon: float


@dataclass(frozen=True, slots=True)
class PackedCFFNSpec:
    """Live parameters needed to select and execute one residual CFFN."""

    input_projection: WidelyLinear
    output_projection: WidelyLinear
    activation: ComplexFFNActivation
    activation_scale: Tensor | float | None
    residual_scale: Tensor


@dataclass(frozen=True, slots=True)
class PackedPostFusionSpec:
    """Atomic capability for the optional post-fusion CFFN."""

    norm: PackedRMSNormSpec
    cffn: PackedCFFNSpec


@dataclass(frozen=True, slots=True)
class PackedPostCarrySpec:
    """Explicit transition-owned contract for the packed execution graph."""

    modes: int
    input_modes: int
    hidden_modes: int
    output_modes: int
    training: bool
    direction_mixer: WidelyLinear
    ffn_norm: PackedRMSNormSpec
    ffn: PackedCFFNSpec
    output_norm: PackedRMSNormSpec
    output_projection: WidelyLinear
    pole_scale: Tensor
    carry_weight: Tensor | None
    carry_projection: ComplexLinear | None
    post_fusion: PackedPostFusionSpec | None = None


@dataclass(frozen=True, slots=True)
class PackedPostCarryWeights:
    """Input-independent packed parameter blocks for one transition."""

    key: tuple[object, ...]
    direction_weight: Tensor
    direction_bias: Tensor
    ffn_input_weight: Tensor
    ffn_input_bias: Tensor
    ffn_output_weight: Tensor
    ffn_output_bias: Tensor
    output_weight: Tensor
    output_bias: Tensor
    ffn_norm_scale: Tensor
    output_norm_scale: Tensor
    layer_scale: Tensor
    # Mode-wise S2D carry, transposed to [4, M].  ``None`` for the
    # strict-complex variant, whose ComplexLinear carry is evaluated as-is.
    #
    # That carry could be folded into one wider GEMM, but the wider weight makes
    # cuBLAS pick a different kernel at some shapes and adds another numerical
    # drift. The pole branch is the bulk of the work, so the carry is left alone.
    carry_weight: Tensor | None
    # Post-fusion mode CFFN, present only on ``S2DPostFusionCFFNTransition``.
    # It is the same norm-then-residual-CFFN shape as the pole branch, so the
    # packed helpers are reused rather than restated.
    post_norm_scale: Tensor | None = None
    post_input_weight: Tensor | None = None
    post_input_bias: Tensor | None = None
    post_output_weight: Tensor | None = None
    post_output_bias: Tensor | None = None
    post_scale: Tensor | None = None


class PackedPostCarryTransition(Protocol):
    """Structural boundary required by the packed inference runtime."""

    _packed_inference_cache: PackedPostCarryWeights | None

    def packed_postcarry_spec(self) -> PackedPostCarrySpec | None: ...

    def modules(self) -> Iterator[torch.nn.Module]: ...


def packed_postcarry_weights_resident_bytes(weights: PackedPostCarryWeights) -> int:
    """Return the physical bytes held by one derived packed-weight cache.

    Storage aliases are counted once.  This reports persistent tensor storage,
    not transient workspace or CUDA allocator reservation, so an autotuner can
    add it to each candidate's steady-step peak instead of losing the packed
    parameter replication when warmup establishes the memory baseline.
    """
    seen_storages: set[tuple[torch.device, int]] = set()
    resident_bytes = 0
    for field in fields(weights):
        value: object = getattr(weights, field.name)
        if not isinstance(value, torch.Tensor):
            continue
        storage = value.untyped_storage()
        storage_key = value.device, storage.data_ptr()
        if storage_key in seen_storages:
            continue
        seen_storages.add(storage_key)
        resident_bytes += storage.nbytes()
    return resident_bytes


def packed_postcarry_cache_resident_bytes(
    transition: PackedPostCarryTransition,
) -> int:
    """Return bytes retained by ``transition``'s current derived cache."""
    cache = transition._packed_inference_cache
    return 0 if cache is None else packed_postcarry_weights_resident_bytes(cache)


def _require_packed_spec(
    transition: PackedPostCarryTransition,
) -> PackedPostCarrySpec:
    spec = transition.packed_postcarry_spec()
    if spec is None:
        message = "transition does not expose a packed post-carry capability"
        raise RuntimeError(message)
    return spec


def _identity(tensor: Tensor) -> tuple[int, int]:
    """Identify a tensor's storage and mutation count without a device sync."""
    return tensor.data_ptr(), tensor._version


def _projection_identity(projection: WidelyLinear) -> tuple[tuple[int, int], ...]:
    tensors = (
        projection.weight_real,
        projection.weight_imag,
        projection.conjugate_real,
        projection.conjugate_imag,
        projection.bias_real,
        projection.bias_imag,
    )
    return tuple(_identity(tensor) for tensor in tensors if tensor is not None)


def _cache_key(
    transition: PackedPostCarryTransition,
    reference: Tensor,
) -> tuple[object, ...]:
    spec = _require_packed_spec(transition)
    projections = (
        spec.direction_mixer,
        spec.ffn.input_projection,
        spec.ffn.output_projection,
        spec.output_projection,
    )
    carry = (
        (spec.carry_projection.weight_real, spec.carry_projection.weight_imag)
        if spec.carry_projection is not None
        else (spec.carry_weight,)
    )
    scalars = (
        spec.ffn_norm.weight,
        spec.output_norm.weight,
        spec.ffn.residual_scale,
        *(tensor for tensor in carry if tensor is not None),
    )
    if spec.post_fusion is not None:
        projections = (
            *projections,
            spec.post_fusion.cffn.input_projection,
            spec.post_fusion.cffn.output_projection,
        )
        scalars = (
            *scalars,
            spec.post_fusion.norm.weight,
            spec.post_fusion.cffn.residual_scale,
        )
    return (
        reference.dtype,
        reference.device,
        *(_projection_identity(projection) for projection in projections),
        *(_identity(tensor) for tensor in scalars),
    )


def _paired_norm_scale(norm: PackedRMSNormSpec, dtype: torch.dtype) -> Tensor:
    scale = norm.weight.to(dtype=dtype)
    return torch.cat((scale, scale), dim=0)


def build_packed_weights(
    transition: PackedPostCarryTransition,
    reference: Tensor,
) -> PackedPostCarryWeights:
    """Fuse every input-independent parameter block exactly once."""
    spec = _require_packed_spec(transition)
    return _build_packed_weights(
        spec,
        reference,
        key=_cache_key(transition, reference),
    )


def _build_packed_weights(
    spec: PackedPostCarrySpec,
    reference: Tensor,
    *,
    key: tuple[object, ...],
) -> PackedPostCarryWeights:
    """Build packed blocks from live parameters without changing autograd."""
    carry_weight = spec.carry_weight
    if spec.carry_projection is not None:
        packed_carry_weight = None
    elif carry_weight is not None:
        packed_carry_weight = carry_weight.transpose(0, 1).to(dtype=reference.dtype).contiguous()
    else:
        message = "packed post-carry capability requires exactly one carry representation"
        raise RuntimeError(message)
    dtype = reference.dtype
    layer_scale = spec.ffn.residual_scale.to(dtype=dtype)
    return PackedPostCarryWeights(
        key=key,
        direction_weight=_packed_projection_weight(spec.direction_mixer),
        direction_bias=_packed_projection_bias(spec.direction_mixer),
        ffn_input_weight=_packed_projection_weight(spec.ffn.input_projection),
        ffn_input_bias=_packed_projection_bias(spec.ffn.input_projection),
        ffn_output_weight=_packed_projection_weight(spec.ffn.output_projection),
        ffn_output_bias=_packed_projection_bias(spec.ffn.output_projection),
        output_weight=_packed_projection_weight(spec.output_projection),
        output_bias=_packed_projection_bias(spec.output_projection),
        ffn_norm_scale=_paired_norm_scale(spec.ffn_norm, dtype),
        output_norm_scale=_paired_norm_scale(spec.output_norm, dtype),
        layer_scale=torch.cat((layer_scale, layer_scale), dim=0),
        carry_weight=packed_carry_weight,
        **_post_fusion_blocks(spec.post_fusion, dtype),
    )


def build_trainable_packed_weights(
    transition: PackedPostCarryTransition,
    reference: Tensor,
) -> PackedPostCarryWeights:
    """Build differentiable packed blocks inside a compiled training graph."""
    return _build_packed_weights(
        _require_packed_spec(transition),
        reference,
        key=(),
    )


def _post_fusion_blocks(
    post_fusion: PackedPostFusionSpec | None,
    dtype: torch.dtype,
) -> dict[str, Tensor]:
    """Fuse the post-fusion CFFN blocks, or nothing when the subclass is absent."""
    if post_fusion is None:
        return {}
    post_scale = post_fusion.cffn.residual_scale.to(dtype=dtype)
    return {
        "post_norm_scale": _paired_norm_scale(post_fusion.norm, dtype),
        "post_input_weight": _packed_projection_weight(post_fusion.cffn.input_projection),
        "post_input_bias": _packed_projection_bias(post_fusion.cffn.input_projection),
        "post_output_weight": _packed_projection_weight(post_fusion.cffn.output_projection),
        "post_output_bias": _packed_projection_bias(post_fusion.cffn.output_projection),
        "post_scale": torch.cat((post_scale, post_scale), dim=0),
    }


def _can_use_packed_transition(
    transition: PackedPostCarryTransition,
    real: Tensor,
    imag: Tensor,
    carry_real: Tensor,
    carry_imag: Tensor,
) -> bool:
    spec = transition.packed_postcarry_spec()
    if spec is None:
        return False
    coordinates = (real, imag, carry_real, carry_imag)
    eligible = (
        os.environ.get(_DISABLE_VARIABLE, "0") != "1"
        and not torch.compiler.is_compiling()
        and not spec.training
        and not torch.is_grad_enabled()
        and all(_same_tensor_contract(real, coordinate) for coordinate in coordinates[1:])
        and real.ndim == 4
        and real.shape[-1] == spec.input_modes
        and real.is_cuda
        and real.dtype is torch.float32
        and not any(tensor.requires_grad for tensor in coordinates)
        and module_calls_are_transparent(*tuple(transition.modules())[1:])
        and all(
            projection.bias_real is not None and projection.bias_imag is not None
            for projection in (
                spec.direction_mixer,
                spec.ffn.input_projection,
                spec.ffn.output_projection,
                spec.output_projection,
            )
        )
    )
    if not eligible:
        return False
    probe = real[..., :1].expand(*real.shape[:-1], spec.hidden_modes)
    return _supports_packed_residual_cffn(
        probe,
        imag=probe,
        input_projection=spec.ffn.input_projection,
        output_projection=spec.ffn.output_projection,
        activation=spec.ffn.activation,
        activation_scale=spec.ffn.activation_scale,
        residual_scale=spec.ffn.residual_scale,
        residual_source=(probe, probe),
    )


def can_use_packed_postcarry_inference(
    transition: PackedPostCarryTransition,
    real: Tensor,
    imag: Tensor,
    carry_real: Tensor,
    carry_imag: Tensor,
) -> bool:
    """Report whether the fused packed forward satisfies its structural contract."""
    return _can_use_packed_transition(
        transition,
        real,
        imag,
        carry_real,
        carry_imag,
    )


def can_use_packed_postfusion_inference(
    transition: PackedPostCarryTransition,
    real: Tensor,
    imag: Tensor,
    carry_real: Tensor,
    carry_imag: Tensor,
) -> bool:
    """Report whether the post-fusion subclass satisfies the packed contract.

    Everything the pole branch requires still applies, so that predicate is
    reused rather than restated; only the appended mode CFFN is checked here.
    """
    spec = transition.packed_postcarry_spec()
    if spec is None:
        return False
    post_fusion = spec.post_fusion
    if post_fusion is None or spec.carry_projection is not None:
        return False
    if not _can_use_packed_transition(
        transition,
        real,
        imag,
        carry_real,
        carry_imag,
    ):
        return False
    probe = real[..., :1].expand(*real.shape[:-1], spec.output_modes)
    return _supports_packed_residual_cffn(
        probe,
        imag=probe,
        input_projection=post_fusion.cffn.input_projection,
        output_projection=post_fusion.cffn.output_projection,
        activation=post_fusion.cffn.activation,
        activation_scale=post_fusion.cffn.activation_scale,
        residual_scale=post_fusion.cffn.residual_scale,
        residual_source=(probe, probe),
    )


def can_use_packed_postfusion_training(
    transition: PackedPostCarryTransition,
    real: Tensor,
    imag: Tensor,
    carry_real: Tensor,
    carry_imag: Tensor,
) -> bool:
    """Select the differentiable packed graph during CUDA compilation.

    Selection is structural: no batch size, image size, or GPU model is baked
    into the decision. Unsupported modules and eager calls retain the generic
    split-coordinate path.
    """
    spec = transition.packed_postcarry_spec()
    if spec is None:
        return False
    post_fusion = spec.post_fusion
    coordinates = (real, imag, carry_real, carry_imag)
    # BF16 already packs each common CFFN inside the compiled fallback. Keeping
    # the entire transition wide increases its large-stage GEMM/layout cost.
    eligible = (
        os.environ.get(_DISABLE_TRAINING_VARIABLE, "0") != "1"
        and torch.compiler.is_compiling()
        and spec.training
        and torch.is_grad_enabled()
        and post_fusion is not None
        and spec.carry_projection is None
        and all(_same_tensor_contract(real, coordinate) for coordinate in coordinates[1:])
        and real.ndim == 4
        and real.shape[-1] == spec.input_modes
        and real.is_cuda
        and real.dtype is torch.float32
        and type(spec.direction_mixer) is WidelyLinear
        and type(spec.output_projection) is WidelyLinear
        and module_calls_are_transparent(*tuple(transition.modules())[1:])
        and all(
            projection.bias_real is not None and projection.bias_imag is not None
            for projection in (
                spec.direction_mixer,
                spec.ffn.input_projection,
                spec.ffn.output_projection,
                spec.output_projection,
                post_fusion.cffn.input_projection,
                post_fusion.cffn.output_projection,
            )
        )
    )
    if not eligible or post_fusion is None:
        return False
    hidden_probe = real[..., :1].expand(*real.shape[:-1], spec.hidden_modes)
    output_probe = real[..., :1].expand(*real.shape[:-1], spec.output_modes)
    return _supports_packed_residual_cffn(
        hidden_probe,
        imag=hidden_probe,
        input_projection=spec.ffn.input_projection,
        output_projection=spec.ffn.output_projection,
        activation=spec.ffn.activation,
        activation_scale=spec.ffn.activation_scale,
        residual_scale=spec.ffn.residual_scale,
        residual_source=(hidden_probe, hidden_probe),
    ) and _supports_packed_residual_cffn(
        output_probe,
        imag=output_probe,
        input_projection=post_fusion.cffn.input_projection,
        output_projection=post_fusion.cffn.output_projection,
        activation=post_fusion.cffn.activation,
        activation_scale=post_fusion.cffn.activation_scale,
        residual_scale=post_fusion.cffn.residual_scale,
        residual_source=(output_probe, output_probe),
    )


def _packed_rms_norm(
    packed: Tensor,
    paired_scale: Tensor,
    epsilon: float,
    width: int,
) -> Tensor:
    """Normalize a packed field with the split path's exact reduction order."""
    paired = packed.unflatten(-1, (2, width))
    energy = paired.float().square().sum(dim=-2).mean(dim=-1, keepdim=True)
    inverse_rms = torch.rsqrt(energy + epsilon).to(dtype=packed.dtype)
    return packed * inverse_rms * paired_scale


def _packed_carry(carry: Tensor, weight: Tensor, modes: int) -> Tensor:
    shape = (*carry.shape[:-1], _CARRY_POSITIONS, modes)
    return (carry.reshape(shape) * weight).sum(dim=-2)


def packed_pole_update(
    transition: PackedPostCarryTransition,
    real: Tensor,
    imag: Tensor,
    *,
    weights: PackedPostCarryWeights,
) -> Tensor:
    """Run the pole branch end to end, returning a packed ``[..., 2M]`` field.

    Shared by the carry-main transition and its post-fusion subclass so both
    dispatch to one implementation of the same graph.
    """
    spec = _require_packed_spec(transition)
    hidden = torch.nn.functional.linear(
        torch.cat((real, imag), dim=-1),
        weights.direction_weight,
        weights.direction_bias,
    )
    unit = _packed_rms_norm(
        hidden,
        weights.ffn_norm_scale,
        spec.ffn_norm.epsilon,
        spec.hidden_modes,
    )
    activated = torch.nn.functional.silu(
        torch.nn.functional.linear(
            unit,
            weights.ffn_input_weight,
        )
        + weights.ffn_input_bias,
    )
    update = torch.nn.functional.linear(
        activated,
        weights.ffn_output_weight,
    ) + weights.ffn_output_bias
    normalized = _packed_rms_norm(
        hidden + weights.layer_scale * update,
        weights.output_norm_scale,
        spec.output_norm.epsilon,
        spec.hidden_modes,
    )
    return torch.nn.functional.linear(
        normalized,
        weights.output_weight,
        weights.output_bias,
    )


def _packed_outer_residual(
    transition: PackedPostCarryTransition,
    real: Tensor,
    imag: Tensor,
    carry_real: Tensor,
    carry_imag: Tensor,
    *,
    weights: PackedPostCarryWeights,
) -> Tensor:
    """Return the packed outer residual ``C + alpha * F`` as ``[..., 2M]``.

    The carry halves are joined at M rather than 4M: the wide S2D field never
    has to be copied, and the merge then costs one elementwise op instead of two.
    """
    if weights.carry_weight is None:
        message = "packed outer residual requires the mode-wise S2D carry"
        raise RuntimeError(message)
    pole = packed_pole_update(transition, real, imag, weights=weights)
    spec = _require_packed_spec(transition)
    state = torch.cat(
        (
            _packed_carry(carry_real, weights.carry_weight, spec.modes),
            _packed_carry(carry_imag, weights.carry_weight, spec.modes),
        ),
        dim=-1,
    )
    return state + spec.pole_scale.to(dtype=pole.dtype) * pole


def packed_postfusion_inference(
    transition: PackedPostCarryTransition,
    real: Tensor,
    imag: Tensor,
    carry_real: Tensor,
    carry_imag: Tensor,
    *,
    weights: PackedPostCarryWeights,
) -> ComplexField:
    """Evaluate the post-fusion subclass entirely in the packed layout.

    The subclass appends one norm-then-residual-CFFN to the outer residual,
    which is the same shape the pole branch already runs packed, so the whole
    transition stays packed from the direction mixer to a single closing split.
    """
    post_input_bias = weights.post_input_bias
    post_output_bias = weights.post_output_bias
    if (
        weights.post_norm_scale is None
        or weights.post_input_weight is None
        or post_input_bias is None
        or weights.post_output_weight is None
        or post_output_bias is None
        or weights.post_scale is None
    ):
        message = "packed post-fusion dispatch requires the post-fusion blocks"
        raise RuntimeError(message)
    spec = _require_packed_spec(transition)
    post_fusion = spec.post_fusion
    if post_fusion is None:
        message = "packed post-fusion dispatch requires its explicit capability"
        raise RuntimeError(message)
    outer = _packed_outer_residual(
        transition,
        real,
        imag,
        carry_real,
        carry_imag,
        weights=weights,
    )
    unit = _packed_rms_norm(
        outer,
        weights.post_norm_scale,
        post_fusion.norm.epsilon,
        spec.output_modes,
    )
    activated = torch.nn.functional.silu(
        torch.nn.functional.linear(
            unit,
            weights.post_input_weight,
        )
        + post_input_bias,
    )
    update = torch.nn.functional.linear(
        activated,
        weights.post_output_weight,
    ) + post_output_bias
    fused_real, fused_imag = (outer + weights.post_scale * update).split(
        spec.output_modes,
        dim=-1,
    )
    return fused_real, fused_imag


def packed_postfusion_training(
    transition: PackedPostCarryTransition,
    real: Tensor,
    imag: Tensor,
    carry_real: Tensor,
    carry_imag: Tensor,
) -> ComplexField:
    """Evaluate compiled training without split-coordinate intermediates."""
    return packed_postfusion_inference(
        transition,
        real,
        imag,
        carry_real,
        carry_imag,
        weights=build_trainable_packed_weights(transition, real),
    )


def packed_postcarry_inference(
    transition: PackedPostCarryTransition,
    real: Tensor,
    imag: Tensor,
    carry_real: Tensor,
    carry_imag: Tensor,
    *,
    weights: PackedPostCarryWeights,
) -> ComplexField:
    """Evaluate the transition without leaving the packed activation layout."""
    spec = _require_packed_spec(transition)
    pole = packed_pole_update(transition, real, imag, weights=weights)
    pole_real, pole_imag = pole.split(spec.output_modes, dim=-1)
    if spec.carry_projection is not None:
        # A ComplexLinear carry is four GEMMs, and packing them into one wider
        # weight makes cuBLAS pick a different kernel at some shapes, which
        # changes the accumulation order.  Exactness outranks the two kernels.
        state_real, state_imag = spec.carry_projection(carry_real, carry_imag)
    else:
        # The mode-wise carry stays split: packing it would copy a full 4M-wide
        # activation to save two kernels, losing more bandwidth than it saves.
        weight = weights.carry_weight
        if weight is None:
            message = "mode-wise carry requires its fused weight"
            raise RuntimeError(message)
        state_real = _packed_carry(carry_real, weight, spec.modes)
        state_imag = _packed_carry(carry_imag, weight, spec.modes)
    scale = spec.pole_scale.to(dtype=pole.dtype)
    return state_real + scale * pole_real, state_imag + scale * pole_imag


def cached_packed_weights(
    transition: PackedPostCarryTransition,
    reference: Tensor,
) -> PackedPostCarryWeights:
    """Return fused blocks, rebuilding them only after a parameter changed.

    Under CUDA Graph replay this check does not run at all, and the recorded
    GEMMs point at whichever fused blocks existed at capture time.  Mutating a
    parameter after capture therefore changes the eager result but not the
    replayed one.  Warm up before capturing, and treat a captured graph as
    frozen against the weights it was captured with.
    """
    cache = transition._packed_inference_cache
    key = _cache_key(transition, reference)
    if cache is not None and cache.key == key:
        return cache
    weights = build_packed_weights(transition, reference)
    transition._packed_inference_cache = weights
    return weights
