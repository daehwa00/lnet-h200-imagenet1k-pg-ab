"""Horizontal-to-vertical D4 product scan composition and memory policy."""

from __future__ import annotations

# The composition boundary intentionally uses package-private primitive ops so
# their intermediate autograd contexts do not retain horizontal scan states.
# pyright: reportPrivateUsage=false
from typing import TYPE_CHECKING, Literal, Protocol, cast

import torch
from torch import Tensor
from torch.library import triton_op

from .pac_product_scan_contracts import (
    DEFAULT_EPSILON,
    ProductGainNormalization,
    gain_kind,
    gain_normalization,
)
from .pac_product_scan_normalization import static_product_scan_auxiliary
from .pac_triton_bidirectional_product_scan import (
    _forward_op as _horizontal_forward_op,
)
from .pac_triton_bidirectional_product_scan import (
    _recomputed_backward_op as _horizontal_recomputed_backward_op,
)
from .pac_triton_bidirectional_product_scan import pac_triton_bidirectional_product_scan
from .pac_triton_grouped_path_cffn import (
    d4_grouped_cell_path_collapse_reference,
    d4_grouped_cell_path_swiglu_reference,
)
from .pac_triton_product_scan_coarse4 import (
    BACKWARD_LAUNCH_NAME,
    DESCRIPTOR_BACKWARD_LAUNCH_NAME,
    FULL16_BACKWARD_LAUNCH_NAME,
    _launch_product_scan4_backward,
    _product_scan_coarse4_op,
    _product_scan_descriptor4_op,
    _product_scan_full16_op,
    _product_scan_path_collapse_backward_op,
    _product_scan_path_collapse_op,
    pac_triton_product_scan_coarse4,
    pac_triton_product_scan_descriptor4,
    pac_triton_product_scan_full16,
)

if TYPE_CHECKING:
    from collections.abc import Callable

ProductScanEpilogue = Literal["coarse", "full16", "descriptor"]
ScanMemoryPolicy = Literal["retain", "recompute"]
ProductScanOutput = tuple[Tensor, Tensor, Tensor] | Tensor
PathCollapseParameters = tuple[Tensor, Tensor, Tensor, Tensor]
PathCollapsePipelineGradients = tuple[
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
]
ProductScanInputGradients = tuple[
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
]


class _AutogradContext(Protocol):
    saved_tensors: tuple[Tensor, ...]
    epsilon: float
    gain_kind: int
    path_swiglu: bool

    def save_for_backward(self, *tensors: Tensor) -> None: ...


def _product_scan_pipeline(
    decay_x_real: Tensor,
    decay_x_imag: Tensor,
    gamma_x_real: Tensor,
    gamma_x_imag: Tensor,
    decay_y_real: Tensor,
    decay_y_imag: Tensor,
    gamma_y_real: Tensor,
    gamma_y_imag: Tensor,
    excitation_real: Tensor,
    excitation_imag: Tensor,
    *,
    epilogue: ProductScanEpilogue,
    gain_normalization: ProductGainNormalization,
    epsilon: float,
) -> ProductScanOutput:
    pole_x = decay_x_real, decay_x_imag, gamma_x_real, gamma_x_imag
    pole_y = decay_y_real, decay_y_imag, gamma_y_real, gamma_y_imag
    positive_real, positive_imag, negative_real, negative_imag = (
        pac_triton_bidirectional_product_scan(
            pole_x,
            (excitation_real, excitation_imag),
        )
    )
    positive_horizontal = positive_real, positive_imag
    negative_horizontal = negative_real, negative_imag
    if epilogue == "coarse":
        return pac_triton_product_scan_coarse4(
            pole_x,
            pole_y,
            positive_horizontal,
            negative_horizontal,
            gain_normalization=gain_normalization,
            epsilon=epsilon,
        )
    if epilogue == "full16":
        return pac_triton_product_scan_full16(
            pole_x,
            pole_y,
            positive_horizontal,
            negative_horizontal,
            gain_normalization=gain_normalization,
            epsilon=epsilon,
        )
    return pac_triton_product_scan_descriptor4(
        pole_x,
        pole_y,
        positive_horizontal,
        negative_horizontal,
        gain_normalization=gain_normalization,
        epsilon=epsilon,
    )


def _recomputed_forward_inputs(
    inputs: tuple[Tensor, ...],
) -> tuple[
    tuple[Tensor, Tensor, Tensor, Tensor],
    tuple[Tensor, Tensor, Tensor, Tensor],
    tuple[Tensor, Tensor],
]:
    return (
        cast("tuple[Tensor, Tensor, Tensor, Tensor]", inputs[:4]),
        cast("tuple[Tensor, Tensor, Tensor, Tensor]", inputs[4:8]),
        cast("tuple[Tensor, Tensor]", inputs[8:10]),
    )


@triton_op("lnet::pac_product_scan_pipeline_coarse_recompute", mutates_args={})
def _coarse_recompute_op(
    decay_x_real: Tensor,
    decay_x_imag: Tensor,
    gamma_x_real: Tensor,
    gamma_x_imag: Tensor,
    decay_y_real: Tensor,
    decay_y_imag: Tensor,
    gamma_y_real: Tensor,
    gamma_y_imag: Tensor,
    excitation_real: Tensor,
    excitation_imag: Tensor,
    epsilon: float,
    gain_kind: int,
) -> tuple[Tensor, Tensor, Tensor]:
    pole_x = decay_x_real, decay_x_imag, gamma_x_real, gamma_x_imag
    pole_y = decay_y_real, decay_y_imag, gamma_y_real, gamma_y_imag
    horizontal = _horizontal_forward_op(*pole_x, excitation_real, excitation_imag)
    variance_x, variance_y, global_inverse_gain = static_product_scan_auxiliary(
        pole_x,
        pole_y,
        excitation_real,
        epsilon=epsilon,
        gain_kind=gain_kind,
    )
    return _product_scan_coarse4_op(
        *pole_y,
        *horizontal,
        variance_x,
        variance_y,
        global_inverse_gain,
        epsilon,
        gain_kind,
    )


@triton_op("lnet::pac_product_scan_pipeline_full16_recompute", mutates_args={})
def _full16_recompute_op(
    decay_x_real: Tensor,
    decay_x_imag: Tensor,
    gamma_x_real: Tensor,
    gamma_x_imag: Tensor,
    decay_y_real: Tensor,
    decay_y_imag: Tensor,
    gamma_y_real: Tensor,
    gamma_y_imag: Tensor,
    excitation_real: Tensor,
    excitation_imag: Tensor,
    epsilon: float,
    gain_kind: int,
) -> tuple[Tensor, Tensor, Tensor]:
    pole_x = decay_x_real, decay_x_imag, gamma_x_real, gamma_x_imag
    pole_y = decay_y_real, decay_y_imag, gamma_y_real, gamma_y_imag
    horizontal = _horizontal_forward_op(*pole_x, excitation_real, excitation_imag)
    variance_x, variance_y, global_inverse_gain = static_product_scan_auxiliary(
        pole_x,
        pole_y,
        excitation_real,
        epsilon=epsilon,
        gain_kind=gain_kind,
    )
    return _product_scan_full16_op(
        *pole_y,
        *horizontal,
        variance_x,
        variance_y,
        global_inverse_gain,
        epsilon,
        gain_kind,
    )


@triton_op("lnet::pac_product_scan_pipeline_path_collapse_recompute", mutates_args={})
def _path_collapse_recompute_op(
    decay_x_real: Tensor,
    decay_x_imag: Tensor,
    gamma_x_real: Tensor,
    gamma_x_imag: Tensor,
    decay_y_real: Tensor,
    decay_y_imag: Tensor,
    gamma_y_real: Tensor,
    gamma_y_imag: Tensor,
    excitation_real: Tensor,
    excitation_imag: Tensor,
    path_input_weight: Tensor,
    path_input_bias: Tensor,
    path_output_weight: Tensor,
    path_output_bias: Tensor,
    epsilon: float,
    gain_kind: int,
    path_swiglu: bool,  # noqa: FBT001
) -> tuple[Tensor, Tensor, Tensor]:
    pole_x = decay_x_real, decay_x_imag, gamma_x_real, gamma_x_imag
    pole_y = decay_y_real, decay_y_imag, gamma_y_real, gamma_y_imag
    horizontal = _horizontal_forward_op(*pole_x, excitation_real, excitation_imag)
    variance_x, variance_y, global_inverse_gain = static_product_scan_auxiliary(
        pole_x,
        pole_y,
        excitation_real,
        epsilon=epsilon,
        gain_kind=gain_kind,
    )
    return _product_scan_path_collapse_op(
        *pole_y,
        *horizontal,
        variance_x,
        variance_y,
        global_inverse_gain,
        path_input_weight,
        path_input_bias,
        path_output_weight,
        path_output_bias,
        epsilon,
        gain_kind,
        path_swiglu,
    )


@triton_op("lnet::pac_product_scan_pipeline_descriptor_recompute", mutates_args={})
def _descriptor_recompute_op(
    decay_x_real: Tensor,
    decay_x_imag: Tensor,
    gamma_x_real: Tensor,
    gamma_x_imag: Tensor,
    decay_y_real: Tensor,
    decay_y_imag: Tensor,
    gamma_y_real: Tensor,
    gamma_y_imag: Tensor,
    excitation_real: Tensor,
    excitation_imag: Tensor,
    epsilon: float,
    gain_kind: int,
) -> Tensor:
    pole_x = decay_x_real, decay_x_imag, gamma_x_real, gamma_x_imag
    pole_y = decay_y_real, decay_y_imag, gamma_y_real, gamma_y_imag
    horizontal = _horizontal_forward_op(*pole_x, excitation_real, excitation_imag)
    variance_x, variance_y, global_inverse_gain = static_product_scan_auxiliary(
        pole_x,
        pole_y,
        excitation_real,
        epsilon=epsilon,
        gain_kind=gain_kind,
    )
    return _product_scan_descriptor4_op(
        *pole_y,
        *horizontal,
        variance_x,
        variance_y,
        global_inverse_gain,
        epsilon,
        gain_kind,
    )


def _setup_recompute_context(
    ctx: _AutogradContext,
    inputs: tuple[Tensor | float | int, ...],
    descriptor: Tensor,
) -> None:
    *tensors, epsilon, gain_kind = inputs
    ctx.epsilon = float(epsilon)
    ctx.gain_kind = int(gain_kind)
    ctx.save_for_backward(*cast("tuple[Tensor, ...]", tuple(tensors)), descriptor)


def _setup_coarse_recompute_context(
    ctx: _AutogradContext,
    inputs: tuple[Tensor | float | int, ...],
    output: tuple[Tensor, Tensor, Tensor],
) -> None:
    _setup_recompute_context(ctx, inputs, output[2])


def _setup_descriptor_recompute_context(
    ctx: _AutogradContext,
    inputs: tuple[Tensor | float | int, ...],
    output: Tensor,
) -> None:
    _setup_recompute_context(ctx, inputs, output)


def _setup_path_collapse_recompute_context(
    ctx: _AutogradContext,
    inputs: tuple[Tensor | float | int | bool, ...],
    output: tuple[Tensor, Tensor, Tensor],
) -> None:
    *tensors, epsilon, gain_kind, path_swiglu = inputs
    ctx.epsilon = float(epsilon)
    ctx.gain_kind = int(gain_kind)
    ctx.path_swiglu = bool(path_swiglu)
    ctx.save_for_backward(*cast("tuple[Tensor, ...]", tuple(tensors)), output[2])


def _reference_recompute_backward(
    ctx: _AutogradContext,
    grad_outputs: tuple[Tensor, ...],
    *,
    epilogue: ProductScanEpilogue,
) -> tuple[Tensor | None, ...]:
    *inputs, _ = ctx.saved_tensors
    differentiable = tuple(value.detach().requires_grad_() for value in inputs)
    pole_x, pole_y, source = _recomputed_forward_inputs(differentiable)
    with torch.enable_grad():
        output = _product_scan_pipeline(
            *pole_x,
            *pole_y,
            *source,
            epilogue=epilogue,
            gain_normalization=gain_normalization(ctx.gain_kind),
            epsilon=ctx.epsilon,
        )
        outputs = output if isinstance(output, tuple) else (output,)
        gradients = torch.autograd.grad(outputs, differentiable, grad_outputs)
    return *gradients, None, None


def _cuda_recompute_backward_impl(
    inputs: tuple[Tensor, ...],
    descriptor: Tensor,
    grad_coarse_real: Tensor,
    grad_coarse_imag: Tensor,
    grad_descriptor: Tensor,
    *,
    emit_coarse: bool,
    full_coarse: bool,
    epsilon: float,
    gain_kind: int,
) -> ProductScanInputGradients:
    pole_x, pole_y, source = _recomputed_forward_inputs(inputs)
    horizontal = _horizontal_forward_op(*pole_x, *source)
    variance_x, variance_y, global_inverse_gain = static_product_scan_auxiliary(
        pole_x,
        pole_y,
        source[0],
        epsilon=epsilon,
        gain_kind=gain_kind,
    )
    height, width = source[0].shape[1:3]
    descriptor_gradient_factor = (
        grad_descriptor * torch.exp(-descriptor) * (1.0 / (height * width))
    ).contiguous()
    if emit_coarse:
        launch_name = FULL16_BACKWARD_LAUNCH_NAME if full_coarse else BACKWARD_LAUNCH_NAME
        vertical_gradients = _launch_product_scan4_backward(
            *pole_y,
            *horizontal,
            variance_x,
            variance_y,
            global_inverse_gain,
            grad_coarse_real.contiguous(),
            grad_coarse_imag.contiguous(),
            descriptor_gradient_factor,
            epsilon,
            gain_kind,
            emit_coarse=True,
            full_coarse=full_coarse,
            launch_name=launch_name,
            source_gradient_buffers=horizontal,
        )
    else:
        vertical_gradients = _launch_product_scan4_backward(
            *pole_y,
            *horizontal,
            variance_x,
            variance_y,
            global_inverse_gain,
            grad_coarse_real,
            grad_coarse_imag,
            descriptor_gradient_factor,
            epsilon,
            gain_kind,
            emit_coarse=False,
            full_coarse=False,
            launch_name=DESCRIPTOR_BACKWARD_LAUNCH_NAME,
            source_gradient_buffers=horizontal,
        )
    horizontal_gradients = _horizontal_recomputed_backward_op(
        *pole_x,
        *source,
        *horizontal,
    )
    return cast(
        "ProductScanInputGradients",
        (
            *horizontal_gradients[:4],
            *vertical_gradients[:4],
            *horizontal_gradients[4:],
        ),
    )


@triton_op("lnet::pac_product_scan_pipeline_coarse_recompute_backward", mutates_args={})
def _coarse_recompute_backward_cuda_op(
    decay_x_real: Tensor,
    decay_x_imag: Tensor,
    gamma_x_real: Tensor,
    gamma_x_imag: Tensor,
    decay_y_real: Tensor,
    decay_y_imag: Tensor,
    gamma_y_real: Tensor,
    gamma_y_imag: Tensor,
    excitation_real: Tensor,
    excitation_imag: Tensor,
    descriptor: Tensor,
    grad_coarse_real: Tensor,
    grad_coarse_imag: Tensor,
    grad_descriptor: Tensor,
    epsilon: float,
    gain_kind: int,
) -> ProductScanInputGradients:
    inputs = (
        decay_x_real,
        decay_x_imag,
        gamma_x_real,
        gamma_x_imag,
        decay_y_real,
        decay_y_imag,
        gamma_y_real,
        gamma_y_imag,
        excitation_real,
        excitation_imag,
    )
    return _cuda_recompute_backward_impl(
        inputs,
        descriptor,
        grad_coarse_real,
        grad_coarse_imag,
        grad_descriptor,
        emit_coarse=True,
        full_coarse=False,
        epsilon=epsilon,
        gain_kind=gain_kind,
    )


@triton_op("lnet::pac_product_scan_pipeline_full16_recompute_backward", mutates_args={})
def _full16_recompute_backward_cuda_op(
    decay_x_real: Tensor,
    decay_x_imag: Tensor,
    gamma_x_real: Tensor,
    gamma_x_imag: Tensor,
    decay_y_real: Tensor,
    decay_y_imag: Tensor,
    gamma_y_real: Tensor,
    gamma_y_imag: Tensor,
    excitation_real: Tensor,
    excitation_imag: Tensor,
    descriptor: Tensor,
    grad_full_real: Tensor,
    grad_full_imag: Tensor,
    grad_descriptor: Tensor,
    epsilon: float,
    gain_kind: int,
) -> ProductScanInputGradients:
    inputs = (
        decay_x_real,
        decay_x_imag,
        gamma_x_real,
        gamma_x_imag,
        decay_y_real,
        decay_y_imag,
        gamma_y_real,
        gamma_y_imag,
        excitation_real,
        excitation_imag,
    )
    return _cuda_recompute_backward_impl(
        inputs,
        descriptor,
        grad_full_real,
        grad_full_imag,
        grad_descriptor,
        emit_coarse=True,
        full_coarse=True,
        epsilon=epsilon,
        gain_kind=gain_kind,
    )


@triton_op("lnet::pac_product_scan_pipeline_path_collapse_recompute_backward", mutates_args={})
def _path_collapse_recompute_backward_cuda_op(
    decay_x_real: Tensor,
    decay_x_imag: Tensor,
    gamma_x_real: Tensor,
    gamma_x_imag: Tensor,
    decay_y_real: Tensor,
    decay_y_imag: Tensor,
    gamma_y_real: Tensor,
    gamma_y_imag: Tensor,
    excitation_real: Tensor,
    excitation_imag: Tensor,
    path_input_weight: Tensor,
    path_input_bias: Tensor,
    path_output_weight: Tensor,
    path_output_bias: Tensor,
    grad_collapsed_real: Tensor,
    grad_collapsed_imag: Tensor,
    epsilon: float,
    gain_kind: int,
    path_swiglu: bool,  # noqa: FBT001
) -> tuple[
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
]:
    pole_x = decay_x_real, decay_x_imag, gamma_x_real, gamma_x_imag
    pole_y = decay_y_real, decay_y_imag, gamma_y_real, gamma_y_imag
    source = excitation_real, excitation_imag
    path = path_input_weight, path_input_bias, path_output_weight, path_output_bias
    horizontal = _horizontal_forward_op(*pole_x, *source)
    variance_x, variance_y, global_inverse_gain = static_product_scan_auxiliary(
        pole_x,
        pole_y,
        excitation_real,
        epsilon=epsilon,
        gain_kind=gain_kind,
    )
    empty_descriptor_gradient = excitation_real.new_empty((0,))
    vertical_gradients = _product_scan_path_collapse_backward_op(
        *pole_y,
        *horizontal,
        variance_x,
        variance_y,
        global_inverse_gain,
        *path,
        grad_collapsed_real,
        grad_collapsed_imag,
        empty_descriptor_gradient,
        epsilon,
        gain_kind,
        path_swiglu,
    )
    horizontal_gradients = _horizontal_recomputed_backward_op(
        *pole_x,
        *source,
        *vertical_gradients[4:8],
    )
    return (
        *horizontal_gradients[:4],
        *vertical_gradients[:4],
        *horizontal_gradients[4:],
        *vertical_gradients[8:12],
    )


@triton_op("lnet::pac_product_scan_pipeline_descriptor_recompute_backward", mutates_args={})
def _descriptor_recompute_backward_cuda_op(
    decay_x_real: Tensor,
    decay_x_imag: Tensor,
    gamma_x_real: Tensor,
    gamma_x_imag: Tensor,
    decay_y_real: Tensor,
    decay_y_imag: Tensor,
    gamma_y_real: Tensor,
    gamma_y_imag: Tensor,
    excitation_real: Tensor,
    excitation_imag: Tensor,
    descriptor: Tensor,
    grad_descriptor: Tensor,
    epsilon: float,
    gain_kind: int,
) -> ProductScanInputGradients:
    inputs = (
        decay_x_real,
        decay_x_imag,
        gamma_x_real,
        gamma_x_imag,
        decay_y_real,
        decay_y_imag,
        gamma_y_real,
        gamma_y_imag,
        excitation_real,
        excitation_imag,
    )
    empty = excitation_real.new_empty((0,))
    return _cuda_recompute_backward_impl(
        inputs,
        descriptor,
        empty,
        empty,
        grad_descriptor,
        emit_coarse=False,
        full_coarse=False,
        epsilon=epsilon,
        gain_kind=gain_kind,
    )


def _coarse_recompute_backward_impl(
    ctx: _AutogradContext,
    grad_coarse_real: Tensor | None,
    grad_coarse_imag: Tensor | None,
    grad_descriptor: Tensor | None,
    *,
    full_coarse: bool,
) -> tuple[Tensor | None, ...]:
    source_real = ctx.saved_tensors[8]
    batch, height, width, modes = source_real.shape
    coarse_shape = (
        (batch, height // 2, width // 2, 4, 4, modes)
        if full_coarse
        else (batch, height // 2, width // 2, 4, modes)
    )
    coarse_real = (
        source_real.new_zeros(coarse_shape) if grad_coarse_real is None else grad_coarse_real
    )
    coarse_imag = (
        source_real.new_zeros(coarse_shape) if grad_coarse_imag is None else grad_coarse_imag
    )
    descriptor = ctx.saved_tensors[-1]
    descriptor_grad = torch.zeros_like(descriptor) if grad_descriptor is None else grad_descriptor
    if not source_real.is_cuda:
        return _reference_recompute_backward(
            ctx,
            (coarse_real, coarse_imag, descriptor_grad),
            epilogue="full16" if full_coarse else "coarse",
        )
    inputs = ctx.saved_tensors
    backward = (
        _full16_recompute_backward_cuda_op if full_coarse else _coarse_recompute_backward_cuda_op
    )
    gradients = backward(
        inputs[0],
        inputs[1],
        inputs[2],
        inputs[3],
        inputs[4],
        inputs[5],
        inputs[6],
        inputs[7],
        inputs[8],
        inputs[9],
        descriptor,
        coarse_real,
        coarse_imag,
        descriptor_grad,
        ctx.epsilon,
        ctx.gain_kind,
    )
    return *gradients, None, None


def _coarse_recompute_backward(
    ctx: _AutogradContext,
    grad_coarse_real: Tensor | None,
    grad_coarse_imag: Tensor | None,
    grad_descriptor: Tensor | None,
) -> tuple[Tensor | None, ...]:
    return _coarse_recompute_backward_impl(
        ctx,
        grad_coarse_real,
        grad_coarse_imag,
        grad_descriptor,
        full_coarse=False,
    )


def _full16_recompute_backward(
    ctx: _AutogradContext,
    grad_full_real: Tensor | None,
    grad_full_imag: Tensor | None,
    grad_descriptor: Tensor | None,
) -> tuple[Tensor | None, ...]:
    return _coarse_recompute_backward_impl(
        ctx,
        grad_full_real,
        grad_full_imag,
        grad_descriptor,
        full_coarse=True,
    )


def _path_collapse_recompute_backward(
    ctx: _AutogradContext,
    grad_collapsed_real: Tensor | None,
    grad_collapsed_imag: Tensor | None,
    grad_descriptor: Tensor | None,
) -> tuple[Tensor | None, ...]:
    del grad_descriptor
    source_real = ctx.saved_tensors[8]
    batch, height, width, modes = source_real.shape
    output_shape = (batch, height, width, 1, modes)
    collapsed_real = (
        source_real.new_zeros(output_shape)
        if grad_collapsed_real is None
        else grad_collapsed_real.contiguous()
    )
    collapsed_imag = (
        source_real.new_zeros(output_shape)
        if grad_collapsed_imag is None
        else grad_collapsed_imag.contiguous()
    )
    if not source_real.is_cuda:
        raise RuntimeError("the fused scan-path recompute boundary requires CUDA")
    inputs = ctx.saved_tensors[:-1]
    backward = cast(
        "Callable[..., PathCollapsePipelineGradients]",
        _path_collapse_recompute_backward_cuda_op,
    )
    gradients = backward(
        *inputs,
        collapsed_real,
        collapsed_imag,
        ctx.epsilon,
        ctx.gain_kind,
        ctx.path_swiglu,
    )
    return *gradients, None, None, None


def _descriptor_recompute_backward(
    ctx: _AutogradContext,
    grad_descriptor: Tensor | None,
) -> tuple[Tensor | None, ...]:
    source_real = ctx.saved_tensors[8]
    descriptor = ctx.saved_tensors[-1]
    descriptor_grad = torch.zeros_like(descriptor) if grad_descriptor is None else grad_descriptor
    if not source_real.is_cuda:
        return _reference_recompute_backward(
            ctx,
            (descriptor_grad,),
            epilogue="descriptor",
        )
    inputs = ctx.saved_tensors
    gradients = _descriptor_recompute_backward_cuda_op(
        inputs[0],
        inputs[1],
        inputs[2],
        inputs[3],
        inputs[4],
        inputs[5],
        inputs[6],
        inputs[7],
        inputs[8],
        inputs[9],
        descriptor,
        descriptor_grad,
        ctx.epsilon,
        ctx.gain_kind,
    )
    return *gradients, None, None


torch.library.register_autograd(
    "lnet::pac_product_scan_pipeline_coarse_recompute",
    _coarse_recompute_backward,
    setup_context=_setup_coarse_recompute_context,
)
torch.library.register_autograd(
    "lnet::pac_product_scan_pipeline_full16_recompute",
    _full16_recompute_backward,
    setup_context=_setup_coarse_recompute_context,
)
torch.library.register_autograd(
    "lnet::pac_product_scan_pipeline_path_collapse_recompute",
    _path_collapse_recompute_backward,
    setup_context=_setup_path_collapse_recompute_context,
)
torch.library.register_autograd(
    "lnet::pac_product_scan_pipeline_descriptor_recompute",
    _descriptor_recompute_backward,
    setup_context=_setup_descriptor_recompute_context,
)


@torch.compiler.disable
def _run_memory_efficient_pipeline(
    pole_x: tuple[Tensor, Tensor, Tensor, Tensor],
    pole_y: tuple[Tensor, Tensor, Tensor, Tensor],
    source: tuple[Tensor, Tensor],
    *,
    epilogue: ProductScanEpilogue,
    epsilon: float,
    gain_kind: int,
) -> ProductScanOutput:
    """Keep donation inside the scan boundary when an outer model is compiled."""
    semantic_args = epsilon, gain_kind
    if epilogue == "coarse":
        return _coarse_recompute_op(*pole_x, *pole_y, *source, *semantic_args)
    if epilogue == "full16":
        return _full16_recompute_op(*pole_x, *pole_y, *source, *semantic_args)
    return _descriptor_recompute_op(*pole_x, *pole_y, *source, *semantic_args)


@torch.compiler.disable
def _run_memory_efficient_path_collapse(
    pole_x: tuple[Tensor, Tensor, Tensor, Tensor],
    pole_y: tuple[Tensor, Tensor, Tensor, Tensor],
    source: tuple[Tensor, Tensor],
    path_collapse: PathCollapseParameters,
    *,
    epsilon: float,
    gain_kind: int,
    path_swiglu: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    """Keep the fused scan-path op opaque to outer Inductor graphs."""
    return _path_collapse_recompute_op(
        *pole_x,
        *pole_y,
        *source,
        *path_collapse,
        epsilon,
        gain_kind,
        path_swiglu,
    )


def run_product_scan_pipeline(
    pole_x: tuple[Tensor, Tensor, Tensor, Tensor],
    pole_y: tuple[Tensor, Tensor, Tensor, Tensor],
    source: tuple[Tensor, Tensor],
    *,
    epilogue: ProductScanEpilogue,
    gain_normalization: ProductGainNormalization,
    memory_policy: ScanMemoryPolicy = "retain",
    epsilon: float = DEFAULT_EPSILON,
) -> ProductScanOutput:
    """Run the optimized D4 scan, optionally replaying only its scan segment."""
    if memory_policy not in {"retain", "recompute"}:
        message = f"unsupported scan memory policy: {memory_policy}"
        raise ValueError(message)
    should_recompute = (
        memory_policy == "recompute"
        and torch.is_grad_enabled()
        and any(value.requires_grad for value in (*pole_x, *pole_y, *source))
    )
    if not should_recompute:
        return _product_scan_pipeline(
            *pole_x,
            *pole_y,
            *source,
            epilogue=epilogue,
            gain_normalization=gain_normalization,
            epsilon=epsilon,
        )
    memory_efficient_pipeline = cast(
        "Callable[..., ProductScanOutput]",
        _run_memory_efficient_pipeline,
    )
    return memory_efficient_pipeline(
        pole_x,
        pole_y,
        source,
        epilogue=epilogue,
        epsilon=epsilon,
        gain_kind=gain_kind(gain_normalization),
    )


def run_product_scan_path_collapse_pipeline(
    pole_x: tuple[Tensor, Tensor, Tensor, Tensor],
    pole_y: tuple[Tensor, Tensor, Tensor, Tensor],
    source: tuple[Tensor, Tensor],
    path_collapse: PathCollapseParameters,
    *,
    gain_normalization: ProductGainNormalization,
    memory_policy: ScanMemoryPolicy = "recompute",
    path_swiglu: bool = False,
    epsilon: float = DEFAULT_EPSILON,
) -> tuple[Tensor, Tensor]:
    """Fuse full-resolution D4 scan and per-mode path collapse."""
    if memory_policy not in {"retain", "recompute"}:
        message = f"unsupported scan memory policy: {memory_policy}"
        raise ValueError(message)
    if not source[0].is_cuda:
        full = run_product_scan_pipeline(
            pole_x,
            pole_y,
            source,
            epilogue="full16",
            gain_normalization=gain_normalization,
            memory_policy="retain",
            epsilon=epsilon,
        )
        if not isinstance(full, tuple):
            raise TypeError("full16 reference did not return directional cells")
        collapse_reference = (
            d4_grouped_cell_path_swiglu_reference
            if path_swiglu
            else d4_grouped_cell_path_collapse_reference
        )
        return collapse_reference(full[0], full[1], *path_collapse)

    semantic_args = epsilon, gain_kind(gain_normalization)
    should_recompute = (
        memory_policy == "recompute"
        and torch.is_grad_enabled()
        and any(value.requires_grad for value in (*pole_x, *pole_y, *source, *path_collapse))
    )
    if should_recompute:
        memory_efficient_path_collapse = cast(
            "Callable[..., tuple[Tensor, Tensor, Tensor]]",
            _run_memory_efficient_path_collapse,
        )
        output = memory_efficient_path_collapse(
            pole_x,
            pole_y,
            source,
            path_collapse,
            epsilon=semantic_args[0],
            gain_kind=semantic_args[1],
            path_swiglu=path_swiglu,
        )
        return output[0], output[1]

    horizontal = _horizontal_forward_op(*pole_x, *source)
    variance_x, variance_y, global_inverse_gain = static_product_scan_auxiliary(
        pole_x,
        pole_y,
        source[0],
        epsilon=epsilon,
        gain_kind=semantic_args[1],
    )
    output = _product_scan_path_collapse_op(
        *pole_y,
        *horizontal,
        variance_x,
        variance_y,
        global_inverse_gain,
        *path_collapse,
        *semantic_args,
        path_swiglu,
    )
    return output[0], output[1]


__all__ = [
    "PathCollapseParameters",
    "ProductScanEpilogue",
    "ProductScanOutput",
    "ScanMemoryPolicy",
    "run_product_scan_path_collapse_pipeline",
    "run_product_scan_pipeline",
]
