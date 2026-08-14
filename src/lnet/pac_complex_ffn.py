"""Shared complex feed-forward network math.

All project CFFN modules keep their parameters and checkpoint names while
delegating projection, activation, residual, and synthesis semantics here.
``torch.compile`` sees one stable graph and owns kernel selection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, cast, override, runtime_checkable

import torch
from torch import Tensor, nn
from torch.nn import functional
from torch.nn.modules import module as nn_module

from .pac_complex_layers import (
    WidelyLinear,
    packed_widely_linear_bias,
    packed_widely_linear_weight,
)

if TYPE_CHECKING:
    from collections.abc import Callable

ComplexField = tuple[Tensor, Tensor]
ComplexFFNActivation = Literal[
    "cartesian_silu",
    "modsilu",
    "modrelu",
    "centered_magnitude",
    "identity_centered_magnitude",
]


@runtime_checkable
class ComplexAffine(Protocol):
    """Structural interface required by the ComplexFFN dispatcher."""

    input_modes: int
    output_modes: int

    @property
    def weight_real(self) -> Tensor: ...

    @property
    def weight_imag(self) -> Tensor: ...

    @property
    def conjugate_real(self) -> Tensor: ...

    @property
    def conjugate_imag(self) -> Tensor: ...

    @property
    def bias_real(self) -> Tensor | None: ...

    @property
    def bias_imag(self) -> Tensor | None: ...

    def __call__(self, real: Tensor, imag: Tensor) -> ComplexField: ...


_MODULE_CALL_HOOK_FIELDS = (
    "_forward_pre_hooks",
    "_forward_hooks",
    "_backward_pre_hooks",
    "_backward_hooks",
)
_GLOBAL_MODULE_CALL_HOOK_FIELDS = (
    "_global_forward_pre_hooks",
    "_global_forward_hooks",
    "_global_backward_pre_hooks",
    "_global_backward_hooks",
)


def module_calls_are_transparent(*modules: nn.Module) -> bool:
    global_hooks = vars(nn_module)
    if any(global_hooks.get(name) for name in _GLOBAL_MODULE_CALL_HOOK_FIELDS):
        return False
    return all(
        not any(vars(module).get(name) for name in _MODULE_CALL_HOOK_FIELDS) for module in modules
    )


def _packed_weight(projection: WidelyLinear) -> Tensor:
    return packed_widely_linear_weight(
        projection.weight_real,
        projection.weight_imag,
        projection.conjugate_real,
        projection.conjugate_imag,
    )


def _packed_bias(projection: WidelyLinear) -> Tensor | None:
    return packed_widely_linear_bias(projection.bias_real, projection.bias_imag)


def _packed_synthesis_weight(
    synthesis_real: Tensor,
    synthesis_imag: Tensor,
) -> Tensor:
    """Return the real-linear form of a mode-wise complex synthesis map."""
    return torch.cat(
        (
            torch.cat((synthesis_real, -synthesis_imag), dim=-1),
            torch.cat((synthesis_imag, synthesis_real), dim=-1),
        ),
        dim=-2,
    )


def packed_cartesian_cffn(
    source: Tensor,
    *,
    input_projection: WidelyLinear,
    output_projection: WidelyLinear,
    residual_scale: Tensor | None = None,
    residual_source: Tensor | None = None,
    synthesis_real: Tensor | None = None,
    synthesis_imag: Tensor | None = None,
) -> Tensor:
    """Evaluate a Cartesian-SiLU CFFN without splitting complex coordinates.

    The last dimension always stores all real coordinates followed by all
    imaginary coordinates.  Widths come exclusively from the projections and
    synthesis tensors, so callers can reuse this graph for arbitrary complex
    feature sizes and leading token layouts.
    """
    input_modes = input_projection.input_modes
    output_modes = output_projection.output_modes
    if source.shape[-1] != 2 * input_modes:
        message = "packed CFFN source width is incompatible"
        raise ValueError(message)
    if input_projection.output_modes != output_projection.input_modes:
        message = "packed CFFN hidden widths are incompatible"
        raise ValueError(message)
    if (synthesis_real is None) != (synthesis_imag is None):
        message = "packed CFFN synthesis requires both coordinates"
        raise ValueError(message)

    hidden = functional.silu(
        functional.linear(
            source,
            _packed_weight(input_projection),
            _packed_bias(input_projection),
        )
    )
    output = functional.linear(
        hidden,
        _packed_weight(output_projection),
        _packed_bias(output_projection),
    )
    if residual_scale is not None:
        anchor = source if residual_source is None else residual_source
        if anchor.shape[-1] != 2 * output_modes:
            message = "packed CFFN residual width is incompatible"
            raise ValueError(message)
        scale = residual_scale.to(dtype=anchor.dtype)
        if scale.ndim != 0:
            if tuple(scale.shape) != (output_modes,):
                message = "packed CFFN residual scale is incompatible"
                raise ValueError(message)
            scale = torch.cat((scale, scale))
        output = anchor + scale * output

    if synthesis_real is None or synthesis_imag is None:
        return output
    if (
        synthesis_real.shape != synthesis_imag.shape
        or synthesis_real.ndim != 3
        or synthesis_real.shape[-1] != output_modes
        or output.ndim < 2
        or output.shape[-2] != synthesis_real.shape[0]
    ):
        message = "packed CFFN synthesis dimensions are incompatible"
        raise ValueError(message)
    return torch.einsum(
        "...mp,mqp->...mq",
        output,
        _packed_synthesis_weight(synthesis_real, synthesis_imag),
    )


def _can_fuse_compiled_cartesian_cffn(
    input_projection: ComplexAffine,
    output_projection: ComplexAffine,
    *,
    activation: ComplexFFNActivation,
    activation_scale: Tensor | float | None,
    synthesis_real: Tensor | None,
    synthesis_imag: Tensor | None,
    hidden_addition: ComplexField | None,
    hidden_transform: Callable[[Tensor, Tensor], ComplexField] | None,
) -> bool:
    return (
        torch.compiler.is_compiling()
        and type(input_projection) is WidelyLinear
        and type(output_projection) is WidelyLinear
        and activation == "cartesian_silu"
        and activation_scale is None
        and ((synthesis_real is None) == (synthesis_imag is None))
        and hidden_addition is None
        and hidden_transform is None
        and module_calls_are_transparent(input_projection, output_projection)
    )


def _packed_cartesian_cffn(
    real: Tensor,
    imag: Tensor,
    *,
    input_projection: WidelyLinear,
    output_projection: WidelyLinear,
    residual_scale: Tensor | None,
    residual_source: ComplexField | None,
    synthesis_real: Tensor | None,
    synthesis_imag: Tensor | None,
) -> ComplexField:
    source = torch.cat((real, imag), dim=-1)
    output = packed_cartesian_cffn(
        source,
        input_projection=input_projection,
        output_projection=output_projection,
        residual_scale=residual_scale,
        residual_source=(
            None if residual_source is None else torch.cat(residual_source, dim=-1)
        ),
        synthesis_real=synthesis_real,
        synthesis_imag=synthesis_imag,
    )
    if synthesis_real is not None:
        output_paths = synthesis_real.shape[-2]
        output_real, output_imag = output.split(output_paths, dim=-1)
        return (
            output_real.transpose(-2, -1).contiguous(),
            output_imag.transpose(-2, -1).contiguous(),
        )
    output_real, output_imag = output.split(output_projection.output_modes, dim=-1)
    return output_real, output_imag


def _validate_residual_contract(
    real: Tensor,
    *,
    output_modes: int,
    residual_scale: Tensor | None,
    residual_source: ComplexField | None,
) -> None:
    if residual_scale is None:
        return
    expected_shape = (*real.shape[:-1], output_modes)
    anchor_shape = real.shape if residual_source is None else residual_source[0].shape
    coordinates_match = residual_source is None or (
        residual_source[0].shape == residual_source[1].shape
    )
    if (
        tuple(anchor_shape) != expected_shape
        or not coordinates_match
        or tuple(residual_scale.shape) not in {(), (output_modes,)}
    ):
        message = "ComplexFFN residual requires matching input/output widths"
        raise ValueError(message)


def _activate(
    real: Tensor,
    imag: Tensor,
    *,
    activation: ComplexFFNActivation,
    activation_scale: Tensor | float | None = None,
    modrelu_bias: Tensor | None,
    centered_log_sharpness: Tensor | None,
    centered_threshold: Tensor | None,
) -> ComplexField:
    if activation == "cartesian_silu":
        activated_real = functional.silu(real)
        activated_imag = functional.silu(imag)
        if activation_scale is None:
            return activated_real, activated_imag
        scale = torch.as_tensor(
            activation_scale,
            dtype=real.dtype,
            device=real.device,
        )
        return scale * activated_real, scale * activated_imag

    magnitude = torch.sqrt(real.float().square() + imag.float().square() + 1.0e-8)
    if activation == "modsilu":
        gain = torch.sigmoid(magnitude)
    elif activation == "modrelu":
        if modrelu_bias is None:
            message = "modReLU ComplexFFN requires a learned bias"
            raise ValueError(message)
        gain = functional.relu(magnitude + modrelu_bias) / magnitude.clamp_min(1.0e-4)
    elif activation == "centered_magnitude":
        if centered_log_sharpness is None or centered_threshold is None:
            message = "centered ComplexFFN requires sharpness and threshold"
            raise ValueError(message)
        sharpness = functional.softplus(centered_log_sharpness)
        gain = torch.sigmoid(sharpness * (magnitude - centered_threshold))
    elif activation == "identity_centered_magnitude":
        # Preserve the complex direction while making the unit-magnitude
        # operating point an identity: 2 * sigmoid(1 - 1) == 1.
        gain = 2.0 * torch.sigmoid(magnitude - 1.0)
    else:
        message = f"unsupported ComplexFFN activation: {activation}"
        raise ValueError(message)
    typed_gain = gain.to(dtype=real.dtype)
    if activation_scale is not None:
        typed_gain = typed_gain * torch.as_tensor(
            activation_scale,
            dtype=real.dtype,
            device=real.device,
        )
    return typed_gain * real, typed_gain * imag


def _synthesize(
    real: Tensor,
    imag: Tensor,
    synthesis_real: Tensor,
    synthesis_imag: Tensor,
) -> ComplexField:
    if real.ndim < 2 or real.shape[-2] != synthesis_real.shape[0]:
        message = "ComplexFFN synthesis mode dimension is incompatible"
        raise ValueError(message)
    if synthesis_real.shape != synthesis_imag.shape:
        message = "ComplexFFN synthesis coordinates must have matching shapes"
        raise ValueError(message)
    if synthesis_real.ndim != 3 or synthesis_real.shape[-1] != real.shape[-1]:
        message = "ComplexFFN synthesis path dimension is incompatible"
        raise ValueError(message)
    return (
        torch.einsum("...mp,mqp->...qm", real, synthesis_real)
        - torch.einsum("...mp,mqp->...qm", imag, synthesis_imag),
        torch.einsum("...mp,mqp->...qm", real, synthesis_imag)
        + torch.einsum("...mp,mqp->...qm", imag, synthesis_real),
    )


def _projection_cffn(
    real: Tensor,
    imag: Tensor,
    *,
    input_projection: ComplexAffine,
    output_projection: ComplexAffine,
    activation: ComplexFFNActivation,
    activation_scale: Tensor | float | None,
    residual_scale: Tensor | None,
    synthesis_real: Tensor | None,
    synthesis_imag: Tensor | None,
    hidden_addition: ComplexField | None,
    residual_source: ComplexField | None,
    modrelu_bias: Tensor | None,
    centered_log_sharpness: Tensor | None,
    centered_threshold: Tensor | None,
    hidden_transform: Callable[[Tensor, Tensor], ComplexField] | None,
) -> ComplexField:
    hidden_real, hidden_imag = input_projection(real, imag)
    if hidden_addition is not None:
        addition_real, addition_imag = hidden_addition
        if addition_real.shape != hidden_real.shape or addition_imag.shape != hidden_imag.shape:
            message = "ComplexFFN hidden addition has an incompatible shape"
            raise ValueError(message)
        hidden_real = hidden_real + addition_real
        hidden_imag = hidden_imag + addition_imag
    if hidden_transform is not None:
        hidden_real, hidden_imag = hidden_transform(hidden_real, hidden_imag)
    hidden_real, hidden_imag = _activate(
        hidden_real,
        hidden_imag,
        activation=activation,
        activation_scale=activation_scale,
        modrelu_bias=modrelu_bias,
        centered_log_sharpness=centered_log_sharpness,
        centered_threshold=centered_threshold,
    )
    output_real, output_imag = output_projection(hidden_real, hidden_imag)
    if residual_scale is not None:
        anchor_real, anchor_imag = residual_source if residual_source is not None else (real, imag)
        scale = residual_scale.to(dtype=anchor_real.dtype)
        output_real = anchor_real + scale * output_real
        output_imag = anchor_imag + scale * output_imag
    if synthesis_real is not None and synthesis_imag is not None:
        return _synthesize(
            output_real,
            output_imag,
            synthesis_real,
            synthesis_imag,
        )
    return output_real, output_imag


def split_projected_complex_ffn(
    real: Tensor,
    imag: Tensor,
    *,
    joint_projection: ComplexAffine,
    output_projection: ComplexAffine,
    base_modes: int,
    residual_scale: Tensor | None,
    activation: ComplexFFNActivation = "cartesian_silu",
) -> ComplexField:
    """Evaluate a joint ``base + hidden`` projection through the common CFFN math.

    Projected transition variants intentionally produce their residual anchor
    and nonlinear hidden state with one affine map.  They cannot use the
    ordinary equal-width packed residual graph, but should still share its
    validation, activation and residual semantics instead of maintaining a
    private SiLU/projection implementation.
    """
    hidden_modes = output_projection.input_modes
    if (
        real.shape != imag.shape
        or real.shape[-1] != joint_projection.input_modes
        or joint_projection.output_modes != base_modes + hidden_modes
        or output_projection.output_modes != base_modes
    ):
        message = "split-projected ComplexFFN dimensions are incompatible"
        raise ValueError(message)
    projected_real, projected_imag = joint_projection(real, imag)
    base_real, hidden_real = projected_real.split((base_modes, hidden_modes), dim=-1)
    base_imag, hidden_imag = projected_imag.split((base_modes, hidden_modes), dim=-1)
    hidden_real, hidden_imag = _activate(
        hidden_real,
        hidden_imag,
        activation=activation,
        modrelu_bias=None,
        centered_log_sharpness=None,
        centered_threshold=None,
    )
    update_real, update_imag = output_projection(hidden_real, hidden_imag)
    if residual_scale is None:
        return base_real + update_real, base_imag + update_imag
    scale = residual_scale.to(dtype=base_real.dtype)
    return base_real + scale * update_real, base_imag + scale * update_imag


def complex_ffn(
    real: Tensor,
    imag: Tensor,
    *,
    input_projection: ComplexAffine,
    output_projection: ComplexAffine,
    activation: ComplexFFNActivation = "cartesian_silu",
    activation_scale: Tensor | float | None = None,
    residual_scale: Tensor | None = None,
    synthesis_real: Tensor | None = None,
    synthesis_imag: Tensor | None = None,
    hidden_addition: ComplexField | None = None,
    residual_source: ComplexField | None = None,
    modrelu_bias: Tensor | None = None,
    centered_log_sharpness: Tensor | None = None,
    centered_threshold: Tensor | None = None,
    hidden_transform: Callable[[Tensor, Tensor], ComplexField] | None = None,
) -> ComplexField:
    """Evaluate an arbitrary-width widely-linear ComplexFFN."""
    if real.shape != imag.shape:
        message = "ComplexFFN inputs must have matching shapes"
        raise ValueError(message)
    if real.shape[-1] != input_projection.input_modes:
        message = "ComplexFFN input width is incompatible"
        raise ValueError(message)
    if input_projection.output_modes != output_projection.input_modes:
        message = "ComplexFFN hidden widths are incompatible"
        raise ValueError(message)
    if (synthesis_real is None) != (synthesis_imag is None):
        message = "ComplexFFN synthesis requires both coordinates"
        raise ValueError(message)
    _validate_residual_contract(
        real,
        output_modes=output_projection.output_modes,
        residual_scale=residual_scale,
        residual_source=residual_source,
    )
    if _can_fuse_compiled_cartesian_cffn(
        input_projection,
        output_projection,
        activation=activation,
        activation_scale=activation_scale,
        synthesis_real=synthesis_real,
        synthesis_imag=synthesis_imag,
        hidden_addition=hidden_addition,
        hidden_transform=hidden_transform,
    ):
        return _packed_cartesian_cffn(
            real,
            imag,
            input_projection=cast("WidelyLinear", input_projection),
            output_projection=cast("WidelyLinear", output_projection),
            residual_scale=residual_scale,
            residual_source=residual_source,
            synthesis_real=synthesis_real,
            synthesis_imag=synthesis_imag,
        )
    return _projection_cffn(
        real,
        imag,
        input_projection=input_projection,
        output_projection=output_projection,
        activation=activation,
        activation_scale=activation_scale,
        residual_scale=residual_scale,
        synthesis_real=synthesis_real,
        synthesis_imag=synthesis_imag,
        hidden_addition=hidden_addition,
        residual_source=residual_source,
        modrelu_bias=modrelu_bias,
        centered_log_sharpness=centered_log_sharpness,
        centered_threshold=centered_threshold,
        hidden_transform=hidden_transform,
    )


class ComplexFFN(nn.Module):
    """Common base for project modules that implement complex FFN blocks.

    Subclasses keep ownership and naming of their projections and parameters,
    while this base supplies one implementation of the shared CFFN operation.
    Specialized FFNs can override ``forward`` without giving up the common
    module identity.
    """

    def __init__(self) -> None:
        super().__init__()
        self._optional_persistent_buffers: set[str] = set()

    def register_optional_persistent_buffer(
        self,
        name: str,
        value: Tensor | None = None,
    ) -> None:
        """Register a persistent buffer whose checkpoint key may appear later."""
        self.register_buffer(name, value, persistent=True)
        self._optional_persistent_buffers.add(name)

    @override
    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Materialize configured optional buffers before PyTorch copies them.

        PyTorch intentionally omits ``None`` buffers from ``state_dict`` and
        rejects a tensor key when the receiving buffer is still ``None``.
        ComplexFFN activation scales are optional at construction but persistent
        once calibrated, so recreate their storage on the module's device before
        delegating to the standard loader.
        """
        anchor = next(self.parameters(), None)
        for name in self._optional_persistent_buffers:
            current = self._buffers.get(name)
            value = state_dict.get(f"{prefix}{name}")
            if current is None and isinstance(value, Tensor):
                device = value.device if anchor is None else anchor.device
                dtype = (
                    anchor.dtype
                    if anchor is not None and value.is_floating_point()
                    else value.dtype
                )
                self._buffers[name] = torch.empty_like(value, device=device, dtype=dtype)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    run_cffn = staticmethod(complex_ffn)
    run_split_projected_cffn = staticmethod(split_projected_complex_ffn)


__all__ = [
    "ComplexAffine",
    "ComplexFFN",
    "ComplexFFNActivation",
    "complex_ffn",
    "module_calls_are_transparent",
    "split_projected_complex_ffn",
]
