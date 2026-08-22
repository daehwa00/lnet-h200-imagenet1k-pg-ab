"""Reusable complex affine layers shared by complex scan CFFNs."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional

ComplexField = tuple[Tensor, Tensor]


def packed_complex_linear_weight(weight_real: Tensor, weight_imag: Tensor) -> Tensor:
    """Return the constrained real matrix for a strict complex-linear map."""
    if weight_real.shape != weight_imag.shape or weight_real.ndim != 2:
        message = "complex-linear weights must be matching matrices"
        raise ValueError(message)
    return torch.cat(
        (
            torch.cat((weight_real, -weight_imag), dim=-1),
            torch.cat((weight_imag, weight_real), dim=-1),
        ),
        dim=0,
    )


def unit_row_complex_linear_weight(
    weight_real: Tensor,
    weight_imag: Tensor,
    *,
    epsilon: float = 1.0e-12,
) -> ComplexField:
    """Return strict-complex weights with unit energy in every output row."""
    if weight_real.shape != weight_imag.shape or weight_real.ndim != 2:
        message = "complex-linear weights must be matching matrices"
        raise ValueError(message)
    if epsilon <= 0.0:
        message = "complex-linear row-energy epsilon must be positive"
        raise ValueError(message)
    row_energy = (
        weight_real.float()
        .square()
        .add(weight_imag.float().square())
        .sum(
            dim=-1,
            keepdim=True,
        )
    )
    inverse_norm = torch.rsqrt(row_energy.clamp_min(epsilon)).to(dtype=weight_real.dtype)
    return weight_real * inverse_norm, weight_imag * inverse_norm


def packed_widely_linear_weight(
    weight_real: Tensor,
    weight_imag: Tensor,
    conjugate_real: Tensor,
    conjugate_imag: Tensor,
) -> Tensor:
    """Return the equivalent real-linear matrix for packed Cartesian input."""
    return torch.cat(
        (
            torch.cat(
                (weight_real + conjugate_real, conjugate_imag - weight_imag),
                dim=-1,
            ),
            torch.cat(
                (weight_imag + conjugate_imag, weight_real - conjugate_real),
                dim=-1,
            ),
        ),
        dim=0,
    )


def packed_widely_linear_bias(
    bias_real: Tensor | None,
    bias_imag: Tensor | None,
) -> Tensor | None:
    """Return the Cartesian bias matching a packed widely-linear weight."""
    if bias_real is None or bias_imag is None:
        return None
    return torch.cat((bias_real, bias_imag))


class ComplexLinear(nn.Module):
    """Bias-free complex linear map represented by two real matrices."""

    def __init__(self, input_modes: int, output_modes: int) -> None:
        super().__init__()
        if input_modes <= 0 or output_modes <= 0:
            message = "complex linear dimensions must be positive"
            raise ValueError(message)
        self.input_modes = input_modes
        self.output_modes = output_modes
        self.weight_real = nn.Parameter(torch.empty(output_modes, input_modes))
        self.weight_imag = nn.Parameter(torch.empty(output_modes, input_modes))
        nn.init.xavier_uniform_(self.weight_real)
        nn.init.xavier_uniform_(self.weight_imag)
        with torch.no_grad():
            self.weight_real.mul_(math.sqrt(0.5))
            self.weight_imag.mul_(math.sqrt(0.5))

    def packed_weight(self) -> Tensor:
        """Return the real matrix mapping ``[real | imag]`` rows to packed rows."""
        return packed_complex_linear_weight(self.weight_real, self.weight_imag)

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        if real.shape != imag.shape or real.shape[-1] != self.input_modes:
            message = "complex linear inputs have incompatible shapes"
            raise ValueError(message)
        return (
            functional.linear(real, self.weight_real) - functional.linear(imag, self.weight_imag),
            functional.linear(real, self.weight_imag) + functional.linear(imag, self.weight_real),
        )


class PackedComplexLinear(ComplexLinear):
    """Evaluate a strict complex map as one packed real GEMM."""

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        if real.shape != imag.shape or real.shape[-1] != self.input_modes:
            message = "packed complex-linear inputs have incompatible shapes"
            raise ValueError(message)
        packed_input = torch.cat((real, imag), dim=-1)
        packed_output = functional.linear(packed_input, self.packed_weight())
        output_real, output_imag = packed_output.split(self.output_modes, dim=-1)
        return output_real, output_imag


class ScaleStableComplexLinear(ComplexLinear):
    """Learn a complex basis map while preserving its initialized row energy."""

    reference_row_norm: Tensor

    def __init__(self, input_modes: int, output_modes: int) -> None:
        super().__init__(input_modes, output_modes)
        reference_row_norm = torch.empty(output_modes, 1)
        self.register_buffer(
            "reference_row_norm",
            reference_row_norm,
        )
        self.reference_row_norm = reference_row_norm
        self.capture_reference_row_norm_()

    def compute_row_norm(self) -> Tensor:
        calculation_dtype = (
            torch.float64 if self.weight_real.dtype is torch.float64 else torch.float32
        )
        energy = self.weight_real.to(calculation_dtype).square() + self.weight_imag.to(
            calculation_dtype
        ).square()
        return energy.sum(dim=-1, keepdim=True).sqrt().to(self.weight_real.dtype)

    @torch.no_grad()
    def capture_reference_row_norm_(self) -> None:
        self.reference_row_norm.copy_(self.compute_row_norm())

    def normalized_weight(self) -> tuple[Tensor, Tensor]:
        current = self.compute_row_norm().clamp_min(torch.finfo(self.weight_real.dtype).tiny)
        scale = self.reference_row_norm / current
        return self.weight_real * scale, self.weight_imag * scale

    @classmethod
    def from_projection(cls, source: ComplexLinear) -> ScaleStableComplexLinear:
        reference = source.weight_real
        with torch.random.fork_rng(devices=[]):
            projection = cls(source.input_modes, source.output_modes).to(
                device=reference.device,
                dtype=reference.dtype,
            )
        with torch.no_grad():
            projection.weight_real.copy_(source.weight_real)
            projection.weight_imag.copy_(source.weight_imag)
            projection.capture_reference_row_norm_()
        return projection

    def packed_weight(self) -> Tensor:
        weight_real, weight_imag = self.normalized_weight()
        return packed_complex_linear_weight(weight_real, weight_imag)

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        if real.shape != imag.shape or real.shape[-1] != self.input_modes:
            raise ValueError("scale-stable complex linear inputs have incompatible shapes")
        weight_real, weight_imag = self.normalized_weight()
        return (
            functional.linear(real, weight_real) - functional.linear(imag, weight_imag),
            functional.linear(real, weight_imag) + functional.linear(imag, weight_real),
        )


@torch.no_grad()
def semi_orthogonal_complex_linear_(layer: ComplexLinear) -> None:
    """Initialize a strict complex map as a real semi-orthogonal projection."""
    nn.init.orthogonal_(layer.weight_real)
    layer.weight_imag.zero_()


@torch.no_grad()
def identity_complex_linear_(layer: ComplexLinear) -> None:
    """Initialize a square strict-complex map as the exact identity."""
    if layer.input_modes != layer.output_modes:
        raise ValueError("identity complex initialization requires a square map")
    layer.weight_real.copy_(
        torch.eye(
            layer.input_modes,
            device=layer.weight_real.device,
            dtype=layer.weight_real.dtype,
        )
    )
    layer.weight_imag.zero_()


class WidelyLinear(nn.Module):
    """General real-linear affine map expressed as Wz + V*conj(z)."""

    def __init__(
        self,
        input_modes: int,
        output_modes: int,
        *,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if input_modes <= 0 or output_modes <= 0:
            message = "widely-linear dimensions must be positive"
            raise ValueError(message)
        self.input_modes = input_modes
        self.output_modes = output_modes
        self.weight_real = nn.Parameter(torch.empty(output_modes, input_modes))
        self.weight_imag = nn.Parameter(torch.empty(output_modes, input_modes))
        self.conjugate_real = nn.Parameter(torch.empty(output_modes, input_modes))
        self.conjugate_imag = nn.Parameter(torch.empty(output_modes, input_modes))
        for weight in (
            self.weight_real,
            self.weight_imag,
            self.conjugate_real,
            self.conjugate_imag,
        ):
            nn.init.xavier_uniform_(weight)
            with torch.no_grad():
                weight.mul_(0.5)
        if bias:
            self.bias_real = nn.Parameter(torch.zeros(output_modes))
            self.bias_imag = nn.Parameter(torch.zeros(output_modes))
        else:
            self.register_parameter("bias_real", None)
            self.register_parameter("bias_imag", None)

    def load_real_affine(self, weight: Tensor, bias: Tensor | None = None) -> None:
        """Load an arbitrary R^(2I)->R^(2O) affine map exactly."""
        expected = (2 * self.output_modes, 2 * self.input_modes)
        if tuple(weight.shape) != expected:
            message = f"real affine weight must have shape {expected}"
            raise ValueError(message)
        output_modes, input_modes = self.output_modes, self.input_modes
        top_left = weight[:output_modes, :input_modes]
        top_right = weight[:output_modes, input_modes:]
        bottom_left = weight[output_modes:, :input_modes]
        bottom_right = weight[output_modes:, input_modes:]
        with torch.no_grad():
            self.weight_real.copy_(0.5 * (top_left + bottom_right))
            self.conjugate_real.copy_(0.5 * (top_left - bottom_right))
            self.weight_imag.copy_(0.5 * (bottom_left - top_right))
            self.conjugate_imag.copy_(0.5 * (bottom_left + top_right))
            if bias is not None:
                if self.bias_real is None or self.bias_imag is None:
                    message = "cannot load a bias into a bias-free widely-linear layer"
                    raise ValueError(message)
                if tuple(bias.shape) != (2 * output_modes,):
                    message = "real affine bias has incompatible shape"
                    raise ValueError(message)
                self.bias_real.copy_(bias[:output_modes])
                self.bias_imag.copy_(bias[output_modes:])

    def packed_weight(self) -> Tensor:
        """Return the real matrix mapping ``[real | imag]`` rows to packed rows."""
        return packed_widely_linear_weight(
            self.weight_real,
            self.weight_imag,
            self.conjugate_real,
            self.conjugate_imag,
        )

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        if real.shape != imag.shape or real.shape[-1] != self.input_modes:
            message = "widely-linear inputs have incompatible shapes"
            raise ValueError(message)
        # Wz + V*conj(z) is an arbitrary real-linear map. Combining its four
        # parameter blocks gives two ordinary GEMMs without packing activations.
        real_weight = torch.cat(
            (
                self.weight_real + self.conjugate_real,
                self.weight_imag + self.conjugate_imag,
            ),
            dim=0,
        )
        imag_weight = torch.cat(
            (
                self.conjugate_imag - self.weight_imag,
                self.weight_real - self.conjugate_real,
            ),
            dim=0,
        )
        bias = None
        if self.bias_real is not None and self.bias_imag is not None:
            bias = torch.cat((self.bias_real, self.bias_imag), dim=0)
        real_output = functional.linear(real, real_weight, bias)
        imag_output = functional.linear(imag, imag_weight)
        real_to_real, real_to_imag = real_output.split(self.output_modes, dim=-1)
        imag_to_real, imag_to_imag = imag_output.split(self.output_modes, dim=-1)
        return real_to_real + imag_to_real, real_to_imag + imag_to_imag


__all__ = [
    "ComplexLinear",
    "PackedComplexLinear",
    "ScaleStableComplexLinear",
    "WidelyLinear",
    "identity_complex_linear_",
    "packed_complex_linear_weight",
    "packed_widely_linear_bias",
    "packed_widely_linear_weight",
    "semi_orthogonal_complex_linear_",
    "unit_row_complex_linear_weight",
]
