"""Local strict-complex readers for product-scan inputs."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional

ComplexField = tuple[Tensor, Tensor]


class PackedComplexConv2dReader(nn.Module):
    """Mix a local complex patch with controlled kernel and optional token gain."""

    _identity_kernel: Tensor

    def __init__(
        self,
        input_modes: int,
        output_modes: int,
        *,
        kernel_size: int = 3,
        variance_epsilon: float = 1.0e-12,
        match_input_rms: bool = False,
    ) -> None:
        super().__init__()
        if input_modes <= 0 or output_modes <= 0:
            message = "complex scan reader dimensions must be positive"
            raise ValueError(message)
        if kernel_size <= 0 or kernel_size % 2 == 0:
            message = "complex scan reader kernel size must be a positive odd integer"
            raise ValueError(message)
        if variance_epsilon <= 0.0:
            message = "complex scan reader variance epsilon must be positive"
            raise ValueError(message)
        self.input_modes = input_modes
        self.output_modes = output_modes
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.variance_epsilon = float(variance_epsilon)
        self.match_input_rms = bool(match_input_rms)
        shape = (output_modes, input_modes, kernel_size, kernel_size)
        self.weight_real = nn.Parameter(torch.empty(shape))
        self.weight_imag = nn.Parameter(torch.empty(shape))
        identity = torch.zeros(shape)
        if input_modes == output_modes:
            indices = torch.arange(input_modes)
            identity[indices, indices, self.padding, self.padding] = 1.0
            nn.init.zeros_(self.weight_real)
            nn.init.zeros_(self.weight_imag)
        else:
            nn.init.xavier_uniform_(self.weight_real)
            nn.init.xavier_uniform_(self.weight_imag)
            with torch.no_grad():
                self.weight_real.mul_(math.sqrt(0.5))
                self.weight_imag.mul_(math.sqrt(0.5))
        self.register_buffer("_identity_kernel", identity, persistent=False)
        self._identity_kernel = identity

    def initialize_identity_(self) -> None:
        """Parameterize the full spatial convolution as identity plus zero delta."""
        if self.input_modes != self.output_modes:
            message = "identity initialization requires equal input and output modes"
            raise ValueError(message)
        with torch.no_grad():
            self.weight_real.zero_()
            self.weight_imag.zero_()

    def normalized_weight(self) -> ComplexField:
        """Return a strict-complex kernel with unit energy per output mode."""
        effective_real = self.weight_real + self._identity_kernel
        row_energy = (
            effective_real.float()
            .square()
            .add(self.weight_imag.float().square())
            .sum(dim=(1, 2, 3), keepdim=True)
        )
        inverse_rms = torch.rsqrt(row_energy.clamp_min(self.variance_epsilon)).to(
            dtype=effective_real.dtype
        )
        return effective_real * inverse_rms, self.weight_imag * inverse_rms

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        if real.shape != imag.shape or real.ndim != 4:
            message = "complex scan reader requires matching NHWM fields"
            raise ValueError(message)
        if real.shape[-1] != self.input_modes:
            message = "complex scan reader input has an incompatible mode dimension"
            raise ValueError(message)

        effective_real, effective_imag = self.normalized_weight()
        top = torch.cat((effective_real, -effective_imag), dim=1)
        bottom = torch.cat((effective_imag, effective_real), dim=1)
        kernel = torch.cat((top, bottom), dim=0)
        packed = torch.cat(
            (real.movedim(-1, 1), imag.movedim(-1, 1)),
            dim=1,
        )
        output = functional.conv2d(packed, kernel, padding=self.padding)
        if self.match_input_rms:
            input_energy = packed.float().square().sum(dim=1, keepdim=True).div(self.input_modes)
            output_energy = output.float().square().sum(dim=1, keepdim=True).div(self.output_modes)
            token_scale = torch.sqrt(
                (input_energy + self.variance_epsilon) / (output_energy + self.variance_epsilon)
            ).to(dtype=output.dtype)
            output = output * token_scale
        output_real, output_imag = output.split(self.output_modes, dim=1)
        # The scan kernels consume dense NHWM fields. Materialize that contract
        # here so their internal boundary does not hide the same copies.
        return (
            output_real.movedim(1, -1).contiguous(),
            output_imag.movedim(1, -1).contiguous(),
        )


__all__ = ["PackedComplexConv2dReader"]
