"""Per-mode complex path projections with automatic D4 kernel dispatch."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional

from .pac_complex_ffn import module_calls_are_transparent
from .pac_d4_path_cffn import (
    d4_grouped_path_collapse,
    supports_d4_grouped_path_collapse,
)

ComplexField = tuple[Tensor, Tensor]


class GroupedWidelyLinear(nn.Module):
    """Apply an independent widely-linear path projection to every mode."""

    def __init__(
        self,
        modes: int,
        input_paths: int,
        output_paths: int,
        *,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if min(modes, input_paths, output_paths) <= 0:
            message = "grouped widely-linear dimensions must be positive"
            raise ValueError(message)
        self.modes = modes
        self.input_paths = input_paths
        self.output_paths = output_paths
        shape = (modes, output_paths, input_paths)
        self.weight_real = nn.Parameter(torch.empty(shape))
        self.weight_imag = nn.Parameter(torch.empty(shape))
        self.conjugate_real = nn.Parameter(torch.empty(shape))
        self.conjugate_imag = nn.Parameter(torch.empty(shape))
        bound = 0.5 * math.sqrt(6.0 / (input_paths + output_paths))
        for weight in (
            self.weight_real,
            self.weight_imag,
            self.conjugate_real,
            self.conjugate_imag,
        ):
            nn.init.uniform_(weight, -bound, bound)
        if bias:
            self.bias_real = nn.Parameter(torch.zeros(modes, output_paths))
            self.bias_imag = nn.Parameter(torch.zeros(modes, output_paths))
        else:
            self.register_parameter("bias_real", None)
            self.register_parameter("bias_imag", None)

    def packed_weight(self) -> Tensor:
        top = torch.cat(
            (
                self.weight_real + self.conjugate_real,
                self.conjugate_imag - self.weight_imag,
            ),
            dim=-1,
        )
        bottom = torch.cat(
            (
                self.weight_imag + self.conjugate_imag,
                self.weight_real - self.conjugate_real,
            ),
            dim=-1,
        )
        return torch.cat((top, bottom), dim=1).reshape(
            2 * self.modes * self.output_paths,
            2 * self.input_paths,
            1,
            1,
        )

    def packed_bias(self) -> Tensor | None:
        if self.bias_real is None or self.bias_imag is None:
            return None
        return torch.cat((self.bias_real, self.bias_imag), dim=-1).reshape(-1)

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        expected = (self.input_paths, self.modes)
        if (
            real.shape != imag.shape
            or real.ndim != 5
            or tuple(real.shape[-2:]) != expected
        ):
            message = "grouped widely-linear inputs must be NHW-path-mode tensors"
            raise ValueError(message)
        batch, height, width, _, _ = real.shape
        packed = torch.cat(
            (
                real.permute(0, 4, 3, 1, 2),
                imag.permute(0, 4, 3, 1, 2),
            ),
            dim=2,
        ).reshape(batch, 2 * self.modes * self.input_paths, height, width)
        output = functional.conv2d(
            packed,
            self.packed_weight(),
            self.packed_bias(),
            groups=self.modes,
        ).reshape(batch, self.modes, 2 * self.output_paths, height, width)
        output_real, output_imag = output.split(self.output_paths, dim=2)
        return (
            output_real.permute(0, 3, 4, 2, 1),
            output_imag.permute(0, 3, 4, 2, 1),
        )


def grouped_cartesian_cffn(
    real: Tensor,
    imag: Tensor,
    *,
    input_projection: GroupedWidelyLinear,
    output_projection: GroupedWidelyLinear,
) -> ComplexField:
    """Run a mode-grouped Cartesian CFFN with semantic CUDA dispatch."""
    if (
        real.shape != imag.shape
        or real.ndim != 5
        or tuple(real.shape[-2:])
        != (input_projection.input_paths, input_projection.modes)
    ):
        message = "grouped CFFN inputs must be NHW-path-mode tensors"
        raise ValueError(message)
    if (
        input_projection.modes != output_projection.modes
        or input_projection.output_paths != output_projection.input_paths
    ):
        message = "grouped CFFN projection dimensions are incompatible"
        raise ValueError(message)
    is_d4_collapse = (
        input_projection.input_paths == 4 and output_projection.output_paths == 1
    )
    projections_are_transparent = module_calls_are_transparent(
        input_projection,
        output_projection,
    )
    if real.is_cuda and is_d4_collapse:
        if not (
            type(input_projection) is GroupedWidelyLinear
            and type(output_projection) is GroupedWidelyLinear
        ):
            message = "CUDA D4 grouped path collapse requires exact grouped projections"
            raise RuntimeError(message)
        if not projections_are_transparent:
            message = "CUDA D4 grouped path collapse requires transparent projections"
            raise RuntimeError(message)
        input_bias = input_projection.packed_bias()
        output_bias = output_projection.packed_bias()
        if input_bias is not None and output_bias is not None:
            modes = input_projection.modes
            packed_hidden = 2 * input_projection.output_paths
            active_real = real.contiguous()
            active_imag = imag.contiguous()
            input_weight = input_projection.packed_weight().reshape(
                modes,
                packed_hidden,
                2 * input_projection.input_paths,
            )
            output_weight = output_projection.packed_weight().reshape(
                modes,
                2 * output_projection.output_paths,
                packed_hidden,
            )
            input_bias = input_bias.reshape(modes, packed_hidden)
            output_bias = output_bias.reshape(
                modes,
                2 * output_projection.output_paths,
            )
            if supports_d4_grouped_path_collapse(
                active_real,
                active_imag,
                input_weight,
                input_bias,
                output_weight,
                output_bias,
            ):
                return d4_grouped_path_collapse(
                    active_real,
                    active_imag,
                    input_weight,
                    input_bias,
                    output_weight,
                    output_bias,
                )
        message = (
            "CUDA D4 grouped path collapse requires the fused BF16-autocast contract; "
            "no alternate CUDA implementation is permitted"
        )
        raise RuntimeError(message)

    if not projections_are_transparent:
        hidden_real, hidden_imag = input_projection(real, imag)
        return output_projection(
            functional.silu(hidden_real),
            functional.silu(hidden_imag),
        )

    return grouped_cartesian_cffn_reference(
        real,
        imag,
        input_projection=input_projection,
        output_projection=output_projection,
    )


def grouped_cartesian_cffn_reference(
    real: Tensor,
    imag: Tensor,
    *,
    input_projection: GroupedWidelyLinear,
    output_projection: GroupedWidelyLinear,
) -> ComplexField:
    """Run the original grouped-convolution implementation without dispatch."""
    if (
        real.shape != imag.shape
        or real.ndim != 5
        or tuple(real.shape[-2:])
        != (input_projection.input_paths, input_projection.modes)
    ):
        message = "grouped CFFN inputs must be NHW-path-mode tensors"
        raise ValueError(message)
    if (
        input_projection.modes != output_projection.modes
        or input_projection.output_paths != output_projection.input_paths
    ):
        message = "grouped CFFN projection dimensions are incompatible"
        raise ValueError(message)
    batch, height, width, _, modes = real.shape
    source = torch.cat(
        (
            real.permute(0, 4, 3, 1, 2),
            imag.permute(0, 4, 3, 1, 2),
        ),
        dim=2,
    ).reshape(batch, 2 * modes * input_projection.input_paths, height, width)
    hidden = functional.silu(
        functional.conv2d(
            source,
            input_projection.packed_weight(),
            input_projection.packed_bias(),
            groups=modes,
        )
    )
    output = functional.conv2d(
        hidden,
        output_projection.packed_weight(),
        output_projection.packed_bias(),
        groups=modes,
    ).reshape(batch, modes, 2 * output_projection.output_paths, height, width)
    output_real, output_imag = output.split(output_projection.output_paths, dim=2)
    return (
        output_real.permute(0, 3, 4, 2, 1),
        output_imag.permute(0, 3, 4, 2, 1),
    )


__all__ = [
    "GroupedWidelyLinear",
    "grouped_cartesian_cffn",
    "grouped_cartesian_cffn_reference",
]
