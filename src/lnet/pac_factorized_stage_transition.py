"""Factorized mode-path-mode transition for product-only complex stages."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .complex_scan_transitions import ComplexRMSNorm
from .pac_complex_ffn import ComplexFFN
from .pac_complex_layers import WidelyLinear
from .pac_grouped_path_cffn import GroupedWidelyLinear, grouped_cartesian_cffn
from .pac_path_cffn import D4PathModeCombiner

ComplexField = tuple[Tensor, Tensor]


class ModeResidualPathCollapse(D4PathModeCombiner):
    """Mix modes per path, then collapse four paths nonlinearly to one."""

    collapses_product_paths = True

    def __init__(
        self,
        modes: int,
        *,
        mode_hidden: int,
        path_hidden: int = 8,
    ) -> None:
        super().__init__()
        if min(modes, mode_hidden, path_hidden) <= 0:
            message = "factorized stage mixer dimensions must be positive"
            raise ValueError(message)
        self.modes = modes
        self.path_count = 4
        self.output_paths = 1
        self.input_modes = 4 * modes
        self.mode_norm = ComplexRMSNorm(modes)
        self.mode_input = WidelyLinear(modes, mode_hidden, bias=True)
        self.mode_output = WidelyLinear(mode_hidden, modes, bias=True)
        self.path_input = GroupedWidelyLinear(
            modes,
            4,
            path_hidden,
            bias=True,
        )
        self.path_output = GroupedWidelyLinear(
            modes,
            path_hidden,
            1,
            bias=True,
        )

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        if real.shape != imag.shape or real.shape[-1] != self.input_modes:
            message = "factorized stage mixer inputs have incompatible shapes"
            raise ValueError(message)
        shape = (*real.shape[:-1], self.path_count, self.modes)
        return self.forward_packed(real.reshape(shape), imag.reshape(shape))

    def forward_packed(self, source_real: Tensor, source_imag: Tensor) -> ComplexField:
        expected = (self.path_count, self.modes)
        if (
            source_real.shape != source_imag.shape
            or source_real.ndim != 5
            or tuple(source_real.shape[-2:]) != expected
        ):
            message = "factorized stage mixer requires NHW-path-mode inputs"
            raise ValueError(message)
        unit_real, unit_imag = self.mode_norm(source_real, source_imag)
        mixed_real, mixed_imag = self.run_cffn(
            unit_real,
            unit_imag,
            input_projection=self.mode_input,
            output_projection=self.mode_output,
            activation="cartesian_silu",
            residual_scale=unit_real.new_ones(()),
            residual_source=(source_real, source_imag),
        )
        return grouped_cartesian_cffn(
            mixed_real,
            mixed_imag,
            input_projection=self.path_input,
            output_projection=self.path_output,
        )


class FactorizedS2DPostFusionTransition(ComplexFFN):
    """Merge collapsed pole state with S2D carry, then mix modes residually."""

    def __init__(self, modes: int, *, post_hidden: int) -> None:
        super().__init__()
        if min(modes, post_hidden) <= 0:
            message = "factorized post-fusion dimensions must be positive"
            raise ValueError(message)
        self.modes = modes
        self.input_modes = modes
        self.output_modes = modes
        self.carry_input_modes = 4 * modes
        self.carry_weight = nn.Parameter(torch.full((modes, 4), 0.25))
        self.post_norm = ComplexRMSNorm(modes)
        self.post_input = WidelyLinear(modes, post_hidden, bias=True)
        self.post_output = WidelyLinear(post_hidden, modes, bias=True)

    def _carry(self, real: Tensor, imag: Tensor) -> ComplexField:
        if real.shape != imag.shape or real.shape[-1] != self.carry_input_modes:
            message = "factorized transition S2D carry has incompatible shape"
            raise ValueError(message)
        shape = (*real.shape[:-1], 4, self.modes)
        weight = self.carry_weight.transpose(0, 1)
        return (
            (real.reshape(shape) * weight).sum(dim=-2),
            (imag.reshape(shape) * weight).sum(dim=-2),
        )

    def forward(
        self,
        real: Tensor,
        imag: Tensor,
        carry_real: Tensor | None = None,
        carry_imag: Tensor | None = None,
    ) -> ComplexField:
        if real.shape != imag.shape or real.shape[-1] != self.modes:
            message = "factorized transition pole state has incompatible shape"
            raise ValueError(message)
        if carry_real is None or carry_imag is None:
            message = "factorized transition requires S2D carry coordinates"
            raise ValueError(message)
        carry_state_real, carry_state_imag = self._carry(carry_real, carry_imag)
        merged_real = carry_state_real + real
        merged_imag = carry_state_imag + imag
        unit_real, unit_imag = self.post_norm(merged_real, merged_imag)
        return self.run_cffn(
            unit_real,
            unit_imag,
            input_projection=self.post_input,
            output_projection=self.post_output,
            activation="cartesian_silu",
            residual_scale=unit_real.new_ones(()),
            residual_source=(merged_real, merged_imag),
        )


__all__ = [
    "FactorizedS2DPostFusionTransition",
    "ModeResidualPathCollapse",
]
