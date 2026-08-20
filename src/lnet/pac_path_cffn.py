"""Path-axis and mode-axis complex FFN combiners."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn

from .pac_complex_ffn import ComplexFFN, ComplexFFNActivation
from .pac_complex_layers import WidelyLinear

ComplexField = tuple[Tensor, Tensor]
_QUADRANT_PATH_COUNT = 4


def _validate_packed_path_field(
    source_real: Tensor,
    source_imag: Tensor,
    *,
    path_count: int,
    modes: int,
    owner: str,
) -> None:
    if (
        source_real.shape != source_imag.shape
        or source_real.ndim < 2
        or tuple(source_real.shape[-2:]) != (path_count, modes)
    ):
        message = f"packed {owner} path/mode CFFN inputs have incompatible shapes"
        raise ValueError(message)


class D4PathModeCombiner(ComplexFFN, ABC):
    """Common module contract for packed D4 path/mode transformations."""

    requires_full_product_cells = False

    @abstractmethod
    def forward_packed(self, source_real: Tensor, source_imag: Tensor) -> ComplexField: ...

    def forward_full_state(
        self,
        source_real: Tensor,
        source_imag: Tensor,
        *,
        pole_x: tuple[Tensor, Tensor, Tensor, Tensor],
        pole_y: tuple[Tensor, Tensor, Tensor, Tensor],
    ) -> ComplexField:
        """Consume direction-relative 2x2 cells when explicitly requested."""
        del source_real, source_imag, pole_x, pole_y
        message = f"{type(self).__name__} does not consume full product cells"
        raise RuntimeError(message)


class FactorizedQuadrantPathModeCFFNCombiner(D4PathModeCombiner):
    """Mix four product paths along their mode and path axes."""

    def __init__(
        self,
        modes: int,
        mode_hidden: int,
        path_hidden: int,
        *,
        layer_scale_initial: float = 1.0e-3,
        mode_activation: ComplexFFNActivation = "cartesian_silu",
        path_activation: ComplexFFNActivation = "cartesian_silu",
    ) -> None:
        super().__init__()
        if min(modes, mode_hidden, path_hidden) <= 0:
            message = "quadrant path/mode CFFN dimensions must be positive"
            raise ValueError(message)
        if layer_scale_initial <= 0.0:
            message = "quadrant path/mode CFFN LayerScale must be positive"
            raise ValueError(message)
        self.modes = modes
        self.path_count = _QUADRANT_PATH_COUNT
        self.output_paths = _QUADRANT_PATH_COUNT
        self.input_modes = _QUADRANT_PATH_COUNT * modes
        self.mode_activation: ComplexFFNActivation = mode_activation
        self.path_activation: ComplexFFNActivation = path_activation
        self.mode_input = WidelyLinear(
            modes,
            mode_hidden,
            bias=True,
        )
        self.mode_output = WidelyLinear(
            mode_hidden,
            modes,
            bias=True,
        )
        self.path_input = WidelyLinear(
            _QUADRANT_PATH_COUNT,
            path_hidden,
            bias=True,
        )
        self.path_output = WidelyLinear(
            path_hidden,
            _QUADRANT_PATH_COUNT,
            bias=True,
        )
        self.mode_layer_scale = nn.Parameter(torch.full((modes,), layer_scale_initial))
        self.path_layer_scale = nn.Parameter(
            torch.full((_QUADRANT_PATH_COUNT,), layer_scale_initial)
        )
        self.path_synthesis_real = nn.Parameter(
            torch.zeros(modes, _QUADRANT_PATH_COUNT, _QUADRANT_PATH_COUNT)
        )
        self.path_synthesis_imag = nn.Parameter(
            torch.zeros(modes, _QUADRANT_PATH_COUNT, _QUADRANT_PATH_COUNT)
        )
        self.mode_activation_scale: Tensor | None
        self.path_activation_scale: Tensor | None
        self.register_optional_persistent_buffer("mode_activation_scale")
        self.register_optional_persistent_buffer("path_activation_scale")
        # Start from identity synthesis across the four D4 product paths.
        with torch.no_grad():
            for quadrant in range(_QUADRANT_PATH_COUNT):
                self.path_synthesis_real[:, quadrant, quadrant] = 1.0

        self.identity_path_handoff = False

    def use_identity_path_handoff_(self) -> FactorizedQuadrantPathModeCFFNCombiner:
        """Pass ModeCFFN product paths through without a path-axis CFFN.

        This is valid only when every input path is retained.  Removing the
        path projections rather than merely bypassing them keeps both the
        parameter count and optimizer state faithful to the active graph.
        """
        if self.path_count != self.output_paths:
            message = "identity path handoff requires equal input and output path counts"
            raise ValueError(message)
        self.path_input = None
        self.path_output = None
        self.register_parameter("path_layer_scale", None)
        self.register_parameter("path_synthesis_real", None)
        self.register_parameter("path_synthesis_imag", None)
        self.identity_path_handoff = True
        return self

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        if real.shape != imag.shape or real.shape[-1] != self.input_modes:
            message = "quadrant path/mode CFFN inputs have incompatible shapes"
            raise ValueError(message)
        shape = (*real.shape[:-1], self.path_count, self.modes)
        return self.forward_packed(real.reshape(shape), imag.reshape(shape))

    def forward_packed(self, source_real: Tensor, source_imag: Tensor) -> ComplexField:
        """Consume an already packed ``[..., path, mode]`` coarse field."""
        _validate_packed_path_field(
            source_real,
            source_imag,
            path_count=self.path_count,
            modes=self.modes,
            owner="quadrant",
        )
        mode_real, mode_imag = self.run_cffn(
            source_real,
            source_imag,
            input_projection=self.mode_input,
            output_projection=self.mode_output,
            activation=self.mode_activation,
            activation_scale=self.mode_activation_scale,
            residual_scale=self.mode_layer_scale,
        )

        if self.identity_path_handoff:
            return mode_real, mode_imag

        if (
            self.path_input is None
            or self.path_output is None
            or self.path_layer_scale is None
            or self.path_synthesis_real is None
            or self.path_synthesis_imag is None
        ):
            message = "active path CFFN is missing parameters"
            raise RuntimeError(message)
        return self.run_cffn(
            mode_real.transpose(-2, -1),
            mode_imag.transpose(-2, -1),
            input_projection=self.path_input,
            output_projection=self.path_output,
            activation=self.path_activation,
            activation_scale=self.path_activation_scale,
            residual_scale=self.path_layer_scale,
            synthesis_real=self.path_synthesis_real,
            synthesis_imag=self.path_synthesis_imag,
        )


__all__ = [
    "D4PathModeCombiner",
    "FactorizedQuadrantPathModeCFFNCombiner",
]
