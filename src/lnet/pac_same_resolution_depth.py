"""Resolution-preserving stages for complex D4 pole backbones.

The scan block keeps every directional state at the input resolution.  Even
grids use the fused full-cell scan and invert its lossless 2x2 packing; odd
grids use the same optimized horizontal scan followed by a short differentiable
vertical recurrence.  The resulting memory is merged with an identity carry
and passed through the same gated PostFusion used by coarsening stages.
"""

from __future__ import annotations

# This module is the public residual-depth boundary around the existing compact
# pole coefficient builder; duplicating that algebra would be the worse API.
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor, nn

from .complex_scan_stage import ComplexScanStage
from .pac_complex_layers import ComplexLinear, semi_orthogonal_complex_linear_
from .pac_factorized_complex_scan_reader import FactorizedComplexConv2dReader
from .pac_gated_post_fusion import GatedComplexPostFusion
from .pac_phase_gated_transition import PathOnlyCollapse
from .pac_product_scan_contracts import DEFAULT_EPSILON, ProductGainNormalization
from .pac_product_scan_normalization import static_variance_tables
from .pac_product_scan_pipeline import (
    run_product_scan_path_collapse_pipeline,
    run_product_scan_pipeline,
)
from .pac_product_scan_reference import bidirectional_product_scan_reference
from .pac_triton_bidirectional_product_scan import (
    pac_triton_bidirectional_product_scan,
)

if TYPE_CHECKING:
    from .complex_scan_types import ComplexField


D4_PATHS = 4
PoleCoefficients = tuple[Tensor, Tensor, Tensor, Tensor]
_PHYSICAL_FROM_RELATIVE = (
    (0, 2, 1, 3),
    (2, 0, 3, 1),
    (1, 3, 0, 2),
    (3, 1, 2, 0),
)


def direction_relative_cells_to_full_resolution(cells: Tensor) -> Tensor:
    """Invert the lossless D4 2x2 cell packing used by ``full16`` scans."""
    if cells.ndim != 6 or tuple(cells.shape[-3:-1]) != (D4_PATHS, D4_PATHS):
        message = "full-cell reconstruction requires BHW-direction-local-mode tensors"
        raise ValueError(message)
    batch, coarse_height, coarse_width, _, _, modes = cells.shape
    directions = []
    for direction, permutation in enumerate(_PHYSICAL_FROM_RELATIVE):
        relative = cells[..., direction, :, :]
        physical = torch.stack(
            tuple(relative[..., index, :] for index in permutation),
            dim=-2,
        )
        directions.append(
            physical.reshape(batch, coarse_height, coarse_width, 2, 2, modes)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(batch, 2 * coarse_height, 2 * coarse_width, modes)
        )
    return torch.stack(directions, dim=-2)


def _vertical_product_scan(
    pole: tuple[Tensor, Tensor, Tensor, Tensor],
    source: ComplexField,
    source_variance: Tensor,
    *,
    reverse: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    """Apply one vertical recurrence without assuming a particular height."""
    source_real, source_imag = source
    if source_real.shape != source_imag.shape or source_real.ndim != 4:
        message = "vertical product scan requires matching NHWM tensors"
        raise ValueError(message)
    if source_variance.shape != source_real.shape:
        message = "vertical product scan variance has an incompatible shape"
        raise ValueError(message)

    storage_dtype = source_real.dtype
    active_real = source_real.float()
    active_imag = source_imag.float()
    active_variance = source_variance.float()
    decay_real, decay_imag, gamma_real, gamma_imag = (
        value.reshape(1, 1, -1).float() for value in pole
    )
    if reverse:
        active_real = active_real.flip(1)
        active_imag = active_imag.flip(1)
        active_variance = active_variance.flip(1)
        decay_imag = -decay_imag
        gamma_imag = -gamma_imag

    batch, height, width, modes = active_real.shape
    state_real = active_real.new_zeros((batch, width, modes))
    state_imag = active_imag.new_zeros((batch, width, modes))
    variance = active_variance.new_zeros((batch, width, modes))
    variance_decay = decay_real.detach().square() + decay_imag.detach().square()
    variance_gamma = gamma_real.detach().square() + gamma_imag.detach().square()
    real_rows = []
    imag_rows = []
    variance_rows = []
    for row in range(height):
        input_real = active_real[:, row]
        input_imag = active_imag[:, row]
        next_real = (
            decay_real * state_real
            - decay_imag * state_imag
            + gamma_real * input_real
            - gamma_imag * input_imag
        )
        next_imag = (
            decay_real * state_imag
            + decay_imag * state_real
            + gamma_real * input_imag
            + gamma_imag * input_real
        )
        variance = variance_decay * variance + variance_gamma * active_variance[:, row]
        state_real, state_imag = next_real, next_imag
        real_rows.append(state_real)
        imag_rows.append(state_imag)
        variance_rows.append(variance)

    output_real = torch.stack(real_rows, dim=1)
    output_imag = torch.stack(imag_rows, dim=1)
    output_variance = torch.stack(variance_rows, dim=1)
    if reverse:
        output_real = output_real.flip(1)
        output_imag = output_imag.flip(1)
        output_variance = output_variance.flip(1)
    return (
        output_real.to(dtype=storage_dtype),
        output_imag.to(dtype=storage_dtype),
        output_variance,
    )


def _normalized_odd_directional_states(
    pole_x: tuple[Tensor, Tensor, Tensor, Tensor],
    pole_y: tuple[Tensor, Tensor, Tensor, Tensor],
    source: ComplexField,
    *,
    gain_normalization: ProductGainNormalization,
    epsilon: float,
) -> ComplexField:
    """Return full D4 states for grids that cannot use lossless 2x2 packing."""
    source_real, _ = source
    batch, height, width, modes = source_real.shape
    horizontal = (
        pac_triton_bidirectional_product_scan(pole_x, source)
        if source_real.is_cuda
        else bidirectional_product_scan_reference(pole_x, source)
    )
    positive_x = horizontal[:2]
    negative_x = horizontal[2:]
    variance_x, _ = static_variance_tables(
        pole_x[0].detach(),
        pole_x[1].detach(),
        pole_x[2].detach(),
        pole_x[3].detach(),
        pole_y[0].detach(),
        pole_y[1].detach(),
        pole_y[2].detach(),
        pole_y[3].detach(),
        width,
        height,
    )
    positive_variance = (
        variance_x[0]
        .view(1, 1, width, modes)
        .expand(
            batch,
            height,
            width,
            modes,
        )
    )
    negative_variance = variance_x[1].view(1, 1, width, modes).expand_as(positive_variance)
    scans = (
        _vertical_product_scan(
            pole_y,
            positive_x,
            positive_variance,
            reverse=False,
        ),
        _vertical_product_scan(
            pole_y,
            negative_x,
            negative_variance,
            reverse=False,
        ),
        _vertical_product_scan(
            pole_y,
            positive_x,
            positive_variance,
            reverse=True,
        ),
        _vertical_product_scan(
            pole_y,
            negative_x,
            negative_variance,
            reverse=True,
        ),
    )
    real_paths = []
    imag_paths = []
    for state_real, state_imag, variance in scans:
        active_variance = (
            variance.mean((1, 2), keepdim=True) if gain_normalization == "global" else variance
        )
        inverse = torch.rsqrt(active_variance.detach().clamp_min(epsilon)).to(
            dtype=state_real.dtype
        )
        real_paths.append(state_real * inverse)
        imag_paths.append(state_imag * inverse)
    return torch.stack(real_paths, dim=-2), torch.stack(imag_paths, dim=-2)


class SameResolutionPoleScanBlock(nn.Module):
    """Run the established pole stage without spatial coarsening.

    A coarsening stage merges collapsed pole memory with an S2D carry before
    PostFusion.  At unchanged resolution the corresponding carry is the input
    excitation itself; every other learned operation remains the same.
    """

    def __init__(
        self,
        modes: int,
        *,
        pole_modes: int | None = None,
        reader_rank: int,
        kernel_size: int,
        pole_template: ComplexScanStage,
        post_hidden: int,
    ) -> None:
        super().__init__()
        active_pole_modes = modes if pole_modes is None else pole_modes
        if min(modes, active_pole_modes, reader_rank, kernel_size, post_hidden) <= 0:
            message = "same-resolution pole stage dimensions must be positive"
            raise ValueError(message)
        self.modes = int(modes)
        self.pole_modes = int(active_pole_modes)
        self.reader = FactorizedComplexConv2dReader(
            modes,
            self.pole_modes,
            rank=reader_rank,
            kernel_size=kernel_size,
            normalize_input=True,
            match_input_rms=False,
        )
        if self.modes == self.pole_modes:
            self.reader.initialize_orthogonal_()
        else:
            self.reader.initialize_semi_orthogonal_()
        self.path_collapse = PathOnlyCollapse(self.pole_modes, path_hidden=2 * D4_PATHS)
        self.memory_projection = (
            None
            if self.pole_modes == self.modes
            else ComplexLinear(self.pole_modes, self.modes)
        )
        if self.memory_projection is not None:
            semi_orthogonal_complex_linear_(self.memory_projection)
        self.post_fusion = GatedComplexPostFusion(modes, post_hidden)
        self.pole_scan = ComplexScanStage(
            self.pole_modes,
            maximum_phase=1.0,
            output_modes=None,
            scan_memory_policy="recompute",
            damping_min=pole_template.damping_min,
            damping_max=pole_template.damping_max,
        )
        self.copy_pole_initialization_(pole_template)

    @torch.no_grad()
    def copy_pole_initialization_(self, source: ComplexScanStage) -> None:
        """Clone a stage's pole atlas while retaining independent parameters."""
        if source.modes != self.pole_modes:
            message = "same-resolution pole template has an incompatible width"
            raise ValueError(message)
        self.pole_scan.damping_min = source.damping_min
        self.pole_scan.damping_max = source.damping_max
        self.pole_scan.product_gain_normalization = source.product_gain_normalization
        for name in ("damping_logits_x", "damping_logits_y", "phase_x", "phase_y"):
            getattr(self.pole_scan, name).copy_(getattr(source, name))

    def _scan_contract(
        self,
        real: Tensor,
        imag: Tensor,
    ) -> tuple[ComplexField, PoleCoefficients, PoleCoefficients, int, int]:
        drive_real, drive_imag = self.reader(real, imag)
        if drive_real.shape != drive_imag.shape or drive_real.ndim != 4:
            message = "same-resolution reader emitted incompatible pole drives"
            raise ValueError(message)
        shape = cast("tuple[int, int, int, int]", tuple(drive_real.shape))
        pole_x, pole_y = self.pole_scan.pole_coefficients(shape)
        height, width = shape[1:3]
        return (drive_real, drive_imag), pole_x, pole_y, height, width

    def _even_cells(
        self,
        source: ComplexField,
        pole_x: PoleCoefficients,
        pole_y: PoleCoefficients,
    ) -> ComplexField:
        full_real, full_imag, _ = cast(
            "tuple[Tensor, Tensor, Tensor]",
            run_product_scan_pipeline(
                pole_x,
                pole_y,
                source,
                epilogue="full16",
                gain_normalization=self.pole_scan.product_gain_normalization,
                memory_policy=self.pole_scan.scan_memory_policy,
            ),
        )
        return full_real, full_imag

    def directional_states(self, real: Tensor, imag: Tensor) -> ComplexField:
        source, pole_x, pole_y, height, width = self._scan_contract(real, imag)
        if height % 2 == 0 and width % 2 == 0:
            full_real, full_imag = self._even_cells(source, pole_x, pole_y)
            return (
                direction_relative_cells_to_full_resolution(full_real),
                direction_relative_cells_to_full_resolution(full_imag),
            )
        return _normalized_odd_directional_states(
            pole_x,
            pole_y,
            source,
            gain_normalization=self.pole_scan.product_gain_normalization,
            epsilon=DEFAULT_EPSILON,
        )

    def collapsed_memory(self, real: Tensor, imag: Tensor) -> ComplexField:
        """Produce collapsed memory through the collapse's declared scan contract."""
        source, pole_x, pole_y, height, width = self._scan_contract(real, imag)
        packed_parameters = getattr(self.path_collapse, "packed_parameters", None)
        if height % 2 == 0 and width % 2 == 0 and callable(packed_parameters):
            active_parameters = cast(
                "tuple[Tensor, Tensor, Tensor, Tensor]",
                packed_parameters(),
            )
            memory_real, memory_imag = run_product_scan_path_collapse_pipeline(
                pole_x,
                pole_y,
                source,
                active_parameters,
                gain_normalization=self.pole_scan.product_gain_normalization,
                memory_policy=self.pole_scan.scan_memory_policy,
                path_swiglu=bool(
                    getattr(self.path_collapse, "path_swiglu", False)
                ),
            )
            return self._project_memory(
                memory_real.squeeze(-2),
                memory_imag.squeeze(-2),
            )
        if height % 2 == 0 and width % 2 == 0:
            full_real, full_imag = self._even_cells(source, pole_x, pole_y)
            paths = (
                direction_relative_cells_to_full_resolution(full_real),
                direction_relative_cells_to_full_resolution(full_imag),
            )
        else:
            paths = _normalized_odd_directional_states(
                pole_x,
                pole_y,
                source,
                gain_normalization=self.pole_scan.product_gain_normalization,
                epsilon=DEFAULT_EPSILON,
            )
        return self.collapse_paths(*paths)

    def collapse_paths(self, real: Tensor, imag: Tensor) -> ComplexField:
        """Apply the canonical D4 collapse and remove its singleton path axis."""
        memory_real, memory_imag = self.path_collapse.forward_packed(real, imag)
        return self._project_memory(
            memory_real.squeeze(-2),
            memory_imag.squeeze(-2),
        )

    def _project_memory(self, real: Tensor, imag: Tensor) -> ComplexField:
        if self.memory_projection is None:
            return real, imag
        return self.memory_projection(real, imag)

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        if real.shape != imag.shape or real.ndim != 4 or real.shape[-1] != self.modes:
            message = "same-resolution pole block requires matching NHWM tensors"
            raise ValueError(message)
        memory_real, memory_imag = self.collapsed_memory(real, imag)
        return self.post_fusion(real + memory_real, imag + memory_imag)


__all__ = [
    "D4_PATHS",
    "SameResolutionPoleScanBlock",
    "direction_relative_cells_to_full_resolution",
]
