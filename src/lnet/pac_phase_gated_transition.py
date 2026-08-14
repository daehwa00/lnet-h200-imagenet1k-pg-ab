"""Phase-Gated replacements for the mode FFNs in factorized transitions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn
from torch.nn import functional

from .complex_scan_transitions import ComplexRMSNorm
from .pac_complex_ffn import ComplexFFN
from .pac_complex_layers import ComplexLinear
from .pac_grouped_path_cffn import GroupedWidelyLinear, grouped_cartesian_cffn
from .pac_mean_one_magnitude_gate import MeanOneMagnitudeGate
from .pac_path_cffn import D4PathModeCombiner
from .pac_phase_gated_cffn import PhaseGatedComplexFFN

if TYPE_CHECKING:
    from .pac_factorized_stage_transition import (
        FactorizedS2DPostFusionTransition,
        ModeResidualPathCollapse,
    )

ComplexField = tuple[Tensor, Tensor]


class PhaseGatedModeResidualPathCollapse(D4PathModeCombiner):
    """Apply Phase-Gated mode mixing, then the established 4-to-1 path block."""

    collapses_product_paths = True

    def __init__(
        self,
        modes: int,
        *,
        mode_hidden: int,
        path_hidden: int = 8,
        self_gated: bool = False,
        unit_row_projections: bool = False,
        residual_scale_max: float | None = None,
    ) -> None:
        super().__init__()
        if min(modes, mode_hidden, path_hidden) <= 0:
            message = "phase-gated stage mixer dimensions must be positive"
            raise ValueError(message)
        self.modes = modes
        self.path_count = 4
        self.output_paths = 1
        self.input_modes = 4 * modes
        self.mode = PhaseGatedComplexFFN(
            modes,
            mode_hidden,
            self_gated=self_gated,
            unit_row_projections=unit_row_projections,
            residual_scale_max=residual_scale_max,
        )
        self.path_input = GroupedWidelyLinear(modes, 4, path_hidden, bias=True)
        self.path_output = GroupedWidelyLinear(modes, path_hidden, 1, bias=True)

    def copy_path_from(self, baseline: ModeResidualPathCollapse) -> None:
        """Copy the unchanged path block from a matched MPM8 control."""
        if baseline.modes != self.modes:
            message = "cannot copy path parameters across different mode counts"
            raise ValueError(message)
        self.path_input.load_state_dict(baseline.path_input.state_dict())
        self.path_output.load_state_dict(baseline.path_output.state_dict())

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        if real.shape != imag.shape or real.shape[-1] != self.input_modes:
            message = "phase-gated stage mixer inputs have incompatible shapes"
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
            message = "phase-gated stage mixer requires NHW-path-mode inputs"
            raise ValueError(message)
        mixed_real, mixed_imag = self.mode(source_real, source_imag)
        return grouped_cartesian_cffn(
            mixed_real,
            mixed_imag,
            input_projection=self.path_input,
            output_projection=self.path_output,
        )


class PathOnlyCollapse(D4PathModeCombiner):
    """Keep the established nonlinear D4 collapse without a mode residual."""

    collapses_product_paths = True

    def __init__(self, modes: int, *, path_hidden: int = 8) -> None:
        super().__init__()
        if min(modes, path_hidden) <= 0:
            message = "path-only collapse dimensions must be positive"
            raise ValueError(message)
        self.modes = modes
        self.path_count = 4
        self.output_paths = 1
        self.input_modes = self.path_count * modes
        self.path_input = GroupedWidelyLinear(modes, 4, path_hidden, bias=True)
        self.path_output = GroupedWidelyLinear(modes, path_hidden, 1, bias=True)

    def copy_path_from(self, baseline: PhaseGatedModeResidualPathCollapse) -> None:
        """Copy the path block while deliberately omitting the PG mode block."""
        if baseline.modes != self.modes:
            message = "cannot copy path parameters across different mode counts"
            raise ValueError(message)
        self.path_input.load_state_dict(baseline.path_input.state_dict())
        self.path_output.load_state_dict(baseline.path_output.state_dict())

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        if real.shape != imag.shape or real.shape[-1] != self.input_modes:
            message = "path-only stage mixer inputs have incompatible shapes"
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
            message = "path-only stage mixer requires NHW-path-mode inputs"
            raise ValueError(message)
        return grouped_cartesian_cffn(
            source_real,
            source_imag,
            input_projection=self.path_input,
            output_projection=self.path_output,
        )


class PathPhaseGatedCollapse(D4PathModeCombiner):
    """Expand D4 paths, mix modes with one shared PG, then collapse paths."""

    collapses_product_paths = True

    def __init__(
        self,
        modes: int,
        *,
        mode_hidden: int,
        path_hidden: int = 8,
        apply_cartesian_silu: bool = True,
    ) -> None:
        super().__init__()
        if min(modes, mode_hidden, path_hidden) <= 0:
            message = "path Phase-Gated collapse dimensions must be positive"
            raise ValueError(message)
        self.modes = modes
        self.path_count = 4
        self.output_paths = 1
        self.input_modes = self.path_count * modes
        self.apply_cartesian_silu = bool(apply_cartesian_silu)
        self.path_input = GroupedWidelyLinear(modes, 4, path_hidden, bias=True)
        self.mode = PhaseGatedComplexFFN(modes, mode_hidden)
        self.path_output = GroupedWidelyLinear(modes, path_hidden, 1, bias=True)

    def copy_from(self, baseline: PhaseGatedModeResidualPathCollapse) -> None:
        """Reuse the matched mode and path parameters while changing their order."""
        if baseline.modes != self.modes:
            message = "cannot copy a path Phase-Gated block across different mode counts"
            raise ValueError(message)
        if baseline.mode.hidden_modes != self.mode.hidden_modes:
            message = "cannot copy a path Phase-Gated block across hidden widths"
            raise ValueError(message)
        self.mode.load_state_dict(baseline.mode.state_dict())
        self.path_input.load_state_dict(baseline.path_input.state_dict())
        self.path_output.load_state_dict(baseline.path_output.state_dict())

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        if real.shape != imag.shape or real.shape[-1] != self.input_modes:
            message = "path Phase-Gated mixer inputs have incompatible shapes"
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
            message = "path Phase-Gated mixer requires NHW-path-mode inputs"
            raise ValueError(message)
        hidden_real, hidden_imag = self.path_input(source_real, source_imag)
        # Grouped path projection returns a strided NHWPM view; packed PG kernels
        # require dense mode rows. Materialize that boundary once before shared PG.
        hidden_real = hidden_real.contiguous()
        hidden_imag = hidden_imag.contiguous()
        mixed_real, mixed_imag = self.mode(hidden_real, hidden_imag)
        if self.apply_cartesian_silu:
            mixed_real = functional.silu(mixed_real)
            mixed_imag = functional.silu(mixed_imag)
        return self.path_output(
            mixed_real,
            mixed_imag,
        )


class PhaseGatedModePathMeanCollapse(D4PathModeCombiner):
    """Apply residual PG blocks on mode and D4 axes, then average the paths."""

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
            message = "all-PG stage mixer dimensions must be positive"
            raise ValueError(message)
        self.modes = modes
        self.path_count = 4
        self.output_paths = 1
        self.input_modes = self.path_count * modes
        self.mode = PhaseGatedComplexFFN(modes, mode_hidden)
        self.path = PhaseGatedComplexFFN(self.path_count, path_hidden)

    def copy_mode_from(self, baseline: PhaseGatedModeResidualPathCollapse) -> None:
        """Copy a shape-compatible Phase-Gated mode block exactly."""
        if baseline.modes != self.modes:
            message = "cannot copy mode parameters across different mode counts"
            raise ValueError(message)
        self.mode.load_state_dict(baseline.mode.state_dict())

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        if real.shape != imag.shape or real.shape[-1] != self.input_modes:
            message = "all-PG stage mixer inputs have incompatible shapes"
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
            message = "all-PG stage mixer requires NHW-path-mode inputs"
            raise ValueError(message)
        mode_real, mode_imag = self.mode(source_real, source_imag)
        path_real, path_imag = self.path(
            mode_real.transpose(-2, -1),
            mode_imag.transpose(-2, -1),
        )
        return (
            path_real.mean(dim=-1).unsqueeze(-2),
            path_imag.mean(dim=-1).unsqueeze(-2),
        )


class PhaseGatedModePathResidualGWLCollapse(PhaseGatedModePathMeanCollapse):
    """Apply residual PG mode/path mixing, then learn the final D4 collapse."""

    def __init__(
        self,
        modes: int,
        *,
        mode_hidden: int,
        path_hidden: int = 8,
    ) -> None:
        super().__init__(
            modes,
            mode_hidden=mode_hidden,
            path_hidden=path_hidden,
        )
        self.collapse = GroupedWidelyLinear(modes, 4, 1, bias=True)

    def forward_packed(self, source_real: Tensor, source_imag: Tensor) -> ComplexField:
        expected = (self.path_count, self.modes)
        if (
            source_real.shape != source_imag.shape
            or source_real.ndim != 5
            or tuple(source_real.shape[-2:]) != expected
        ):
            message = "phase-gated path residual requires NHW-path-mode inputs"
            raise ValueError(message)
        mode_real, mode_imag = self.mode(source_real, source_imag)
        path_real, path_imag = self.path(
            mode_real.transpose(-2, -1),
            mode_imag.transpose(-2, -1),
        )
        return self.collapse(
            path_real.transpose(-2, -1),
            path_imag.transpose(-2, -1),
        )


class PhaseGatedModePathResidualComplexLinearCollapse(PhaseGatedModePathMeanCollapse):
    """Mix modes and paths residually, then collapse D4 complex-linearly."""

    def __init__(
        self,
        modes: int,
        *,
        mode_hidden: int,
        path_hidden: int = 8,
    ) -> None:
        super().__init__(
            modes,
            mode_hidden=mode_hidden,
            path_hidden=path_hidden,
        )
        self.collapse = ComplexLinear(self.path_count, 1)
        with torch.no_grad():
            self.collapse.weight_real.fill_(1.0 / self.path_count)
            self.collapse.weight_imag.zero_()

    def forward_packed(self, source_real: Tensor, source_imag: Tensor) -> ComplexField:
        expected = (self.path_count, self.modes)
        if (
            source_real.shape != source_imag.shape
            or source_real.ndim != 5
            or tuple(source_real.shape[-2:]) != expected
        ):
            message = "phase-gated complex-linear collapse requires NHW-path-mode inputs"
            raise ValueError(message)
        mode_real, mode_imag = self.mode(source_real, source_imag)
        path_real, path_imag = self.path(
            mode_real.transpose(-2, -1),
            mode_imag.transpose(-2, -1),
        )
        collapsed_real, collapsed_imag = self.collapse(path_real, path_imag)
        return collapsed_real.transpose(-2, -1), collapsed_imag.transpose(-2, -1)


class PhaseGatedS2DPostFusionTransition(ComplexFFN):
    """Merge the established S2D carry and apply one Phase-Gated residual FFN."""

    def __init__(self, modes: int, *, post_hidden: int) -> None:
        super().__init__()
        if min(modes, post_hidden) <= 0:
            message = "phase-gated post-fusion dimensions must be positive"
            raise ValueError(message)
        self.modes = modes
        self.input_modes = modes
        self.output_modes = modes
        self.carry_input_modes = 4 * modes
        self.carry_weight = nn.Parameter(torch.full((modes, 4), 0.25))
        self.post = PhaseGatedComplexFFN(modes, post_hidden)

    def copy_carry_from(self, baseline: FactorizedS2DPostFusionTransition) -> None:
        """Copy the unchanged S2D carry weights from a matched control."""
        if baseline.modes != self.modes:
            message = "cannot copy carry parameters across different mode counts"
            raise ValueError(message)
        with torch.no_grad():
            self.carry_weight.copy_(baseline.carry_weight)

    def _carry(self, real: Tensor, imag: Tensor) -> ComplexField:
        if real.shape != imag.shape or real.shape[-1] != self.carry_input_modes:
            message = "phase-gated transition S2D carry has incompatible shape"
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
            message = "phase-gated transition pole state has incompatible shape"
            raise ValueError(message)
        if carry_real is None or carry_imag is None:
            message = "phase-gated transition requires S2D carry coordinates"
            raise ValueError(message)
        carry_state_real, carry_state_imag = self._carry(carry_real, carry_imag)
        return self.post(real + carry_state_real, imag + carry_state_imag)


class AveragePoolPhaseGatedRefinementTransition(ComplexFFN):
    """Add a fixed 2x2 average carry, then apply a unit-scale PG refinement."""

    def __init__(self, modes: int, *, refine_hidden: int) -> None:
        super().__init__()
        if min(modes, refine_hidden) <= 0:
            message = "average-carry Phase-Gated dimensions must be positive"
            raise ValueError(message)
        self.modes = modes
        self.input_modes = modes
        self.output_modes = modes
        self.carry_input_modes = 4 * modes
        self.refine = PhaseGatedComplexFFN(
            modes,
            refine_hidden,
            residual_scale_init=1.0,
            learnable_residual_scale=False,
        )

    def _carry(self, real: Tensor, imag: Tensor) -> ComplexField:
        if real.shape != imag.shape or real.shape[-1] != self.carry_input_modes:
            message = "average-carry transition S2D input has incompatible shape"
            raise ValueError(message)
        shape = (*real.shape[:-1], 4, self.modes)
        return real.reshape(shape).mean(dim=-2), imag.reshape(shape).mean(dim=-2)

    def forward(
        self,
        real: Tensor,
        imag: Tensor,
        carry_real: Tensor | None = None,
        carry_imag: Tensor | None = None,
    ) -> ComplexField:
        if real.shape != imag.shape or real.shape[-1] != self.modes:
            message = "average-carry transition memory state has incompatible shape"
            raise ValueError(message)
        if carry_real is None or carry_imag is None:
            message = "average-carry transition requires S2D carry coordinates"
            raise ValueError(message)
        local_real, local_imag = self._carry(carry_real, carry_imag)
        merged_real = real + local_real
        merged_imag = imag + local_imag
        # Product scan accumulation is FP32 while the packed PG kernels execute
        # in BF16 under autocast. Keep that precision boundary explicit.
        if (
            merged_real.is_cuda
            and torch.is_autocast_enabled("cuda")
            and torch.get_autocast_dtype("cuda") is torch.bfloat16
        ):
            merged_real = merged_real.to(torch.bfloat16)
            merged_imag = merged_imag.to(torch.bfloat16)
        return self.refine(merged_real.contiguous(), merged_imag.contiguous())


class PureMagnitudeGateTransition(ComplexFFN):
    """Apply an identity-centered magnitude gate to a transition state."""

    def __init__(self, modes: int) -> None:
        super().__init__()
        if modes <= 0:
            message = "pure magnitude gate requires positive modes"
            raise ValueError(message)
        self.modes = modes
        self.input_modes = modes
        self.output_modes = modes
        self.norm = ComplexRMSNorm(modes)
        self.gate = MeanOneMagnitudeGate(modes)

    def _apply_gate(
        self,
        real: Tensor,
        imag: Tensor,
    ) -> ComplexField:
        if real.shape != imag.shape or real.shape[-1] != self.modes:
            message = "pure magnitude gate memory has incompatible shape"
            raise ValueError(message)
        if (
            real.is_cuda
            and torch.is_autocast_enabled("cuda")
            and torch.get_autocast_dtype("cuda") is torch.bfloat16
        ):
            real = real.to(torch.bfloat16)
            imag = imag.to(torch.bfloat16)
        real = real.contiguous()
        imag = imag.contiguous()
        normalized_real, normalized_imag = self.norm(real, imag)
        gain = self.gate(normalized_real, normalized_imag)
        return real * gain, imag * gain


class AveragePoolMagnitudeGateTransition(PureMagnitudeGateTransition):
    """Merge fixed local evidence and apply a pure magnitude gate."""

    def __init__(self, modes: int) -> None:
        super().__init__(modes)
        self.carry_input_modes = 4 * modes

    def _carry(self, real: Tensor, imag: Tensor) -> ComplexField:
        if real.shape != imag.shape or real.shape[-1] != self.carry_input_modes:
            message = "average-carry magnitude gate S2D input has incompatible shape"
            raise ValueError(message)
        shape = (*real.shape[:-1], 4, self.modes)
        return real.reshape(shape).mean(dim=-2), imag.reshape(shape).mean(dim=-2)

    def forward(
        self,
        real: Tensor,
        imag: Tensor,
        carry_real: Tensor | None = None,
        carry_imag: Tensor | None = None,
    ) -> ComplexField:
        if carry_real is None or carry_imag is None:
            message = "average-carry magnitude gate requires S2D carry coordinates"
            raise ValueError(message)
        local_real, local_imag = self._carry(carry_real, carry_imag)
        return self._apply_gate(real + local_real, imag + local_imag)


class MemoryOnlyMagnitudeGateTransition(PureMagnitudeGateTransition):
    """Gate collapsed scan memory without an S2D residual branch."""

    def forward(
        self,
        real: Tensor,
        imag: Tensor,
        _carry_real: Tensor | None = None,
        _carry_imag: Tensor | None = None,
    ) -> ComplexField:
        return self._apply_gate(real, imag)


__all__ = [
    "AveragePoolMagnitudeGateTransition",
    "AveragePoolPhaseGatedRefinementTransition",
    "MemoryOnlyMagnitudeGateTransition",
    "PathOnlyCollapse",
    "PathPhaseGatedCollapse",
    "PhaseGatedModePathMeanCollapse",
    "PhaseGatedModePathResidualComplexLinearCollapse",
    "PhaseGatedModePathResidualGWLCollapse",
    "PhaseGatedModeResidualPathCollapse",
    "PhaseGatedS2DPostFusionTransition",
    "PureMagnitudeGateTransition",
]
