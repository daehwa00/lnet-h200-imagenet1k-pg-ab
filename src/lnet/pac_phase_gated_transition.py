"""Phase-Gated replacements for the mode FFNs in factorized transitions."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn
from torch.nn import functional

from .complex_scan_transitions import ComplexRMSNorm
from .pac_complex_ffn import ComplexFFN
from .pac_d4_path_cffn import (
    d4_grouped_path_collapse,
    d4_grouped_path_swiglu,
    d4_grouped_path_swiglu_reference,
    supports_d4_grouped_path_collapse,
    supports_d4_grouped_path_swiglu,
)
from .pac_grouped_path_cffn import (
    GroupedWidelyLinear,
    grouped_cartesian_cell_cffn,
    grouped_cartesian_cffn,
)
from .pac_mean_one_magnitude_gate import MeanOneMagnitudeGate
from .pac_path_cffn import D4PathModeCombiner
from .pac_phase_gated_cffn import PhaseGatedComplexFFN

if TYPE_CHECKING:
    from .pac_factorized_stage_transition import ModeResidualPathCollapse

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

    def __init__(
        self,
        modes: int | PhaseGatedModeResidualPathCollapse,
        *,
        path_hidden: int = 8,
    ) -> None:
        super().__init__()
        if isinstance(modes, PhaseGatedModeResidualPathCollapse):
            self.modes = modes.modes
            self.path_count = modes.path_count
            self.output_paths = modes.output_paths
            self.input_modes = modes.input_modes
            self.path_input = modes.path_input
            self.path_output = modes.path_output
            return
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

    def forward_cells(self, source_real: Tensor, source_imag: Tensor) -> ComplexField:
        """Collapse direction-relative 2x2 cells directly at full resolution."""
        return grouped_cartesian_cell_cffn(
            source_real,
            source_imag,
            input_projection=self.path_input,
            output_projection=self.path_output,
        )

    def packed_parameters(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return the packed projection contract consumed by fused scan epilogues."""
        input_bias = self.path_input.packed_bias()
        output_bias = self.path_output.packed_bias()
        if input_bias is None or output_bias is None:
            raise RuntimeError("path collapse fusion requires both projection biases")
        packed_hidden = 2 * self.path_input.output_paths
        return (
            self.path_input.packed_weight().reshape(
                self.modes,
                packed_hidden,
                2 * self.path_input.input_paths,
            ),
            input_bias.reshape(self.modes, packed_hidden),
            self.path_output.packed_weight().reshape(
                self.modes,
                2 * self.path_output.output_paths,
                packed_hidden,
            ),
            output_bias.reshape(self.modes, 2 * self.path_output.output_paths),
        )


class DirectPathCollapse(D4PathModeCombiner):
    """Collapse four D4 paths with one learned real-linear complex projection."""

    collapses_product_paths = True

    def __init__(self, modes: int) -> None:
        super().__init__()
        if modes <= 0:
            raise ValueError("direct path collapse requires positive modes")
        self.modes = modes
        self.path_count = 4
        self.output_paths = 1
        self.input_modes = self.path_count * modes
        self.projection = GroupedWidelyLinear(modes, self.path_count, 1, bias=True)
        identity_output = torch.tensor(
            ((1.0, -1.0, 0.0, 0.0), (0.0, 0.0, 1.0, -1.0)),
        )
        self._identity_output: Tensor
        self.register_buffer("_identity_output", identity_output, persistent=False)

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        if real.shape != imag.shape or real.shape[-1] != self.input_modes:
            raise ValueError("direct path collapse inputs have incompatible shapes")
        shape = (*real.shape[:-1], self.path_count, self.modes)
        return self.forward_packed(real.reshape(shape), imag.reshape(shape))

    def forward_packed(self, source_real: Tensor, source_imag: Tensor) -> ComplexField:
        expected = (self.path_count, self.modes)
        if (
            source_real.shape != source_imag.shape
            or source_real.ndim != 5
            or tuple(source_real.shape[-2:]) != expected
        ):
            raise ValueError("direct path collapse requires NHW-path-mode inputs")
        if source_real.is_cuda:
            parameters = tuple(value.contiguous() for value in self.packed_parameters())
            if not supports_d4_grouped_path_collapse(
                source_real,
                source_imag,
                *parameters,
            ):
                raise RuntimeError(
                    "CUDA direct path collapse requires the fused BF16-autocast contract"
                )
            return d4_grouped_path_collapse(
                source_real.contiguous(),
                source_imag.contiguous(),
                *parameters,
            )
        output_real, output_imag = self.projection(source_real, source_imag)
        return output_real.contiguous(), output_imag.contiguous()

    def packed_parameters(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Encode the direct map exactly for the existing fused SiLU epilogue.

        ``SiLU(x) - SiLU(-x) == x`` lets the two-layer fused kernel evaluate
        the one-layer projection without adding trainable hidden parameters.
        """
        direct_bias = self.projection.packed_bias()
        if direct_bias is None:
            raise RuntimeError("direct path collapse unexpectedly lost its bias")
        direct_weight = self.projection.packed_weight().reshape(
            self.modes,
            2,
            2 * self.path_count,
        )
        direct_bias = direct_bias.reshape(self.modes, 2)
        input_weight = torch.stack(
            (
                direct_weight[:, 0],
                -direct_weight[:, 0],
                direct_weight[:, 1],
                -direct_weight[:, 1],
            ),
            dim=1,
        )
        input_bias = torch.stack(
            (
                direct_bias[:, 0],
                -direct_bias[:, 0],
                direct_bias[:, 1],
                -direct_bias[:, 1],
            ),
            dim=1,
        )
        output_weight = self._identity_output.expand(self.modes, -1, -1).contiguous()
        output_bias = direct_bias.new_zeros((self.modes, 2))
        return input_weight, input_bias, output_weight, output_bias


class PathSwiGLUCollapse(D4PathModeCombiner):
    """Collapse paths with a PostFusion-style complex value and real gate."""

    collapses_product_paths = True
    path_swiglu = True

    def __init__(
        self,
        modes: int,
        *,
        path_hidden: int = 4,
    ) -> None:
        super().__init__()
        if min(modes, path_hidden) <= 0:
            raise ValueError("Path-SwiGLU dimensions must be positive")
        self.modes = modes
        self.path_count = 4
        self.output_paths = 1
        self.input_modes = self.path_count * modes
        self.path_hidden = path_hidden
        self.value = GroupedWidelyLinear(
            modes,
            self.path_count,
            path_hidden,
            bias=False,
        )
        self.gate_weight = nn.Parameter(
            torch.empty(modes, path_hidden, 2 * self.path_count)
        )
        nn.init.uniform_(
            self.gate_weight,
            -1.0 / math.sqrt(2 * self.path_count),
            1.0 / math.sqrt(2 * self.path_count),
        )
        self.output = GroupedWidelyLinear(modes, path_hidden, 1, bias=False)

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        if real.shape != imag.shape or real.shape[-1] != self.input_modes:
            raise ValueError("Path-SwiGLU inputs have incompatible shapes")
        shape = (*real.shape[:-1], self.path_count, self.modes)
        return self.forward_packed(real.reshape(shape), imag.reshape(shape))

    def forward_packed(self, source_real: Tensor, source_imag: Tensor) -> ComplexField:
        expected = (self.path_count, self.modes)
        if (
            source_real.shape != source_imag.shape
            or source_real.ndim != 5
            or tuple(source_real.shape[-2:]) != expected
        ):
            raise ValueError("Path-SwiGLU requires NHW-path-mode inputs")
        parameters = self.packed_parameters()
        if source_real.is_cuda:
            if not supports_d4_grouped_path_swiglu(
                source_real,
                source_imag,
                *parameters,
            ):
                raise RuntimeError(
                    "CUDA Path-SwiGLU requires the fused BF16-autocast contract"
                )
            return d4_grouped_path_swiglu(
                source_real.contiguous(),
                source_imag.contiguous(),
                *parameters,
            )
        return d4_grouped_path_swiglu_reference(
            source_real,
            source_imag,
            *parameters,
        )

    def packed_parameters(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        value_weight = self.value.packed_weight().reshape(
            self.modes,
            2 * self.path_hidden,
            2 * self.path_count,
        )
        joint_weight = torch.cat(
            (
                value_weight,
                self.gate_weight,
            ),
            dim=1,
        ).contiguous()
        joint_bias = value_weight.new_zeros((self.modes, 3 * self.path_hidden))
        output_weight = self.output.packed_weight().reshape(
            self.modes,
            2,
            2 * self.path_hidden,
        ).contiguous()
        output_bias = value_weight.new_zeros((self.modes, 2))
        return joint_weight, joint_bias, output_weight, output_bias


class PathPhaseMagnitudeCollapse(D4PathModeCombiner):
    """Collapse four paths through a phase-invariant mean-one detector gate.

    The value and detector projections share one grouped launch, but remain
    semantically separate.  The detector magnitude can only redistribute gain
    among the hidden paths; it cannot close or amplify the whole collapse.
    """

    collapses_product_paths = True

    def __init__(self, modes: int, *, path_hidden: int = 4) -> None:
        super().__init__()
        if min(modes, path_hidden) <= 0:
            raise ValueError("Path-PG dimensions must be positive")
        self.modes = modes
        self.path_count = 4
        self.output_paths = 1
        self.input_modes = self.path_count * modes
        self.path_hidden = path_hidden
        self.value_detector = GroupedWidelyLinear(
            modes,
            self.path_count,
            2 * path_hidden,
            bias=True,
        )
        self.gate = MeanOneMagnitudeGate(path_hidden)
        self.output = GroupedWidelyLinear(modes, path_hidden, 1, bias=True)

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        if real.shape != imag.shape or real.shape[-1] != self.input_modes:
            raise ValueError("Path-PG inputs have incompatible shapes")
        shape = (*real.shape[:-1], self.path_count, self.modes)
        return self.forward_packed(real.reshape(shape), imag.reshape(shape))

    def forward_packed(self, source_real: Tensor, source_imag: Tensor) -> ComplexField:
        expected = (self.path_count, self.modes)
        if (
            source_real.shape != source_imag.shape
            or source_real.ndim != 5
            or tuple(source_real.shape[-2:]) != expected
        ):
            raise ValueError("Path-PG requires NHW-path-mode inputs")
        joint_real, joint_imag = self.value_detector(source_real, source_imag)
        value_real, detector_real = joint_real.split(self.path_hidden, dim=-2)
        value_imag, detector_imag = joint_imag.split(self.path_hidden, dim=-2)
        gate = self.gate(
            detector_real.transpose(-2, -1),
            detector_imag.transpose(-2, -1),
        ).transpose(-2, -1)
        output_real, output_imag = self.output(value_real * gate, value_imag * gate)
        return output_real.contiguous(), output_imag.contiguous()


class PathSelfMagnitudeCollapse(D4PathModeCombiner):
    """Use one H-wide value projection as its own mean-one path detector."""

    collapses_product_paths = True

    def __init__(self, modes: int, *, path_hidden: int = 4) -> None:
        super().__init__()
        if min(modes, path_hidden) <= 0:
            raise ValueError("self-gated path dimensions must be positive")
        self.modes = modes
        self.path_count = 4
        self.output_paths = 1
        self.input_modes = self.path_count * modes
        self.path_hidden = path_hidden
        self.value = GroupedWidelyLinear(
            modes,
            self.path_count,
            path_hidden,
            bias=True,
        )
        self.gate = MeanOneMagnitudeGate(path_hidden)
        self.output = GroupedWidelyLinear(modes, path_hidden, 1, bias=True)

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        if real.shape != imag.shape or real.shape[-1] != self.input_modes:
            raise ValueError("self-gated path inputs have incompatible shapes")
        shape = (*real.shape[:-1], self.path_count, self.modes)
        return self.forward_packed(real.reshape(shape), imag.reshape(shape))

    def forward_packed(self, source_real: Tensor, source_imag: Tensor) -> ComplexField:
        expected = (self.path_count, self.modes)
        if (
            source_real.shape != source_imag.shape
            or source_real.ndim != 5
            or tuple(source_real.shape[-2:]) != expected
        ):
            raise ValueError("self-gated path collapse requires NHW-path-mode inputs")
        value_real, value_imag = self.value(source_real, source_imag)
        gate = self.gate(
            value_real.transpose(-2, -1),
            value_imag.transpose(-2, -1),
        ).transpose(-2, -1)
        output_real, output_imag = self.output(value_real * gate, value_imag * gate)
        return output_real.contiguous(), output_imag.contiguous()


class DirectMagnitudePathCollapse(D4PathModeCombiner):
    """Gate the four raw path magnitudes before one widely-linear collapse."""

    collapses_product_paths = True

    def __init__(self, modes: int) -> None:
        super().__init__()
        if modes <= 0:
            raise ValueError("magnitude-gated direct path modes must be positive")
        self.modes = modes
        self.path_count = 4
        self.output_paths = 1
        self.input_modes = self.path_count * modes
        self.gate = MeanOneMagnitudeGate(self.path_count)
        self.output = GroupedWidelyLinear(modes, self.path_count, 1, bias=True)

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        if real.shape != imag.shape or real.shape[-1] != self.input_modes:
            raise ValueError("magnitude-gated direct path inputs have incompatible shapes")
        shape = (*real.shape[:-1], self.path_count, self.modes)
        return self.forward_packed(real.reshape(shape), imag.reshape(shape))

    def forward_packed(self, source_real: Tensor, source_imag: Tensor) -> ComplexField:
        expected = (self.path_count, self.modes)
        if (
            source_real.shape != source_imag.shape
            or source_real.ndim != 5
            or tuple(source_real.shape[-2:]) != expected
        ):
            raise ValueError(
                "magnitude-gated direct collapse requires NHW-path-mode inputs"
            )
        gate = self.gate(
            source_real.transpose(-2, -1),
            source_imag.transpose(-2, -1),
        ).transpose(-2, -1)
        output_real, output_imag = self.output(source_real * gate, source_imag * gate)
        return output_real.contiguous(), output_imag.contiguous()


class PathCoherenceGateCollapse(D4PathModeCombiner):
    """Collapse D4 paths with a global-phase-invariant coherence gate.

    The score for path ``d`` is evaluated as

    ``Re(z_d * conj(sum_e A[d,e] z_e))``.

    This is algebraically equal to a learned contraction of the four-by-four
    path Gram matrix, but it never materializes that quadratic activation.
    """

    collapses_product_paths = True
    redistribution = 0.5

    def __init__(self, modes: int) -> None:
        super().__init__()
        if modes <= 0:
            raise ValueError("coherence-gated path modes must be positive")
        self.modes = modes
        self.path_count = 4
        self.output_paths = 1
        self.input_modes = self.path_count * modes
        self.coherence_weight = nn.Parameter(
            torch.zeros(modes, self.path_count, self.path_count)
        )
        self.output = GroupedWidelyLinear(modes, self.path_count, 1, bias=True)

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        if real.shape != imag.shape or real.shape[-1] != self.input_modes:
            raise ValueError("coherence-gated path inputs have incompatible shapes")
        shape = (*real.shape[:-1], self.path_count, self.modes)
        return self.forward_packed(real.reshape(shape), imag.reshape(shape))

    def gate_values(self, source_real: Tensor, source_imag: Tensor) -> Tensor:
        """Return an exact mean-one gate without constructing a Gram tensor."""
        expected = (self.path_count, self.modes)
        if (
            source_real.shape != source_imag.shape
            or source_real.ndim != 5
            or tuple(source_real.shape[-2:]) != expected
        ):
            raise ValueError("coherence gate requires NHW-path-mode inputs")
        real = source_real.transpose(-2, -1).float()
        imag = source_imag.transpose(-2, -1).float()
        inverse_rms = torch.rsqrt(
            real.square().add(imag.square()).mean(dim=-1, keepdim=True).clamp_min(1.0e-12)
        )
        unit_real = real * inverse_rms
        unit_imag = imag * inverse_rms
        mixed_real = torch.einsum(
            "...mp,mdp->...md",
            unit_real,
            self.coherence_weight,
        )
        mixed_imag = torch.einsum(
            "...mp,mdp->...md",
            unit_imag,
            self.coherence_weight,
        )
        score = unit_real * mixed_real + unit_imag * mixed_imag
        centered = score - score.mean(dim=-1, keepdim=True)
        relative = 1.0 + self.redistribution * torch.tanh(centered)
        gate = relative / relative.mean(dim=-1, keepdim=True)
        return gate.transpose(-2, -1).to(dtype=source_real.dtype)

    def forward_packed(self, source_real: Tensor, source_imag: Tensor) -> ComplexField:
        gate = self.gate_values(source_real, source_imag)
        output_real, output_imag = self.output(source_real * gate, source_imag * gate)
        return output_real.contiguous(), output_imag.contiguous()


class ResidualPathCoherenceGateCollapse(PathCoherenceGateCollapse):
    """Keep a direct collapse and add only the coherence-driven redistribution."""

    def __init__(
        self,
        modes: int,
        *,
        correction_scale: float = 0.1,
        coherence_init: float = 0.01,
    ) -> None:
        if correction_scale <= 0.0 or coherence_init <= 0.0:
            raise ValueError("coherence correction initialization must be positive")
        super().__init__(modes)
        nn.init.normal_(self.coherence_weight, std=coherence_init)
        self.correction = GroupedWidelyLinear(
            modes,
            self.path_count,
            1,
            bias=False,
        )
        self.correction_scale = nn.Parameter(torch.tensor(float(correction_scale)))

    def forward_packed(self, source_real: Tensor, source_imag: Tensor) -> ComplexField:
        gate = self.gate_values(source_real, source_imag)
        base_real, base_imag = self.output(source_real, source_imag)
        correction_real, correction_imag = self.correction(
            source_real * (gate - 1.0),
            source_imag * (gate - 1.0),
        )
        scale = self.correction_scale.to(dtype=base_real.dtype)
        return (
            (base_real + scale * correction_real).contiguous(),
            (base_imag + scale * correction_imag).contiguous(),
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
    "DirectMagnitudePathCollapse",
    "DirectPathCollapse",
    "MemoryOnlyMagnitudeGateTransition",
    "PathCoherenceGateCollapse",
    "PathOnlyCollapse",
    "PathPhaseGatedCollapse",
    "PathPhaseMagnitudeCollapse",
    "PathSelfMagnitudeCollapse",
    "PathSwiGLUCollapse",
    "PhaseGatedModeResidualPathCollapse",
    "PureMagnitudeGateTransition",
    "ResidualPathCoherenceGateCollapse",
]
