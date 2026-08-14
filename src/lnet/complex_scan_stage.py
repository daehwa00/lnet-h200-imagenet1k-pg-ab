"""Recurrence, descriptor, coarsening, and pole-bank implementation."""

from __future__ import annotations

# Private helpers in this implementation module are deliberately re-exported
# through the public complex_scan module.
# pyright: reportUnusedFunction=false
import math
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor, nn
from torch.nn import functional

from .complex_scan_transitions import (
    AugmentedComplexTransition,
    ComplexInteractionTransition,
    ComplexModulatedTransition,
    ComplexResidualFFN,
    ComplexRMSNorm,
)
from .pac_complex_layers import ComplexLinear, WidelyLinear
from .pac_path_cffn import D4PathModeCombiner, FactorizedQuadrantPathModeCFFNCombiner
from .pac_product_scan_pipeline import ScanMemoryPolicy, run_product_scan_pipeline
from .pac_real2d_math import discrete_pole_real2d
from .pac_triton_product_scan_coarse4 import (
    ProductGainNormalization,
    supports_pac_triton_product_scan_coarse4,
    supports_pac_triton_product_scan_descriptor4,
)

if TYPE_CHECKING:
    from .complex_scan_types import ComplexCarryBasis, ComplexCarryMerge, ComplexField
    from .pac_complex_ffn import ComplexFFN


def pole_aligned_complex_downsample(
    real: Tensor,
    imag: Tensor,
    phase_x: Tensor,
    phase_y: Tensor,
) -> ComplexField:
    """Downsample a modal field after aligning each pole's local carrier phase.

    The fixed binomial low-pass is modulated by the conjugate unit pole phase.
    Consequently a response that rotates according to ``phase_x/phase_y`` is
    averaged in its baseband frame instead of cancelling in image coordinates.
    """
    if real.shape != imag.shape or real.ndim != 4:
        message = "pole-aligned downsample requires matching NHWM tensors"
        raise ValueError(message)
    modes = real.shape[-1]
    if phase_x.shape != (modes,) or phase_y.shape != (modes,):
        message = "pole-aligned downsample phases do not match the modal width"
        raise ValueError(message)
    if real.shape[1] % 2 or real.shape[2] % 2:
        message = "pole-aligned downsample requires even spatial dimensions"
        raise ValueError(message)

    coordinate = torch.arange(-1, 2, device=real.device, dtype=phase_x.dtype)
    offset_y, offset_x = torch.meshgrid(coordinate, coordinate, indexing="ij")
    angle = (
        phase_x[:, None, None] * offset_x[None, :, :]
        + phase_y[:, None, None] * offset_y[None, :, :]
    )
    low_pass = torch.tensor(
        ((1.0, 2.0, 1.0), (2.0, 4.0, 2.0), (1.0, 2.0, 1.0)),
        device=real.device,
        dtype=phase_x.dtype,
    ).div(16.0)
    weight_real = low_pass[None, :, :] * torch.cos(angle)
    weight_imag = -low_pass[None, :, :] * torch.sin(angle)
    # One grouped convolution evaluates the complex depthwise filter.  Within
    # each group the channels are ordered (real, imag), as are the outputs.
    weights = torch.stack(
        (
            torch.stack((weight_real, -weight_imag), dim=1),
            torch.stack((weight_imag, weight_real), dim=1),
        ),
        dim=1,
    ).reshape(2 * modes, 2, 3, 3)
    weights = weights.to(dtype=real.dtype)
    packed = torch.stack((real, imag), dim=-1).permute(0, 3, 4, 1, 2)
    packed = packed.reshape(real.shape[0], 2 * modes, real.shape[1], real.shape[2])
    coarse = functional.conv2d(
        packed,
        weights,
        stride=2,
        padding=1,
        groups=modes,
    )
    coarse = coarse.reshape(
        real.shape[0],
        modes,
        2,
        real.shape[1] // 2,
        real.shape[2] // 2,
    ).permute(0, 3, 4, 1, 2)
    return coarse[..., 0], coarse[..., 1]


def _validate_bank_transition(
    modes: int,
    output_modes: int | None,
    *,
    transition_width: int | None,
    interaction_rank: int | None,
    widely_linear: bool,
    augmented_width: int | None,
) -> None:
    if modes <= 0:
        message = "complex scan stage requires positive modes"
        raise ValueError(message)
    if output_modes is not None and output_modes <= 0:
        message = "complex pole bridge output modes must be positive"
        raise ValueError(message)
    if transition_width is not None and (output_modes is None or transition_width <= 0):
        message = "non-terminal complex pole transition width must be positive"
        raise ValueError(message)
    if interaction_rank is not None and (output_modes is None or interaction_rank <= 0):
        message = "non-terminal complex pole interaction rank must be positive"
        raise ValueError(message)
    if transition_width is not None and interaction_rank is not None:
        message = "complex scan stage cannot use both wide and interaction transitions"
        raise ValueError(message)
    if widely_linear and output_modes is None:
        message = "terminal complex scan stage cannot use a widely-linear bridge"
        raise ValueError(message)
    if augmented_width is not None and (output_modes is None or augmented_width <= 0):
        message = "non-terminal augmented complex width must be positive"
        raise ValueError(message)
    enabled = sum(
        (
            transition_width is not None,
            interaction_rank is not None,
            widely_linear,
            augmented_width is not None,
        )
    )
    if enabled > 1:
        message = "complex scan stage accepts only one transition family"
        raise ValueError(message)


def _phase_atlas(modes: int, maximum_phase: float) -> tuple[Tensor, Tensor]:
    levels = modes // 4
    radial = torch.logspace(
        math.log10(maximum_phase / 8.0),
        math.log10(maximum_phase),
        levels,
    ).repeat_interleave(4)
    orientation = torch.tensor((0.0, math.pi / 4.0, math.pi / 2.0, 3.0 * math.pi / 4.0)).repeat(
        levels
    )
    return radial * torch.cos(orientation), radial * torch.sin(orientation)


def complex_carry_coordinates(
    real: Tensor,
    imag: Tensor,
    basis: ComplexCarryBasis,
) -> ComplexField:
    """Rearrange each complex 2x2 block into lossless S2D coordinates."""
    if basis == "none":
        message = "complex carry coordinates require the S2D basis"
        raise ValueError(message)
    if real.shape != imag.shape or real.ndim != 4:
        message = "complex carry inputs must be matching NHWM tensors"
        raise ValueError(message)
    batch, height, width, modes = real.shape
    if height % 2 or width % 2:
        message = "complex carry requires even spatial dimensions"
        raise ValueError(message)

    def space_to_depth(values: Tensor) -> Tensor:
        blocks = values.reshape(batch, height // 2, 2, width // 2, 2, modes)
        return blocks.permute(0, 1, 3, 2, 4, 5).reshape(
            batch,
            height // 2,
            width // 2,
            4 * modes,
        )

    if basis != "s2d":
        message = f"unsupported complex carry basis: {basis}"
        raise ValueError(message)
    return space_to_depth(real), space_to_depth(imag)


class ComplexCCCNDownsample(nn.Module):
    """Mode-preserving complex depthwise projection shortcut."""

    def __init__(self, input_modes: int, output_modes: int) -> None:
        super().__init__()
        if input_modes <= 0 or output_modes != input_modes:
            message = "complex CCCN shortcut requires equal positive mode counts"
            raise ValueError(message)
        self.input_modes = input_modes
        self.output_modes = output_modes
        self.depthwise_weight_real = nn.Parameter(torch.zeros(input_modes, 1, 3, 3))
        self.depthwise_weight_imag = nn.Parameter(torch.zeros(input_modes, 1, 3, 3))
        blur = torch.tensor(((1.0, 2.0, 1.0), (2.0, 4.0, 2.0), (1.0, 2.0, 1.0)))
        blur = blur.div(16.0)
        with torch.no_grad():
            self.depthwise_weight_real.copy_(blur.view(1, 1, 3, 3).expand(input_modes, -1, -1, -1))

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        if real.shape != imag.shape or real.shape[-1] != self.input_modes:
            message = "complex CCCN shortcut inputs have incompatible shapes"
            raise ValueError(message)
        real_nchw = real.permute(0, 3, 1, 2)
        imag_nchw = imag.permute(0, 3, 1, 2)
        depthwise_real = functional.conv2d(
            real_nchw,
            self.depthwise_weight_real,
            stride=2,
            padding=1,
            groups=self.input_modes,
        ) - functional.conv2d(
            imag_nchw,
            self.depthwise_weight_imag,
            stride=2,
            padding=1,
            groups=self.input_modes,
        )
        depthwise_imag = functional.conv2d(
            real_nchw,
            self.depthwise_weight_imag,
            stride=2,
            padding=1,
            groups=self.input_modes,
        ) + functional.conv2d(
            imag_nchw,
            self.depthwise_weight_real,
            stride=2,
            padding=1,
            groups=self.input_modes,
        )
        return (
            depthwise_real.permute(0, 2, 3, 1),
            depthwise_imag.permute(0, 2, 3, 1),
        )


class ComplexScanStage(nn.Module):
    """Reusable two-dimensional complex product-scan stage."""

    def __init__(  # noqa: C901, PLR0912, PLR0915
        self,
        modes: int,
        *,
        maximum_phase: float,
        output_modes: int | None,
        transition_width: int | None = None,
        interaction_rank: int | None = None,
        widely_linear: bool = False,
        augmented_width: int | None = None,
        carry_basis: ComplexCarryBasis = "none",
        carry_merge: ComplexCarryMerge = "pole_main",
        carry_scale_initial: float = 1.0e-2,
        coherence_gated_carry: bool = False,
        use_pole_aligned_shortcut: bool = False,
        use_cccn_shortcut: bool = False,
        use_zero_gated_pole_aligned_residual: bool = False,
        quadrant_path_mode_cffn_width: int | None = None,
        quadrant_path_cffn_width: int | None = None,
        stage_residual_scale_initial: float = 0.1,
        post_transition_width: int | None = None,
        scan_memory_policy: ScanMemoryPolicy = "retain",
        gate_sharpness: float = 8.0,
        damping_min: float = 0.01,
        damping_max: float = 0.7,
    ) -> None:
        super().__init__()
        _validate_bank_transition(
            modes,
            output_modes,
            transition_width=transition_width,
            interaction_rank=interaction_rank,
            widely_linear=widely_linear,
            augmented_width=augmented_width,
        )
        if carry_basis not in {"none", "s2d"}:
            message = f"unsupported complex carry basis: {carry_basis}"
            raise ValueError(message)
        if carry_basis != "none" and augmented_width is None:
            message = "complex stage carry requires an augmented transition"
            raise ValueError(message)
        if carry_merge not in {"pole_main", "carry_main"}:
            message = f"unsupported complex carry merge: {carry_merge}"
            raise ValueError(message)
        if carry_scale_initial < 0.0:
            message = "complex carry scale cannot be negative"
            raise ValueError(message)
        if use_pole_aligned_shortcut and output_modes != modes:
            message = "pole-aligned shortcut requires equal input and output mode counts"
            raise ValueError(message)
        if use_pole_aligned_shortcut and carry_basis != "none":
            message = "pole-aligned shortcut replaces the existing S2D carry"
            raise ValueError(message)
        if use_cccn_shortcut and output_modes != modes:
            message = "complex CCCN shortcut requires equal input and output mode counts"
            raise ValueError(message)
        if use_cccn_shortcut and (carry_basis != "none" or use_pole_aligned_shortcut):
            message = "complex CCCN shortcut is exclusive with carry and pole-aligned shortcut"
            raise ValueError(message)
        if use_zero_gated_pole_aligned_residual and output_modes != modes:
            message = "zero-gated pole-aligned residual requires equal mode counts"
            raise ValueError(message)
        if use_zero_gated_pole_aligned_residual and use_pole_aligned_shortcut:
            message = "zero-gated and replacement pole-aligned residuals are exclusive"
            raise ValueError(message)
        if (quadrant_path_mode_cffn_width is None) != (quadrant_path_cffn_width is None):
            message = "quadrant path/mode CFFN requires both hidden widths"
            raise ValueError(message)
        if quadrant_path_mode_cffn_width is not None and (
            quadrant_path_mode_cffn_width <= 0
            or quadrant_path_cffn_width is None
            or quadrant_path_cffn_width <= 0
            or output_modes is None
        ):
            message = "quadrant path/mode CFFN requires positive widths and a non-terminal bank"
            raise ValueError(message)
        if post_transition_width is not None and (
            post_transition_width <= 0 or output_modes is None
        ):
            message = "post-transition CFFN requires a positive width between stages"
            raise ValueError(message)
        if stage_residual_scale_initial <= 0.0:
            message = "complex stage residual scale must be positive"
            raise ValueError(message)
        if scan_memory_policy not in {"retain", "recompute"}:
            message = f"unsupported scan memory policy: {scan_memory_policy}"
            raise ValueError(message)
        self.modes = modes
        self.output_modes = output_modes
        self.carry_basis: ComplexCarryBasis = carry_basis
        self.use_pole_aligned_shortcut = use_pole_aligned_shortcut
        self.use_cccn_shortcut = use_cccn_shortcut
        self.use_zero_gated_pole_aligned_residual = use_zero_gated_pole_aligned_residual
        self.product_gain_normalization: ProductGainNormalization = "pointwise"
        self.scan_memory_policy: ScanMemoryPolicy = scan_memory_policy
        self.gate_sharpness = gate_sharpness
        self.damping_min = damping_min
        self.damping_max = damping_max
        base_damping = torch.logspace(math.log10(0.04), math.log10(0.35), modes)
        ratio = ((base_damping - damping_min) / (damping_max - damping_min)).clamp(
            1.0e-4,
            1.0 - 1.0e-4,
        )
        base_logits = torch.logit(ratio)
        self.damping_logits_x = nn.Parameter(base_logits.clone())
        self.damping_logits_y = nn.Parameter(base_logits.clone())
        phase_x, phase_y = _phase_atlas(modes, maximum_phase)
        self.phase_x = nn.Parameter(phase_x)
        self.phase_y = nn.Parameter(phase_y)
        pole_input_modes = 4 * modes
        self.transition = (
            ComplexModulatedTransition(pole_input_modes, transition_width, output_modes)
            if output_modes is not None and transition_width is not None
            else None
        )
        self.interaction = (
            ComplexInteractionTransition(
                pole_input_modes,
                pole_input_modes,
                output_modes,
                coherence_rank=interaction_rank,
            )
            if output_modes is not None and interaction_rank is not None
            else None
        )
        self.widely_bridge = (
            WidelyLinear(pole_input_modes, output_modes, bias=False)
            if output_modes is not None and widely_linear
            else None
        )
        self.augmented: ComplexFFN | None = (
            AugmentedComplexTransition(
                pole_input_modes,
                augmented_width,
                output_modes,
                carry_input_modes=4 * modes if carry_basis != "none" else None,
                carry_merge=carry_merge,
                carry_scale_initial=carry_scale_initial,
                coherence_gated_carry=coherence_gated_carry,
            )
            if output_modes is not None and augmented_width is not None
            else None
        )
        self.bridge = (
            ComplexLinear(pole_input_modes, output_modes)
            if output_modes is not None
            and transition_width is None
            and interaction_rank is None
            and not widely_linear
            and augmented_width is None
            else None
        )
        self.output_norm = (
            ComplexRMSNorm(output_modes)
            if output_modes is not None
            and transition_width is None
            and interaction_rank is None
            and augmented_width is None
            else None
        )
        self.post_transition_ffn = (
            ComplexResidualFFN(output_modes, post_transition_width)
            if output_modes is not None and post_transition_width is not None
            else None
        )
        if use_pole_aligned_shortcut or use_cccn_shortcut:
            self.stage_residual_scale = nn.Parameter(
                torch.full((modes,), stage_residual_scale_initial)
            )
        else:
            self.register_parameter("stage_residual_scale", None)
        with torch.random.fork_rng(devices=[]):
            self.cccn_shortcut = (
                ComplexCCCNDownsample(modes, output_modes)
                if use_cccn_shortcut and output_modes is not None
                else None
            )
            self.cccn_pole_input_norm = ComplexRMSNorm(modes) if use_cccn_shortcut else None
        if use_zero_gated_pole_aligned_residual:
            self.aligned_residual_norm = ComplexRMSNorm(modes)
            self.aligned_residual_gate = nn.Parameter(torch.zeros(modes))
        else:
            self.aligned_residual_norm = None
            self.register_parameter("aligned_residual_gate", None)
        with torch.random.fork_rng(devices=[]):
            self.quadrant_path_mode_combiner: D4PathModeCombiner | None = (
                FactorizedQuadrantPathModeCFFNCombiner(
                    modes,
                    quadrant_path_mode_cffn_width,
                    quadrant_path_cffn_width,
                )
                if quadrant_path_mode_cffn_width is not None
                and quadrant_path_cffn_width is not None
                else None
            )
        # Optional strict coupling for experiments where each pole reads the
        # complete excitation vector.  It is deliberately confined to the
        # scan branch: S2D carry continues to transport the original local
        # excitation.
        self.pole_input_projection: nn.Module | None = None

    def _damping_fields(self) -> tuple[Tensor, Tensor]:
        """Return the mode-static damping tensors used by the associative scan."""
        logits = torch.stack((self.damping_logits_x, self.damping_logits_y))
        ratio = torch.sigmoid(logits).view(2, 1, 1, 1, -1)
        damping = self.damping_min + (self.damping_max - self.damping_min) * ratio
        return damping[0], damping[1]

    def _pole_coefficients(
        self,
        shape: tuple[int, int, int, int],
    ) -> tuple[
        tuple[Tensor, Tensor, Tensor, Tensor],
        tuple[Tensor, Tensor, Tensor, Tensor],
    ]:
        """Build the compact mode-static pole tensors for the scan pipeline."""
        _, height, width, _ = shape
        spacing_x, spacing_y = 1.0 / width, 1.0 / height
        damping_x, damping_y = self._damping_fields()
        frequency_x = (self.phase_x / spacing_x).view(1, 1, 1, -1)
        frequency_y = (self.phase_y / spacing_y).view(1, 1, 1, -1)
        positive_x = discrete_pole_real2d(
            damping_x / spacing_x,
            frequency_x.expand_as(damping_x),
            spacing_x,
        )
        positive_y = discrete_pole_real2d(
            damping_y / spacing_y,
            frequency_y.expand_as(damping_y),
            spacing_y,
        )
        return positive_x, positive_y

    def _project_coarse(
        self,
        real: Tensor,
        imag: Tensor,
        carry_coordinates: ComplexField | None = None,
    ) -> ComplexField:
        if self.transition is not None:
            return self.transition(real, imag)
        if self.interaction is not None:
            return self.interaction(real, imag)
        if self.augmented is not None:
            if carry_coordinates is None:
                return self.augmented(real, imag)
            return self.augmented(real, imag, *carry_coordinates)
        if self.widely_bridge is not None:
            if self.output_norm is None:
                message = "widely-linear pole bridge is missing output normalization"
                raise RuntimeError(message)
            return self.output_norm(*self.widely_bridge(real, imag))
        if self.bridge is None or self.output_norm is None:
            message = "non-terminal complex scan stage is missing its bridge"
            raise RuntimeError(message)
        return self.output_norm(*self.bridge(real, imag))

    def forward(  # noqa: C901, PLR0912, PLR0915
        self,
        real: Tensor,
        imag: Tensor,
    ) -> tuple[ComplexField | None, Tensor]:
        if real.shape != imag.shape or real.ndim != 4 or real.shape[-1] != self.modes:
            message = "complex scan stage inputs must be matching NHWM tensors"
            raise ValueError(message)
        shortcut_input = (real, imag)
        if self.use_cccn_shortcut:
            if self.cccn_pole_input_norm is None:
                message = "complex CCCN shortcut is missing pole-branch pre-normalization"
                raise RuntimeError(message)
            real, imag = self.cccn_pole_input_norm(real, imag)
        carry_input = (real, imag)
        if self.pole_input_projection is not None:
            real, imag = self.pole_input_projection(real, imag)

        full_product_cells = bool(
            self.output_modes is not None
            and self.quadrant_path_mode_combiner is not None
            and self.quadrant_path_mode_combiner.requires_full_product_cells
        )
        epilogue = (
            "full16"
            if full_product_cells
            else "coarse" if self.output_modes is not None else "descriptor"
        )
        backend_available = (
            not real.is_cuda
            or (
                supports_pac_triton_product_scan_coarse4(real, imag)
                if epilogue != "descriptor"
                else supports_pac_triton_product_scan_descriptor4(real, imag)
            )
        )
        if not backend_available:
            message = "D4 associative product scan is unavailable for these tensors"
            raise RuntimeError(message)

        input_shape = cast("tuple[int, int, int, int]", tuple(real.shape))
        positive_x, positive_y = self._pole_coefficients(input_shape)
        scan_output = run_product_scan_pipeline(
            positive_x,
            positive_y,
            (real, imag),
            epilogue=epilogue,
            gain_normalization=self.product_gain_normalization,
            memory_policy=self.scan_memory_policy,
        )
        if self.output_modes is None:
            return None, cast("Tensor", scan_output)

        coarse_real, coarse_imag, descriptor = cast(
            "tuple[Tensor, Tensor, Tensor]",
            scan_output,
        )
        path_real, path_imag = coarse_real, coarse_imag
        collapse_product_paths = False
        if self.quadrant_path_mode_combiner is not None:
            if full_product_cells:
                path_real, path_imag = self.quadrant_path_mode_combiner.forward_full_state(
                    path_real,
                    path_imag,
                    pole_x=positive_x,
                    pole_y=positive_y,
                )
            else:
                path_real, path_imag = self.quadrant_path_mode_combiner.forward_packed(
                    path_real,
                    path_imag,
                )
            collapse_product_paths = bool(
                getattr(
                    self.quadrant_path_mode_combiner,
                    "collapses_product_paths",
                    False,
                )
            )
        if collapse_product_paths:
            if (
                path_real.shape != path_imag.shape
                or tuple(path_real.shape[-2:]) != (1, self.modes)
            ):
                message = "collapsed D4 path combiner must emit one product path"
                raise RuntimeError(message)
            concatenated_real = path_real.squeeze(-2)
            concatenated_imag = path_imag.squeeze(-2)
        else:
            if (
                path_real.shape != path_imag.shape
                or tuple(path_real.shape[-2:]) != (4, self.modes)
            ):
                message = "D4 path combiner must preserve four product paths"
                raise RuntimeError(message)
            carry_vector = torch.sigmoid(
                self.gate_sharpness * (math.pi / 2.0 - self.phase_x.abs())
            ) * torch.sigmoid(
                self.gate_sharpness * (math.pi / 2.0 - self.phase_y.abs())
            )
            active_carry = carry_vector.view(1, 1, 1, -1, 1).to(
                dtype=path_real.dtype
            )
            packed_real = active_carry * path_real.transpose(-2, -1)
            packed_imag = active_carry * path_imag.transpose(-2, -1)
            concatenated_real = packed_real.permute(0, 1, 2, 4, 3).flatten(-2)
            concatenated_imag = packed_imag.permute(0, 1, 2, 4, 3).flatten(-2)

        carry_coordinates = (
            complex_carry_coordinates(*carry_input, self.carry_basis)
            if self.carry_basis != "none"
            else None
        )
        projected_real, projected_imag = self._project_coarse(
            concatenated_real,
            concatenated_imag,
            carry_coordinates,
        )
        if self.post_transition_ffn is not None:
            projected_real, projected_imag = self.post_transition_ffn(
                projected_real,
                projected_imag,
            )
        if self.use_pole_aligned_shortcut:
            if self.stage_residual_scale is None:
                message = "pole-aligned shortcut is missing its residual scale"
                raise RuntimeError(message)
            shortcut_real, shortcut_imag = pole_aligned_complex_downsample(
                real,
                imag,
                self.phase_x,
                self.phase_y,
            )
            scale = self.stage_residual_scale.to(dtype=projected_real.dtype)
            projected_real = shortcut_real + scale * projected_real
            projected_imag = shortcut_imag + scale * projected_imag
        if self.use_cccn_shortcut:
            if self.stage_residual_scale is None or self.cccn_shortcut is None:
                message = "complex CCCN shortcut is missing its projection or residual scale"
                raise RuntimeError(message)
            shortcut_real, shortcut_imag = self.cccn_shortcut(*shortcut_input)
            scale = self.stage_residual_scale.to(dtype=projected_real.dtype)
            projected_real = shortcut_real + scale * projected_real
            projected_imag = shortcut_imag + scale * projected_imag
        if self.use_zero_gated_pole_aligned_residual:
            if self.aligned_residual_norm is None or self.aligned_residual_gate is None:
                message = "zero-gated pole-aligned residual is missing its gate"
                raise RuntimeError(message)
            aligned_real, aligned_imag = pole_aligned_complex_downsample(
                real,
                imag,
                self.phase_x,
                self.phase_y,
            )
            aligned_real, aligned_imag = self.aligned_residual_norm(
                aligned_real,
                aligned_imag,
            )
            gate = torch.tanh(self.aligned_residual_gate).to(dtype=projected_real.dtype)
            projected_real = projected_real + gate * aligned_real
            projected_imag = projected_imag + gate * aligned_imag
        return (projected_real, projected_imag), descriptor
