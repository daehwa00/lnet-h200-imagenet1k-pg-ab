"""PolePyramid-A-Tiny for CIFAR with block-conditioned stable damping."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import torch
from torch import Tensor, nn
from torch.nn import functional
from torch.nn.utils.parametrizations import orthogonal

from .image_layers import (
    CifarConvStem,
    LowRankQuadraticModalHead,
    StandardizedAffineModalHead,
)
from .pac_directional import direction_aligned_endpoints
from .pac_real2d_math import discrete_pole_real2d
from .pac_recurrence import (
    recurrence_real2d_directional,
    recurrence_real2d_state_variance_directional,
)

if TYPE_CHECKING:
    from .pac_recurrence import RecurrenceBackend

ComplexField = tuple[Tensor, Tensor]
DirectionalScan = tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
PyramidTransport = Literal["pole", "average"]
ComplementResidual = Literal["none", "full", "orthogonal"]
TransportBranchMask = Literal["combined", "pole_only", "residual_only"]
_DIRECTIONS = ((1, 1), (-1, 1), (1, -1), (-1, -1))


def _phase_atlas(modes: int, maximum_phase: float) -> tuple[Tensor, Tensor]:
    levels = modes // 4
    radial = torch.logspace(
        math.log10(maximum_phase / 8.0), math.log10(maximum_phase), levels
    ).repeat_interleave(4)
    orientation = torch.tensor((0.0, math.pi / 4.0, math.pi / 2.0, 3.0 * math.pi / 4.0)).repeat(
        levels
    )
    return radial * torch.cos(orientation), radial * torch.sin(orientation)


def _axis_scan_dynamic(
    input_real: Tensor,
    input_imag: Tensor,
    decay_real: Tensor,
    decay_imag: Tensor,
    *,
    direction: int,
    backend: RecurrenceBackend,
) -> ComplexField:
    if direction not in {-1, 1}:
        message = "scan direction must be -1 or 1"
        raise ValueError(message)
    return recurrence_real2d_directional(
        decay_real,
        decay_imag,
        input_real,
        input_imag,
        backend,
        "forward" if direction == 1 else "backward",
    )


def _axis_scan_state_variance_dynamic(
    input_real: Tensor,
    input_imag: Tensor,
    decay_real: Tensor,
    decay_imag: Tensor,
    variance_input: Tensor,
    variance_decay: Tensor,
    *,
    direction: int,
    backend: RecurrenceBackend,
) -> tuple[Tensor, Tensor, Tensor]:
    if direction not in {-1, 1}:
        message = "scan direction must be -1 or 1"
        raise ValueError(message)
    return recurrence_real2d_state_variance_directional(
        decay_real,
        decay_imag,
        input_real,
        input_imag,
        variance_decay,
        variance_input,
        backend,
        "forward" if direction == 1 else "backward",
    )


class PoleDown2D(nn.Module):
    """Mode-static fine recurrence, normalized energy, and exact endpoint carry."""

    def __init__(  # noqa: C901, PLR0912, PLR0915
        self,
        input_width: int,
        output_width: int,
        modes: int,
        *,
        maximum_phase: float,
        recurrence_backend: RecurrenceBackend,
        dynamic_gain_normalization: bool,
        damping_min: float,
        damping_max: float,
        gate_sharpness: float,
        layer_scale_init: float,
        transport: PyramidTransport = "pole",
        stop_gradient_gain_normalization: bool = False,
        quadrant_scan_fusion: bool = True,
        fuse_state_variance_recurrence: bool = False,
        complement_residual: ComplementResidual = "none",
        complement_detach_projector: bool = True,
        complement_scale_init: float = 1.0e-2,
        modal_carry_rank: int = 0,
        modal_carry_learned: bool = True,
        tcir_innovation_reweighting: bool = False,
        tcir_radius: float = 0.5,
    ) -> None:
        super().__init__()
        self.modes = modes
        self.recurrence_backend: RecurrenceBackend = recurrence_backend
        self.dynamic_gain_normalization = dynamic_gain_normalization
        self.damping_min = damping_min
        self.damping_max = damping_max
        self.gate_sharpness = gate_sharpness
        self.transport = transport
        self.stop_gradient_gain_normalization = stop_gradient_gain_normalization
        self.quadrant_scan_fusion = quadrant_scan_fusion
        self.fuse_state_variance_recurrence = fuse_state_variance_recurrence
        self.complement_residual = complement_residual
        self.complement_detach_projector = complement_detach_projector
        self.modal_carry_rank = modal_carry_rank
        self.branch_mask: TransportBranchMask = "combined"
        self.tcir_radius = tcir_radius
        if fuse_state_variance_recurrence and not (
            dynamic_gain_normalization and stop_gradient_gain_normalization and quadrant_scan_fusion
        ):
            message = (
                "state/variance fusion requires dynamic stop-gradient gain "
                "normalization and quadrant scan fusion"
            )
            raise ValueError(message)
        if complement_residual not in {"none", "full", "orthogonal"}:
            message = f"unsupported complement residual: {complement_residual}"
            raise ValueError(message)
        if complement_residual != "none" and transport != "pole":
            message = "complement residual requires pole transport"
            raise ValueError(message)
        if complement_scale_init < 0.0:
            message = "complement residual scale cannot be negative"
            raise ValueError(message)
        if not 0 <= modal_carry_rank <= 2 * modes:
            message = "modal carry rank must be in [0, 2M]"
            raise ValueError(message)
        if modal_carry_rank and complement_residual != "orthogonal":
            message = "modal carry requires the orthogonal-complement residual"
            raise ValueError(message)
        if tcir_innovation_reweighting and transport != "pole":
            message = "TCIR requires pole transport"
            raise ValueError(message)
        if not 0.0 < tcir_radius <= 1.0:
            message = "TCIR radius must be in (0, 1]"
            raise ValueError(message)
        self.input_norm = nn.RMSNorm(input_width)
        self.analysis = nn.Linear(input_width, 2 * modes, bias=False)
        nn.init.orthogonal_(self.analysis.weight)
        orthogonal(self.analysis, "weight", orthogonal_map="matrix_exp", use_trivialization=True)
        base_damping = torch.logspace(math.log10(0.04), math.log10(0.35), modes)
        ratio = ((base_damping - damping_min) / (damping_max - damping_min)).clamp(
            1.0e-4, 1.0 - 1.0e-4
        )
        base_logits = torch.logit(ratio)
        self.damping_logits_x = nn.Parameter(base_logits.clone())
        self.damping_logits_y = nn.Parameter(base_logits.clone())
        phase_x, phase_y = _phase_atlas(modes, maximum_phase)
        self.phase_x = nn.Parameter(phase_x)
        self.phase_y = nn.Parameter(phase_y)
        if tcir_innovation_reweighting:
            self.tcir_innovation_logits = nn.Parameter(torch.zeros(modes))
        else:
            self.register_parameter("tcir_innovation_logits", None)
        self.direction_mix = nn.Linear(4 * input_width, output_width)
        self.modal_carry_projection: nn.Linear | None
        if modal_carry_rank:
            self.modal_carry_projection = nn.Linear(
                2 * modes,
                modal_carry_rank,
                bias=False,
            )
            nn.init.orthogonal_(self.modal_carry_projection.weight)
            if modal_carry_learned:
                orthogonal(
                    self.modal_carry_projection,
                    "weight",
                    orthogonal_map="matrix_exp",
                    use_trivialization=True,
                )
            else:
                self.modal_carry_projection.weight.requires_grad = False
        else:
            self.modal_carry_projection = None
        self.residual_projection: nn.Linear | None
        self.residual_scale: nn.Parameter | None
        if complement_residual == "none":
            self.residual_projection = None
            self.register_parameter("residual_scale", None)
        else:
            residual_width = (
                input_width
                if complement_residual == "full"
                else input_width - 2 * modes + modal_carry_rank
            )
            self.residual_projection = nn.Linear(4 * residual_width, output_width)
            self.residual_scale = nn.Parameter(torch.full((output_width,), complement_scale_init))
        self.mlp_norm = nn.RMSNorm(output_width)
        self.mlp = nn.Sequential(
            nn.Linear(output_width, 2 * output_width),
            nn.SiLU(),
            nn.Linear(2 * output_width, output_width),
        )
        self.mlp_scale = nn.Parameter(torch.full((output_width,), layer_scale_init))

    def _damping_fields(self, normalized: Tensor) -> tuple[Tensor, Tensor]:
        batch, height, width, _ = normalized.shape
        shape = (batch, height // 2, width // 2, self.modes)

        def static(logits: Tensor) -> Tensor:
            ratio = torch.sigmoid(logits).view(1, 1, 1, -1)
            damping = self.damping_min + (self.damping_max - self.damping_min) * ratio
            return damping.expand(shape)

        return static(self.damping_logits_x), static(self.damping_logits_y)

    @staticmethod
    def _expand_blocks(values: Tensor) -> Tensor:
        return values.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)

    def _scan_direction(
        self,
        excitation_real: Tensor,
        excitation_imag: Tensor,
        damping_x: Tensor,
        damping_y: Tensor,
        *,
        direction_x: int,
        direction_y: int,
    ) -> DirectionalScan:
        batch, height, width, modes = excitation_real.shape
        spacing_x, spacing_y = 1.0 / width, 1.0 / height
        fine_damping_x = self._expand_blocks(damping_x)
        fine_damping_y = self._expand_blocks(damping_y)
        frequency_x = (direction_x * self.phase_x / spacing_x).view(1, 1, 1, -1)
        frequency_y = (direction_y * self.phase_y / spacing_y).view(1, 1, 1, -1)
        decay_x_real, decay_x_imag, gamma_x_real, gamma_x_imag = discrete_pole_real2d(
            fine_damping_x / spacing_x,
            frequency_x.expand_as(fine_damping_x),
            spacing_x,
        )
        drive_x_real = gamma_x_real * excitation_real - gamma_x_imag * excitation_imag
        drive_x_imag = gamma_x_real * excitation_imag + gamma_x_imag * excitation_real
        horizontal_real, horizontal_imag = _axis_scan_dynamic(
            drive_x_real.reshape(batch * height, width, modes),
            drive_x_imag.reshape(batch * height, width, modes),
            decay_x_real.reshape(batch * height, width, modes),
            decay_x_imag.reshape(batch * height, width, modes),
            direction=direction_x,
            backend=self.recurrence_backend,
        )
        horizontal_real = horizontal_real.reshape(batch, height, width, modes)
        horizontal_imag = horizontal_imag.reshape(batch, height, width, modes)
        decay_y_real, decay_y_imag, gamma_y_real, gamma_y_imag = discrete_pole_real2d(
            fine_damping_y / spacing_y,
            frequency_y.expand_as(fine_damping_y),
            spacing_y,
        )
        drive_y_real = gamma_y_real * horizontal_real - gamma_y_imag * horizontal_imag
        drive_y_imag = gamma_y_real * horizontal_imag + gamma_y_imag * horizontal_real
        vertical_shape = (batch * width, height, modes)
        state_real, state_imag = _axis_scan_dynamic(
            drive_y_real.permute(0, 2, 1, 3).reshape(vertical_shape),
            drive_y_imag.permute(0, 2, 1, 3).reshape(vertical_shape),
            decay_y_real.permute(0, 2, 1, 3).reshape(vertical_shape),
            decay_y_imag.permute(0, 2, 1, 3).reshape(vertical_shape),
            direction=direction_y,
            backend=self.recurrence_backend,
        )
        state_real = state_real.reshape(batch, width, height, modes).permute(0, 2, 1, 3)
        state_imag = state_imag.reshape(batch, width, height, modes).permute(0, 2, 1, 3)
        innovation_real, innovation_imag = self._block_innovation(
            excitation_real,
            excitation_imag,
            decay_x_real,
            decay_x_imag,
            gamma_x_real,
            gamma_x_imag,
            decay_y_real,
            decay_y_imag,
            gamma_y_real,
            gamma_y_imag,
            direction_x=direction_x,
            direction_y=direction_y,
        )
        if not self.dynamic_gain_normalization:
            variance = torch.ones_like(state_real)
            return state_real, state_imag, variance, innovation_real, innovation_imag
        zero_x = torch.zeros_like(decay_x_real)
        variance_x, _ = _axis_scan_dynamic(
            gamma_x_real.square().add(gamma_x_imag.square()).reshape(batch * height, width, modes),
            torch.zeros_like(excitation_real).reshape(batch * height, width, modes),
            decay_x_real.square().add(decay_x_imag.square()).reshape(batch * height, width, modes),
            zero_x.reshape(batch * height, width, modes),
            direction=direction_x,
            backend=self.recurrence_backend,
        )
        variance_x = variance_x.reshape(batch, height, width, modes)
        variance_drive = gamma_y_real.square().add(gamma_y_imag.square()) * variance_x
        zero_y = torch.zeros_like(decay_y_real)
        variance, _ = _axis_scan_dynamic(
            variance_drive.permute(0, 2, 1, 3).reshape(vertical_shape),
            torch.zeros_like(variance_drive).permute(0, 2, 1, 3).reshape(vertical_shape),
            decay_y_real.square()
            .add(decay_y_imag.square())
            .permute(0, 2, 1, 3)
            .reshape(vertical_shape),
            zero_y.permute(0, 2, 1, 3).reshape(vertical_shape),
            direction=direction_y,
            backend=self.recurrence_backend,
        )
        variance = variance.reshape(batch, width, height, modes).permute(0, 2, 1, 3)
        return state_real, state_imag, variance, innovation_real, innovation_imag

    def _scan_quadrants(  # noqa: PLR0915
        self,
        excitation_real: Tensor,
        excitation_imag: Tensor,
        damping_x: Tensor,
        damping_y: Tensor,
    ) -> list[DirectionalScan]:
        """Evaluate four quadrants while sharing axis-identical scan work."""
        batch, height, width, modes = excitation_real.shape
        spacing_x, spacing_y = 1.0 / width, 1.0 / height
        fine_damping_x = self._expand_blocks(damping_x)
        fine_damping_y = self._expand_blocks(damping_y)
        horizontal: dict[int, ComplexField] = {}
        horizontal_poles: dict[int, tuple[Tensor, Tensor, Tensor, Tensor]] = {}
        variance_x: dict[int, Tensor] = {}
        for direction_x in (1, -1):
            frequency_x = (direction_x * self.phase_x / spacing_x).view(1, 1, 1, -1)
            decay_x_real, decay_x_imag, gamma_x_real, gamma_x_imag = discrete_pole_real2d(
                fine_damping_x / spacing_x,
                frequency_x.expand_as(fine_damping_x),
                spacing_x,
            )
            drive_x_real = gamma_x_real * excitation_real - gamma_x_imag * excitation_imag
            drive_x_imag = gamma_x_real * excitation_imag + gamma_x_imag * excitation_real
            state_shape_x = (batch * height, width, modes)
            if self.fuse_state_variance_recurrence:
                magnitude_decay_x = decay_x_real.square().add(decay_x_imag.square())
                magnitude_gamma_x = gamma_x_real.square().add(gamma_x_imag.square())
                horizontal_real, horizontal_imag, current_variance_x = (
                    _axis_scan_state_variance_dynamic(
                        drive_x_real.reshape(state_shape_x),
                        drive_x_imag.reshape(state_shape_x),
                        decay_x_real.reshape(state_shape_x),
                        decay_x_imag.reshape(state_shape_x),
                        magnitude_gamma_x.reshape(state_shape_x),
                        magnitude_decay_x.reshape(state_shape_x),
                        direction=direction_x,
                        backend=self.recurrence_backend,
                    )
                )
                variance_x[direction_x] = current_variance_x.reshape(batch, height, width, modes)
            else:
                horizontal_real, horizontal_imag = _axis_scan_dynamic(
                    drive_x_real.reshape(state_shape_x),
                    drive_x_imag.reshape(state_shape_x),
                    decay_x_real.reshape(state_shape_x),
                    decay_x_imag.reshape(state_shape_x),
                    direction=direction_x,
                    backend=self.recurrence_backend,
                )
            horizontal[direction_x] = (
                horizontal_real.reshape(batch, height, width, modes),
                horizontal_imag.reshape(batch, height, width, modes),
            )
            horizontal_poles[direction_x] = (
                decay_x_real,
                decay_x_imag,
                gamma_x_real,
                gamma_x_imag,
            )
            if self.dynamic_gain_normalization and not self.fuse_state_variance_recurrence:
                magnitude_decay_x = decay_x_real.square().add(decay_x_imag.square())
                magnitude_gamma_x = gamma_x_real.square().add(gamma_x_imag.square())
                current_variance_x, _ = _axis_scan_dynamic(
                    magnitude_gamma_x.reshape(batch * height, width, modes),
                    torch.zeros_like(magnitude_gamma_x).reshape(batch * height, width, modes),
                    magnitude_decay_x.reshape(batch * height, width, modes),
                    torch.zeros_like(magnitude_decay_x).reshape(batch * height, width, modes),
                    direction=direction_x,
                    backend=self.recurrence_backend,
                )
                variance_x[direction_x] = current_variance_x.reshape(batch, height, width, modes)

        x_directions = (1, -1)
        quadrants: list[DirectionalScan] = []
        for direction_y in (1, -1):
            frequency_y = (direction_y * self.phase_y / spacing_y).view(1, 1, 1, -1)
            decay_y_real, decay_y_imag, gamma_y_real, gamma_y_imag = discrete_pole_real2d(
                fine_damping_y / spacing_y,
                frequency_y.expand_as(fine_damping_y),
                spacing_y,
            )
            horizontal_real = torch.stack(
                [horizontal[direction_x][0] for direction_x in x_directions]
            )
            horizontal_imag = torch.stack(
                [horizontal[direction_x][1] for direction_x in x_directions]
            )
            drive_y_real = (
                gamma_y_real.unsqueeze(0) * horizontal_real
                - gamma_y_imag.unsqueeze(0) * horizontal_imag
            )
            drive_y_imag = (
                gamma_y_real.unsqueeze(0) * horizontal_imag
                + gamma_y_imag.unsqueeze(0) * horizontal_real
            )
            vertical_shape = (2 * batch * width, height, modes)
            decay_y_real_batched = (
                decay_y_real.unsqueeze(0)
                .expand(2, -1, -1, -1, -1)
                .permute(0, 1, 3, 2, 4)
                .reshape(vertical_shape)
            )
            decay_y_imag_batched = (
                decay_y_imag.unsqueeze(0)
                .expand(2, -1, -1, -1, -1)
                .permute(0, 1, 3, 2, 4)
                .reshape(vertical_shape)
            )
            vertical_drive_real = drive_y_real.permute(0, 1, 3, 2, 4).reshape(vertical_shape)
            vertical_drive_imag = drive_y_imag.permute(0, 1, 3, 2, 4).reshape(vertical_shape)
            variance = vertical_drive_real
            if self.fuse_state_variance_recurrence:
                current_variance_x = torch.stack(
                    [variance_x[direction_x] for direction_x in x_directions]
                )
                magnitude_gamma_y = gamma_y_real.square().add(gamma_y_imag.square())
                variance_drive = magnitude_gamma_y.unsqueeze(0) * current_variance_x
                magnitude_decay_y = decay_y_real.square().add(decay_y_imag.square())
                magnitude_decay_y_batched = (
                    magnitude_decay_y.unsqueeze(0)
                    .expand(2, -1, -1, -1, -1)
                    .permute(0, 1, 3, 2, 4)
                    .reshape(vertical_shape)
                )
                state_real, state_imag, variance = _axis_scan_state_variance_dynamic(
                    vertical_drive_real,
                    vertical_drive_imag,
                    decay_y_real_batched,
                    decay_y_imag_batched,
                    variance_drive.permute(0, 1, 3, 2, 4).reshape(vertical_shape),
                    magnitude_decay_y_batched,
                    direction=direction_y,
                    backend=self.recurrence_backend,
                )
            else:
                state_real, state_imag = _axis_scan_dynamic(
                    vertical_drive_real,
                    vertical_drive_imag,
                    decay_y_real_batched,
                    decay_y_imag_batched,
                    direction=direction_y,
                    backend=self.recurrence_backend,
                )
            state_real = state_real.reshape(2, batch, width, height, modes).permute(0, 1, 3, 2, 4)
            state_imag = state_imag.reshape(2, batch, width, height, modes).permute(0, 1, 3, 2, 4)
            if self.fuse_state_variance_recurrence:
                variance = variance.reshape(2, batch, width, height, modes).permute(0, 1, 3, 2, 4)
            elif self.dynamic_gain_normalization:
                current_variance_x = torch.stack(
                    [variance_x[direction_x] for direction_x in x_directions]
                )
                magnitude_gamma_y = gamma_y_real.square().add(gamma_y_imag.square())
                variance_drive = magnitude_gamma_y.unsqueeze(0) * current_variance_x
                magnitude_decay_y = decay_y_real.square().add(decay_y_imag.square())
                magnitude_decay_y_batched = (
                    magnitude_decay_y.unsqueeze(0)
                    .expand(2, -1, -1, -1, -1)
                    .permute(0, 1, 3, 2, 4)
                    .reshape(vertical_shape)
                )
                variance, _ = _axis_scan_dynamic(
                    variance_drive.permute(0, 1, 3, 2, 4).reshape(vertical_shape),
                    torch.zeros_like(variance_drive).permute(0, 1, 3, 2, 4).reshape(vertical_shape),
                    magnitude_decay_y_batched,
                    torch.zeros_like(magnitude_decay_y_batched),
                    direction=direction_y,
                    backend=self.recurrence_backend,
                )
                variance = variance.reshape(2, batch, width, height, modes).permute(0, 1, 3, 2, 4)
            else:
                variance = torch.ones_like(state_real)
            for index, direction_x in enumerate(x_directions):
                decay_x_real, decay_x_imag, gamma_x_real, gamma_x_imag = horizontal_poles[
                    direction_x
                ]
                innovation_real, innovation_imag = self._block_innovation(
                    excitation_real,
                    excitation_imag,
                    decay_x_real,
                    decay_x_imag,
                    gamma_x_real,
                    gamma_x_imag,
                    decay_y_real,
                    decay_y_imag,
                    gamma_y_real,
                    gamma_y_imag,
                    direction_x=direction_x,
                    direction_y=direction_y,
                )
                quadrants.append(
                    (
                        state_real[index],
                        state_imag[index],
                        variance[index],
                        innovation_real,
                        innovation_imag,
                    )
                )
        return quadrants

    @staticmethod
    def _complex_multiply(
        left_real: Tensor,
        left_imag: Tensor,
        right_real: Tensor,
        right_imag: Tensor,
    ) -> ComplexField:
        return (
            left_real * right_real - left_imag * right_imag,
            left_real * right_imag + left_imag * right_real,
        )

    @classmethod
    def _block_innovation(
        cls,
        excitation_real: Tensor,
        excitation_imag: Tensor,
        decay_x_real: Tensor,
        decay_x_imag: Tensor,
        gamma_x_real: Tensor,
        gamma_x_imag: Tensor,
        decay_y_real: Tensor,
        decay_y_imag: Tensor,
        gamma_y_real: Tensor,
        gamma_y_imag: Tensor,
        *,
        direction_x: int,
        direction_y: int,
    ) -> ComplexField:
        """Return the exact contribution injected by each current 2x2 block."""
        batch, height, width, modes = excitation_real.shape
        real_blocks = excitation_real.reshape(batch, height // 2, 2, width // 2, 2, modes)
        imag_blocks = excitation_imag.reshape(batch, height // 2, 2, width // 2, 2, modes)
        if direction_y == -1:
            real_blocks = real_blocks.flip(2)
            imag_blocks = imag_blocks.flip(2)
        if direction_x == -1:
            real_blocks = real_blocks.flip(4)
            imag_blocks = imag_blocks.flip(4)

        coefficient_slice = (slice(None), slice(None, None, 2), slice(None, None, 2))
        px_real = decay_x_real[coefficient_slice]
        px_imag = decay_x_imag[coefficient_slice]
        gx_real = gamma_x_real[coefficient_slice]
        gx_imag = gamma_x_imag[coefficient_slice]
        py_real = decay_y_real[coefficient_slice]
        py_imag = decay_y_imag[coefficient_slice]
        gy_real = gamma_y_real[coefficient_slice]
        gy_imag = gamma_y_imag[coefficient_slice]

        horizontal: list[ComplexField] = []
        for local_y in (0, 1):
            earlier_real = real_blocks[:, :, local_y, :, 0]
            earlier_imag = imag_blocks[:, :, local_y, :, 0]
            later_real = real_blocks[:, :, local_y, :, 1]
            later_imag = imag_blocks[:, :, local_y, :, 1]
            weighted_real, weighted_imag = cls._complex_multiply(
                px_real,
                px_imag,
                earlier_real,
                earlier_imag,
            )
            horizontal.append(
                cls._complex_multiply(
                    gx_real,
                    gx_imag,
                    weighted_real + later_real,
                    weighted_imag + later_imag,
                )
            )

        weighted_real, weighted_imag = cls._complex_multiply(
            py_real,
            py_imag,
            horizontal[0][0],
            horizontal[0][1],
        )
        return cls._complex_multiply(
            gy_real,
            gy_imag,
            weighted_real + horizontal[1][0],
            weighted_imag + horizontal[1][1],
        )

    def tcir_innovation_multiplier(self) -> Tensor | None:
        """Return the bounded mode-wise innovation multiplier, if enabled."""
        if self.tcir_innovation_logits is None:
            return None
        return 1.0 + self.tcir_radius * torch.tanh(self.tcir_innovation_logits)

    @staticmethod
    def space_to_depth(inputs: Tensor) -> Tensor:
        batch, height, width, channels = inputs.shape
        if height % 2 or width % 2:
            message = "complement residual requires even spatial dimensions"
            raise ValueError(message)
        blocks = inputs.reshape(batch, height // 2, 2, width // 2, 2, channels)
        return blocks.permute(0, 1, 3, 2, 4, 5).reshape(
            batch,
            height // 2,
            width // 2,
            4 * channels,
        )

    def _residual_coordinates(self, normalized: Tensor, modal_frame: Tensor) -> Tensor:
        if self.complement_residual == "full":
            return normalized
        projector_frame = modal_frame.detach() if self.complement_detach_projector else modal_frame
        frozen_frame = modal_frame.detach()
        # CUDA QR has no BF16 implementation.  The basis is also a numerical
        # coordinate construction rather than a learned mixed-precision matmul,
        # so materialize it in FP32 and cast only the completed basis back to the
        # active feature dtype.
        complement_basis = (
            torch.linalg.qr(frozen_frame.float().mT, mode="complete")
            .Q[:, frozen_frame.shape[0] :]
            .to(dtype=normalized.dtype)
        )
        projected = functional.linear(normalized, projector_frame)
        complement = normalized - torch.matmul(projected, projector_frame)
        complement_coordinates = torch.matmul(complement, complement_basis)
        if self.modal_carry_projection is None:
            return complement_coordinates
        # The local modal branch follows the preregistered stop-gradient frame:
        # it learns which modal coordinates to carry without changing A through
        # this residual path.  Linear.weight is B^T, so its orthonormal rows are
        # equivalent to the column-orthonormal B in C_parallel B.
        modal_coordinates = functional.linear(normalized, frozen_frame)
        carried = self.modal_carry_projection(modal_coordinates)
        return torch.cat((complement_coordinates, carried), dim=-1)

    def _complement_residual(self, normalized: Tensor, modal_frame: Tensor) -> Tensor | None:
        if self.residual_projection is None:
            return None
        coordinates = self._residual_coordinates(normalized, modal_frame)
        return self.residual_projection(self.space_to_depth(coordinates))

    def set_branch_mask(self, mask: TransportBranchMask) -> None:
        if mask not in {"combined", "pole_only", "residual_only"}:
            message = f"unsupported transport branch mask: {mask}"
            raise ValueError(message)
        if mask == "residual_only" and self.residual_projection is None:
            message = "residual-only masking requires a residual branch"
            raise ValueError(message)
        self.branch_mask = mask

    def _merge_transport_branches(
        self,
        pole: Tensor,
        residual: Tensor | None,
    ) -> Tensor:
        if residual is None or self.branch_mask == "pole_only":
            return pole
        if self.residual_scale is None:
            message = "residual projection requires a layer scale"
            raise RuntimeError(message)
        scaled_residual = self.residual_scale * residual
        if self.branch_mask == "residual_only":
            return scaled_residual
        return pole + scaled_residual

    @staticmethod
    def _synthesize(real: Tensor, imag: Tensor, modal_frame: Tensor) -> Tensor:
        frame_real, frame_imag = modal_frame.chunk(2, dim=0)
        return torch.matmul(real, frame_real) + torch.matmul(imag, frame_imag)

    def directional_readout_states(
        self,
        states: list[ComplexField],
    ) -> list[ComplexField]:
        return states

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        normalized = self.input_norm(inputs)
        modal_frame = self.analysis.weight
        excitation_real, excitation_imag = functional.linear(normalized, modal_frame).chunk(
            2, dim=-1
        )
        residual = self._complement_residual(normalized, modal_frame)
        descriptors = []
        synthesized = []
        if self.transport == "average":
            energy = (excitation_real.float().square() + excitation_imag.float().square()).mean(
                (1, 2)
            )
            coarse_real = functional.avg_pool2d(excitation_real.permute(0, 3, 1, 2), 2, 2).permute(
                0, 2, 3, 1
            )
            coarse_imag = functional.avg_pool2d(excitation_imag.permute(0, 3, 1, 2), 2, 2).permute(
                0, 2, 3, 1
            )
            synthesis = self._synthesize(coarse_real, coarse_imag, modal_frame)
            for _ in _DIRECTIONS:
                descriptors.append(torch.log1p(energy))
                synthesized.append(synthesis)
            coarse = self.direction_mix(torch.cat(synthesized, dim=-1))
            coarse = self._merge_transport_branches(coarse, residual)
            output = coarse + self.mlp_scale * self.mlp(self.mlp_norm(coarse))
            return output, torch.cat(descriptors, dim=-1)
        damping_x, damping_y = self._damping_fields(normalized)
        carry = (
            torch.sigmoid(self.gate_sharpness * (math.pi / 2.0 - self.phase_x.abs()))
            * torch.sigmoid(self.gate_sharpness * (math.pi / 2.0 - self.phase_y.abs()))
        ).view(1, 1, 1, -1)
        quadrant_states = (
            self._scan_quadrants(excitation_real, excitation_imag, damping_x, damping_y)
            if self.quadrant_scan_fusion
            else [
                self._scan_direction(
                    excitation_real,
                    excitation_imag,
                    damping_x,
                    damping_y,
                    direction_x=direction_x,
                    direction_y=direction_y,
                )
                for direction_x, direction_y in _DIRECTIONS
            ]
        )
        normalized_states: list[ComplexField] = []
        for (direction_x, direction_y), (
            real,
            imag,
            variance,
            innovation_real,
            innovation_imag,
        ) in zip(_DIRECTIONS, quadrant_states, strict=True):
            gain_variance = variance.detach() if self.stop_gradient_gain_normalization else variance
            inverse_gain = torch.rsqrt(gain_variance.clamp_min(1.0e-8))
            normalized_real = real * inverse_gain
            normalized_imag = imag * inverse_gain
            normalized_states.append((normalized_real, normalized_imag))
            coarse_real, coarse_imag = direction_aligned_endpoints(
                normalized_real,
                normalized_imag,
                direction_x=direction_x,
                direction_y=direction_y,
            )
            innovation_inverse_gain, _ = direction_aligned_endpoints(
                inverse_gain,
                inverse_gain,
                direction_x=direction_x,
                direction_y=direction_y,
            )
            multiplier = self.tcir_innovation_multiplier()
            if multiplier is not None:
                normalized_innovation_real = innovation_real * innovation_inverse_gain
                normalized_innovation_imag = innovation_imag * innovation_inverse_gain
                delta = (multiplier - 1.0).view(1, 1, 1, -1)
                coarse_real = coarse_real + delta * normalized_innovation_real
                coarse_imag = coarse_imag + delta * normalized_innovation_imag
            synthesized.append(
                self._synthesize(
                    carry * coarse_real,
                    carry * coarse_imag,
                    modal_frame,
                )
            )
        for normalized_real, normalized_imag in self.directional_readout_states(normalized_states):
            energy = (normalized_real.float().square() + normalized_imag.float().square()).mean(
                (1, 2)
            )
            descriptors.append(torch.log1p(energy))
        coarse = self.direction_mix(torch.cat(synthesized, dim=-1))
        coarse = self._merge_transport_branches(coarse, residual)
        output = coarse + self.mlp_scale * self.mlp(self.mlp_norm(coarse))
        return output, torch.cat(descriptors, dim=-1)


class ResidualOnlyDown2D(nn.Module):
    """Independently trained S2D path with no early pole recurrence."""

    def __init__(
        self,
        input_width: int,
        output_width: int,
        *,
        layer_scale_init: float,
    ) -> None:
        super().__init__()
        self.input_norm = nn.RMSNorm(input_width)
        self.residual_projection = nn.Linear(4 * input_width, output_width)
        self.mlp_norm = nn.RMSNorm(output_width)
        self.mlp = nn.Sequential(
            nn.Linear(output_width, 2 * output_width),
            nn.SiLU(),
            nn.Linear(2 * output_width, output_width),
        )
        self.mlp_scale = nn.Parameter(torch.full((output_width,), layer_scale_init))

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        normalized = self.input_norm(inputs)
        coarse = self.residual_projection(PoleDown2D.space_to_depth(normalized))
        output = coarse + self.mlp_scale * self.mlp(self.mlp_norm(coarse))
        empty_descriptor = inputs.new_empty((inputs.shape[0], 0))
        return output, empty_descriptor


class TerminalProductPoleBank(PoleDown2D):
    """Static product-pole analysis that emits energy without a coarse feature map."""

    def __init__(
        self,
        input_width: int,
        modes: int,
        *,
        maximum_phase: float,
        recurrence_backend: RecurrenceBackend,
        dynamic_gain_normalization: bool,
        damping_min: float,
        damping_max: float,
        grid_size: int = 1,
        transport: PyramidTransport = "pole",
        stop_gradient_gain_normalization: bool = False,
        quadrant_scan_fusion: bool = True,
        fuse_state_variance_recurrence: bool = False,
    ) -> None:
        nn.Module.__init__(self)
        self.modes = modes
        self.recurrence_backend: RecurrenceBackend = recurrence_backend
        self.dynamic_gain_normalization = dynamic_gain_normalization
        self.damping_min = damping_min
        self.damping_max = damping_max
        self.grid_size = grid_size
        self.transport = transport
        self.stop_gradient_gain_normalization = stop_gradient_gain_normalization
        self.quadrant_scan_fusion = quadrant_scan_fusion
        self.fuse_state_variance_recurrence = fuse_state_variance_recurrence
        if fuse_state_variance_recurrence and not (
            dynamic_gain_normalization and stop_gradient_gain_normalization and quadrant_scan_fusion
        ):
            message = (
                "state/variance fusion requires dynamic stop-gradient gain "
                "normalization and quadrant scan fusion"
            )
            raise ValueError(message)
        self.input_norm = nn.RMSNorm(input_width)
        self.analysis = nn.Linear(input_width, 2 * modes, bias=False)
        nn.init.orthogonal_(self.analysis.weight)
        orthogonal(self.analysis, "weight", orthogonal_map="matrix_exp", use_trivialization=True)
        base_damping = torch.logspace(math.log10(0.04), math.log10(0.35), modes)
        ratio = ((base_damping - damping_min) / (damping_max - damping_min)).clamp(
            1.0e-4, 1.0 - 1.0e-4
        )
        base_logits = torch.logit(ratio)
        self.damping_logits_x = nn.Parameter(base_logits.clone())
        self.damping_logits_y = nn.Parameter(base_logits.clone())
        phase_x, phase_y = _phase_atlas(modes, maximum_phase)
        self.phase_x = nn.Parameter(phase_x)
        self.phase_y = nn.Parameter(phase_y)

    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, inputs: Tensor
    ) -> Tensor:
        normalized = self.input_norm(inputs)
        excitation_real, excitation_imag = self.analysis(normalized).chunk(2, dim=-1)
        if self.transport == "average":
            energy_map = excitation_real.float().square() + excitation_imag.float().square()
            descriptor = self._regional_energy(energy_map)
            return torch.cat([descriptor for _ in _DIRECTIONS], dim=-1)
        damping_x, damping_y = self._damping_fields(normalized)
        descriptors = []
        quadrant_states = (
            self._scan_quadrants(excitation_real, excitation_imag, damping_x, damping_y)
            if self.quadrant_scan_fusion
            else [
                self._scan_direction(
                    excitation_real,
                    excitation_imag,
                    damping_x,
                    damping_y,
                    direction_x=direction_x,
                    direction_y=direction_y,
                )
                for direction_x, direction_y in _DIRECTIONS
            ]
        )
        normalized_states: list[ComplexField] = []
        for real, imag, variance, _, _ in quadrant_states:
            gain_variance = variance.detach() if self.stop_gradient_gain_normalization else variance
            inverse_gain = torch.rsqrt(gain_variance.clamp_min(1.0e-8))
            normalized_states.append((real * inverse_gain, imag * inverse_gain))
        for normalized_real, normalized_imag in self.directional_readout_states(normalized_states):
            energy_map = normalized_real.float().square() + normalized_imag.float().square()
            descriptors.append(self._regional_energy(energy_map))
        return torch.cat(descriptors, dim=-1)

    def _regional_energy(self, energy_map: Tensor) -> Tensor:
        pooled = functional.adaptive_avg_pool2d(
            energy_map.permute(0, 3, 1, 2), (self.grid_size, self.grid_size)
        )
        return torch.log1p(pooled.permute(0, 2, 3, 1).flatten(1))


@dataclass(frozen=True, slots=True)
class PolePyramidATinyConfig:
    output_dim: int = 100
    image_size: int = 32
    widths: tuple[int, ...] = (64, 96, 128, 192)
    modes: tuple[int, ...] = (16, 24, 32)
    dynamic_gain_normalization: bool = True
    damping_min: float = 0.01
    damping_max: float = 0.7
    gate_sharpness: float = 8.0
    layer_scale_init: float = 1.0e-3
    recurrence_backend: RecurrenceBackend = "auto"
    transport: PyramidTransport = "pole"
    stop_gradient_gain_normalization: bool = False
    quadrant_scan_fusion: bool = True
    fuse_state_variance_recurrence: bool = False

    def validate(self) -> None:
        if self.image_size != 32:
            message = "PolePyramid-A-Tiny is fixed to 32x32 CIFAR input"
            raise ValueError(message)
        if len(self.widths) != 4 or len(self.modes) != 3:
            message = "PolePyramid-A-Tiny requires three PoleDown stages"
            raise ValueError(message)
        if any(
            2 * modes > width for width, modes in zip(self.widths[:-1], self.modes, strict=True)
        ):
            message = "each stage must satisfy 2M <= D"
            raise ValueError(message)
        if self.transport not in {"pole", "average"}:
            message = f"unsupported transport: {self.transport}"
            raise ValueError(message)
        if self.fuse_state_variance_recurrence and not (
            self.dynamic_gain_normalization
            and self.stop_gradient_gain_normalization
            and self.quadrant_scan_fusion
        ):
            message = (
                "state/variance fusion requires dynamic stop-gradient gain "
                "normalization and quadrant scan fusion"
            )
            raise ValueError(message)


class PolePyramidATiny(nn.Module):
    """Three-stage static-pole pyramid with a 288-coordinate affine head."""

    downs: nn.ModuleList

    def __init__(self, config: PolePyramidATinyConfig | None = None) -> None:
        super().__init__()
        active = config or PolePyramidATinyConfig()
        active.validate()
        self.config = active
        self.stem = CifarConvStem(active.widths[0])
        maximum_phases = (math.pi * 0.75, math.pi * 0.70, math.pi * 0.65)
        self.downs = nn.ModuleList(
            [
                PoleDown2D(
                    input_width,
                    output_width,
                    modes,
                    maximum_phase=maximum_phase,
                    recurrence_backend=active.recurrence_backend,
                    dynamic_gain_normalization=active.dynamic_gain_normalization,
                    damping_min=active.damping_min,
                    damping_max=active.damping_max,
                    gate_sharpness=active.gate_sharpness,
                    layer_scale_init=active.layer_scale_init,
                    transport=active.transport,
                    stop_gradient_gain_normalization=active.stop_gradient_gain_normalization,
                    quadrant_scan_fusion=active.quadrant_scan_fusion,
                    fuse_state_variance_recurrence=active.fuse_state_variance_recurrence,
                )
                for input_width, output_width, modes, maximum_phase in zip(
                    active.widths[:-1],
                    active.widths[1:],
                    active.modes,
                    maximum_phases,
                    strict=True,
                )
            ]
        )
        self.descriptor_dim = 4 * sum(active.modes)
        self.classifier = nn.Linear(self.descriptor_dim, active.output_dim)

    def raw_descriptor(self, inputs: Tensor) -> Tensor:
        features = self.stem(inputs)
        descriptors = []
        for down in self.downs:
            features, descriptor = down(features)
            descriptors.append(descriptor)
        return torch.cat(descriptors, dim=-1)

    def forward(self, inputs: Tensor) -> Tensor:
        descriptor = self.raw_descriptor(inputs)
        with torch.autocast(device_type=descriptor.device.type, enabled=False):
            return self.classifier(descriptor.float())


@dataclass(frozen=True, slots=True)
class PolePyramidATerminalTinyConfig:
    output_dim: int = 100
    image_size: int = 32
    stem_strides: tuple[int, int] = (1, 1)
    widths: tuple[int, ...] = (64, 96, 128)
    modes: tuple[int, ...] = (16, 24, 32)
    dynamic_gain_normalization: bool = True
    damping_min: float = 0.01
    damping_max: float = 0.7
    gate_sharpness: float = 8.0
    layer_scale_init: float = 1.0e-3
    recurrence_backend: RecurrenceBackend = "auto"
    terminal_grid_size: int = 1
    transport: PyramidTransport = "pole"
    stop_gradient_gain_normalization: bool = False
    quadrant_scan_fusion: bool = True
    fuse_state_variance_recurrence: bool = False
    quadratic_rank: int = 0
    standardize_descriptor: bool = False
    complement_residual: ComplementResidual = "none"
    complement_detach_projector: bool = True
    complement_scale_init: float = 1.0e-2
    modal_carry_ranks: tuple[int, ...] = (0, 0)
    modal_carry_learned: bool = True
    residual_only_early: bool = False
    tcir_innovation_reweighting: bool = False
    tcir_radius: float = 0.5

    def validate(self) -> None:  # noqa: C901, PLR0912, PLR0915
        if self.image_size < 32:
            message = "PolePyramid-A-Terminal-Tiny requires images of at least 32x32"
            raise ValueError(message)
        if len(self.stem_strides) != 2 or any(stride not in {1, 2} for stride in self.stem_strides):
            message = "stem strides must contain two values chosen from 1 or 2"
            raise ValueError(message)
        total_reduction = math.prod(self.stem_strides) * 4
        if self.image_size % total_reduction:
            message = "image size must be divisible by the stem and PoleDown reduction"
            raise ValueError(message)
        if len(self.widths) != 3 or len(self.modes) != 3:
            message = "PolePyramid-A-Terminal-Tiny requires two PoleDowns and one terminal bank"
            raise ValueError(message)
        if any(2 * modes > width for width, modes in zip(self.widths, self.modes, strict=True)):
            message = "each stage must satisfy 2M <= D"
            raise ValueError(message)
        if self.terminal_grid_size not in {1, 2, 4}:
            message = "terminal grid size must be one of 1, 2, or 4"
            raise ValueError(message)
        if self.transport not in {"pole", "average"}:
            message = f"unsupported transport: {self.transport}"
            raise ValueError(message)
        if self.quadratic_rank < 0:
            message = "quadratic rank cannot be negative"
            raise ValueError(message)
        if self.quadratic_rank and self.standardize_descriptor:
            message = "LRQ already standardizes the descriptor"
            raise ValueError(message)
        if self.complement_residual not in {"none", "full", "orthogonal"}:
            message = f"unsupported complement residual: {self.complement_residual}"
            raise ValueError(message)
        if self.complement_residual != "none" and self.transport != "pole":
            message = "complement residual requires pole transport"
            raise ValueError(message)
        if len(self.modal_carry_ranks) != len(self.widths) - 1:
            message = "modal carry ranks must provide one value per PoleDown"
            raise ValueError(message)
        if any(
            not 0 <= rank <= 2 * modes
            for rank, modes in zip(
                self.modal_carry_ranks,
                self.modes[:-1],
                strict=True,
            )
        ):
            message = "each modal carry rank must be in [0, 2M]"
            raise ValueError(message)
        if any(self.modal_carry_ranks) and self.complement_residual != "orthogonal":
            message = "modal carry requires the orthogonal-complement residual"
            raise ValueError(message)
        if self.residual_only_early and self.complement_residual != "full":
            message = "residual-only early transport requires the full S2D residual"
            raise ValueError(message)
        if self.residual_only_early and any(self.modal_carry_ranks):
            message = "residual-only early transport does not use modal carry ranks"
            raise ValueError(message)
        if self.complement_scale_init < 0.0:
            message = "complement residual scale cannot be negative"
            raise ValueError(message)
        if self.tcir_innovation_reweighting and self.transport != "pole":
            message = "TCIR requires pole transport"
            raise ValueError(message)
        if self.fuse_state_variance_recurrence and not (
            self.dynamic_gain_normalization
            and self.stop_gradient_gain_normalization
            and self.quadrant_scan_fusion
        ):
            message = (
                "state/variance fusion requires dynamic stop-gradient gain "
                "normalization and quadrant scan fusion"
            )
            raise ValueError(message)
        if not 0.0 < self.tcir_radius <= 1.0:
            message = "TCIR radius must be in (0, 1]"
            raise ValueError(message)


class PolePyramidATerminalTiny(nn.Module):
    """Two static PoleDowns followed by an analysis-only terminal pole bank."""

    downs: nn.ModuleList

    def __init__(self, config: PolePyramidATerminalTinyConfig | None = None) -> None:
        super().__init__()
        active = config or PolePyramidATerminalTinyConfig()
        active.validate()
        self.config = active
        self.stem = CifarConvStem(active.widths[0], active.stem_strides)
        maximum_phases = (math.pi * 0.75, math.pi * 0.70, math.pi * 0.65)
        if active.residual_only_early:
            self.downs = nn.ModuleList(
                [
                    ResidualOnlyDown2D(
                        input_width,
                        output_width,
                        layer_scale_init=active.layer_scale_init,
                    )
                    for input_width, output_width in zip(
                        active.widths[:-1],
                        active.widths[1:],
                        strict=True,
                    )
                ]
            )
        else:
            self.downs = nn.ModuleList(
                [
                    PoleDown2D(
                        input_width,
                        output_width,
                        modes,
                        maximum_phase=maximum_phase,
                        recurrence_backend=active.recurrence_backend,
                        dynamic_gain_normalization=active.dynamic_gain_normalization,
                        damping_min=active.damping_min,
                        damping_max=active.damping_max,
                        gate_sharpness=active.gate_sharpness,
                        layer_scale_init=active.layer_scale_init,
                        transport=active.transport,
                        stop_gradient_gain_normalization=active.stop_gradient_gain_normalization,
                        quadrant_scan_fusion=active.quadrant_scan_fusion,
                        fuse_state_variance_recurrence=active.fuse_state_variance_recurrence,
                        complement_residual=active.complement_residual,
                        complement_detach_projector=active.complement_detach_projector,
                        complement_scale_init=active.complement_scale_init,
                        modal_carry_rank=modal_carry_rank,
                        modal_carry_learned=active.modal_carry_learned,
                        tcir_innovation_reweighting=active.tcir_innovation_reweighting,
                        tcir_radius=active.tcir_radius,
                    )
                    for (
                        input_width,
                        output_width,
                        modes,
                        maximum_phase,
                        modal_carry_rank,
                    ) in zip(
                        active.widths[:-1],
                        active.widths[1:],
                        active.modes[:-1],
                        maximum_phases[:-1],
                        active.modal_carry_ranks,
                        strict=True,
                    )
                ]
            )
        self.terminal = TerminalProductPoleBank(
            active.widths[-1],
            active.modes[-1],
            maximum_phase=maximum_phases[-1],
            recurrence_backend=active.recurrence_backend,
            dynamic_gain_normalization=active.dynamic_gain_normalization,
            damping_min=active.damping_min,
            damping_max=active.damping_max,
            grid_size=active.terminal_grid_size,
            transport=active.transport,
            stop_gradient_gain_normalization=active.stop_gradient_gain_normalization,
            quadrant_scan_fusion=active.quadrant_scan_fusion,
            fuse_state_variance_recurrence=active.fuse_state_variance_recurrence,
        )
        early_descriptor_dim = 0 if active.residual_only_early else 4 * sum(active.modes[:-1])
        self.descriptor_dim = early_descriptor_dim + (
            4 * active.modes[-1] * active.terminal_grid_size**2
        )
        self.classifier: nn.Module
        if active.quadratic_rank:
            self.classifier = LowRankQuadraticModalHead(
                self.descriptor_dim,
                active.output_dim,
                active.quadratic_rank,
            )
        elif active.standardize_descriptor:
            self.classifier = StandardizedAffineModalHead(
                self.descriptor_dim,
                active.output_dim,
            )
        else:
            self.classifier = nn.Linear(self.descriptor_dim, active.output_dim)

    def set_transport_branch_mask(self, mask: TransportBranchMask) -> None:
        for module in self.downs:
            down = cast("PoleDown2D", module)
            down.set_branch_mask(mask)

    def transport_features(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        features = self.stem(inputs)
        descriptors = []
        for down in self.downs:
            features, descriptor = down(features)
            descriptors.append(descriptor)
        return features, torch.cat(descriptors, dim=-1)

    def raw_descriptor(self, inputs: Tensor) -> Tensor:
        features, early_descriptor = self.transport_features(inputs)
        descriptors = [early_descriptor]
        descriptors.append(self.terminal(features))
        return torch.cat(descriptors, dim=-1)

    def forward(self, inputs: Tensor) -> Tensor:
        descriptor = self.raw_descriptor(inputs)
        with torch.autocast(device_type=descriptor.device.type, enabled=False):
            return self.classifier(descriptor.float())


__all__ = [
    "ComplementResidual",
    "PoleDown2D",
    "PolePyramidATerminalTiny",
    "PolePyramidATerminalTinyConfig",
    "PolePyramidATiny",
    "PolePyramidATinyConfig",
    "PyramidTransport",
    "TerminalProductPoleBank",
    "TransportBranchMask",
]
