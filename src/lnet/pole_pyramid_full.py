"""Full PolePyramid with fine-detail readout and shared physical poles."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import torch
from torch import Tensor, nn
from torch.nn import functional

from .alphabet2d import product_pole_scan_2d
from .pac_directional import direction_aligned_endpoints
from .pac_real2d_math import discrete_pole_real2d

if TYPE_CHECKING:
    from .pac_recurrence import RecurrenceBackend

FullPyramidTransport = Literal["pole", "average"]
_QUADRANTS = ((1, 1), (-1, 1), (1, -1), (-1, -1))
_LAGS = ((1, 0), (0, 1), (1, 1), (1, -1))


def _radial_log_complex(real: Tensor, imag: Tensor) -> tuple[Tensor, Tensor]:
    magnitude = torch.sqrt(real.square() + imag.square())
    scale = torch.log1p(magnitude) / magnitude.clamp_min(torch.finfo(real.dtype).eps)
    return scale * real, scale * imag


def fine_modal_moments(real: Tensor, imag: Tensor) -> Tensor:
    """Return log-energy and four unnormalized complex lag moments."""
    if real.shape != imag.shape or real.ndim != 4:
        message = "modal responses must share [B,H,W,M] shape"
        raise ValueError(message)
    real = real.float()
    imag = imag.float()
    height, width = real.shape[1:3]
    pieces = [torch.log1p((real.square() + imag.square()).mean(dim=(1, 2)))]
    for delta_x, delta_y in _LAGS:
        if abs(delta_x) >= width or abs(delta_y) >= height:
            message = f"lag ({delta_x}, {delta_y}) exceeds {height}x{width} field"
            raise ValueError(message)
        current_y = slice(max(delta_y, 0), height + min(delta_y, 0))
        previous_y = slice(max(-delta_y, 0), height - max(delta_y, 0))
        current_x = slice(max(delta_x, 0), width + min(delta_x, 0))
        previous_x = slice(max(-delta_x, 0), width - max(delta_x, 0))
        current_real = real[:, current_y, current_x]
        current_imag = imag[:, current_y, current_x]
        previous_real = real[:, previous_y, previous_x]
        previous_imag = imag[:, previous_y, previous_x]
        correlation_real = (
            current_real * previous_real + current_imag * previous_imag
        ).mean(dim=(1, 2))
        correlation_imag = (
            current_imag * previous_real - current_real * previous_imag
        ).mean(dim=(1, 2))
        pieces.extend(_radial_log_complex(correlation_real, correlation_imag))
    return torch.cat(pieces, dim=-1)


class SharedPhysicalPoleGeometry(nn.Module):
    """One continuous-space pole atlas reused at every pyramid spacing."""

    def __init__(self, modes: int) -> None:
        super().__init__()
        if modes < 4 or modes % 4:
            message = "shared pole count must be a positive multiple of four"
            raise ValueError(message)
        levels = modes // 4
        radial = torch.logspace(math.log10(math.pi / 16.0), math.log10(math.pi / 2.0), levels)
        radial = radial.repeat_interleave(4)
        orientation = torch.tensor(
            (0.0, math.pi / 4.0, math.pi / 2.0, 3.0 * math.pi / 4.0)
        ).repeat(levels)
        base_frequency_x = radial * torch.cos(orientation)
        base_frequency_y = radial * torch.sin(orientation)
        base_damping = (radial / 2.0).clamp_min(1.0 / 64.0)
        self.register_buffer("base_damping_x", base_damping.clone())
        self.register_buffer("base_damping_y", base_damping.clone())
        self.register_buffer("base_frequency_x", base_frequency_x)
        self.register_buffer("base_frequency_y", base_frequency_y)
        self.log_damping_offset_x = nn.Parameter(torch.zeros(modes))
        self.log_damping_offset_y = nn.Parameter(torch.zeros(modes))
        self.frequency_offset_x = nn.Parameter(torch.zeros(modes))
        self.frequency_offset_y = nn.Parameter(torch.zeros(modes))

    def damping_x(self) -> Tensor:
        return self.get_buffer("base_damping_x") * torch.exp(
            math.log(2.0) * torch.tanh(self.log_damping_offset_x)
        )

    def damping_y(self) -> Tensor:
        return self.get_buffer("base_damping_y") * torch.exp(
            math.log(2.0) * torch.tanh(self.log_damping_offset_y)
        )

    def frequency_x(self) -> Tensor:
        return self.get_buffer("base_frequency_x") + (math.pi / 16.0) * torch.tanh(
            self.frequency_offset_x
        )

    def frequency_y(self) -> Tensor:
        return self.get_buffer("base_frequency_y") + (math.pi / 16.0) * torch.tanh(
            self.frequency_offset_y
        )

    def descriptor_gain(self, spacing: float) -> Tensor:
        """Analytic inverse L2 gain of the separable ZOH pole kernel."""
        decay_x_real, decay_x_imag, gamma_x_real, gamma_x_imag = discrete_pole_real2d(
            self.damping_x(),
            self.frequency_x(),
            spacing,
        )
        decay_y_real, decay_y_imag, gamma_y_real, gamma_y_imag = discrete_pole_real2d(
            self.damping_y(),
            self.frequency_y(),
            spacing,
        )
        decay_energy_x = decay_x_real.square() + decay_x_imag.square()
        decay_energy_y = decay_y_real.square() + decay_y_imag.square()
        gamma_energy_x = gamma_x_real.square() + gamma_x_imag.square()
        gamma_energy_y = gamma_y_real.square() + gamma_y_imag.square()
        numerator = ((1.0 - decay_energy_x) * (1.0 - decay_energy_y)).clamp_min(1.0e-12)
        denominator = (gamma_energy_x * gamma_energy_y).clamp_min(1.0e-12)
        return torch.sqrt(numerator / denominator)


class FullPoleDown2D(nn.Module):
    """Analyze fine detail and carry only exact low-mode block endpoints."""

    def __init__(
        self,
        input_width: int,
        output_width: int,
        modes: int,
        *,
        stage: int,
        geometry: SharedPhysicalPoleGeometry,
        transport: FullPyramidTransport,
        recurrence_backend: RecurrenceBackend,
    ) -> None:
        super().__init__()
        self.modes = modes
        self.stage = stage
        self.geometry = geometry
        self.transport = transport
        self.recurrence_backend: RecurrenceBackend = recurrence_backend
        self.input_norm = nn.RMSNorm(input_width)
        self.analysis = nn.Linear(input_width, 2 * modes, bias=False)
        nn.init.orthogonal_(self.analysis.weight)
        self.point_mix = nn.Linear(4 * 2 * (modes // 2), output_width)

    @property
    def spacing(self) -> float:
        return float(2**self.stage)

    def _pole_direction(
        self,
        real: Tensor,
        imag: Tensor,
        direction_x: int,
        direction_y: int,
    ) -> tuple[Tensor, Tensor]:
        return product_pole_scan_2d(
            real,
            imag,
            damping_x=self.geometry.damping_x(),
            damping_y=self.geometry.damping_y(),
            frequency_x=self.geometry.frequency_x(),
            frequency_y=self.geometry.frequency_y(),
            spacing_x=self.spacing,
            spacing_y=self.spacing,
            direction_x=direction_x,
            direction_y=direction_y,
            recurrence_backend=self.recurrence_backend,
        )

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        excitation_real, excitation_imag = self.analysis(self.input_norm(inputs)).chunk(2, dim=-1)
        descriptors = []
        carried = []
        descriptor_gain = self.geometry.descriptor_gain(self.spacing).view(1, 1, 1, -1)
        for direction_x, direction_y in _QUADRANTS:
            if self.transport == "pole":
                fine_real, fine_imag = self._pole_direction(
                    excitation_real,
                    excitation_imag,
                    direction_x,
                    direction_y,
                )
                detail_real = fine_real * descriptor_gain
                detail_imag = fine_imag * descriptor_gain
                coarse_real, coarse_imag = direction_aligned_endpoints(
                    fine_real,
                    fine_imag,
                    direction_x=direction_x,
                    direction_y=direction_y,
                )
            else:
                detail_real, detail_imag = excitation_real, excitation_imag
                coarse_real = functional.avg_pool2d(
                    excitation_real.permute(0, 3, 1, 2), 2, 2
                ).permute(0, 2, 3, 1)
                coarse_imag = functional.avg_pool2d(
                    excitation_imag.permute(0, 3, 1, 2), 2, 2
                ).permute(0, 2, 3, 1)
            descriptors.append(fine_modal_moments(detail_real, detail_imag))
            low_modes = self.modes // 2
            carried.extend((coarse_real[..., :low_modes], coarse_imag[..., :low_modes]))
        output = functional.silu(self.point_mix(torch.cat(carried, dim=-1)))
        return output, torch.cat(descriptors, dim=-1)


@dataclass(frozen=True, slots=True)
class FullPolePyramidConfig:
    input_channels: int = 3
    output_dim: int = 100
    image_size: int = 32
    widths: tuple[int, ...] = (32, 48, 64, 96, 128)
    modes: int = 16
    stages: int = 4
    transport: FullPyramidTransport = "pole"
    recurrence_backend: RecurrenceBackend = "auto"

    def validate(self) -> None:
        if len(self.widths) != self.stages + 1:
            message = "widths must contain one more entry than stages"
            raise ValueError(message)
        if self.image_size % (2**self.stages):
            message = "image size must be divisible by the pyramid stride"
            raise ValueError(message)
        if self.modes < 4 or self.modes % 4:
            message = "mode count must be a positive multiple of four"
            raise ValueError(message)
        if self.transport not in {"pole", "average"}:
            message = f"unsupported pyramid transport: {self.transport}"
            raise ValueError(message)


class FullPolePyramid(nn.Module):
    """Pole-only image pyramid with fine-scale detail and an affine head."""

    blocks: nn.ModuleList

    def __init__(self, config: FullPolePyramidConfig | None = None) -> None:
        super().__init__()
        active = config or FullPolePyramidConfig()
        active.validate()
        self.config = active
        self.geometry = SharedPhysicalPoleGeometry(active.modes)
        self.input_lift = nn.Linear(active.input_channels, active.widths[0])
        self.blocks = nn.ModuleList(
            [
                FullPoleDown2D(
                    active.widths[stage],
                    active.widths[stage + 1],
                    active.modes,
                    stage=stage,
                    geometry=self.geometry,
                    transport=active.transport,
                    recurrence_backend=active.recurrence_backend,
                )
                for stage in range(active.stages)
            ]
        )
        self.descriptor_dim = active.stages * 4 * 9 * active.modes
        self.classifier = nn.Linear(self.descriptor_dim, active.output_dim)

    def forward_features(self, inputs: Tensor) -> Tensor:
        features = functional.silu(self.input_lift(inputs.permute(0, 2, 3, 1)))
        descriptors = []
        for block in self.blocks:
            features, descriptor = cast("FullPoleDown2D", block)(features)
            descriptors.append(descriptor)
        return torch.cat(descriptors, dim=-1)

    def forward(self, inputs: Tensor) -> Tensor:
        descriptor = self.forward_features(inputs)
        with torch.autocast(device_type=descriptor.device.type, enabled=False):
            return self.classifier(descriptor.float())


__all__ = [
    "FullPoleDown2D",
    "FullPolePyramid",
    "FullPolePyramidConfig",
    "FullPyramidTransport",
    "SharedPhysicalPoleGeometry",
    "fine_modal_moments",
]
