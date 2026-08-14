"""Conv-stem ALPHABET image pyramid with pole residuals and Q/R/C readout."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
from torch import Tensor, nn
from torch.nn import functional

from .alphabet2d import product_pole_scan_2d
from .image_layers import LayerNorm2d
from .pac_directional import direction_aligned_endpoints
from .pole_pyramid_full import SharedPhysicalPoleGeometry
from .spatialalphabet_h import cross_mode_edges, regional_pole_moments

if TYPE_CHECKING:
    from .pac_recurrence import RecurrenceBackend

ConvStemTransport = Literal["pole", "average"]
Direction = tuple[int, int]
ComplexField = tuple[Tensor, Tensor]

_DIRECTIONS: tuple[Direction, ...] = ((1, 1), (-1, 1), (1, -1), (-1, -1))
_LAGS = ((1, 0), (0, 1), (1, 1), (1, -1))


class ImageNetPoleStem(nn.Module):
    """The prescribed Conv-Norm-GELU-Conv-Norm sensory stem."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = channels // 2
        self.first = nn.Conv2d(3, hidden, kernel_size=3, stride=2, padding=1, bias=False)
        self.first_norm = LayerNorm2d(hidden)
        self.activation = nn.GELU()
        self.second = nn.Conv2d(
            hidden,
            channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.second_norm = LayerNorm2d(channels)

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = self.activation(self.first_norm(self.first(inputs)))
        # Deliberately no activation: the pole bank receives signed excitation.
        return self.second_norm(self.second(hidden))


def _imagenet_geometry(modes: int) -> SharedPhysicalPoleGeometry:
    """Initialize one physical pole atlas below the final-stage Nyquist limit."""
    geometry = SharedPhysicalPoleGeometry(modes)
    levels = modes // 4
    radial = torch.logspace(math.log10(math.pi / 128.0), math.log10(math.pi / 32.0), levels)
    radial = radial.repeat_interleave(4)
    orientation = torch.tensor(
        (0.0, math.pi / 4.0, math.pi / 2.0, 3.0 * math.pi / 4.0)
    ).repeat(levels)
    with torch.no_grad():
        geometry.get_buffer("base_damping_x").copy_((radial / 2.0).clamp_min(1.0 / 256.0))
        geometry.get_buffer("base_damping_y").copy_((radial / 2.0).clamp_min(1.0 / 256.0))
        geometry.get_buffer("base_frequency_x").copy_(radial * torch.cos(orientation))
        geometry.get_buffer("base_frequency_y").copy_(radial * torch.sin(orientation))
    return geometry


def _descriptor_size(modes: int) -> int:
    edges = cross_mode_edges(modes)
    # Q plus complex R for each lag plus sparse complex cross-pole C.
    per_direction = modes + 2 * len(_LAGS) * modes + 2 * len(edges)
    return len(_DIRECTIONS) * per_direction


class PoleField2D(nn.Module):
    """Project features and produce four direction-aware complex modal fields."""

    def __init__(
        self,
        width: int,
        modes: int,
        *,
        spacing: float,
        geometry: SharedPhysicalPoleGeometry,
        transport: ConvStemTransport,
        recurrence_backend: RecurrenceBackend,
    ) -> None:
        super().__init__()
        self.modes = modes
        self.spacing = spacing
        self.geometry = geometry
        self.transport = transport
        self.recurrence_backend: RecurrenceBackend = recurrence_backend
        self.input_norm = nn.RMSNorm(width)
        self.analysis = nn.Linear(width, 2 * modes, bias=False)
        nn.init.orthogonal_(self.analysis.weight)

    def forward(self, inputs: Tensor) -> tuple[ComplexField, ...]:
        excitation_real, excitation_imag = self.analysis(self.input_norm(inputs)).chunk(2, dim=-1)
        if self.transport == "average":
            return tuple((excitation_real, excitation_imag) for _ in _DIRECTIONS)
        fields = []
        for direction_x, direction_y in _DIRECTIONS:
            fields.append(
                product_pole_scan_2d(
                    excitation_real,
                    excitation_imag,
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
            )
        return tuple(fields)


class PoleResidualStage2D(nn.Module):
    """Full-resolution modal scan, learned pointwise synthesis, and nonlinear mixing."""

    def __init__(
        self,
        width: int,
        modes: int,
        *,
        spacing: float,
        geometry: SharedPhysicalPoleGeometry,
        transport: ConvStemTransport,
        recurrence_backend: RecurrenceBackend,
        layer_scale_init: float,
    ) -> None:
        super().__init__()
        self.field = PoleField2D(
            width,
            modes,
            spacing=spacing,
            geometry=geometry,
            transport=transport,
            recurrence_backend=recurrence_backend,
        )
        self.synthesis = nn.Linear(8 * modes, width, bias=False)
        self.pole_scale = nn.Parameter(torch.full((width,), layer_scale_init))
        hidden = 2 * width
        self.mix_norm = nn.RMSNorm(width)
        self.pointwise_mix = nn.Sequential(
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Linear(hidden, width),
        )
        self.mix_scale = nn.Parameter(torch.full((width,), layer_scale_init))

    def forward(self, inputs: Tensor) -> Tensor:
        fields = self.field(inputs)
        modal = torch.cat([coordinate for field in fields for coordinate in field], dim=-1)
        features = inputs + self.pole_scale * self.synthesis(modal)
        return features + self.mix_scale * self.pointwise_mix(self.mix_norm(features))


def _qrc_descriptor(
    fields: tuple[ComplexField, ...],
    *,
    gain: Tensor,
    edges: tuple[tuple[int, int], ...],
) -> Tensor:
    descriptors = []
    gain_view = gain.view(1, 1, 1, -1)
    for real, imag in fields:
        descriptors.append(
            regional_pole_moments(
                real * gain_view,
                imag * gain_view,
                lags=_LAGS,
                edges=edges,
                regional=False,
            )
        )
    return torch.cat(descriptors, dim=-1)


class PoleDownQRC2D(nn.Module):
    """Keep fine Q/R/C detail while carrying exact low-mode block endpoints."""

    def __init__(
        self,
        input_width: int,
        output_width: int,
        modes: int,
        *,
        spacing: float,
        geometry: SharedPhysicalPoleGeometry,
        transport: ConvStemTransport,
        recurrence_backend: RecurrenceBackend,
    ) -> None:
        super().__init__()
        self.modes = modes
        self.spacing = spacing
        self.geometry = geometry
        self.transport = transport
        self.edges = cross_mode_edges(modes)
        self.field = PoleField2D(
            input_width,
            modes,
            spacing=spacing,
            geometry=geometry,
            transport=transport,
            recurrence_backend=recurrence_backend,
        )
        self.carry_mix = nn.Linear(4 * 2 * (modes // 2), output_width)
        self.output_norm = nn.RMSNorm(output_width)

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        fields = self.field(inputs)
        descriptor = _qrc_descriptor(
            fields,
            gain=self.geometry.descriptor_gain(self.spacing),
            edges=self.edges,
        )
        carried = []
        low_modes = self.modes // 2
        for (direction_x, direction_y), (real, imag) in zip(
            _DIRECTIONS, fields, strict=True
        ):
            if self.transport == "pole":
                coarse_real, coarse_imag = direction_aligned_endpoints(
                    real,
                    imag,
                    direction_x=direction_x,
                    direction_y=direction_y,
                )
            else:
                coarse_real = functional.avg_pool2d(real.permute(0, 3, 1, 2), 2, 2).permute(
                    0, 2, 3, 1
                )
                coarse_imag = functional.avg_pool2d(imag.permute(0, 3, 1, 2), 2, 2).permute(
                    0, 2, 3, 1
                )
            carried.extend((coarse_real[..., :low_modes], coarse_imag[..., :low_modes]))
        coarse = self.carry_mix(torch.cat(carried, dim=-1))
        return self.output_norm(functional.gelu(coarse)), descriptor


class PoleDescriptorReader2D(nn.Module):
    """Terminal pole bank exposing only conditioned Q/R/C coordinates."""

    def __init__(
        self,
        width: int,
        modes: int,
        *,
        spacing: float,
        geometry: SharedPhysicalPoleGeometry,
        transport: ConvStemTransport,
        recurrence_backend: RecurrenceBackend,
    ) -> None:
        super().__init__()
        self.spacing = spacing
        self.geometry = geometry
        self.edges = cross_mode_edges(modes)
        self.field = PoleField2D(
            width,
            modes,
            spacing=spacing,
            geometry=geometry,
            transport=transport,
            recurrence_backend=recurrence_backend,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return _qrc_descriptor(
            self.field(inputs),
            gain=self.geometry.descriptor_gain(self.spacing),
            edges=self.edges,
        )


@dataclass(frozen=True, slots=True)
class ConvStemPolePyramidConfig:
    output_dim: int = 100
    image_size: int = 224
    stem_width: int = 64
    widths: tuple[int, ...] = (64, 96, 128)
    modes: int = 16
    layer_scale_init: float = 1.0e-2
    transport: ConvStemTransport = "pole"
    recurrence_backend: RecurrenceBackend = "auto"

    def validate(self) -> None:
        if self.image_size != 224:
            message = "the confirmatory ConvStem PolePyramid contract requires 224px input"
            raise ValueError(message)
        if self.stem_width % 2:
            message = "stem width must be even"
            raise ValueError(message)
        if len(self.widths) != 3 or self.widths[0] != self.stem_width:
            message = "widths must define the 56, 28, and 14 grids"
            raise ValueError(message)
        if self.modes < 4 or self.modes % 4:
            message = "mode count must be a positive multiple of four"
            raise ValueError(message)
        if self.transport not in {"pole", "average"}:
            message = f"unsupported transport: {self.transport}"
            raise ValueError(message)


class ConvStemPolePyramid(nn.Module):
    """Three pole-residual scales, two exact PoleDowns, and an affine Q/R/C head."""

    stages: nn.ModuleList
    downs: nn.ModuleList

    def __init__(self, config: ConvStemPolePyramidConfig | None = None) -> None:
        super().__init__()
        active = config or ConvStemPolePyramidConfig()
        active.validate()
        self.config = active
        self.stem = ImageNetPoleStem(active.stem_width)
        self.geometry = _imagenet_geometry(active.modes)
        spacings = (4.0, 8.0, 16.0)
        self.stages = nn.ModuleList(
            [
                PoleResidualStage2D(
                    width,
                    active.modes,
                    spacing=spacing,
                    geometry=self.geometry,
                    transport=active.transport,
                    recurrence_backend=active.recurrence_backend,
                    layer_scale_init=active.layer_scale_init,
                )
                for width, spacing in zip(active.widths, spacings, strict=True)
            ]
        )
        self.downs = nn.ModuleList(
            [
                PoleDownQRC2D(
                    active.widths[index],
                    active.widths[index + 1],
                    active.modes,
                    spacing=spacings[index],
                    geometry=self.geometry,
                    transport=active.transport,
                    recurrence_backend=active.recurrence_backend,
                )
                for index in range(2)
            ]
        )
        self.reader = PoleDescriptorReader2D(
            active.widths[-1],
            active.modes,
            spacing=spacings[-1],
            geometry=self.geometry,
            transport=active.transport,
            recurrence_backend=active.recurrence_backend,
        )
        self.descriptor_dim = 3 * _descriptor_size(active.modes)
        self.classifier = nn.Linear(self.descriptor_dim, active.output_dim)

    def forward_features(self, inputs: Tensor) -> Tensor:
        features = self.stem(inputs).permute(0, 2, 3, 1)
        descriptors = []
        for stage, down in zip(self.stages[:2], self.downs, strict=True):
            features = stage(features)
            features, descriptor = down(features)
            descriptors.append(descriptor)
        features = self.stages[2](features)
        descriptors.append(self.reader(features))
        return torch.cat(descriptors, dim=-1)

    def forward(self, inputs: Tensor) -> Tensor:
        descriptor = self.forward_features(inputs)
        with torch.autocast(device_type=descriptor.device.type, enabled=False):
            return self.classifier(descriptor.float())


__all__ = [
    "ConvStemPolePyramid",
    "ConvStemPolePyramidConfig",
    "ConvStemTransport",
    "ImageNetPoleStem",
    "PoleDescriptorReader2D",
    "PoleDownQRC2D",
    "PoleResidualStage2D",
]
