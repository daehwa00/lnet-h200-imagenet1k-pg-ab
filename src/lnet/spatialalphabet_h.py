"""Hierarchical native-2D pole moments for auditable image recognition."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import torch
from torch import Tensor, nn
from torch.nn import functional
from torch.nn.utils.parametrizations import orthogonal

from .alphabet2d import product_pole_scan_2d

if TYPE_CHECKING:
    from .pac_recurrence import RecurrenceBackend

Transport = Literal["product", "pole_free"]
DescriptorStandardization = Literal["none", "running"]
SpatialLag = tuple[int, int]
_QUADRANTS = ((1, 1), (-1, 1), (1, -1), (-1, -1))
_D1: tuple[SpatialLag, ...] = ((1, 0), (0, 1), (1, 1), (1, -1))
_D2: tuple[SpatialLag, ...] = (*_D1, (2, 0), (0, 2))


@dataclass(frozen=True, slots=True)
class SpatialAlphabetHConfig:
    """Frozen SPATIALPHABET-H architecture contract."""

    input_channels: int = 3
    output_dim: int = 100
    image_size: int = 224
    widths: tuple[int, ...] = (64, 128, 256, 384)
    depths: tuple[int, ...] = (1, 2, 4, 2)
    modes: tuple[int, ...] = (8, 12, 16, 24)
    mlp_ratio: float = 2.0
    layer_scale_init: float = 1.0e-2
    recurrence_backend: RecurrenceBackend = "auto"
    transport: Transport = "product"
    descriptor_standardization: DescriptorStandardization = "none"
    standardizer_momentum: float = 0.01

    def validate(self) -> None:
        if len(self.widths) != 4:
            message = "SPATIALPHABET-H requires exactly four stages"
            raise ValueError(message)
        if len(self.depths) != 4 or len(self.modes) != 4:
            message = "widths, depths, and modes must contain four entries"
            raise ValueError(message)
        if self.image_size < 64 or self.image_size % 32:
            message = "image_size must be at least 64 and divisible by 32"
            raise ValueError(message)
        if any(width < 2 * mode for width, mode in zip(self.widths, self.modes, strict=True)):
            message = "each stage width must be at least twice its mode count"
            raise ValueError(message)
        if any(mode < 4 or mode % 4 for mode in self.modes):
            message = "each mode count must be a positive multiple of four"
            raise ValueError(message)
        if min(self.depths) < 1:
            message = "each stage requires at least one local block"
            raise ValueError(message)
        if self.transport not in {"product", "pole_free"}:
            message = f"unsupported transport: {self.transport}"
            raise ValueError(message)
        if self.descriptor_standardization not in {"none", "running"}:
            message = "descriptor_standardization must be either 'none' or 'running'"
            raise ValueError(message)
        if not 0.0 < self.standardizer_momentum <= 1.0:
            message = "standardizer momentum must be in (0, 1]"
            raise ValueError(message)


def cross_mode_edges(modes: int) -> tuple[tuple[int, int], ...]:
    """Orientation-cycle and adjacent-scale edges for a four-orientation atlas."""
    if modes < 4 or modes % 4:
        message = "cross-mode graph requires a positive multiple of four modes"
        raise ValueError(message)
    levels = modes // 4
    edges: set[tuple[int, int]] = set()
    for level in range(levels):
        base = 4 * level
        for orientation in range(4):
            left = base + orientation
            right = base + ((orientation + 1) % 4)
            edges.add((min(left, right), max(left, right)))
    for level in range(levels - 1):
        for orientation in range(4):
            edges.add((4 * level + orientation, 4 * (level + 1) + orientation))
    return tuple(sorted(edges))


def _stage_atlas(stage: int, modes: int) -> tuple[Tensor, Tensor, Tensor]:
    radial_cycles = (
        (4.0, 12.0),
        (2.0, 5.0, 10.0),
        (1.0, 2.5, 4.5, 6.0),
        (0.25, 0.75, 1.25, 1.75, 2.25, 2.75),
    )[stage]
    levels = modes // 4
    if levels != len(radial_cycles):
        message = f"stage {stage} requires {4 * len(radial_cycles)} modes"
        raise ValueError(message)
    grid_size = (56, 28, 14, 7)[stage]
    cells = torch.logspace(
        math.log10(2.0),
        math.log10(max(2.0, grid_size / 2.0)),
        levels,
    )
    level = torch.arange(modes) // 4
    orientation = torch.arange(modes) % 4
    angles = orientation * (math.pi / 4.0)
    radial = torch.tensor(radial_cycles)[level]
    damping = (grid_size / cells)[level]
    omega = 2.0 * math.pi * radial
    return damping, omega * torch.cos(angles), omega * torch.sin(angles)


class HybridPhysicalModalField2D(nn.Module):
    """Half-fixed, half-anchored field with an L2-normalized complex direction mixer."""

    learned_mask: Tensor
    base_damping: Tensor
    base_frequency_x: Tensor
    base_frequency_y: Tensor
    analysis: nn.Linear
    direction_mix_real: nn.Parameter
    direction_mix_imag: nn.Parameter
    log_damping_offset_x: nn.Parameter
    log_damping_offset_y: nn.Parameter
    frequency_offset_x: nn.Parameter
    frequency_offset_y: nn.Parameter
    recurrence_backend: RecurrenceBackend
    transport: Transport

    def __init__(
        self,
        width: int,
        modes: int,
        *,
        stage: int,
        transport: Transport,
        recurrence_backend: RecurrenceBackend,
    ) -> None:
        super().__init__()
        self.width = width
        self.modes = modes
        self.transport = transport
        self.recurrence_backend = recurrence_backend
        self.analysis = nn.Linear(width, 2 * modes, bias=False)
        nn.init.orthogonal_(self.analysis.weight)
        orthogonal(
            self.analysis,
            "weight",
            orthogonal_map="matrix_exp",
            use_trivialization=True,
        )
        damping, omega_x, omega_y = _stage_atlas(stage, modes)
        self.register_buffer("base_damping", damping)
        self.register_buffer("base_frequency_x", omega_x)
        self.register_buffer("base_frequency_y", omega_y)
        self.base_damping = self.get_buffer("base_damping")
        self.base_frequency_x = self.get_buffer("base_frequency_x")
        self.base_frequency_y = self.get_buffer("base_frequency_y")
        self.log_damping_offset_x = nn.Parameter(torch.zeros(modes))
        self.log_damping_offset_y = nn.Parameter(torch.zeros(modes))
        self.frequency_offset_x = nn.Parameter(torch.zeros(modes))
        self.frequency_offset_y = nn.Parameter(torch.zeros(modes))
        level = torch.arange(modes) // 4
        orientation = torch.arange(modes) % 4
        learned = ((level + orientation) % 2).to(torch.float32)
        self.register_buffer("learned_mask", learned)
        self.learned_mask = self.get_buffer("learned_mask")
        self.direction_mix_real = nn.Parameter(torch.full((modes, 4), 0.5))
        self.direction_mix_imag = nn.Parameter(torch.zeros(modes, 4))

    def damping_x(self) -> Tensor:
        offset = self.learned_mask * torch.tanh(self.log_damping_offset_x)
        return self.base_damping * torch.exp(math.log(2.0) * offset)

    def damping_y(self) -> Tensor:
        offset = self.learned_mask * torch.tanh(self.log_damping_offset_y)
        return self.base_damping * torch.exp(math.log(2.0) * offset)

    def frequency_x(self) -> Tensor:
        offset = self.learned_mask * torch.tanh(self.frequency_offset_x)
        return self.base_frequency_x + 0.5 * math.pi * offset

    def frequency_y(self) -> Tensor:
        offset = self.learned_mask * torch.tanh(self.frequency_offset_y)
        return self.base_frequency_y + 0.5 * math.pi * offset

    def direction_mixer(self) -> tuple[Tensor, Tensor]:
        denominator = torch.sqrt(
            (self.direction_mix_real.square() + self.direction_mix_imag.square()).sum(
                dim=-1, keepdim=True
            )
        ).clamp_min(1.0e-8)
        return (
            self.direction_mix_real / denominator,
            self.direction_mix_imag / denominator,
        )

    def _pointwise_quadrants(
        self,
        excitation_real: Tensor,
        excitation_imag: Tensor,
        *,
        spacing_x: float,
        spacing_y: float,
    ) -> tuple[Tensor, Tensor]:
        amplitude = torch.exp(-self.damping_x() * spacing_x - self.damping_y() * spacing_y)
        real_states = []
        imag_states = []
        for direction_x, direction_y in _QUADRANTS:
            phase = (
                direction_x * self.frequency_x() * spacing_x
                + direction_y * self.frequency_y() * spacing_y
            )
            cosine = torch.cos(phase) * amplitude
            sine = torch.sin(phase) * amplitude
            real_states.append(excitation_real * cosine - excitation_imag * sine)
            imag_states.append(excitation_real * sine + excitation_imag * cosine)
        return torch.stack(real_states, dim=1), torch.stack(imag_states, dim=1)

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        excitation_real, excitation_imag = self.analysis(inputs).chunk(2, dim=-1)
        spacing_x = 1.0 / inputs.shape[2]
        spacing_y = 1.0 / inputs.shape[1]
        if self.transport == "product":
            quadrants = [
                product_pole_scan_2d(
                    excitation_real,
                    excitation_imag,
                    damping_x=self.damping_x(),
                    damping_y=self.damping_y(),
                    frequency_x=self.frequency_x(),
                    frequency_y=self.frequency_y(),
                    spacing_x=spacing_x,
                    spacing_y=spacing_y,
                    direction_x=direction_x,
                    direction_y=direction_y,
                    recurrence_backend=self.recurrence_backend,
                )
                for direction_x, direction_y in _QUADRANTS
            ]
            quadrant_real = torch.stack([state[0] for state in quadrants], dim=1)
            quadrant_imag = torch.stack([state[1] for state in quadrants], dim=1)
        else:
            quadrant_real, quadrant_imag = self._pointwise_quadrants(
                excitation_real,
                excitation_imag,
                spacing_x=spacing_x,
                spacing_y=spacing_y,
            )
        mixer_real, mixer_imag = self.direction_mixer()
        mixer_real = mixer_real.mT.view(1, 4, 1, 1, self.modes)
        mixer_imag = mixer_imag.mT.view(1, 4, 1, 1, self.modes)
        return (
            (quadrant_real * mixer_real - quadrant_imag * mixer_imag).sum(dim=1),
            (quadrant_real * mixer_imag + quadrant_imag * mixer_real).sum(dim=1),
        )

    def synthesize(self, real: Tensor, imag: Tensor) -> Tensor:
        frame_real, frame_imag = self.analysis.weight.chunk(2, dim=0)
        return torch.matmul(real, frame_real) + torch.matmul(imag, frame_imag)

    def audit(self) -> dict[str, Tensor]:
        mixer_real, mixer_imag = self.direction_mixer()
        return {
            "damping_x": self.damping_x().detach(),
            "damping_y": self.damping_y().detach(),
            "frequency_x": self.frequency_x().detach(),
            "frequency_y": self.frequency_y().detach(),
            "learned_mask": self.learned_mask.detach(),
            "direction_mix_real": mixer_real.detach(),
            "direction_mix_imag": mixer_imag.detach(),
        }


class _BlurPool(nn.Module):
    kernel: Tensor

    def __init__(self, channels: int, *, stride: int = 2) -> None:
        super().__init__()
        kernel_1d = torch.tensor((1.0, 2.0, 1.0))
        kernel = torch.outer(kernel_1d, kernel_1d)
        kernel = kernel / kernel.sum()
        self.register_buffer(
            "kernel",
            kernel.view(1, 1, 3, 3).expand(channels, 1, 3, 3).contiguous(),
        )
        self.kernel = self.get_buffer("kernel")
        self.channels = channels
        self.stride = stride

    def forward(self, inputs: Tensor) -> Tensor:
        return functional.conv2d(
            inputs,
            self.kernel,
            stride=self.stride,
            padding=1,
            groups=self.channels,
        )


class _AntiAliasedStem(nn.Module):
    def __init__(self, input_channels: int, width: int) -> None:
        super().__init__()
        self.first = nn.Conv2d(input_channels, width, 3, stride=2, padding=1)
        self.blur = _BlurPool(width)
        self.second = nn.Conv2d(width, width, 3, padding=1)
        self.norm = nn.RMSNorm(width)

    def forward(self, inputs: Tensor) -> Tensor:
        features = functional.silu(self.first(inputs))
        features = self.second(self.blur(features)).permute(0, 2, 3, 1)
        return self.norm(functional.silu(features))


class _LocalBlock(nn.Module):
    def __init__(self, width: int, *, mlp_ratio: float, layer_scale: float) -> None:
        super().__init__()
        hidden = round(width * mlp_ratio)
        self.depthwise = nn.Conv2d(width, width, 3, padding=1, groups=width)
        self.norm = nn.RMSNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Linear(hidden, width),
        )
        self.scale = nn.Parameter(torch.full((width,), layer_scale))

    def forward(self, inputs: Tensor) -> Tensor:
        local = self.depthwise(inputs.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        update = self.mlp(self.norm(functional.silu(local)))
        return inputs + self.scale * update


class SpatialPoleStage(nn.Module):
    local_blocks: nn.ModuleList
    field: HybridPhysicalModalField2D

    def __init__(
        self,
        width: int,
        modes: int,
        depth: int,
        *,
        stage: int,
        config: SpatialAlphabetHConfig,
    ) -> None:
        super().__init__()
        self.local_blocks = nn.ModuleList(
            [
                _LocalBlock(
                    width,
                    mlp_ratio=config.mlp_ratio,
                    layer_scale=config.layer_scale_init,
                )
                for _ in range(depth)
            ]
        )
        self.pole_norm = nn.RMSNorm(width)
        self.field = HybridPhysicalModalField2D(
            width,
            modes,
            stage=stage,
            transport=config.transport,
            recurrence_backend=config.recurrence_backend,
        )
        self.pole_scale = nn.Parameter(torch.full((width,), config.layer_scale_init))

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        features = inputs
        for block in self.local_blocks:
            features = cast("_LocalBlock", block)(features)
        pole_input = self.pole_norm(features)
        real, imag = self.field(pole_input)
        features = features + self.pole_scale * self.field.synthesize(real, imag)
        return features, real, imag


class AntiAliasedMerge(nn.Module):
    def __init__(self, input_width: int, output_width: int) -> None:
        super().__init__()
        self.blur = _BlurPool(input_width)
        self.projection = nn.Conv2d(input_width, output_width, 1)
        self.norm = nn.RMSNorm(output_width)

    def forward(self, inputs: Tensor) -> Tensor:
        channels_first = inputs.permute(0, 3, 1, 2)
        merged = self.projection(self.blur(channels_first)).permute(0, 2, 3, 1)
        return self.norm(merged)


def _region_slices(
    height: int,
    width: int,
    *,
    regional: bool,
) -> tuple[tuple[str, slice, slice], ...]:
    global_region = ("global", slice(0, height), slice(0, width))
    if not regional:
        return (global_region,)
    middle_y = max(1, height // 2)
    middle_x = max(1, width // 2)
    return (
        global_region,
        ("top_left", slice(0, middle_y), slice(0, middle_x)),
        ("top_right", slice(0, middle_y), slice(middle_x, width)),
        ("bottom_left", slice(middle_y, height), slice(0, middle_x)),
        ("bottom_right", slice(middle_y, height), slice(middle_x, width)),
    )


def _radial_log_complex(real: Tensor, imag: Tensor) -> tuple[Tensor, Tensor]:
    magnitude = torch.sqrt(real.square() + imag.square())
    scale = torch.log1p(magnitude) / magnitude.clamp_min(1.0e-8)
    return real * scale, imag * scale


def _lag_pair(
    real: Tensor,
    imag: Tensor,
    delta_x: int,
    delta_y: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    height, width = real.shape[1:3]
    if abs(delta_x) >= width or abs(delta_y) >= height:
        message = f"lag ({delta_x}, {delta_y}) exceeds {height}x{width} region"
        raise ValueError(message)
    current_y = slice(max(delta_y, 0), height + min(delta_y, 0))
    previous_y = slice(max(-delta_y, 0), height - max(delta_y, 0))
    current_x = slice(max(delta_x, 0), width + min(delta_x, 0))
    previous_x = slice(max(-delta_x, 0), width - max(delta_x, 0))
    return (
        real[:, current_y, current_x],
        imag[:, current_y, current_x],
        real[:, previous_y, previous_x],
        imag[:, previous_y, previous_x],
    )


def regional_pole_moments(
    real: Tensor,
    imag: Tensor,
    *,
    lags: tuple[SpatialLag, ...],
    edges: tuple[tuple[int, int], ...],
    regional: bool,
) -> Tensor:
    """Return conditioned Q, normalized R, and sparse normalized C coordinates."""
    if real.shape != imag.shape or real.ndim != 4:
        message = "modal responses must share [B,H,W,M] shape"
        raise ValueError(message)
    real = real.float()
    imag = imag.float()
    pieces = []
    epsilon = max(1.0e-8, torch.finfo(real.dtype).eps)
    for _name, row_slice, column_slice in _region_slices(
        real.shape[1],
        real.shape[2],
        regional=regional,
    ):
        region_real = real[:, row_slice, column_slice]
        region_imag = imag[:, row_slice, column_slice]
        energy = (region_real.square() + region_imag.square()).mean(dim=(1, 2))
        region_pieces = [torch.log1p(energy)]
        for delta_x, delta_y in lags:
            current_real, current_imag, previous_real, previous_imag = _lag_pair(
                region_real,
                region_imag,
                delta_x,
                delta_y,
            )
            correlation_real = (current_real * previous_real + current_imag * previous_imag).mean(
                dim=(1, 2)
            ) / energy.clamp_min(epsilon)
            correlation_imag = (current_imag * previous_real - current_real * previous_imag).mean(
                dim=(1, 2)
            ) / energy.clamp_min(epsilon)
            radial_real, radial_imag = _radial_log_complex(
                correlation_real,
                correlation_imag,
            )
            region_pieces.extend((radial_real, radial_imag))
        if edges:
            left = torch.tensor(
                [edge[0] for edge in edges],
                device=real.device,
            )
            right = torch.tensor(
                [edge[1] for edge in edges],
                device=real.device,
            )
            covariance_real = (
                region_real[..., left] * region_real[..., right]
                + region_imag[..., left] * region_imag[..., right]
            ).mean(dim=(1, 2))
            covariance_imag = (
                region_imag[..., left] * region_real[..., right]
                - region_real[..., left] * region_imag[..., right]
            ).mean(dim=(1, 2))
            denominator = torch.sqrt(
                (energy[:, left] * energy[:, right]).clamp_min(epsilon * epsilon)
            )
            covariance_real = covariance_real / denominator
            covariance_imag = covariance_imag / denominator
            radial_real, radial_imag = _radial_log_complex(
                covariance_real,
                covariance_imag,
            )
            region_pieces.extend((radial_real, radial_imag))
        pieces.append(torch.cat(region_pieces, dim=-1))
    return torch.cat(pieces, dim=-1)


def descriptor_coordinates(
    modes: tuple[int, ...] = (8, 12, 16, 24),
) -> tuple[str, ...]:
    coordinates = []
    for stage, mode_count in enumerate(modes):
        lags = _D1 if stage < 2 else _D2
        edges = () if stage < 2 else cross_mode_edges(mode_count)
        regions = (
            ("global",)
            if stage < 2
            else (
                "global",
                "top_left",
                "top_right",
                "bottom_left",
                "bottom_right",
            )
        )
        for region in regions:
            coordinates.extend(
                f"stage{stage + 1}/{region}/mode{mode}/Q" for mode in range(mode_count)
            )
            for delta_x, delta_y in lags:
                coordinates.extend(
                    f"stage{stage + 1}/{region}/mode{mode}/R({delta_x},{delta_y})/real"
                    for mode in range(mode_count)
                )
                coordinates.extend(
                    f"stage{stage + 1}/{region}/mode{mode}/R({delta_x},{delta_y})/imag"
                    for mode in range(mode_count)
                )
            coordinates.extend(
                f"stage{stage + 1}/{region}/edge{left}-{right}/C/real" for left, right in edges
            )
            coordinates.extend(
                f"stage{stage + 1}/{region}/edge{left}-{right}/C/imag" for left, right in edges
            )
    return tuple(coordinates)


class CoordinateStandardizer(nn.Module):
    """Shared running coordinate standardization for train and evaluation."""

    running_mean: Tensor
    running_variance: Tensor
    batches_seen: Tensor

    def __init__(self, dimensions: int, momentum: float) -> None:
        super().__init__()
        self.momentum = momentum
        self.register_buffer("running_mean", torch.zeros(dimensions))
        self.register_buffer("running_variance", torch.ones(dimensions))
        self.register_buffer("batches_seen", torch.zeros((), dtype=torch.long))
        self.running_mean = self.get_buffer("running_mean")
        self.running_variance = self.get_buffer("running_variance")
        self.batches_seen = self.get_buffer("batches_seen")

    def forward(self, inputs: Tensor) -> Tensor:
        working = inputs.float()
        epsilon = 1.0e-6
        if self.training:
            mean = working.detach().mean(dim=0)
            variance = working.detach().var(dim=0, unbiased=False)
            with torch.no_grad():
                first_batch = self.batches_seen == 0
                updated_mean = torch.lerp(
                    self.running_mean,
                    mean,
                    self.momentum,
                )
                updated_variance = torch.lerp(
                    self.running_variance,
                    variance.clamp_min(epsilon),
                    self.momentum,
                )
                self.running_mean.copy_(torch.where(first_batch, mean, updated_mean))
                self.running_variance.copy_(
                    torch.where(
                        first_batch,
                        variance.clamp_min(epsilon),
                        updated_variance,
                    )
                )
                self.batches_seen.add_(1)
        return (working - self.running_mean) / torch.sqrt(self.running_variance.clamp_min(epsilon))


class SpatialAlphabetH(nn.Module):
    """Four-stage SPATIALPHABET-H with a fixed modal-moment affine interface."""

    stages: nn.ModuleList
    merges: nn.ModuleList
    stem: _AntiAliasedStem
    standardizer: nn.Module
    classifier: nn.Linear

    def __init__(self, config: SpatialAlphabetHConfig | None = None) -> None:
        super().__init__()
        active = config or SpatialAlphabetHConfig()
        active.validate()
        self.config = active
        self.stem = _AntiAliasedStem(active.input_channels, active.widths[0])
        self.stages = nn.ModuleList(
            [
                SpatialPoleStage(
                    width,
                    modes,
                    depth,
                    stage=stage,
                    config=active,
                )
                for stage, (width, modes, depth) in enumerate(
                    zip(active.widths, active.modes, active.depths, strict=True)
                )
            ]
        )
        self.merges = nn.ModuleList(
            [
                AntiAliasedMerge(input_width, output_width)
                for input_width, output_width in zip(
                    active.widths[:-1],
                    active.widths[1:],
                    strict=True,
                )
            ]
        )
        self.coordinate_names = descriptor_coordinates(active.modes)
        self.descriptor_dim = len(self.coordinate_names)
        self.standardizer = (
            CoordinateStandardizer(
                self.descriptor_dim,
                active.standardizer_momentum,
            )
            if active.descriptor_standardization == "running"
            else nn.Identity()
        )
        self.classifier = nn.Linear(self.descriptor_dim, active.output_dim)

    def raw_descriptor(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        features = self.stem(inputs)
        descriptors = []
        for stage_index, stage in enumerate(self.stages):
            active_stage = cast("SpatialPoleStage", stage)
            features, real, imag = active_stage(features)
            descriptors.append(
                regional_pole_moments(
                    real,
                    imag,
                    lags=_D1 if stage_index < 2 else _D2,
                    edges=(
                        () if stage_index < 2 else cross_mode_edges(self.config.modes[stage_index])
                    ),
                    regional=stage_index >= 2,
                )
            )
            if stage_index < len(self.merges):
                features = cast("AntiAliasedMerge", self.merges[stage_index])(features)
        return torch.cat(descriptors, dim=-1), features

    def forward_features(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        descriptor, terminal = self.raw_descriptor(inputs)
        return self.standardizer(descriptor), terminal

    def forward(self, inputs: Tensor) -> Tensor:
        descriptor, _ = self.forward_features(inputs)
        with torch.autocast(device_type=descriptor.device.type, enabled=False):
            return self.classifier(descriptor.float())

    def decompose_margin(
        self,
        descriptor: Tensor,
        winner: int,
        runner_up: int,
    ) -> tuple[Tensor, Tensor]:
        weight = self.classifier.weight[winner] - self.classifier.weight[runner_up]
        base = self.classifier.bias[winner] - self.classifier.bias[runner_up]
        return base.expand(descriptor.shape[0]), descriptor * weight

    def pole_audit(self) -> dict[str, dict[str, Tensor]]:
        return {
            f"stage{index + 1}": cast("SpatialPoleStage", stage).field.audit()
            for index, stage in enumerate(self.stages)
        }


def build_spatialalphabet_h(
    config: SpatialAlphabetHConfig | None = None,
) -> SpatialAlphabetH:
    return SpatialAlphabetH(config)


__all__ = [
    "AntiAliasedMerge",
    "CoordinateStandardizer",
    "DescriptorStandardization",
    "HybridPhysicalModalField2D",
    "SpatialAlphabetH",
    "SpatialAlphabetHConfig",
    "SpatialPoleStage",
    "build_spatialalphabet_h",
    "cross_mode_edges",
    "descriptor_coordinates",
    "regional_pole_moments",
]
