"""ALPHABET-2D-Tiny: hierarchical product-pole writers with a modal affine head."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor, nn
from torch.nn import functional
from torch.nn.utils.parametrizations import orthogonal

from .alphabet2d import product_pole_scan_2d
from .image_layers import LayerNorm2d
from .pac_real2d_math import discrete_pole_real2d

if TYPE_CHECKING:
    from .pac_recurrence import RecurrenceBackend

ComplexField = tuple[Tensor, Tensor]
_DIRECTIONS = ((1, 1), (-1, 1), (1, -1), (-1, -1))
_LAGS = ((1, 0), (0, 1), (1, 1), (1, -1))


class DropPath(nn.Module):
    """Per-example stochastic depth without adding a dependency."""

    def __init__(self, probability: float) -> None:
        super().__init__()
        self.probability = probability

    def forward(self, inputs: Tensor) -> Tensor:
        if not self.training or self.probability == 0.0:
            return inputs
        keep = 1.0 - self.probability
        mask_shape = (inputs.shape[0],) + (1,) * (inputs.ndim - 1)
        mask = torch.empty(mask_shape, dtype=inputs.dtype, device=inputs.device).bernoulli_(keep)
        return inputs * mask / keep


class OverlappingConvStem(nn.Module):
    """Conv-LN-GELU-Conv-LN stem leaving signed excitation for the first writer."""

    def __init__(self, output_width: int) -> None:
        super().__init__()
        hidden = output_width // 2
        self.layers = nn.Sequential(
            nn.Conv2d(3, hidden, 3, stride=2, padding=1, bias=False),
            LayerNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, output_width, 3, stride=2, padding=1, bias=False),
            LayerNorm2d(output_width),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs).permute(0, 2, 3, 1)


def _pole_atlas(modes: int, maximum_cycles: float) -> tuple[Tensor, Tensor, Tensor]:
    levels = modes // 4
    radial = torch.logspace(math.log10(maximum_cycles / 8.0), math.log10(maximum_cycles), levels)
    radial = radial.repeat_interleave(4)
    orientation = torch.tensor(
        (0.0, math.pi / 4.0, math.pi / 2.0, 3.0 * math.pi / 4.0)
    ).repeat(levels)
    omega = 2.0 * math.pi * radial
    damping = (omega / 2.0).clamp_min(0.5)
    return damping, omega * torch.cos(orientation), omega * torch.sin(orientation)


class ProductPoleBank2D(nn.Module):
    """Independent stable pole atlas and semi-orthogonal complex analysis frame."""

    def __init__(
        self,
        width: int,
        modes: int,
        *,
        maximum_cycles: float,
        recurrence_backend: RecurrenceBackend,
    ) -> None:
        super().__init__()
        self.modes = modes
        self.recurrence_backend: RecurrenceBackend = recurrence_backend
        self.analysis = nn.Linear(width, 2 * modes, bias=False)
        nn.init.orthogonal_(self.analysis.weight)
        orthogonal(self.analysis, "weight", orthogonal_map="matrix_exp", use_trivialization=True)
        damping, frequency_x, frequency_y = _pole_atlas(modes, maximum_cycles)
        self.raw_damping_x = nn.Parameter(torch.log(torch.expm1(damping)))
        self.raw_damping_y = nn.Parameter(torch.log(torch.expm1(damping)))
        self.frequency_x = nn.Parameter(frequency_x)
        self.frequency_y = nn.Parameter(frequency_y)

    def damping_x(self) -> Tensor:
        return functional.softplus(self.raw_damping_x) + 1.0e-4

    def damping_y(self) -> Tensor:
        return functional.softplus(self.raw_damping_y) + 1.0e-4

    def gain(self, spacing_x: float, spacing_y: float) -> Tensor:
        decay_x_real, decay_x_imag, gamma_x_real, gamma_x_imag = discrete_pole_real2d(
            self.damping_x(), self.frequency_x, spacing_x
        )
        decay_y_real, decay_y_imag, gamma_y_real, gamma_y_imag = discrete_pole_real2d(
            self.damping_y(), self.frequency_y, spacing_y
        )
        gain_x = (gamma_x_real.square() + gamma_x_imag.square()) / (
            1.0 - decay_x_real.square() - decay_x_imag.square()
        ).clamp_min(1.0e-8)
        gain_y = (gamma_y_real.square() + gamma_y_imag.square()) / (
            1.0 - decay_y_real.square() - decay_y_imag.square()
        ).clamp_min(1.0e-8)
        return (gain_x * gain_y).clamp_min(1.0e-8)

    def forward(self, inputs: Tensor) -> tuple[ComplexField, ...]:
        real, imag = self.analysis(inputs).chunk(2, dim=-1)
        spacing_x = 1.0 / inputs.shape[2]
        spacing_y = 1.0 / inputs.shape[1]
        inverse_gain = torch.rsqrt(self.gain(spacing_x, spacing_y)).view(1, 1, 1, -1)
        states = []
        for direction_x, direction_y in _DIRECTIONS:
            state_real, state_imag = product_pole_scan_2d(
                real,
                imag,
                damping_x=self.damping_x(),
                damping_y=self.damping_y(),
                frequency_x=self.frequency_x,
                frequency_y=self.frequency_y,
                spacing_x=spacing_x,
                spacing_y=spacing_y,
                direction_x=direction_x,
                direction_y=direction_y,
                recurrence_backend=self.recurrence_backend,
            )
            states.append((state_real * inverse_gain, state_imag * inverse_gain))
        return tuple(states)

    def synthesize(self, real: Tensor, imag: Tensor) -> Tensor:
        frame_real, frame_imag = self.analysis.weight.chunk(2, dim=0)
        return torch.matmul(real, frame_real) + torch.matmul(imag, frame_imag)


class ProductPoleWriterBlock(nn.Module):
    """Local lift, four-direction pole transport, tied synthesis, and channel MLP."""

    def __init__(
        self,
        width: int,
        modes: int,
        *,
        maximum_cycles: float,
        recurrence_backend: RecurrenceBackend,
        layer_scale_init: float,
        drop_path: float,
    ) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(width, width, 3, padding=1, groups=width)
        self.local_norm = nn.RMSNorm(width)
        self.bank = ProductPoleBank2D(
            width,
            modes,
            maximum_cycles=maximum_cycles,
            recurrence_backend=recurrence_backend,
        )
        self.feedthrough = nn.Parameter(torch.zeros(width))
        self.pole_scale = nn.Parameter(torch.full((width,), layer_scale_init))
        self.pole_drop = DropPath(drop_path)
        self.mlp_norm = nn.RMSNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, 2 * width),
            nn.SiLU(),
            nn.Linear(2 * width, width),
        )
        self.mlp_scale = nn.Parameter(torch.full((width,), layer_scale_init))
        self.mlp_drop = DropPath(drop_path)

    def forward(self, inputs: Tensor) -> tuple[Tensor, tuple[ComplexField, ...]]:
        local = self.depthwise(inputs.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        lifted = self.local_norm(functional.silu(local))
        states = self.bank(lifted)
        mean_real = torch.stack([state[0] for state in states]).mean(dim=0)
        mean_imag = torch.stack([state[1] for state in states]).mean(dim=0)
        synthesis = self.bank.synthesize(mean_real, mean_imag) + self.feedthrough * lifted
        features = inputs + self.pole_drop(self.pole_scale * synthesis)
        mlp_update = self.mlp(self.mlp_norm(features))
        return features + self.mlp_drop(self.mlp_scale * mlp_update), states


class AntiAliasedDownsample(nn.Module):
    """Channel LayerNorm, fixed binomial blur, and learned 2x2 stride-two projection."""

    def __init__(self, input_width: int, output_width: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_width)
        kernel_1d = torch.tensor((1.0, 2.0, 1.0))
        kernel = torch.outer(kernel_1d, kernel_1d) / 16.0
        self.register_buffer(
            "blur_kernel",
            kernel.view(1, 1, 3, 3).expand(input_width, 1, 3, 3).contiguous(),
        )
        self.projection = nn.Conv2d(input_width, output_width, 2, stride=2)

    def forward(self, inputs: Tensor) -> Tensor:
        channels_first = self.norm(inputs).permute(0, 3, 1, 2)
        blurred = functional.conv2d(
            channels_first,
            self.get_buffer("blur_kernel"),
            padding=1,
            groups=channels_first.shape[1],
        )
        return self.projection(blurred).permute(0, 2, 3, 1)


class TerminalPoleReader(nn.Module):
    """Independent no-synthesis terminal pole bank."""

    def __init__(
        self,
        width: int,
        modes: int,
        *,
        maximum_cycles: float,
        recurrence_backend: RecurrenceBackend,
    ) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(width, width, 3, padding=1, groups=width)
        self.norm = nn.RMSNorm(width)
        self.bank = ProductPoleBank2D(
            width,
            modes,
            maximum_cycles=maximum_cycles,
            recurrence_backend=recurrence_backend,
        )

    def forward(self, inputs: Tensor) -> tuple[ComplexField, ...]:
        local = self.depthwise(inputs.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return self.bank(self.norm(functional.silu(local)))


def _radial_log(real: Tensor, imag: Tensor) -> tuple[Tensor, Tensor]:
    magnitude = torch.sqrt(real.square() + imag.square())
    scale = torch.log1p(magnitude) / magnitude.clamp_min(1.0e-8)
    return real * scale, imag * scale


def _regions(height: int, width: int) -> tuple[tuple[slice, slice], ...]:
    middle_y, middle_x = height // 2, width // 2
    return (
        (slice(0, height), slice(0, width)),
        (slice(0, middle_y), slice(0, middle_x)),
        (slice(0, middle_y), slice(middle_x, width)),
        (slice(middle_y, height), slice(0, middle_x)),
        (slice(middle_y, height), slice(middle_x, width)),
    )


def spatial_modal_moments(states: tuple[ComplexField, ...]) -> Tensor:
    """Return five-window Q/R moments while retaining bank, direction, and mode axes."""
    descriptors = []
    for state_real, state_imag in states:
        working_real, working_imag = state_real.float(), state_imag.float()
        for rows, columns in _regions(working_real.shape[1], working_real.shape[2]):
            region_real = working_real[:, rows, columns]
            region_imag = working_imag[:, rows, columns]
            pieces = [torch.log1p((region_real.square() + region_imag.square()).mean((1, 2)))]
            height, width = region_real.shape[1:3]
            for delta_x, delta_y in _LAGS:
                current_y = slice(max(delta_y, 0), height + min(delta_y, 0))
                previous_y = slice(max(-delta_y, 0), height - max(delta_y, 0))
                current_x = slice(max(delta_x, 0), width + min(delta_x, 0))
                previous_x = slice(max(-delta_x, 0), width - max(delta_x, 0))
                current_real = region_real[:, current_y, current_x]
                current_imag = region_imag[:, current_y, current_x]
                previous_real = region_real[:, previous_y, previous_x]
                previous_imag = region_imag[:, previous_y, previous_x]
                correlation_real = (
                    current_real * previous_real + current_imag * previous_imag
                ).mean((1, 2))
                correlation_imag = (
                    current_imag * previous_real - current_real * previous_imag
                ).mean((1, 2))
                pieces.extend(_radial_log(correlation_real, correlation_imag))
            descriptors.append(torch.cat(pieces, dim=-1))
    return torch.cat(descriptors, dim=-1)


@dataclass(frozen=True, slots=True)
class Alphabet2DTinyConfig:
    output_dim: int = 100
    image_size: int = 224
    widths: tuple[int, ...] = (64, 128, 192)
    modes: tuple[int, ...] = (8, 16, 16)
    depths: tuple[int, ...] = (2, 4, 2)
    layer_scale_init: float = 1.0e-3
    drop_path_rate: float = 0.1
    recurrence_backend: RecurrenceBackend = "auto"

    def validate(self) -> None:
        if self.image_size < 64 or self.image_size % 16:
            message = "image size must be at least 64 and divisible by 16"
            raise ValueError(message)
        if not (len(self.widths) == len(self.modes) == len(self.depths) == 3):
            message = "ALPHABET-2D-Tiny requires exactly three stages"
            raise ValueError(message)
        if any(2 * modes > width for width, modes in zip(self.widths, self.modes, strict=True)):
            message = "each stage must satisfy 2M <= D"
            raise ValueError(message)


class Alphabet2DTiny(nn.Module):
    """The prescribed 2+4+2 writer hierarchy and independent terminal reader."""

    stages: nn.ModuleList
    downsamples: nn.ModuleList

    def __init__(self, config: Alphabet2DTinyConfig | None = None) -> None:
        super().__init__()
        active = config or Alphabet2DTinyConfig()
        active.validate()
        self.config = active
        self.stem = OverlappingConvStem(active.widths[0])
        maximum_cycles = (16.0, 10.0, 6.0)
        total_blocks = sum(active.depths)
        drop_rates = torch.linspace(0.0, active.drop_path_rate, total_blocks).tolist()
        block_index = 0
        stages = []
        for width, modes, depth, cycles in zip(
            active.widths, active.modes, active.depths, maximum_cycles, strict=True
        ):
            blocks = []
            for _ in range(depth):
                blocks.append(
                    ProductPoleWriterBlock(
                        width,
                        modes,
                        maximum_cycles=cycles,
                        recurrence_backend=active.recurrence_backend,
                        layer_scale_init=active.layer_scale_init,
                        drop_path=drop_rates[block_index],
                    )
                )
                block_index += 1
            stages.append(nn.ModuleList(blocks))
        self.stages = nn.ModuleList(stages)
        self.downsamples = nn.ModuleList(
            [
                AntiAliasedDownsample(input_width, output_width)
                for input_width, output_width in zip(
                    active.widths[:-1], active.widths[1:], strict=True
                )
            ]
        )
        self.reader = TerminalPoleReader(
            active.widths[-1],
            active.modes[-1],
            maximum_cycles=maximum_cycles[-1],
            recurrence_backend=active.recurrence_backend,
        )
        self.descriptor_dim = 2 * 4 * active.modes[-1] * 5 * 9
        self.classifier = nn.Linear(self.descriptor_dim, active.output_dim)

    def raw_descriptor(self, inputs: Tensor) -> Tensor:
        features = self.stem(inputs)
        writer_states: tuple[ComplexField, ...] | None = None
        for stage_index, stage in enumerate(self.stages):
            for block in cast("nn.ModuleList", stage):
                features, writer_states = cast("ProductPoleWriterBlock", block)(features)
            if stage_index < len(self.downsamples):
                features = cast("AntiAliasedDownsample", self.downsamples[stage_index])(features)
        if writer_states is None:
            message = "writer hierarchy produced no modal state"
            raise RuntimeError(message)
        reader_states = self.reader(features)
        return torch.cat(
            (spatial_modal_moments(writer_states), spatial_modal_moments(reader_states)),
            dim=-1,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        descriptor = self.raw_descriptor(inputs)
        with torch.autocast(device_type=descriptor.device.type, enabled=False):
            return self.classifier(descriptor.float())


__all__ = [
    "Alphabet2DTiny",
    "Alphabet2DTinyConfig",
    "AntiAliasedDownsample",
    "OverlappingConvStem",
    "ProductPoleBank2D",
    "ProductPoleWriterBlock",
    "TerminalPoleReader",
    "spatial_modal_moments",
]
