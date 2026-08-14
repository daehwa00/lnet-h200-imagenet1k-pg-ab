"""PolePyramid-C-Tiny with exact pole-state coarsening after a shallow stem."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor, nn

from .alphabet2d_tiny import ComplexField, OverlappingConvStem, ProductPoleBank2D
from .pac_directional import direction_aligned_endpoints

if TYPE_CHECKING:
    from .pac_recurrence import RecurrenceBackend

_DIRECTIONS = ((1, 1), (-1, 1), (1, -1), (-1, -1))
_LAGS = ((1, 0), (0, 1), (1, 1), (1, -1))


def _radial_log(real: Tensor, imag: Tensor) -> tuple[Tensor, Tensor]:
    magnitude = torch.sqrt(real.square() + imag.square())
    scale = torch.log1p(magnitude) / magnitude.clamp_min(1.0e-8)
    return real * scale, imag * scale


def global_modal_moments(states: tuple[ComplexField, ...]) -> Tensor:
    """Global Q and four unnormalized complex spatial-lag moments."""
    descriptors = []
    for state_real, state_imag in states:
        real, imag = state_real.float(), state_imag.float()
        height, width = real.shape[1:3]
        pieces = [torch.log1p((real.square() + imag.square()).mean((1, 2)))]
        for delta_x, delta_y in _LAGS:
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
            ).mean((1, 2))
            correlation_imag = (
                current_imag * previous_real - current_real * previous_imag
            ).mean((1, 2))
            pieces.extend(_radial_log(correlation_real, correlation_imag))
        descriptors.append(torch.cat(pieces, dim=-1))
    return torch.cat(descriptors, dim=-1)


class PoleDown2D(nn.Module):
    """Fine-grid product-pole analysis with exact endpoint carry and tied synthesis."""

    def __init__(
        self,
        input_width: int,
        output_width: int,
        modes: int,
        *,
        maximum_cycles: float,
        recurrence_backend: RecurrenceBackend,
        layer_scale_init: float,
        gate_sharpness: float,
    ) -> None:
        super().__init__()
        self.input_norm = nn.RMSNorm(input_width)
        self.bank = ProductPoleBank2D(
            input_width,
            modes,
            maximum_cycles=maximum_cycles,
            recurrence_backend=recurrence_backend,
        )
        self.direction_mix = nn.Linear(4 * input_width, output_width)
        self.mlp_norm = nn.RMSNorm(output_width)
        self.mlp = nn.Sequential(
            nn.Linear(output_width, 2 * output_width),
            nn.SiLU(),
            nn.Linear(2 * output_width, output_width),
        )
        self.mlp_scale = nn.Parameter(torch.full((output_width,), layer_scale_init))
        self.gate_sharpness = gate_sharpness

    def _carry_gate(self, height: int, width: int) -> Tensor:
        spacing_x, spacing_y = 1.0 / width, 1.0 / height
        gate_x = torch.sigmoid(
            self.gate_sharpness * (math.pi / 2.0 - self.bank.frequency_x.abs() * spacing_x)
        )
        gate_y = torch.sigmoid(
            self.gate_sharpness * (math.pi / 2.0 - self.bank.frequency_y.abs() * spacing_y)
        )
        return (gate_x * gate_y).view(1, 1, 1, -1)

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        states = self.bank(self.input_norm(inputs))
        descriptor = global_modal_moments(states)
        gate = self._carry_gate(inputs.shape[1], inputs.shape[2])
        synthesized = []
        for (direction_x, direction_y), (real, imag) in zip(
            _DIRECTIONS, states, strict=True
        ):
            coarse_real, coarse_imag = direction_aligned_endpoints(
                real,
                imag,
                direction_x=direction_x,
                direction_y=direction_y,
            )
            synthesized.append(self.bank.synthesize(gate * coarse_real, gate * coarse_imag))
        coarse = self.direction_mix(torch.cat(synthesized, dim=-1))
        mixed = coarse + self.mlp_scale * self.mlp(self.mlp_norm(coarse))
        return mixed, descriptor


class CascadedPoleReader2D(nn.Module):
    """Independent terminal bank with no synthesis or feature update."""

    def __init__(
        self,
        width: int,
        modes: int,
        *,
        maximum_cycles: float,
        recurrence_backend: RecurrenceBackend,
    ) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(width)
        self.bank = ProductPoleBank2D(
            width,
            modes,
            maximum_cycles=maximum_cycles,
            recurrence_backend=recurrence_backend,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return global_modal_moments(self.bank(self.norm(inputs)))


@dataclass(frozen=True, slots=True)
class PolePyramidCTinyConfig:
    output_dim: int = 100
    image_size: int = 224
    widths: tuple[int, ...] = (64, 96, 128, 192)
    modes: tuple[int, ...] = (16, 24, 32)
    terminal_modes: int = 32
    layer_scale_init: float = 1.0e-3
    gate_sharpness: float = 8.0
    recurrence_backend: RecurrenceBackend = "auto"

    def validate(self) -> None:
        if self.image_size < 64 or self.image_size % 32:
            message = "image size must be at least 64 and divisible by 32"
            raise ValueError(message)
        if len(self.widths) != 4 or len(self.modes) != 3:
            message = "PolePyramid-C-Tiny requires three PoleDown stages"
            raise ValueError(message)
        if any(
            2 * modes > width
            for width, modes in zip(self.widths[:-1], self.modes, strict=True)
        ):
            message = "each PoleDown must satisfy 2M <= input width"
            raise ValueError(message)
        if 2 * self.terminal_modes > self.widths[-1]:
            message = "terminal bank must satisfy 2M <= input width"
            raise ValueError(message)


class PolePyramidCTiny(nn.Module):
    """Stem followed solely by three PoleDowns and a cascaded modal reader."""

    downs: nn.ModuleList

    def __init__(self, config: PolePyramidCTinyConfig | None = None) -> None:
        super().__init__()
        active = config or PolePyramidCTinyConfig()
        active.validate()
        self.config = active
        self.stem = OverlappingConvStem(active.widths[0])
        maximum_cycles = (16.0, 8.0, 4.0)
        self.downs = nn.ModuleList(
            [
                PoleDown2D(
                    input_width,
                    output_width,
                    modes,
                    maximum_cycles=cycles,
                    recurrence_backend=active.recurrence_backend,
                    layer_scale_init=active.layer_scale_init,
                    gate_sharpness=active.gate_sharpness,
                )
                for input_width, output_width, modes, cycles in zip(
                    active.widths[:-1],
                    active.widths[1:],
                    active.modes,
                    maximum_cycles,
                    strict=True,
                )
            ]
        )
        self.reader = CascadedPoleReader2D(
            active.widths[-1],
            active.terminal_modes,
            maximum_cycles=3.0,
            recurrence_backend=active.recurrence_backend,
        )
        self.descriptor_dim = 4 * 9 * (sum(active.modes) + active.terminal_modes)
        self.classifier = nn.Linear(self.descriptor_dim, active.output_dim)

    def raw_descriptor(self, inputs: Tensor) -> Tensor:
        features = self.stem(inputs)
        descriptors = []
        for down in self.downs:
            features, descriptor = cast("PoleDown2D", down)(features)
            descriptors.append(descriptor)
        descriptors.append(self.reader(features))
        return torch.cat(descriptors, dim=-1)

    def forward(self, inputs: Tensor) -> Tensor:
        descriptor = self.raw_descriptor(inputs)
        with torch.autocast(device_type=descriptor.device.type, enabled=False):
            return self.classifier(descriptor.float())


__all__ = [
    "CascadedPoleReader2D",
    "PoleDown2D",
    "PolePyramidCTiny",
    "PolePyramidCTinyConfig",
    "global_modal_moments",
]
