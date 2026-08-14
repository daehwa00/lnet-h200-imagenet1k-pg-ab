"""Deep PolePyramid-C-Tiny with static direction merging and no excitation gate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor, nn
from torch.nn import functional

from .alphabet2d_tiny import (
    ComplexField,
    DropPath,
    OverlappingConvStem,
    ProductPoleBank2D,
)
from .pac_directional import direction_aligned_endpoints
from .polepyramid_c_tiny import global_modal_moments

if TYPE_CHECKING:
    from .pac_recurrence import RecurrenceBackend

_DIRECTIONS = ((1, 1), (-1, 1), (1, -1), (-1, -1))


def _static_merge(
    states: tuple[ComplexField, ...], direction_logits: Tensor
) -> ComplexField:
    weights = torch.softmax(direction_logits, dim=-1).mT.view(4, 1, 1, 1, -1)
    real = torch.stack([state[0] for state in states])
    imag = torch.stack([state[1] for state in states])
    return (weights * real).sum(dim=0), (weights * imag).sum(dim=0)


class StaticDirectionPoleBlock2D(nn.Module):
    """Same-resolution writer with fixed learned mode-wise direction weights."""

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
        self.direction_logits = nn.Parameter(torch.zeros(modes, 4))
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

    def forward(self, inputs: Tensor) -> Tensor:
        local = self.depthwise(inputs.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        lifted = self.local_norm(functional.silu(local))
        merged_real, merged_imag = _static_merge(self.bank(lifted), self.direction_logits)
        synthesis = self.bank.synthesize(merged_real, merged_imag) + self.feedthrough * lifted
        features = inputs + self.pole_drop(self.pole_scale * synthesis)
        return features + self.mlp_drop(
            self.mlp_scale * self.mlp(self.mlp_norm(features))
        )


class DeepPoleDown2D(nn.Module):
    """Independent local pole bank, exact endpoint coarsening, and static merge."""

    def __init__(
        self,
        input_width: int,
        output_width: int,
        modes: int,
        *,
        maximum_cycles: float,
        recurrence_backend: RecurrenceBackend,
        gate_sharpness: float,
    ) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            input_width, input_width, 3, padding=1, groups=input_width
        )
        self.local_norm = nn.RMSNorm(input_width)
        self.bank = ProductPoleBank2D(
            input_width,
            modes,
            maximum_cycles=maximum_cycles,
            recurrence_backend=recurrence_backend,
        )
        self.direction_logits = nn.Parameter(torch.zeros(modes, 4))
        self.output_projection = nn.Linear(input_width, output_width)
        self.output_norm = nn.LayerNorm(output_width)
        self.gate_sharpness = gate_sharpness

    def _carry_gate(self, height: int, width: int) -> Tensor:
        spacing_x, spacing_y = 1.0 / width, 1.0 / height
        gate_x = torch.sigmoid(
            self.gate_sharpness
            * (torch.pi / 2.0 - self.bank.frequency_x.abs() * spacing_x)
        )
        gate_y = torch.sigmoid(
            self.gate_sharpness
            * (torch.pi / 2.0 - self.bank.frequency_y.abs() * spacing_y)
        )
        return (gate_x * gate_y).view(1, 1, 1, -1)

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        local = self.depthwise(inputs.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        states = self.bank(self.local_norm(functional.silu(local)))
        descriptor = global_modal_moments(states)
        gate = self._carry_gate(inputs.shape[1], inputs.shape[2])
        endpoints = []
        for (direction_x, direction_y), (real, imag) in zip(
            _DIRECTIONS, states, strict=True
        ):
            coarse_real, coarse_imag = direction_aligned_endpoints(
                real,
                imag,
                direction_x=direction_x,
                direction_y=direction_y,
            )
            endpoints.append((gate * coarse_real, gate * coarse_imag))
        merged_real, merged_imag = _static_merge(tuple(endpoints), self.direction_logits)
        synthesized = self.bank.synthesize(merged_real, merged_imag)
        projected = self.output_projection(synthesized)
        return functional.silu(self.output_norm(projected)), descriptor


class DeepCascadedPoleReader2D(nn.Module):
    """Independent local terminal bank exposing moments without synthesis."""

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

    def forward(self, inputs: Tensor) -> Tensor:
        local = self.depthwise(inputs.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        states = self.bank(self.norm(functional.silu(local)))
        return global_modal_moments(states)


@dataclass(frozen=True, slots=True)
class PolePyramidCDeepTinyConfig:
    output_dim: int = 100
    image_size: int = 224
    widths: tuple[int, ...] = (64, 96, 128, 192)
    modes: tuple[int, ...] = (16, 24, 32, 32)
    depths: tuple[int, ...] = (2, 2, 4, 2)
    layer_scale_init: float = 1.0e-3
    drop_path_rate: float = 0.1
    gate_sharpness: float = 8.0
    recurrence_backend: RecurrenceBackend = "auto"

    def validate(self) -> None:
        if self.image_size < 64 or self.image_size % 32:
            message = "image size must be at least 64 and divisible by 32"
            raise ValueError(message)
        if not (len(self.widths) == len(self.modes) == len(self.depths) == 4):
            message = "deep PolePyramid-C-Tiny requires four stages"
            raise ValueError(message)
        if any(
            2 * modes > width
            for width, modes in zip(self.widths, self.modes, strict=True)
        ):
            message = "every stage must satisfy 2M <= D"
            raise ValueError(message)


class PolePyramidCDeepTiny(nn.Module):
    """The prescribed 2+2+4+2 hierarchy with three exact PoleDowns."""

    stages: nn.ModuleList
    downs: nn.ModuleList

    def __init__(self, config: PolePyramidCDeepTinyConfig | None = None) -> None:
        super().__init__()
        active = config or PolePyramidCDeepTinyConfig()
        active.validate()
        self.config = active
        self.stem = OverlappingConvStem(active.widths[0])
        maximum_cycles = (16.0, 8.0, 4.0, 3.0)
        rates = torch.linspace(0.0, active.drop_path_rate, sum(active.depths)).tolist()
        rate_index = 0
        stages = []
        for width, modes, depth, cycles in zip(
            active.widths, active.modes, active.depths, maximum_cycles, strict=True
        ):
            blocks = []
            for _ in range(depth):
                blocks.append(
                    StaticDirectionPoleBlock2D(
                        width,
                        modes,
                        maximum_cycles=cycles,
                        recurrence_backend=active.recurrence_backend,
                        layer_scale_init=active.layer_scale_init,
                        drop_path=rates[rate_index],
                    )
                )
                rate_index += 1
            stages.append(nn.ModuleList(blocks))
        self.stages = nn.ModuleList(stages)
        self.downs = nn.ModuleList(
            [
                DeepPoleDown2D(
                    input_width,
                    output_width,
                    modes,
                    maximum_cycles=cycles,
                    recurrence_backend=active.recurrence_backend,
                    gate_sharpness=active.gate_sharpness,
                )
                for input_width, output_width, modes, cycles in zip(
                    active.widths[:-1],
                    active.widths[1:],
                    active.modes[:-1],
                    maximum_cycles[:-1],
                    strict=True,
                )
            ]
        )
        self.reader = DeepCascadedPoleReader2D(
            active.widths[-1],
            active.modes[-1],
            maximum_cycles=maximum_cycles[-1],
            recurrence_backend=active.recurrence_backend,
        )
        self.descriptor_dim = 4 * 9 * (sum(active.modes[:-1]) + active.modes[-1])
        self.classifier = nn.Linear(self.descriptor_dim, active.output_dim)

    def raw_descriptor(self, inputs: Tensor) -> Tensor:
        features = self.stem(inputs)
        descriptors = []
        for stage_index, stage in enumerate(self.stages):
            for block in cast("nn.ModuleList", stage):
                features = cast("StaticDirectionPoleBlock2D", block)(features)
            if stage_index < len(self.downs):
                features, descriptor = cast("DeepPoleDown2D", self.downs[stage_index])(
                    features
                )
                descriptors.append(descriptor)
        descriptors.append(self.reader(features))
        return torch.cat(descriptors, dim=-1)

    def forward(self, inputs: Tensor) -> Tensor:
        descriptor = self.raw_descriptor(inputs)
        with torch.autocast(device_type=descriptor.device.type, enabled=False):
            return self.classifier(descriptor.float())


__all__ = [
    "DeepCascadedPoleReader2D",
    "DeepPoleDown2D",
    "PolePyramidCDeepTiny",
    "PolePyramidCDeepTinyConfig",
    "StaticDirectionPoleBlock2D",
]
