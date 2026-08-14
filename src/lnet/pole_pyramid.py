"""Exact endpoint coarsening for separable two-dimensional pole transport."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import torch
from torch import Tensor, nn
from torch.nn import functional

from .alphabet2d import product_pole_scan_2d
from .pac_real2d_math import discrete_pole_real2d
from .pac_recurrence import recurrence_real2d_directional
from .spatialalphabet_h import regional_pole_moments

if TYPE_CHECKING:
    from .pac_recurrence import RecurrenceBackend

PyramidTransport = Literal["pole", "average"]
_QUADRANTS = ((1, 1), (-1, 1), (1, -1), (-1, -1))
_LAGS = ((1, 0), (0, 1), (1, 1), (1, -1))


def _complex_multiply(
    left_real: Tensor,
    left_imag: Tensor,
    right_real: Tensor,
    right_imag: Tensor,
) -> tuple[Tensor, Tensor]:
    return (
        left_real * right_real - left_imag * right_imag,
        left_real * right_imag + left_imag * right_real,
    )


def _axis_scan(
    input_real: Tensor,
    input_imag: Tensor,
    decay_real: Tensor,
    decay_imag: Tensor,
    *,
    direction: int,
    recurrence_backend: RecurrenceBackend,
) -> tuple[Tensor, Tensor]:
    expanded_real = decay_real.view(1, 1, -1).expand_as(input_real)
    expanded_imag = decay_imag.view(1, 1, -1).expand_as(input_imag)
    return recurrence_real2d_directional(
        expanded_real,
        expanded_imag,
        input_real,
        input_imag,
        recurrence_backend,
        "forward" if direction == 1 else "backward",
    )


def _complex_powers(
    real: Tensor,
    imag: Tensor,
    maximum: int,
) -> tuple[tuple[Tensor, Tensor], ...]:
    powers = [(torch.ones_like(real), torch.zeros_like(imag))]
    for _ in range(maximum):
        powers.append(
            _complex_multiply(
                powers[-1][0],
                powers[-1][1],
                real,
                imag,
            )
        )
    return tuple(powers)


def _strided_axis_scan(
    input_real: Tensor,
    input_imag: Tensor,
    decay_real: Tensor,
    decay_imag: Tensor,
    *,
    stride: int,
    direction: int,
    recurrence_backend: RecurrenceBackend,
) -> tuple[Tensor, Tensor]:
    """Coarsen a driven recurrence while preserving selected endpoint states."""
    if input_real.shape != input_imag.shape or input_real.ndim != 3:
        message = "axis inputs must share [N,L,M] shape"
        raise ValueError(message)
    if stride < 1 or input_real.shape[1] % stride:
        message = "axis length must be divisible by the positive stride"
        raise ValueError(message)
    if direction not in {-1, 1}:
        message = "scan direction must be -1 or 1"
        raise ValueError(message)
    batches, length, modes = input_real.shape
    blocks = length // stride
    grouped_real = input_real.reshape(batches, blocks, stride, modes)
    grouped_imag = input_imag.reshape(batches, blocks, stride, modes)
    powers = _complex_powers(decay_real, decay_imag, stride)
    block_real = torch.zeros_like(grouped_real[:, :, 0])
    block_imag = torch.zeros_like(grouped_imag[:, :, 0])
    for position in range(stride):
        exponent = stride - 1 - position if direction == 1 else position
        weighted_real, weighted_imag = _complex_multiply(
            grouped_real[:, :, position],
            grouped_imag[:, :, position],
            powers[exponent][0],
            powers[exponent][1],
        )
        block_real = block_real + weighted_real
        block_imag = block_imag + weighted_imag
    return _axis_scan(
        block_real,
        block_imag,
        powers[stride][0],
        powers[stride][1],
        direction=direction,
        recurrence_backend=recurrence_backend,
    )


def strided_product_pole_scan_2d(
    excitation_real: Tensor,
    excitation_imag: Tensor,
    *,
    damping_x: Tensor,
    damping_y: Tensor,
    frequency_x: Tensor,
    frequency_y: Tensor,
    spacing_x: float,
    spacing_y: float,
    stride: int,
    direction_x: int,
    direction_y: int,
    recurrence_backend: RecurrenceBackend = "auto",
) -> tuple[Tensor, Tensor]:
    """Compute full-scan block endpoints directly on a coarse two-dimensional grid."""
    if excitation_real.shape != excitation_imag.shape or excitation_real.ndim != 4:
        message = "excitations must share [B,H,W,M] shape"
        raise ValueError(message)
    batch, height, width, modes = excitation_real.shape
    if height % stride or width % stride:
        message = "height and width must be divisible by stride"
        raise ValueError(message)
    decay_x_real, decay_x_imag, gamma_x_real, gamma_x_imag = discrete_pole_real2d(
        damping_x, frequency_x, spacing_x
    )
    drive_x_real, drive_x_imag = _complex_multiply(
        excitation_real,
        excitation_imag,
        gamma_x_real,
        gamma_x_imag,
    )
    horizontal_real, horizontal_imag = _strided_axis_scan(
        drive_x_real.reshape(batch * height, width, modes),
        drive_x_imag.reshape(batch * height, width, modes),
        decay_x_real,
        decay_x_imag,
        stride=stride,
        direction=direction_x,
        recurrence_backend=recurrence_backend,
    )
    coarse_width = width // stride
    horizontal_real = horizontal_real.reshape(
        batch,
        height,
        coarse_width,
        modes,
    )
    horizontal_imag = horizontal_imag.reshape(
        batch,
        height,
        coarse_width,
        modes,
    )

    decay_y_real, decay_y_imag, gamma_y_real, gamma_y_imag = discrete_pole_real2d(
        damping_y, frequency_y, spacing_y
    )
    drive_y_real, drive_y_imag = _complex_multiply(
        horizontal_real,
        horizontal_imag,
        gamma_y_real,
        gamma_y_imag,
    )
    vertical_real, vertical_imag = _strided_axis_scan(
        drive_y_real.permute(0, 2, 1, 3).reshape(
            batch * coarse_width,
            height,
            modes,
        ),
        drive_y_imag.permute(0, 2, 1, 3).reshape(
            batch * coarse_width,
            height,
            modes,
        ),
        decay_y_real,
        decay_y_imag,
        stride=stride,
        direction=direction_y,
        recurrence_backend=recurrence_backend,
    )
    coarse_height = height // stride
    output_shape = (batch, coarse_width, coarse_height, modes)
    return (
        vertical_real.reshape(output_shape).permute(0, 2, 1, 3).contiguous(),
        vertical_imag.reshape(output_shape).permute(0, 2, 1, 3).contiguous(),
    )


def decimated_product_pole_scan_2d(
    excitation_real: Tensor,
    excitation_imag: Tensor,
    *,
    damping_x: Tensor,
    damping_y: Tensor,
    frequency_x: Tensor,
    frequency_y: Tensor,
    spacing_x: float,
    spacing_y: float,
    stride: int,
    direction_x: int,
    direction_y: int,
    recurrence_backend: RecurrenceBackend = "auto",
) -> tuple[Tensor, Tensor]:
    """Reference full scan followed by direction-aligned endpoint selection."""
    full_real, full_imag = product_pole_scan_2d(
        excitation_real,
        excitation_imag,
        damping_x=damping_x,
        damping_y=damping_y,
        frequency_x=frequency_x,
        frequency_y=frequency_y,
        spacing_x=spacing_x,
        spacing_y=spacing_y,
        direction_x=direction_x,
        direction_y=direction_y,
        recurrence_backend=recurrence_backend,
    )
    start_x = stride - 1 if direction_x == 1 else 0
    start_y = stride - 1 if direction_y == 1 else 0
    return (
        full_real[:, start_y::stride, start_x::stride],
        full_imag[:, start_y::stride, start_x::stride],
    )


@dataclass(frozen=True, slots=True)
class PolePyramidConfig:
    input_channels: int = 3
    output_dim: int = 100
    image_size: int = 32
    widths: tuple[int, ...] = (32, 48, 64, 96, 128)
    modes: tuple[int, ...] = (8, 8, 16, 16)
    transport: PyramidTransport = "pole"
    recurrence_backend: RecurrenceBackend = "auto"

    def validate(self) -> None:
        if len(self.widths) != len(self.modes) + 1:
            message = "widths must contain one more entry than modes"
            raise ValueError(message)
        if self.image_size % (2 ** len(self.modes)):
            message = "image size must be divisible by the pyramid stride"
            raise ValueError(message)
        if any(mode < 8 or mode % 8 for mode in self.modes):
            message = "each stage mode count must be divisible by eight"
            raise ValueError(message)
        if self.transport not in {"pole", "average"}:
            message = f"unsupported pyramid transport: {self.transport}"
            raise ValueError(message)


def _pole_atlas(modes: int, grid_size: int) -> tuple[Tensor, Tensor, Tensor]:
    levels = modes // 4
    maximum_cycles = max(0.5, grid_size / 8.0)
    radial_cycles = torch.logspace(
        math.log10(0.25),
        math.log10(maximum_cycles),
        levels,
    ).repeat_interleave(4)
    orientations = torch.tensor((0.0, math.pi / 4.0, math.pi / 2.0, 3.0 * math.pi / 4.0)).repeat(
        levels
    )
    radial_frequency = 2.0 * math.pi * radial_cycles
    damping = (radial_frequency / 2.0).clamp_min(0.5)
    return (
        damping,
        radial_frequency * torch.cos(orientations),
        radial_frequency * torch.sin(orientations),
    )


class PoleDown2D(nn.Module):
    """One exact endpoint-coarsening stage with an average-pool control."""

    def __init__(
        self,
        input_width: int,
        output_width: int,
        modes: int,
        *,
        grid_size: int,
        transport: PyramidTransport,
        recurrence_backend: RecurrenceBackend,
    ) -> None:
        super().__init__()
        self.modes = modes
        self.grid_size = grid_size
        self.transport = transport
        self.recurrence_backend: RecurrenceBackend = recurrence_backend
        self.input_norm = nn.RMSNorm(input_width)
        self.analysis = nn.Linear(input_width, 2 * modes, bias=False)
        nn.init.orthogonal_(self.analysis.weight)
        damping, frequency_x, frequency_y = _pole_atlas(modes, grid_size)
        self.register_buffer("base_damping", damping)
        self.register_buffer("base_frequency_x", frequency_x)
        self.register_buffer("base_frequency_y", frequency_y)
        self.base_damping = self.get_buffer("base_damping")
        self.base_frequency_x = self.get_buffer("base_frequency_x")
        self.base_frequency_y = self.get_buffer("base_frequency_y")
        self.log_damping_offset = nn.Parameter(torch.zeros(modes))
        self.frequency_offset_x = nn.Parameter(torch.zeros(modes))
        self.frequency_offset_y = nn.Parameter(torch.zeros(modes))
        learned = ((torch.arange(modes) // 4) % 2).to(torch.float32)
        self.register_buffer("learned_mask", learned)
        self.learned_mask = self.get_buffer("learned_mask")
        angles = torch.arange(4, dtype=torch.float32) * (math.pi / 2.0)
        self.direction_gain_real = nn.Parameter(
            torch.cos(angles).view(4, 1).expand(4, modes).clone()
        )
        self.direction_gain_imag = nn.Parameter(
            torch.sin(angles).view(4, 1).expand(4, modes).clone()
        )
        low_modes = modes // 2
        self.point_mix = nn.Linear(4 * 2 * low_modes, output_width)
        self.output_norm = nn.RMSNorm(output_width)

    def damping(self) -> Tensor:
        offset = self.learned_mask * torch.tanh(self.log_damping_offset)
        return self.base_damping * torch.exp(math.log(2.0) * offset)

    def frequency_x(self) -> Tensor:
        offset = self.learned_mask * torch.tanh(self.frequency_offset_x)
        return self.base_frequency_x + math.pi * offset

    def frequency_y(self) -> Tensor:
        offset = self.learned_mask * torch.tanh(self.frequency_offset_y)
        return self.base_frequency_y + math.pi * offset

    def _direction_gain(self) -> tuple[Tensor, Tensor]:
        denominator = torch.sqrt(
            self.direction_gain_real.square() + self.direction_gain_imag.square()
        ).clamp_min(1.0e-8)
        return (
            self.direction_gain_real / denominator,
            self.direction_gain_imag / denominator,
        )

    def _average_directions(
        self,
        real: Tensor,
        imag: Tensor,
    ) -> tuple[tuple[Tensor, Tensor], ...]:
        pooled_real = functional.avg_pool2d(
            real.permute(0, 3, 1, 2),
            2,
            2,
        ).permute(0, 2, 3, 1)
        pooled_imag = functional.avg_pool2d(
            imag.permute(0, 3, 1, 2),
            2,
            2,
        ).permute(0, 2, 3, 1)
        gain_real, gain_imag = self._direction_gain()
        return tuple(
            _complex_multiply(
                pooled_real,
                pooled_imag,
                gain_real[index],
                gain_imag[index],
            )
            for index in range(4)
        )

    def _pole_directions(
        self,
        real: Tensor,
        imag: Tensor,
    ) -> tuple[tuple[Tensor, Tensor], ...]:
        gain_real, gain_imag = self._direction_gain()
        outputs = []
        for index, (direction_x, direction_y) in enumerate(_QUADRANTS):
            state_real, state_imag = strided_product_pole_scan_2d(
                real,
                imag,
                damping_x=self.damping(),
                damping_y=self.damping(),
                frequency_x=self.frequency_x(),
                frequency_y=self.frequency_y(),
                spacing_x=1.0 / self.grid_size,
                spacing_y=1.0 / self.grid_size,
                stride=2,
                direction_x=direction_x,
                direction_y=direction_y,
                recurrence_backend=self.recurrence_backend,
            )
            outputs.append(
                _complex_multiply(
                    state_real,
                    state_imag,
                    gain_real[index],
                    gain_imag[index],
                )
            )
        return tuple(outputs)

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        real, imag = self.analysis(self.input_norm(inputs)).chunk(2, dim=-1)
        directions = (
            self._pole_directions(real, imag)
            if self.transport == "pole"
            else self._average_directions(real, imag)
        )
        descriptor = torch.cat(
            [
                regional_pole_moments(
                    state_real,
                    state_imag,
                    lags=_LAGS,
                    edges=(),
                    regional=False,
                )
                for state_real, state_imag in directions
            ],
            dim=-1,
        )
        low_modes = self.modes // 2
        coarse = torch.cat(
            [component[..., :low_modes] for state in directions for component in state],
            dim=-1,
        )
        output = self.output_norm(functional.silu(self.point_mix(coarse)))
        return output, descriptor


class PolePyramid(nn.Module):
    """Convolution-free multiresolution classifier built only from PoleDown stages."""

    blocks: nn.ModuleList

    def __init__(self, config: PolePyramidConfig | None = None) -> None:
        super().__init__()
        active = config or PolePyramidConfig()
        active.validate()
        self.config = active
        self.input_lift = nn.Linear(active.input_channels, active.widths[0])
        self.blocks = nn.ModuleList(
            [
                PoleDown2D(
                    active.widths[index],
                    active.widths[index + 1],
                    modes,
                    grid_size=active.image_size // (2**index),
                    transport=active.transport,
                    recurrence_backend=active.recurrence_backend,
                )
                for index, modes in enumerate(active.modes)
            ]
        )
        descriptor_dim = 4 * 9 * sum(active.modes)
        self.descriptor_dim = descriptor_dim
        self.classifier = nn.Linear(descriptor_dim, active.output_dim)

    def forward_features(self, inputs: Tensor) -> Tensor:
        features = self.input_lift(inputs.permute(0, 2, 3, 1))
        descriptors = []
        for block in self.blocks:
            features, descriptor = cast("PoleDown2D", block)(features)
            descriptors.append(descriptor)
        return torch.cat(descriptors, dim=-1)

    def forward(self, inputs: Tensor) -> Tensor:
        descriptor = self.forward_features(inputs)
        with torch.autocast(device_type=descriptor.device.type, enabled=False):
            return self.classifier(descriptor.float())


__all__ = [
    "PoleDown2D",
    "PolePyramid",
    "PolePyramidConfig",
    "PyramidTransport",
    "decimated_product_pole_scan_2d",
    "strided_product_pole_scan_2d",
]
