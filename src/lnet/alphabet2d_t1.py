"""Hierarchical physical Laplace pyramid with a compact affine modal interface."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal, cast

import torch
from torch import Tensor, nn
from torch.nn import functional
from torch.nn.utils.parametrizations import orthogonal

from .alphabet2d import product_pole_scan_2d

if TYPE_CHECKING:
    from .pac_recurrence import RecurrenceBackend

Transport = Literal["product", "pole_free"]
_QUADRANTS = ((1, 1), (-1, 1), (1, -1), (-1, -1))
_ORIENTED_LAGS = ((1, 0), (1, 1), (0, 1), (-1, 1))


@dataclass(frozen=True, slots=True)
class Alphabet2DT1Config:
    """Frozen compact T1 architecture contract."""

    input_channels: int = 3
    output_dim: int = 100
    image_size: int = 224
    widths: tuple[int, ...] = (48, 64, 96, 128)
    depths: tuple[int, ...] = (1, 1, 2, 2)
    modes: tuple[int, ...] = (8, 8, 12, 12)
    mlp_ratio: float = 2.0
    recurrence_backend: RecurrenceBackend = "auto"
    layer_scale_init: float = 1.0e-2
    transport: Transport = "product"

    def validate(self) -> None:
        stage_count = len(self.widths)
        if stage_count != 4:
            message = "T1 requires exactly four spatial stages"
            raise ValueError(message)
        if len(self.depths) != stage_count or len(self.modes) != stage_count:
            message = "widths, depths, and modes must have identical lengths"
            raise ValueError(message)
        if self.image_size % 4 != 0 or self.image_size < 32:
            message = "image_size must be divisible by four and at least 32"
            raise ValueError(message)
        if min(self.widths, default=0) < 1 or min(self.depths, default=0) < 1:
            message = "stage widths and depths must be positive"
            raise ValueError(message)
        if any(mode < 4 or mode % 4 for mode in self.modes):
            message = "each stage mode count must be a positive multiple of four"
            raise ValueError(message)
        if any(
            width < 2 * mode
            for width, mode in zip(self.widths, self.modes, strict=True)
        ):
            message = "each stage width must be at least twice its mode count"
            raise ValueError(message)
        if self.transport not in {"product", "pole_free"}:
            message = f"unsupported transport: {self.transport}"
            raise ValueError(message)


def _stage_atlas(stage: int, modes: int) -> tuple[Tensor, Tensor, Tensor]:
    cycles_by_stage = (
        (4.0, 12.0),
        (2.0, 6.0),
        (1.0, 3.0, 5.0),
        (0.5, 1.5, 2.5),
    )
    memory_by_stage = (
        (2.0 / 56.0, 6.0 / 56.0),
        (2.0 / 28.0, 6.0 / 28.0),
        (2.0 / 14.0, 5.0 / 14.0, 12.0 / 14.0),
        (2.0 / 7.0, 5.0 / 7.0, 2.0),
    )
    cycles = cycles_by_stage[stage]
    memories = memory_by_stage[stage]
    levels = modes // 4
    if levels > len(cycles) or levels > len(memories):
        message = f"stage {stage} atlas does not match {modes} modes"
        raise ValueError(message)
    cycles = cycles[:levels]
    memories = memories[:levels]
    orientation = torch.arange(modes) % 4
    level = torch.arange(modes) // 4
    angles = orientation * (math.pi / 4.0)
    radial = torch.tensor(cycles)[level]
    damping = torch.tensor([1.0 / memory for memory in memories])[level]
    omega = 2.0 * math.pi * radial
    return damping, omega * torch.cos(angles), omega * torch.sin(angles)


def _complex_rotate(
    real: Tensor,
    imag: Tensor,
    phase: Tensor,
    amplitude: Tensor,
) -> tuple[Tensor, Tensor]:
    cosine = torch.cos(phase) * amplitude
    sine = torch.sin(phase) * amplitude
    return real * cosine - imag * sine, real * sine + imag * cosine


class PhysicalModalField2D(nn.Module):
    """Bounded physical pole atlas with product or pointwise matched transport."""

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
        self.log_damping_offset_x = nn.Parameter(torch.zeros(modes))
        self.log_damping_offset_y = nn.Parameter(torch.zeros(modes))
        self.frequency_offset_x = nn.Parameter(torch.zeros(modes))
        self.frequency_offset_y = nn.Parameter(torch.zeros(modes))
        self.quadrant_mix = nn.Parameter(torch.ones(modes, 4))

    def damping_x(self) -> Tensor:
        return self.base_damping * torch.exp(
            math.log(2.0) * torch.tanh(self.log_damping_offset_x)
        )

    def damping_y(self) -> Tensor:
        return self.base_damping * torch.exp(
            math.log(2.0) * torch.tanh(self.log_damping_offset_y)
        )

    def frequency_x(self) -> Tensor:
        return self.base_frequency_x + 0.5 * math.pi * torch.tanh(
            self.frequency_offset_x
        )

    def frequency_y(self) -> Tensor:
        return self.base_frequency_y + 0.5 * math.pi * torch.tanh(
            self.frequency_offset_y
        )

    def _pointwise_quadrants(
        self,
        excitation_real: Tensor,
        excitation_imag: Tensor,
        *,
        spacing_x: float,
        spacing_y: float,
    ) -> tuple[Tensor, Tensor]:
        amplitude = torch.exp(
            -self.damping_x() * spacing_x - self.damping_y() * spacing_y
        )
        real_states = []
        imag_states = []
        for direction_x, direction_y in _QUADRANTS:
            phase = (
                direction_x * self.frequency_x() * spacing_x
                + direction_y * self.frequency_y() * spacing_y
            )
            real, imag = _complex_rotate(
                excitation_real,
                excitation_imag,
                phase,
                amplitude,
            )
            real_states.append(real)
            imag_states.append(imag)
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
        denominator = self.quadrant_mix.abs().sum(dim=-1, keepdim=True).clamp_min(
            1.0e-6
        )
        mixing = (self.quadrant_mix / denominator).mT.view(
            1,
            4,
            1,
            1,
            self.modes,
        )
        return (
            (quadrant_real * mixing).sum(dim=1),
            (quadrant_imag * mixing).sum(dim=1),
        )

    def synthesize(self, real: Tensor, imag: Tensor) -> Tensor:
        frame_real, frame_imag = self.analysis.weight.chunk(2, dim=0)
        return torch.matmul(real, frame_real) + torch.matmul(imag, frame_imag)

    def audit(self) -> dict[str, Tensor]:
        return {
            "damping_x": self.damping_x().detach(),
            "damping_y": self.damping_y().detach(),
            "frequency_x": self.frequency_x().detach(),
            "frequency_y": self.frequency_y().detach(),
            "quadrant_mix": self.quadrant_mix.detach()
            / self.quadrant_mix.detach().abs().sum(dim=-1, keepdim=True),
        }


class _LaplaceBlock(nn.Module):
    def __init__(
        self,
        width: int,
        modes: int,
        *,
        stage: int,
        config: Alphabet2DT1Config,
    ) -> None:
        super().__init__()
        hidden = round(width * config.mlp_ratio)
        self.local = nn.Conv2d(width, width, 3, padding=1, groups=width)
        self.norm = nn.RMSNorm(width)
        self.field = PhysicalModalField2D(
            width,
            modes,
            stage=stage,
            transport=config.transport,
            recurrence_backend=config.recurrence_backend,
        )
        self.pole_scale = nn.Parameter(torch.full((width,), config.layer_scale_init))
        self.mlp_norm = nn.RMSNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Linear(hidden, width),
        )
        self.mlp_scale = nn.Parameter(torch.full((width,), config.layer_scale_init))

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        local = self.local(inputs.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        normalized = self.norm(functional.silu(local))
        real, imag = self.field(normalized)
        updated = inputs + self.pole_scale * self.field.synthesize(real, imag)
        output = updated + self.mlp_scale * self.mlp(self.mlp_norm(updated))
        return output, real, imag


def _dct_basis(height: int, width: int) -> Tensor:
    y = torch.arange(height, dtype=torch.float32)
    x = torch.arange(width, dtype=torch.float32)
    basis_y = {
        k: torch.cos(math.pi * k * (y + 0.5) / height)
        for k in (0, 1)
    }
    basis_x = {
        k: torch.cos(math.pi * k * (x + 0.5) / width)
        for k in (0, 1)
    }
    basis = torch.stack(
        (
            torch.outer(basis_y[0], basis_x[1]),
            torch.outer(basis_y[1], basis_x[0]),
            torch.outer(basis_y[1], basis_x[1]),
        )
    )
    return basis / basis.square().sum(dim=(-2, -1), keepdim=True).sqrt()


def _oriented_lag_moments(real: Tensor, imag: Tensor) -> Tensor:
    batch, _height, _width, modes = real.shape
    energy = (real.square() + imag.square()).mean(dim=(1, 2))
    pieces = [torch.log1p(energy)]
    correlations_real = real.new_empty((batch, modes))
    correlations_imag = real.new_empty((batch, modes))
    epsilon = max(1.0e-8, torch.finfo(real.dtype).eps)
    for orientation, (delta_x, delta_y) in enumerate(_ORIENTED_LAGS):
        indices = torch.arange(orientation, modes, 4, device=real.device)
        current_y = slice(max(delta_y, 0), real.shape[1] + min(delta_y, 0))
        previous_y = slice(max(-delta_y, 0), real.shape[1] - max(delta_y, 0))
        current_x = slice(max(delta_x, 0), real.shape[2] + min(delta_x, 0))
        previous_x = slice(max(-delta_x, 0), real.shape[2] - max(delta_x, 0))
        current_real = real[:, current_y, current_x, indices]
        current_imag = imag[:, current_y, current_x, indices]
        previous_real = real[:, previous_y, previous_x, indices]
        previous_imag = imag[:, previous_y, previous_x, indices]
        current_energy = (current_real.square() + current_imag.square()).mean(
            dim=(1, 2)
        )
        previous_energy = (previous_real.square() + previous_imag.square()).mean(
            dim=(1, 2)
        )
        denominator = torch.sqrt(
            (current_energy * previous_energy).clamp_min(epsilon * epsilon)
        )
        correlations_real[:, indices] = (
            current_real * previous_real + current_imag * previous_imag
        ).mean(dim=(1, 2)) / denominator
        correlations_imag[:, indices] = (
            current_imag * previous_real - current_real * previous_imag
        ).mean(dim=(1, 2)) / denominator
    pieces.extend((correlations_real, correlations_imag))
    return torch.stack(pieces, dim=-1).reshape(batch, 3 * modes)


def _energy_dct_moments(real: Tensor, imag: Tensor, basis: Tensor) -> Tensor:
    envelope = real.square() + imag.square()
    energy = envelope.mean(dim=(1, 2))
    denominator = energy.clamp_min(max(1.0e-8, torch.finfo(real.dtype).eps))
    dct = torch.einsum(
        "bhwm,khw->bmk",
        envelope,
        basis.to(dtype=envelope.dtype),
    )
    dct = dct / (envelope.shape[1] * envelope.shape[2] * denominator[..., None])
    return torch.cat((torch.log1p(energy)[..., None], dct), dim=-1).flatten(1)


class _CompactStageReader(nn.Module):
    def __init__(
        self,
        width: int,
        modes: int,
        grid_size: int,
        *,
        stage: int,
        config: Alphabet2DT1Config,
    ) -> None:
        super().__init__()
        self.modes = modes
        self.cascade_projection = nn.Linear(modes, width, bias=False)
        self.cascade_field = PhysicalModalField2D(
            width,
            modes,
            stage=stage,
            transport=config.transport,
            recurrence_backend=config.recurrence_backend,
        )
        self.terminal_local = nn.Conv2d(width, width, 3, padding=1, groups=width)
        self.terminal_norm = nn.RMSNorm(width)
        self.terminal_field = PhysicalModalField2D(
            width,
            modes,
            stage=stage,
            transport=config.transport,
            recurrence_backend=config.recurrence_backend,
        )
        self.register_buffer("dct_basis", _dct_basis(grid_size, grid_size))

    def forward(
        self,
        features: Tensor,
        direct_real: Tensor,
        direct_imag: Tensor,
    ) -> Tensor:
        modulus = torch.sqrt(
            direct_real.square() + direct_imag.square() + 1.0e-6
        )
        cascade_input = self.cascade_projection(modulus)
        cascade_real, cascade_imag = self.cascade_field(cascade_input)
        terminal_input = self.terminal_local(
            features.permute(0, 3, 1, 2)
        ).permute(0, 2, 3, 1)
        terminal_input = self.terminal_norm(functional.silu(terminal_input))
        terminal_real, terminal_imag = self.terminal_field(terminal_input)
        return torch.cat(
            (
                _oriented_lag_moments(direct_real, direct_imag),
                _energy_dct_moments(
                    cascade_real,
                    cascade_imag,
                    self.dct_basis,
                ),
                _energy_dct_moments(
                    terminal_real,
                    terminal_imag,
                    self.dct_basis,
                ),
            ),
            dim=-1,
        )


class _AntiAliasedMerge(nn.Module):
    def __init__(self, input_width: int, output_width: int) -> None:
        super().__init__()
        kernel_1d = torch.tensor((1.0, 2.0, 1.0))
        kernel = torch.outer(kernel_1d, kernel_1d)
        kernel = kernel / kernel.sum()
        self.register_buffer(
            "kernel",
            kernel.view(1, 1, 3, 3).expand(input_width, 1, 3, 3).contiguous(),
        )
        self.projection = nn.Conv2d(input_width, output_width, 1)

    def forward(self, inputs: Tensor) -> Tensor:
        channels_first = inputs.permute(0, 3, 1, 2)
        blurred = functional.conv2d(
            channels_first,
            self.kernel,
            stride=2,
            padding=1,
            groups=channels_first.shape[1],
        )
        return self.projection(blurred).permute(0, 2, 3, 1)


class Alphabet2DT1Compact(nn.Module):
    """Four-stage Laplace pyramid with a 440-coordinate affine classifier."""

    def __init__(self, config: Alphabet2DT1Config | None = None) -> None:
        super().__init__()
        active = config or Alphabet2DT1Config()
        active.validate()
        self.config = active
        first_width = active.widths[0]
        stem_width = max(16, first_width // 2)
        self.stem = nn.Sequential(
            nn.Conv2d(active.input_channels, stem_width, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(stem_width, first_width, 3, stride=2, padding=1),
            nn.GELU(),
        )
        grid_sizes = tuple(active.image_size // (4 * (2**stage)) for stage in range(4))
        self.stages = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        _LaplaceBlock(
                            width,
                            modes,
                            stage=stage,
                            config=active,
                        )
                        for _ in range(depth)
                    ]
                )
                for stage, (width, depth, modes) in enumerate(
                    zip(active.widths, active.depths, active.modes, strict=True)
                )
            ]
        )
        self.readers = nn.ModuleList(
            [
                _CompactStageReader(
                    width,
                    modes,
                    grid_size,
                    stage=stage,
                    config=active,
                )
                for stage, (width, modes, grid_size) in enumerate(
                    zip(active.widths, active.modes, grid_sizes, strict=True)
                )
            ]
        )
        self.merges = nn.ModuleList(
            [
                _AntiAliasedMerge(input_width, output_width)
                for input_width, output_width in zip(
                    active.widths[:-1],
                    active.widths[1:],
                    strict=True,
                )
            ]
        )
        self.descriptor_dim = 11 * sum(active.modes)
        self.classifier = nn.Linear(self.descriptor_dim, active.output_dim)

    def forward_features(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        features = self.stem(inputs).permute(0, 2, 3, 1)
        descriptors = []
        for stage_index, blocks in enumerate(self.stages):
            direct_real: Tensor | None = None
            direct_imag: Tensor | None = None
            for block in blocks:
                features, direct_real, direct_imag = cast(
                    "_LaplaceBlock",
                    block,
                )(features)
            if direct_real is None or direct_imag is None:
                message = "every T1 stage requires at least one block"
                raise RuntimeError(message)
            descriptors.append(
                cast("_CompactStageReader", self.readers[stage_index])(
                    features,
                    direct_real,
                    direct_imag,
                )
            )
            if stage_index < len(self.merges):
                features = cast("_AntiAliasedMerge", self.merges[stage_index])(
                    features
                )
        return torch.cat(descriptors, dim=-1), features

    def forward(self, inputs: Tensor) -> Tensor:
        descriptor, _ = self.forward_features(inputs)
        return self.classifier(descriptor)

    def pole_audit(self) -> dict[str, dict[str, Tensor]]:
        audit: dict[str, dict[str, Tensor]] = {}
        for stage_index, blocks in enumerate(self.stages):
            for block_index, block in enumerate(blocks):
                audit[f"stage{stage_index}.block{block_index}"] = cast(
                    "_LaplaceBlock",
                    block,
                ).field.audit()
            reader = cast("_CompactStageReader", self.readers[stage_index])
            audit[f"stage{stage_index}.cascade"] = reader.cascade_field.audit()
            audit[f"stage{stage_index}.terminal"] = reader.terminal_field.audit()
        return audit


def build_alphabet2d_t1(
    variant: Transport = "product",
    config: Alphabet2DT1Config | None = None,
) -> Alphabet2DT1Compact:
    active = config or Alphabet2DT1Config(transport=variant)
    if active.transport != variant:
        active = replace(active, transport=variant)
    return Alphabet2DT1Compact(active)


__all__ = [
    "Alphabet2DT1Compact",
    "Alphabet2DT1Config",
    "PhysicalModalField2D",
    "build_alphabet2d_t1",
]
