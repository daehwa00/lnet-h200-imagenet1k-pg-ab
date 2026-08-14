from __future__ import annotations

from typing import TYPE_CHECKING, Final

import torch
from torch import Tensor, nn
from torch.nn import functional

if TYPE_CHECKING:
    from .pac_types import PACExperimentConfig

QPRL_MODELS: Final[tuple[str, ...]] = ("pac_qprl_depth2_pyramid_evidence",)


class QPRLClassifier(nn.Module):
    def __init__(self, config: PACExperimentConfig, class_count: int) -> None:
        super().__init__()
        self.carrier_dim = max(config.model_dim, 2 * config.modes)
        self.modes = config.modes
        self.stem = _Stem(config.raw_input_dim, self.carrier_dim)
        total_dim = self.carrier_dim + 2 * self.modes
        tap_size = min(config.tap_kernel_size, 5)
        self.blocks = nn.ModuleList(
            _QPRLBlock(total_dim, self.carrier_dim, self.modes, tap_size) for _ in range(2)
        )
        self.head = _QPRLHead(total_dim, self.carrier_dim, self.modes, class_count)

    def forward(self, inputs: Tensor) -> Tensor:
        carrier = self.stem(inputs)
        zeros = carrier.new_zeros(carrier.shape[0], carrier.shape[1], 2 * self.modes)
        features = torch.cat((carrier, zeros), dim=-1)
        for block in self.blocks:
            features = block(features)
        return self.head(features)


class _QPRLBlock(nn.Module):
    def __init__(self, total_dim: int, carrier_dim: int, modes: int, tap_size: int) -> None:
        super().__init__()
        self.carrier_dim = carrier_dim
        self.modes = modes
        self.tap_size = tap_size
        rank = max(4, min(16, total_dim // 2))
        energy_rank = max(4, min(8, modes))
        writer_rank = max(4, min(16, 2 * modes))
        self.norm = _PairRMSNorm(carrier_dim, modes)
        self.controller = nn.Linear(total_dim, rank)
        self.damping_head = nn.Linear(rank, modes)
        self.write_head = nn.Linear(rank, modes)
        self.read_head = nn.Linear(rank, modes)
        self.reader_real = nn.Parameter(0.02 * torch.randn(modes, total_dim))
        self.reader_imag = nn.Parameter(0.02 * torch.randn(modes, total_dim))
        self.tap_logits = nn.Parameter(torch.zeros(modes, tap_size))
        self.raw_decay = nn.Parameter(torch.linspace(-2.0, 1.0, modes))
        self.raw_frequency = nn.Parameter(torch.linspace(0.0, 0.7, modes))
        self.energy_v = nn.Parameter(0.02 * torch.randn(modes, energy_rank))
        self.energy_u = nn.Parameter(0.02 * torch.randn(energy_rank, modes))
        self.writer = nn.Sequential(
            nn.Linear(2 * modes, writer_rank), nn.Linear(writer_rank, carrier_dim)
        )
        self.carrier_scale = nn.Parameter(torch.full((carrier_dim,), 1.0e-2))
        self.modal_scale = nn.Parameter(torch.full((modes,), 1.0e-2))

    def forward(self, inputs: Tensor) -> Tensor:
        normalized = self.norm(inputs)
        control = functional.silu(self.controller(normalized))
        damping = _bounded_damping(self.raw_decay, torch.tanh(self.damping_head(control)))
        frequency = torch.pi * torch.tanh(self.raw_frequency).view(1, 1, self.modes)
        decay_real, decay_imag, gamma_real, gamma_imag = _discretize(damping, frequency)
        drive_real, drive_imag = self._drive(normalized)
        write_gate = 2.0 * torch.sigmoid(self.write_head(control))
        input_real = write_gate * (gamma_real * drive_real - gamma_imag * drive_imag)
        input_imag = write_gate * (gamma_real * drive_imag + gamma_imag * drive_real)
        states_real, states_imag = _recurrence(decay_real, decay_imag, input_real, input_imag)
        energy = torch.log1p(states_real.square() + states_imag.square())
        read_gate = 2.0 * torch.sigmoid(
            self.read_head(control)
            + torch.matmul(torch.matmul(energy, self.energy_v), self.energy_u)
        )
        gated_real = read_gate * states_real
        gated_imag = read_gate * states_imag
        carrier, slots_real, slots_imag = _split(inputs, self.carrier_dim, self.modes)
        carrier_update = self.writer(torch.cat((gated_real, gated_imag), dim=-1))
        next_carrier = carrier + self.carrier_scale.view(1, 1, -1) * carrier_update
        scale = self.modal_scale.view(1, 1, -1)
        return torch.cat(
            (next_carrier, slots_real + scale * gated_real, slots_imag + scale * gated_imag), dim=-1
        )

    def _drive(self, normalized: Tensor) -> tuple[Tensor, Tensor]:
        instant_real = torch.einsum("bnd,md->bnm", normalized, self.reader_real)
        instant_imag = torch.einsum("bnd,md->bnm", normalized, self.reader_imag)
        weights = torch.softmax(self.tap_logits, dim=-1).to(
            device=normalized.device, dtype=normalized.dtype
        )
        kernel = torch.flip(weights, dims=(-1,)).view(self.modes, 1, self.tap_size)
        return (
            _tap(instant_real, kernel, self.tap_size),
            _tap(instant_imag, kernel, self.tap_size),
        )


class _Stem(nn.Module):
    def __init__(self, input_dim: int, carrier_dim: int) -> None:
        super().__init__()
        self.depthwise = _CausalConv(input_dim, input_dim, 5, input_dim)
        self.pointwise = nn.Linear(input_dim, carrier_dim)
        self.norm = nn.LayerNorm(carrier_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        return functional.silu(self.norm(self.pointwise(self.depthwise(inputs))))


class _CausalConv(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, kernel_size: int, groups: int) -> None:
        super().__init__()
        self.left_pad = kernel_size - 1
        self.conv = nn.Conv1d(input_dim, output_dim, kernel_size, groups=groups)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(functional.pad(inputs.transpose(1, 2), (self.left_pad, 0))).transpose(1, 2)


class _PairRMSNorm(nn.Module):
    def __init__(self, carrier_dim: int, modes: int, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.carrier_weight = nn.Parameter(torch.ones(carrier_dim))
        self.mode_weight = nn.Parameter(torch.ones(modes))
        self.carrier_dim = carrier_dim
        self.modes = modes
        self.eps = eps

    def forward(self, inputs: Tensor) -> Tensor:
        carrier, real, imag = _split(inputs, self.carrier_dim, self.modes)
        scale = torch.rsqrt(inputs.square().mean(dim=-1, keepdim=True) + self.eps)
        return torch.cat(
            (
                carrier * scale * self.carrier_weight,
                real * scale * self.mode_weight,
                imag * scale * self.mode_weight,
            ),
            dim=-1,
        )


class _QPRLHead(nn.Module):
    def __init__(self, total_dim: int, carrier_dim: int, modes: int, class_count: int) -> None:
        super().__init__()
        self.carrier_dim = carrier_dim
        self.modes = modes
        input_dim = 7 * (carrier_dim + 3 * modes)
        hidden = max(32, min(128, input_dim // 2))
        self.global_head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, class_count),
        )
        self.timestep_head = nn.Linear(total_dim, class_count)
        self.local_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, inputs: Tensor) -> Tensor:
        return self.global_head(
            _pyramid(inputs, self.carrier_dim, self.modes)
        ) + self.local_scale * torch.logsumexp(self.timestep_head(inputs), dim=1)


def build_qprl_classifier(
    name: str, config: PACExperimentConfig, class_count: int
) -> nn.Module | None:
    if name not in QPRL_MODELS:
        return None
    return QPRLClassifier(config, class_count)


def _split(inputs: Tensor, carrier_dim: int, modes: int) -> tuple[Tensor, Tensor, Tensor]:
    carrier = inputs[..., :carrier_dim]
    real = inputs[..., carrier_dim : carrier_dim + modes]
    imag = inputs[..., carrier_dim + modes : carrier_dim + 2 * modes]
    return carrier, real, imag


def _bounded_damping(raw_decay: Tensor, control: Tensor) -> Tensor:
    return 1.0e-3 + (2.0 - 1.0e-3) * torch.sigmoid(raw_decay.view(1, 1, -1) + control)


def _discretize(damping: Tensor, frequency: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    radius = torch.exp(-damping)
    decay_real = radius * torch.cos(frequency)
    decay_imag = radius * torch.sin(frequency)
    denom = (damping.square() + frequency.square()).clamp_min(1.0e-6)
    gamma_real = (damping * (1.0 - decay_real) + frequency * decay_imag) / denom
    gamma_imag = (-frequency * (decay_real - 1.0) - damping * decay_imag) / denom
    return decay_real, decay_imag, gamma_real, gamma_imag


def _tap(instant: Tensor, kernel: Tensor, tap_size: int) -> Tensor:
    by_mode = instant.transpose(1, 2)
    return functional.conv1d(
        functional.pad(by_mode, (tap_size - 1, 0)), kernel, groups=instant.shape[-1]
    ).transpose(1, 2)


def _recurrence(
    decay_real: Tensor, decay_imag: Tensor, input_real: Tensor, input_imag: Tensor
) -> tuple[Tensor, Tensor]:
    state_real = torch.zeros_like(input_real[:, 0])
    state_imag = torch.zeros_like(input_imag[:, 0])
    real_states: list[Tensor] = []
    imag_states: list[Tensor] = []
    for time_index in range(input_real.shape[1]):
        previous_real = state_real
        state_real = (
            decay_real[:, time_index] * state_real
            - decay_imag[:, time_index] * state_imag
            + input_real[:, time_index]
        )
        state_imag = (
            decay_imag[:, time_index] * previous_real
            + decay_real[:, time_index] * state_imag
            + input_imag[:, time_index]
        )
        real_states.append(state_real)
        imag_states.append(state_imag)
    return torch.stack(real_states, dim=1), torch.stack(imag_states, dim=1)


def _pyramid(inputs: Tensor, carrier_dim: int, modes: int) -> Tensor:
    pieces: list[Tensor] = []
    for segments in (1, 2, 4):
        for chunk in torch.tensor_split(inputs, segments, dim=1):
            carrier, real, imag = _split(chunk, carrier_dim, modes)
            pieces.extend((carrier.mean(dim=1), real.mean(dim=1), imag.mean(dim=1)))
            pieces.append(torch.log1p(real.square() + imag.square()).mean(dim=1))
    return torch.cat(pieces, dim=-1)
