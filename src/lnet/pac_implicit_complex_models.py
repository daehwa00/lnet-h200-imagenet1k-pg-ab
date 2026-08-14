from __future__ import annotations

from typing import TYPE_CHECKING, Final, Literal

import torch
from torch import Tensor, nn
from torch.nn import functional

from .pac_real2d_math import discrete_pole_real2d
from .pac_recurrence import recurrence_real2d

if TYPE_CHECKING:
    from .pac_types import PACExperimentConfig

Direction = Literal["forward", "backward"]
IMPLICIT_COMPLEX_MODELS: Final[tuple[str, ...]] = ("pac_implicit_complex_depth2_pyramid_terminal",)


class ImplicitComplexConfigError(ValueError):
    def __init__(self, model_dim: int) -> None:
        self.model_dim = model_dim
        super().__init__(f"model_dim must be at least 4, got {model_dim}")


class ImplicitComplexClassifier(nn.Module):
    def __init__(self, config: PACExperimentConfig, class_count: int) -> None:
        super().__init__()
        if config.model_dim < 4:
            raise ImplicitComplexConfigError(config.model_dim)
        self.model_dim = config.model_dim
        self.modes = max(1, min(config.modes, config.model_dim // 4))
        self.directions: tuple[Direction, Direction] = ("forward", "backward")
        self.stem = _CausalStem(config.raw_input_dim, self.model_dim)
        self.forward_block = _ImplicitComplexBlock(self.model_dim, self.modes, "forward")
        self.backward_block = _ImplicitComplexBlock(self.model_dim, self.modes, "backward")
        self.final_norm = nn.RMSNorm(self.model_dim)
        self.head = _ImplicitComplexHead(self.model_dim, self.modes, class_count)

    def forward(self, inputs: Tensor) -> Tensor:
        features = self.stem(inputs)
        features, forward_terminal = self.forward_block(features)
        features, backward_terminal = self.backward_block(features)
        return self.head(self.final_norm(features), [forward_terminal, backward_terminal])


class _ImplicitComplexBlock(nn.Module):
    def __init__(self, model_dim: int, modes: int, direction: Direction) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.modes = modes
        self.direction = direction
        self.norm = nn.RMSNorm(model_dim)
        self.local = nn.Conv1d(model_dim, 2 * model_dim, 5, groups=model_dim)
        frame, _ = torch.linalg.qr(torch.randn(model_dim, 2 * modes), mode="reduced")
        self.writer_real = nn.Parameter(frame[:, :modes].contiguous())
        self.writer_imag = nn.Parameter(frame[:, modes:].contiguous())
        initial_decay = torch.logspace(-1.3, 0.3, modes)
        self.raw_decay = nn.Parameter(torch.log(torch.expm1(initial_decay)))
        frequency_grid = torch.linspace(0.0, 0.75, modes).clamp(max=0.999)
        self.raw_frequency = nn.Parameter(torch.atanh(frequency_grid))
        self.direct_scale = nn.Parameter(torch.zeros(model_dim))
        self.layer_scale = nn.Parameter(torch.full((model_dim,), 1.0e-2))

    def complex_frame(self) -> Tensor:
        return torch.cat((self.writer_real, self.writer_imag), dim=1)

    def forward(self, inputs: Tensor) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        work = torch.flip(inputs, dims=(1,)) if self.direction == "backward" else inputs
        normalized = self.norm(work)
        main, gate_logits = self._local_features(normalized)
        drive = torch.matmul(main, self.writer_real)
        damping = (1.0e-3 + functional.softplus(self.raw_decay)).view(1, 1, -1)
        damping = damping.expand_as(drive)
        frequency = torch.pi * torch.tanh(self.raw_frequency).view(1, 1, -1).expand_as(damping)
        decay_real, decay_imag, gamma_real, gamma_imag = discrete_pole_real2d(
            damping, frequency, 1.0
        )
        states_real, states_imag = recurrence_real2d(
            decay_real,
            decay_imag,
            gamma_real * drive,
            gamma_imag * drive,
            "auto",
        )
        modal = 2.0 * (
            torch.matmul(states_real, self.writer_real.transpose(0, 1))
            - torch.matmul(states_imag, self.writer_imag.transpose(0, 1))
        )
        gate = 2.0 * torch.sigmoid(gate_logits)
        update = gate * (modal + self.direct_scale.view(1, 1, -1) * main)
        output = work + self.layer_scale.view(1, 1, -1) * update
        if self.direction == "backward":
            output = torch.flip(output, dims=(1,))
        return output, (states_real[:, -1], states_imag[:, -1])

    def _local_features(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        convolved = self.local(
            functional.pad(inputs.transpose(1, 2), (self.local.kernel_size[0] - 1, 0))
        )
        paired = convolved.view(convolved.shape[0], self.model_dim, 2, convolved.shape[-1])
        main, gate = paired.unbind(dim=2)
        return functional.silu(main.transpose(1, 2)), gate.transpose(1, 2)


class _CausalStem(nn.Module):
    def __init__(self, input_dim: int, model_dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(input_dim, model_dim, 9, stride=2)

    def forward(self, inputs: Tensor) -> Tensor:
        padded = functional.pad(inputs.transpose(1, 2), (self.conv.kernel_size[0] - 1, 0))
        return functional.silu(self.conv(padded).transpose(1, 2))


class _ImplicitComplexHead(nn.Module):
    def __init__(self, model_dim: int, modes: int, class_count: int) -> None:
        super().__init__()
        self.pool = nn.Linear(7 * model_dim, class_count)
        self.phase_real = nn.Parameter(0.02 * torch.randn(2, modes, class_count))
        self.phase_imag = nn.Parameter(0.02 * torch.randn(2, modes, class_count))
        self.energy = nn.Parameter(0.02 * torch.randn(2, modes, class_count))

    def forward(self, inputs: Tensor, terminals: list[tuple[Tensor, Tensor]]) -> Tensor:
        pooled = _ordered_temporal_pool(inputs)
        logits = self.pool(pooled)
        for index, (real, imag) in enumerate(terminals):
            logits = logits + 2.0 * (
                torch.matmul(real, self.phase_real[index])
                - torch.matmul(imag, self.phase_imag[index])
            )
            logits = logits + torch.matmul(real.square() + imag.square(), self.energy[index])
        return logits


def _ordered_temporal_pool(inputs: Tensor) -> Tensor:
    summaries: list[Tensor] = []
    empty = inputs.new_zeros(inputs.shape[0], inputs.shape[2])
    for level in (1, 2, 4):
        summaries.extend(
            chunk.mean(dim=1) if chunk.shape[1] else empty
            for chunk in torch.tensor_split(inputs, level, dim=1)
        )
    return torch.cat(summaries, dim=-1)


def build_implicit_complex_classifier(
    name: str, config: PACExperimentConfig, class_count: int
) -> nn.Module | None:
    if name not in IMPLICIT_COMPLEX_MODELS:
        return None
    return ImplicitComplexClassifier(config, class_count)
