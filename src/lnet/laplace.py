from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional


@dataclass(frozen=True, slots=True)
class LaplaceShapeError(ValueError):
    actual_shape: tuple[int, ...]
    expected_rank: int
    expected_features: int

    def __str__(self) -> str:
        return (
            "expected rank "
            f"{self.expected_rank} with trailing feature size {self.expected_features}, "
            f"received shape {self.actual_shape}"
        )


@dataclass(frozen=True, slots=True)
class LaplaceParameterError(ValueError):
    reason: str

    def __str__(self) -> str:
        return self.reason


def _inverse_softplus(value: Tensor) -> Tensor:
    return torch.log(torch.expm1(value))


def _phi1(argument: Tensor) -> Tensor:
    small_mask = argument.abs() < 1.0e-4
    safe_denominator = torch.where(small_mask, torch.ones_like(argument), argument)
    safe_ratio = torch.expm1(argument) / safe_denominator
    series = 1.0 + (argument / 2.0) + (argument.square() / 6.0)
    return torch.where(small_mask, series, safe_ratio)


def _resolve_activation(name: Literal["identity", "tanh"]) -> nn.Module:
    match name:
        case "identity":
            return nn.Identity()
        case "tanh":
            return nn.Tanh()
        case unreachable:
            raise LaplaceParameterError(reason=f"unsupported activation: {unreachable}")


class PoleResidueTemporalMixing(nn.Module):
    def __init__(
        self,
        model_dim: int,
        modes: int,
        *,
        dt: float = 1.0,
        min_decay: float = 1.0e-3,
    ) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.modes = modes
        self.dt = dt
        self.min_decay = min_decay

        initial_decay = torch.linspace(0.2, 1.0, modes, dtype=torch.float64)
        self.raw_decay = nn.Parameter(_inverse_softplus(initial_decay))
        self.residues = nn.Parameter(
            0.05 * torch.randn(model_dim, modes, model_dim, dtype=torch.float64),
        )
        self.direct_term = nn.Parameter(torch.zeros(model_dim, model_dim, dtype=torch.float64))
        self.bias = nn.Parameter(torch.zeros(model_dim, dtype=torch.float64))

    def continuous_poles(self) -> Tensor:
        return -(functional.softplus(self.raw_decay) + self.min_decay)

    def load_transfer_parameters(
        self,
        *,
        poles: Tensor,
        residues: Tensor,
        direct_term: Tensor,
        bias: Tensor,
    ) -> None:
        expected_residue_shape = (self.model_dim, self.modes, self.model_dim)
        if residues.shape != expected_residue_shape:
            raise LaplaceParameterError(
                reason=(
                    "residue tensor shape mismatch: expected "
                    f"{expected_residue_shape}, received {tuple(residues.shape)}"
                ),
            )
        if direct_term.shape != (self.model_dim, self.model_dim):
            raise LaplaceParameterError(
                reason=(
                    "direct term shape mismatch: expected "
                    f"{(self.model_dim, self.model_dim)}, received {tuple(direct_term.shape)}"
                ),
            )
        if bias.shape != (self.model_dim,):
            raise LaplaceParameterError(
                reason=(
                    f"bias shape mismatch: expected {(self.model_dim,)}, "
                    f"received {tuple(bias.shape)}"
                ),
            )
        if poles.shape != (self.modes,):
            raise LaplaceParameterError(
                reason=(
                    f"pole shape mismatch: expected {(self.modes,)}, received {tuple(poles.shape)}"
                ),
            )
        if not torch.all(poles < 0.0):
            raise LaplaceParameterError(reason="all poles must be strictly negative")

        with torch.no_grad():
            self.raw_decay.copy_(_inverse_softplus((-poles) - self.min_decay))
            self.residues.copy_(residues)
            self.direct_term.copy_(direct_term)
            self.bias.copy_(bias)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 3:
            raise LaplaceShapeError(
                actual_shape=tuple(inputs.shape),
                expected_rank=3,
                expected_features=self.model_dim,
            )
        if inputs.shape[-1] != self.model_dim:
            raise LaplaceShapeError(
                actual_shape=tuple(inputs.shape),
                expected_rank=3,
                expected_features=self.model_dim,
            )

        inputs_64 = inputs.to(dtype=torch.float64)
        poles = self.continuous_poles()
        scaled_poles = poles * self.dt
        discrete_decay = torch.exp(scaled_poles).view(1, self.modes, 1)
        discrete_drive = (self.dt * _phi1(scaled_poles)).view(1, self.modes, 1)

        state = torch.zeros(
            inputs_64.shape[0],
            self.modes,
            self.model_dim,
            dtype=torch.float64,
            device=inputs_64.device,
        )
        outputs: list[Tensor] = []
        for current_input in inputs_64.unbind(dim=1):
            state = (discrete_decay * state) + (discrete_drive * current_input.unsqueeze(1))
            modal_output = torch.einsum("omi,bmi->bo", self.residues, state)
            direct_output = torch.matmul(current_input, self.direct_term.transpose(0, 1))
            outputs.append(modal_output + direct_output + self.bias)
        return torch.stack(outputs, dim=1).to(dtype=inputs.dtype)


class ProjectedPRLBlock(nn.Module):
    def __init__(
        self,
        *,
        raw_input_dim: int,
        model_dim: int,
        output_dim: int,
        modes: int,
        activation: Literal["identity", "tanh"] = "tanh",
        dt: float = 1.0,
    ) -> None:
        super().__init__()
        self.raw_input_dim = raw_input_dim
        self.model_dim = model_dim
        self.output_dim = output_dim
        self.modes = modes
        self.input_projection = nn.Linear(raw_input_dim, model_dim)
        self.temporal_mixer = PoleResidueTemporalMixing(model_dim, modes, dt=dt)
        self.activation_name = activation
        self.activation = _resolve_activation(activation)
        self.output_projection = nn.Linear(model_dim, output_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 3:
            raise LaplaceShapeError(
                actual_shape=tuple(inputs.shape),
                expected_rank=3,
                expected_features=self.raw_input_dim,
            )
        if inputs.shape[-1] != self.raw_input_dim:
            raise LaplaceShapeError(
                actual_shape=tuple(inputs.shape),
                expected_rank=3,
                expected_features=self.raw_input_dim,
            )

        projected = self.input_projection(inputs)
        mixed = self.temporal_mixer(projected)
        activated = self.activation(mixed)
        return self.output_projection(activated)
