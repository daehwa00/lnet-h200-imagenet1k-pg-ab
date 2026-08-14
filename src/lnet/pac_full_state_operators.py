"""Structured complex operators for full-cell transition experiments."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional

from .pac_grouped_path_cffn import GroupedWidelyLinear

ComplexField = tuple[Tensor, Tensor]


def _complex_xavier_(real: Tensor, imag: Tensor) -> None:
    for index in range(real.shape[0]):
        nn.init.xavier_uniform_(real[index])
        nn.init.xavier_uniform_(imag[index])
    with torch.no_grad():
        real.mul_(math.sqrt(0.5))
        imag.mul_(math.sqrt(0.5))


class GroupedComplexLinear(nn.Module):
    """Apply an independent strict complex-linear path map to every mode."""

    def __init__(self, groups: int, input_paths: int, output_paths: int) -> None:
        super().__init__()
        if min(groups, input_paths, output_paths) <= 0:
            message = "grouped complex-linear dimensions must be positive"
            raise ValueError(message)
        self.groups = groups
        self.input_paths = input_paths
        self.output_paths = output_paths
        shape = (groups, output_paths, input_paths)
        self.weight_real = nn.Parameter(torch.empty(shape))
        self.weight_imag = nn.Parameter(torch.empty(shape))
        _complex_xavier_(self.weight_real, self.weight_imag)

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        if (
            real.shape != imag.shape
            or real.ndim < 2
            or tuple(real.shape[-2:]) != (self.input_paths, self.groups)
        ):
            message = "grouped complex-linear inputs must end in path-group axes"
            raise ValueError(message)
        real_real = torch.einsum("...ig,goi->...og", real, self.weight_real)
        imag_imag = torch.einsum("...ig,goi->...og", imag, self.weight_imag)
        real_imag = torch.einsum("...ig,goi->...og", real, self.weight_imag)
        imag_real = torch.einsum("...ig,goi->...og", imag, self.weight_real)
        return real_real - imag_imag, real_imag + imag_real


class GroupedPhaseGatedComplexFFN(nn.Module):
    """Mode-specific Phase-Gated residuals over a short structured axis."""

    gate_redistribution = 0.5

    def __init__(
        self,
        groups: int,
        modes: int,
        hidden_modes: int,
        *,
        epsilon: float = 1.0e-6,
        alpha_init: float = 0.075,
        residual_scale_init: float = 0.01,
    ) -> None:
        super().__init__()
        if min(groups, modes, hidden_modes) <= 0:
            message = "grouped Phase-Gated dimensions must be positive"
            raise ValueError(message)
        if epsilon <= 0.0 or residual_scale_init <= 0.0:
            message = "grouped Phase-Gated scales must be positive"
            raise ValueError(message)
        self.groups = groups
        self.modes = modes
        self.hidden_modes = hidden_modes
        self.epsilon = float(epsilon)
        self.norm_weight = nn.Parameter(torch.ones(groups, modes))
        input_shape = (groups, 2 * hidden_modes, modes)
        output_shape = (groups, modes, hidden_modes)
        self.input_weight_real = nn.Parameter(torch.empty(input_shape))
        self.input_weight_imag = nn.Parameter(torch.empty(input_shape))
        self.output_weight_real = nn.Parameter(torch.empty(output_shape))
        self.output_weight_imag = nn.Parameter(torch.empty(output_shape))
        _complex_xavier_(self.input_weight_real, self.input_weight_imag)
        _complex_xavier_(self.output_weight_real, self.output_weight_imag)
        self.alpha = nn.Parameter(torch.full((groups, hidden_modes), alpha_init))
        self.gamma = nn.Parameter(torch.full((groups,), residual_scale_init))
        self.gate_mean: Tensor
        self.gate_std: Tensor
        self.update_ratio: Tensor
        self.register_buffer("gate_mean", torch.zeros(()), persistent=False)
        self.register_buffer("gate_std", torch.zeros(()), persistent=False)
        self.register_buffer("update_ratio", torch.zeros(()), persistent=False)
        self.diagnostics_enabled = False

    def _project(
        self,
        real: Tensor,
        imag: Tensor,
        weight_real: Tensor,
        weight_imag: Tensor,
    ) -> ComplexField:
        real_real = torch.einsum("...ig,goi->...og", real, weight_real)
        imag_imag = torch.einsum("...ig,goi->...og", imag, weight_imag)
        real_imag = torch.einsum("...ig,goi->...og", real, weight_imag)
        imag_real = torch.einsum("...ig,goi->...og", imag, weight_real)
        return real_real - imag_imag, real_imag + imag_real

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        if (
            real.shape != imag.shape
            or real.ndim < 2
            or tuple(real.shape[-2:]) != (self.modes, self.groups)
        ):
            message = "grouped Phase-Gated inputs must end in mode-group axes"
            raise ValueError(message)
        energy = real.float().square() + imag.float().square()
        inverse_rms = torch.rsqrt(energy.mean(dim=-2, keepdim=True) + self.epsilon)
        weight = self.norm_weight.transpose(0, 1)
        unit_real = (real.float() * inverse_rms * weight).to(dtype=real.dtype)
        unit_imag = (imag.float() * inverse_rms * weight).to(dtype=imag.dtype)
        projected_real, projected_imag = self._project(
            unit_real,
            unit_imag,
            self.input_weight_real,
            self.input_weight_imag,
        )
        value_real, gate_real = projected_real.split(self.hidden_modes, dim=-2)
        value_imag, gate_imag = projected_imag.split(self.hidden_modes, dim=-2)
        magnitude = torch.log1p(gate_real.float().square() + gate_imag.float().square())
        centered = magnitude - magnitude.mean(dim=-2, keepdim=True)
        relative = 1.0 + self.gate_redistribution * torch.tanh(
            self.alpha.transpose(0, 1).float() * centered
        )
        gate = relative / relative.mean(dim=-2, keepdim=True)
        gated_real = value_real * gate.to(dtype=value_real.dtype)
        gated_imag = value_imag * gate.to(dtype=value_imag.dtype)
        delta_real, delta_imag = self._project(
            gated_real,
            gated_imag,
            self.output_weight_real,
            self.output_weight_imag,
        )
        gamma = self.gamma.to(dtype=delta_real.dtype)
        update_real = delta_real * gamma
        update_imag = delta_imag * gamma
        if self.diagnostics_enabled:
            with torch.no_grad():
                sampled_gate = gate.reshape(-1, self.hidden_modes, self.groups)[:128]
                source_energy = energy.reshape(-1, self.modes, self.groups)[:128].mean()
                update_energy = (
                    (update_real.float().square() + update_imag.float().square())
                    .reshape(-1, self.modes, self.groups)[:128]
                    .mean()
                )
                self.gate_mean.copy_(sampled_gate.mean())
                self.gate_std.copy_(sampled_gate.std(unbiased=False))
                self.update_ratio.copy_(
                    torch.sqrt(update_energy / source_energy.clamp_min(1.0e-12))
                )
        return real + update_real, imag + update_imag

    def diagnostic_metrics(self) -> dict[str, float]:
        return {
            "gate_mean": float(self.gate_mean),
            "gate_std": float(self.gate_std),
            "update_ratio": float(self.update_ratio),
            "alpha_mean": float(self.alpha.detach().float().mean()),
            "gamma_mean": float(self.gamma.detach().float().mean()),
        }


class GroupedWidelyLinearResidual(nn.Module):
    """Grouped widely-linear Cartesian FFN with an identity residual."""

    def __init__(self, groups: int, modes: int, hidden_modes: int) -> None:
        super().__init__()
        self.groups = groups
        self.modes = modes
        self.hidden_modes = hidden_modes
        self.input_projection = GroupedWidelyLinear(groups, modes, hidden_modes, bias=True)
        self.output_projection = GroupedWidelyLinear(groups, hidden_modes, modes, bias=True)

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        if (
            real.shape != imag.shape
            or real.ndim < 2
            or tuple(real.shape[-2:]) != (self.modes, self.groups)
        ):
            message = "grouped widely-linear residual inputs must end in mode-group axes"
            raise ValueError(message)
        leading = real.shape[:-2]
        packed_shape = (-1, 1, 1, self.modes, self.groups)
        hidden_real, hidden_imag = self.input_projection(
            real.reshape(packed_shape),
            imag.reshape(packed_shape),
        )
        update_real, update_imag = self.output_projection(
            functional.silu(hidden_real),
            functional.silu(hidden_imag),
        )
        output_shape = (*leading, self.modes, self.groups)
        return real + update_real.reshape(output_shape), imag + update_imag.reshape(output_shape)


__all__ = [
    "GroupedComplexLinear",
    "GroupedPhaseGatedComplexFFN",
    "GroupedWidelyLinearResidual",
]
