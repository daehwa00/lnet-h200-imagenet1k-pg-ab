from __future__ import annotations

import torch
from torch import Tensor, nn

from .pac_model import PACHybridPRLBlock
from .pac_recurrence import recurrence_states


class TimedPACHybridPRL(nn.Module):
    def __init__(self, model: PACHybridPRLBlock, delta: Tensor) -> None:
        super().__init__()
        self.model = model
        self.time_delta: Tensor
        self.register_buffer("time_delta", delta)

    def forward(self, inputs: Tensor) -> Tensor:
        projected = self.model.input_projection(inputs)
        fused = self._fused(projected)
        residual = projected + self.model.residual_scale * self.model.activation(
            self.model.output_projection(fused)
        )
        return self.model.readout_projection(residual)

    def _fused(self, projected: Tensor) -> Tensor:
        scales = self.model.branch_scales.to(device=projected.device, dtype=projected.dtype)
        fused = torch.zeros_like(projected)
        if "prl" in self.model.active_branches:
            fused = fused + _timed_prl_output(self.model, projected, self.time_delta) * scales[0]
        if "fir" in self.model.active_branches:
            fused = (
                fused
                + self.model.branch_outputs(projected, ("prl",)).get(
                    "fir", torch.zeros_like(projected)
                )
                * scales[1]
            )
        if "mlp" in self.model.active_branches:
            fused = (
                fused
                + self.model.branch_outputs(projected, ("prl", "fir")).get(
                    "mlp", torch.zeros_like(projected)
                )
                * scales[2]
            )
        return fused


def timed_pac_model(model: nn.Module, delta: Tensor) -> nn.Module:
    if isinstance(model, PACHybridPRLBlock):
        return TimedPACHybridPRL(model, delta)
    return model


def _timed_prl_output(model: PACHybridPRLBlock, projected: Tensor, delta: Tensor) -> Tensor:
    branch = model.require_prl_branch()
    inputs_work = projected.to(dtype=branch.reader.dtype)
    step = delta.to(device=inputs_work.device, dtype=inputs_work.dtype).unsqueeze(-1)
    damping = branch.effective_damping_values(inputs_work)
    frequency = branch.frequency_values().to(device=inputs_work.device, dtype=damping.dtype)
    poles = torch.complex(-damping, frequency.view(1, 1, branch.modes).expand_as(damping))
    decay = torch.exp(poles * step)
    gamma = _stable_expm1_over_p(poles, step)
    instant_drive = torch.einsum("bnd,md->bnm", inputs_work, branch.reader)
    tapped_drive = branch.tapped_drive_sequence(instant_drive).to(dtype=gamma.dtype)
    states = recurrence_states(decay, gamma * tapped_drive, branch.recurrence_backend)
    writer = torch.complex(branch.writer_real, branch.writer_imag)
    modal = 2.0 * torch.einsum("bnm,md->bnd", states, writer).real
    direct = torch.matmul(inputs_work, branch.direct_term.transpose(0, 1)) + branch.bias
    return (modal + direct).to(dtype=projected.dtype)


def _stable_expm1_over_p(poles: Tensor, delta: Tensor, threshold: float = 1.0e-6) -> Tensor:
    scaled = poles * delta
    small = torch.abs(scaled) < threshold
    safe_poles = torch.where(small, torch.ones_like(poles), poles)
    raw = torch.expm1(scaled) / safe_poles
    approx = delta.to(dtype=raw.real.dtype).expand_as(raw.real)
    return torch.where(small, torch.complex(approx, torch.zeros_like(approx)), raw)
