"""Identity-centered magnitude gates for complex feature fields."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def mean_one_magnitude_gate(
    real: Tensor,
    imag: Tensor,
    alpha: Tensor,
    *,
    redistribution: float,
) -> Tensor:
    """Return a positive relative-energy gate with exact row mean one."""
    if real.shape != imag.shape or real.ndim < 1:
        message = "mean-one magnitude gate requires matching complex coordinates"
        raise ValueError(message)
    if alpha.shape != (real.shape[-1],):
        message = "mean-one magnitude gate alpha has incompatible shape"
        raise ValueError(message)
    if not 0.0 < redistribution < 1.0:
        message = "mean-one magnitude redistribution must be strictly between zero and one"
        raise ValueError(message)
    magnitude = torch.log1p(real.float().square() + imag.float().square())
    centered = magnitude - magnitude.mean(dim=-1, keepdim=True)
    relative = 1.0 + redistribution * torch.tanh(alpha.float() * centered)
    return (relative / relative.mean(dim=-1, keepdim=True)).to(dtype=real.dtype)


class MeanOneMagnitudeGate(nn.Module):
    """Learn per-mode gate sensitivity without changing mean branch gain."""

    gate_mean: Tensor
    gate_std: Tensor
    gate_min: Tensor
    gate_max: Tensor
    diagnostic_updates: Tensor

    def __init__(
        self,
        modes: int,
        *,
        alpha_init: float = 0.075,
        redistribution: float = 0.5,
    ) -> None:
        super().__init__()
        if modes <= 0:
            message = "mean-one magnitude gate requires positive modes"
            raise ValueError(message)
        if not 0.0 < redistribution < 1.0:
            message = "mean-one magnitude redistribution must be strictly between zero and one"
            raise ValueError(message)
        self.modes = modes
        self.redistribution = float(redistribution)
        self.alpha = nn.Parameter(torch.full((modes,), alpha_init))
        self.register_buffer("gate_mean", torch.zeros(()), persistent=False)
        self.gate_mean = self.get_buffer("gate_mean")
        self.register_buffer("gate_std", torch.zeros(()), persistent=False)
        self.gate_std = self.get_buffer("gate_std")
        self.register_buffer("gate_min", torch.zeros(()), persistent=False)
        self.gate_min = self.get_buffer("gate_min")
        self.register_buffer("gate_max", torch.zeros(()), persistent=False)
        self.gate_max = self.get_buffer("gate_max")
        self.register_buffer("diagnostic_updates", torch.zeros(()), persistent=False)
        self.diagnostic_updates = self.get_buffer("diagnostic_updates")

    @torch.no_grad()
    def _update_diagnostics(self, gate: Tensor) -> None:
        sampled = gate.detach().reshape(-1, self.modes)[:1024].float()
        values = (
            sampled.mean(),
            sampled.std(unbiased=False),
            sampled.min(),
            sampled.max(),
        )
        count = self.diagnostic_updates
        decay = torch.where(count > 0, count.new_tensor(0.95), count.new_zeros(()))
        for target, value in zip(
            (self.gate_mean, self.gate_std, self.gate_min, self.gate_max),
            values,
            strict=True,
        ):
            target.mul_(decay).add_(value * (1.0 - decay))
        self.diagnostic_updates.add_(1)

    def forward(self, real: Tensor, imag: Tensor) -> Tensor:
        if real.shape != imag.shape or real.shape[-1] != self.modes:
            message = "mean-one magnitude gate inputs have incompatible shapes"
            raise ValueError(message)
        gate = mean_one_magnitude_gate(
            real,
            imag,
            self.alpha,
            redistribution=self.redistribution,
        )
        self._update_diagnostics(gate)
        return gate

    @torch.no_grad()
    def diagnostic_metrics(self) -> dict[str, float]:
        return {
            "gate_mean": float(self.gate_mean),
            "gate_std": float(self.gate_std),
            "gate_min": float(self.gate_min),
            "gate_max": float(self.gate_max),
            "alpha_mean": float(self.alpha.detach().float().mean()),
            "alpha_std": float(self.alpha.detach().float().std(unbiased=False)),
            "redistribution": self.redistribution,
        }

    @torch.no_grad()
    def gradient_metrics(self) -> dict[str, float]:
        if self.alpha.grad is None:
            return {}
        return {"alpha_grad_norm": float(torch.linalg.vector_norm(self.alpha.grad.float()))}


__all__ = ["MeanOneMagnitudeGate", "mean_one_magnitude_gate"]
