"""Physics-clean damped Fourier classifier for irregular point samples."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class DampedSpectrumClassifier(nn.Module):
    """Classify terminal power from a learnable damped Fourier bank.

    Each mode evaluates the point-sample measure directly:

        z_m(T) = sum_k exp((-alpha_m + i omega_m) (T - t_k)) e_{k,m}.

    There is no hold interpolation, temporal convolution, reader recurrence, or
    lag-moment quadrature in this control.
    """

    def __init__(
        self,
        input_dim: int,
        modes: int,
        output_dim: int,
        *,
        minimum_period_days: float = 0.05,
        maximum_period_days: float = 10.0,
        damping_min_per_day: float = 1.0 / 3000.0,
        damping_max_per_day: float = 2.0,
        near_undamped_modes: int = 0,
        near_undamped_alpha_per_day: float = 1.0e-6,
        freeze_frequencies: bool = False,
    ) -> None:
        super().__init__()
        if input_dim < 1 or modes < 2 or output_dim < 2:
            message = "input_dim, modes, and output_dim must define a nontrivial classifier"
            raise ValueError(message)
        if not 0 <= near_undamped_modes <= modes:
            message = "near_undamped_modes must lie between zero and the mode count"
            raise ValueError(message)
        self.input_dim = input_dim
        self.modes = modes
        self.output_dim = output_dim
        self.damping_min = damping_min_per_day
        self.damping_max = damping_max_per_day
        self.frequency_min = 2.0 * math.pi / maximum_period_days
        self.frequency_max = 2.0 * math.pi / minimum_period_days
        self.near_undamped_modes = near_undamped_modes
        self.near_undamped_alpha = near_undamped_alpha_per_day

        target_damping = torch.logspace(
            math.log10(damping_min_per_day * 1.01),
            math.log10(damping_max_per_day * 0.99),
            modes,
        )
        damping_unit = (target_damping - self.damping_min) / (
            self.damping_max - self.damping_min
        )
        self.raw_decay = nn.Parameter(
            torch.logit(damping_unit.clamp(1.0e-6, 1.0 - 1.0e-6))
        )
        target_frequency = torch.logspace(
            math.log10(self.frequency_min * 1.001),
            math.log10(self.frequency_max * 0.999),
            modes,
        )
        frequency_unit = (target_frequency - self.frequency_min) / (
            self.frequency_max - self.frequency_min
        )
        self.raw_frequency = nn.Parameter(
            torch.logit(frequency_unit.clamp(1.0e-6, 1.0 - 1.0e-6)),
            requires_grad=not freeze_frequencies,
        )
        self.spectrum_norm = nn.RMSNorm(modes)
        self.classifier = nn.Linear(modes, output_dim)

    def damping_values(self) -> Tensor:
        damping = self.damping_min + (self.damping_max - self.damping_min) * torch.sigmoid(
            self.raw_decay
        )
        if not self.near_undamped_modes:
            return damping
        fixed = torch.full_like(damping[: self.near_undamped_modes], self.near_undamped_alpha)
        return torch.cat((fixed, damping[self.near_undamped_modes :]))

    def frequency_values(self) -> Tensor:
        return self.frequency_min + (self.frequency_max - self.frequency_min) * torch.sigmoid(
            self.raw_frequency
        )

    def modal_coordinates(
        self,
        flux: Tensor,
        *,
        time_delta: Tensor,
        observation_mask: Tensor,
        valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return terminal real, imaginary, and power coordinates."""
        if flux.ndim != 3 or flux.shape[-1] != self.input_dim:
            message = "flux must have shape [B,T,input_dim]"
            raise ValueError(message)
        if time_delta.shape != (*flux.shape[:2], 1):
            message = "time_delta must have shape [B,T,1]"
            raise ValueError(message)
        if observation_mask.shape != flux.shape:
            message = "observation_mask must match flux"
            raise ValueError(message)
        if valid_mask.shape != (*flux.shape[:2], 1):
            message = "valid_mask must have shape [B,T,1]"
            raise ValueError(message)

        observed = observation_mask.to(dtype=flux.dtype) * valid_mask
        count = observed.sum(dim=1).clamp_min(1.0)
        mean = (flux * observed).sum(dim=1) / count
        centered = (flux - mean.unsqueeze(1)) * observed
        scale = torch.sqrt(centered.square().sum(dim=1) / count).clamp_min(
            torch.finfo(flux.dtype).eps
        )
        normalized = centered / scale.unsqueeze(1)

        relative_delta = torch.cat(
            (torch.zeros_like(time_delta[:, :1]), time_delta[:, 1:]),
            dim=1,
        )
        timestamps = relative_delta.cumsum(dim=1)
        final_time = (timestamps * valid_mask).amax(dim=1, keepdim=True)
        age = (final_time - timestamps).clamp_min(0.0)
        damping = self.damping_values().view(1, 1, -1)
        frequency = self.frequency_values().view(1, 1, -1)
        envelope = torch.exp(-damping * age)
        phase = frequency * age
        transition_real = envelope * torch.cos(phase)
        transition_imag = -envelope * torch.sin(phase)
        state_real = torch.einsum("btc,btm->bcm", normalized, transition_real)
        state_imag = torch.einsum("btc,btm->bcm", normalized, transition_imag)
        power = (state_real.square() + state_imag.square()).sum(dim=1)
        return state_real, state_imag, power

    def forward(
        self,
        flux: Tensor,
        *,
        time_delta: Tensor,
        observation_mask: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        *_, power = self.modal_coordinates(
            flux,
            time_delta=time_delta,
            observation_mask=observation_mask,
            valid_mask=valid_mask,
        )
        return self.classifier(self.spectrum_norm(torch.log1p(power)))
