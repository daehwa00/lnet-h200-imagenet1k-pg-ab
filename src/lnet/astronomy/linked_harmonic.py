"""Linked impulse-pole coordinates for harmonic light-curve morphology."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def fuse_linked_logits(
    alphabet_logits: Tensor,
    harmonic_logits: Tensor,
    beta: float,
) -> Tensor:
    """Combine frozen ALPHABET and linked-branch logits."""
    if alphabet_logits.shape != harmonic_logits.shape:
        message = "alphabet and harmonic logits must have identical shapes"
        raise ValueError(message)
    if beta < 0.0:
        message = "fusion beta must be nonnegative"
        raise ValueError(message)
    return alphabet_logits + beta * harmonic_logits


class LinkedImpulseHarmonicBranch(nn.Module):
    """A fixed linked-pole bank with object-wise harmonic-family selection."""

    def __init__(
        self,
        input_dim: int,
        base_modes: int,
        output_dim: int,
        *,
        harmonics: int = 4,
        minimum_period_days: float = 0.05,
        maximum_period_days: float = 10.0,
        damping_per_day: float = 1.0e-6,
        phase_coupling: bool = True,
        capacity_matched: bool = False,
    ) -> None:
        super().__init__()
        if input_dim < 1 or base_modes < 2 or output_dim < 2 or harmonics < 2:
            message = "linked harmonic dimensions must be nontrivial"
            raise ValueError(message)
        self.input_dim = input_dim
        self.base_modes = base_modes
        self.output_dim = output_dim
        self.harmonics = harmonics
        self.phase_coupling = phase_coupling
        self.capacity_matched = capacity_matched
        period = torch.logspace(
            math.log10(maximum_period_days),
            math.log10(minimum_period_days),
            base_modes,
        )
        self.base_frequency: Tensor
        self.damping: Tensor
        self.register_buffer("base_frequency", 2.0 * math.pi / period)
        self.register_buffer("damping", torch.tensor(damping_per_day))
        amplitude_dim = harmonics + 2
        phase_dim = 2 * (harmonics - 1)
        self.descriptor_dim = amplitude_dim + (
            phase_dim if phase_coupling or capacity_matched else 0
        )
        self.norm = nn.RMSNorm(self.descriptor_dim)
        self.classifier = nn.Linear(self.descriptor_dim, output_dim)

    def _modal_coordinates(
        self,
        flux: Tensor,
        time_delta: Tensor,
        observation_mask: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
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
        harmonic = torch.arange(
            1,
            self.harmonics + 1,
            dtype=flux.dtype,
            device=flux.device,
        )
        frequency = self.base_frequency.to(dtype=flux.dtype).view(1, 1, -1, 1)
        phase = torch.remainder(
            age.unsqueeze(-1) * frequency * harmonic.view(1, 1, 1, -1),
            2.0 * math.pi,
        )
        envelope = torch.exp(-self.damping.to(dtype=flux.dtype) * age).unsqueeze(-1)
        transition = torch.polar(envelope, -phase)
        return torch.einsum(
            "btc,btmk->bcmk",
            normalized.to(dtype=transition.dtype),
            transition,
        )

    def descriptor(
        self,
        flux: Tensor,
        *,
        time_delta: Tensor,
        observation_mask: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        state = self._modal_coordinates(
            flux,
            time_delta,
            observation_mask,
            valid_mask,
        )
        epsilon = torch.finfo(flux.dtype).eps
        band_amplitude = state.abs()
        amplitude = torch.sqrt(band_amplitude.square().sum(dim=1) + epsilon)
        family_score = torch.log1p(amplitude.square()).sum(dim=-1)
        selected = family_score.argmax(dim=-1)
        gather_amplitude = selected[:, None, None].expand(
            -1,
            1,
            self.harmonics,
        )
        chosen_amplitude = amplitude.gather(1, gather_amplitude).squeeze(1)
        period = 2.0 * math.pi / self.base_frequency
        chosen_period = period[selected].to(dtype=flux.dtype)
        ratios = torch.log(chosen_amplitude[:, 1:].clamp_min(epsilon))
        ratios -= torch.log(chosen_amplitude[:, :1].clamp_min(epsilon))
        peak_fraction = torch.softmax(family_score, dim=-1).amax(dim=-1)
        amplitude_features = torch.cat(
            (
                torch.log(chosen_period).unsqueeze(-1),
                torch.log1p(chosen_amplitude[:, :1].square()),
                ratios,
                peak_fraction.unsqueeze(-1),
            ),
            dim=-1,
        )
        if not self.phase_coupling:
            if self.capacity_matched:
                zeros = amplitude_features.new_zeros(
                    amplitude_features.shape[0],
                    2 * (self.harmonics - 1),
                )
                return torch.cat((amplitude_features, zeros), dim=-1)
            return amplitude_features
        gather_state = selected[:, None, None, None].expand(
            -1,
            self.input_dim,
            1,
            self.harmonics,
        )
        chosen_state = state.gather(2, gather_state).squeeze(2)
        fundamental = chosen_state[..., :1]
        fundamental_unit = fundamental / fundamental.abs().clamp_min(epsilon)
        phase_features: list[Tensor] = []
        for index in range(1, self.harmonics):
            harmonic_state = chosen_state[..., index : index + 1]
            harmonic_unit = harmonic_state / harmonic_state.abs().clamp_min(epsilon)
            cross = harmonic_unit * fundamental_unit.conj().pow(index + 1)
            weight = harmonic_state.abs() * fundamental.abs()
            aggregate = (weight * cross).sum(dim=1) / weight.sum(dim=1).clamp_min(
                epsilon
            )
            phase_features.extend((aggregate.real, aggregate.imag))
        return torch.cat((amplitude_features, *phase_features), dim=-1)

    def forward(
        self,
        flux: Tensor,
        *,
        time_delta: Tensor,
        observation_mask: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        descriptor = self.descriptor(
            flux,
            time_delta=time_delta,
            observation_mask=observation_mask,
            valid_mask=valid_mask,
        )
        return self.classifier(self.norm(descriptor))
