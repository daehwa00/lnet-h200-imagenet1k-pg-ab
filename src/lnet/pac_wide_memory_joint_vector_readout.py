"""Wide pole memory with a rank-two separable complex readout between stages."""

from __future__ import annotations

# pyright: reportPrivateUsage=false
# ruff: noqa: SLF001
import math

import torch
from torch import Tensor, nn

from .complex_scan_stage import complex_carry_coordinates
from .pac_complex_layers import (
    ComplexLinear,
    PackedComplexLinear,
    semi_orthogonal_complex_linear_,
)
from .pac_factorized_wide_pole_memory import FactorizedWidePoleMemoryStage
from .pac_phase_gated_cffn import PhaseGatedComplexFFN

ComplexField = tuple[Tensor, Tensor]

_DIRECTION_COUNT = 4
_DESCRIPTOR_MODES = 96
_DESCRIPTOR_POLE_MODES = 4
_READOUT_RANK = 2


class RankTwoSeparableReadout(nn.Module):
    """Compress direction-content-pole memory with a rank-two strict map.

    Two shared content maps are followed by output-dependent complex pole
    pooling.  The resulting effective content-by-pole kernel has rank at most
    two for every directional output.
    """

    def __init__(
        self,
        input_modes: int,
        poles: int,
        output_modes: int,
        *,
        rank: int = _READOUT_RANK,
    ) -> None:
        super().__init__()
        if output_modes % _DIRECTION_COUNT:
            message = "separable readout output must split evenly across four directions"
            raise ValueError(message)
        if rank != _READOUT_RANK:
            message = "this experiment fixes the separable readout rank at two"
            raise ValueError(message)
        self.input_modes = input_modes
        self.poles = poles
        self.output_modes = output_modes
        self.rank = rank
        self.direction_modes = output_modes // _DIRECTION_COUNT
        self.content_projection = PackedComplexLinear(
            input_modes,
            rank * self.direction_modes,
        )
        semi_orthogonal_complex_linear_(self.content_projection)
        weight_shape = (_DIRECTION_COUNT, self.direction_modes, rank, poles)
        self.pole_weight_real = nn.Parameter(torch.empty(weight_shape))
        self.pole_weight_imag = nn.Parameter(torch.zeros(weight_shape))
        self._initialize_pole_basis()

    @torch.no_grad()
    def _initialize_pole_basis(self) -> None:
        coordinate = torch.arange(self.poles, dtype=self.pole_weight_real.dtype)
        angle = 2.0 * torch.pi * (coordinate + 0.5) / self.poles
        basis = torch.stack(
            (
                torch.full_like(coordinate, 1.0 / math.sqrt(self.poles)),
                math.sqrt(2.0 / self.poles) * torch.cos(angle),
            )
        ).div(math.sqrt(self.rank))
        self.pole_weight_real.copy_(
            basis.view(1, 1, self.rank, self.poles).expand_as(self.pole_weight_real)
        )

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        expected_tail = (_DIRECTION_COUNT, self.input_modes, self.poles)
        if real.shape != imag.shape or tuple(real.shape[-3:]) != expected_tail:
            message = "rank-two separable readout inputs have incompatible semantic axes"
            raise ValueError(message)

        projected_real, projected_imag = self.content_projection(
            real.transpose(-1, -2),
            imag.transpose(-1, -2),
        )
        projected_shape = (
            *projected_real.shape[:-1],
            self.rank,
            self.direction_modes,
        )
        projected_real = projected_real.reshape(projected_shape)
        projected_imag = projected_imag.reshape(projected_shape)
        weight_real = self.pole_weight_real.to(dtype=projected_real.dtype)
        weight_imag = self.pole_weight_imag.to(dtype=projected_real.dtype)
        output_real = torch.einsum(
            "...drso,dosr->...do",
            projected_real,
            weight_real,
        ) - torch.einsum("...drso,dosr->...do", projected_imag, weight_imag)
        output_imag = torch.einsum(
            "...drso,dosr->...do",
            projected_real,
            weight_imag,
        ) + torch.einsum("...drso,dosr->...do", projected_imag, weight_real)
        return output_real.flatten(-2), output_imag.flatten(-2)


class WideMemoryJointVectorReadoutStage(FactorizedWidePoleMemoryStage):
    """Process pole/path axes, then apply a rank-two separable re-encoding."""

    excitation_rms: Tensor
    excitation_effective_rank: Tensor

    def __init__(
        self,
        content_modes: int,
        poles: int,
        *,
        next_modes: int | None,
        stage_index: int,
        maximum_phase: float,
        frequency_scale: float,
        damping_scale: float,
        terminal: bool,
        scan_memory_policy: str = "recompute",
        damping_min: float = 0.01,
        damping_max: float = 0.7,
    ) -> None:
        if terminal != (next_modes is None):
            message = "only the terminal joint-readout stage may omit next_modes"
            raise ValueError(message)
        super().__init__(
            content_modes,
            poles,
            stage_index=stage_index,
            maximum_phase=maximum_phase,
            frequency_scale=frequency_scale,
            damping_scale=damping_scale,
            terminal=terminal,
            scan_memory_policy=scan_memory_policy,
            damping_min=damping_min,
            damping_max=damping_max,
        )
        self.next_modes = next_modes
        self.output_modes = next_modes

        # Separable readout already mixes content, so the wide Content PG is absent.
        self.content_pg = None
        # The requested 4->8->4 PG has four value and four gate coordinates.
        self.path_pg = PhaseGatedComplexFFN(_DIRECTION_COUNT, 4)

        self.descriptor_readout_real = None
        self.descriptor_readout_imag = None
        self.descriptor_pole_readout = PackedComplexLinear(
            poles,
            _DESCRIPTOR_POLE_MODES,
        )
        self.descriptor_readout = PackedComplexLinear(
            content_modes * _DESCRIPTOR_POLE_MODES,
            _DESCRIPTOR_MODES,
        )
        semi_orthogonal_complex_linear_(self.descriptor_pole_readout)
        semi_orthogonal_complex_linear_(self.descriptor_readout)

        self.direction_readout = None
        self.pole_readout_real = None
        self.pole_readout_imag = None
        if terminal:
            self.joint_readout = None
            self.carry_projection = None
            self.post_pg = None
        else:
            if next_modes is None:
                message = "non-terminal joint-readout stage requires next_modes"
                raise RuntimeError(message)
            self.joint_readout = RankTwoSeparableReadout(
                content_modes,
                poles,
                next_modes,
            )
            self.carry_projection = PackedComplexLinear(content_modes, next_modes)
            semi_orthogonal_complex_linear_(self.carry_projection)
            self.carry_weight = nn.Parameter(torch.full((content_modes, 4), 0.25))
            self.post_pg = PhaseGatedComplexFFN(next_modes, next_modes)

        self.register_buffer("excitation_rms", torch.zeros(()), persistent=False)
        self.register_buffer("excitation_effective_rank", torch.zeros(()), persistent=False)

    def _process_memory(
        self,
        real: Tensor,
        imag: Tensor,
        *,
        update_diagnostics: bool = True,
    ) -> ComplexField:
        # Raw scan layout is D,R,K. Pole PG changes it to D,K,R; Path PG then
        # exposes D as its final mode axis and returns canonical D,K,R memory.
        pole_real, pole_imag = self.pole_pg._forward(
            real.transpose(-1, -2).contiguous(),
            imag.transpose(-1, -2).contiguous(),
            update_diagnostics=update_diagnostics,
        )
        rank = real.ndim
        leading = tuple(range(rank - 3))
        direction_axis, content_axis, pole_axis = range(rank - 3, rank)
        path_real, path_imag = self.path_pg._forward(
            pole_real.permute(*leading, content_axis, pole_axis, direction_axis).contiguous(),
            pole_imag.permute(*leading, content_axis, pole_axis, direction_axis).contiguous(),
            update_diagnostics=update_diagnostics,
        )
        return (
            path_real.permute(*leading, pole_axis, direction_axis, content_axis).contiguous(),
            path_imag.permute(*leading, pole_axis, direction_axis, content_axis).contiguous(),
        )

    def _descriptor(self, real: Tensor, imag: Tensor) -> Tensor:
        pole_real, pole_imag = self.descriptor_pole_readout(real, imag)
        descriptor_real, descriptor_imag = self.descriptor_readout(
            pole_real.flatten(-2),
            pole_imag.flatten(-2),
        )
        energy = descriptor_real.float().square().add(descriptor_imag.float().square())
        return torch.log1p(energy.mean((1, 2))).flatten(1)

    def _memory_readout(self, real: Tensor, imag: Tensor) -> ComplexField:
        if self.joint_readout is None:
            message = "terminal joint-readout stage has no next excitation"
            raise RuntimeError(message)
        return self.joint_readout(real, imag)

    def _carry(self, real: Tensor, imag: Tensor) -> ComplexField:
        if self.carry_weight is None or self.carry_projection is None:
            message = "terminal joint-readout stage has no projected S2D carry"
            raise RuntimeError(message)
        carry_real, carry_imag = complex_carry_coordinates(real, imag, "s2d")
        shape = (*carry_real.shape[:-1], 4, self.content_modes)
        weight = self.carry_weight.transpose(0, 1).to(dtype=real.dtype)
        pooled_real = (carry_real.reshape(shape) * weight).sum(-2)
        pooled_imag = (carry_imag.reshape(shape) * weight).sum(-2)
        return self.carry_projection(pooled_real, pooled_imag)

    @torch.no_grad()
    def _update_excitation_diagnostics(self, real: Tensor, imag: Tensor) -> None:
        height = min(4, real.shape[1])
        width = min(4, real.shape[2])
        sample_real = real[:1, :height, :width].detach().float().reshape(-1, self.content_modes)
        sample_imag = imag[:1, :height, :width].detach().float().reshape(-1, self.content_modes)
        rms = sample_real.square().add(sample_imag.square()).mean().sqrt()
        gram_real = sample_real.mT @ sample_real + sample_imag.mT @ sample_imag
        gram_imag = sample_real.mT @ sample_imag - sample_imag.mT @ sample_real
        trace = gram_real.diagonal().sum()
        frobenius = gram_real.square().add(gram_imag.square()).sum().clamp_min(1.0e-12)
        self._ema(self.excitation_rms, rms)
        self._ema(self.excitation_effective_rank, trace.square() / frobenius)

    @staticmethod
    @torch.no_grad()
    def _singular_metrics(layer: ComplexLinear, prefix: str) -> dict[str, float]:
        weight = torch.complex(
            layer.weight_real.detach().float(),
            layer.weight_imag.detach().float(),
        )
        gram = weight @ weight.mH
        squared = torch.linalg.eigvalsh(gram).clamp_min(0.0)
        singular = torch.sqrt(squared)
        effective = squared.sum().square() / squared.square().sum().clamp_min(1.0e-12)
        return {
            f"{prefix}_singular_min": float(singular.min()),
            f"{prefix}_singular_mean": float(singular.mean()),
            f"{prefix}_singular_max": float(singular.max()),
            f"{prefix}_effective_rank": float(effective),
        }

    def diagnostic_metrics(self) -> dict[str, float]:
        damping_x = self.damping_min + (self.damping_max - self.damping_min) * torch.sigmoid(
            self.damping_logits_x.detach().float()
        )
        damping_y = self.damping_min + (self.damping_max - self.damping_min) * torch.sigmoid(
            self.damping_logits_y.detach().float()
        )
        metrics = {
            "content_modes": float(self.content_modes),
            "poles": float(self.poles),
            "excitation_rms": float(self.excitation_rms),
            "excitation_effective_rank": float(self.excitation_effective_rank),
            "pole_damping_mean": float(torch.cat((damping_x, damping_y)).mean()),
            "pole_damping_std": float(torch.cat((damping_x, damping_y)).std(unbiased=False)),
            "pole_phase_mean": float(
                torch.cat((self.phase_x.detach().float(), self.phase_y.detach().float())).mean()
            ),
            "pole_phase_std": float(
                torch.cat((self.phase_x.detach().float(), self.phase_y.detach().float())).std(
                    unbiased=False
                )
            ),
            "pole_response_rms_mean": float(self.response_rms_mean),
            "pole_response_rms_std": float(self.response_rms_std),
            "pole_response_correlation_mean": float(self.response_correlation_mean),
            "pole_response_correlation_max": float(self.response_correlation_max),
            "pole_effective_rank": float(self.pole_effective_rank),
            "raw_memory_rms": float(self.raw_memory_rms),
            "processed_memory_rms": float(self.processed_memory_rms),
        }
        metrics.update(
            self._singular_metrics(self.descriptor_pole_readout, "descriptor_pole_readout")
        )
        metrics.update(self._singular_metrics(self.descriptor_readout, "descriptor_readout"))
        if self.joint_readout is not None:
            metrics.update(
                {
                    "joint_readout_rms": float(self.memory_readout_rms),
                    "carry_rms": float(self.carry_rms),
                    "joint_readout_carry_rms_ratio": float(self.memory_carry_rms_ratio),
                }
            )
            metrics.update(
                self._singular_metrics(
                    self.joint_readout.content_projection,
                    "readout_content_projection",
                )
            )
            pole_energy = self.joint_readout.pole_weight_real.detach().float().square().add(
                self.joint_readout.pole_weight_imag.detach().float().square()
            )
            pole_probability = pole_energy / pole_energy.sum(-1, keepdim=True).clamp_min(1.0e-12)
            metrics.update(
                {
                    "readout_pole_effective_count": float(
                        pole_probability.square().sum(-1).reciprocal().mean()
                    ),
                    "readout_pole_concentration": float(
                        pole_probability.max(-1).values.mean()
                    ),
                }
            )
        return metrics

    def _forward_impl(
        self,
        real: Tensor,
        imag: Tensor,
        *,
        update_diagnostics: bool,
    ) -> tuple[ComplexField | None, Tensor]:
        if update_diagnostics:
            self._update_excitation_diagnostics(real, imag)
        return super()._forward_impl(real, imag, update_diagnostics=update_diagnostics)

    def forward(self, real: Tensor, imag: Tensor) -> tuple[ComplexField | None, Tensor]:
        if real.shape != imag.shape or real.ndim != 4 or real.shape[-1] != self.content_modes:
            message = "joint-readout stage inputs must be matching NHW-content tensors"
            raise ValueError(message)
        # This architecture peaks well below the 24 GiB device budget. Retain
        # stage activations and avoid replaying scan, PG, joint readout, and Q.
        return self._forward_impl(real, imag, update_diagnostics=self.training)


__all__ = [
    "RankTwoSeparableReadout",
    "WideMemoryJointVectorReadoutStage",
]
