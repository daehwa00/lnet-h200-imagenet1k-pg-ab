"""Vector-input 2D pole memory with one learned drive per pole."""

from __future__ import annotations

# pyright: reportPrivateUsage=false
# ruff: noqa: SLF001
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor, nn

from .pac_complex_layers import (
    ComplexLinear,
    PackedComplexLinear,
    semi_orthogonal_complex_linear_,
)
from .pac_factorized_wide_pole_memory import FactorizedWidePoleMemoryStage
from .pac_phase_gated_cffn import PhaseGatedComplexFFN
from .pac_product_scan_pipeline import ScanMemoryPolicy, run_product_scan_pipeline

if TYPE_CHECKING:
    from .complex_scan_types import ComplexField

_DIRECTION_COUNT = 4
_DESCRIPTOR_MODES = 96
_POLE_PG_HIDDEN = 16
_PATH_PG_HIDDEN = 4


class VectorInputPoleMemoryStage(FactorizedWidePoleMemoryStage):
    """Project the complete content vector into pole drives before the 2D scan."""

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
        scan_memory_policy: ScanMemoryPolicy = "retain",
        damping_min: float = 0.01,
        damping_max: float = 0.7,
    ) -> None:
        if terminal != (next_modes is None):
            message = "only the terminal vector-input pole stage may omit next_modes"
            raise ValueError(message)
        if next_modes is not None and next_modes % _DIRECTION_COUNT:
            message = "next excitation width must split evenly across four directions"
            raise ValueError(message)
        nn.Module.__init__(self)
        self._initialize_pole_parameters(
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

        self.pole_input = PackedComplexLinear(content_modes, poles)
        semi_orthogonal_complex_linear_(self.pole_input)
        self.pole_pg = PhaseGatedComplexFFN(poles, _POLE_PG_HIDDEN)
        self.path_pg = PhaseGatedComplexFFN(_DIRECTION_COUNT, _PATH_PG_HIDDEN)

        self.descriptor_projection = PackedComplexLinear(poles, _DESCRIPTOR_MODES)
        semi_orthogonal_complex_linear_(self.descriptor_projection)

        if terminal:
            self.direction_projection = None
        else:
            if next_modes is None:
                message = "non-terminal vector-input pole stage requires next_modes"
                raise RuntimeError(message)
            self.direction_projection = PackedComplexLinear(
                poles,
                next_modes // _DIRECTION_COUNT,
            )
            semi_orthogonal_complex_linear_(self.direction_projection)
        self._register_diagnostic_buffers()

    def _scan(self, real: Tensor, imag: Tensor) -> ComplexField:
        shape = cast("tuple[int, int, int, int]", tuple(real.shape))
        pole_x, pole_y = self._compact_pole_coefficients(shape)
        excitation = self.pole_input(real, imag)
        if self.terminal:
            return self._terminal_full_states(pole_x, pole_y, excitation)
        coarse_real, coarse_imag, _ = cast(
            "tuple[Tensor, Tensor, Tensor]",
            run_product_scan_pipeline(
                pole_x,
                pole_y,
                excitation,
                epilogue="coarse",
                gain_normalization="pointwise",
                memory_policy=self.scan_memory_policy,
            ),
        )
        return coarse_real, coarse_imag

    def _process_memory(
        self,
        real: Tensor,
        imag: Tensor,
        *,
        update_diagnostics: bool = True,
    ) -> ComplexField:
        pole_real, pole_imag = self.pole_pg._forward(
            real,
            imag,
            update_diagnostics=update_diagnostics,
        )
        path_real, path_imag = self.path_pg._forward(
            pole_real.movedim(-2, -1).contiguous(),
            pole_imag.movedim(-2, -1).contiguous(),
            update_diagnostics=update_diagnostics,
        )
        return path_real.movedim(-1, -2), path_imag.movedim(-1, -2)

    def _descriptor(self, real: Tensor, imag: Tensor) -> Tensor:
        projected_real, projected_imag = self.descriptor_projection(real, imag)
        energy = projected_real.float().square().add(projected_imag.float().square())
        return torch.log1p(energy.mean((1, 2))).flatten(1)

    def _memory_readout(self, real: Tensor, imag: Tensor) -> ComplexField:
        if self.direction_projection is None:
            message = "terminal vector-input pole stage has no next excitation readout"
            raise RuntimeError(message)
        output_real, output_imag = self.direction_projection(real, imag)
        return output_real.flatten(-2), output_imag.flatten(-2)

    @torch.no_grad()
    def _update_response_diagnostics(self, real: Tensor, imag: Tensor) -> None:
        height = min(2, real.shape[1])
        width = min(2, real.shape[2])
        sampled_real = real[:1, :height, :width].detach().float()
        sampled_imag = imag[:1, :height, :width].detach().float()
        matrix_real = sampled_real.reshape(-1, self.poles)
        matrix_imag = sampled_imag.reshape(-1, self.poles)
        gram_real = matrix_real.mT @ matrix_real + matrix_imag.mT @ matrix_imag
        gram_imag = matrix_real.mT @ matrix_imag - matrix_imag.mT @ matrix_real
        diagonal = gram_real.diagonal().clamp_min(1.0e-12)
        denominator = torch.sqrt(diagonal[:, None] * diagonal[None, :])
        correlation = torch.sqrt(gram_real.square() + gram_imag.square()) / denominator
        active_correlation = correlation[self._off_diagonal]
        frobenius = gram_real.square().add(gram_imag.square()).sum().clamp_min(1.0e-12)
        response_rms = torch.sqrt(diagonal / max(1, matrix_real.shape[0]))
        self._ema(self.response_rms_mean, response_rms.mean())
        self._ema(self.response_rms_std, response_rms.std(unbiased=False))
        self._ema(self.response_correlation_mean, active_correlation.mean())
        self._ema(self.response_correlation_max, active_correlation.max())
        self._ema(self.pole_effective_rank, diagonal.sum().square() / frobenius)
        self._ema(
            self.raw_memory_rms,
            sampled_real.square().add(sampled_imag.square()).mean().sqrt(),
        )

    @staticmethod
    def _linear_metrics(layer: ComplexLinear, prefix: str) -> dict[str, float]:
        weight = torch.complex(
            layer.weight_real.detach().float(),
            layer.weight_imag.detach().float(),
        )
        singular = torch.linalg.svdvals(weight)
        squared = singular.square()
        effective_rank = squared.sum().square() / squared.square().sum().clamp_min(1.0e-12)
        return {
            f"{prefix}_singular_min": float(singular.min()),
            f"{prefix}_singular_mean": float(singular.mean()),
            f"{prefix}_singular_max": float(singular.max()),
            f"{prefix}_effective_rank": float(effective_rank),
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
            "pole_damping_mean": float(torch.cat((damping_x, damping_y)).mean()),
            "pole_damping_std": float(torch.cat((damping_x, damping_y)).std(unbiased=False)),
            "pole_response_rms_mean": float(self.response_rms_mean),
            "pole_response_rms_std": float(self.response_rms_std),
            "pole_response_correlation_mean": float(self.response_correlation_mean),
            "pole_response_correlation_max": float(self.response_correlation_max),
            "pole_effective_rank": float(self.pole_effective_rank),
            "raw_memory_rms": float(self.raw_memory_rms),
            "processed_memory_rms": float(self.processed_memory_rms),
        }
        metrics.update(self._linear_metrics(self.pole_input, "pole_input"))
        metrics.update(self._linear_metrics(self.descriptor_projection, "descriptor"))
        if self.direction_projection is not None:
            metrics.update(self._linear_metrics(self.direction_projection, "direction_readout"))
            metrics["next_excitation_rms"] = float(self.memory_readout_rms)
        return metrics

    def _forward_impl(
        self,
        real: Tensor,
        imag: Tensor,
        *,
        update_diagnostics: bool,
    ) -> tuple[ComplexField | None, Tensor]:
        raw_memory = self._scan(real, imag)
        if update_diagnostics:
            self._update_response_diagnostics(*raw_memory)
        memory = self._process_memory(
            *raw_memory,
            update_diagnostics=update_diagnostics,
        )
        descriptor = self._descriptor(*memory)
        if self.terminal:
            if update_diagnostics:
                self._update_transition_diagnostics(memory, None, None)
                self.diagnostic_updates.add_(1)
            return None, descriptor
        output = self._memory_readout(*memory)
        if update_diagnostics:
            output_rms = (
                output[0].detach().float().square().add(output[1].detach().float().square())
            ).mean().sqrt()
            self._ema(self.memory_readout_rms, output_rms)
            self._update_transition_diagnostics(memory, None, None)
            self.diagnostic_updates.add_(1)
        return output, descriptor

    def forward(self, real: Tensor, imag: Tensor) -> tuple[ComplexField | None, Tensor]:
        if real.shape != imag.shape or real.ndim != 4 or real.shape[-1] != self.content_modes:
            message = "vector-input pole stage inputs must be matching NHW-content tensors"
            raise ValueError(message)
        return self._forward_impl(real, imag, update_diagnostics=self.training)


__all__ = ["VectorInputPoleMemoryStage"]
