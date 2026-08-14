"""Normalized vector-input pole memory with local residual re-excitation."""

from __future__ import annotations

# pyright: reportPrivateUsage=false
# ruff: noqa: SLF001
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor, nn

from .complex_scan_stage import complex_carry_coordinates
from .complex_scan_transitions import ComplexRMSNorm
from .pac_complex_layers import PackedComplexLinear, semi_orthogonal_complex_linear_
from .pac_factorized_wide_pole_memory import FactorizedWidePoleMemoryStage
from .pac_phase_gated_cffn import PhaseGatedComplexFFN
from .pac_product_scan_pipeline import ScanMemoryPolicy, run_product_scan_pipeline
from .pac_vector_input_pole_memory import VectorInputPoleMemoryStage

if TYPE_CHECKING:
    from .complex_scan_types import ComplexField

_DIRECTIONS = 4
_POLE_PG_HIDDEN = 16


class ReexcitationPoleMemoryStage(FactorizedWidePoleMemoryStage):
    """Separate normalized pole memory from local-evidence re-excitation."""

    excitation_rms: Tensor
    normalized_input_rms: Tensor
    merged_excitation_rms: Tensor
    reexcited_rms: Tensor

    def __init__(
        self,
        content_modes: int,
        poles: int,
        *,
        next_modes: int | None,
        reexcitation_hidden: int | None,
        stage_index: int,
        maximum_phase: float,
        frequency_scale: float,
        damping_scale: float,
        terminal: bool,
        scan_memory_policy: ScanMemoryPolicy = "retain",
        damping_min: float = 0.01,
        damping_max: float = 0.7,
    ) -> None:
        if terminal != (next_modes is None) or terminal != (reexcitation_hidden is None):
            message = "only the terminal re-excitation stage may omit its output contract"
            raise ValueError(message)
        if next_modes is not None and next_modes != _DIRECTIONS * poles:
            message = "re-excitation width must equal the four-direction pole memory width"
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
        self.reexcitation_hidden = reexcitation_hidden

        self.input_norm = ComplexRMSNorm(content_modes)
        self.pole_input = PackedComplexLinear(content_modes, poles)
        semi_orthogonal_complex_linear_(self.pole_input)

        if terminal:
            self.memory_pg = None
            self.carry_projection = None
            self.reexcitation_pg = None
        else:
            if next_modes is None or reexcitation_hidden is None:
                message = "non-terminal re-excitation stages require output and hidden widths"
                raise RuntimeError(message)
            self.memory_pg = PhaseGatedComplexFFN(poles, _POLE_PG_HIDDEN)
            if next_modes == content_modes:
                self.carry_projection = None
            else:
                self.carry_projection = nn.Linear(content_modes, next_modes, bias=False)
                nn.init.orthogonal_(self.carry_projection.weight)
            self.reexcitation_pg = PhaseGatedComplexFFN(next_modes, reexcitation_hidden)

        self._register_diagnostic_buffers()
        self.register_buffer("excitation_rms", torch.zeros(()), persistent=False)
        self.excitation_rms = self.get_buffer("excitation_rms")
        self.register_buffer("normalized_input_rms", torch.zeros(()), persistent=False)
        self.normalized_input_rms = self.get_buffer("normalized_input_rms")
        self.register_buffer("merged_excitation_rms", torch.zeros(()), persistent=False)
        self.merged_excitation_rms = self.get_buffer("merged_excitation_rms")
        self.register_buffer("reexcited_rms", torch.zeros(()), persistent=False)
        self.reexcited_rms = self.get_buffer("reexcited_rms")

    def _scan_normalized(
        self,
        normalized: ComplexField,
        source_shape: tuple[int, int, int, int],
    ) -> ComplexField:
        pole_x, pole_y = self._compact_pole_coefficients(source_shape)
        excitation = self.pole_input(*normalized)
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

    def _scan(self, real: Tensor, imag: Tensor) -> ComplexField:
        shape = cast("tuple[int, int, int, int]", tuple(real.shape))
        return self._scan_normalized(self.input_norm(real, imag), shape)

    def _descriptor(self, real: Tensor, imag: Tensor) -> Tensor:
        energy = real.float().square().add(imag.float().square())
        # Direction-major [B,4P], with one coordinate per raw pole memory.
        return torch.log1p(energy.mean((1, 2))).flatten(1)

    def _carry(self, real: Tensor, imag: Tensor) -> ComplexField:
        carry_real, carry_imag = complex_carry_coordinates(real, imag, "s2d")
        shape = (*carry_real.shape[:-1], _DIRECTIONS, self.content_modes)
        # Preserve the constant local signal while folding the 2x2 neighborhood.
        pooled_real = carry_real.reshape(shape).mean(-2)
        pooled_imag = carry_imag.reshape(shape).mean(-2)
        if self.carry_projection is None:
            return pooled_real, pooled_imag
        packed = torch.stack((pooled_real, pooled_imag), dim=-2)
        projected = self.carry_projection(packed)
        return projected[..., 0, :], projected[..., 1, :]

    @staticmethod
    def _rms(real: Tensor, imag: Tensor) -> Tensor:
        return real.detach().float().square().add(imag.detach().float().square()).mean().sqrt()

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

    @torch.no_grad()
    def _update_excitation_diagnostics(
        self,
        source: ComplexField,
        normalized: ComplexField,
        merged: ComplexField | None,
        output: ComplexField | None,
    ) -> None:
        self._ema(self.excitation_rms, self._rms(*source))
        self._ema(self.normalized_input_rms, self._rms(*normalized))
        if merged is not None:
            self._ema(self.merged_excitation_rms, self._rms(*merged))
        if output is not None:
            self._ema(self.reexcited_rms, self._rms(*output))

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
            "memory_candidate_rms": float(self.memory_readout_rms),
            "carry_rms": float(self.carry_rms),
            "memory_carry_rms_ratio": float(self.memory_carry_rms_ratio),
            "excitation_rms": float(self.excitation_rms),
            "normalized_input_rms": float(self.normalized_input_rms),
            "merged_excitation_rms": float(self.merged_excitation_rms),
            "reexcited_rms": float(self.reexcited_rms),
        }
        metrics.update(VectorInputPoleMemoryStage._linear_metrics(self.pole_input, "pole_input"))
        if self.carry_projection is not None:
            singular = torch.linalg.svdvals(self.carry_projection.weight.detach().float())
            squared = singular.square()
            metrics.update(
                {
                    "carry_projection_singular_min": float(singular.min()),
                    "carry_projection_singular_mean": float(singular.mean()),
                    "carry_projection_singular_max": float(singular.max()),
                    "carry_projection_effective_rank": float(
                        squared.sum().square() / squared.square().sum().clamp_min(1.0e-12)
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
        normalized = self.input_norm(real, imag)
        shape = cast("tuple[int, int, int, int]", tuple(real.shape))
        raw_memory = self._scan_normalized(normalized, shape)
        descriptor = self._descriptor(*raw_memory)
        if update_diagnostics:
            self._update_response_diagnostics(*raw_memory)

        if self.terminal:
            if update_diagnostics:
                self._update_transition_diagnostics(raw_memory, None, None)
                self._update_excitation_diagnostics((real, imag), normalized, None, None)
                self.diagnostic_updates.add_(1)
            return None, descriptor

        if self.memory_pg is None or self.reexcitation_pg is None:
            message = "non-terminal re-excitation stage is missing its PG blocks"
            raise RuntimeError(message)
        processed = self.memory_pg._forward(
            *raw_memory,
            update_diagnostics=update_diagnostics,
        )
        candidate = processed[0].flatten(-2), processed[1].flatten(-2)
        carry = self._carry(real, imag)
        merged = candidate[0] + carry[0], candidate[1] + carry[1]
        output = self.reexcitation_pg._forward(
            *merged,
            update_diagnostics=update_diagnostics,
        )
        if update_diagnostics:
            self._update_transition_diagnostics(processed, candidate, carry)
            self._update_excitation_diagnostics((real, imag), normalized, merged, output)
            self.diagnostic_updates.add_(1)
        return output, descriptor

    def forward(self, real: Tensor, imag: Tensor) -> tuple[ComplexField | None, Tensor]:
        if real.shape != imag.shape or real.ndim != 4 or real.shape[-1] != self.content_modes:
            message = "re-excitation pole stage inputs must be matching NHW-content tensors"
            raise ValueError(message)
        return self._forward_impl(real, imag, update_diagnostics=self.training)


__all__ = ["ReexcitationPoleMemoryStage"]
