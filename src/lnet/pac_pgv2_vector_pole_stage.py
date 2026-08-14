"""PGv2 transition wrapped around a vector-input D4 pole scan."""

from __future__ import annotations

# pyright: reportPrivateUsage=false
# ruff: noqa: SLF001
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor, nn
from torch.nn.utils.parametrizations import orthogonal

from .complex_scan_stage import complex_carry_coordinates
from .complex_scan_transitions import FixedComplexRMSNorm
from .pac_complex_ffn import complex_ffn
from .pac_complex_layers import (
    PackedComplexLinear,
    WidelyLinear,
    semi_orthogonal_complex_linear_,
)
from .pac_factorized_wide_pole_memory import FactorizedWidePoleMemoryStage
from .pac_grouped_path_cffn import grouped_cartesian_cffn
from .pac_phase_gated_transition import PhaseGatedModeResidualPathCollapse
from .pac_product_scan_pipeline import ScanMemoryPolicy, run_product_scan_pipeline

if TYPE_CHECKING:
    from .complex_scan_types import ComplexField

_DIRECTIONS = 4


class PGv2VectorPoleStage(FactorizedWidePoleMemoryStage):
    """Project normalized excitation with a real-shared pole-drive FC."""

    def __init__(
        self,
        content_modes: int,
        poles: int,
        *,
        next_modes: int | None,
        path_hidden: int,
        post_hidden: int | None,
        stage_index: int,
        maximum_phase: float,
        frequency_scale: float,
        damping_scale: float,
        terminal: bool,
        scan_memory_policy: ScanMemoryPolicy = "retain",
        damping_min: float = 0.01,
        damping_max: float = 0.7,
    ) -> None:
        if terminal != (next_modes is None) or terminal != (post_hidden is None):
            message = "only the terminal PGv2 vector-pole stage may omit its output contract"
            raise ValueError(message)
        if min(content_modes, poles, path_hidden) <= 0:
            message = "PGv2 vector-pole widths must be positive"
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
        self.mode_hidden = poles
        self.path_hidden = path_hidden
        self.post_hidden = post_hidden

        self.pole_input = nn.Linear(content_modes, poles, bias=False)
        nn.init.orthogonal_(self.pole_input.weight)
        orthogonal(
            self.pole_input,
            "weight",
            orthogonal_map="matrix_exp",
            use_trivialization=True,
        )

        if terminal:
            self.transition = None
            self.memory_adapter = None
            self.register_parameter("carry_logits", None)
            self.carry_projection = None
            self.post_norm = None
            self.post_input = None
            self.post_output = None
            self.register_parameter("post_scale", None)
        else:
            if next_modes is None or post_hidden is None:
                message = "non-terminal PGv2 vector-pole stage requires transition widths"
                raise RuntimeError(message)
            self.transition = PhaseGatedModeResidualPathCollapse(
                poles,
                mode_hidden=poles,
                path_hidden=path_hidden,
            )
            self.memory_adapter = PackedComplexLinear(poles, next_modes)
            semi_orthogonal_complex_linear_(self.memory_adapter)
            self.carry_logits = nn.Parameter(torch.zeros(content_modes, _DIRECTIONS))
            self.carry_projection = (
                None
                if content_modes == next_modes
                else PackedComplexLinear(content_modes, next_modes)
            )
            if self.carry_projection is not None:
                semi_orthogonal_complex_linear_(self.carry_projection)
            self.post_norm = FixedComplexRMSNorm(next_modes)
            self.post_input = WidelyLinear(next_modes, post_hidden, bias=True)
            self.post_output = WidelyLinear(post_hidden, next_modes, bias=True)
            self.post_scale = nn.Parameter(torch.full((next_modes,), 0.1))

        self._register_diagnostic_buffers()

    def _scan_memory_and_descriptor(
        self,
        real: Tensor,
        imag: Tensor,
    ) -> tuple[ComplexField, Tensor]:
        shape = cast("tuple[int, int, int, int]", tuple(real.shape))
        pole_x, pole_y = self._compact_pole_coefficients(shape)
        excitation = self.pole_input(real), self.pole_input(imag)
        if self.terminal:
            full_states = self._terminal_full_states(pole_x, pole_y, excitation)
            return full_states, self._descriptor(*full_states)
        coarse_real, coarse_imag, descriptor = cast(
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
        return (coarse_real, coarse_imag), descriptor

    def _descriptor(self, real: Tensor, imag: Tensor) -> Tensor:
        energy = real.float().square().add(imag.float().square())
        return torch.log1p(energy.mean((1, 2))).flatten(1)

    def _collapse_paths(
        self,
        real: Tensor,
        imag: Tensor,
        *,
        update_diagnostics: bool,
    ) -> ComplexField:
        if self.transition is None:
            message = "terminal PGv2 vector-pole stage has no transition"
            raise RuntimeError(message)
        mixed_real, mixed_imag = self.transition.mode._forward(
            real,
            imag,
            update_diagnostics=update_diagnostics,
        )
        collapsed_real, collapsed_imag = grouped_cartesian_cffn(
            mixed_real,
            mixed_imag,
            input_projection=self.transition.path_input,
            output_projection=self.transition.path_output,
        )
        return collapsed_real.squeeze(-2), collapsed_imag.squeeze(-2)

    def _carry(self, real: Tensor, imag: Tensor) -> ComplexField:
        if self.carry_logits is None:
            message = "terminal PGv2 vector-pole stage has no S2D carry"
            raise RuntimeError(message)
        carry_real, carry_imag = complex_carry_coordinates(real, imag, "s2d")
        shape = (*carry_real.shape[:-1], _DIRECTIONS, self.content_modes)
        weight = torch.softmax(self.carry_logits.float(), dim=-1).to(dtype=real.dtype).mT
        pooled = (
            (carry_real.reshape(shape) * weight).sum(-2),
            (carry_imag.reshape(shape) * weight).sum(-2),
        )
        if self.carry_projection is None:
            return pooled
        return self.carry_projection(*pooled)

    def _post_fusion(self, real: Tensor, imag: Tensor) -> ComplexField:
        if (
            self.post_norm is None
            or self.post_input is None
            or self.post_output is None
            or self.post_scale is None
        ):
            message = "terminal PGv2 vector-pole stage has no PostFusion"
            raise RuntimeError(message)
        normalized = self.post_norm(real, imag)
        return complex_ffn(
            *normalized,
            input_projection=self.post_input,
            output_projection=self.post_output,
            activation="cartesian_silu",
            residual_scale=self.post_scale,
            residual_source=(real, imag),
        )

    @staticmethod
    def _linear_metrics(layer: PackedComplexLinear, prefix: str) -> dict[str, float]:
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

    @staticmethod
    def _real_linear_metrics(layer: nn.Linear, prefix: str) -> dict[str, float]:
        singular = torch.linalg.svdvals(layer.weight.detach().float())
        squared = singular.square()
        effective_rank = squared.sum().square() / squared.square().sum().clamp_min(1.0e-12)
        return {
            f"{prefix}_singular_min": float(singular.min()),
            f"{prefix}_singular_mean": float(singular.mean()),
            f"{prefix}_singular_max": float(singular.max()),
            f"{prefix}_effective_rank": float(effective_rank),
        }

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
            "memory_adapter_rms": float(self.memory_readout_rms),
            "s2d_carry_rms": float(self.carry_rms),
            "memory_carry_rms_ratio": float(self.memory_carry_rms_ratio),
        }
        metrics.update(self._real_linear_metrics(self.pole_input, "pole_input"))
        if self.carry_logits is not None:
            carry = torch.softmax(self.carry_logits.detach().float(), dim=-1)
            metrics.update(
                {
                    "carry_coefficient_min": float(carry.min()),
                    "carry_coefficient_max": float(carry.max()),
                    "carry_sum_error": float((carry.sum(-1) - 1.0).abs().max()),
                }
            )
        if self.memory_adapter is not None:
            metrics.update(self._linear_metrics(self.memory_adapter, "memory_adapter"))
        if self.carry_projection is not None:
            metrics.update(self._linear_metrics(self.carry_projection, "carry_projection"))
        return metrics

    def _forward_impl(
        self,
        real: Tensor,
        imag: Tensor,
        *,
        update_diagnostics: bool,
    ) -> tuple[ComplexField | None, Tensor]:
        raw_memory, descriptor = self._scan_memory_and_descriptor(real, imag)
        if update_diagnostics:
            self._update_response_diagnostics(*raw_memory)
        if self.terminal:
            if update_diagnostics:
                self._update_transition_diagnostics(raw_memory, None, None)
                self.diagnostic_updates.add_(1)
            return None, descriptor

        if self.memory_adapter is None:
            message = "non-terminal PGv2 vector-pole stage is missing its memory adapter"
            raise RuntimeError(message)
        collapsed = self._collapse_paths(
            *raw_memory,
            update_diagnostics=update_diagnostics,
        )
        memory = self.memory_adapter(*collapsed)
        carry = self._carry(real, imag)
        merged = memory[0] + carry[0], memory[1] + carry[1]
        output = self._post_fusion(*merged)
        if update_diagnostics:
            self._update_transition_diagnostics(raw_memory, memory, carry)
            self.diagnostic_updates.add_(1)
        return output, descriptor

    def forward(self, real: Tensor, imag: Tensor) -> tuple[ComplexField | None, Tensor]:
        if real.shape != imag.shape or real.ndim != 4 or real.shape[-1] != self.content_modes:
            message = "PGv2 vector-pole stage inputs must be matching NHW-content tensors"
            raise ValueError(message)
        return self._forward_impl(real, imag, update_diagnostics=self.training)


__all__ = ["PGv2VectorPoleStage"]
