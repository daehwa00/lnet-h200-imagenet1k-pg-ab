"""Phase-equivariant complex FFN with automatic packed CUDA execution."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional

from .complex_scan_transitions import ComplexRMSNorm
from .pac_complex_ffn import module_calls_are_transparent
from .pac_complex_layers import (
    ComplexLinear,
    packed_complex_linear_weight,
    unit_row_complex_linear_weight,
)
from .pac_triton_complex_rmsnorm import packed_complex_rms_norm
from .pac_triton_hardware import diagnostic_sample_rows
from .pac_triton_phase_gate_linear import phase_gate_output_linear
from .pac_triton_phase_gate_residual_fused import (
    fused_phase_gate_output_residual,
    supports_fused_phase_gate_output_residual,
)
from .pac_triton_phase_gated_cffn_fused import (
    fused_phase_gated_cffn,
    supports_fused_phase_gated_cffn,
)
from .pac_triton_rmsnorm_linear_fused import (
    fused_rmsnorm_input_linear,
    supports_fused_rmsnorm_input_linear,
)

ComplexField = tuple[Tensor, Tensor]
GATE_REDISTRIBUTION = 0.5
PROJECTION_ROW_EPSILON = 1.0e-12


def _strict_complex_linear(
    real: Tensor,
    imag: Tensor,
    weight_real: Tensor,
    weight_imag: Tensor,
) -> ComplexField:
    return (
        functional.linear(real, weight_real) - functional.linear(imag, weight_imag),
        functional.linear(real, weight_imag) + functional.linear(imag, weight_real),
    )


def _magnitude_gate(
    gate_real: Tensor,
    gate_imag: Tensor,
    alpha: Tensor,
    redistribution: float,
) -> Tensor:
    magnitude = torch.log1p(gate_real.float().square() + gate_imag.float().square())
    centered = magnitude - magnitude.mean(dim=-1, keepdim=True)
    relative = 1.0 + redistribution * torch.tanh(alpha.float() * centered)
    gate = relative / relative.mean(dim=-1, keepdim=True)
    return gate.to(dtype=gate_real.dtype)


def _phase_gate_coordinates(
    projected_real: Tensor,
    projected_imag: Tensor,
    hidden_modes: int,
    *,
    self_gated: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if self_gated:
        return projected_real, projected_imag, projected_real, projected_imag
    value_real, gate_real = projected_real.split(hidden_modes, dim=-1)
    value_imag, gate_imag = projected_imag.split(hidden_modes, dim=-1)
    return value_real, value_imag, gate_real, gate_imag


def phase_gated_complex_ffn_reference(
    module: PhaseGatedComplexFFN,
    real: Tensor,
    imag: Tensor,
    *,
    update_diagnostics: bool = False,
) -> ComplexField:
    """Evaluate the authoritative PyTorch equation without CUDA dispatch."""
    unit_real, unit_imag = module.norm(real, imag)
    input_weight, output_weight = module.effective_projection_weights()
    projected_real, projected_imag = _strict_complex_linear(
        unit_real,
        unit_imag,
        *input_weight,
    )
    value_real, value_imag, gate_real, gate_imag = _phase_gate_coordinates(
        projected_real,
        projected_imag,
        module.hidden_modes,
        self_gated=module.self_gated,
    )
    gate = _magnitude_gate(
        gate_real,
        gate_imag,
        module.alpha,
        module.gate_redistribution,
    )
    delta_real, delta_imag = _strict_complex_linear(
        value_real * gate,
        value_imag * gate,
        *output_weight,
    )
    residual_scale = module.effective_residual_scale()
    output_real = real + residual_scale * delta_real
    output_imag = imag + residual_scale * delta_imag
    if update_diagnostics:
        module._update_diagnostics(  # pyright: ignore[reportPrivateUsage]
            gate,
            real,
            imag,
            delta_real,
            delta_imag,
            output_real,
            output_imag,
        )
    return output_real, output_imag


def _validate_residual_scale_contract(
    initial: float,
    maximum: float | None,
    *,
    learnable: bool,
) -> None:
    if initial <= 0.0:
        message = "phase-gated residual scale must be positive"
        raise ValueError(message)
    if maximum is not None and maximum <= initial:
        message = "phase-gated residual scale maximum must exceed its initial value"
        raise ValueError(message)
    if not learnable and initial != 1.0:
        message = "fixed residual scale must be exactly one"
        raise ValueError(message)
    if not learnable and maximum is not None:
        message = "fixed unit residual scale cannot also be bounded"
        raise ValueError(message)


class PhaseGatedComplexFFN(nn.Module):
    """A globally phase-equivariant residual FFN with a magnitude-only gate."""

    gate_mean: Tensor
    gate_std: Tensor
    gate_min: Tensor
    gate_max: Tensor
    update_ratio: Tensor
    energy_ratio: Tensor
    diagnostic_updates: Tensor
    gamma: nn.Parameter | None
    output_gain_logits: nn.Parameter | None

    def __init__(
        self,
        modes: int,
        hidden_modes: int,
        *,
        alpha_init: float = 0.075,
        gate_redistribution: float = GATE_REDISTRIBUTION,
        residual_scale_init: float = 0.01,
        self_gated: bool = False,
        unit_row_projections: bool = False,
        residual_scale_max: float | None = None,
        learnable_residual_scale: bool = True,
    ) -> None:
        super().__init__()
        if min(modes, hidden_modes) <= 0:
            message = "phase-gated FFN dimensions must be positive"
            raise ValueError(message)
        if not 0.0 < gate_redistribution < 1.0:
            message = "phase-gated redistribution must be strictly between zero and one"
            raise ValueError(message)
        _validate_residual_scale_contract(
            residual_scale_init,
            residual_scale_max,
            learnable=learnable_residual_scale,
        )
        self.modes = modes
        self.hidden_modes = hidden_modes
        self.self_gated = self_gated
        self.gate_redistribution = float(gate_redistribution)
        self.unit_row_projections = bool(unit_row_projections)
        self.projected_direction_rows = False
        self.learnable_residual_scale = bool(learnable_residual_scale)
        self.residual_scale_max = None if residual_scale_max is None else float(residual_scale_max)
        self.output_gain_max: float | None = None
        self.norm = ComplexRMSNorm(modes)
        self.input_projection = ComplexLinear(
            modes,
            hidden_modes if self_gated else 2 * hidden_modes,
        )
        self.output_projection = ComplexLinear(hidden_modes, modes)
        if self.unit_row_projections:
            self._normalize_raw_projection_rows_()
        self.alpha = nn.Parameter(torch.full((hidden_modes,), alpha_init))
        gamma_parameter_init = float(residual_scale_init)
        if self.residual_scale_max is not None:
            gamma_parameter_init = math.atanh(gamma_parameter_init / self.residual_scale_max)
        if self.learnable_residual_scale:
            self.gamma = nn.Parameter(torch.tensor(gamma_parameter_init))
        else:
            self.register_parameter("gamma", None)
        self.output_gain_logits = None
        self.register_buffer("gate_mean", torch.zeros(()), persistent=False)
        self.gate_mean = self.get_buffer("gate_mean")
        self.register_buffer("gate_std", torch.zeros(()), persistent=False)
        self.gate_std = self.get_buffer("gate_std")
        self.register_buffer("gate_min", torch.zeros(()), persistent=False)
        self.gate_min = self.get_buffer("gate_min")
        self.register_buffer("gate_max", torch.zeros(()), persistent=False)
        self.gate_max = self.get_buffer("gate_max")
        self.register_buffer("update_ratio", torch.zeros(()), persistent=False)
        self.update_ratio = self.get_buffer("update_ratio")
        self.register_buffer("energy_ratio", torch.zeros(()), persistent=False)
        self.energy_ratio = self.get_buffer("energy_ratio")
        self.register_buffer("diagnostic_updates", torch.zeros(()), persistent=False)
        self.diagnostic_updates = self.get_buffer("diagnostic_updates")

    def _gate(self, real: Tensor, imag: Tensor) -> Tensor:
        return _magnitude_gate(
            real,
            imag,
            self.alpha,
            self.gate_redistribution,
        )

    @torch.no_grad()
    def _normalize_raw_projection_rows_(self) -> None:
        for projection in (self.input_projection, self.output_projection):
            weight_real, weight_imag = unit_row_complex_linear_weight(
                projection.weight_real,
                projection.weight_imag,
                epsilon=PROJECTION_ROW_EPSILON,
            )
            projection.weight_real.copy_(weight_real)
            projection.weight_imag.copy_(weight_imag)

    @torch.no_grad()
    def enable_unit_row_contract_(
        self,
        *,
        residual_scale_max: float,
    ) -> PhaseGatedComplexFFN:
        """Constrain an initialized PG block without changing its learned directions."""
        if self.unit_row_projections or self.residual_scale_max is not None:
            message = "phase-gated unit-row contract is already enabled"
            raise RuntimeError(message)
        gamma = self.gamma
        if gamma is None:
            message = "phase-gated global residual scale is missing"
            raise RuntimeError(message)
        effective_gamma = float(gamma.detach())
        if residual_scale_max <= abs(effective_gamma):
            message = "phase-gated residual scale maximum must exceed the current scale"
            raise ValueError(message)
        self._normalize_raw_projection_rows_()
        gamma.copy_(gamma.new_tensor(math.atanh(effective_gamma / residual_scale_max)))
        self.unit_row_projections = True
        self.residual_scale_max = float(residual_scale_max)
        return self

    @torch.no_grad()
    def enable_direction_output_gain_contract_(
        self,
        *,
        output_gain_max: float,
    ) -> PhaseGatedComplexFFN:
        """Use unit-sphere directions and one bounded gain per output mode."""
        if (
            self.unit_row_projections
            or self.projected_direction_rows
            or self.residual_scale_max is not None
            or self.output_gain_max is not None
        ):
            message = "phase-gated projection scale contract is already enabled"
            raise RuntimeError(message)
        if self.gamma is None:
            message = "phase-gated global residual scale is missing"
            raise RuntimeError(message)
        initial_gain = float(self.gamma.detach())
        if output_gain_max <= abs(initial_gain):
            message = "phase-gated output gain maximum must exceed the initial gain"
            raise ValueError(message)
        self._normalize_raw_projection_rows_()
        initial_logit = math.atanh(initial_gain / output_gain_max)
        self.output_gain_logits = nn.Parameter(self.alpha.new_full((self.modes,), initial_logit))
        self.gamma = None
        self.projected_direction_rows = True
        self.output_gain_max = float(output_gain_max)
        return self

    @torch.no_grad()
    def project_direction_rows_(self) -> None:
        """Retract direction parameters to the complex unit-row sphere."""
        if self.projected_direction_rows:
            self._normalize_raw_projection_rows_()

    def effective_output_gains(self) -> Tensor | None:
        if self.output_gain_max is None:
            return None
        if self.output_gain_logits is None:
            message = "phase-gated output gain logits are missing"
            raise RuntimeError(message)
        return self.output_gain_max * torch.tanh(self.output_gain_logits)

    def effective_projection_weights(self) -> tuple[ComplexField, ComplexField]:
        if self.projected_direction_rows:
            output_gains = self.effective_output_gains()
            if output_gains is None:
                message = "phase-gated output gains are missing"
                raise RuntimeError(message)
            output_scale = output_gains.unsqueeze(-1)
            return (
                (
                    self.input_projection.weight_real,
                    self.input_projection.weight_imag,
                ),
                (
                    self.output_projection.weight_real * output_scale,
                    self.output_projection.weight_imag * output_scale,
                ),
            )
        if not self.unit_row_projections:
            return (
                (
                    self.input_projection.weight_real,
                    self.input_projection.weight_imag,
                ),
                (
                    self.output_projection.weight_real,
                    self.output_projection.weight_imag,
                ),
            )
        input_weight = unit_row_complex_linear_weight(
            self.input_projection.weight_real,
            self.input_projection.weight_imag,
            epsilon=PROJECTION_ROW_EPSILON,
        )
        output_weight = unit_row_complex_linear_weight(
            self.output_projection.weight_real,
            self.output_projection.weight_imag,
            epsilon=PROJECTION_ROW_EPSILON,
        )
        return input_weight, output_weight

    def effective_residual_scale(self) -> Tensor:
        if not self.learnable_residual_scale:
            return self.alpha.new_ones(())
        if self.output_gain_max is not None:
            return self.alpha.new_ones(())
        if self.gamma is None:
            message = "phase-gated global residual scale is missing"
            raise RuntimeError(message)
        if self.residual_scale_max is None:
            return self.gamma
        return self.residual_scale_max * torch.tanh(self.gamma)

    def _validate_input(self, real: Tensor, imag: Tensor) -> None:
        if real.ndim < 1 or real.shape != imag.shape or real.shape[-1] != self.modes:
            message = "phase-gated FFN inputs have incompatible shapes"
            raise ValueError(message)
        if real.device != imag.device or real.dtype != imag.dtype:
            message = "phase-gated FFN coordinates must share one device and dtype"
            raise ValueError(message)

    def _optimized_forward(
        self,
        real: Tensor,
        imag: Tensor,
        *,
        collect_diagnostics: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        input_coordinates, output_coordinates = self.effective_projection_weights()
        input_weight = packed_complex_linear_weight(*input_coordinates).to(dtype=torch.bfloat16)
        output_weight = packed_complex_linear_weight(*output_coordinates).to(dtype=torch.bfloat16)
        residual_scale = self.effective_residual_scale()
        if supports_fused_phase_gated_cffn(
            real,
            imag,
            self.norm.weight,
            input_weight,
            self.alpha,
            output_weight,
            residual_scale,
            epsilon=self.norm.epsilon,
            redistribution=self.gate_redistribution,
            self_gated=self.self_gated,
        ):
            output_real, output_imag, projected, packed_delta = fused_phase_gated_cffn(
                real,
                imag,
                self.norm.weight,
                input_weight,
                self.alpha,
                output_weight,
                residual_scale,
                epsilon=self.norm.epsilon,
                redistribution=self.gate_redistribution,
                self_gated=self.self_gated,
                collect_diagnostics=collect_diagnostics,
            )
            delta_real, delta_imag = packed_delta.split(self.modes, dim=-1)
            return output_real, output_imag, projected, delta_real, delta_imag
        if supports_fused_rmsnorm_input_linear(
            real,
            imag,
            self.norm.weight,
            input_weight,
            epsilon=self.norm.epsilon,
        ):
            projected = fused_rmsnorm_input_linear(
                real,
                imag,
                self.norm.weight,
                input_weight,
                epsilon=self.norm.epsilon,
            )
        else:
            normalized = packed_complex_rms_norm(
                real,
                imag,
                self.norm.weight,
                self.norm.epsilon,
            )
            projected = functional.linear(normalized, input_weight)
        if supports_fused_phase_gate_output_residual(
            projected,
            self.alpha,
            output_weight,
            real,
            imag,
            residual_scale,
            redistribution=self.gate_redistribution,
            self_gated=self.self_gated,
        ):
            output_real, output_imag, packed_delta = fused_phase_gate_output_residual(
                projected,
                self.alpha,
                output_weight,
                real,
                imag,
                residual_scale,
                redistribution=self.gate_redistribution,
                self_gated=self.self_gated,
            )
        else:
            packed_delta = phase_gate_output_linear(
                projected,
                self.alpha,
                output_weight,
                redistribution=self.gate_redistribution,
                self_gated=self.self_gated,
            )
            packed_update = packed_delta * residual_scale.to(dtype=packed_delta.dtype)
            update_real, update_imag = packed_update.split(self.modes, dim=-1)
            output_real = real + update_real
            output_imag = imag + update_imag
        delta_real, delta_imag = packed_delta.split(self.modes, dim=-1)
        return (
            output_real,
            output_imag,
            projected,
            delta_real,
            delta_imag,
        )

    def _uses_bf16_cuda_execution(self, real: Tensor) -> bool:
        bf16_source = real.dtype is torch.bfloat16
        fp32_autocast_source = (
            real.dtype is torch.float32
            and torch.is_autocast_enabled("cuda")
            and torch.get_autocast_dtype("cuda") is torch.bfloat16
        )
        return real.is_cuda and real.numel() > 0 and (bf16_source or fp32_autocast_source)

    def _validate_packed_components(self) -> None:
        exact_components = (
            type(self.norm) is ComplexRMSNorm
            and type(self.input_projection) is ComplexLinear
            and type(self.output_projection) is ComplexLinear
        )
        transparent_calls = module_calls_are_transparent(
            self.norm,
            self.input_projection,
            self.output_projection,
        )
        fp32_parameters = all(
            parameter.dtype is torch.float32 and parameter.is_contiguous()
            for parameter in self.parameters()
        )
        if not exact_components or not transparent_calls or not fp32_parameters:
            message = (
                "BF16 CUDA Phase-Gated CFFN requires exact, hook-free components "
                "with contiguous FP32 parameters"
            )
            raise RuntimeError(message)

    def _reference_forward_with_auxiliary(
        self,
        real: Tensor,
        imag: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        unit_real, unit_imag = self.norm(real, imag)
        input_weight, output_weight = self.effective_projection_weights()
        projected_real, projected_imag = _strict_complex_linear(
            unit_real,
            unit_imag,
            *input_weight,
        )
        value_real, value_imag, gate_real, gate_imag = _phase_gate_coordinates(
            projected_real,
            projected_imag,
            self.hidden_modes,
            self_gated=self.self_gated,
        )
        gate = self._gate(gate_real, gate_imag)
        delta_real, delta_imag = _strict_complex_linear(
            value_real * gate,
            value_imag * gate,
            *output_weight,
        )
        residual_scale = self.effective_residual_scale()
        return (
            real + residual_scale * delta_real,
            imag + residual_scale * delta_imag,
            gate,
            delta_real,
            delta_imag,
        )

    def _sample_gate(self, projected: Tensor) -> Tensor:
        sample_rows = diagnostic_sample_rows(projected)
        sample = projected.reshape(-1, projected.shape[-1])[:sample_rows]
        if self.self_gated:
            gate_real, gate_imag = sample.split(self.hidden_modes, dim=-1)
        else:
            _, gate_real, _, gate_imag = sample.split(self.hidden_modes, dim=-1)
        return self._gate(gate_real, gate_imag)

    @torch.no_grad()
    def _update_diagnostics(
        self,
        gate: Tensor,
        source_real: Tensor,
        source_imag: Tensor,
        delta_real: Tensor,
        delta_imag: Tensor,
        output_real: Tensor,
        output_imag: Tensor,
    ) -> None:
        if gate.numel() == 0:
            return
        sample_rows = diagnostic_sample_rows(source_real)
        sampled_gate = gate.detach().reshape(-1, self.hidden_modes)[:sample_rows].float()
        sampled_source_real = source_real.detach().reshape(-1, self.modes)[:sample_rows].float()
        sampled_source_imag = source_imag.detach().reshape(-1, self.modes)[:sample_rows].float()
        sampled_delta_real = delta_real.detach().reshape(-1, self.modes)[:sample_rows]
        sampled_delta_imag = delta_imag.detach().reshape(-1, self.modes)[:sample_rows]
        residual_scale = self.effective_residual_scale().detach().to(dtype=sampled_delta_real.dtype)
        sampled_update_real = (sampled_delta_real * residual_scale).float()
        sampled_update_imag = (sampled_delta_imag * residual_scale).float()
        sampled_output_real = output_real.detach().reshape(-1, self.modes)[:sample_rows].float()
        sampled_output_imag = output_imag.detach().reshape(-1, self.modes)[:sample_rows].float()
        source_energy = (sampled_source_real.square() + sampled_source_imag.square()).mean()
        update_energy = (sampled_update_real.square() + sampled_update_imag.square()).mean()
        output_energy = (sampled_output_real.square() + sampled_output_imag.square()).mean()
        values = (
            sampled_gate.mean(),
            sampled_gate.std(unbiased=False),
            sampled_gate.min(),
            sampled_gate.max(),
            torch.sqrt(update_energy / source_energy.clamp_min(1.0e-12)),
            output_energy / source_energy.clamp_min(1.0e-12),
        )
        count = self.diagnostic_updates
        decay = torch.where(count > 0, count.new_tensor(0.95), count.new_zeros(()))
        for target, value in zip(
            (
                self.gate_mean,
                self.gate_std,
                self.gate_min,
                self.gate_max,
                self.update_ratio,
                self.energy_ratio,
            ),
            values,
            strict=True,
        ):
            target.mul_(decay).add_(value * (1.0 - decay))
        self.diagnostic_updates.add_(1)

    def diagnostic_metrics(self) -> dict[str, float]:
        metrics = {
            "gate_mean": float(self.gate_mean),
            "gate_std": float(self.gate_std),
            "gate_min": float(self.gate_min),
            "gate_max": float(self.gate_max),
            "update_ratio": float(self.update_ratio),
            "energy_ratio": float(self.energy_ratio),
            "alpha_mean": float(self.alpha.detach().float().mean()),
            "alpha_std": float(self.alpha.detach().float().std(unbiased=False)),
            "gamma_max": self.residual_scale_max or 0.0,
            "unit_row_projections": float(self.unit_row_projections),
            "projected_direction_rows": float(self.projected_direction_rows),
            "gate_redistribution": self.gate_redistribution,
        }
        output_gains = self.effective_output_gains()
        if not self.learnable_residual_scale:
            metrics.update(
                {
                    "gamma": 1.0,
                    "fixed_unit_residual_scale": 1.0,
                }
            )
        elif output_gains is not None:
            gains = output_gains.detach().float()
            metrics.update(
                {
                    "output_gain_mean": float(gains.mean()),
                    "output_gain_std": float(gains.std(unbiased=False)),
                    "output_gain_abs_mean": float(gains.abs().mean()),
                    "output_gain_min": float(gains.min()),
                    "output_gain_max": float(gains.max()),
                    "output_gain_bound": self.output_gain_max or 0.0,
                }
            )
        else:
            if self.gamma is None:
                message = "phase-gated global residual scale is missing"
                raise RuntimeError(message)
            metrics.update(
                {
                    "gamma": float(self.effective_residual_scale().detach()),
                    "gamma_parameter": float(self.gamma.detach()),
                }
            )
        return metrics

    @staticmethod
    def _gradient_norm(*gradients: Tensor) -> float:
        active = [gradient.detach().float() for gradient in gradients]
        squared_norm = active[0].square().sum()
        for gradient in active[1:]:
            squared_norm = squared_norm + gradient.square().sum()
        return float(torch.sqrt(squared_norm))

    @torch.no_grad()
    def gradient_metrics(self) -> dict[str, float]:
        """Report the latest Phase-Gated branch gradients after a train epoch."""
        input_real = self.input_projection.weight_real.grad
        input_imag = self.input_projection.weight_imag.grad
        output_real = self.output_projection.weight_real.grad
        output_imag = self.output_projection.weight_imag.grad
        alpha = self.alpha.grad
        scale_parameter = (
            self.output_gain_logits if self.output_gain_logits is not None else self.gamma
        )
        scale_gradient = None if scale_parameter is None else scale_parameter.grad
        if (
            input_real is None
            or input_imag is None
            or output_real is None
            or output_imag is None
            or alpha is None
            or (self.learnable_residual_scale and scale_gradient is None)
        ):
            return {}
        value_norm = self._gradient_norm(
            input_real[: self.hidden_modes],
            input_imag[: self.hidden_modes],
        )
        metrics = {
            "input_value_weight_grad_norm": value_norm,
            "output_weight_grad_norm": self._gradient_norm(output_real, output_imag),
            "alpha_grad_norm": self._gradient_norm(alpha),
        }
        if scale_gradient is not None:
            metrics[
                "output_gain_grad_norm"
                if self.output_gain_logits is not None
                else "gamma_grad_norm"
            ] = self._gradient_norm(scale_gradient)
        if not self.self_gated:
            gate_norm = self._gradient_norm(
                input_real[self.hidden_modes :],
                input_imag[self.hidden_modes :],
            )
            metrics.update(
                {
                    "input_gate_weight_grad_norm": gate_norm,
                    "input_gate_to_value_grad_ratio": gate_norm
                    / max(value_norm, torch.finfo(torch.float32).tiny),
                }
            )
        return metrics

    def _forward(
        self,
        real: Tensor,
        imag: Tensor,
        *,
        update_diagnostics: bool,
    ) -> ComplexField:
        self._validate_input(real, imag)
        if self._uses_bf16_cuda_execution(real):
            self._validate_packed_components()
            output_real, output_imag, projected, delta_real, delta_imag = self._optimized_forward(
                real,
                imag,
                collect_diagnostics=update_diagnostics,
            )
            gate = self._sample_gate(projected) if update_diagnostics else None
        else:
            output_real, output_imag, gate, delta_real, delta_imag = (
                self._reference_forward_with_auxiliary(real, imag)
            )
        if update_diagnostics:
            if gate is None:
                message = "Phase-Gated diagnostics require a sampled gate"
                raise RuntimeError(message)
            self._update_diagnostics(
                gate,
                real,
                imag,
                delta_real,
                delta_imag,
                output_real,
                output_imag,
            )
        return output_real, output_imag

    def _forward_without_diagnostics(self, real: Tensor, imag: Tensor) -> ComplexField:
        return self._forward(real, imag, update_diagnostics=False)

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        return self._forward(real, imag, update_diagnostics=True)


__all__ = [
    "GATE_REDISTRIBUTION",
    "PhaseGatedComplexFFN",
    "phase_gated_complex_ffn_reference",
]
