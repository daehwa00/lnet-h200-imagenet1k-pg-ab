"""Configurable vector-input pole memory stage for controlled ablations."""

from __future__ import annotations

# pyright: reportPrivateUsage=false
# ruff: noqa: SLF001
from typing import TYPE_CHECKING, Literal, cast

import torch
from torch import Tensor, nn

from .complex_scan_stage import complex_carry_coordinates
from .complex_scan_transitions import ComplexRMSNorm
from .pac_complex_layers import PackedComplexLinear, semi_orthogonal_complex_linear_
from .pac_factorized_wide_pole_memory import FactorizedWidePoleMemoryStage
from .pac_phase_gated_cffn import PhaseGatedComplexFFN
from .pac_product_scan_pipeline import ScanMemoryPolicy, run_product_scan_pipeline

if TYPE_CHECKING:
    from .complex_scan_types import ComplexField

MemoryAdapter = Literal["complex", "real_shared"]
CarryKind = Literal["learned_s2d", "fixed_average", "none"]
DescriptorKind = Literal["direct", "projected"]

_DIRECTIONS = 4


class RealSharedLinear(nn.Module):
    """Apply one real matrix to both coordinates of a complex vector."""

    def __init__(self, input_modes: int, output_modes: int) -> None:
        super().__init__()
        if min(input_modes, output_modes) <= 0:
            message = "real-shared linear dimensions must be positive"
            raise ValueError(message)
        self.input_modes = input_modes
        self.output_modes = output_modes
        self.linear = nn.Linear(input_modes, output_modes, bias=False)
        nn.init.orthogonal_(self.linear.weight)

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        if real.shape != imag.shape or real.shape[-1] != self.input_modes:
            message = "real-shared linear inputs have incompatible shapes"
            raise ValueError(message)
        packed = torch.stack((real, imag), dim=-2)
        projected = self.linear(packed)
        return projected[..., 0, :], projected[..., 1, :]


class PoleBackboneAblationStage(FactorizedWidePoleMemoryStage):
    """Map normalized vector excitations to pole memory and back to a vector."""

    excitation_rms: Tensor
    normalized_input_rms: Tensor
    memory_adapter_rms: Tensor
    carry_memory_rms_ratio: Tensor
    memory_carry_correlation_real: Tensor
    memory_carry_correlation_magnitude: Tensor
    merged_excitation_rms: Tensor
    output_excitation_rms: Tensor
    q_rms: Tensor
    q_variance: Tensor
    q_pole_entropy: Tensor
    q_pole_effective_count: Tensor
    q_pole_max_share: Tensor
    q_pole_energy: Tensor
    q_pole_share: Tensor

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
        pole_pg_hidden: int | None,
        memory_adapter: MemoryAdapter,
        precarry_memory_pg_hidden: int | None,
        carry_kind: CarryKind,
        reexcitation_hidden: int | None,
        descriptor_kind: DescriptorKind,
        descriptor_modes: int = 96,
        scan_memory_policy: ScanMemoryPolicy = "retain",
        damping_min: float = 0.01,
        damping_max: float = 0.7,
    ) -> None:
        self._validate_constructor(
            terminal=terminal,
            next_modes=next_modes,
            pole_pg_hidden=pole_pg_hidden,
            precarry_memory_pg_hidden=precarry_memory_pg_hidden,
            reexcitation_hidden=reexcitation_hidden,
            descriptor_kind=descriptor_kind,
            descriptor_modes=descriptor_modes,
        )
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
        self.memory_adapter_kind = memory_adapter
        self.carry_kind = carry_kind
        self.descriptor_kind = descriptor_kind
        self.descriptor_modes = descriptor_modes
        self.input_norm = ComplexRMSNorm(content_modes)
        self.pole_input = PackedComplexLinear(content_modes, poles)
        semi_orthogonal_complex_linear_(self.pole_input)
        self._initialize_descriptor(descriptor_kind, descriptor_modes, poles)
        self._initialize_transition(
            terminal=terminal,
            content_modes=content_modes,
            poles=poles,
            next_modes=next_modes,
            pole_pg_hidden=pole_pg_hidden,
            memory_adapter=memory_adapter,
            precarry_memory_pg_hidden=precarry_memory_pg_hidden,
            carry_kind=carry_kind,
            reexcitation_hidden=reexcitation_hidden,
        )
        self._register_diagnostic_buffers()
        self._register_ablation_diagnostics(poles)
        self.excitation_rms = self.get_buffer("excitation_rms")
        self.normalized_input_rms = self.get_buffer("normalized_input_rms")
        self.memory_adapter_rms = self.get_buffer("memory_adapter_rms")
        self.carry_memory_rms_ratio = self.get_buffer("carry_memory_rms_ratio")
        self.memory_carry_correlation_real = self.get_buffer(
            "memory_carry_correlation_real"
        )
        self.memory_carry_correlation_magnitude = self.get_buffer(
            "memory_carry_correlation_magnitude"
        )
        self.merged_excitation_rms = self.get_buffer("merged_excitation_rms")
        self.output_excitation_rms = self.get_buffer("output_excitation_rms")
        self.q_rms = self.get_buffer("q_rms")
        self.q_variance = self.get_buffer("q_variance")
        self.q_pole_entropy = self.get_buffer("q_pole_entropy")
        self.q_pole_effective_count = self.get_buffer("q_pole_effective_count")
        self.q_pole_max_share = self.get_buffer("q_pole_max_share")
        self.q_pole_energy = self.get_buffer("q_pole_energy")
        self.q_pole_share = self.get_buffer("q_pole_share")

    @staticmethod
    def _validate_constructor(
        *,
        terminal: bool,
        next_modes: int | None,
        pole_pg_hidden: int | None,
        precarry_memory_pg_hidden: int | None,
        reexcitation_hidden: int | None,
        descriptor_kind: DescriptorKind,
        descriptor_modes: int,
    ) -> None:
        if terminal != (next_modes is None):
            message = "only the terminal pole-ablation stage may omit next_modes"
            raise ValueError(message)
        if terminal and any(
            value is not None
            for value in (pole_pg_hidden, precarry_memory_pg_hidden, reexcitation_hidden)
        ):
            message = "terminal pole-ablation stage cannot contain transition PG blocks"
            raise ValueError(message)
        if not terminal and reexcitation_hidden is not None and reexcitation_hidden <= 0:
            message = "re-excitation hidden width must be positive when enabled"
            raise ValueError(message)
        if descriptor_kind == "projected" and descriptor_modes <= 0:
            message = "projected descriptor width must be positive"
            raise ValueError(message)

    def _initialize_descriptor(
        self,
        descriptor_kind: DescriptorKind,
        descriptor_modes: int,
        poles: int,
    ) -> None:
        self.descriptor_projection = (
            PackedComplexLinear(poles, descriptor_modes)
            if descriptor_kind == "projected"
            else None
        )
        if self.descriptor_projection is not None:
            semi_orthogonal_complex_linear_(self.descriptor_projection)

    def _initialize_transition(
        self,
        *,
        terminal: bool,
        content_modes: int,
        poles: int,
        next_modes: int | None,
        pole_pg_hidden: int | None,
        memory_adapter: MemoryAdapter,
        precarry_memory_pg_hidden: int | None,
        carry_kind: CarryKind,
        reexcitation_hidden: int | None,
    ) -> None:
        self.memory_pole_pg = (
            PhaseGatedComplexFFN(poles, pole_pg_hidden)
            if pole_pg_hidden is not None
            else None
        )
        if terminal:
            self.memory_adapter = None
            self.precarry_memory_pg = None
            self.learned_carry = None
            self.fixed_carry_projection = None
            self.reexcitation_pg = None
        else:
            if next_modes is None:
                message = "non-terminal pole-ablation stage requires next_modes"
                raise RuntimeError(message)
            adapter_input = _DIRECTIONS * poles
            if memory_adapter == "complex":
                self.memory_adapter = PackedComplexLinear(adapter_input, next_modes)
                semi_orthogonal_complex_linear_(self.memory_adapter)
            else:
                self.memory_adapter = RealSharedLinear(adapter_input, next_modes)
            self.precarry_memory_pg = (
                PhaseGatedComplexFFN(next_modes, precarry_memory_pg_hidden)
                if precarry_memory_pg_hidden is not None
                else None
            )
            if carry_kind == "learned_s2d":
                self.learned_carry = PackedComplexLinear(
                    _DIRECTIONS * content_modes,
                    next_modes,
                )
                semi_orthogonal_complex_linear_(self.learned_carry)
                self.fixed_carry_projection = None
            elif carry_kind == "fixed_average":
                self.learned_carry = None
                self.fixed_carry_projection = (
                    RealSharedLinear(content_modes, next_modes)
                    if content_modes != next_modes
                    else None
                )
            else:
                self.learned_carry = None
                self.fixed_carry_projection = None
            self.reexcitation_pg = (
                PhaseGatedComplexFFN(next_modes, reexcitation_hidden)
                if reexcitation_hidden is not None
                else None
            )

    def _register_ablation_diagnostics(self, poles: int) -> None:
        for name in (
            "excitation_rms",
            "normalized_input_rms",
            "memory_adapter_rms",
            "carry_memory_rms_ratio",
            "memory_carry_correlation_real",
            "memory_carry_correlation_magnitude",
            "merged_excitation_rms",
            "output_excitation_rms",
            "q_rms",
            "q_variance",
            "q_pole_entropy",
            "q_pole_effective_count",
            "q_pole_max_share",
        ):
            self.register_buffer(name, torch.zeros(()), persistent=False)
        self.register_buffer("q_pole_energy", torch.zeros(poles), persistent=False)
        self.register_buffer("q_pole_share", torch.zeros(poles), persistent=False)

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

    def _descriptor_and_raw_q(self, real: Tensor, imag: Tensor) -> tuple[Tensor, Tensor]:
        raw_energy = real.float().square().add(imag.float().square()).mean((1, 2))
        raw_q = torch.log1p(raw_energy)
        if self.descriptor_projection is None:
            return raw_q.flatten(1), raw_q
        projected_real, projected_imag = self.descriptor_projection(real, imag)
        projected_energy = projected_real.float().square().add(projected_imag.float().square())
        descriptor = torch.log1p(projected_energy.mean((1, 2))).flatten(1)
        return descriptor, raw_q

    def _process_memory(
        self,
        real: Tensor,
        imag: Tensor,
        *,
        update_diagnostics: bool = True,
    ) -> ComplexField:
        if self.memory_pole_pg is None:
            processed = real, imag
        else:
            processed = self.memory_pole_pg._forward(
                real,
                imag,
                update_diagnostics=update_diagnostics,
            )
        return processed

    def _adapt_memory(self, real: Tensor, imag: Tensor) -> ComplexField:
        if self.memory_adapter is None:
            message = "terminal pole-ablation stage has no memory adapter"
            raise RuntimeError(message)
        return self.memory_adapter(real.flatten(-2), imag.flatten(-2))

    def _transition_carry(self, real: Tensor, imag: Tensor) -> ComplexField | None:
        if self.carry_kind == "none":
            return None
        carry_real, carry_imag = complex_carry_coordinates(real, imag, "s2d")
        if self.carry_kind == "learned_s2d":
            if self.learned_carry is None:
                message = "learned S2D carry is missing its projection"
                raise RuntimeError(message)
            return self.learned_carry(carry_real, carry_imag)
        shape = (*carry_real.shape[:-1], _DIRECTIONS, self.content_modes)
        averaged = carry_real.reshape(shape).mean(-2), carry_imag.reshape(shape).mean(-2)
        if self.fixed_carry_projection is None:
            return averaged
        return self.fixed_carry_projection(*averaged)

    @staticmethod
    def _rms(field: ComplexField) -> Tensor:
        energy = field[0].detach().float().square().add(field[1].detach().float().square())
        return energy.mean().sqrt()

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
        raw_energy = sampled_real.square().add(sampled_imag.square())
        self._ema(self.raw_memory_rms, raw_energy.mean().sqrt())

    @torch.no_grad()
    def _update_q_diagnostics(self, descriptor: Tensor, raw_q: Tensor) -> None:
        active = descriptor.detach().float()
        self._ema(self.q_rms, active.square().mean().sqrt())
        self._ema(self.q_variance, active.var(unbiased=False))
        pole_energy = torch.expm1(raw_q.detach().float()).mean((0, 1))
        pole_share = pole_energy / pole_energy.sum().clamp_min(1.0e-12)
        entropy = -(pole_share * pole_share.clamp_min(1.0e-12).log()).sum()
        self._ema(self.q_pole_entropy, entropy)
        self._ema(self.q_pole_effective_count, pole_share.square().sum().reciprocal())
        self._ema(self.q_pole_max_share, pole_share.max())
        self._ema(self.q_pole_energy, pole_energy)
        self._ema(self.q_pole_share, pole_share)

    @torch.no_grad()
    def _update_transition_diagnostics_compact(
        self,
        source: ComplexField,
        normalized: ComplexField,
        processed: ComplexField,
        adapted: ComplexField | None,
        carry: ComplexField | None,
        merged: ComplexField | None,
        output: ComplexField | None,
    ) -> None:
        self._ema(self.excitation_rms, self._rms(source))
        self._ema(self.normalized_input_rms, self._rms(normalized))
        self._ema(self.processed_memory_rms, self._rms(processed))
        if adapted is None:
            return
        adapted_rms = self._rms(adapted)
        self._ema(self.memory_adapter_rms, adapted_rms)
        self._ema(self.memory_readout_rms, adapted_rms)
        if carry is not None:
            carry_rms = self._rms(carry)
            self._ema(self.carry_rms, carry_rms)
            self._ema(self.memory_carry_rms_ratio, adapted_rms / carry_rms.clamp_min(1.0e-12))
            self._ema(self.carry_memory_rms_ratio, carry_rms / adapted_rms.clamp_min(1.0e-12))
            height = min(2, adapted[0].shape[1])
            width = min(2, adapted[0].shape[2])
            ar = adapted[0][:1, :height, :width].detach().float()
            ai = adapted[1][:1, :height, :width].detach().float()
            cr = carry[0][:1, :height, :width].detach().float()
            ci = carry[1][:1, :height, :width].detach().float()
            dot_real = (ar * cr + ai * ci).mean()
            dot_imag = (ai * cr - ar * ci).mean()
            denominator = ar.square().add(ai.square()).mean().mul(
                cr.square().add(ci.square()).mean()
            ).sqrt().clamp_min(1.0e-12)
            self._ema(self.memory_carry_correlation_real, dot_real / denominator)
            self._ema(
                self.memory_carry_correlation_magnitude,
                torch.sqrt(dot_real.square() + dot_imag.square()) / denominator,
            )
        if merged is not None:
            self._ema(self.merged_excitation_rms, self._rms(merged))
        if output is not None:
            self._ema(self.output_excitation_rms, self._rms(output))

    @staticmethod
    def _linear_metrics(
        layer: PackedComplexLinear | RealSharedLinear,
        prefix: str,
    ) -> dict[str, float]:
        if isinstance(layer, PackedComplexLinear):
            weight = torch.complex(
                layer.weight_real.detach().float(),
                layer.weight_imag.detach().float(),
            )
            singular = torch.linalg.svdvals(weight)
        else:
            singular = torch.linalg.svdvals(layer.linear.weight.detach().float())
        squared = singular.square()
        return {
            f"{prefix}_singular_min": float(singular.min()),
            f"{prefix}_singular_mean": float(singular.mean()),
            f"{prefix}_singular_max": float(singular.max()),
            f"{prefix}_effective_rank": float(
                squared.sum().square() / squared.square().sum().clamp_min(1.0e-12)
            ),
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
            "normalized_input_rms": float(self.normalized_input_rms),
            "raw_memory_rms": float(self.raw_memory_rms),
            "processed_memory_rms": float(self.processed_memory_rms),
            "memory_adapter_rms": float(self.memory_adapter_rms),
            "s2d_carry_rms": float(self.carry_rms),
            "memory_carry_rms_ratio": float(self.memory_carry_rms_ratio),
            "carry_memory_rms_ratio": float(self.carry_memory_rms_ratio),
            "memory_carry_correlation_real": float(self.memory_carry_correlation_real),
            "memory_carry_correlation_magnitude": float(
                self.memory_carry_correlation_magnitude
            ),
            "merged_excitation_rms": float(self.merged_excitation_rms),
            "output_excitation_rms": float(self.output_excitation_rms),
            "q_rms": float(self.q_rms),
            "q_variance": float(self.q_variance),
            "q_pole_entropy": float(self.q_pole_entropy),
            "q_pole_effective_count": float(self.q_pole_effective_count),
            "q_pole_max_share": float(self.q_pole_max_share),
            "pole_damping_mean": float(torch.cat((damping_x, damping_y)).mean()),
            "pole_damping_std": float(torch.cat((damping_x, damping_y)).std(unbiased=False)),
            "pole_response_rms_mean": float(self.response_rms_mean),
            "pole_response_rms_std": float(self.response_rms_std),
            "pole_response_correlation_mean": float(self.response_correlation_mean),
            "pole_response_correlation_max": float(self.response_correlation_max),
            "pole_effective_rank": float(self.pole_effective_rank),
        }
        metrics.update(self._linear_metrics(self.pole_input, "pole_input"))
        if self.memory_adapter is not None:
            metrics.update(self._linear_metrics(self.memory_adapter, "memory_adapter"))
        if self.learned_carry is not None:
            metrics.update(self._linear_metrics(self.learned_carry, "learned_carry"))
        if self.fixed_carry_projection is not None:
            metrics.update(
                self._linear_metrics(self.fixed_carry_projection, "fixed_carry_projection")
            )
        for index, (energy, share) in enumerate(
            zip(self.q_pole_energy, self.q_pole_share, strict=True)
        ):
            metrics[f"q_pole_energy_{index:03d}"] = float(energy)
            metrics[f"q_pole_share_{index:03d}"] = float(share)
        return metrics

    def _forward_impl(
        self,
        real: Tensor,
        imag: Tensor,
        *,
        update_diagnostics: bool,
    ) -> tuple[ComplexField | None, Tensor]:
        source = real, imag
        normalized = self.input_norm(real, imag)
        shape = cast("tuple[int, int, int, int]", tuple(real.shape))
        raw_memory = self._scan_normalized(normalized, shape)
        descriptor, raw_q = self._descriptor_and_raw_q(*raw_memory)
        if update_diagnostics:
            self._update_response_diagnostics(*raw_memory)
            self._update_q_diagnostics(descriptor, raw_q)

        if self.terminal:
            if update_diagnostics:
                self._update_transition_diagnostics_compact(
                    source,
                    normalized,
                    raw_memory,
                    None,
                    None,
                    None,
                    None,
                )
                self.diagnostic_updates.add_(1)
            return None, descriptor

        processed = self._process_memory(
            *raw_memory,
            update_diagnostics=update_diagnostics,
        )
        adapted = self._adapt_memory(*processed)
        if self.precarry_memory_pg is not None:
            adapted = self.precarry_memory_pg._forward(
                *adapted,
                update_diagnostics=update_diagnostics,
            )
        carry = self._transition_carry(real, imag)
        merged = adapted if carry is None else (adapted[0] + carry[0], adapted[1] + carry[1])
        output = (
            self.reexcitation_pg._forward(
                *merged,
                update_diagnostics=update_diagnostics,
            )
            if self.reexcitation_pg is not None
            else merged
        )
        if update_diagnostics:
            self._update_transition_diagnostics_compact(
                source,
                normalized,
                processed,
                adapted,
                carry,
                merged,
                output,
            )
            self.diagnostic_updates.add_(1)
        return output, descriptor

    def forward(self, real: Tensor, imag: Tensor) -> tuple[ComplexField | None, Tensor]:
        if real.shape != imag.shape or real.ndim != 4 or real.shape[-1] != self.content_modes:
            message = "pole-ablation stage inputs must be matching NHW-content tensors"
            raise ValueError(message)
        return self._forward_impl(real, imag, update_diagnostics=self.training)


__all__ = [
    "CarryKind",
    "DescriptorKind",
    "MemoryAdapter",
    "PoleBackboneAblationStage",
    "RealSharedLinear",
]
