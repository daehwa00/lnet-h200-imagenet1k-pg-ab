"""Factorized wide pole memory with compact shared excitations.

Each stage expands 32 complex content channels across a stage-specific bank
of shared pole dynamics.  The optimized product scan sees the Cartesian
content-by-pole axis as one wide mode dimension, while all processing after
the scan restores the semantic ``direction x content x pole`` axes.
"""

from __future__ import annotations

# pyright: reportPrivateUsage=false
# ruff: noqa: SLF001
import math
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .complex_scan_stage import complex_carry_coordinates
from .pac_complex_layers import ComplexLinear
from .pac_phase_gated_cffn import PhaseGatedComplexFFN
from .pac_product_scan_normalization import static_variance_tables
from .pac_product_scan_pipeline import ScanMemoryPolicy, run_product_scan_pipeline
from .pac_real2d_math import discrete_pole_real2d
from .pac_triton_bidirectional_product_scan import (
    pac_triton_bidirectional_product_scan,
)

if TYPE_CHECKING:
    from .complex_scan_types import ComplexField

Pole = tuple[Tensor, Tensor, Tensor, Tensor]

_DIRECTIONS = ((0, 0), (1, 0), (0, 1), (1, 1))
_ORIENTATIONS = 8
_DESCRIPTOR_DYNAMICS = 3


def _descriptor_basis(poles: int) -> Tensor:
    """Return three distinct real readouts with matched row energy."""
    coordinate = torch.arange(poles, dtype=torch.float32)
    angle = 2.0 * math.pi * (coordinate + 0.5) / poles
    scale = math.sqrt(2.0) / poles
    return torch.stack(
        (
            torch.full((poles,), 1.0 / poles),
            scale * torch.cos(angle),
            scale * torch.sin(angle),
        )
    )


class FactorizedWidePoleMemoryStage(nn.Module):
    """Expand compact content into a shared wide pole bank and re-excite it."""

    scan_memory_policy: ScanMemoryPolicy
    carry_weight: nn.Parameter | None
    response_rms_mean: Tensor
    response_rms_std: Tensor
    response_correlation_mean: Tensor
    response_correlation_max: Tensor
    pole_effective_rank: Tensor
    raw_memory_rms: Tensor
    processed_memory_rms: Tensor
    memory_readout_rms: Tensor
    carry_rms: Tensor
    memory_carry_rms_ratio: Tensor
    diagnostic_updates: Tensor
    _off_diagonal: Tensor

    def __init__(
        self,
        content_modes: int,
        poles: int,
        *,
        stage_index: int,
        maximum_phase: float,
        frequency_scale: float,
        damping_scale: float,
        terminal: bool,
        scan_memory_policy: ScanMemoryPolicy = "recompute",
        damping_min: float = 0.01,
        damping_max: float = 0.7,
    ) -> None:
        super().__init__()
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

        # PGv2 hidden sizes implement 32->64->32, R->32->R, and 4->16->4.
        self.content_pg = PhaseGatedComplexFFN(content_modes, content_modes)
        self.pole_pg = PhaseGatedComplexFFN(poles, 16)
        self.path_pg = PhaseGatedComplexFFN(4, 8)

        descriptor = _descriptor_basis(poles).unsqueeze(0).repeat(content_modes, 1, 1)
        self.descriptor_readout_real = nn.Parameter(descriptor)
        self.descriptor_readout_imag = nn.Parameter(torch.zeros_like(descriptor))

        if terminal:
            self.direction_readout = None
            self.register_parameter("pole_readout_real", None)
            self.register_parameter("pole_readout_imag", None)
            self.register_parameter("carry_weight", None)
            self.post_pg = None
        else:
            self.direction_readout = ComplexLinear(4, 1)
            with torch.no_grad():
                self.direction_readout.weight_real.fill_(0.25)
                self.direction_readout.weight_imag.zero_()
            self.pole_readout_real = nn.Parameter(torch.full((content_modes, poles), 1.0 / poles))
            self.pole_readout_imag = nn.Parameter(torch.zeros(content_modes, poles))
            self.carry_weight = nn.Parameter(torch.full((content_modes, 4), 0.25))
            self.post_pg = PhaseGatedComplexFFN(content_modes, content_modes)

        self._register_diagnostic_buffers()

    def _initialize_pole_parameters(
        self,
        content_modes: int,
        poles: int,
        *,
        stage_index: int,
        maximum_phase: float,
        frequency_scale: float,
        damping_scale: float,
        terminal: bool,
        scan_memory_policy: ScanMemoryPolicy,
        damping_min: float,
        damping_max: float,
    ) -> None:
        if content_modes <= 0 or poles < _ORIENTATIONS or poles % _ORIENTATIONS:
            message = "wide pole memory requires positive content and complete 8-pole groups"
            raise ValueError(message)
        if stage_index not in range(4):
            message = "wide pole memory stage index must be in [0, 3]"
            raise ValueError(message)
        if scan_memory_policy not in {"retain", "recompute"}:
            message = f"unsupported scan memory policy: {scan_memory_policy}"
            raise ValueError(message)
        self.content_modes = content_modes
        self.poles = poles
        self.stage_index = stage_index
        self.modes = content_modes
        self.output_modes = None if terminal else content_modes
        self.terminal = terminal
        self.scan_memory_policy = scan_memory_policy
        self.damping_min = damping_min
        self.damping_max = damping_max

        radial_levels = poles // _ORIENTATIONS
        calibrated_phase = maximum_phase * frequency_scale
        radial = torch.logspace(
            math.log10(calibrated_phase / 8.0),
            math.log10(calibrated_phase),
            radial_levels,
        ).repeat_interleave(_ORIENTATIONS)
        orientation = torch.linspace(0.0, math.pi / 2.0, _ORIENTATIONS).repeat(radial_levels)
        damping = torch.logspace(
            math.log10(0.04 * damping_scale),
            math.log10(0.35 * damping_scale),
            radial_levels,
        ).repeat_interleave(_ORIENTATIONS)
        ratio = ((damping - damping_min) / (damping_max - damping_min)).clamp(
            1.0e-4,
            1.0 - 1.0e-4,
        )
        damping_logits = torch.logit(ratio)
        self.damping_logits_x = nn.Parameter(damping_logits.clone())
        self.damping_logits_y = nn.Parameter(damping_logits.clone())
        self.phase_x = nn.Parameter(radial * torch.cos(orientation))
        self.phase_y = nn.Parameter(radial * torch.sin(orientation))

    def _register_diagnostic_buffers(self) -> None:
        for name in (
            "response_rms_mean",
            "response_rms_std",
            "response_correlation_mean",
            "response_correlation_max",
            "pole_effective_rank",
            "raw_memory_rms",
            "processed_memory_rms",
            "memory_readout_rms",
            "carry_rms",
            "memory_carry_rms_ratio",
        ):
            self.register_buffer(name, torch.zeros(()), persistent=False)
        self.response_rms_mean = self.get_buffer("response_rms_mean")
        self.response_rms_std = self.get_buffer("response_rms_std")
        self.response_correlation_mean = self.get_buffer("response_correlation_mean")
        self.response_correlation_max = self.get_buffer("response_correlation_max")
        self.pole_effective_rank = self.get_buffer("pole_effective_rank")
        self.raw_memory_rms = self.get_buffer("raw_memory_rms")
        self.processed_memory_rms = self.get_buffer("processed_memory_rms")
        self.memory_readout_rms = self.get_buffer("memory_readout_rms")
        self.carry_rms = self.get_buffer("carry_rms")
        self.memory_carry_rms_ratio = self.get_buffer("memory_carry_rms_ratio")
        self.register_buffer(
            "diagnostic_updates",
            torch.zeros((), dtype=torch.int64),
            persistent=False,
        )
        self.diagnostic_updates = self.get_buffer("diagnostic_updates")
        self.register_buffer(
            "_off_diagonal",
            ~torch.eye(self.poles, dtype=torch.bool),
            persistent=False,
        )
        self._off_diagonal = self.get_buffer("_off_diagonal")

    def _compact_pole_coefficients(
        self,
        shape: tuple[int, int, int, int],
    ) -> tuple[Pole, Pole]:
        _, height, width, modes = shape
        if modes != self.content_modes:
            message = "wide pole scan received the wrong content width"
            raise ValueError(message)
        span = self.damping_max - self.damping_min
        damping_x = self.damping_min + span * torch.sigmoid(self.damping_logits_x)
        damping_y = self.damping_min + span * torch.sigmoid(self.damping_logits_y)
        spacing_x, spacing_y = 1.0 / width, 1.0 / height
        pole_x = discrete_pole_real2d(
            damping_x.view(1, 1, 1, -1) / spacing_x,
            (self.phase_x / spacing_x).view(1, 1, 1, -1),
            spacing_x,
        )
        pole_y = discrete_pole_real2d(
            damping_y.view(1, 1, 1, -1) / spacing_y,
            (self.phase_y / spacing_y).view(1, 1, 1, -1),
            spacing_y,
        )
        return pole_x, pole_y

    def _pole_coefficients(self, shape: tuple[int, int, int, int]) -> tuple[Pole, Pole]:
        pole_x, pole_y = self._compact_pole_coefficients(shape)

        def across_content(pole: Pole) -> Pole:
            return cast(
                "Pole",
                tuple(value.repeat_interleave(self.content_modes, dim=-1) for value in pole),
            )

        return across_content(pole_x), across_content(pole_y)

    def _expanded_excitation(self, value: Tensor) -> Tensor:
        shape = (*value.shape[:-1], self.poles, self.content_modes)
        return value.unsqueeze(-2).expand(shape).reshape(*value.shape[:-1], -1)

    def _terminal_full_states(
        self,
        pole_x: Pole,
        pole_y: Pole,
        source: ComplexField,
    ) -> ComplexField:
        horizontal = pac_triton_bidirectional_product_scan(pole_x, source)

        def vertical(active: ComplexField) -> tuple[Tensor, Tensor, Tensor, Tensor]:
            transposed = active[0].transpose(1, 2), active[1].transpose(1, 2)
            result = pac_triton_bidirectional_product_scan(pole_y, transposed)
            return cast(
                "tuple[Tensor, Tensor, Tensor, Tensor]",
                tuple(value.transpose(1, 2) for value in result),
            )

        positive_x = vertical((horizontal[0], horizontal[1]))
        negative_x = vertical((horizontal[2], horizontal[3]))
        directional = (
            (positive_x[0], positive_x[1]),
            (negative_x[0], negative_x[1]),
            (positive_x[2], positive_x[3]),
            (negative_x[2], negative_x[3]),
        )
        _, height, width, _ = source[0].shape
        coefficients = cast(
            "tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]",
            tuple(value.detach() for value in (*pole_x, *pole_y)),
        )
        variance_x, variance_y = static_variance_tables(
            *coefficients,
            width,
            height,
        )
        real_paths = []
        imag_paths = []
        for (real, imag), (x_sign, y_sign) in zip(directional, _DIRECTIONS, strict=True):
            variance = variance_y[y_sign, :, None, :] * variance_x[x_sign, None, :, :]
            inverse = torch.rsqrt(variance.clamp_min(1.0e-8)).to(real.dtype)
            real_paths.append(real * inverse)
            imag_paths.append(imag * inverse)
        return torch.stack(real_paths, dim=-2), torch.stack(imag_paths, dim=-2)

    def _scan(self, real: Tensor, imag: Tensor) -> ComplexField:
        shape = cast("tuple[int, int, int, int]", tuple(real.shape))
        pole_x, pole_y = self._pole_coefficients(shape)
        expanded = self._expanded_excitation(real), self._expanded_excitation(imag)
        if self.terminal:
            state_real, state_imag = self._terminal_full_states(pole_x, pole_y, expanded)
        else:
            state_real, state_imag, _ = cast(
                "tuple[Tensor, Tensor, Tensor]",
                run_product_scan_pipeline(
                    pole_x,
                    pole_y,
                    expanded,
                    epilogue="coarse",
                    gain_normalization="pointwise",
                    memory_policy=self.scan_memory_policy,
                ),
            )
        semantic_shape = (*state_real.shape[:-1], self.poles, self.content_modes)
        return state_real.reshape(semantic_shape), state_imag.reshape(semantic_shape)

    def _process_memory(
        self,
        real: Tensor,
        imag: Tensor,
        *,
        update_diagnostics: bool = True,
    ) -> ComplexField:
        def phase_gated(
            block: PhaseGatedComplexFFN,
            block_real: Tensor,
            block_imag: Tensor,
        ) -> ComplexField:
            return block._forward(
                block_real,
                block_imag,
                update_diagnostics=update_diagnostics,
            )

        # Materialize each PG's mode axis as canonical rows, then restore the
        # semantic direction-content-pole layout consumed by the readouts.
        content_real, content_imag = phase_gated(self.content_pg, real, imag)
        pole_real, pole_imag = phase_gated(
            self.pole_pg,
            content_real.transpose(-1, -2).contiguous(),
            content_imag.transpose(-1, -2).contiguous(),
        )
        rank = real.ndim
        leading = tuple(range(rank - 3))
        path_order = (*leading, rank - 2, rank - 1, rank - 3)
        semantic_order = (*leading, rank - 1, rank - 3, rank - 2)
        path_real, path_imag = phase_gated(
            self.path_pg,
            pole_real.permute(path_order).contiguous(),
            pole_imag.permute(path_order).contiguous(),
        )
        return (
            path_real.permute(semantic_order).contiguous(),
            path_imag.permute(semantic_order).contiguous(),
        )

    def _descriptor(self, real: Tensor, imag: Tensor) -> Tensor:
        weight_real = self.descriptor_readout_real.to(dtype=real.dtype)
        weight_imag = self.descriptor_readout_imag.to(dtype=real.dtype)
        contracted_real = torch.einsum(
            "bhwdkr,kqr->bhwdkq",
            real,
            weight_real,
        ) - torch.einsum("bhwdkr,kqr->bhwdkq", imag, weight_imag)
        contracted_imag = torch.einsum(
            "bhwdkr,kqr->bhwdkq",
            real,
            weight_imag,
        ) + torch.einsum("bhwdkr,kqr->bhwdkq", imag, weight_real)
        energy = contracted_real.float().square().add(contracted_imag.float().square())
        # B,D,K,3 -> B,384, direction-major as in the matched head contract.
        return torch.log1p(energy.mean((1, 2))).flatten(1)

    def _memory_readout(self, real: Tensor, imag: Tensor) -> ComplexField:
        if (
            self.direction_readout is None
            or self.pole_readout_real is None
            or self.pole_readout_imag is None
        ):
            message = "terminal wide pole stage has no next-excitation readout"
            raise RuntimeError(message)
        weight_real = self.pole_readout_real.to(dtype=real.dtype)
        weight_imag = self.pole_readout_imag.to(dtype=real.dtype)
        # The pole and direction maps are bias-free complex-linear maps over
        # independent axes.  Reduce R before D so the direction projection
        # consumes [B,H,W,K,D], not the full [B,H,W,K,R,D] memory.
        contracted_real = torch.einsum(
            "bhwdkr,kr->bhwdk", real, weight_real
        ) - torch.einsum("bhwdkr,kr->bhwdk", imag, weight_imag)
        contracted_imag = torch.einsum(
            "bhwdkr,kr->bhwdk", real, weight_imag
        ) + torch.einsum("bhwdkr,kr->bhwdk", imag, weight_real)
        direction_real, direction_imag = self.direction_readout(
            contracted_real.transpose(-1, -2),
            contracted_imag.transpose(-1, -2),
        )
        return direction_real.squeeze(-1), direction_imag.squeeze(-1)

    def _carry(self, real: Tensor, imag: Tensor) -> ComplexField:
        if self.carry_weight is None:
            message = "terminal wide pole stage has no S2D carry"
            raise RuntimeError(message)
        carry_real, carry_imag = complex_carry_coordinates(real, imag, "s2d")
        shape = (*carry_real.shape[:-1], 4, self.content_modes)
        weight = self.carry_weight.transpose(0, 1).to(dtype=real.dtype)
        return (
            (carry_real.reshape(shape) * weight).sum(-2),
            (carry_imag.reshape(shape) * weight).sum(-2),
        )

    def _ema(self, target: Tensor, value: Tensor) -> None:
        decay = torch.where(
            self.diagnostic_updates > 0,
            value.new_tensor(0.95),
            value.new_zeros(()),
        )
        target.mul_(decay).add_(value * (1.0 - decay))

    @torch.no_grad()
    def _update_response_diagnostics(self, real: Tensor, imag: Tensor) -> None:
        height = min(2, real.shape[1])
        width = min(2, real.shape[2])
        contents = min(4, self.content_modes)
        sampled_real = real[:1, :height, :width, :, :, :contents].detach().float()
        sampled_imag = imag[:1, :height, :width, :, :, :contents].detach().float()
        matrix_real = sampled_real.movedim(-2, -1).reshape(-1, self.poles)
        matrix_imag = sampled_imag.movedim(-2, -1).reshape(-1, self.poles)
        gram_real = matrix_real.mT @ matrix_real + matrix_imag.mT @ matrix_imag
        gram_imag = matrix_real.mT @ matrix_imag - matrix_imag.mT @ matrix_real
        diagonal = gram_real.diagonal().clamp_min(1.0e-12)
        denominator = torch.sqrt(diagonal[:, None] * diagonal[None, :])
        correlation = torch.sqrt(gram_real.square() + gram_imag.square()) / denominator
        active_correlation = correlation[self._off_diagonal]
        frobenius = gram_real.square().add(gram_imag.square()).sum().clamp_min(1.0e-12)
        effective_rank = diagonal.sum().square() / frobenius
        response_rms = torch.sqrt(diagonal / max(1, matrix_real.shape[0]))
        self._ema(self.response_rms_mean, response_rms.mean())
        self._ema(self.response_rms_std, response_rms.std(unbiased=False))
        self._ema(self.response_correlation_mean, active_correlation.mean())
        self._ema(self.response_correlation_max, active_correlation.max())
        self._ema(self.pole_effective_rank, effective_rank)
        raw_sample = sampled_real.square().add(sampled_imag.square()).mean().sqrt()
        self._ema(self.raw_memory_rms, raw_sample)

    @torch.no_grad()
    def _update_transition_diagnostics(
        self,
        memory: ComplexField,
        readout: ComplexField | None,
        carry: ComplexField | None,
    ) -> None:
        real, imag = memory
        height = min(2, real.shape[1])
        width = min(2, real.shape[2])
        sample_real = real[:1, :height, :width].detach().float()
        sample_imag = imag[:1, :height, :width].detach().float()
        memory_rms = sample_real.square().add(sample_imag.square()).mean().sqrt()
        self._ema(self.processed_memory_rms, memory_rms)
        if readout is None or carry is None:
            return
        readout_rms = (
            (readout[0].detach().float().square().add(readout[1].detach().float().square()))
            .mean()
            .sqrt()
        )
        carry_rms = (
            (carry[0].detach().float().square().add(carry[1].detach().float().square()))
            .mean()
            .sqrt()
        )
        self._ema(self.memory_readout_rms, readout_rms)
        self._ema(self.carry_rms, carry_rms)
        self._ema(
            self.memory_carry_rms_ratio,
            readout_rms / carry_rms.clamp_min(1.0e-12),
        )

    @staticmethod
    def _effective_count(weight_real: Tensor, weight_imag: Tensor) -> Tensor:
        energy = weight_real.float().square().add(weight_imag.float().square())
        probability = energy / energy.sum(-1, keepdim=True).clamp_min(1.0e-12)
        return probability.square().sum(-1).reciprocal()

    @staticmethod
    def _descriptor_effective_rank(weight_real: Tensor, weight_imag: Tensor) -> Tensor:
        real = weight_real.float()
        imag = weight_imag.float()
        gram_real = real @ real.mT + imag @ imag.mT
        gram_imag = real @ imag.mT - imag @ real.mT
        trace = gram_real.diagonal(dim1=-2, dim2=-1).sum(-1)
        frobenius = gram_real.square().add(gram_imag.square()).sum((-2, -1))
        return trace.square() / frobenius.clamp_min(1.0e-12)

    def diagnostic_metrics(self) -> dict[str, float]:
        damping_x = self.damping_min + (self.damping_max - self.damping_min) * torch.sigmoid(
            self.damping_logits_x.detach().float()
        )
        damping_y = self.damping_min + (self.damping_max - self.damping_min) * torch.sigmoid(
            self.damping_logits_y.detach().float()
        )
        descriptor_effective = self._effective_count(
            self.descriptor_readout_real.detach(),
            self.descriptor_readout_imag.detach(),
        )
        descriptor_rank = self._descriptor_effective_rank(
            self.descriptor_readout_real.detach(),
            self.descriptor_readout_imag.detach(),
        )
        metrics = {
            "poles": float(self.poles),
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
            "descriptor_readout_effective_poles": float(descriptor_effective.mean()),
            "descriptor_readout_effective_rank": float(descriptor_rank.mean()),
        }
        if self.pole_readout_real is not None and self.pole_readout_imag is not None:
            readout_effective = self._effective_count(
                self.pole_readout_real.detach(),
                self.pole_readout_imag.detach(),
            )
            readout_energy = (
                self.pole_readout_real.detach()
                .float()
                .square()
                .add(self.pole_readout_imag.detach().float().square())
            )
            readout_probability = readout_energy / readout_energy.sum(-1, keepdim=True).clamp_min(
                1.0e-12
            )
            metrics.update(
                {
                    "memory_readout_rms": float(self.memory_readout_rms),
                    "carry_rms": float(self.carry_rms),
                    "memory_carry_rms_ratio": float(self.memory_carry_rms_ratio),
                    "pole_readout_effective_poles": float(readout_effective.mean()),
                    "pole_readout_concentration": float(readout_probability.max(-1).values.mean()),
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

        readout = self._memory_readout(*memory)
        carry = self._carry(real, imag)
        if self.post_pg is None:
            message = "non-terminal wide pole stage is missing Post PG"
            raise RuntimeError(message)
        post_real = readout[0] + carry[0]
        post_imag = readout[1] + carry[1]
        # Strict pole contraction can promote a BF16 direction readout when
        # the scan residual is FP32.  Restore the requested BF16 activation
        # precision before entering the packed Post PG kernel.
        if (
            post_real.is_cuda
            and torch.is_autocast_enabled("cuda")
            and torch.get_autocast_dtype("cuda") is torch.bfloat16
        ):
            post_real = post_real.to(torch.bfloat16)
            post_imag = post_imag.to(torch.bfloat16)
        post_real = post_real.contiguous()
        post_imag = post_imag.contiguous()
        if update_diagnostics:
            output = self.post_pg(post_real, post_imag)
        else:
            output = self.post_pg._forward_without_diagnostics(
                post_real,
                post_imag,
            )
        if update_diagnostics:
            self._update_transition_diagnostics(memory, readout, carry)
            self.diagnostic_updates.add_(1)
        return output, descriptor

    def _checkpointed_nonterminal(
        self,
        real: Tensor,
        imag: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        # Keep the wide scan memory inside the checkpoint so its full output
        # is not retained until backward.  Diagnostics belong to the
        # grad-enabled replay exactly once per logical step.
        state, descriptor = self._forward_impl(
            real,
            imag,
            update_diagnostics=torch.is_grad_enabled(),
        )
        if state is None:
            message = "non-terminal checkpoint produced no next excitation"
            raise RuntimeError(message)
        return state[0], state[1], descriptor

    def _checkpointed_terminal(
        self,
        real: Tensor,
        imag: Tensor,
    ) -> Tensor:
        state, descriptor = self._forward_impl(
            real,
            imag,
            update_diagnostics=torch.is_grad_enabled(),
        )
        if state is not None:
            message = "terminal checkpoint unexpectedly produced an excitation"
            raise RuntimeError(message)
        return descriptor

    def forward(self, real: Tensor, imag: Tensor) -> tuple[ComplexField | None, Tensor]:
        if real.shape != imag.shape or real.ndim != 4 or real.shape[-1] != self.content_modes:
            message = "wide pole stage inputs must be matching NHW-content tensors"
            raise ValueError(message)
        if not self.training or not torch.is_grad_enabled():
            return self._forward_impl(real, imag, update_diagnostics=self.training)
        if self.terminal:
            descriptor = cast(
                "Tensor",
                checkpoint(
                    self._checkpointed_terminal,
                    real,
                    imag,
                    use_reentrant=True,
                ),
            )
            return None, descriptor
        output_real, output_imag, descriptor = cast(
            "tuple[Tensor, Tensor, Tensor]",
            checkpoint(
                self._checkpointed_nonterminal,
                real,
                imag,
                use_reentrant=True,
            ),
        )
        return (output_real, output_imag), descriptor


__all__ = ["FactorizedWidePoleMemoryStage"]
