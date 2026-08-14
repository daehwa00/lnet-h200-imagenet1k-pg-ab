from __future__ import annotations

from typing import TYPE_CHECKING, Final, Literal

import torch
from torch import Tensor, nn
from torch.nn import functional

from .pac_headroom_efficient_models import (
    EDGE_FRAME_VARIANT,
    EdgeFramePAC,
    _apply_raw_mask,  # pyright: ignore[reportPrivateUsage]
)
from .pac_laplace_native_input import (
    RawRepeatedTwoForwardPAC,
    _RawForcingStem,  # pyright: ignore[reportPrivateUsage]
)
from .pac_recurrence import recurrence_real2d_directional
from .pac_tight_frame_models import (
    _BlockVariant,  # pyright: ignore[reportPrivateUsage]
    _directional_depthwise_conv,  # pyright: ignore[reportPrivateUsage]
    _InvariantMomentHead,  # pyright: ignore[reportPrivateUsage]
    _masked_modal_moments,  # pyright: ignore[reportPrivateUsage]
    _modal_mask,  # pyright: ignore[reportPrivateUsage]
    _modal_moments,  # pyright: ignore[reportPrivateUsage]
    _modal_time_inputs,  # pyright: ignore[reportPrivateUsage]
    _MomentVariant,  # pyright: ignore[reportPrivateUsage]
    _TightFrameBlock,  # pyright: ignore[reportPrivateUsage]
)

if TYPE_CHECKING:
    from .pac_headroom_models import HeadroomObjective
    from .pac_types import PACExperimentConfig


RawEfficiencyVariant = Literal[
    "terminal_analysis",
    "terminal_analysis_no_rmsnorm",
    "efp16_rmsnorm_control",
    "efp16_no_rmsnorm",
    "terminal_second_moments_only",
    "one_modal_refinement",
    "preprocessed_single_modal",
    "mamba_exact_single_block",
    "mamba_exact_single_block_no_rmsnorm",
    "mamba_exact_terminal",
    "mamba_exact_two_block",
    "mamba_exact_two_block_no_rmsnorm",
    "mamba_exact_scan_only_m8_first",
    "mamba_exact_scan_only_m8_second",
    "mamba_exact_scan_only_m8_both",
    "mamba_exact_local_terminal_m8_both",
    "mamba_exact_projected_local_terminal_m8_both",
    "mamba_first_full_terminal_control",
    "mamba_first_no_gate_terminal",
    "mamba_first_diagonal_output_terminal",
    "mamba_first_no_gate_diagonal_terminal",
    "mamba_first_no_gate_diagonal_no_main_projection_terminal",
    "compact_raw_branches_all_gate_terminal",
    "compact_raw_branches_modal_only_gate_terminal",
    "fused_scalar_stem",
    "fixed_layer_scale",
    "shared_analysis_synthesis",
    "first_moments_only",
    "second_moments_only",
    "terminal_fused_scalar_stem",
]
RAW_EFFICIENCY_VARIANTS: Final[tuple[RawEfficiencyVariant, ...]] = (
    "terminal_analysis",
    "terminal_analysis_no_rmsnorm",
    "efp16_rmsnorm_control",
    "efp16_no_rmsnorm",
    "terminal_second_moments_only",
    "one_modal_refinement",
    "preprocessed_single_modal",
    "mamba_exact_single_block",
    "mamba_exact_single_block_no_rmsnorm",
    "mamba_exact_terminal",
    "mamba_exact_two_block",
    "mamba_exact_two_block_no_rmsnorm",
    "mamba_exact_scan_only_m8_first",
    "mamba_exact_scan_only_m8_second",
    "mamba_exact_scan_only_m8_both",
    "mamba_exact_local_terminal_m8_both",
    "mamba_exact_projected_local_terminal_m8_both",
    "mamba_first_full_terminal_control",
    "mamba_first_no_gate_terminal",
    "mamba_first_diagonal_output_terminal",
    "mamba_first_no_gate_diagonal_terminal",
    "mamba_first_no_gate_diagonal_no_main_projection_terminal",
    "compact_raw_branches_all_gate_terminal",
    "compact_raw_branches_modal_only_gate_terminal",
    "fused_scalar_stem",
    "fixed_layer_scale",
    "shared_analysis_synthesis",
    "first_moments_only",
    "second_moments_only",
    "terminal_fused_scalar_stem",
)


class FusedScalarRawStem(nn.Module):
    """Fold a scalar projection and depthwise local map into one convolution."""

    def __init__(self, source: _RawForcingStem) -> None:
        super().__init__()
        if source.projection.in_features != 1:
            message = "the fused raw stem requires scalar inputs"
            raise ValueError(message)
        local = source.local
        padding = local.padding if isinstance(local.padding, str) else local.padding[0]
        self.convolution = nn.Conv1d(
            1,
            source.projection.out_features,
            kernel_size=local.kernel_size[0],
            stride=local.stride[0],
            padding=padding,
            dilation=local.dilation[0],
            bias=local.bias is not None,
            device=local.weight.device,
            dtype=local.weight.dtype,
        )
        with torch.no_grad():
            projected_scale = source.projection.weight[:, :1].unsqueeze(-1)
            self.convolution.weight.copy_(source.local.weight * projected_scale)
            if self.convolution.bias is not None and source.local.bias is not None:
                self.convolution.bias.copy_(source.local.bias)

    def forward(self, inputs: Tensor) -> Tensor:
        local = self.convolution(inputs.transpose(1, 2)).transpose(1, 2)
        return functional.silu(local)

    @torch.no_grad()
    def project_weight_(self) -> None:
        """Keep the inherited EdgeFramePAC constraint hook a no-op."""


def _replace_parameter_with_buffer(
    module: nn.Module,
    name: str,
    value: Tensor,
) -> None:
    parameter = getattr(module, name, None)
    if not isinstance(parameter, nn.Parameter):
        message = f"{name} is not a registered parameter"
        raise TypeError(message)
    delattr(module, name)
    module.register_buffer(name, value.detach().clone())


def _fold_and_fix_layer_scale(block: _TightFrameBlock) -> None:
    synthesis = block.independent_synthesis_frame
    layer_scale = block.layer_scale
    direct_scale = block.direct_scale
    if synthesis is None or layer_scale is None or direct_scale is None:
        message = "fixed layer scale requires the canonical untied residual block"
        raise RuntimeError(message)
    with torch.no_grad():
        synthesis.mul_(layer_scale.view(-1, 1))
        direct_scale.mul_(layer_scale)
    _replace_parameter_with_buffer(block, "layer_scale", torch.ones_like(layer_scale))


def _make_terminal_analysis(block: _TightFrameBlock) -> None:
    if block.split_residual_scales:
        message = "terminal analysis requires the coupled residual block"
        raise RuntimeError(message)
    direct_scale = block.direct_scale
    layer_scale = block.layer_scale
    if direct_scale is None or layer_scale is None:
        message = "terminal analysis residual parameters are unavailable"
        raise RuntimeError(message)

    # Moments do not depend on synthesis or residual parameters. The terminal
    # forward path asks the block to return before synthesis, so these tensors can
    # be removed without changing the retained second recurrence or its moments.
    block.register_parameter("independent_synthesis_frame", None)
    _replace_parameter_with_buffer(block, "direct_scale", torch.zeros_like(direct_scale))
    _replace_parameter_with_buffer(block, "layer_scale", torch.zeros_like(layer_scale))


def _make_external_modal_core(block: _TightFrameBlock) -> None:
    """Expose recurrence states while removing the inherited residual controls."""
    if block.split_residual_scales:
        message = "the Mamba scaffold requires the coupled residual block"
        raise RuntimeError(message)
    direct_scale = block.direct_scale
    layer_scale = block.layer_scale
    if direct_scale is None or layer_scale is None:
        message = "modal residual parameters are unavailable"
        raise RuntimeError(message)
    block.use_input_norm = False
    block.norm = nn.Identity()
    block.local = None
    _replace_parameter_with_buffer(block, "direct_scale", torch.zeros_like(direct_scale))
    _replace_parameter_with_buffer(block, "layer_scale", torch.zeros_like(layer_scale))


def _stable_discrete_pole_real2d(
    damping: Tensor,
    frequency: Tensor,
    step: float | Tensor,
    *,
    threshold: float = 1.0e-6,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Evaluate the exact ZOH pole and gain without cancellation near zero."""
    active_step = (
        step.to(dtype=damping.dtype, device=damping.device) if isinstance(step, Tensor) else step
    )
    scaled_real = -damping * active_step
    phase = frequency * active_step
    cosine = torch.cos(phase)
    scaled_decay = torch.exp(scaled_real)
    decay_real = scaled_decay * cosine
    decay_imag = scaled_decay * torch.sin(phase)

    # Re[expm1(a + ib)] = expm1(a) cos(b) + cos(b) - 1.  Writing
    # cos(b) - 1 as -2 sin^2(b/2) avoids a second cancellation.
    shifted_real = torch.expm1(scaled_real) * cosine - 2.0 * torch.sin(0.5 * phase).square()
    shifted_imag = decay_imag
    pole_real = -damping
    pole_imag = frequency
    denominator = pole_real.square() + pole_imag.square()
    small = torch.sqrt(scaled_real.square() + phase.square()) < threshold
    safe_denominator = torch.where(small, torch.ones_like(denominator), denominator)
    gain_real = (shifted_real * pole_real + shifted_imag * pole_imag) / safe_denominator
    gain_imag = (shifted_imag * pole_real - shifted_real * pole_imag) / safe_denominator
    step_tensor = torch.ones_like(gain_real) * active_step
    return (
        decay_real,
        decay_imag,
        torch.where(small, step_tensor, gain_real),
        torch.where(small, torch.zeros_like(gain_imag), gain_imag),
    )


class RawProjectionEmbedding(nn.Module):
    """Semi-orthogonal raw embedding without a separate local stem."""

    def __init__(self, input_dim: int, model_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, model_dim, bias=False)
        nn.init.orthogonal_(self.projection.weight)
        self.project_weight_()

    def forward(self, inputs: Tensor) -> Tensor:
        return self.projection(inputs)

    @torch.no_grad()
    def project_weight_(self) -> None:
        weight = self.projection.weight
        active = weight.float() if weight.shape[0] >= weight.shape[1] else weight.float().T
        frame, upper = torch.linalg.qr(active, mode="reduced")
        diagonal = torch.diagonal(upper)
        signs = torch.where(
            diagonal >= 0.0,
            torch.ones_like(diagonal),
            -torch.ones_like(diagonal),
        )
        projected = frame * signs.unsqueeze(0)
        if weight.shape[0] < weight.shape[1]:
            projected = projected.T
        weight.copy_(projected.to(dtype=weight.dtype))


class _MambaExactPoleBlock(nn.Module):
    """Pre-norm gated residual scaffold around the existing exact-pole core."""

    def __init__(
        self,
        core: _TightFrameBlock,
        model_dim: int,
        *,
        kernel_size: int = 5,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        _make_external_modal_core(core)
        if core.independent_synthesis_frame is None:
            message = "the gated exact-pole block requires learned synthesis"
            raise RuntimeError(message)
        if core.direction != "forward":
            message = "the single-direction Mamba scaffold requires a forward modal core"
            raise RuntimeError(message)
        if core.mode_gate_bias is not None:
            message = "the single-direction Mamba scaffold requires static ungated poles"
            raise RuntimeError(message)
        self.core = core
        self.norm = nn.RMSNorm(model_dim)
        self.input_projection = nn.Linear(model_dim, 2 * model_dim, bias=False)
        self.local = nn.Conv1d(
            model_dim,
            model_dim,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=model_dim,
        )
        self.skip_scale = nn.Parameter(torch.ones(model_dim))
        self.output_projection = nn.Linear(model_dim, model_dim, bias=False)
        nn.init.orthogonal_(self.input_projection.weight)
        nn.init.orthogonal_(self.output_projection.weight)
        with torch.no_grad():
            self.output_projection.weight.mul_(1.0e-2)

    def _exact_pole_scan(
        self,
        inputs: Tensor,
        *,
        time_delta: Tensor | None,
        observation_mask: Tensor | None,
        valid_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Return modal coordinates and moments from one recurrence evaluation."""
        core = self.core
        excitation = torch.matmul(inputs, core.frame_matrix())
        excitation_real, excitation_imag = excitation.chunk(2, dim=-1)
        active_delta, active_observation = _modal_time_inputs(
            inputs,
            time_delta,
            observation_mask,
            direction=core.direction,
        )
        active_valid = _modal_mask(inputs, valid_mask, name="valid_mask")

        damping = core.damping_min + (core.damping_max - core.damping_min) * torch.sigmoid(
            core.raw_decay.view(1, 1, -1)
        )
        damping = damping.expand_as(excitation_real)
        frequency = torch.pi * torch.tanh(core.raw_frequency).view(1, 1, -1)
        decay_real, decay_imag, gain_real, gain_imag = _stable_discrete_pole_real2d(
            damping,
            frequency,
            active_delta,
        )
        input_real = gain_real * excitation_real - gain_imag * excitation_imag
        input_imag = gain_real * excitation_imag + gain_imag * excitation_real
        if active_observation is not None:
            input_real = active_observation * input_real
            input_imag = active_observation * input_imag

        states_real, states_imag = recurrence_real2d_directional(
            decay_real,
            decay_imag,
            input_real,
            input_imag,
            core.recurrence_backend,
            core.direction,
        )
        moment_variant = _MomentVariant(
            "forward" if core.align_moments else core.direction,
            core.log_energy,
            core.normalize_autocorrelation,
            core.moment_lags,
        )
        moments = (
            _modal_moments(states_real, states_imag, moment_variant)
            if active_valid is None
            else _masked_modal_moments(
                states_real,
                states_imag,
                active_valid,
                moment_variant,
            )
        )
        return torch.cat((states_real, states_imag), dim=-1), moments

    def forward(
        self,
        inputs: Tensor,
        *,
        time_delta: Tensor | None = None,
        observation_mask: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        projected, gate_pre_activation = self.input_projection(self.norm(inputs)).chunk(2, dim=-1)
        channels_first = projected.transpose(1, 2)
        left_padding = self.local.dilation[0] * (self.local.kernel_size[0] - 1)
        local = functional.silu(
            self.local(functional.pad(channels_first, (left_padding, 0))).transpose(1, 2)
        )
        if valid_mask is not None:
            active = valid_mask if valid_mask.ndim == 3 else valid_mask.unsqueeze(-1)
            local = local * active.to(device=local.device, dtype=local.dtype)
        modal_coordinates, moments = self._exact_pole_scan(
            local,
            time_delta=time_delta,
            observation_mask=observation_mask,
            valid_mask=valid_mask,
        )
        synthesis = self.core.synthesis_frame_matrix()
        modal = torch.matmul(modal_coordinates, synthesis.transpose(0, 1))
        if self.core.synthesis_scale != 1.0:
            modal = self.core.synthesis_scale * modal
        feedthrough = modal + self.skip_scale.view(1, 1, -1) * local
        gate = functional.silu(gate_pre_activation)
        return inputs + self.output_projection(feedthrough * gate), moments

    def retract_frame(self) -> None:
        self.core.retract_frame()

    def finalize_frame(self) -> None:
        self.core.finalize_frame()


class _MambaExactPoleTerminalAnalysis(nn.Module):
    """Apply the Mamba main-branch lift but retain only terminal modal moments."""

    def __init__(self, core: _TightFrameBlock, model_dim: int) -> None:
        super().__init__()
        _make_terminal_analysis(core)
        core.use_input_norm = False
        core.norm = nn.Identity()
        core.local = None
        self.core = core
        self.norm = nn.RMSNorm(model_dim)
        self.input_projection = nn.Linear(model_dim, model_dim, bias=False)
        self.local = nn.Conv1d(
            model_dim,
            model_dim,
            kernel_size=5,
            groups=model_dim,
        )
        nn.init.orthogonal_(self.input_projection.weight)

    def forward(
        self,
        inputs: Tensor,
        *,
        time_delta: Tensor | None = None,
        observation_mask: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        projected = self.input_projection(self.norm(inputs))
        local = functional.silu(
            _directional_depthwise_conv(projected, self.local, self.core.direction)
        )
        if valid_mask is not None:
            active = valid_mask if valid_mask.ndim == 3 else valid_mask.unsqueeze(-1)
            local = local * active.to(device=local.device, dtype=local.dtype)
        moments = self.core(
            local,
            time_delta=time_delta,
            observation_mask=observation_mask,
            valid_mask=valid_mask,
            return_moments_only=True,
        )
        return local, moments

    def retract_frame(self) -> None:
        self.core.retract_frame()

    def finalize_frame(self) -> None:
        self.core.finalize_frame()


def _select_single_moment_copy(
    model: RawRepeatedTwoForwardPAC,
    source: Literal["first", "second"],
) -> None:
    head = model.head
    if not isinstance(head, _InvariantMomentHead):
        message = "single-moment candidates require the invariant moment head"
        raise TypeError(message)
    classifier = head.classifier
    moment_dim = model.modes * (1 + 2 * len(model.forward_block.moment_lags))
    pooled_dim = model.model_dim
    expected_features = pooled_dim + 2 * moment_dim
    if classifier.in_features != expected_features:
        message = "unexpected two-moment classifier width"
        raise RuntimeError(message)

    replacement = nn.Linear(
        pooled_dim + moment_dim,
        classifier.out_features,
        bias=classifier.bias is not None,
        device=classifier.weight.device,
        dtype=classifier.weight.dtype,
    )
    moment_start = pooled_dim if source == "first" else pooled_dim + moment_dim
    with torch.no_grad():
        replacement.weight[:, :pooled_dim].copy_(classifier.weight[:, :pooled_dim])
        replacement.weight[:, pooled_dim:].copy_(
            classifier.weight[:, moment_start : moment_start + moment_dim]
        )
        if replacement.bias is not None and classifier.bias is not None:
            replacement.bias.copy_(classifier.bias)
    head.classifier = replacement
    head.use_modal_moments = True
    head.use_backward_moments = False


class TerminalAnalysisRawRepeatedPAC(RawRepeatedTwoForwardPAC):
    """Use the second recurrence for moments but not for the terminal stream."""

    synthesis_compute_elided: Final[bool] = True

    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        *,
        objective: HeadroomObjective = "classification",
    ) -> None:
        super().__init__(config, output_dim, objective=objective)
        _make_terminal_analysis(self.backward_block)

    def forward(
        self,
        inputs: Tensor,
        *,
        time_delta: Tensor | None = None,
        observation_mask: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        stem_inputs = _apply_raw_mask(inputs, observation_mask, valid_mask)
        first_local = self._mask_features(self.stem(stem_inputs), valid_mask)
        first_stream, first_moments = self.forward_block(
            first_local,
            time_delta=time_delta,
            observation_mask=observation_mask,
            valid_mask=valid_mask,
        )
        second_projected = self.second_projection(first_stream)
        second_local = functional.silu(
            self.second_local(second_projected.transpose(1, 2)).transpose(1, 2)
        )
        encoded = self._mask_features(second_local, valid_mask)
        second_moments = self.backward_block(
            encoded,
            time_delta=time_delta,
            observation_mask=observation_mask,
            valid_mask=valid_mask,
            return_moments_only=True,
        )
        return self._readout(encoded, first_moments, second_moments, valid_mask)


class TerminalAnalysisNoRMSNormRawRepeatedPAC(TerminalAnalysisRawRepeatedPAC):
    """Remove both analysis norms and the terminal readout norm."""

    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        *,
        objective: HeadroomObjective = "classification",
    ) -> None:
        super().__init__(config, output_dim, objective=objective)
        for block in (self.forward_block, self.backward_block):
            block.use_input_norm = False
            block.norm = nn.Identity()  # pyright: ignore[reportAttributeAccessIssue]
        self.final_norm = None  # pyright: ignore[reportIncompatibleUnannotatedOverride]

    def _readout(
        self,
        inputs: Tensor,
        forward_moments: Tensor,
        backward_moments: Tensor,
        valid_mask: Tensor | None,
    ) -> Tensor:
        if valid_mask is None:
            pooled = inputs.mean(dim=1)
        else:
            weight = valid_mask.to(device=inputs.device, dtype=inputs.dtype)
            if weight.ndim == 2:
                weight = weight.unsqueeze(-1)
            pooled = (inputs * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)
        return self.head(pooled, forward_moments, backward_moments)


class EFP16RMSNormControlPAC(EdgeFramePAC):
    """Canonical EFP16 control under the raw-efficiency screen recipe."""

    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        *,
        objective: HeadroomObjective = "classification",
    ) -> None:
        super().__init__(
            config,
            output_dim,
            modes=16,
            semi_orthogonal=True,
            objective=objective,
        )


class EFP16NoRMSNormPAC(EFP16RMSNormControlPAC):
    """Canonical EFP16 with only its three RMS normalizations removed."""

    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        *,
        objective: HeadroomObjective = "classification",
    ) -> None:
        super().__init__(config, output_dim, objective=objective)
        for block in (self.forward_block, self.backward_block):
            block.use_input_norm = False
            block.norm = nn.Identity()  # pyright: ignore[reportAttributeAccessIssue]
        self.final_norm = None  # pyright: ignore[reportIncompatibleUnannotatedOverride]

    def _readout(
        self,
        inputs: Tensor,
        forward_moments: Tensor,
        backward_moments: Tensor,
        valid_mask: Tensor | None,
    ) -> Tensor:
        if valid_mask is None:
            pooled = inputs.mean(dim=1)
        else:
            weight = valid_mask.to(device=inputs.device, dtype=inputs.dtype)
            if weight.ndim == 2:
                weight = weight.unsqueeze(-1)
            pooled = (inputs * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)
        return self.head(pooled, forward_moments, backward_moments)


class TerminalSecondMomentReadoutRawRepeatedPAC(TerminalAnalysisRawRepeatedPAC):
    """Keep only the terminal recurrence moments alongside the real stream."""

    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        *,
        objective: HeadroomObjective = "classification",
    ) -> None:
        super().__init__(config, output_dim, objective=objective)
        _select_single_moment_copy(self, "second")

    def _readout(
        self,
        inputs: Tensor,
        forward_moments: Tensor,  # noqa: ARG002
        backward_moments: Tensor,
        valid_mask: Tensor | None,
    ) -> Tensor:
        return super()._readout(inputs, backward_moments, backward_moments, valid_mask)


class OneModalRefinementRawRepeatedPAC(RawRepeatedTwoForwardPAC):
    """Keep block-one modal memory and use project/local as the final refinement."""

    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        *,
        objective: HeadroomObjective = "classification",
    ) -> None:
        super().__init__(config, output_dim, objective=objective)
        _select_single_moment_copy(self, "first")
        self.backward_block = nn.Identity()  # pyright: ignore[reportIncompatibleUnannotatedOverride]

    def forward(
        self,
        inputs: Tensor,
        *,
        time_delta: Tensor | None = None,
        observation_mask: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        stem_inputs = _apply_raw_mask(inputs, observation_mask, valid_mask)
        first_local = self._mask_features(self.stem(stem_inputs), valid_mask)
        first_stream, first_moments = self.forward_block(
            first_local,
            time_delta=time_delta,
            observation_mask=observation_mask,
            valid_mask=valid_mask,
        )
        second_projected = self.second_projection(first_stream)
        second_local = functional.silu(
            self.second_local(second_projected.transpose(1, 2)).transpose(1, 2)
        )
        encoded = self._mask_features(second_local, valid_mask)
        return self._readout(encoded, first_moments, first_moments, valid_mask)

    def post_optimizer_step(self) -> None:
        self.forward_block.retract_frame()
        stem = self.stem
        if not isinstance(stem, _RawForcingStem):
            message = "one-modal refinement requires the canonical raw stem"
            raise TypeError(message)
        stem.project_weight_()

    def finalize_constraints(self) -> None:
        self.forward_block.finalize_frame()
        stem = self.stem
        if not isinstance(stem, _RawForcingStem):
            message = "one-modal refinement requires the canonical raw stem"
            raise TypeError(message)
        stem.project_weight_()


class PreprocessedSingleModalRawRepeatedPAC(RawRepeatedTwoForwardPAC):
    """Apply both local lifts before one terminal modal recurrence."""

    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        *,
        objective: HeadroomObjective = "classification",
    ) -> None:
        super().__init__(config, output_dim, objective=objective)
        _select_single_moment_copy(self, "first")
        self.pre_modal_norm = nn.RMSNorm(self.model_dim)
        self.pre_modal_scale = nn.Parameter(torch.ones(self.model_dim))
        self.backward_block = nn.Identity()  # pyright: ignore[reportIncompatibleUnannotatedOverride]

    def forward(
        self,
        inputs: Tensor,
        *,
        time_delta: Tensor | None = None,
        observation_mask: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        stem_inputs = _apply_raw_mask(inputs, observation_mask, valid_mask)
        first_local = self._mask_features(self.stem(stem_inputs), valid_mask)
        second_projected = self.second_projection(self.pre_modal_norm(first_local))
        second_local = functional.silu(
            self.second_local(second_projected.transpose(1, 2)).transpose(1, 2)
        )
        lifted = first_local + self.pre_modal_scale.view(1, 1, -1) * second_local
        lifted = self._mask_features(lifted, valid_mask)
        encoded, moments = self.forward_block(
            lifted,
            time_delta=time_delta,
            observation_mask=observation_mask,
            valid_mask=valid_mask,
        )
        encoded = self._mask_features(encoded, valid_mask)
        return self._readout(encoded, moments, moments, valid_mask)

    def post_optimizer_step(self) -> None:
        self.forward_block.retract_frame()
        stem = self.stem
        if not isinstance(stem, _RawForcingStem):
            message = "preprocessed single-modal model requires the canonical raw stem"
            raise TypeError(message)
        stem.project_weight_()

    def finalize_constraints(self) -> None:
        self.forward_block.finalize_frame()
        stem = self.stem
        if not isinstance(stem, _RawForcingStem):
            message = "preprocessed single-modal model requires the canonical raw stem"
            raise TypeError(message)
        stem.project_weight_()


class _MambaExactRawPACBase(RawRepeatedTwoForwardPAC):
    """Common raw embedding and constraint handling for gated exact-pole candidates."""

    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        *,
        objective: HeadroomObjective,
    ) -> None:
        super().__init__(config, output_dim, objective=objective)
        self.stem = RawProjectionEmbedding(config.raw_input_dim, self.model_dim)
        self.second_projection = nn.Identity()
        self.second_local = nn.Identity()  # pyright: ignore[reportIncompatibleVariableOverride]

    def post_optimizer_step(self) -> None:
        self.forward_block.retract_frame()
        self.backward_block.retract_frame()
        self.stem.project_weight_()

    def finalize_constraints(self) -> None:
        self.forward_block.finalize_frame()
        self.backward_block.finalize_frame()
        self.stem.project_weight_()


class MambaExactSingleBlockRawPAC(_MambaExactRawPACBase):
    """One forward-only gated exact-pole block with same-scan moments."""

    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        *,
        objective: HeadroomObjective = "classification",
    ) -> None:
        super().__init__(config, output_dim, objective=objective)
        _select_single_moment_copy(self, "first")
        self.forward_block = _MambaExactPoleBlock(
            self.forward_block,
            self.model_dim,
            kernel_size=4,
            dilation=1,
        )
        self.backward_block = nn.Identity()

    def forward(
        self,
        inputs: Tensor,
        *,
        time_delta: Tensor | None = None,
        observation_mask: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        stem_inputs = _apply_raw_mask(inputs, observation_mask, valid_mask)
        encoded = self._mask_features(self.stem(stem_inputs), valid_mask)
        encoded, moments = self.forward_block(
            encoded,
            time_delta=time_delta,
            observation_mask=observation_mask,
            valid_mask=valid_mask,
        )
        encoded = self._mask_features(encoded, valid_mask)
        readout_mask = (
            valid_mask.unsqueeze(-1)
            if valid_mask is not None and valid_mask.ndim == 2
            else valid_mask
        )
        return self._readout(encoded, moments, torch.zeros_like(moments), readout_mask)

    def post_optimizer_step(self) -> None:
        self.forward_block.retract_frame()
        self.stem.project_weight_()

    def finalize_constraints(self) -> None:
        self.forward_block.finalize_frame()
        self.stem.project_weight_()


class MambaExactSingleBlockNoRMSNormRawPAC(MambaExactSingleBlockRawPAC):
    """Remove both pre-scan and final-readout RMS normalization."""

    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        *,
        objective: HeadroomObjective = "classification",
    ) -> None:
        super().__init__(config, output_dim, objective=objective)
        self.forward_block.norm = nn.Identity()
        self.final_norm = None

    def _readout(
        self,
        inputs: Tensor,
        forward_moments: Tensor,
        backward_moments: Tensor,
        valid_mask: Tensor | None,
    ) -> Tensor:
        if valid_mask is None:
            pooled = inputs.mean(dim=1)
        else:
            weight = valid_mask.to(device=inputs.device, dtype=inputs.dtype)
            pooled = (inputs * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)
        return self.head(pooled, forward_moments, backward_moments)


class MambaExactTerminalRawPAC(_MambaExactRawPACBase):
    """One gated exact-pole update followed by a moments-only modal analysis."""

    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        *,
        objective: HeadroomObjective = "classification",
    ) -> None:
        super().__init__(config, output_dim, objective=objective)
        self.forward_block = _MambaExactPoleBlock(self.forward_block, self.model_dim)
        self.backward_block = _MambaExactPoleTerminalAnalysis(
            self.backward_block,
            self.model_dim,
        )

    def forward(
        self,
        inputs: Tensor,
        *,
        time_delta: Tensor | None = None,
        observation_mask: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        stem_inputs = _apply_raw_mask(inputs, observation_mask, valid_mask)
        encoded = self._mask_features(self.stem(stem_inputs), valid_mask)
        encoded, first_moments = self.forward_block(
            encoded,
            time_delta=time_delta,
            observation_mask=observation_mask,
            valid_mask=valid_mask,
        )
        encoded = self._mask_features(encoded, valid_mask)
        encoded, second_moments = self.backward_block(
            encoded,
            time_delta=time_delta,
            observation_mask=observation_mask,
            valid_mask=valid_mask,
        )
        encoded = self._mask_features(encoded, valid_mask)
        return self._readout(encoded, first_moments, second_moments, valid_mask)


class MambaExactTwoBlockRawPAC(_MambaExactRawPACBase):
    """Repeat the full Mamba-shaped exact-pole block twice."""

    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        *,
        objective: HeadroomObjective = "classification",
        block_kernel_size: int = 5,
        block_dilation: int = 1,
    ) -> None:
        super().__init__(config, output_dim, objective=objective)
        self.forward_block = _MambaExactPoleBlock(
            self.forward_block,
            self.model_dim,
            kernel_size=block_kernel_size,
            dilation=block_dilation,
        )
        self.backward_block = _MambaExactPoleBlock(
            self.backward_block,
            self.model_dim,
            kernel_size=block_kernel_size,
            dilation=block_dilation,
        )

    def forward(
        self,
        inputs: Tensor,
        *,
        time_delta: Tensor | None = None,
        observation_mask: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        stem_inputs = _apply_raw_mask(inputs, observation_mask, valid_mask)
        encoded = self._mask_features(self.stem(stem_inputs), valid_mask)
        encoded, first_moments = self.forward_block(
            encoded,
            time_delta=time_delta,
            observation_mask=observation_mask,
            valid_mask=valid_mask,
        )
        encoded = self._mask_features(encoded, valid_mask)
        encoded, second_moments = self.backward_block(
            encoded,
            time_delta=time_delta,
            observation_mask=observation_mask,
            valid_mask=valid_mask,
        )
        encoded = self._mask_features(encoded, valid_mask)
        return self._readout(encoded, first_moments, second_moments, valid_mask)


class MambaExactTwoBlockNoRMSNormRawPAC(MambaExactTwoBlockRawPAC):
    """Stack two K4/D1 gated exact-pole blocks without RMS normalization."""

    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        *,
        objective: HeadroomObjective = "classification",
    ) -> None:
        super().__init__(
            config,
            output_dim,
            objective=objective,
            block_kernel_size=4,
            block_dilation=1,
        )
        self.forward_block.norm = nn.Identity()
        self.backward_block.norm = nn.Identity()
        self.final_norm = None

    def _readout(
        self,
        inputs: Tensor,
        forward_moments: Tensor,
        backward_moments: Tensor,
        valid_mask: Tensor | None,
    ) -> Tensor:
        if valid_mask is None:
            pooled = inputs.mean(dim=1)
        else:
            weight = valid_mask.to(device=inputs.device, dtype=inputs.dtype)
            if weight.ndim == 2:
                weight = weight.unsqueeze(-1)
            pooled = (inputs * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)
        return self.head(pooled, forward_moments, backward_moments)


ScanOnlyMomentReadout = Literal["first", "second", "both"]


class _AsymmetricMomentHead(nn.Module):
    """Linear head for independently sized first and terminal moment vectors."""

    def __init__(
        self,
        pooled_dim: int,
        first_moment_dim: int,
        second_moment_dim: int,
        output_dim: int,
        readout: ScanOnlyMomentReadout,
    ) -> None:
        super().__init__()
        self.readout = readout
        selected_dim = {
            "first": first_moment_dim,
            "second": second_moment_dim,
            "both": first_moment_dim + second_moment_dim,
        }[readout]
        self.classifier = nn.Linear(pooled_dim + selected_dim, output_dim)

    def forward(
        self,
        pooled: Tensor,
        first_moments: Tensor,
        second_moments: Tensor,
    ) -> Tensor:
        match self.readout:
            case "first":
                features = torch.cat((pooled, first_moments), dim=-1)
            case "second":
                features = torch.cat((pooled, second_moments), dim=-1)
            case "both":
                features = torch.cat((pooled, first_moments, second_moments), dim=-1)
        return self.classifier(features)


class _ScanOnlyTerminalAnalyzer(nn.Module):
    """A learned modal analysis and exact scan with no feature-stream update."""

    def __init__(self, model_dim: int, modes: int) -> None:
        super().__init__()
        core = _TightFrameBlock(
            model_dim,
            modes,
            _BlockVariant("forward", EDGE_FRAME_VARIANT),
        )
        _make_terminal_analysis(core)
        core.use_input_norm = False
        core.norm = nn.Identity()  # pyright: ignore[reportAttributeAccessIssue]
        core.local = None
        self.core = core
        self.modes = modes

    def forward(
        self,
        inputs: Tensor,
        *,
        time_delta: Tensor | None = None,
        observation_mask: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        return self.core(
            inputs,
            time_delta=time_delta,
            observation_mask=observation_mask,
            valid_mask=valid_mask,
            return_moments_only=True,
        )

    def retract_frame(self) -> None:
        self.core.retract_frame()

    def finalize_frame(self) -> None:
        self.core.finalize_frame()


class _LocalTerminalAnalyzer(_ScanOnlyTerminalAnalyzer):
    """Causal local preparation followed by the scan-only terminal core."""

    def __init__(self, model_dim: int, modes: int, *, learned_projection: bool) -> None:
        super().__init__(model_dim, modes)
        self.input_projection: nn.Module = (
            nn.Linear(model_dim, model_dim, bias=False) if learned_projection else nn.Identity()
        )
        if isinstance(self.input_projection, nn.Linear):
            nn.init.orthogonal_(self.input_projection.weight)
        self.local = nn.Conv1d(
            model_dim,
            model_dim,
            kernel_size=4,
            dilation=1,
            groups=model_dim,
        )

    def forward(
        self,
        inputs: Tensor,
        *,
        time_delta: Tensor | None = None,
        observation_mask: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        projected = self.input_projection(inputs)
        local = functional.silu(
            self.local(functional.pad(projected.transpose(1, 2), (3, 0))).transpose(1, 2)
        )
        if valid_mask is not None:
            active = valid_mask if valid_mask.ndim == 3 else valid_mask.unsqueeze(-1)
            local = local * active.to(device=local.device, dtype=local.dtype)
        return super().forward(
            local,
            time_delta=time_delta,
            observation_mask=observation_mask,
            valid_mask=valid_mask,
        )


class MambaExactScanOnlyM8RawPAC(_MambaExactRawPACBase):
    """One full no-RMS block followed by an M=8 scan-only analyzer."""

    terminal_modes: Final[int] = 8

    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        *,
        readout: ScanOnlyMomentReadout,
        objective: HeadroomObjective = "classification",
    ) -> None:
        super().__init__(config, output_dim, objective=objective)
        self.forward_block = _MambaExactPoleBlock(
            self.forward_block,
            self.model_dim,
            kernel_size=4,
            dilation=1,
        )
        self.forward_block.norm = nn.Identity()  # pyright: ignore[reportAttributeAccessIssue]
        self.backward_block = _ScanOnlyTerminalAnalyzer(  # pyright: ignore[reportIncompatibleUnannotatedOverride]
            self.model_dim,
            self.terminal_modes,
        )
        self.final_norm = None  # pyright: ignore[reportIncompatibleUnannotatedOverride]
        first_moment_dim = self.modes * (1 + 2 * len(self.forward_block.core.moment_lags))
        second_moment_dim = self.terminal_modes * (
            1 + 2 * len(self.backward_block.core.moment_lags)
        )
        self.head = _AsymmetricMomentHead(
            self.model_dim,
            first_moment_dim,
            second_moment_dim,
            output_dim,
            readout,
        )

    def forward(
        self,
        inputs: Tensor,
        *,
        time_delta: Tensor | None = None,
        observation_mask: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        stem_inputs = _apply_raw_mask(inputs, observation_mask, valid_mask)
        encoded = self._mask_features(self.stem(stem_inputs), valid_mask)
        encoded, first_moments = self.forward_block(
            encoded,
            time_delta=time_delta,
            observation_mask=observation_mask,
            valid_mask=valid_mask,
        )
        encoded = self._mask_features(encoded, valid_mask)
        second_moments = self.backward_block(
            encoded,
            time_delta=time_delta,
            observation_mask=observation_mask,
            valid_mask=valid_mask,
        )
        if valid_mask is None:
            pooled = encoded.mean(dim=1)
        else:
            weight = valid_mask.to(device=encoded.device, dtype=encoded.dtype)
            if weight.ndim == 2:
                weight = weight.unsqueeze(-1)
            pooled = (encoded * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)
        return self.head(pooled, first_moments, second_moments)


class MambaExactScanOnlyM8FirstRawPAC(MambaExactSingleBlockNoRMSNormRawPAC):
    """No-terminal-scan control: pooled H1 plus first-scan moments."""

    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        *,
        objective: HeadroomObjective = "classification",
    ) -> None:
        super().__init__(config, output_dim, objective=objective)


class MambaExactScanOnlyM8SecondRawPAC(MambaExactScanOnlyM8RawPAC):
    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        *,
        objective: HeadroomObjective = "classification",
    ) -> None:
        super().__init__(config, output_dim, readout="second", objective=objective)


class MambaExactScanOnlyM8BothRawPAC(MambaExactScanOnlyM8RawPAC):
    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        *,
        objective: HeadroomObjective = "classification",
    ) -> None:
        super().__init__(config, output_dim, readout="both", objective=objective)


class MambaExactLocalTerminalM8BothRawPAC(MambaExactScanOnlyM8BothRawPAC):
    """Add only a K4/D1 causal depthwise map before the terminal M=8 scan."""

    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        *,
        objective: HeadroomObjective = "classification",
    ) -> None:
        super().__init__(config, output_dim, objective=objective)
        self.backward_block = _LocalTerminalAnalyzer(  # pyright: ignore[reportIncompatibleUnannotatedOverride]
            self.model_dim,
            self.terminal_modes,
            learned_projection=False,
        )


class MambaExactProjectedLocalTerminalM8BothRawPAC(MambaExactScanOnlyM8BothRawPAC):
    """Add a dense analysis lift and K4/D1 local map before the M=8 scan."""

    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        *,
        objective: HeadroomObjective = "classification",
    ) -> None:
        super().__init__(config, output_dim, objective=objective)
        self.backward_block = _LocalTerminalAnalyzer(  # pyright: ignore[reportIncompatibleUnannotatedOverride]
            self.model_dim,
            self.terminal_modes,
            learned_projection=True,
        )


class _FirstBlockMixerAblation(_MambaExactPoleBlock):
    """Controlled gate/output/main-projection ablation of the first exact-pole block."""

    def __init__(
        self,
        core: _TightFrameBlock,
        model_dim: int,
        *,
        use_gate: bool,
        dense_output: bool,
        use_main_projection: bool,
    ) -> None:
        super().__init__(core, model_dim, kernel_size=4, dilation=1)
        self.norm = nn.Identity()  # pyright: ignore[reportAttributeAccessIssue]
        source_projection = self.input_projection
        main_projection = nn.Linear(model_dim, model_dim, bias=False)
        gate_projection = nn.Linear(model_dim, model_dim, bias=False)
        with torch.no_grad():
            main_projection.weight.copy_(source_projection.weight[:model_dim])
            gate_projection.weight.copy_(source_projection.weight[model_dim:])
        del self.input_projection
        self.main_projection: nn.Module = main_projection if use_main_projection else nn.Identity()
        self.gate_projection: nn.Module | None = gate_projection if use_gate else None
        self.use_gate = use_gate
        self.dense_output = dense_output
        self.use_main_projection = use_main_projection
        if dense_output:
            self.register_parameter("residual_scale", None)
        else:
            self.output_projection = nn.Identity()  # pyright: ignore[reportIncompatibleUnannotatedOverride]
            self.residual_scale = nn.Parameter(torch.full((model_dim,), 1.0e-2))

    def forward(
        self,
        inputs: Tensor,
        *,
        time_delta: Tensor | None = None,
        observation_mask: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        projected = self.main_projection(inputs)
        left_padding = self.local.dilation[0] * (self.local.kernel_size[0] - 1)
        local = functional.silu(
            self.local(functional.pad(projected.transpose(1, 2), (left_padding, 0))).transpose(1, 2)
        )
        if valid_mask is not None:
            active = valid_mask if valid_mask.ndim == 3 else valid_mask.unsqueeze(-1)
            local = local * active.to(device=local.device, dtype=local.dtype)
        modal_coordinates, moments = self._exact_pole_scan(
            local,
            time_delta=time_delta,
            observation_mask=observation_mask,
            valid_mask=valid_mask,
        )
        modal = torch.matmul(
            modal_coordinates,
            self.core.synthesis_frame_matrix().transpose(0, 1),
        )
        if self.core.synthesis_scale != 1.0:
            modal = self.core.synthesis_scale * modal
        update = modal + self.skip_scale.view(1, 1, -1) * local
        if self.gate_projection is not None:
            update = update * functional.silu(self.gate_projection(inputs))
        if self.residual_scale is None:
            update = self.output_projection(update)
        else:
            update = self.residual_scale.view(1, 1, -1) * update
        return inputs + update, moments


class MambaFirstBlockTerminalControlRawPAC(TerminalAnalysisRawRepeatedPAC):
    """Fix the best terminal analyzer while ablating only the first mixer."""

    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        *,
        use_gate: bool,
        dense_output: bool,
        use_main_projection: bool = True,
        objective: HeadroomObjective = "classification",
    ) -> None:
        super().__init__(config, output_dim, objective=objective)
        self.stem = RawProjectionEmbedding(config.raw_input_dim, self.model_dim)
        self.forward_block = _FirstBlockMixerAblation(  # pyright: ignore[reportIncompatibleUnannotatedOverride]
            self.forward_block,
            self.model_dim,
            use_gate=use_gate,
            dense_output=dense_output,
            use_main_projection=use_main_projection,
        )


class MambaFirstFullTerminalControlRawPAC(MambaFirstBlockTerminalControlRawPAC):
    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        *,
        objective: HeadroomObjective = "classification",
    ) -> None:
        super().__init__(
            config,
            output_dim,
            use_gate=True,
            dense_output=True,
            objective=objective,
        )


class MambaFirstNoGateTerminalRawPAC(MambaFirstBlockTerminalControlRawPAC):
    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        *,
        objective: HeadroomObjective = "classification",
    ) -> None:
        super().__init__(
            config,
            output_dim,
            use_gate=False,
            dense_output=True,
            objective=objective,
        )


class MambaFirstDiagonalOutputTerminalRawPAC(MambaFirstBlockTerminalControlRawPAC):
    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        *,
        objective: HeadroomObjective = "classification",
    ) -> None:
        super().__init__(
            config,
            output_dim,
            use_gate=True,
            dense_output=False,
            objective=objective,
        )


class MambaFirstNoGateDiagonalTerminalRawPAC(MambaFirstBlockTerminalControlRawPAC):
    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        *,
        objective: HeadroomObjective = "classification",
    ) -> None:
        super().__init__(
            config,
            output_dim,
            use_gate=False,
            dense_output=False,
            objective=objective,
        )


class MambaFirstNoGateDiagonalNoMainProjectionTerminalRawPAC(MambaFirstBlockTerminalControlRawPAC):
    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        *,
        objective: HeadroomObjective = "classification",
    ) -> None:
        super().__init__(
            config,
            output_dim,
            use_gate=False,
            dense_output=False,
            use_main_projection=False,
            objective=objective,
        )


class _BranchSpecificRawProjection(nn.Module):
    """Independent residual, main, and gate embeddings of the raw input."""

    def __init__(self, input_dim: int, model_dim: int) -> None:
        super().__init__()
        self.residual_projection = nn.Linear(input_dim, model_dim, bias=False)
        self.main_projection = nn.Linear(input_dim, model_dim, bias=False)
        self.gate_projection = nn.Linear(input_dim, model_dim, bias=False)
        nn.init.orthogonal_(self.residual_projection.weight)
        # Match the initialization induced by W_in @ W_raw in the unfused
        # parameterization.  W_in has shape (2D, D) with orthonormal columns,
        # so the vertically concatenated main/gate raw maps—not each map
        # independently—must have orthonormal columns.
        joint_main_gate = torch.empty(
            2 * model_dim,
            input_dim,
            device=self.main_projection.weight.device,
            dtype=self.main_projection.weight.dtype,
        )
        nn.init.orthogonal_(joint_main_gate)
        with torch.no_grad():
            self.main_projection.weight.copy_(joint_main_gate[:model_dim])
            self.gate_projection.weight.copy_(joint_main_gate[model_dim:])
        self.project_weight_()

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return (
            self.residual_projection(inputs),
            self.main_projection(inputs),
            self.gate_projection(inputs),
        )

    @torch.no_grad()
    def project_weight_(self) -> None:
        """Preserve only the inherited residual-stream embedding constraint."""
        weight = self.residual_projection.weight
        active = weight.float() if weight.shape[0] >= weight.shape[1] else weight.float().T
        frame, upper = torch.linalg.qr(active, mode="reduced")
        diagonal = torch.diagonal(upper)
        signs = torch.where(
            diagonal >= 0.0,
            torch.ones_like(diagonal),
            -torch.ones_like(diagonal),
        )
        projected = frame * signs.unsqueeze(0)
        if weight.shape[0] < weight.shape[1]:
            projected = projected.T
        weight.copy_(projected.to(dtype=weight.dtype))


class _CompactRawExactPoleBlock(_MambaExactPoleBlock):
    """Consume precomputed raw branches and apply a diagonal residual update."""

    def __init__(
        self,
        core: _TightFrameBlock,
        model_dim: int,
        *,
        modal_only_gate: bool,
    ) -> None:
        super().__init__(core, model_dim, kernel_size=4, dilation=1)
        self.norm = nn.Identity()  # pyright: ignore[reportAttributeAccessIssue]
        del self.input_projection
        self.output_projection = nn.Identity()  # pyright: ignore[reportIncompatibleUnannotatedOverride]
        self.residual_scale = nn.Parameter(torch.full((model_dim,), 1.0e-2))
        self.modal_only_gate = modal_only_gate

    def forward(
        self,
        residual_stream: Tensor,
        main_inputs: Tensor,
        gate_pre_activation: Tensor,
        *,
        time_delta: Tensor | None = None,
        observation_mask: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        left_padding = self.local.dilation[0] * (self.local.kernel_size[0] - 1)
        local = functional.silu(
            self.local(functional.pad(main_inputs.transpose(1, 2), (left_padding, 0))).transpose(
                1, 2
            )
        )
        if valid_mask is not None:
            active = valid_mask if valid_mask.ndim == 3 else valid_mask.unsqueeze(-1)
            local = local * active.to(device=local.device, dtype=local.dtype)
        modal_coordinates, moments = self._exact_pole_scan(
            local,
            time_delta=time_delta,
            observation_mask=observation_mask,
            valid_mask=valid_mask,
        )
        modal = torch.matmul(
            modal_coordinates,
            self.core.synthesis_frame_matrix().transpose(0, 1),
        )
        if self.core.synthesis_scale != 1.0:
            modal = self.core.synthesis_scale * modal
        gate = functional.silu(gate_pre_activation)
        local_skip = self.skip_scale.view(1, 1, -1) * local
        update = modal * gate + local_skip if self.modal_only_gate else (modal + local_skip) * gate
        return residual_stream + self.residual_scale.view(1, 1, -1) * update, moments


class CompactRawBranchesTerminalPAC(TerminalAnalysisRawRepeatedPAC):
    """Fold the first block's linear raw branches into compact embeddings."""

    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        *,
        modal_only_gate: bool,
        objective: HeadroomObjective = "classification",
    ) -> None:
        super().__init__(config, output_dim, objective=objective)
        self.stem = _BranchSpecificRawProjection(config.raw_input_dim, self.model_dim)
        self.forward_block = _CompactRawExactPoleBlock(  # pyright: ignore[reportIncompatibleUnannotatedOverride]
            self.forward_block,
            self.model_dim,
            modal_only_gate=modal_only_gate,
        )

    def forward(
        self,
        inputs: Tensor,
        *,
        time_delta: Tensor | None = None,
        observation_mask: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        stem_inputs = _apply_raw_mask(inputs, observation_mask, valid_mask)
        residual, main, gate = self.stem(stem_inputs)
        residual = self._mask_features(residual, valid_mask)
        main = self._mask_features(main, valid_mask)
        gate = self._mask_features(gate, valid_mask)
        first_stream, first_moments = self.forward_block(
            residual,
            main,
            gate,
            time_delta=time_delta,
            observation_mask=observation_mask,
            valid_mask=valid_mask,
        )
        second_projected = self.second_projection(first_stream)
        second_local = functional.silu(
            self.second_local(second_projected.transpose(1, 2)).transpose(1, 2)
        )
        encoded = self._mask_features(second_local, valid_mask)
        second_moments = self.backward_block(
            encoded,
            time_delta=time_delta,
            observation_mask=observation_mask,
            valid_mask=valid_mask,
            return_moments_only=True,
        )
        return self._readout(encoded, first_moments, second_moments, valid_mask)


class CompactRawBranchesAllGateTerminalPAC(CompactRawBranchesTerminalPAC):
    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        *,
        objective: HeadroomObjective = "classification",
    ) -> None:
        super().__init__(
            config,
            output_dim,
            modal_only_gate=False,
            objective=objective,
        )


class CompactRawBranchesModalOnlyGateTerminalPAC(CompactRawBranchesTerminalPAC):
    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        *,
        objective: HeadroomObjective = "classification",
    ) -> None:
        super().__init__(
            config,
            output_dim,
            modal_only_gate=True,
            objective=objective,
        )


@torch.no_grad()
def fold_diagonal_first_block_raw_projections_(
    source: MambaFirstDiagonalOutputTerminalRawPAC,
    target: CompactRawBranchesAllGateTerminalPAC,
) -> None:
    """Convert a dense-input diagonal-output model without changing its function."""
    target.load_state_dict(source.state_dict(), strict=False)
    source_stem = source.stem
    if not isinstance(source_stem, RawProjectionEmbedding):
        message = "source model does not expose the expected raw projection"
        raise TypeError(message)
    source_block = source.forward_block
    if not isinstance(source_block.main_projection, nn.Linear):
        message = "source model does not expose a dense main projection"
        raise TypeError(message)
    if not isinstance(source_block.gate_projection, nn.Linear):
        message = "source model does not expose a dense gate projection"
        raise TypeError(message)
    raw_weight = source_stem.projection.weight
    target.stem.residual_projection.weight.copy_(raw_weight)
    target.stem.main_projection.weight.copy_(source_block.main_projection.weight @ raw_weight)
    target.stem.gate_projection.weight.copy_(source_block.gate_projection.weight @ raw_weight)


class FusedScalarStemRawRepeatedPAC(RawRepeatedTwoForwardPAC):
    """Replace scalar P1 plus depthwise local filtering with one exact convolution."""

    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        *,
        objective: HeadroomObjective = "classification",
    ) -> None:
        super().__init__(config, output_dim, objective=objective)
        if not isinstance(self.stem, _RawForcingStem):
            message = "the raw repeated model did not build its canonical stem"
            raise TypeError(message)
        self.stem = FusedScalarRawStem(self.stem)


class TerminalFusedScalarStemRawRepeatedPAC(TerminalAnalysisRawRepeatedPAC):
    """Combine the two independently successful efficiency interventions."""

    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        *,
        objective: HeadroomObjective = "classification",
    ) -> None:
        super().__init__(config, output_dim, objective=objective)
        stem = self.stem
        if not isinstance(stem, _RawForcingStem):
            message = "the terminal raw model did not build its canonical stem"
            raise TypeError(message)
        self.stem = FusedScalarRawStem(stem)


class FixedLayerScaleRawRepeatedPAC(RawRepeatedTwoForwardPAC):
    """Fold both trainable layer scales into S and the normalized direct path."""

    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        *,
        objective: HeadroomObjective = "classification",
    ) -> None:
        super().__init__(config, output_dim, objective=objective)
        _fold_and_fix_layer_scale(self.forward_block)
        _fold_and_fix_layer_scale(self.backward_block)


class SharedAnalysisSynthesisRawRepeatedPAC(RawRepeatedTwoForwardPAC):
    """Share modal analysis R and untied synthesis S across both recurrences."""

    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        *,
        objective: HeadroomObjective = "classification",
    ) -> None:
        super().__init__(config, output_dim, objective=objective)
        first_synthesis = self.forward_block.independent_synthesis_frame
        if first_synthesis is None:
            message = "frame sharing requires independent synthesis parameters"
            raise RuntimeError(message)
        self.backward_block.frame = self.forward_block.frame
        self.backward_block.independent_synthesis_frame = first_synthesis


class FirstMomentReadoutRawRepeatedPAC(RawRepeatedTwoForwardPAC):
    """Retain only first-block modal moments in the classifier input."""

    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        *,
        objective: HeadroomObjective = "classification",
    ) -> None:
        super().__init__(config, output_dim, objective=objective)
        _select_single_moment_copy(self, "first")


class SecondMomentReadoutRawRepeatedPAC(RawRepeatedTwoForwardPAC):
    """Retain only second-block modal moments in the classifier input."""

    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        *,
        objective: HeadroomObjective = "classification",
    ) -> None:
        super().__init__(config, output_dim, objective=objective)
        _select_single_moment_copy(self, "second")

    def _readout(
        self,
        inputs: Tensor,
        forward_moments: Tensor,  # noqa: ARG002
        backward_moments: Tensor,
        valid_mask: Tensor | None,
    ) -> Tensor:
        return super()._readout(inputs, backward_moments, backward_moments, valid_mask)


def build_raw_efficiency_candidate(
    variant: RawEfficiencyVariant,
    config: PACExperimentConfig,
    output_dim: int,
    *,
    objective: HeadroomObjective = "classification",
) -> nn.Module:
    builders: dict[RawEfficiencyVariant, type[nn.Module]] = {
        "terminal_analysis": TerminalAnalysisRawRepeatedPAC,
        "terminal_analysis_no_rmsnorm": TerminalAnalysisNoRMSNormRawRepeatedPAC,
        "efp16_rmsnorm_control": EFP16RMSNormControlPAC,
        "efp16_no_rmsnorm": EFP16NoRMSNormPAC,
        "terminal_second_moments_only": TerminalSecondMomentReadoutRawRepeatedPAC,
        "one_modal_refinement": OneModalRefinementRawRepeatedPAC,
        "preprocessed_single_modal": PreprocessedSingleModalRawRepeatedPAC,
        "mamba_exact_single_block": MambaExactSingleBlockRawPAC,
        "mamba_exact_single_block_no_rmsnorm": MambaExactSingleBlockNoRMSNormRawPAC,
        "mamba_exact_terminal": MambaExactTerminalRawPAC,
        "mamba_exact_two_block": MambaExactTwoBlockRawPAC,
        "mamba_exact_two_block_no_rmsnorm": MambaExactTwoBlockNoRMSNormRawPAC,
        "mamba_exact_scan_only_m8_first": MambaExactScanOnlyM8FirstRawPAC,
        "mamba_exact_scan_only_m8_second": MambaExactScanOnlyM8SecondRawPAC,
        "mamba_exact_scan_only_m8_both": MambaExactScanOnlyM8BothRawPAC,
        "mamba_exact_local_terminal_m8_both": MambaExactLocalTerminalM8BothRawPAC,
        "mamba_exact_projected_local_terminal_m8_both": (
            MambaExactProjectedLocalTerminalM8BothRawPAC
        ),
        "mamba_first_full_terminal_control": MambaFirstFullTerminalControlRawPAC,
        "mamba_first_no_gate_terminal": MambaFirstNoGateTerminalRawPAC,
        "mamba_first_diagonal_output_terminal": (MambaFirstDiagonalOutputTerminalRawPAC),
        "mamba_first_no_gate_diagonal_terminal": (MambaFirstNoGateDiagonalTerminalRawPAC),
        "mamba_first_no_gate_diagonal_no_main_projection_terminal": (
            MambaFirstNoGateDiagonalNoMainProjectionTerminalRawPAC
        ),
        "compact_raw_branches_all_gate_terminal": CompactRawBranchesAllGateTerminalPAC,
        "compact_raw_branches_modal_only_gate_terminal": (
            CompactRawBranchesModalOnlyGateTerminalPAC
        ),
        "fused_scalar_stem": FusedScalarStemRawRepeatedPAC,
        "fixed_layer_scale": FixedLayerScaleRawRepeatedPAC,
        "shared_analysis_synthesis": SharedAnalysisSynthesisRawRepeatedPAC,
        "first_moments_only": FirstMomentReadoutRawRepeatedPAC,
        "second_moments_only": SecondMomentReadoutRawRepeatedPAC,
        "terminal_fused_scalar_stem": TerminalFusedScalarStemRawRepeatedPAC,
    }
    try:
        builder = builders[variant]
    except KeyError as error:
        message = f"unknown raw efficiency variant: {variant}"
        raise ValueError(message) from error
    return builder(config, output_dim, objective=objective)
