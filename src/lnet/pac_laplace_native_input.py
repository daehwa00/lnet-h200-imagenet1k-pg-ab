from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final, Literal

import torch
from torch import Tensor, nn
from torch.nn import functional

from .pac_headroom_efficient_models import (
    EDGE_FRAME_VARIANT,
    EdgeFramePAC,
    EdgeFrameStem,
    _apply_raw_mask,
    _combined_edge_mask,
    _degree_normalized_edge_analysis,
    _edge_or_singleton_mask,
)
from .pac_headroom_models import HEADROOM_SPECS, HeadroomObjective, HeadroomPACClassifier
from .pac_metrics import count_parameters

if TYPE_CHECKING:
    from .pac_types import PACExperimentConfig

LaplaceInputVariant = Literal[
    "raw_zoh",
    "value_innovation_zoh",
    "initialized_raw_zoh",
    "gated_raw_local_innovation_zoh",
    "direct_parallel_zoh",
    "direct_parallel_scaled_zoh",
    "mamba_gated_parallel_zoh",
    "two_layer_causal_mamba_zoh",
    "edge_forward_backward_zoh",
    "edge_two_forward_zoh",
    "edge_projected_two_forward_zoh",
    "edge_repeated_identity_two_forward_zoh",
    "edge_repeated_learned_two_forward_zoh",
    "raw_repeated_learned_two_forward_zoh",
]
LAPLACE_INPUT_VARIANTS: Final[tuple[LaplaceInputVariant, ...]] = (
    "raw_zoh",
    "value_innovation_zoh",
    "initialized_raw_zoh",
    "gated_raw_local_innovation_zoh",
    "direct_parallel_zoh",
    "direct_parallel_scaled_zoh",
    "mamba_gated_parallel_zoh",
    "two_layer_causal_mamba_zoh",
    "edge_forward_backward_zoh",
    "edge_two_forward_zoh",
    "edge_projected_two_forward_zoh",
    "edge_repeated_identity_two_forward_zoh",
    "edge_repeated_learned_two_forward_zoh",
    "raw_repeated_learned_two_forward_zoh",
)


@dataclass(frozen=True, slots=True)
class LaplaceInputMetadata:
    variant: LaplaceInputVariant
    drive: str
    transition: str
    params_trainable: int


def _validate_dwconv_geometry(kernel_size: object, dilation: object) -> None:
    """Validate a same-length depthwise-convolution geometry."""
    if type(kernel_size) is not int or kernel_size < 1:
        message = "dwconv_kernel_size must be a positive integer"
        raise ValueError(message)
    if type(dilation) is not int or dilation < 1:
        message = "dwconv_dilation must be a positive integer"
        raise ValueError(message)


def _make_same_length_depthwise_conv1d(
    model_dim: int,
    *,
    kernel_size: int,
    dilation: int,
) -> nn.Conv1d:
    """Construct a depthwise Conv1d that preserves every input timestep."""
    _validate_dwconv_geometry(kernel_size, dilation)
    receptive_span = dilation * (kernel_size - 1)
    padding: int | str = receptive_span // 2 if receptive_span % 2 == 0 else "same"
    return nn.Conv1d(
        model_dim,
        model_dim,
        kernel_size=kernel_size,
        dilation=dilation,
        padding=padding,
        groups=model_dim,
    )


class _RawForcingStem(nn.Module):
    """Raw-value projection with the same local map used by the edge-frame stem."""

    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        *,
        dwconv_kernel_size: int = 5,
        dwconv_dilation: int = 4,
    ) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, model_dim, bias=False)
        nn.init.orthogonal_(self.projection.weight)
        self.local = _make_same_length_depthwise_conv1d(
            model_dim,
            kernel_size=dwconv_kernel_size,
            dilation=dwconv_dilation,
        )
        self.project_weight_()

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

    def forward(self, inputs: Tensor) -> Tensor:
        projected = self.projection(inputs)
        local = self.local(projected.transpose(1, 2)).transpose(1, 2)
        return functional.silu(local)


class _InitializedRawForcingStem(nn.Module):
    """Raw ZOH forcing with learned local context and first-sample injection."""

    def __init__(self, input_dim: int, model_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, model_dim, bias=False)
        nn.init.orthogonal_(self.projection.weight)
        self.local = nn.Conv1d(
            model_dim,
            model_dim,
            kernel_size=5,
            dilation=4,
            padding=8,
            groups=model_dim,
        )
        self.initial_scale = nn.Parameter(torch.zeros(model_dim))

    def forward_with_skip(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        raw_skip = self.projection(inputs)
        local = functional.silu(self.local(raw_skip.transpose(1, 2)).transpose(1, 2))
        forcing = raw_skip + local
        if forcing.shape[1] > 0:
            initial = (
                forcing[:, :1] + torch.tanh(self.initial_scale).view(1, 1, -1) * raw_skip[:, :1]
            )
            forcing = torch.cat((initial, forcing[:, 1:]), dim=1)
        return forcing, raw_skip

    def forward(self, inputs: Tensor) -> Tensor:
        forcing, _ = self.forward_with_skip(inputs)
        return forcing


class _GatedRawLocalInnovationStem(nn.Module):
    """Mix raw, innovation, and learned local forcing with a task-level gate."""

    def __init__(self, input_dim: int, model_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.projection = nn.Linear(2 * input_dim, model_dim, bias=False)
        nn.init.orthogonal_(self.projection.weight)
        self.local = nn.Conv1d(
            model_dim,
            model_dim,
            kernel_size=5,
            dilation=4,
            padding=8,
            groups=model_dim,
        )
        self.gate_logits = nn.Parameter(torch.tensor([0.0, 0.0, 2.0]))

    def component_weights(self) -> Tensor:
        return torch.softmax(self.gate_logits, dim=0)

    def forward(self, inputs: Tensor) -> Tensor:
        raw, innovation = inputs.split(self.input_dim, dim=-1)
        raw_projection = functional.linear(raw, self.projection.weight[:, : self.input_dim])
        innovation_projection = functional.linear(
            innovation, self.projection.weight[:, self.input_dim :]
        )
        local_input = raw_projection + innovation_projection
        local = functional.silu(self.local(local_input.transpose(1, 2)).transpose(1, 2))
        weights = self.component_weights().to(dtype=inputs.dtype)
        return weights[0] * raw_projection + weights[1] * innovation_projection + weights[2] * local


class LaplaceNativePAC(HeadroomPACClassifier):
    """Drive the stable modal core from samples rather than graph-edge coordinates."""

    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        *,
        input_variant: LaplaceInputVariant,
        objective: HeadroomObjective = "classification",
    ) -> None:
        if input_variant not in LAPLACE_INPUT_VARIANTS:
            message = f"unknown Laplace input variant: {input_variant}"
            raise ValueError(message)
        super().__init__(
            config,
            output_dim,
            HEADROOM_SPECS["B"],
            objective=objective,
            pac_variant=EDGE_FRAME_VARIANT,
            mode_divisor=2,
        )
        self.input_variant = input_variant
        if input_variant == "raw_zoh":
            self.stem = _RawForcingStem(config.raw_input_dim, config.model_dim)
        elif input_variant == "initialized_raw_zoh":
            self.stem = _InitializedRawForcingStem(config.raw_input_dim, config.model_dim)
        elif input_variant == "gated_raw_local_innovation_zoh":
            self.stem = _GatedRawLocalInnovationStem(config.raw_input_dim, config.model_dim)
        else:
            self.stem = EdgeFrameStem(
                config.raw_input_dim,
                config.model_dim,
                semi_orthogonal=True,
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
        drive = (
            _value_innovation_drive(
                stem_inputs,
                time_delta=time_delta,
                observation_mask=observation_mask,
                valid_mask=valid_mask,
            )
            if self.input_variant in {"value_innovation_zoh", "gated_raw_local_innovation_zoh"}
            else stem_inputs
        )
        raw_skip: Tensor | None = None
        if isinstance(self.stem, _InitializedRawForcingStem):
            encoded, raw_skip = self.stem.forward_with_skip(drive)
        else:
            encoded = self.stem(drive)
        if valid_mask is not None:
            active_valid = valid_mask if valid_mask.ndim == 3 else valid_mask.unsqueeze(-1)
            encoded = encoded * active_valid.to(device=encoded.device, dtype=encoded.dtype)
            if raw_skip is not None:
                raw_skip = raw_skip * active_valid.to(device=raw_skip.device, dtype=raw_skip.dtype)
        encoded, forward_moments = self.forward_block(
            encoded,
            time_delta=time_delta,
            observation_mask=observation_mask,
            valid_mask=valid_mask,
        )
        encoded, backward_moments = self.backward_block(
            encoded,
            time_delta=time_delta,
            observation_mask=observation_mask,
            valid_mask=valid_mask,
        )
        if raw_skip is not None:
            encoded = encoded + raw_skip
        return self._readout(encoded, forward_moments, backward_moments, valid_mask)


class DirectParallelPAC(HeadroomPACClassifier):
    """Direct raw forcing with parallel forward and backward modal residuals."""

    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        *,
        objective: HeadroomObjective = "classification",
        scale_preserving_init: bool = False,
    ) -> None:
        direct_variant = replace(EDGE_FRAME_VARIANT, use_local_convolution=False)
        super().__init__(
            config,
            output_dim,
            HEADROOM_SPECS["B"],
            objective=objective,
            pac_variant=direct_variant,
            mode_divisor=2,
        )
        self.input_variant: LaplaceInputVariant = (
            "direct_parallel_scaled_zoh" if scale_preserving_init else "direct_parallel_zoh"
        )
        self.stem = nn.Linear(config.raw_input_dim, config.model_dim, bias=False)
        nn.init.orthogonal_(self.stem.weight)
        if scale_preserving_init:
            with torch.no_grad():
                self.stem.weight.mul_(math.sqrt(config.model_dim))
        self.forward_block.use_input_norm = False
        self.backward_block.use_input_norm = False
        self.forward_block.norm = nn.Identity()
        self.backward_block.norm = nn.Identity()

    def forward(
        self,
        inputs: Tensor,
        *,
        time_delta: Tensor | None = None,
        observation_mask: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        stem_inputs = _apply_raw_mask(inputs, observation_mask, valid_mask)
        base = self.stem(stem_inputs)
        if valid_mask is not None:
            active_valid = valid_mask if valid_mask.ndim == 3 else valid_mask.unsqueeze(-1)
            base = base * active_valid.to(device=base.device, dtype=base.dtype)
        forward_stream, forward_moments = self.forward_block(
            base,
            time_delta=time_delta,
            observation_mask=observation_mask,
            valid_mask=valid_mask,
        )
        backward_stream, backward_moments = self.backward_block(
            base,
            time_delta=time_delta,
            observation_mask=observation_mask,
            valid_mask=valid_mask,
        )
        fused = forward_stream + backward_stream - base
        return self._readout(fused, forward_moments, backward_moments, valid_mask)


class MambaGatedParallelPAC(HeadroomPACClassifier):
    """One local feature path, fixed-pole parallel memory, and an input gate."""

    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        *,
        objective: HeadroomObjective = "classification",
    ) -> None:
        modal_variant = replace(EDGE_FRAME_VARIANT, use_local_convolution=False)
        super().__init__(
            config,
            output_dim,
            HEADROOM_SPECS["B"],
            objective=objective,
            pac_variant=modal_variant,
            mode_divisor=2,
        )
        self.input_variant: LaplaceInputVariant = "mamba_gated_parallel_zoh"
        self.stem = nn.Identity()
        self.input_projection = nn.Linear(config.raw_input_dim, 2 * config.model_dim, bias=False)
        self.local = nn.Conv1d(
            config.model_dim,
            config.model_dim,
            kernel_size=5,
            groups=config.model_dim,
        )
        self.output_projection = nn.Linear(config.model_dim, config.model_dim, bias=False)
        self.forward_block.use_input_norm = False
        self.backward_block.use_input_norm = False
        self.forward_block.norm = nn.Identity()
        self.backward_block.norm = nn.Identity()

    def forward(
        self,
        inputs: Tensor,
        *,
        time_delta: Tensor | None = None,
        observation_mask: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        stem_inputs = _apply_raw_mask(inputs, observation_mask, valid_mask)
        content, gate = self.input_projection(stem_inputs).chunk(2, dim=-1)
        causal_content = functional.pad(content.transpose(1, 2), (self.local.kernel_size[0] - 1, 0))
        local = functional.silu(self.local(causal_content).transpose(1, 2))
        if valid_mask is not None:
            active_valid = valid_mask if valid_mask.ndim == 3 else valid_mask.unsqueeze(-1)
            local = local * active_valid.to(device=local.device, dtype=local.dtype)
        forward_stream, forward_moments = self.forward_block(
            local,
            time_delta=time_delta,
            observation_mask=observation_mask,
            valid_mask=valid_mask,
        )
        backward_stream, backward_moments = self.backward_block(
            local,
            time_delta=time_delta,
            observation_mask=observation_mask,
            valid_mask=valid_mask,
        )
        modal_update = forward_stream + backward_stream - 2.0 * local
        encoded = content + self.output_projection(functional.silu(gate) * modal_update)
        if valid_mask is not None:
            encoded = encoded * active_valid.to(device=encoded.device, dtype=encoded.dtype)
        return self._readout(encoded, forward_moments, backward_moments, valid_mask)


class TwoLayerCausalMambaPAC(HeadroomPACClassifier):
    """Two stacked causal fixed-pole Mamba shells with exact-ZOH memory."""

    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        *,
        objective: HeadroomObjective = "classification",
    ) -> None:
        modal_variant = replace(EDGE_FRAME_VARIANT, use_local_convolution=False)
        super().__init__(
            config,
            output_dim,
            HEADROOM_SPECS["B"],
            objective=objective,
            pac_variant=modal_variant,
            mode_divisor=2,
        )
        self.input_variant: LaplaceInputVariant = "two_layer_causal_mamba_zoh"
        self.stem = nn.Linear(config.raw_input_dim, config.model_dim, bias=False)
        self.layer_norms = nn.ModuleList(nn.RMSNorm(config.model_dim) for _ in range(2))
        self.input_projections = nn.ModuleList(
            nn.Linear(config.model_dim, 2 * config.model_dim, bias=False) for _ in range(2)
        )
        self.local_convolutions = nn.ModuleList(
            nn.Conv1d(
                config.model_dim,
                config.model_dim,
                kernel_size=5,
                groups=config.model_dim,
            )
            for _ in range(2)
        )
        self.output_projections = nn.ModuleList(
            nn.Linear(config.model_dim, config.model_dim, bias=False) for _ in range(2)
        )

        # The inherited second slot remains independent, but it is now a second
        # causal layer rather than a reverse-time pass.
        self.backward_block.direction = "forward"
        for block in (self.forward_block, self.backward_block):
            block.use_input_norm = False
            block.norm = nn.Identity()

    @staticmethod
    def _apply_valid_mask(features: Tensor, valid_mask: Tensor | None) -> Tensor:
        if valid_mask is None:
            return features
        active = valid_mask if valid_mask.ndim == 3 else valid_mask.unsqueeze(-1)
        return features * active.to(device=features.device, dtype=features.dtype)

    def _causal_layer(
        self,
        features: Tensor,
        layer_index: int,
        *,
        time_delta: Tensor | None,
        observation_mask: Tensor | None,
        valid_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        normalized = self.layer_norms[layer_index](features)
        content, gate = self.input_projections[layer_index](normalized).chunk(2, dim=-1)
        convolution = self.local_convolutions[layer_index]
        causal_content = functional.pad(
            content.transpose(1, 2), (convolution.kernel_size[0] - 1, 0)
        )
        local = functional.silu(convolution(causal_content).transpose(1, 2))
        local = self._apply_valid_mask(local, valid_mask)
        block = self.forward_block if layer_index == 0 else self.backward_block
        modal_stream, moments = block(
            local,
            time_delta=time_delta,
            observation_mask=observation_mask,
            valid_mask=valid_mask,
        )
        modal_update = modal_stream - local
        gated_update = functional.silu(gate) * modal_update
        output = features + self.output_projections[layer_index](gated_update)
        return self._apply_valid_mask(output, valid_mask), moments

    def forward(
        self,
        inputs: Tensor,
        *,
        time_delta: Tensor | None = None,
        observation_mask: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        stem_inputs = _apply_raw_mask(inputs, observation_mask, valid_mask)
        encoded = self._apply_valid_mask(self.stem(stem_inputs), valid_mask)
        encoded, first_layer_moments = self._causal_layer(
            encoded,
            0,
            time_delta=time_delta,
            observation_mask=observation_mask,
            valid_mask=valid_mask,
        )
        encoded, second_layer_moments = self._causal_layer(
            encoded,
            1,
            time_delta=time_delta,
            observation_mask=observation_mask,
            valid_mask=valid_mask,
        )
        return self._readout(
            encoded,
            first_layer_moments,
            second_layer_moments,
            valid_mask,
        )


class EdgeTwoForwardPAC(EdgeFramePAC):
    """Canonical Edge-Parseval model with only the second direction changed."""

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
            model_dim=32,
            pac_variant=EDGE_FRAME_VARIANT,
        )
        self.input_variant: LaplaceInputVariant = "edge_two_forward_zoh"
        self.backward_block.direction = "forward"


class EdgeForwardBackwardPAC(EdgeFramePAC):
    """Canonical Edge-Parseval direction-matched reference for the control."""

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
            model_dim=32,
            pac_variant=EDGE_FRAME_VARIANT,
        )
        self.input_variant: LaplaceInputVariant = "edge_forward_backward_zoh"


class EdgeProjectedTwoForwardPAC(EdgeFramePAC):
    """Two project-local-forward-modal blocks with one local map per block."""

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
            model_dim=32,
            pac_variant=EDGE_FRAME_VARIANT,
        )
        self.input_variant: LaplaceInputVariant = "edge_projected_two_forward_zoh"
        self.backward_block.direction = "forward"

        # Block 1 uses the edge stem's joint projection and dilated local map.
        self.forward_block.local = None

        # Block 2 receives a fresh channel mixture before its one causal local map.
        self.second_norm = nn.RMSNorm(self.model_dim)
        self.second_projection = nn.Linear(self.model_dim, self.model_dim, bias=False)
        nn.init.orthogonal_(self.second_projection.weight)
        self.second_local = nn.Conv1d(
            self.model_dim,
            self.model_dim,
            kernel_size=5,
            groups=self.model_dim,
        )
        self.backward_block.use_input_norm = False
        self.backward_block.norm = nn.Identity()
        self.backward_block.local = None

    @staticmethod
    def _mask_features(features: Tensor, valid_mask: Tensor | None) -> Tensor:
        if valid_mask is None:
            return features
        active = valid_mask if valid_mask.ndim == 3 else valid_mask.unsqueeze(-1)
        return features * active.to(device=features.device, dtype=features.dtype)

    def forward(
        self,
        inputs: Tensor,
        *,
        time_delta: Tensor | None = None,
        observation_mask: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        stem_inputs = _apply_raw_mask(inputs, observation_mask, valid_mask)
        level, detail, active_delta = _degree_normalized_edge_analysis(
            stem_inputs,
            time_delta,
        )
        active_observation = _edge_or_singleton_mask(observation_mask)
        active_valid = _edge_or_singleton_mask(valid_mask)
        edge_mask = _combined_edge_mask(active_observation, active_valid)
        edge_features = torch.cat((level, detail), dim=-1)
        if edge_mask is not None:
            edge_features = edge_features * edge_mask.to(
                device=edge_features.device,
                dtype=edge_features.dtype,
            )
        first_local = self._mask_features(self.stem(edge_features), active_valid)
        first_stream, first_moments = self.forward_block(
            first_local,
            time_delta=active_delta,
            observation_mask=active_observation,
            valid_mask=active_valid,
        )

        second_mixed = self.second_projection(self.second_norm(first_stream))
        causal_second = functional.pad(
            second_mixed.transpose(1, 2),
            (self.second_local.kernel_size[0] - 1, 0),
        )
        second_local = functional.silu(self.second_local(causal_second).transpose(1, 2))
        second_local = self._mask_features(second_local, active_valid)
        second_modal_stream, second_moments = self.backward_block(
            second_local,
            time_delta=active_delta,
            observation_mask=active_observation,
            valid_mask=active_valid,
        )
        second_update = second_modal_stream - second_local
        encoded = self._mask_features(first_stream + second_update, active_valid)
        return self._readout(encoded, first_moments, second_moments, active_valid)


class EdgeRepeatedTwoForwardPAC(EdgeFramePAC):
    """Literal project-local-modal blocks with the second stream kept intact."""

    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        *,
        learned_projection: bool,
        objective: HeadroomObjective = "classification",
    ) -> None:
        super().__init__(
            config,
            output_dim,
            modes=config.modes,
            semi_orthogonal=True,
            objective=objective,
            model_dim=config.model_dim,
            pac_variant=EDGE_FRAME_VARIANT,
            mode_divisor=2,
        )
        self.input_variant: LaplaceInputVariant = (
            "edge_repeated_learned_two_forward_zoh"
            if learned_projection
            else "edge_repeated_identity_two_forward_zoh"
        )
        self.backward_block.direction = "forward"
        self.forward_block.local = None
        self.backward_block.local = None

        # Construct the common local map before the optional projection so that
        # same-seed identity and learned variants start from the same function.
        self.second_local = nn.Conv1d(
            self.model_dim,
            self.model_dim,
            kernel_size=5,
            dilation=4,
            padding=8,
            groups=self.model_dim,
        )
        if learned_projection:
            projection = nn.Linear(self.model_dim, self.model_dim, bias=False)
            nn.init.eye_(projection.weight)
            self.second_projection: nn.Module = projection
        else:
            self.second_projection = nn.Identity()

    @staticmethod
    def _mask_features(features: Tensor, valid_mask: Tensor | None) -> Tensor:
        if valid_mask is None:
            return features
        active = valid_mask if valid_mask.ndim == 3 else valid_mask.unsqueeze(-1)
        return features * active.to(device=features.device, dtype=features.dtype)

    def forward(
        self,
        inputs: Tensor,
        *,
        time_delta: Tensor | None = None,
        observation_mask: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        stem_inputs = _apply_raw_mask(inputs, observation_mask, valid_mask)
        level, detail, active_delta = _degree_normalized_edge_analysis(
            stem_inputs,
            time_delta,
        )
        active_observation = _edge_or_singleton_mask(observation_mask)
        active_valid = _edge_or_singleton_mask(valid_mask)
        edge_mask = _combined_edge_mask(active_observation, active_valid)
        edge_features = torch.cat((level, detail), dim=-1)
        if edge_mask is not None:
            edge_features = edge_features * edge_mask.to(
                device=edge_features.device,
                dtype=edge_features.dtype,
            )

        first_local = self._mask_features(self.stem(edge_features), active_valid)
        first_stream, first_moments = self.forward_block(
            first_local,
            time_delta=active_delta,
            observation_mask=active_observation,
            valid_mask=active_valid,
        )
        second_projected = self.second_projection(first_stream)
        second_local = functional.silu(
            self.second_local(second_projected.transpose(1, 2)).transpose(1, 2)
        )
        second_local = self._mask_features(second_local, active_valid)
        encoded, second_moments = self.backward_block(
            second_local,
            time_delta=active_delta,
            observation_mask=active_observation,
            valid_mask=active_valid,
        )
        encoded = self._mask_features(encoded, active_valid)
        return self._readout(encoded, first_moments, second_moments, active_valid)


class RawRepeatedTwoForwardPAC(EdgeRepeatedTwoForwardPAC):
    """Repeated two-forward blocks driven directly by projected raw samples."""

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
            learned_projection=True,
            objective=objective,
        )
        self.input_variant: LaplaceInputVariant = "raw_repeated_learned_two_forward_zoh"
        self.stem = _RawForcingStem(config.raw_input_dim, self.model_dim)

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
        second_local = self._mask_features(second_local, valid_mask)
        encoded, second_moments = self.backward_block(
            second_local,
            time_delta=time_delta,
            observation_mask=observation_mask,
            valid_mask=valid_mask,
        )
        encoded = self._mask_features(encoded, valid_mask)
        return self._readout(encoded, first_moments, second_moments, valid_mask)


def _value_innovation_drive(
    inputs: Tensor,
    *,
    time_delta: Tensor | None,
    observation_mask: Tensor | None,
    valid_mask: Tensor | None,
) -> Tensor:
    innovation = torch.zeros_like(inputs)
    if inputs.shape[1] > 1:
        innovation[:, 1:] = inputs[:, 1:] - inputs[:, :-1]
        if time_delta is not None:
            delta = time_delta if time_delta.ndim == 3 else time_delta.unsqueeze(-1)
            innovation[:, 1:] = innovation[:, 1:] / delta[:, 1:].clamp_min(
                torch.finfo(inputs.dtype).eps
            ).to(dtype=inputs.dtype)
        active_mask = observation_mask if observation_mask is not None else valid_mask
        if active_mask is not None:
            mask = active_mask if active_mask.ndim == 3 else active_mask.unsqueeze(-1)
            pair_mask = mask[:, 1:] * mask[:, :-1]
            innovation[:, 1:] = innovation[:, 1:] * pair_mask.to(
                device=inputs.device, dtype=inputs.dtype
            )
    return torch.cat((inputs, innovation), dim=-1)


def build_laplace_native_classifier(
    variant: LaplaceInputVariant,
    config: PACExperimentConfig,
    output_dim: int,
    *,
    objective: HeadroomObjective = "classification",
) -> tuple[
    LaplaceNativePAC
    | DirectParallelPAC
    | MambaGatedParallelPAC
    | TwoLayerCausalMambaPAC
    | EdgeTwoForwardPAC
    | EdgeForwardBackwardPAC
    | EdgeProjectedTwoForwardPAC
    | EdgeRepeatedTwoForwardPAC
    | RawRepeatedTwoForwardPAC,
    LaplaceInputMetadata,
]:
    if variant == "raw_repeated_learned_two_forward_zoh":
        model: (
            LaplaceNativePAC
            | DirectParallelPAC
            | MambaGatedParallelPAC
            | TwoLayerCausalMambaPAC
            | EdgeTwoForwardPAC
            | EdgeForwardBackwardPAC
            | EdgeProjectedTwoForwardPAC
            | EdgeRepeatedTwoForwardPAC
            | RawRepeatedTwoForwardPAC
        ) = RawRepeatedTwoForwardPAC(
            config,
            output_dim,
            objective=objective,
        )
    elif variant in {
        "edge_repeated_identity_two_forward_zoh",
        "edge_repeated_learned_two_forward_zoh",
    }:
        model: (
            LaplaceNativePAC
            | DirectParallelPAC
            | MambaGatedParallelPAC
            | TwoLayerCausalMambaPAC
            | EdgeTwoForwardPAC
            | EdgeForwardBackwardPAC
            | EdgeProjectedTwoForwardPAC
            | EdgeRepeatedTwoForwardPAC
        ) = EdgeRepeatedTwoForwardPAC(
            config,
            output_dim,
            learned_projection=variant == "edge_repeated_learned_two_forward_zoh",
            objective=objective,
        )
    elif variant == "edge_projected_two_forward_zoh":
        model = EdgeProjectedTwoForwardPAC(config, output_dim, objective=objective)
    elif variant == "edge_forward_backward_zoh":
        model = EdgeForwardBackwardPAC(config, output_dim, objective=objective)
    elif variant == "edge_two_forward_zoh":
        model = EdgeTwoForwardPAC(config, output_dim, objective=objective)
    elif variant == "two_layer_causal_mamba_zoh":
        model: (
            LaplaceNativePAC
            | DirectParallelPAC
            | MambaGatedParallelPAC
            | TwoLayerCausalMambaPAC
            | EdgeTwoForwardPAC
            | EdgeForwardBackwardPAC
            | EdgeProjectedTwoForwardPAC
            | EdgeRepeatedTwoForwardPAC
        ) = TwoLayerCausalMambaPAC(config, output_dim, objective=objective)
    elif variant == "mamba_gated_parallel_zoh":
        model = MambaGatedParallelPAC(config, output_dim, objective=objective)
    elif variant in {"direct_parallel_zoh", "direct_parallel_scaled_zoh"}:
        model = DirectParallelPAC(
            config,
            output_dim,
            objective=objective,
            scale_preserving_init=variant == "direct_parallel_scaled_zoh",
        )
    else:
        model = LaplaceNativePAC(
            config,
            output_dim,
            input_variant=variant,
            objective=objective,
        )
    return model, LaplaceInputMetadata(
        variant=variant,
        drive=(
            "raw samples as piecewise-constant forcing"
            if variant == "raw_zoh"
            else (
                "raw ZOH with learned local forcing, first-sample injection, and raw skip"
                if variant == "initialized_raw_zoh"
                else (
                    "learned gated mixture of raw, local, and time-normalized innovation forcing"
                    if variant == "gated_raw_local_innovation_zoh"
                    else (
                        "direct linear raw forcing with parallel bidirectional modal residuals"
                        if variant in {"direct_parallel_zoh", "direct_parallel_scaled_zoh"}
                        else (
                            "single causal local forcing with input-gated parallel modal output"
                            if variant == "mamba_gated_parallel_zoh"
                            else (
                                "two stacked causal local-gated exact-ZOH modal layers"
                                if variant == "two_layer_causal_mamba_zoh"
                                else (
                                    "canonical Edge-Parseval drive with two forward modal layers"
                                    if variant == "edge_two_forward_zoh"
                                    else (
                                        "canonical Edge-Parseval drive with "
                                        "forward/backward modal layers"
                                        if variant == "edge_forward_backward_zoh"
                                        else (
                                            "two project-local-forward-modal blocks after "
                                            "canonical Edge-Parseval analysis"
                                            if variant == "edge_projected_two_forward_zoh"
                                            else (
                                                "literal repeated project-local-forward-modal "
                                                "blocks with an identity second projection"
                                                if variant
                                                == "edge_repeated_identity_two_forward_zoh"
                                                else (
                                                    "literal repeated project-local-forward-modal "
                                                    "blocks with a learned second projection"
                                                    if variant
                                                    == "edge_repeated_learned_two_forward_zoh"
                                                    else (
                                                        "raw samples projected directly into two "
                                                        "repeated local-forward-modal blocks"
                                                        if variant
                                                        == "raw_repeated_learned_two_forward_zoh"
                                                        else (
                                                            "raw value and time-normalized "
                                                            "innovation "
                                                            "as piecewise-constant forcing"
                                                        )
                                                    )
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        ),
        transition="stable learned poles with exact zero-order-hold transition",
        params_trainable=count_parameters(model),
    )
