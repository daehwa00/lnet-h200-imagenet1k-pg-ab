"""Parity-locked four-stage builder used by current R2K3 models."""

from __future__ import annotations

# The builder intentionally reproduces the historical construction order.
# pyright: reportAttributeAccessIssue=false
import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor, nn

from lnet.a2d_q_heads import A2DAffineQClassifier
from lnet.complex_scan import (
    ComplexScanBackbone,
    ComplexScanConfig,
    ComplexScanStage,
    ModalFusionHead,
)
from lnet.complex_scan_transitions import (
    AugmentedComplexTransition,
    S2DPostCFFNCarryMainTransition,
    S2DPostFusionCFFNTransition,
)
from lnet.image_layers import StandardizedAffineModalHead

if TYPE_CHECKING:
    from lnet.complex_scan_types import ComplexField


PATH_CFFN_WIDTH = 16
POLE_SCALE_INITIAL = 1.0
POST_LAYER_SCALE_INITIAL = 0.1
FIRST_FUSION_WIDTH = 384
SECOND_FUSION_WIDTH = 256
AFFINE_AUXILIARY_WEIGHT = 1.0


@dataclass(frozen=True, slots=True)
class Deep4BackboneSpec:
    modes: tuple[int, int, int, int]
    stem_width: int
    mode_cffn_widths: tuple[int, int, int]
    augmented_widths: tuple[int, int, int]
    post_ffn_widths: tuple[int, int, int]

    @property
    def descriptor_dim(self) -> int:
        return 4 * sum(self.modes)


class VariableFourStageA2D(ComplexScanBackbone):
    """Join three adaptive transitions to a mode-matched terminal bank."""

    def __init__(
        self,
        source: ComplexScanBackbone,
        stage3: ComplexScanStage,
        terminal: ComplexScanStage,
        descriptor_dim: int,
    ) -> None:
        nn.Module.__init__(self)
        self.config = source.config
        self.stem = source.stem
        self.input_norm = source.input_norm
        self.precomplex_fc = source.precomplex_fc
        self.analysis = source.analysis
        self.stage1 = source.stage1
        self.stage2 = source.stage2
        self.stage3 = stage3
        self.terminal = terminal
        self.descriptor_dim = descriptor_dim
        self.classifier = source.classifier

    def complex_features(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        inputs: Tensor,
    ) -> tuple[ComplexField, ComplexField, ComplexField]:
        excitation = self._initial_excitation(inputs)
        state2, _ = self.stage1(*excitation)
        state2 = self._require_state(state2)
        state3, _ = self.stage2(*state2)
        state3 = self._require_state(state3)
        state4, _ = self.stage3(*state3)
        return state2, state3, self._require_state(state4)

    def raw_descriptor(self, inputs: Tensor) -> Tensor:
        excitation = self._initial_excitation(inputs)
        state2, descriptor1 = self.stage1(*excitation)
        state2 = self._require_state(state2)
        state3, descriptor2 = self.stage2(*state2)
        state3 = self._require_state(state3)
        state4, descriptor3 = self.stage3(*state3)
        state4 = self._require_state(state4)
        _, descriptor4 = self.terminal(*state4)
        descriptor = torch.cat((descriptor1, descriptor2, descriptor3, descriptor4), dim=-1)
        if descriptor.shape[-1] != self.descriptor_dim:
            raise RuntimeError("four-stage descriptor contract changed")
        return descriptor


class DeepModalFusionHead(ModalFusionHead):
    """Preserve Fusion384, then refine it through a 256-wide hidden layer."""

    def __init__(self, source: ModalFusionHead, output_dim: int) -> None:
        nn.Module.__init__(self)
        if source.hidden_dim != FIRST_FUSION_WIDTH:
            raise ValueError("deep fusion source is not the matched Fusion384 head")
        self.input_dim = source.input_dim
        self.hidden_dim = source.hidden_dim
        self.second_hidden_dim = SECOND_FUSION_WIDTH
        self.output_dim = output_dim
        self.standardizer = source.standardizer
        self.fusion = source.fusion
        self.activation = source.activation
        self.norm = source.norm
        self.refinement = nn.Linear(FIRST_FUSION_WIDTH, SECOND_FUSION_WIDTH)
        self.refinement_activation = nn.GELU()
        self.refinement_norm = nn.RMSNorm(SECOND_FUSION_WIDTH)
        self.classifier = nn.Linear(SECOND_FUSION_WIDTH, output_dim)

    def forward(self, descriptor: Tensor) -> Tensor:
        standardized = self.standardizer(descriptor)
        first = self.norm(self.activation(self.fusion(standardized)))
        second = self.refinement_norm(self.refinement_activation(self.refinement(first)))
        return self.classifier(second)


def _active_config(config: ComplexScanConfig, spec: Deep4BackboneSpec) -> ComplexScanConfig:
    return replace(
        config,
        stem_width=spec.stem_width,
        use_precomplex_fc=True,
        precomplex_fc_layers=2,
        modes=spec.modes[:3],
        augmented_widths=spec.augmented_widths[:2],
        carry_bases=("s2d", "s2d"),
        quadrant_path_mode_cffn_widths=spec.mode_cffn_widths[:2],
        quadrant_path_cffn_widths=(PATH_CFFN_WIDTH, PATH_CFFN_WIDTH),
        quadratic_rank=64,
        fusion_width=FIRST_FUSION_WIDTH,
        dual_fusion_lrq_head=True,
    )


def _remove_final_precomplex_gelu(model: ComplexScanBackbone) -> None:
    if not isinstance(model.precomplex_fc, nn.Sequential):
        raise TypeError("seeded builder is missing its pre-complex projection")
    layers = list(model.precomplex_fc.children())
    if (
        len(layers) != 4
        or not isinstance(layers[0], nn.Linear)
        or not isinstance(layers[1], nn.GELU)
        or not isinstance(layers[2], nn.Linear)
        or not isinstance(layers[3], nn.GELU)
    ):
        raise RuntimeError("pre-complex projection layout changed")
    model.precomplex_fc = nn.Sequential(layers[0], layers[1], layers[2], nn.Identity())


def _install_postcarry(stage: ComplexScanStage) -> None:
    previous = stage.augmented
    if not isinstance(previous, AugmentedComplexTransition) or stage.carry_basis != "s2d":
        raise TypeError("seeded builder requires an augmented S2D transition")
    transition = S2DPostCFFNCarryMainTransition(
        modes=stage.modes,
        hidden_modes=previous.hidden_modes,
        output_modes=previous.output_modes,
        pole_paths=previous.input_modes // stage.modes,
        expansion=previous.ffn_input.output_modes // previous.hidden_modes,
        pole_scale_initial=POLE_SCALE_INITIAL,
    )
    transition.copy_pole_branch_from(previous)
    stage.augmented = transition


def _install_postfusion(stage: ComplexScanStage, post_hidden_modes: int) -> None:
    previous = stage.augmented
    if not isinstance(previous, S2DPostCFFNCarryMainTransition):
        raise TypeError("seeded builder requires a completed PostCarry transition")
    transition = S2DPostFusionCFFNTransition(
        modes=stage.modes,
        hidden_modes=previous.hidden_modes,
        output_modes=previous.output_modes,
        pole_paths=previous.input_modes // stage.modes,
        expansion=previous.ffn_input.output_modes // previous.hidden_modes,
        pole_scale_initial=POLE_SCALE_INITIAL,
        post_hidden_modes=post_hidden_modes,
        post_layer_scale_initial=POST_LAYER_SCALE_INITIAL,
    )
    transition.copy_pole_branch_from(previous)
    if transition.carry_weight is not None and previous.carry_weight is not None:
        transition.carry_weight.data.copy_(previous.carry_weight.data)
    elif transition.carry_projection is not None and previous.carry_projection is not None:
        transition.carry_projection.load_state_dict(previous.carry_projection.state_dict())
    else:
        raise TypeError("PostCarry and PostFusion carry projections do not match")
    transition.pole_scale.data.copy_(previous.pole_scale.data)
    stage.augmented = transition


def _install_transition(stage: ComplexScanStage, post_hidden_modes: int) -> None:
    _install_postcarry(stage)
    _install_postfusion(stage, post_hidden_modes)


def _make_stage3(active: ComplexScanConfig, spec: Deep4BackboneSpec) -> ComplexScanStage:
    stage = ComplexScanStage(
        spec.modes[2],
        maximum_phase=math.pi * 0.60,
        output_modes=spec.modes[3],
        augmented_width=spec.augmented_widths[2],
        carry_basis=active.carry_bases[-1],
        carry_merge=active.carry_merge,
        carry_scale_initial=active.carry_scale_initial,
        coherence_gated_carry=active.coherence_gated_carry,
        quadrant_path_mode_cffn_width=spec.mode_cffn_widths[2],
        quadrant_path_cffn_width=PATH_CFFN_WIDTH,
        stage_residual_scale_initial=active.stage_residual_scale_initial,
        scan_memory_policy=active.scan_memory_policy,
        damping_min=active.damping_min,
        damping_max=active.damping_max,
    )
    _install_transition(stage, spec.post_ffn_widths[2])
    return stage


def _make_terminal(active: ComplexScanConfig, modes: int) -> ComplexScanStage:
    return ComplexScanStage(
        modes,
        maximum_phase=math.pi * 0.65,
        output_modes=None,
        scan_memory_policy=active.scan_memory_policy,
        damping_min=active.damping_min,
        damping_max=active.damping_max,
    )


def build(spec: Deep4BackboneSpec, config: ComplexScanConfig) -> ComplexScanBackbone:
    """Construct the historical Deep4 scaffold in its exact seeded order."""
    active = _active_config(config, spec)
    source = ComplexScanBackbone(active)
    _remove_final_precomplex_gelu(source)
    _install_transition(source.stage1, spec.post_ffn_widths[0])
    _install_transition(source.stage2, spec.post_ffn_widths[1])
    model = VariableFourStageA2D(
        source,
        _make_stage3(active, spec),
        _make_terminal(active, spec.modes[3]),
        spec.descriptor_dim,
    )
    affine = StandardizedAffineModalHead(spec.descriptor_dim, config.output_dim)
    fusion_source = ModalFusionHead(
        spec.descriptor_dim,
        FIRST_FUSION_WIDTH,
        config.output_dim,
    )
    fusion = DeepModalFusionHead(fusion_source, config.output_dim)
    model.classifier = A2DAffineQClassifier(
        spec.descriptor_dim,
        config.output_dim,
        main="fusion",
        affine=affine,
        fusion=fusion,
        lrq=None,
        beta_lrq=None,
        affine_auxiliary_weight=AFFINE_AUXILIARY_WEIGHT,
    )
    return cast("ComplexScanBackbone", model)


__all__ = ["Deep4BackboneSpec", "build"]
