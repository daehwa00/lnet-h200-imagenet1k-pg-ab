#!/usr/bin/env python3
"""Train matched Deep4 A2D pole-budget and coarse-CFFN variants."""

# ruff: noqa: SLF001

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import run_a2d_resaux1_deep4_imagenet100 as deep4
import run_a2d_resaux1_imagenet100 as resaux_base
import run_double_prefc_imagenet100 as a2d_base
import torch
from torch import Tensor, nn

from lnet.complex_scan import (
    ComplexScanBackbone,
    ComplexScanConfig,
    ComplexScanStage,
    ModalFusionHead,
    S2DPostCFFNCarryMainTransition,
    S2DPostFusionCFFNTransition,
)
from lnet.image_layers import StandardizedAffineModalHead

if TYPE_CHECKING:
    from argparse import Namespace

    from lnet.complex_scan import ComplexField


COARSE_MODES = "D4-CoarseModes"
UNIFORM_M64 = "D4-M64"
COARSE_FFN = "D4-CoarseFFN"
VARIANTS = (COARSE_MODES, UNIFORM_M64, COARSE_FFN)
SEEDS = (501,)
AFFINE_AUXILIARY_WEIGHT = 1.0
PATH_CFFN_WIDTH = 16
heads = deep4.heads


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


SPECS = {
    COARSE_MODES: Deep4BackboneSpec(
        modes=(32, 48, 64, 80),
        stem_width=96,
        mode_cffn_widths=(64, 96, 128),
        augmented_widths=(64, 96, 128),
        post_ffn_widths=(96, 128, 160),
    ),
    UNIFORM_M64: Deep4BackboneSpec(
        modes=(64, 64, 64, 64),
        # ComplexScanBackbone requires 2 * first-stage modes <= stem width.
        stem_width=128,
        mode_cffn_widths=(128, 128, 128),
        augmented_widths=(128, 128, 128),
        post_ffn_widths=(128, 128, 128),
    ),
    COARSE_FFN: Deep4BackboneSpec(
        modes=(48, 48, 48, 48),
        stem_width=96,
        mode_cffn_widths=(96, 96, 96),
        augmented_widths=(96, 96, 96),
        # Only the added 14-to-7 transition receives the wider PostFFN.
        post_ffn_widths=(96, 96, 192),
    ),
}


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

    def complex_features(
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
        descriptor = torch.cat(
            (descriptor1, descriptor2, descriptor3, descriptor4),
            dim=-1,
        )
        if descriptor.shape[-1] != self.descriptor_dim:
            message = "Deep4 variable-mode descriptor contract changed"
            raise RuntimeError(message)
        return descriptor


def _active_config(
    config: ComplexScanConfig,
    spec: Deep4BackboneSpec,
) -> ComplexScanConfig:
    postffn = resaux_base.backbone
    matched = postffn._variant_config(postffn.VARIANT, config)
    return replace(
        matched,
        stem_width=spec.stem_width,
        modes=spec.modes[:3],
        augmented_widths=spec.augmented_widths[:2],
        scan_memory_policy=config.scan_memory_policy,
        quadrant_path_mode_cffn_widths=spec.mode_cffn_widths[:2],
        quadrant_path_cffn_widths=(PATH_CFFN_WIDTH, PATH_CFFN_WIDTH),
    )


def _replace_postfusion(
    stage: ComplexScanStage,
    post_hidden_modes: int,
) -> None:
    previous = stage.augmented
    if not isinstance(previous, S2DPostCFFNCarryMainTransition):
        message = "Deep4 PostFFN requires a completed PostCarry transition"
        raise TypeError(message)
    transition = S2DPostFusionCFFNTransition(
        modes=stage.modes,
        hidden_modes=previous.hidden_modes,
        output_modes=previous.output_modes,
        pole_paths=previous.input_modes // stage.modes,
        expansion=previous.ffn_input.output_modes // previous.hidden_modes,
        pole_scale_initial=resaux_base.backbone.POLE_SCALE_INITIAL,
        post_hidden_modes=post_hidden_modes,
        post_layer_scale_initial=(resaux_base.backbone.POST_LAYER_SCALE_INITIAL),
    )
    transition.copy_pole_branch_from(previous)
    if transition.carry_weight is not None and previous.carry_weight is not None:
        transition.carry_weight.data.copy_(previous.carry_weight.data)
    elif transition.carry_projection is not None and previous.carry_projection is not None:
        transition.carry_projection.load_state_dict(previous.carry_projection.state_dict())
    else:
        message = "Deep4 PostCarry and PostFFN carry projections do not match"
        raise TypeError(message)
    transition.pole_scale.data.copy_(previous.pole_scale.data)
    stage.augmented = transition


def _install_transition(
    stage: ComplexScanStage,
    post_hidden_modes: int,
) -> None:
    postffn = resaux_base.backbone
    postffn.postcarry._replace_transition(stage)
    _replace_postfusion(stage, post_hidden_modes)


def _make_stage3(
    active: ComplexScanConfig,
    spec: Deep4BackboneSpec,
) -> ComplexScanStage:
    stage = ComplexScanStage(
        spec.modes[2],
        maximum_phase=deep4.STAGE3_MAXIMUM_PHASE,
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


def _make_terminal(
    active: ComplexScanConfig,
    modes: int,
) -> ComplexScanStage:
    return ComplexScanStage(
        modes,
        maximum_phase=math.pi * 0.65,
        output_modes=None,
        scan_memory_policy=active.scan_memory_policy,
        damping_min=active.damping_min,
        damping_max=active.damping_max,
    )


def _build(variant: str, config: ComplexScanConfig) -> nn.Module:
    try:
        spec = SPECS[variant]
    except KeyError as error:
        message = f"unsupported A2D Deep4 backbone variant: {variant}"
        raise ValueError(message) from error
    active = _active_config(config, spec)
    source = ComplexScanBackbone(active)
    source = a2d_base._remove_final_precomplex_gelu(source)
    _install_transition(source.stage1, spec.post_ffn_widths[0])
    _install_transition(source.stage2, spec.post_ffn_widths[1])
    descriptor_dim = spec.descriptor_dim
    model = VariableFourStageA2D(
        source,
        _make_stage3(active, spec),
        _make_terminal(active, spec.modes[3]),
        descriptor_dim,
    )
    affine = StandardizedAffineModalHead(descriptor_dim, config.output_dim)
    fusion_source = ModalFusionHead(
        descriptor_dim,
        deep4.baseline.FIRST_WIDTH,
        config.output_dim,
    )
    fusion = deep4.baseline.DeepModalFusionHead(fusion_source, config.output_dim)
    model.classifier = resaux_base.heads.A2DAffineQClassifier(
        descriptor_dim,
        config.output_dim,
        main="fusion",
        affine=affine,
        fusion=fusion,
        lrq=None,
        beta_lrq=None,
        affine_auxiliary_weight=AFFINE_AUXILIARY_WEIGHT,
    )
    return model


def _wandb_model_metrics(model: nn.Module) -> dict[str, float]:
    return deep4._wandb_model_metrics(model)


def _contract(args: Namespace) -> dict[str, Any]:
    payload = deep4._contract(args)
    config = ComplexScanConfig(
        output_dim=100,
        stem_strides=(2, 2),
    )
    models = {variant: _build(variant, config) for variant in VARIANTS}
    payload.update(
        {
            "schema": "lnet.a2d.deep4_backbone_variants.imagenet100.v1",
            "evidence_status": "one-seed 100-epoch Deep4 backbone ablation",
            "variants": list(VARIANTS),
            "seeds": list(SEEDS),
        }
    )
    payload["variant_configs"] = {
        variant: {
            "backbone": {
                "name": "A2D-D4-PostCarry-PostFFN-4Stage",
                "modes": list(spec.modes),
                "stem_width": spec.stem_width,
                "mode_cffn_widths": list(spec.mode_cffn_widths),
                "augmented_widths": list(spec.augmented_widths),
                "post_ffn_widths": list(spec.post_ffn_widths),
                "spatial_resolutions": [56, 28, 14, 7],
                "descriptor_dim": spec.descriptor_dim,
            },
            "head": {
                "main": f"Fusion{spec.descriptor_dim}-384-256",
                "affine_auxiliary_weight": AFFINE_AUXILIARY_WEIGHT,
                "lrq": False,
            },
        }
        for variant, spec in SPECS.items()
    }
    payload["parameter_counts"] = {
        variant: sum(parameter.numel() for parameter in model.parameters())
        for variant, model in models.items()
    }
    payload["architecture"] = {
        COARSE_MODES: (
            "Matched A2D-Deep4 with a 32/48/64/80 coarse-heavy pole budget; "
            "mode-bearing CFFNs scale with their stage while the original "
            "Fusion384-256 head and affine auxiliary CE 1.0 remain unchanged."
        ),
        UNIFORM_M64: (
            "Matched A2D-Deep4 with 64 modes at all four stages, 128-wide "
            "mode and PostFFN hidden states, and the unchanged Fusion384-256 "
            "head plus affine auxiliary CE 1.0."
        ),
        COARSE_FFN: (
            "Matched 48-mode A2D-Deep4 with only the added 14-to-7 stage's "
            "PostFFN widened from 48-96-48 to 48-192-48; all other backbone "
            "and head dimensions remain unchanged."
        ),
    }
    payload["source_sha256"]["a2d_deep4_backbone_variants_runner"] = heads.harness._digest(
        Path(__file__)
    )
    return json.loads(json.dumps(payload))


def main() -> None:
    source = resaux_base
    residuals = a2d_base.residuals
    harness = source.heads.harness
    source.heads.VARIANTS = VARIANTS
    source.heads.SEEDS = SEEDS
    source.structured._training_objective = source.heads._training_objective
    source.structured._after_training_batch = source.heads._after_training_batch
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    harness.main(
        harness.runner_bindings(
            variants=VARIANTS,
            seeds=SEEDS,
            model_config=ComplexScanConfig,
            build_model=_build,
            contract=_contract,
            build_optimizer=residuals.optimizer_source._build_optimizer,
            prepare_model=source._prepare_model,
            train_epoch=source.structured._train_epoch,
            evaluate=source.heads._evaluate,
            wandb_model_metrics=_wandb_model_metrics,
            summarize=source.heads._summarize,
        )
    )


if __name__ == "__main__":
    main()
