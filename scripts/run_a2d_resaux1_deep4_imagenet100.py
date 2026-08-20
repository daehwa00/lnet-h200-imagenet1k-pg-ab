#!/usr/bin/env python3
"""Train A2D-DeepHead with one additional pole transition and terminal stage."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import run_a2d_affine_qhead_imagenet100 as heads
import run_a2d_resaux1_deephead_imagenet100 as baseline
import run_a2d_resaux1_imagenet100 as resaux
import run_double_prefc_imagenet100 as a2d_base
import torch
from torch import Tensor, nn

from lnet.complex_scan import (
    ComplexScanBackbone,
    ComplexScanConfig,
    ComplexScanStage,
    ModalFusionHead,
    S2DPostFusionCFFNTransition,
)
from lnet.image_layers import StandardizedAffineModalHead

if TYPE_CHECKING:
    from argparse import Namespace

    from lnet.complex_scan import ComplexField


VARIANT = "A2D-Deep4"
SEEDS = (501,)
STAGE_MODES = 48
STAGE3_MAXIMUM_PHASE = math.pi * 0.60
DESCRIPTOR_DIM = 4 * 4 * STAGE_MODES
AFFINE_AUXILIARY_WEIGHT = 1.0


class FourStageA2D(ComplexScanBackbone):
    """Extend a matched three-stage A2D instance with a third transition."""

    def __init__(
        self,
        source: ComplexScanBackbone,
        stage3: ComplexScanStage,
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
        # The original analysis-only stage 3 becomes analysis-only stage 4.
        self.terminal = source.terminal
        self.descriptor_dim = DESCRIPTOR_DIM
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
        return torch.cat(
            (descriptor1, descriptor2, descriptor3, descriptor4),
            dim=-1,
        )


def _make_stage3(config: ComplexScanConfig) -> ComplexScanStage:
    postffn = resaux.backbone
    active = postffn._variant_config(postffn.VARIANT, config)
    if active.modes != (STAGE_MODES, STAGE_MODES, STAGE_MODES):
        message = "A2D-Deep4 requires the matched 48-mode A2D backbone"
        raise ValueError(message)
    stage = ComplexScanStage(
        STAGE_MODES,
        maximum_phase=STAGE3_MAXIMUM_PHASE,
        output_modes=STAGE_MODES,
        augmented_width=active.augmented_widths[-1] if active.augmented_widths else 96,
        carry_basis=active.carry_bases[-1],
        carry_merge=active.carry_merge,
        carry_scale_initial=active.carry_scale_initial,
        coherence_gated_carry=active.coherence_gated_carry,
        quadrant_path_mode_cffn_width=(
            active.quadrant_path_mode_cffn_widths[-1]
            if active.quadrant_path_mode_cffn_widths
            else None
        ),
        quadrant_path_cffn_width=(
            active.quadrant_path_cffn_widths[-1] if active.quadrant_path_cffn_widths else None
        ),
        stage_residual_scale_initial=active.stage_residual_scale_initial,
        damping_min=active.damping_min,
        damping_max=active.damping_max,
    )
    # Reproduce the established transition ladder exactly: the PostFFN
    # replacement consumes a completed PostCarry transition, not the original
    # augmented pole-main transition.
    postffn.postcarry._replace_transition(stage)
    postffn._replace_transition(stage)
    return stage


def _build(variant: str, config: ComplexScanConfig) -> nn.Module:
    if variant != VARIANT:
        message = f"unsupported A2D four-stage variant: {variant}"
        raise ValueError(message)
    source = baseline._build(baseline.VARIANT, config)
    model = FourStageA2D(source, _make_stage3(config))
    affine = StandardizedAffineModalHead(DESCRIPTOR_DIM, config.output_dim)
    fusion_source = ModalFusionHead(
        DESCRIPTOR_DIM,
        baseline.FIRST_WIDTH,
        config.output_dim,
    )
    fusion = baseline.DeepModalFusionHead(fusion_source, config.output_dim)
    model.classifier = heads.A2DAffineQClassifier(
        DESCRIPTOR_DIM,
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
    metrics = resaux._wandb_model_metrics(model)
    transition = model.stage3.augmented
    if not isinstance(transition, S2DPostFusionCFFNTransition):
        message = "A2D-Deep4 stage 3 lost its PostCarry/PostFFN transition"
        raise TypeError(message)
    prefix = "postcarry_postffn/stage3"
    if transition.pole_scale is not None:
        metrics[f"{prefix}/alpha"] = float(transition.pole_scale.detach())
    beta = transition.post_ffn_scale.detach().float()
    metrics[f"{prefix}/beta_mean"] = float(beta.mean())
    metrics[f"{prefix}/beta_min"] = float(beta.min())
    metrics[f"{prefix}/beta_max"] = float(beta.max())
    return metrics


def _contract(args: Namespace) -> dict[str, Any]:
    payload = baseline._contract(args)
    config = ComplexScanConfig(
        output_dim=100,
        stem_strides=(2, 2),
    )
    model = _build(VARIANT, config)
    payload.update(
        {
            "schema": "lnet.a2d.deep4.imagenet100.v1",
            "evidence_status": "one-seed 100-epoch four-stage depth comparison",
            "variants": [VARIANT],
            "seeds": list(SEEDS),
        }
    )
    payload["variant_configs"] = {
        VARIANT: {
            "backbone": {
                "name": "A2D-D4-PathMix-PostCarry-PostFFN-4Stage",
                "modes": [STAGE_MODES] * 4,
                "spatial_resolutions": [56, 28, 14, 7],
                "descriptor_dim": DESCRIPTOR_DIM,
                "added_transition": "matched stage3 D4-PathMix/PostCarry/PostFFN",
            },
            "head": {
                "main": "Fusion768-384-256",
                "affine_auxiliary_weight": AFFINE_AUXILIARY_WEIGHT,
                "lrq": False,
            },
        }
    }
    payload["parameter_counts"] = {
        VARIANT: sum(parameter.numel() for parameter in model.parameters())
    }
    payload["architecture"] = {
        VARIANT: (
            "A2D-DeepHead with a third matched 48-mode D4-PathMix/PostCarry/"
            "PostFFN transition at 14-to-7 resolution and the former terminal "
            "bank moved to stage 4. Four 192-coordinate Q descriptors are "
            "concatenated and classified by Fusion 768-to-384-to-256-to-100 "
            "with affine auxiliary CE weight 1.0 and no LRQ."
        )
    }
    payload["source_sha256"]["a2d_deep4_runner"] = baseline.heads.harness._digest(Path(__file__))
    return json.loads(json.dumps(payload))


def main() -> None:
    residuals = a2d_base.residuals
    harness = resaux.heads.harness
    resaux.heads.VARIANTS = (VARIANT,)
    resaux.heads.SEEDS = SEEDS
    resaux.structured._training_objective = resaux.heads._training_objective
    resaux.structured._after_training_batch = resaux.heads._after_training_batch
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    harness.main(
        harness.RunnerBindings(
            variants=(VARIANT,),
            seeds=SEEDS,
            model_config=ComplexScanConfig,
            build_model=_build,
            contract=_contract,
            build_optimizer=residuals.optimizer_source._build_optimizer,
            prepare_model=resaux._prepare_model,
            train_epoch=resaux.structured._train_epoch,
            evaluate=resaux.heads._evaluate,
            wandb_model_metrics=_wandb_model_metrics,
            summarize=resaux.heads._summarize,
        )
    )


if __name__ == "__main__":
    main()
