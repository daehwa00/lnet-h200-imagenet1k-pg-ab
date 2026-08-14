#!/usr/bin/env python3
"""Train the three prioritized Deep4 head follow-ups on ImageNet-100."""

# ruff: noqa: SLF001

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import run_a2d_resaux1_deep4_imagenet100 as deep4
import run_a2d_resaux1_imagenet100 as resaux_base
import run_double_prefc_imagenet100 as a2d_base
import torch
from torch import Tensor, nn

from lnet.complex_scan import ComplexScanConfig, ModalFusionHead
from lnet.image_layers import StandardizedAffineModalHead

if TYPE_CHECKING:
    from argparse import Namespace


VARIANTS = (
    "D4-W768",
    "D4-SR64-Aux",
    "D4-DirectionHead",
)
SEEDS = (501,)
CLASSES = 100
STAGES = 4
DIRECTIONS = 4
MODES = deep4.STAGE_MODES
STAGE_DIM = DIRECTIONS * MODES
DESCRIPTOR_DIM = STAGES * STAGE_DIM
WIDE_WIDTH = 768
STAGE_RESIDUAL_WIDTH = 64
DIRECTION_WIDTH = 192
AFFINE_AUXILIARY_WEIGHT = 1.0
heads = deep4.heads


class IndependentStageResidualAuxHead(nn.Module):
    """Reuse one affine path for joint prediction and auxiliary supervision."""

    def __init__(self, input_dim: int, output_dim: int, width: int) -> None:
        super().__init__()
        if input_dim != DESCRIPTOR_DIM:
            message = f"Deep4 stage residual expects {DESCRIPTOR_DIM} coordinates"
            raise ValueError(message)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.width = width
        self.affine = StandardizedAffineModalHead(input_dim, output_dim)
        self.stage_residuals = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(STAGE_DIM, width),
                    nn.GELU(),
                    nn.RMSNorm(width),
                    nn.Linear(width, output_dim, bias=False),
                )
                for _ in range(STAGES)
            ]
        )
        self.beta = nn.Parameter(torch.tensor(0.1))

    def joint_and_affine(self, descriptor: Tensor) -> tuple[Tensor, Tensor]:
        standardized = self.affine.standardizer(descriptor)
        affine_logits = self.affine.linear(standardized)
        stages = standardized.split(STAGE_DIM, dim=-1)
        correction = sum(
            (branch(stage) for branch, stage in zip(self.stage_residuals, stages, strict=True)),
            start=torch.zeros_like(affine_logits),
        )
        return affine_logits + self.beta * correction, affine_logits

    def forward(self, descriptor: Tensor) -> Tensor:
        joint, _ = self.joint_and_affine(descriptor)
        return joint


class SharedAffineAuxClassifier(heads.A2DAffineQClassifier):
    """Expose an SR head's existing affine logits without a duplicate head."""

    def __init__(self, head: IndependentStageResidualAuxHead) -> None:
        super().__init__(
            head.input_dim,
            head.output_dim,
            main="fusion",
            affine=head.affine,
            fusion=cast("Any", head),
            lrq=None,
            beta_lrq=None,
            affine_auxiliary_weight=AFFINE_AUXILIARY_WEIGHT,
        )

    def branch_logits(self, descriptor: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        head = cast("IndependentStageResidualAuxHead", self.fusion)
        joint, affine = head.joint_and_affine(descriptor)
        zero = torch.zeros_like(joint)
        return joint, affine, joint, zero


class Deep4DirectionFusionHead(nn.Module):
    """Mix the four stages independently within each scan direction."""

    def __init__(self, input_dim: int, output_dim: int, width: int) -> None:
        super().__init__()
        if input_dim != DESCRIPTOR_DIM:
            message = f"Deep4 direction head expects {DESCRIPTOR_DIM} coordinates"
            raise ValueError(message)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.width = width
        self.standardizer = nn.BatchNorm1d(input_dim, affine=False)
        self.direction_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(STAGES * MODES, width),
                    nn.GELU(),
                )
                for _ in range(DIRECTIONS)
            ]
        )
        self.norm = nn.RMSNorm(DIRECTIONS * width)
        self.classifier = nn.Linear(DIRECTIONS * width, output_dim)

    @staticmethod
    def direction_groups(descriptor: Tensor) -> tuple[Tensor, ...]:
        shaped = descriptor.reshape(descriptor.shape[0], STAGES, DIRECTIONS, MODES)
        return tuple(shaped[:, :, direction, :].flatten(1) for direction in range(DIRECTIONS))

    def forward(self, descriptor: Tensor) -> Tensor:
        groups = self.direction_groups(self.standardizer(descriptor))
        hidden = torch.cat(
            [
                projection(group)
                for projection, group in zip(
                    self.direction_projections,
                    groups,
                    strict=True,
                )
            ],
            dim=-1,
        )
        return self.classifier(self.norm(hidden))


def _auxiliary_classifier(
    fusion: nn.Module,
    *,
    output_dim: int,
) -> heads.A2DAffineQClassifier:
    return heads.A2DAffineQClassifier(
        DESCRIPTOR_DIM,
        output_dim,
        main="fusion",
        affine=StandardizedAffineModalHead(DESCRIPTOR_DIM, output_dim),
        fusion=cast("Any", fusion),
        lrq=None,
        beta_lrq=None,
        affine_auxiliary_weight=AFFINE_AUXILIARY_WEIGHT,
    )


def _build(variant: str, config: ComplexScanConfig) -> nn.Module:
    if variant not in VARIANTS:
        message = f"unsupported Deep4 head follow-up: {variant}"
        raise ValueError(message)
    model = deep4._build(deep4.VARIANT, config)
    if model.descriptor_dim != DESCRIPTOR_DIM:
        message = "Deep4 descriptor layout changed"
        raise RuntimeError(message)

    if variant == "D4-W768":
        fusion: nn.Module = ModalFusionHead(
            DESCRIPTOR_DIM,
            WIDE_WIDTH,
            config.output_dim,
        )
        model.classifier = _auxiliary_classifier(
            fusion,
            output_dim=config.output_dim,
        )
    elif variant == "D4-SR64-Aux":
        stage_residual = IndependentStageResidualAuxHead(
            DESCRIPTOR_DIM,
            config.output_dim,
            STAGE_RESIDUAL_WIDTH,
        )
        model.classifier = SharedAffineAuxClassifier(stage_residual)
    else:
        direction = Deep4DirectionFusionHead(
            DESCRIPTOR_DIM,
            config.output_dim,
            DIRECTION_WIDTH,
        )
        model.classifier = _auxiliary_classifier(
            direction,
            output_dim=config.output_dim,
        )
    return model


def _head_contract(variant: str) -> dict[str, Any]:
    common = {
        "affine_auxiliary_weight": AFFINE_AUXILIARY_WEIGHT,
        "lrq": False,
    }
    if variant == "D4-W768":
        return {
            **common,
            "main": "BN-Linear768-GELU-RMSNorm-Linear100",
            "affine_auxiliary": "independent standardized affine head",
        }
    if variant == "D4-SR64-Aux":
        return {
            **common,
            "main": "Affine plus four independent StageResidual64 branches",
            "affine_auxiliary": "same affine logits reused; no duplicate weights",
            "residual_scale_initial": 0.1,
        }
    return {
        **common,
        "main": "four direction-wise 192-to-192 projections, RMSNorm768, Linear100",
        "affine_auxiliary": "independent standardized affine head",
    }


def _contract(args: Namespace) -> dict[str, Any]:
    payload = deep4._contract(args)
    config = ComplexScanConfig(
        output_dim=CLASSES,
        stem_strides=(2, 2),
    )
    models = {variant: _build(variant, config) for variant in VARIANTS}
    payload.update(
        {
            "schema": "lnet.a2d.deep4_head_followups.imagenet100.v1",
            "evidence_status": "one-seed 100-epoch Deep4 head comparison",
            "variants": list(VARIANTS),
            "seeds": list(SEEDS),
        }
    )
    backbone = {
        "name": "A2D-D4-PathMix-PostCarry-PostFFN-4Stage",
        "modes": [MODES] * STAGES,
        "spatial_resolutions": [56, 28, 14, 7],
        "descriptor_layout": [STAGES, DIRECTIONS, MODES],
        "descriptor_dim": DESCRIPTOR_DIM,
    }
    payload["variant_configs"] = {
        variant: {"backbone": backbone, "head": _head_contract(variant)} for variant in VARIANTS
    }
    payload["parameter_counts"] = {
        variant: sum(parameter.numel() for parameter in model.parameters())
        for variant, model in models.items()
    }
    payload["architecture"] = {
        "D4-W768": (
            "Deep4 Q768 with one shallow 768-wide GELU/RMSNorm fusion and a "
            "separate affine auxiliary classifier weighted 1.0."
        ),
        "D4-SR64-Aux": (
            "Deep4 Q768 with one affine main path plus four independent "
            "192-to-64-to-100 stage residuals; the exact same affine logits "
            "receive auxiliary CE weight 1.0, without duplicate affine weights."
        ),
        "D4-DirectionHead": (
            "Deep4 Q768 reshaped as stage-by-direction-by-mode; each direction "
            "mixes its four 48-mode stage blocks through 192 hidden coordinates, "
            "then four direction embeddings are concatenated and classified."
        ),
    }
    payload["source_sha256"]["a2d_deep4_head_followups_runner"] = (
        deep4.baseline.heads.harness._digest(Path(__file__))
    )
    return json.loads(json.dumps(payload))


def _wandb_model_metrics(model: nn.Module) -> dict[str, float]:
    metrics = deep4._wandb_model_metrics(model)
    classifier = cast("Any", model).classifier
    if isinstance(classifier, SharedAffineAuxClassifier):
        head = cast("IndependentStageResidualAuxHead", classifier.fusion)
        metrics["head/stage_residual_beta"] = float(head.beta.detach())
    return metrics


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
