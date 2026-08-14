#!/usr/bin/env python3
"""Train four-stage A2D with one joint cross-stage residual head."""

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

from lnet.complex_scan import ComplexScanConfig

if TYPE_CHECKING:
    from argparse import Namespace


VARIANT = "D4-SR64-Joint"
SEEDS = (501,)
STAGE_WIDTH = 64
STAGE_DIM = 4 * deep4.STAGE_MODES
STAGE_COUNT = 4
heads = deep4.heads


class JointStageResidualHead(nn.Module):
    """Affine logits plus one nonlinear residual formed jointly across stages."""

    def __init__(self, input_dim: int, output_dim: int, width: int) -> None:
        super().__init__()
        if input_dim != STAGE_COUNT * STAGE_DIM:
            message = (
                "Deep4 joint stage residual requires four equal "
                f"{STAGE_DIM}-coordinate stage blocks"
            )
            raise ValueError(message)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.width = width
        self.stage_dim = STAGE_DIM
        self.stage_count = STAGE_COUNT
        self.standardizer = nn.BatchNorm1d(input_dim, affine=False)
        self.affine = nn.Linear(input_dim, output_dim)
        self.stage_embeddings = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(STAGE_DIM, width),
                    nn.GELU(),
                )
                for _ in range(STAGE_COUNT)
            ]
        )
        self.residual = nn.Sequential(
            nn.RMSNorm(STAGE_COUNT * width),
            nn.Linear(STAGE_COUNT * width, output_dim),
        )
        self.beta = nn.Parameter(torch.tensor(0.1))

    def joint_and_affine(self, descriptor: Tensor) -> tuple[Tensor, Tensor]:
        standardized = self.standardizer(descriptor)
        stages = standardized.split(self.stage_dim, dim=-1)
        hidden = torch.cat(
            [
                embedding(stage)
                for embedding, stage in zip(
                    self.stage_embeddings,
                    stages,
                    strict=True,
                )
            ],
            dim=-1,
        )
        affine_logits = self.affine(standardized)
        return affine_logits + self.beta * self.residual(hidden), affine_logits

    def forward(self, descriptor: Tensor) -> Tensor:
        joint, _ = self.joint_and_affine(descriptor)
        return joint


def _build(variant: str, config: ComplexScanConfig) -> nn.Module:
    if variant != VARIANT:
        message = f"unsupported A2D Deep4 joint stage-residual variant: {variant}"
        raise ValueError(message)
    model = deep4._build(deep4.VARIANT, config)
    joint_residual = JointStageResidualHead(
        deep4.DESCRIPTOR_DIM,
        config.output_dim,
        STAGE_WIDTH,
    )
    model.classifier = resaux_base.heads.A2DAffineQClassifier(
        deep4.DESCRIPTOR_DIM,
        config.output_dim,
        main="fusion",
        affine=None,
        fusion=cast("Any", joint_residual),
        lrq=None,
        beta_lrq=None,
        affine_auxiliary_weight=0.0,
    )
    return model


def _wandb_model_metrics(model: nn.Module) -> dict[str, float]:
    metrics = deep4._wandb_model_metrics(model)
    head = cast("JointStageResidualHead", cast("Any", model).classifier.fusion)
    metrics["head/stage_residual_beta"] = float(head.beta.detach())
    return metrics


def _contract(args: Namespace) -> dict[str, Any]:
    payload = deep4._contract(args)
    config = ComplexScanConfig(
        output_dim=100,
        stem_strides=(2, 2),
    )
    model = _build(VARIANT, config)
    payload.update(
        {
            "schema": "lnet.a2d.deep4_joint_stage_residual.imagenet100.v1",
            "evidence_status": "one-seed 100-epoch four-stage joint-residual head",
            "variants": [VARIANT],
            "seeds": list(SEEDS),
        }
    )
    payload["variant_configs"] = {
        VARIANT: {
            "backbone": {
                "name": "A2D-D4-PathMix-PostCarry-PostFFN-4Stage",
                "modes": [deep4.STAGE_MODES] * STAGE_COUNT,
                "spatial_resolutions": [56, 28, 14, 7],
                "descriptor_dim": deep4.DESCRIPTOR_DIM,
            },
            "head": {
                "normalizer": "BatchNorm1d(768, affine=False)",
                "main": "Affine768-to-100",
                "stage_embeddings": "4 independent 192-to-64-GELU projections",
                "joint_residual": "RMSNorm256-to-Linear100",
                "residual_scale_initial": 0.1,
                "affine_auxiliary_weight": 0.0,
                "lrq": False,
            },
        }
    }
    payload["parameter_counts"] = {
        VARIANT: sum(parameter.numel() for parameter in model.parameters())
    }
    payload["architecture"] = {
        VARIANT: (
            "A2D-Deep4 with its 768-coordinate descriptor standardized by "
            "non-affine BatchNorm. An affine 768-to-100 main path is corrected "
            "by beta times one joint residual: each of four 192-coordinate "
            "stages is embedded to 64 coordinates, the four embeddings are "
            "concatenated, RMS-normalized, and mapped to 100 logits. Training "
            "uses joint cross-entropy only, with no auxiliary affine loss or LRQ."
        )
    }
    payload["source_sha256"]["a2d_deep4_joint_stage_residual_runner"] = (
        deep4.baseline.heads.harness._digest(Path(__file__))
    )
    return json.loads(json.dumps(payload))


def main() -> None:
    source = resaux_base
    residuals = a2d_base.residuals
    harness = source.heads.harness
    source.heads.VARIANTS = (VARIANT,)
    source.heads.SEEDS = SEEDS
    source.structured._training_objective = source.heads._training_objective
    source.structured._after_training_batch = source.heads._after_training_batch
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    harness.main(
        harness.runner_bindings(
            variants=(VARIANT,),
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
