#!/usr/bin/env python3
"""Train A2D-ResAux1 with a 384-then-256 deep fusion head."""

# ruff: noqa: SLF001

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import run_a2d_resaux1_imagenet100 as baseline
import run_double_prefc_imagenet100 as a2d_base
import torch
from torch import Tensor, nn

from lnet.complex_scan import (
    ComplexScanConfig,
    ModalFusionHead,
    ParallelFusionLRQHead,
)
from lnet.image_layers import StandardizedAffineModalHead

if TYPE_CHECKING:
    from argparse import Namespace


VARIANT = "A2D-DeepHead"
SEEDS = (501,)
FIRST_WIDTH = 384
SECOND_WIDTH = 256
AFFINE_AUXILIARY_WEIGHT = 1.0
# Generic experiment smoke tooling expects the runner to expose its objective
# module directly, as the canonical ResAux1 runner does.
heads = baseline.heads


class DeepModalFusionHead(ModalFusionHead):
    """Preserve Fusion384, then refine it through a 256-wide hidden layer."""

    def __init__(self, source: ModalFusionHead, output_dim: int) -> None:
        nn.Module.__init__(self)
        if source.hidden_dim != FIRST_WIDTH:
            message = "deep fusion source is not the matched Fusion384 head"
            raise ValueError(message)
        self.input_dim = source.input_dim
        self.hidden_dim = source.hidden_dim
        self.second_hidden_dim = SECOND_WIDTH
        self.output_dim = output_dim
        # Reuse the matched Fusion384 input transform exactly.  Only the
        # refinement and final classifier are new.
        self.standardizer = source.standardizer
        self.fusion = source.fusion
        self.activation = source.activation
        self.norm = source.norm
        self.refinement = nn.Linear(FIRST_WIDTH, SECOND_WIDTH)
        self.refinement_activation = nn.GELU()
        self.refinement_norm = nn.RMSNorm(SECOND_WIDTH)
        self.classifier = nn.Linear(SECOND_WIDTH, output_dim)

    def forward(self, descriptor: Tensor) -> Tensor:
        standardized = self.standardizer(descriptor)
        first = self.norm(self.activation(self.fusion(standardized)))
        second = self.refinement_norm(
            self.refinement_activation(self.refinement(first))
        )
        return self.classifier(second)


def _build(
    variant: str,
    config: ComplexScanConfig,
) -> nn.Module:
    if variant != VARIANT:
        message = f"unsupported A2D deep-head variant: {variant}"
        raise ValueError(message)
    model = baseline.backbone._build(baseline.backbone.VARIANT, config)
    current = model.classifier
    if not isinstance(current, ParallelFusionLRQHead):
        message = "residual A2D backbone no longer exposes Fusion384+LRQ64"
        raise TypeError(message)
    # Construct the auxiliary head before the new refinement layers so its
    # initialization stays matched to A2D-ResAux1 under the same seed.
    affine = StandardizedAffineModalHead(model.descriptor_dim, config.output_dim)
    fusion = DeepModalFusionHead(current.fusion, config.output_dim)
    model.classifier = baseline.heads.A2DAffineQClassifier(
        model.descriptor_dim,
        config.output_dim,
        main="fusion",
        affine=affine,
        fusion=fusion,
        lrq=None,
        beta_lrq=None,
        affine_auxiliary_weight=AFFINE_AUXILIARY_WEIGHT,
    )
    return model


def _contract(args: Namespace) -> dict[str, Any]:
    payload = baseline._contract(args)
    config = ComplexScanConfig(
        output_dim=100,
        stem_strides=(2, 2),
    )
    model = _build(VARIANT, config)
    original = payload["variant_configs"][baseline.VARIANT]
    payload.update(
        {
            "schema": "lnet.a2d.deephead.imagenet100.v1",
            "evidence_status": "one-seed 100-epoch deep-fusion comparison",
            "variants": [VARIANT],
            "seeds": list(SEEDS),
        }
    )
    payload["variant_configs"] = {
        VARIANT: {
            "backbone": original["backbone"],
            "head": {
                "main": "Fusion384-256",
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
            "A2D-ResAux1 with its matched descriptor-to-384 GELU/RMSNorm "
            "fusion retained, followed by a 384-to-256 GELU/RMSNorm "
            "refinement and a 256-to-100 classifier. The standardized affine "
            "Q auxiliary head keeps weight 1.0; no LRQ."
        )
    }
    payload["source_sha256"]["a2d_deephead_runner"] = (
        baseline.heads.harness._digest(Path(__file__))
    )
    return json.loads(json.dumps(payload))


def main() -> None:
    residuals = a2d_base.residuals
    harness = baseline.heads.harness
    baseline.heads.VARIANTS = (VARIANT,)
    baseline.heads.SEEDS = SEEDS
    baseline.structured._training_objective = baseline.heads._training_objective
    baseline.structured._after_training_batch = baseline.heads._after_training_batch
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
            prepare_model=baseline._prepare_model,
            train_epoch=baseline.structured._train_epoch,
            evaluate=baseline.heads._evaluate,
            wandb_model_metrics=baseline._wandb_model_metrics,
            summarize=baseline.heads._summarize,
        )
    )


if __name__ == "__main__":
    main()
