#!/usr/bin/env python3
"""Train PGv2-H96-K3-RMSMatch with an unnormalized Q4 linear head."""

from __future__ import annotations

# pyright: reportArgumentType=false, reportExplicitAny=false
# pyright: reportImplicitRelativeImport=false, reportPrivateLocalImportUsage=false
# pyright: reportPrivateUsage=false
# ruff: noqa: SLF001
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import run_a2d_affine_qhead_imagenet100 as head_runner
import run_a2d_pgv2_h96_k3_rmsmatch_q4_affine_imagenet100 as q4
import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from argparse import Namespace

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig


VARIANT = "PGv2-H96-K3-RMSMatch-Q4Linear"
VARIANTS = (VARIANT,)
SEEDS = q4.SEEDS


class UnnormalizedLinearHead(nn.Module):
    """Expose the affine-head API while applying no descriptor normalization."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.standardizer = nn.Identity()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, descriptor: Tensor) -> Tensor:
        return self.linear(descriptor)


class Q4OnlyLinearClassifier(q4.StageOnlyAffineClassifier):
    """Select terminal Q4 and classify it with one unconstrained linear map."""

    def __init__(self, output_dim: int) -> None:
        head_runner.A2DAffineQClassifier.__init__(
            self,
            q4.Q4_DIM,
            output_dim,
            main="affine",
            affine=cast("Any", UnnormalizedLinearHead(q4.Q4_DIM, output_dim)),
            fusion=None,
            lrq=None,
            beta_lrq=None,
            affine_auxiliary_weight=0.0,
        )
        self.full_descriptor_dim = q4.FULL_DESCRIPTOR_DIM
        self.stage_index = 3


def _configure_ramp() -> None:
    q4.local_reader._configure_ramp()
    ramp = q4.local_reader.control.control.control.stemres.uniform.base
    ramp.VARIANT = VARIANT
    ramp.VARIANTS = VARIANTS
    ramp.SEEDS = SEEDS


def _assert_model(model: ComplexScanBackbone) -> None:
    if model.descriptor_dim != q4.FULL_DESCRIPTOR_DIM:
        message = "Q4 linear experiment changed the four-stage descriptor"
        raise RuntimeError(message)
    for name in ("stage1", "stage2", "stage3", "terminal"):
        reader = getattr(model, name).pole_input_projection
        if (
            not isinstance(reader, q4.PackedComplexConv2dReader)
            or reader.input_modes != q4.P
            or reader.output_modes != q4.P
            or reader.kernel_size != q4.local_reader.KERNEL_SIZE
            or not reader.match_input_rms
        ):
            message = f"{name} changed the K3 RMS-matched reader control"
            raise RuntimeError(message)
    classifier = model.classifier
    if not isinstance(classifier, Q4OnlyLinearClassifier):
        message = "Q4 linear experiment lost its declared classifier"
        raise TypeError(message)
    affine = classifier.affine
    if (
        not isinstance(affine, UnnormalizedLinearHead)
        or type(affine.standardizer) is not nn.Identity
        or affine.linear.in_features != q4.Q4_DIM
        or affine.linear.out_features != model.config.output_dim
        or classifier.stage_index != 3
        or classifier.fusion is not None
        or classifier.lrq is not None
        or classifier.affine_auxiliary_weight != 0.0
    ):
        message = "Q4 unnormalized linear-head contract changed"
        raise RuntimeError(message)


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    if variant != VARIANT:
        message = f"unsupported PGv2-H96 Q4 linear variant: {variant}"
        raise ValueError(message)
    model = q4.local_reader._build(q4.local_reader.RMS_MATCH_VARIANT, config)
    cast("Any", model).classifier = Q4OnlyLinearClassifier(config.output_dim)
    _configure_ramp()
    _assert_model(model)
    return model


def _variant_config() -> dict[str, Any]:
    payload = deepcopy(q4.local_reader._variant_config(q4.local_reader.RMS_MATCH_VARIANT))
    payload["backbone"]["name"] = "A2D-PGv2-H96-K3-RMSMatch"
    payload["head"] = {
        "descriptor_source": "terminal Q4 only; exact last 384 coordinates of Q1536",
        "operator": "Linear384-to-100",
        "normalization": "none",
        "fusion": False,
        "lrq": False,
        "auxiliary": False,
        "objective": "MixUp cross-entropy on the raw-Q4 linear logits",
    }
    return payload


def _contract(args: Namespace) -> dict[str, Any]:
    payload = q4.local_reader.control._contract(args)
    ramp = q4.local_reader.control.control.control.stemres.uniform.base
    config = ramp.PoleModelConfig(output_dim=100, stem_strides=(2, 2))
    model = _build(VARIANT, config)
    payload["schema"] = "lnet.a2d.pgv2_h96.k3_rmsmatch_q4_linear.imagenet100.v1"
    payload["evidence_status"] = "untrained Q4 linear-head normalization ablation"
    payload["variants"] = [VARIANT]
    payload["seeds"] = list(SEEDS)
    payload["variant_configs"] = {VARIANT: _variant_config()}
    payload["parameter_counts"] = {
        VARIANT: sum(parameter.numel() for parameter in model.parameters())
    }
    payload["architecture"] = {
        VARIANT: (
            "Exact completed PGv2-H96-K3-RMSMatch-Q4Affine control except that "
            "BatchNorm1d(384, affine=False) is removed. Raw terminal Q4 is sent "
            "directly to Linear384-to-100; Fusion, LRQ, and auxiliary loss remain absent."
        )
    }
    payload["recipe"]["head_objective"] = (
        "one MixUp cross-entropy on raw terminal-Q4 linear logits; no auxiliary term"
    )
    digest = ramp.heads.harness._digest
    payload["source_sha256"]["q4_linear_runner"] = digest(Path(__file__))
    return payload


def main() -> None:
    _configure_ramp()
    ramp = q4.local_reader.control.control.control.stemres.uniform.base
    source = ramp.canonical8.fair_init.backbone.deep4.baseline.baseline
    residuals = ramp.backbone.a2d_base.residuals
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
            model_config=ramp.PoleModelConfig,
            build_model=_build,
            contract=_contract,
            build_optimizer=residuals.optimizer_source._build_optimizer,
            prepare_model=source._prepare_model,
            train_epoch=source.structured._train_epoch,
            evaluate=source.heads._evaluate,
            wandb_model_metrics=q4.local_reader.control.control._wandb_model_metrics,
            summarize=source.heads._summarize,
        )
    )


if __name__ == "__main__":
    main()
