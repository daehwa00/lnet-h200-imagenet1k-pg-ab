#!/usr/bin/env python3
"""Train Q4Affine with pole geometry at the base learning rate."""

from __future__ import annotations

# pyright: reportArgumentType=false, reportExplicitAny=false
# pyright: reportImplicitRelativeImport=false, reportPrivateLocalImportUsage=false
# pyright: reportPrivateUsage=false
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import run_a2d_affine_qhead_imagenet100 as head_runner
import run_a2d_deep4_calibrated_uniform_p96_phase_gated_h96_local_reader_imagenet100 as local_reader
import torch
from torch import Tensor

from lnet.image_layers import StandardizedAffineModalHead
from lnet.pac_complex_scan_reader import PackedComplexConv2dReader

if TYPE_CHECKING:
    from argparse import Namespace

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig


VARIANT = "PGv2-H96-K3-RMSMatch-Q4Affine-PoleLR1"
VARIANTS = (VARIANT,)
POLE_GEOMETRY_LR_MULTIPLIER = 1.0
SEEDS = local_reader.SEEDS
P = local_reader.P
STAGE_COUNT = 4
Q4_DIM = 4 * P
FULL_DESCRIPTOR_DIM = STAGE_COUNT * Q4_DIM


class Q4OnlyAffineClassifier(head_runner.A2DAffineQClassifier):
    """Select terminal Q4 and apply one standardized affine classifier."""

    def __init__(self, output_dim: int) -> None:
        super().__init__(
            Q4_DIM,
            output_dim,
            main="affine",
            affine=StandardizedAffineModalHead(Q4_DIM, output_dim),
            fusion=None,
            lrq=None,
            beta_lrq=None,
            affine_auxiliary_weight=0.0,
        )
        self.full_descriptor_dim = FULL_DESCRIPTOR_DIM

    @staticmethod
    def select_q4(descriptor: Tensor) -> Tensor:
        if descriptor.shape[-1] != FULL_DESCRIPTOR_DIM:
            message = (
                "Q4-only affine head requires the ordered Q1-Q4 descriptor "
                f"with width {FULL_DESCRIPTOR_DIM}"
            )
            raise ValueError(message)
        return descriptor[..., -Q4_DIM:]

    def forward(self, descriptor: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        return super().forward(self.select_q4(descriptor))


def _configure_ramp() -> None:
    local_reader._configure_ramp()
    ramp = local_reader.control.control.control.stemres.uniform.base
    ramp.VARIANT = VARIANT
    ramp.VARIANTS = VARIANTS
    ramp.SEEDS = SEEDS


def _assert_model(model: ComplexScanBackbone) -> None:
    if model.descriptor_dim != FULL_DESCRIPTOR_DIM:
        message = "Q4 affine experiment changed the four-stage backbone descriptor"
        raise RuntimeError(message)
    for name in ("stage1", "stage2", "stage3", "terminal"):
        reader = getattr(model, name).pole_input_projection
        if (
            not isinstance(reader, PackedComplexConv2dReader)
            or reader.input_modes != P
            or reader.output_modes != P
            or reader.kernel_size != local_reader.KERNEL_SIZE
            or not reader.match_input_rms
        ):
            message = f"{name} changed the K3 RMS-matched scan-reader control"
            raise RuntimeError(message)
    classifier = model.classifier
    if not isinstance(classifier, Q4OnlyAffineClassifier):
        message = "Q4 affine experiment lost its terminal-only classifier"
        raise TypeError(message)
    affine = classifier.affine
    if (
        classifier.input_dim != Q4_DIM
        or classifier.main != "affine"
        or classifier.fusion is not None
        or classifier.lrq is not None
        or classifier.beta_lrq is not None
        or classifier.affine_auxiliary_weight != 0.0
        or not isinstance(affine, StandardizedAffineModalHead)
        or affine.standardizer.affine
        or affine.linear.in_features != Q4_DIM
        or affine.linear.out_features != model.config.output_dim
    ):
        message = "Q4 affine classifier contract changed"
        raise RuntimeError(message)


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    if variant != VARIANT:
        message = f"unsupported PGv2-H96 Q4 affine variant: {variant}"
        raise ValueError(message)
    model = local_reader._build(local_reader.RMS_MATCH_VARIANT, config)
    cast("Any", model).classifier = Q4OnlyAffineClassifier(config.output_dim)
    _configure_ramp()
    _assert_model(model)
    return model


def _variant_config() -> dict[str, Any]:
    payload = deepcopy(local_reader._variant_config(local_reader.RMS_MATCH_VARIANT))
    payload["backbone"]["name"] = "A2D-PGv2-H96-K3-RMSMatch"
    payload["head"] = {
        "descriptor_source": "terminal Q4 only; exact last 384 coordinates of Q1536",
        "operator": "BatchNorm384-affine-false-Linear100",
        "main": "affine",
        "fusion": False,
        "lrq": False,
        "auxiliary": False,
        "objective": "MixUp cross-entropy on the Q4 affine logits",
    }
    payload.setdefault("optimizer", {})["pole_geometry_learning_rate_multiplier"] = (
        POLE_GEOMETRY_LR_MULTIPLIER
    )
    return payload


def _build_optimizer(
    model: torch.nn.Module,
    recipe: dict[str, Any],
) -> torch.optim.Optimizer:
    adjusted_recipe = deepcopy(recipe)
    adjusted_recipe["pole_geometry_learning_rate_multiplier"] = POLE_GEOMETRY_LR_MULTIPLIER
    ramp = local_reader.control.control.control.stemres.uniform.base
    return ramp.backbone.a2d_base.residuals.optimizer_source._build_optimizer(
        model,
        adjusted_recipe,
    )


def _contract(args: Namespace) -> dict[str, Any]:
    payload = local_reader.control._contract(args)
    ramp = local_reader.control.control.control.stemres.uniform.base
    config = ramp.PoleModelConfig(output_dim=100, stem_strides=(2, 2))
    model = _build(VARIANT, config)
    payload["schema"] = "lnet.a2d.pgv2_h96.k3_rmsmatch_q4_affine_polelr1.imagenet100.v1"
    payload["evidence_status"] = "untrained pole-geometry base-LR ablation"
    payload["variants"] = [VARIANT]
    payload["seeds"] = list(SEEDS)
    payload["variant_configs"] = {VARIANT: _variant_config()}
    payload["parameter_counts"] = {
        VARIANT: sum(parameter.numel() for parameter in model.parameters())
    }
    payload["architecture"] = {
        VARIANT: (
            "Exact PGv2-H96-K3-RMSMatch stem, readers, D4 scans, transitions, and "
            "Q1-Q4 descriptor computation. Only the classifier changes: it selects "
            "terminal Q4 (the final 384 coordinates) and applies parameter-free "
            "BatchNorm standardization followed by Linear384-to-100. Fusion, LRQ, "
            "and auxiliary loss are absent."
        )
    }
    payload["recipe"]["head_objective"] = (
        "one MixUp cross-entropy on terminal-Q4 affine logits; no auxiliary term"
    )
    payload["recipe"]["pole_geometry_learning_rate_multiplier"] = POLE_GEOMETRY_LR_MULTIPLIER
    digest = ramp.heads.harness._digest
    payload["source_sha256"]["q4_affine_runner"] = digest(Path(__file__))
    return payload


def main() -> None:
    _configure_ramp()
    ramp = local_reader.control.control.control.stemres.uniform.base
    source = ramp.canonical8.fair_init.backbone.deep4.baseline.baseline
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
            build_optimizer=_build_optimizer,
            prepare_model=source._prepare_model,
            train_epoch=source.structured._train_epoch,
            evaluate=source.heads._evaluate,
            wandb_model_metrics=local_reader.control.control._wandb_model_metrics,
            summarize=source.heads._summarize,
        )
    )


if __name__ == "__main__":
    main()
