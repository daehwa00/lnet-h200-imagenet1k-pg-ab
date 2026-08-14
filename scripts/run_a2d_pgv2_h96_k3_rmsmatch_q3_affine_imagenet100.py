#!/usr/bin/env python3
"""Train the exact PGv2-H96-K3-RMSMatch backbone with a Q3-only affine head."""

from __future__ import annotations

# pyright: reportArgumentType=false, reportExplicitAny=false
# pyright: reportImplicitRelativeImport=false, reportPrivateLocalImportUsage=false
# pyright: reportPrivateUsage=false
# ruff: noqa: SLF001
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import run_a2d_pgv2_h96_k3_rmsmatch_q4_affine_imagenet100 as q4
import torch

from lnet.image_layers import StandardizedAffineModalHead
from lnet.pac_complex_scan_reader import PackedComplexConv2dReader

if TYPE_CHECKING:
    from argparse import Namespace

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig


VARIANT = "PGv2-H96-K3-RMSMatch-Q3Affine"
VARIANTS = (VARIANT,)
SEEDS = q4.SEEDS
Q3_INDEX = 2


class Q3OnlyAffineClassifier(q4.StageOnlyAffineClassifier):
    """Select Q3 and apply one standardized affine classifier."""

    def __init__(self, output_dim: int) -> None:
        super().__init__(output_dim, stage_index=Q3_INDEX)


def _configure_ramp() -> None:
    q4.local_reader._configure_ramp()
    ramp = q4.local_reader.control.control.control.stemres.uniform.base
    ramp.VARIANT = VARIANT
    ramp.VARIANTS = VARIANTS
    ramp.SEEDS = SEEDS


def _assert_model(model: ComplexScanBackbone) -> None:
    if model.descriptor_dim != q4.FULL_DESCRIPTOR_DIM:
        message = "Q3 affine experiment changed the four-stage backbone descriptor"
        raise RuntimeError(message)
    for name in ("stage1", "stage2", "stage3", "terminal"):
        reader = getattr(model, name).pole_input_projection
        if (
            not isinstance(reader, PackedComplexConv2dReader)
            or reader.input_modes != q4.P
            or reader.output_modes != q4.P
            or reader.kernel_size != q4.local_reader.KERNEL_SIZE
            or not reader.match_input_rms
        ):
            message = f"{name} changed the K3 RMS-matched scan-reader control"
            raise RuntimeError(message)
    classifier = model.classifier
    if not isinstance(classifier, Q3OnlyAffineClassifier):
        message = "Q3 affine experiment lost its stage-three-only classifier"
        raise TypeError(message)
    affine = classifier.affine
    if (
        classifier.input_dim != q4.Q4_DIM
        or classifier.stage_index != Q3_INDEX
        or classifier.main != "affine"
        or classifier.fusion is not None
        or classifier.lrq is not None
        or classifier.beta_lrq is not None
        or classifier.affine_auxiliary_weight != 0.0
        or not isinstance(affine, StandardizedAffineModalHead)
        or affine.standardizer.affine
        or affine.linear.in_features != q4.Q4_DIM
        or affine.linear.out_features != model.config.output_dim
    ):
        message = "Q3 affine classifier contract changed"
        raise RuntimeError(message)


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    if variant != VARIANT:
        message = f"unsupported PGv2-H96 Q3 affine variant: {variant}"
        raise ValueError(message)
    model = q4.local_reader._build(q4.local_reader.RMS_MATCH_VARIANT, config)
    cast("Any", model).classifier = Q3OnlyAffineClassifier(config.output_dim)
    # Q4 is outside the objective. Mark its exclusive terminal bank inactive so
    # the optimizer and gradient-connectivity checks contain no dead parameters;
    # torch.compile can then eliminate the discarded terminal branch.
    model.terminal.requires_grad_(requires_grad=False)
    _configure_ramp()
    _assert_model(model)
    return model


def _variant_config() -> dict[str, Any]:
    payload = deepcopy(q4.local_reader._variant_config(q4.local_reader.RMS_MATCH_VARIANT))
    payload["backbone"]["name"] = "A2D-PGv2-H96-K3-RMSMatch"
    payload["head"] = {
        "descriptor_source": "Q3 only; exact coordinates 768:1152 of Q1536",
        "operator": "BatchNorm384-affine-false-Linear100",
        "main": "affine",
        "fusion": False,
        "lrq": False,
        "auxiliary": False,
        "objective": "MixUp cross-entropy on the Q3 affine logits",
    }
    return payload


def _contract(args: Namespace) -> dict[str, Any]:
    payload = q4.local_reader.control._contract(args)
    ramp = q4.local_reader.control.control.control.stemres.uniform.base
    config = ramp.PoleModelConfig(output_dim=100, stem_strides=(2, 2))
    model = _build(VARIANT, config)
    payload["schema"] = "lnet.a2d.pgv2_h96.k3_rmsmatch_q3_affine.imagenet100.v1"
    payload["evidence_status"] = "untrained stage-three-Q affine-head ablation"
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
            "Q3 (coordinates 768:1152) and applies parameter-free BatchNorm "
            "standardization followed by Linear384-to-100. Fusion, LRQ, and "
            "auxiliary loss are absent."
        )
    }
    payload["recipe"]["head_objective"] = (
        "one MixUp cross-entropy on Q3 affine logits; no auxiliary term"
    )
    digest = ramp.heads.harness._digest
    payload["source_sha256"]["q3_affine_runner"] = digest(Path(__file__))
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
