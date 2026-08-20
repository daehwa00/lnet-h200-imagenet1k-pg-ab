#!/usr/bin/env python3
"""Train the uniform-P96 model with automatic Phase-Gated mode transitions."""

# pyright: reportArgumentType=false, reportExplicitAny=false
# pyright: reportImplicitRelativeImport=false, reportPrivateLocalImportUsage=false
# pyright: reportPrivateUsage=false
# pyright: reportUnnecessaryIsInstance=false

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import run_a2d_affine_qhead_imagenet100 as head_runner
import run_a2d_deep4_calibrated_uniform_p96_factorized_imagenet100 as control
import run_a2d_resaux1_deephead_imagenet100 as deephead
import torch
from torch import nn

from lnet.image_layers import StandardizedAffineModalHead
from lnet.pac_factorized_stage_transition import (
    FactorizedS2DPostFusionTransition,
    ModeResidualPathCollapse,
)
from lnet.pac_phase_gated_transition import PhaseGatedModeResidualPathCollapse

if TYPE_CHECKING:
    from argparse import Namespace

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanStage


VARIANT = "D4-Cal-U96-Stem32-MPM8-PGv2-H192-Aux05"
VARIANTS = (VARIANT,)
SEEDS = control.SEEDS
P = control.P
MODE_HIDDEN = 192
PATH_HIDDEN = control.PATH_HIDDEN
AFFINE_AUXILIARY_WEIGHT = 0.5


def _configure_ramp() -> None:
    control._configure_ramp()
    ramp = control.stemres.uniform.base
    ramp.VARIANT = VARIANT
    ramp.VARIANTS = VARIANTS
    ramp.SEEDS = SEEDS


def _install_phase_gated_mode(stage: ComplexScanStage) -> None:
    baseline = stage.quadrant_path_mode_combiner
    if not isinstance(baseline, ModeResidualPathCollapse):
        message = "Phase-Gated mode transition requires the MPM8 control"
        raise TypeError(message)
    mixer = PhaseGatedModeResidualPathCollapse(
        P,
        mode_hidden=MODE_HIDDEN,
        path_hidden=PATH_HIDDEN,
    )
    mixer.copy_path_from(baseline)
    stage.quadrant_path_mode_combiner = mixer


def _assert_model(model: ComplexScanBackbone) -> None:
    control.stemres._assert_stem(model)
    for name in ("stage1", "stage2", "stage3"):
        stage = getattr(model, name)
        mixer = stage.quadrant_path_mode_combiner
        if not isinstance(mixer, PhaseGatedModeResidualPathCollapse):
            message = f"{name} is missing its Phase-Gated mode transition"
            raise TypeError(message)
        if mixer.mode.hidden_modes != MODE_HIDDEN:
            message = f"{name} has the wrong Phase-Gated hidden width"
            raise RuntimeError(message)
        if (
            mixer.mode.gate_redistribution != 0.5
            or mixer.mode.gamma.shape
            or hasattr(mixer.mode, "beta")
        ):
            message = f"{name} changed the Phase-Gated v2 parameterization"
            raise RuntimeError(message)
        if type(stage.augmented) is not FactorizedS2DPostFusionTransition:
            message = f"{name} changed the MPM8 post-fusion control"
            raise TypeError(message)
    if model.terminal.output_modes is not None:
        message = "Phase-Gated mode transition changed the terminal descriptor"
        raise RuntimeError(message)
    classifier = model.classifier
    if not isinstance(classifier, head_runner.A2DAffineQClassifier):
        message = "Phase-Gated model lost its main/auxiliary classifier"
        raise TypeError(message)
    fusion = classifier.fusion
    affine = classifier.affine
    if (
        model.descriptor_dim != 1536
        or not isinstance(fusion, deephead.DeepModalFusionHead)
        or not isinstance(fusion.standardizer, nn.BatchNorm1d)
        or fusion.standardizer.affine
        or fusion.fusion.in_features != 1536
        or fusion.fusion.out_features != 384
        or not isinstance(fusion.norm, nn.RMSNorm)
        or fusion.refinement.in_features != 384
        or fusion.refinement.out_features != 256
        or not isinstance(fusion.refinement_norm, nn.RMSNorm)
        or fusion.classifier.in_features != 256
        or fusion.classifier.out_features != 100
    ):
        message = "Phase-Gated main classifier contract changed"
        raise RuntimeError(message)
    if (
        not isinstance(affine, StandardizedAffineModalHead)
        or not isinstance(affine.standardizer, nn.BatchNorm1d)
        or affine.standardizer.affine
        or affine.linear.in_features != 1536
        or affine.linear.out_features != 100
        or classifier.affine_auxiliary_weight != AFFINE_AUXILIARY_WEIGHT
    ):
        message = "Phase-Gated affine auxiliary contract changed"
        raise RuntimeError(message)


def _build(
    variant: str,
    config: control.stemres.uniform.base.PoleModelConfig,
) -> ComplexScanBackbone:
    if variant != VARIANT:
        message = f"unsupported Phase-Gated variant: {variant}"
        raise ValueError(message)
    model = control._build(control.VARIANT, config)
    for name in ("stage1", "stage2", "stage3"):
        _install_phase_gated_mode(getattr(model, name))
    classifier = model.classifier
    if not isinstance(classifier, head_runner.A2DAffineQClassifier):
        message = "Phase-Gated model requires the established main/auxiliary classifier"
        raise TypeError(message)
    classifier.affine_auxiliary_weight = AFFINE_AUXILIARY_WEIGHT
    _configure_ramp()
    _assert_model(model)
    return model


def _variant_config() -> dict[str, Any]:
    payload = deepcopy(control._variant_config())
    payload["backbone"]["name"] = "A2D-Cal-U96-Stem32-MPM8-PGv2-H192-Aux05"
    payload["head"] = {
        "main": "BatchNorm1536-Linear384-GELU-RMSNorm-Linear256-GELU-RMSNorm-Linear100",
        "auxiliary": "BatchNorm1536-affine-false-Linear100",
        "affine_auxiliary_weight": AFFINE_AUXILIARY_WEIGHT,
    }
    transition = payload["backbone"]["stage_transition"]
    transition["mode_residual"] = f"PhaseGatedComplexFFNv2-{P}-{MODE_HIDDEN}-{P}"
    transition["mode_gate"] = {
        "alpha_init": 0.075,
        "beta": "removed",
        "centered_log1p_energy": "exact mean subtraction with exact centered gradient",
        "gate_equation": "g0=1+0.5*tanh(alpha*c); g=g0/mean(g0)",
        "gate_mean": 1.0,
        "gate_statistics_dtype": "float32",
        "relative_gate_range_before_mean_normalization": [0.5, 1.5],
        "gate_hidden_width": MODE_HIDDEN,
        "input_projection_complex_width": 2 * MODE_HIDDEN,
        "output_projection_initialization": "standard ComplexLinear initialization",
        "residual_scale": {"name": "gamma", "initial": 0.01, "learnable": True},
        "redistribution": 0.5,
        "value_hidden_width": MODE_HIDDEN,
    }
    return payload


def _wandb_model_metrics(model: nn.Module) -> dict[str, float]:
    metrics = {
        "model/parameters": float(sum(parameter.numel() for parameter in model.parameters())),
        "model/trainable_parameters": float(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
    }
    metrics.update(
        {
            f"train/{name}": float(value)
            for name, value in getattr(model, "_latest_training_diagnostics", {}).items()
        }
    )
    for index, name in enumerate(("stage1", "stage2", "stage3"), start=1):
        mixer = getattr(model, name).quadrant_path_mode_combiner
        if isinstance(mixer, PhaseGatedModeResidualPathCollapse):
            for metric, value in mixer.mode.diagnostic_metrics().items():
                metrics[f"phase_gated/stage{index}/mode_{metric}"] = value
            for metric, value in mixer.mode.gradient_metrics().items():
                metrics[f"phase_gated/stage{index}/mode_{metric}"] = value
    return metrics


def _contract(args: Namespace) -> dict[str, Any]:
    payload = control._contract(args)
    config = control.stemres.uniform.base.PoleModelConfig(
        output_dim=100,
        stem_strides=(2, 2),
    )
    model = _build(VARIANT, config)
    payload["schema"] = "lnet.a2d.cal_u96_mpm8.phase_gated_mode.imagenet100.v3"
    payload["evidence_status"] = "untrained automatic Phase-Gated mode candidate"
    payload["variants"] = [VARIANT]
    payload["seeds"] = list(SEEDS)
    payload["variant_configs"] = {VARIANT: _variant_config()}
    payload["parameter_counts"] = {
        VARIANT: sum(parameter.numel() for parameter in model.parameters())
    }
    recipe = payload["recipe"]
    recipe["phase_gated_optimizer"] = {
        "alpha_gamma_crmsnorm_learning_rate": recipe["learning_rate"],
        "alpha_gamma_crmsnorm_weight_decay": 0.0,
        "projection_learning_rate": recipe["learning_rate"],
        "projection_weight_decay": 0.0,
        "selection": (
            "PhaseGatedComplexFFN type-owned projections use base LR without weight "
            "decay; this prevents gamma-gated branches from decaying while task "
            "gradients are suppressed"
        ),
    }
    payload["architecture"] = {
        VARIANT: (
            "Uniform-P96 Stem32 MPM8 with Stage 1-3 mode residuals replaced by "
            "Phase-Gated v2 H192 blocks. Exact energy centering and mean-one relative "
            "gates remove the common-scale gate direction; beta is absent. A normally "
            "initialized output projection is controlled by a learnable real gamma "
            "initialized to 0.01. Path, post-fusion, and terminal semantics remain "
            "unchanged. The 1536-384-256-100 main classifier is trained with a "
            "1536-100 affine auxiliary MixUp CE at weight 0.5."
        )
    }
    digest = control.stemres.uniform.base.heads.harness._digest
    for name in (
        "pac_phase_gated_cffn.py",
        "pac_phase_gated_transition.py",
        "pac_reduction_tiling.py",
        "pac_triton_complex_rmsnorm.py",
        "pac_triton_hardware.py",
        "pac_triton_phase_gate.py",
        "pac_triton_phase_gate_linear.py",
    ):
        payload["source_sha256"][name.removesuffix(".py")] = digest(Path("src/lnet") / name)
    payload["source_sha256"]["phase_gated_runner"] = digest(Path(__file__))
    payload["source_sha256"]["optimizer_source"] = digest(
        Path("scripts/run_complex_scan_augmented_cifar100.py")
    )
    return payload


def main() -> None:
    _configure_ramp()
    ramp = control.stemres.uniform.base
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
            wandb_model_metrics=_wandb_model_metrics,
            summarize=source.heads._summarize,
        )
    )


if __name__ == "__main__":
    main()
