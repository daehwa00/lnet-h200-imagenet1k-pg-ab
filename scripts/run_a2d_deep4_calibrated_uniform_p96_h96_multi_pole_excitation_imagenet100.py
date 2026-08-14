#!/usr/bin/env python3
"""Train matched H96 controls with two or four poles per excitation."""

# pyright: reportArgumentType=false, reportExplicitAny=false
# pyright: reportImplicitRelativeImport=false, reportPrivateLocalImportUsage=false
# pyright: reportPrivateUsage=false
# ruff: noqa: EM102, SLF001, TRY003

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import run_a2d_deep4_calibrated_uniform_p96_h96_path_pg_modewise_complex_linear_imagenet100 as control  # noqa: E501
import torch
from torch import nn

from lnet.pac_modewise_path_collapse import (
    PhaseGatedModePathResidualModeWiseCollapse,
)
from lnet.pac_multi_pole_excitation import MultiPoleExcitationStage
from lnet.pac_phase_gated_transition import PhaseGatedS2DPostFusionTransition

if TYPE_CHECKING:
    from argparse import Namespace

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig


VARIANT_R2 = "PGv2-H96-MultiPoleE-R2"
VARIANT_R4 = "PGv2-H96-MultiPoleE-R4"
VARIANTS = (VARIANT_R2, VARIANT_R4)
MULTIPLICITY = {VARIANT_R2: 2, VARIANT_R4: 4}
SEEDS = control.SEEDS
P = control.P
MODE_HIDDEN = control.MODE_HIDDEN
PATH_HIDDEN = control.PATH_HIDDEN


def _assert_model(model: ComplexScanBackbone, multiplicity: int) -> None:
    for name in ("stage1", "stage2", "stage3", "terminal"):
        stage = getattr(model, name)
        if not isinstance(stage, MultiPoleExcitationStage):
            raise TypeError(f"{name} is missing its multi-pole wrapper")
        if stage.multiplicity != multiplicity or stage.modes != P:
            raise RuntimeError(f"{name} changed the requested pole multiplicity")
        expected_real = torch.full_like(stage.fusion_weight_real, 1.0 / multiplicity)
        if not torch.equal(stage.fusion_weight_real, expected_real) or torch.count_nonzero(
            stage.fusion_weight_imag
        ):
            raise RuntimeError(f"{name} lost exact uniform strict-complex initialization")
    for name in ("stage1", "stage2", "stage3"):
        inner = getattr(model, name).stage
        mixer = inner.quadrant_path_mode_combiner
        if not isinstance(mixer, PhaseGatedModePathResidualModeWiseCollapse):
            raise TypeError(f"{name} changed the matched ModePG/PathPG/MWCL control")
        if (
            mixer.mode.hidden_modes != MODE_HIDDEN
            or mixer.path.hidden_modes != PATH_HIDDEN
            or not isinstance(inner.augmented, PhaseGatedS2DPostFusionTransition)
            or inner.augmented.post.hidden_modes != MODE_HIDDEN
        ):
            raise RuntimeError(f"{name} changed downstream capacity")


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    try:
        multiplicity = MULTIPLICITY[variant]
    except KeyError as error:
        raise ValueError(f"unsupported multi-pole variant: {variant}") from error
    model = control._build(control.VARIANT, config)
    control._assert_model(model)
    for name in ("stage1", "stage2", "stage3", "terminal"):
        setattr(
            model,
            name,
            MultiPoleExcitationStage(getattr(model, name), multiplicity),
        )
    _assert_model(model, multiplicity)
    return model


def _variant_config(variant: str) -> dict[str, Any]:
    multiplicity = MULTIPLICITY[variant]
    payload = deepcopy(control._variant_config())
    payload["backbone"]["name"] = f"A2D-PGv2-H96-MultiPoleE-R{multiplicity}"
    payload["backbone"]["scan"] = {
        "excitation_modes": P,
        "poles_per_excitation": multiplicity,
        "raw_dynamical_states": P * multiplicity,
        "fusion": f"mode-wise strict ComplexLinear {multiplicity}-to-1",
        "fusion_initialization": f"exact mean: {1.0 / multiplicity:g}+0i",
        "conjugate_branch": False,
        "complex_bias": False,
        "initialization": (
            "each replica is a stratified permutation of the exact baseline pole atlas; "
            "the marginal pole distribution is unchanged"
        ),
        "execution": (
            "existing optimized D4 full16 product-scan kernels per replica; streaming "
            "strict-complex fusion avoids materializing NHW4MR; optimized bidirectional "
            "scan kernels provide the odd-sized terminal full state"
        ),
    }
    payload["backbone"]["unchanged_after_fusion"] = [
        "ModePG-H96",
        "PathPG-H8",
        "mode-wise strict D4 readout",
        "S2D carry",
        "PostPG-H96",
        "Q384 per stage",
        "classifier/head",
    ]
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
    for index, name in enumerate(("stage1", "stage2", "stage3", "terminal"), start=1):
        multi = getattr(model, name)
        if not isinstance(multi, MultiPoleExcitationStage):
            continue
        for metric, value in multi.diagnostic_metrics().items():
            metrics[f"multi_pole/stage{index}/{metric}"] = value
        if name == "terminal":
            continue
        mixer = multi.stage.quadrant_path_mode_combiner
        post = multi.stage.augmented
        if isinstance(mixer, PhaseGatedModePathResidualModeWiseCollapse) and isinstance(
            post,
            PhaseGatedS2DPostFusionTransition,
        ):
            for axis, block in (("mode", mixer.mode), ("path", mixer.path), ("post", post.post)):
                for metric, value in block.diagnostic_metrics().items():
                    metrics[f"phase_gated/stage{index}/{axis}_{metric}"] = value
                for metric, value in block.gradient_metrics().items():
                    metrics[f"phase_gated/stage{index}/{axis}_{metric}"] = value
    return metrics


def _contract(args: Namespace) -> dict[str, Any]:
    payload = control._contract(args)
    ramp = control.base.base.control.control.stemres.uniform.base
    config = ramp.PoleModelConfig(output_dim=100, stem_strides=(2, 2))
    models = {variant: _build(variant, config) for variant in VARIANTS}
    payload["schema"] = "lnet.a2d.pgv2_h96.multi_pole_excitation.imagenet100.v1"
    payload["evidence_status"] = "two-arm one-seed multi-pole causal experiment"
    payload["variants"] = list(VARIANTS)
    payload["seeds"] = list(SEEDS)
    payload["variant_configs"] = {variant: _variant_config(variant) for variant in VARIANTS}
    payload["parameter_counts"] = {
        variant: sum(parameter.numel() for parameter in model.parameters())
        for variant, model in models.items()
    }
    payload["architecture"] = {
        variant: (
            f"Each of 96 shared complex excitations is read by {MULTIPLICITY[variant]} "
            "independently learned D4 pole replicas. Their normalized full-grid responses "
            "are fused by a mode-wise bias-free strict complex map initialized to the exact "
            "mean before the unchanged H96 Mode PG, H8 Path PG, mode-wise D4 readout, S2D "
            "carry, H96 Post PG, Q384 measurement, and classifier."
        )
        for variant in VARIANTS
    }
    digest = ramp.heads.harness._digest
    payload["source_sha256"]["pac_multi_pole_excitation"] = digest(
        Path("src/lnet/pac_multi_pole_excitation.py")
    )
    payload["source_sha256"]["multi_pole_runner"] = digest(Path(__file__))
    return payload


def main() -> None:
    control.base.base._configure_ramp()
    ramp = control.base.base.control.control.stemres.uniform.base
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
