#!/usr/bin/env python3
"""Train the two E32 factorized wide-pole memory pyramids."""

# pyright: reportAny=false, reportExplicitAny=false
# pyright: reportImplicitRelativeImport=false, reportPrivateLocalImportUsage=false
# pyright: reportPrivateUsage=false
# ruff: noqa: SLF001

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import run_a2d_deep4_calibrated_uniform_p96_h96_path_pg_modewise_complex_linear_imagenet100 as control  # noqa: E501
import run_a2d_deep4_m64_canonical8_calibrated_imagenet100 as calibrated
import run_a2d_deep4_m64_canonical8_imagenet100 as canonical8
import torch
from torch import nn
from torch.nn.utils import parametrize
from torch.nn.utils.parametrizations import orthogonal

from lnet.pac_factorized_wide_pole_memory import FactorizedWidePoleMemoryStage

if TYPE_CHECKING:
    from argparse import Namespace

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig
    from lnet.pac_phase_gated_cffn import PhaseGatedComplexFFN


VARIANT_BALANCED = "E32-PolePyr-8-16-32-64"
VARIANT_ISOSCAN = "E32-PoleIsoScan-8-32-128-256"
VARIANTS = (VARIANT_BALANCED, VARIANT_ISOSCAN)
POLE_SCHEDULES = {
    VARIANT_BALANCED: (8, 16, 32, 64),
    VARIANT_ISOSCAN: (8, 32, 128, 256),
}
SEEDS = (501,)
CONTENT_MODES = 32
STEM_REAL_WIDTH = 192
DESCRIPTOR_DYNAMICS = 3
STAGE_NAMES = ("stage1", "stage2", "stage3", "terminal")


def _replace_analysis(model: ComplexScanBackbone) -> None:
    if (
        model.analysis is None
        or model.analysis.in_features != STEM_REAL_WIDTH
        or model.analysis.out_features != STEM_REAL_WIDTH
    ):
        message = "wide-pole experiment requires the retained 192-wide stem interface"
        raise TypeError(message)
    analysis = nn.Linear(STEM_REAL_WIDTH, 2 * CONTENT_MODES, bias=False)
    nn.init.orthogonal_(analysis.weight)
    orthogonal(
        analysis,
        "weight",
        orthogonal_map="matrix_exp",
        use_trivialization=True,
    )
    model.analysis = analysis


def _assert_model(
    model: ComplexScanBackbone,
    schedule: tuple[int, ...],
) -> None:
    stemres = control.base.base.control.control.stemres
    if not isinstance(model.stem, stemres.ModeScaledTwoConvStem):
        message = "factorized wide memory lost the 3-to-32-to-192 stem"
        raise TypeError(message)
    if model.stem.output_width != STEM_REAL_WIDTH:
        message = "factorized wide memory changed stem spatial capacity"
        raise RuntimeError(message)
    if (
        model.analysis is None
        or model.analysis.in_features != STEM_REAL_WIDTH
        or model.analysis.out_features != 2 * CONTENT_MODES
        or model.analysis.bias is not None
        or not parametrize.is_parametrized(model.analysis, "weight")
    ):
        message = "factorized wide memory requires orthogonal 192-to-64 analysis"
        raise TypeError(message)
    for index, (name, poles) in enumerate(zip(STAGE_NAMES, schedule, strict=True)):
        stage = getattr(model, name)
        if not isinstance(stage, FactorizedWidePoleMemoryStage):
            message = f"{name} is missing its factorized wide-memory stage"
            raise TypeError(message)
        if (
            stage.content_modes != CONTENT_MODES
            or stage.poles != poles
            or stage.terminal != (name == "terminal")
            or stage.content_pg.hidden_modes != CONTENT_MODES
            or stage.pole_pg.hidden_modes != 16
            or stage.path_pg.hidden_modes != 8
            or stage.stage_index != index
        ):
            message = f"{name} changed the requested factorized axis contract"
            raise RuntimeError(message)
        expected_descriptor = (CONTENT_MODES, DESCRIPTOR_DYNAMICS, poles)
        if tuple(stage.descriptor_readout_real.shape) != expected_descriptor:
            message = f"{name} changed its R-to-3 Q readout"
            raise RuntimeError(message)
        if name != "terminal" and (
            stage.pole_readout_real is None
            or tuple(stage.pole_readout_real.shape) != (CONTENT_MODES, poles)
            or stage.post_pg is None
            or stage.post_pg.modes != CONTENT_MODES
        ):
            message = f"{name} changed its structured next-excitation readout"
            raise RuntimeError(message)
    if model.descriptor_dim != 4 * 4 * CONTENT_MODES * DESCRIPTOR_DYNAMICS:
        message = "factorized wide memory changed the matched Q1536 head interface"
        raise RuntimeError(message)


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    try:
        schedule = POLE_SCHEDULES[variant]
    except KeyError as error:
        message = f"unsupported factorized wide-pole variant: {variant}"
        raise ValueError(message) from error
    control.base.base._configure_ramp()
    model = control._build(control.VARIANT, config)
    control._assert_model(model)
    _replace_analysis(model)
    for index, (name, poles) in enumerate(zip(STAGE_NAMES, schedule, strict=True)):
        setattr(
            model,
            name,
            FactorizedWidePoleMemoryStage(
                CONTENT_MODES,
                poles,
                stage_index=index,
                maximum_phase=canonical8.MAXIMUM_PHASES[index],
                frequency_scale=calibrated.FREQUENCY_SCALES[index],
                damping_scale=calibrated.DAMPING_SCALES[index],
                terminal=name == "terminal",
                scan_memory_policy="recompute",
                damping_min=config.damping_min,
                damping_max=config.damping_max,
            ),
        )
    _assert_model(model, schedule)
    return model


def _variant_config(variant: str) -> dict[str, Any]:
    schedule = POLE_SCHEDULES[variant]
    return {
        "backbone": {
            "name": variant,
            "stem": "3-to-32-to-192 Conv-LN-GELU; residual real 192 mixer; RMSNorm",
            "analysis": "orthogonal real 192-to-64, split into complex E32",
            "content_modes": CONTENT_MODES,
            "pole_schedule": list(schedule),
            "memory_axes": ["D4 direction", "32 content", "stage pole bank"],
            "scan": (
                "one optimized coarse4 D4 product scan over the flattened R-by-32 axis; "
                "odd terminal uses optimized bidirectional product scans"
            ),
            "scan_memory_policy": "recompute",
            "activation_checkpoint": (
                "whole-stage recomputation only; retain E32/Q384 between stages and "
                "avoid nested memory/PG replay"
            ),
            "pole_workspace_chunk": 32,
            "memory_processing": {
                "content_pg": (
                    "32-to-64-to-32 shared across direction and pole; exact pole-axis "
                    "workspace tiling with optimized fused kernels"
                ),
                "pole_pg": "R-to-32-to-R with fixed hidden workspace 16",
                "path_pg": (
                    "4-to-16-to-4 shared across content and pole; exact pole-axis "
                    "workspace tiling with optimized fused kernels"
                ),
            },
            "next_excitation": (
                "strict shared direction 4-to-1, content-specific strict pole R-to-1, "
                "content-wise real S2D carry, PostPGv2 H32"
            ),
            "descriptor": "raw D4 then content-specific strict pole R-to-3; Q384/stage",
            "descriptor_dim": 1536,
            "persistent_sidecar": False,
        },
        "head": {
            "descriptor_dim": 1536,
            "classifier": "exact matched H96 control head",
        },
    }


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
    for index, name in enumerate(STAGE_NAMES, start=1):
        stage = getattr(model, name)
        if not isinstance(stage, FactorizedWidePoleMemoryStage):
            continue
        for metric, value in stage.diagnostic_metrics().items():
            metrics[f"wide_memory/stage{index}/{metric}"] = value
        blocks: tuple[tuple[str, PhaseGatedComplexFFN | None], ...] = (
            ("content", stage.content_pg),
            ("pole", stage.pole_pg),
            ("path", stage.path_pg),
            ("post", stage.post_pg),
        )
        for axis, block in blocks:
            if block is None:
                continue
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
    payload["schema"] = "lnet.a2d.factorized_wide_pole_memory.imagenet100.v1"
    payload["evidence_status"] = "two-arm one-seed factorized wide-memory experiment"
    payload["variants"] = list(VARIANTS)
    payload["seeds"] = list(SEEDS)
    payload["variant_configs"] = {variant: _variant_config(variant) for variant in VARIANTS}
    payload["parameter_counts"] = {
        variant: sum(parameter.numel() for parameter in model.parameters())
        for variant, model in models.items()
    }
    payload["architecture"] = {
        VARIANT_BALANCED: (
            "Compact complex E32 feeds shared pole banks 8/16/32/64. Each coarse memory "
            "keeps D4, content, and dynamics axes factorized through Content/Pole/Path PG, "
            "structured readout, S2D carry, PostPG H32, and Q384 per stage."
        ),
        VARIANT_ISOSCAN: (
            "The same E32 factorized architecture with an 8/32/128/256 pole schedule, "
            "trading lower spatial area for wider shared dynamical memory while preserving "
            "the exact same Q1536 classifier interface."
        ),
    }
    payload["recipe"] = deepcopy(payload["recipe"])
    payload["recipe"]["kernel"] = (
        "optimized coarse4 state-plus-stop-gradient-variance Triton recurrence with "
        "recomputed scan intermediates"
    )
    digest = ramp.heads.harness._digest
    payload["source_sha256"]["pac_factorized_wide_pole_memory"] = digest(
        Path("src/lnet/pac_factorized_wide_pole_memory.py")
    )
    payload["source_sha256"]["factorized_wide_pole_runner"] = digest(Path(__file__))
    for source_name in (
        "pac_phase_gated_cffn",
        "pac_triton_phase_gated_cffn_fused",
        "pac_reduction_tiling",
        "pac_triton_hardware",
        "pac_triton_complex_rmsnorm",
        "pac_triton_phase_gate_linear",
        "pac_triton_phase_gate_linear_fused",
        "pac_triton_phase_gate_residual_fused",
        "pac_triton_rmsnorm_linear_fused",
    ):
        payload["source_sha256"][source_name] = digest(Path(f"src/lnet/{source_name}.py"))
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
