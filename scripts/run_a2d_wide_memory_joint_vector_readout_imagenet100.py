#!/usr/bin/env python3
"""Train the ALPHABET wide-memory joint-vector-readout experiment."""

# pyright: reportAny=false, reportExplicitAny=false
# pyright: reportImplicitRelativeImport=false, reportPrivateLocalImportUsage=false
# pyright: reportPrivateUsage=false
# ruff: noqa: C901, SLF001

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

from lnet.pac_wide_memory_joint_vector_readout import (
    RankTwoSeparableReadout,
    WideMemoryJointVectorReadoutStage,
)

if TYPE_CHECKING:
    from argparse import Namespace

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig
    from lnet.pac_phase_gated_cffn import PhaseGatedComplexFFN


VARIANT = "E32-64-128-256-Rank2SepReadout-R8-8-16-16-PackedQ4-NoCkpt"
VARIANTS = (VARIANT,)
SEEDS = (501,)
EXCITATION_SCHEDULE = (32, 64, 128, 256)
POLE_SCHEDULE = (8, 8, 16, 16)
NEXT_EXCITATIONS = (64, 128, 256, None)
STEM_REAL_WIDTH = 192
DESCRIPTOR_MODES = 96
STAGE_NAMES = ("stage1", "stage2", "stage3", "terminal")


def _replace_analysis(model: ComplexScanBackbone) -> None:
    if (
        model.analysis is None
        or model.analysis.in_features != STEM_REAL_WIDTH
        or model.analysis.out_features != STEM_REAL_WIDTH
    ):
        message = "joint-readout experiment requires the retained 192-wide stem interface"
        raise TypeError(message)
    analysis = nn.Linear(STEM_REAL_WIDTH, 2 * EXCITATION_SCHEDULE[0], bias=False)
    nn.init.orthogonal_(analysis.weight)
    orthogonal(
        analysis,
        "weight",
        orthogonal_map="matrix_exp",
        use_trivialization=True,
    )
    model.analysis = analysis


def _assert_model(model: ComplexScanBackbone) -> None:
    stemres = control.base.base.control.control.stemres
    if not isinstance(model.stem, stemres.ModeScaledTwoConvStem):
        message = "joint-readout model lost the 3-to-32-to-192 stem"
        raise TypeError(message)
    if model.stem.output_width != STEM_REAL_WIDTH:
        message = "joint-readout model changed stem spatial capacity"
        raise RuntimeError(message)
    if (
        model.analysis is None
        or model.analysis.in_features != STEM_REAL_WIDTH
        or model.analysis.out_features != 2 * EXCITATION_SCHEDULE[0]
        or model.analysis.bias is not None
        or not parametrize.is_parametrized(model.analysis, "weight")
    ):
        message = "joint-readout model requires orthogonal 192-to-64 analysis"
        raise TypeError(message)

    for index, (name, modes, poles, next_modes) in enumerate(
        zip(
            STAGE_NAMES,
            EXCITATION_SCHEDULE,
            POLE_SCHEDULE,
            NEXT_EXCITATIONS,
            strict=True,
        )
    ):
        stage = getattr(model, name)
        if not isinstance(stage, WideMemoryJointVectorReadoutStage):
            message = f"{name} is missing its joint-vector-readout stage"
            raise TypeError(message)
        if (
            stage.content_modes != modes
            or stage.poles != poles
            or stage.next_modes != next_modes
            or stage.output_modes != next_modes
            or stage.terminal != (name == "terminal")
            or stage.content_pg is not None
            or stage.pole_pg.hidden_modes != 16
            or stage.path_pg.hidden_modes != 4
            or stage.stage_index != index
            or stage.descriptor_pole_readout.input_modes != poles
            or stage.descriptor_pole_readout.output_modes != 4
            or stage.descriptor_readout.input_modes != modes * 4
            or stage.descriptor_readout.output_modes != DESCRIPTOR_MODES
        ):
            message = f"{name} changed the requested joint-readout axis contract"
            raise RuntimeError(message)
        if next_modes is None:
            if (
                stage.joint_readout is not None
                or stage.carry_projection is not None
                or stage.post_pg is not None
            ):
                message = "terminal stage unexpectedly constructs a next excitation"
                raise RuntimeError(message)
            continue
        if (
            not isinstance(stage.joint_readout, RankTwoSeparableReadout)
            or stage.joint_readout.input_modes != modes
            or stage.joint_readout.poles != poles
            or stage.joint_readout.output_modes != next_modes
            or stage.joint_readout.rank != 2
            or stage.joint_readout.direction_modes != next_modes // 4
            or stage.carry_projection is None
            or stage.carry_projection.input_modes != modes
            or stage.carry_projection.output_modes != next_modes
            or stage.post_pg is None
            or stage.post_pg.modes != next_modes
            or stage.post_pg.hidden_modes != next_modes
        ):
            message = f"{name} changed its strict vector re-encoding contract"
            raise RuntimeError(message)
    if model.descriptor_dim != 4 * len(STAGE_NAMES) * DESCRIPTOR_MODES:
        message = "joint-readout model changed the matched Q1536 head interface"
        raise RuntimeError(message)


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    if variant != VARIANT:
        message = f"unsupported joint-vector-readout variant: {variant}"
        raise ValueError(message)
    control.base.base._configure_ramp()
    model = control._build(control.VARIANT, config)
    control._assert_model(model)
    _replace_analysis(model)
    for index, (name, modes, poles, next_modes) in enumerate(
        zip(
            STAGE_NAMES,
            EXCITATION_SCHEDULE,
            POLE_SCHEDULE,
            NEXT_EXCITATIONS,
            strict=True,
        )
    ):
        setattr(
            model,
            name,
            WideMemoryJointVectorReadoutStage(
                modes,
                poles,
                next_modes=next_modes,
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
    _assert_model(model)
    return model


def _variant_config() -> dict[str, Any]:
    return {
        "backbone": {
            "name": VARIANT,
            "stem": "3-to-32-to-192 Conv-LN-GELU; residual real 192 mixer; RMSNorm",
            "analysis": "orthogonal real 192-to-64, split into complex E32",
            "excitation_schedule": list(EXCITATION_SCHEDULE),
            "pole_schedule": list(POLE_SCHEDULE),
            "memory_axes": ["D4 direction", "stage content", "stage pole bank"],
            "scan": "optimized coarse4 D4 product scan; optimized bidirectional terminal scan",
            "scan_memory_policy": "recompute",
            "activation_checkpoint": "disabled; retain all stage activations",
            "memory_processing": {
                "content_pg": None,
                "pole_pg": "R-to-32-to-R with fixed hidden workspace 16",
                "path_pg": "4-to-8-to-4 residual PG shared across content and poles",
            },
            "separable_readout": [
                "rank-2: raw D4; shared strict 32-to-(2x16); output pole 8-to-1",
                "rank-2: raw D4; shared strict 64-to-(2x32); output pole 8-to-1",
                "rank-2: raw D4; shared strict 128-to-(2x64); output pole 16-to-1",
            ],
            "carry": "mode-wise average S2D then packed strict K-to-K_next ComplexLinear",
            "post_pg": "PGv2 H64/H128/H256 with gamma=0.01",
            "descriptor": (
                "raw D4 then packed strict shared R-to-4 and "
                "packed strict (K*4)-to-96; Q384/stage"
            ),
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
        if not isinstance(stage, WideMemoryJointVectorReadoutStage):
            continue
        for metric, value in stage.diagnostic_metrics().items():
            metrics[f"joint_memory/stage{index}/{metric}"] = value
        blocks: tuple[tuple[str, PhaseGatedComplexFFN | None], ...] = (
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
    model = _build(VARIANT, config)
    payload["schema"] = "lnet.a2d.rank2_separable_wide_memory_readout.imagenet100.v1"
    payload["evidence_status"] = "one-arm one-seed rank-two separable-readout experiment"
    payload["variants"] = list(VARIANTS)
    payload["seeds"] = list(SEEDS)
    payload["variant_configs"] = {VARIANT: _variant_config()}
    payload["parameter_counts"] = {
        VARIANT: sum(parameter.numel() for parameter in model.parameters())
    }
    payload["architecture"] = {
        VARIANT: (
            "Complex excitations widen 32/64/128/256 while shared pole banks remain "
            "8/8/16/16. Pole and Path PG preserve structured memory before normalized "
            "raw directional states, rank-two shared-content/output-pole re-encoding, projected "
            "S2D carry, PostPG, and Q384/stage."
        )
    }
    payload["recipe"] = deepcopy(payload["recipe"])
    payload["recipe"]["kernel"] = (
        "optimized coarse4 product scan, optimized bidirectional terminal scan, latest "
        "packed/fused Phase-Gated CFFN kernels, packed shared-content projections, "
        "and compiled output-dependent pole pooling"
    )
    digest = ramp.heads.harness._digest
    payload["source_sha256"]["pac_wide_memory_joint_vector_readout"] = digest(
        Path("src/lnet/pac_wide_memory_joint_vector_readout.py")
    )
    payload["source_sha256"]["joint_vector_readout_runner"] = digest(Path(__file__))
    for source_name in (
        "pac_factorized_wide_pole_memory",
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
