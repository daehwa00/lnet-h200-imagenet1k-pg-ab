#!/usr/bin/env python3
"""Train the vector-input 2D pole-scan ImageNet-100 experiment."""

from __future__ import annotations

# pyright: reportAny=false, reportExplicitAny=false
# pyright: reportImplicitRelativeImport=false, reportPrivateLocalImportUsage=false
# pyright: reportPrivateUsage=false
# ruff: noqa: SLF001
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import run_a2d_wide_memory_joint_vector_readout_imagenet100 as joint
import torch
from torch import nn
from torch.nn.utils import parametrize
from torch.nn.utils.parametrizations import orthogonal

from lnet.pac_vector_input_pole_memory import VectorInputPoleMemoryStage

if TYPE_CHECKING:
    from argparse import Namespace

    import numpy as np
    from torch.utils.data import DataLoader

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig
    from lnet.pac_phase_gated_cffn import PhaseGatedComplexFFN
    from lnet.pac_product_scan_pipeline import ScanMemoryPolicy

VARIANT = "VecPole-D64-64-128-128-P16-32-64-64"
VARIANTS = (VARIANT,)
SEEDS = (501,)
EXCITATION_SCHEDULE = (64, 64, 128, 128)
POLE_SCHEDULE = (16, 32, 64, 64)
NEXT_EXCITATIONS = (64, 128, 128, None)
STAGE_NAMES = ("stage1", "stage2", "stage3", "terminal")
STEM_REAL_WIDTH = 192
DESCRIPTOR_MODES = 96
SCAN_MEMORY_POLICY: ScanMemoryPolicy = "retain"


def _replace_analysis(model: ComplexScanBackbone) -> None:
    if (
        model.analysis is None
        or model.analysis.in_features != STEM_REAL_WIDTH
    ):
        message = "compact vector-pole model requires the retained 192-wide stem interface"
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


def _assert_model(model: ComplexScanBackbone, *, memory_policy: ScanMemoryPolicy) -> None:
    if (
        model.analysis is None
        or model.analysis.in_features != STEM_REAL_WIDTH
        or model.analysis.out_features != 2 * EXCITATION_SCHEDULE[0]
        or model.analysis.bias is not None
        or not parametrize.is_parametrized(model.analysis, "weight")
    ):
        message = "compact vector-pole model requires orthogonal 192-to-128 analysis"
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
        if not isinstance(stage, VectorInputPoleMemoryStage):
            message = f"{name} is missing its vector-input pole stage"
            raise TypeError(message)
        if (
            stage.content_modes != modes
            or stage.poles != poles
            or stage.next_modes != next_modes
            or stage.output_modes != next_modes
            or stage.terminal != (name == "terminal")
            or stage.scan_memory_policy != memory_policy
            or stage.pole_input.input_modes != modes
            or stage.pole_input.output_modes != poles
            or stage.pole_pg.modes != poles
            or stage.pole_pg.hidden_modes != 16
            or stage.path_pg.modes != 4
            or stage.path_pg.hidden_modes != 4
            or stage.descriptor_projection.input_modes != poles
            or stage.descriptor_projection.output_modes != DESCRIPTOR_MODES
            or stage.stage_index != index
        ):
            message = f"{name} changed the vector-input pole contract"
            raise RuntimeError(message)
        if next_modes is None:
            if stage.direction_projection is not None:
                message = "terminal vector-input pole stage constructs a next excitation"
                raise RuntimeError(message)
            continue
        if (
            stage.direction_projection is None
            or stage.direction_projection.input_modes != poles
            or stage.direction_projection.output_modes != next_modes // 4
        ):
            message = f"{name} changed its shared directional readout contract"
            raise RuntimeError(message)
    if model.descriptor_dim != 4 * len(STAGE_NAMES) * DESCRIPTOR_MODES:
        message = "vector-input pole model changed the matched Q1536 head interface"
        raise RuntimeError(message)


def _build_with_policy(
    variant: str,
    config: ComplexScanConfig,
    memory_policy: ScanMemoryPolicy,
) -> ComplexScanBackbone:
    if variant != VARIANT:
        message = f"unsupported vector-input pole variant: {variant}"
        raise ValueError(message)
    model = joint._build(joint.VARIANT, config)
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
            VectorInputPoleMemoryStage(
                modes,
                poles,
                next_modes=next_modes,
                stage_index=index,
                maximum_phase=joint.canonical8.MAXIMUM_PHASES[index],
                frequency_scale=joint.calibrated.FREQUENCY_SCALES[index],
                damping_scale=joint.calibrated.DAMPING_SCALES[index],
                terminal=name == "terminal",
                scan_memory_policy=memory_policy,
                damping_min=config.damping_min,
                damping_max=config.damping_max,
            ),
        )
    _assert_model(model, memory_policy=memory_policy)
    return model


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    return _build_with_policy(variant, config, SCAN_MEMORY_POLICY)


def _variant_config() -> dict[str, Any]:
    return {
        "backbone": {
            "name": VARIANT,
            "excitation_schedule": list(EXCITATION_SCHEDULE),
            "pole_schedule": list(POLE_SCHEDULE),
            "scan_memory_policy": SCAN_MEMORY_POLICY,
            "pole_input": [
                "packed strict 64-to-16",
                "packed strict 64-to-32",
                "packed strict 128-to-64",
                "packed strict 128-to-64",
            ],
            "scan": "P independent learned 2D poles; optimized D4 product scan",
            "memory_processing": "Pole PG P-to-32-to-P; fused Path PG 4-to-8-to-4; raw D4",
            "directional_readout": [
                "shared packed strict 16-to-16 per direction; concat to E64",
                "shared packed strict 32-to-32 per direction; concat to E128",
                "shared packed strict 64-to-32 per direction; concat to E128",
            ],
            "transition": "direct concat of four shared pole readouts into the next dim",
            "carry": "absent",
            "post_pg": "absent",
            "descriptor": "shared packed strict P-to-96 per raw direction; Q384/stage",
            "descriptor_dim": 1536,
            "activation_checkpoint": "disabled; compact P-state activations retained",
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
        if not isinstance(stage, VectorInputPoleMemoryStage):
            continue
        for metric, value in stage.diagnostic_metrics().items():
            metrics[f"vector_pole/stage{index}/{metric}"] = value
        blocks: tuple[tuple[str, PhaseGatedComplexFFN], ...] = (
            ("pole", stage.pole_pg),
            ("path", stage.path_pg),
        )
        for axis, block in blocks:
            for metric, value in block.diagnostic_metrics().items():
                metrics[f"phase_gated/stage{index}/{axis}_{metric}"] = value
    metrics.update(getattr(model, "_vector_pole_gradient_metrics", {}))
    return metrics


def _capture_pg_gradient_metrics(model: nn.Module) -> None:
    """Copy graph-owned gradient statistics before validation replays the graph."""
    metrics: dict[str, float] = {}
    for index, name in enumerate(STAGE_NAMES, start=1):
        stage = getattr(model, name)
        if not isinstance(stage, VectorInputPoleMemoryStage):
            continue
        blocks: tuple[tuple[str, PhaseGatedComplexFFN], ...] = (
            ("pole", stage.pole_pg),
            ("path", stage.path_pg),
        )
        for axis, block in blocks:
            for metric, value in block.gradient_metrics().items():
                metrics[f"phase_gated/stage{index}/{axis}_{metric}"] = value
    cast("Any", model)._vector_pole_gradient_metrics = metrics


def _train_epoch(
    model: nn.Module,
    runtime: nn.Module,
    loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    mixup_generator: np.random.Generator,
    mixup_alpha: float,
    precision: str,
    gradient_accumulation_steps: int = 1,
    channels_last: bool = False,
) -> dict[str, float]:
    source = joint.control.base.base.control.control.stemres.uniform.base
    result = source.canonical8.fair_init.backbone.deep4.baseline.baseline.structured._train_epoch(
        model,
        runtime,
        loader,
        optimizer,
        device=device,
        mixup_generator=mixup_generator,
        mixup_alpha=mixup_alpha,
        precision=precision,
        gradient_accumulation_steps=gradient_accumulation_steps,
        channels_last=channels_last,
    )
    _capture_pg_gradient_metrics(model)
    return result


def _contract(args: Namespace) -> dict[str, Any]:
    payload = joint._contract(args)
    ramp = joint.control.base.base.control.control.stemres.uniform.base
    config = ramp.PoleModelConfig(output_dim=100, stem_strides=(2, 2))
    model = _build(VARIANT, config)
    payload["schema"] = "lnet.a2d.vector_input_pole_scan.imagenet100.v3"
    payload["evidence_status"] = "compact vector-input pole-scan experiment"
    payload["variants"] = list(VARIANTS)
    payload["seeds"] = list(SEEDS)
    payload["variant_configs"] = {VARIANT: _variant_config()}
    payload["parameter_counts"] = {
        VARIANT: sum(parameter.numel() for parameter in model.parameters())
    }
    payload["architecture"] = {
        VARIANT: (
            "Complex excitation widths 64/64/128/128 project into 16/32/64/64 learned "
            "pole drives before the optimized D4 scan. Pole/Path PG process those memories, "
            "then four shared raw-direction readouts concatenate directly without Carry or PostPG."
        )
    }
    payload["recipe"] = deepcopy(payload["recipe"])
    payload["recipe"]["kernel"] = (
        "packed complex pole-input/readout GEMMs, optimized associative D4 scan, "
        "and automatic packed/fused Phase-Gated kernels"
    )
    digest = ramp.heads.harness._digest
    payload["source_sha256"]["pac_vector_input_pole_memory"] = digest(
        Path("src/lnet/pac_vector_input_pole_memory.py")
    )
    payload["source_sha256"]["vector_input_pole_runner"] = digest(Path(__file__))
    payload["source_sha256"]["pac_complex_layers"] = digest(
        Path("src/lnet/pac_complex_layers.py")
    )
    return payload


def main() -> None:
    joint.control.base.base._configure_ramp()
    ramp = joint.control.base.base.control.control.stemres.uniform.base
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
            train_epoch=_train_epoch,
            evaluate=source.heads._evaluate,
            wandb_model_metrics=_wandb_model_metrics,
            summarize=source.heads._summarize,
        )
    )


if __name__ == "__main__":
    main()
