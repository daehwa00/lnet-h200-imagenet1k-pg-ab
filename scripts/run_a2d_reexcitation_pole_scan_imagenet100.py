#!/usr/bin/env python3
"""Train normalized pole-memory to local re-excitation on ImageNet-100."""

from __future__ import annotations

# pyright: reportAny=false, reportExplicitAny=false
# pyright: reportImplicitRelativeImport=false, reportPrivateLocalImportUsage=false
# pyright: reportPrivateUsage=false
# ruff: noqa: SLF001
import os
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import run_a2d_affine_qhead_imagenet100 as head_runner
import run_a2d_resaux1_deephead_imagenet100 as deephead
import run_a2d_vector_input_pole_scan_imagenet100 as base
import torch
from torch import nn
from torch.nn.utils import parametrize

from lnet.complex_scan import ModalFusionHead
from lnet.image_layers import StandardizedAffineModalHead
from lnet.pac_phase_gated_cffn import PhaseGatedComplexFFN
from lnet.pac_reexcitation_pole_memory import ReexcitationPoleMemoryStage

if TYPE_CHECKING:
    from argparse import Namespace

    import numpy as np
    from torch.utils.data import DataLoader

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig
    from lnet.pac_product_scan_pipeline import ScanMemoryPolicy

VARIANT = "ReX-Q704-P16-32-64x2"
VARIANTS = (VARIANT,)
SEEDS = (501,)
EXCITATION_SCHEDULE = (64, 64, 128, 256)
POLE_SCHEDULE = (16, 32, 64, 64)
NEXT_EXCITATIONS = (64, 128, 256, None)
REEXCITATION_HIDDEN = (32, 64, 64, None)
STAGE_NAMES = ("stage1", "stage2", "stage3", "terminal")
DESCRIPTOR_DIM = 4 * sum(POLE_SCHEDULE)
SCAN_MEMORY_POLICY: ScanMemoryPolicy = "retain"
COMPILE_MODE = "reduce-overhead"


def _replace_head(model: ComplexScanBackbone, output_dim: int) -> None:
    fusion_source = ModalFusionHead(DESCRIPTOR_DIM, deephead.FIRST_WIDTH, output_dim)
    fusion = deephead.DeepModalFusionHead(fusion_source, output_dim)
    affine = StandardizedAffineModalHead(DESCRIPTOR_DIM, output_dim)
    model.descriptor_dim = DESCRIPTOR_DIM
    cast("Any", model).classifier = head_runner.A2DAffineQClassifier(
        DESCRIPTOR_DIM,
        output_dim,
        main="fusion",
        affine=affine,
        fusion=fusion,
        lrq=None,
        beta_lrq=None,
        affine_auxiliary_weight=0.5,
    )


def _assert_model(model: ComplexScanBackbone, *, memory_policy: ScanMemoryPolicy) -> None:
    if (
        model.analysis is None
        or model.analysis.in_features != base.STEM_REAL_WIDTH
        or model.analysis.out_features != 2 * EXCITATION_SCHEDULE[0]
        or not parametrize.is_parametrized(model.analysis, "weight")
    ):
        message = "re-excitation model changed its orthogonal stem interface"
        raise RuntimeError(message)
    for index, (name, modes, poles, next_modes, hidden) in enumerate(
        zip(
            STAGE_NAMES,
            EXCITATION_SCHEDULE,
            POLE_SCHEDULE,
            NEXT_EXCITATIONS,
            REEXCITATION_HIDDEN,
            strict=True,
        )
    ):
        stage = getattr(model, name)
        if not isinstance(stage, ReexcitationPoleMemoryStage):
            message = f"{name} is missing its re-excitation pole stage"
            raise TypeError(message)
        if (
            stage.content_modes != modes
            or stage.poles != poles
            or stage.next_modes != next_modes
            or stage.output_modes != next_modes
            or stage.reexcitation_hidden != hidden
            or stage.terminal != (index == 3)
            or stage.scan_memory_policy != memory_policy
            or stage.pole_input.input_modes != modes
            or stage.pole_input.output_modes != poles
            or stage.stage_index != index
        ):
            message = f"{name} changed the re-excitation stage contract"
            raise RuntimeError(message)
        if next_modes is None:
            if any(
                block is not None
                for block in (stage.memory_pg, stage.carry_projection, stage.reexcitation_pg)
            ):
                message = "terminal re-excitation stage contains transition operators"
                raise RuntimeError(message)
            continue
        if (
            not isinstance(stage.memory_pg, PhaseGatedComplexFFN)
            or stage.memory_pg.modes != poles
            or stage.memory_pg.hidden_modes != 16
            or not isinstance(stage.reexcitation_pg, PhaseGatedComplexFFN)
            or stage.reexcitation_pg.modes != next_modes
            or stage.reexcitation_pg.hidden_modes != hidden
            or (stage.carry_projection is None) != (modes == next_modes)
        ):
            message = f"{name} changed its memory/carry/re-excitation operators"
            raise RuntimeError(message)
    classifier = model.classifier
    if (
        model.descriptor_dim != DESCRIPTOR_DIM
        or not isinstance(classifier, head_runner.A2DAffineQClassifier)
        or not isinstance(classifier.affine, StandardizedAffineModalHead)
        or classifier.affine.linear.in_features != DESCRIPTOR_DIM
        or not isinstance(classifier.fusion, deephead.DeepModalFusionHead)
        or classifier.fusion.fusion.in_features != DESCRIPTOR_DIM
        or classifier.affine_auxiliary_weight != 0.5
    ):
        message = "re-excitation model changed its direct-Q704 head"
        raise RuntimeError(message)


def _build_with_policy(
    variant: str,
    config: ComplexScanConfig,
    memory_policy: ScanMemoryPolicy,
) -> ComplexScanBackbone:
    if variant != VARIANT:
        message = f"unsupported re-excitation pole variant: {variant}"
        raise ValueError(message)
    model = base.joint._build(base.joint.VARIANT, config)
    base._replace_analysis(model)
    for index, (name, modes, poles, next_modes, hidden) in enumerate(
        zip(
            STAGE_NAMES,
            EXCITATION_SCHEDULE,
            POLE_SCHEDULE,
            NEXT_EXCITATIONS,
            REEXCITATION_HIDDEN,
            strict=True,
        )
    ):
        setattr(
            model,
            name,
            ReexcitationPoleMemoryStage(
                modes,
                poles,
                next_modes=next_modes,
                reexcitation_hidden=hidden,
                stage_index=index,
                maximum_phase=base.joint.canonical8.MAXIMUM_PHASES[index],
                frequency_scale=base.joint.calibrated.FREQUENCY_SCALES[index],
                damping_scale=base.joint.calibrated.DAMPING_SCALES[index],
                terminal=index == 3,
                scan_memory_policy=memory_policy,
                damping_min=config.damping_min,
                damping_max=config.damping_max,
            ),
        )
    _replace_head(model, config.output_dim)
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
            "pole_input": "learned CRMSNorm followed by packed strict ComplexLinear K-to-P",
            "scan": "P independent learned 2D poles; optimized D4 product scan",
            "descriptor": "raw scan state and direct per-direction log-energy Q4P",
            "memory_processing": "Pole PG P-to-32-to-P; no Path PG",
            "memory_pack": "lossless direction-major reshape [4,P] to [4P]",
            "carry": "fixed 2x2 local mean then shared real Linear K-to-4P when widths differ",
            "reexcitation": [
                "PG 64-to-64-to-64 (u32+v32)",
                "PG 128-to-128-to-128 (u64+v64)",
                "PG 256-to-128-to-256 (u64+v64)",
            ],
            "terminal": "normalized 256-to-64 pole drive; scan and direct Q only",
            "descriptor_dim": DESCRIPTOR_DIM,
            "compile_mode": COMPILE_MODE,
        },
        "head": {
            "descriptor_dim": DESCRIPTOR_DIM,
            "classifier": "BatchNorm-Fusion384-GELU-RMSNorm-256-GELU-RMSNorm-100",
            "auxiliary": "BatchNorm(affine=False)-Linear704-to-100 at weight 0.5",
        },
    }


def _pg_blocks(
    stage: ReexcitationPoleMemoryStage,
) -> tuple[tuple[str, PhaseGatedComplexFFN], ...]:
    blocks: list[tuple[str, PhaseGatedComplexFFN]] = []
    if stage.memory_pg is not None:
        blocks.append(("pole", stage.memory_pg))
    if stage.reexcitation_pg is not None:
        blocks.append(("reexcitation", stage.reexcitation_pg))
    return tuple(blocks)


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
        if not isinstance(stage, ReexcitationPoleMemoryStage):
            continue
        for metric, value in stage.diagnostic_metrics().items():
            metrics[f"reexcitation_pole/stage{index}/{metric}"] = value
        for axis, block in _pg_blocks(stage):
            for metric, value in block.diagnostic_metrics().items():
                metrics[f"phase_gated/stage{index}/{axis}_{metric}"] = value
    metrics.update(getattr(model, "_reexcitation_gradient_metrics", {}))
    return metrics


def _capture_pg_gradient_metrics(model: nn.Module) -> None:
    metrics: dict[str, float] = {}
    for index, name in enumerate(STAGE_NAMES, start=1):
        stage = getattr(model, name)
        if not isinstance(stage, ReexcitationPoleMemoryStage):
            continue
        for axis, block in _pg_blocks(stage):
            for metric, value in block.gradient_metrics().items():
                metrics[f"phase_gated/stage{index}/{axis}_{metric}"] = value
    cast("Any", model)._reexcitation_gradient_metrics = metrics


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
    source = base.joint.control.base.base.control.control.stemres.uniform.base
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
    payload = base.joint._contract(args)
    ramp = base.joint.control.base.base.control.control.stemres.uniform.base
    config = ramp.PoleModelConfig(output_dim=100, stem_strides=(2, 2))
    model = _build(VARIANT, config)
    payload["schema"] = "lnet.a2d.reexcitation_pole_scan.imagenet100.v1"
    payload["evidence_status"] = "one-seed normalized memory-to-re-excitation experiment"
    payload["variants"] = list(VARIANTS)
    payload["seeds"] = list(SEEDS)
    payload["variant_configs"] = {VARIANT: _variant_config()}
    payload["parameter_counts"] = {
        VARIANT: sum(parameter.numel() for parameter in model.parameters())
    }
    payload["architecture"] = {
        VARIANT: (
            "Normalized K-to-P pole drives create raw D4 memories and direct Q704. "
            "Pole PG memories reshape without projection, merge with local S2D carry, "
            "and narrow PG re-excites the next vector."
        )
    }
    payload["recipe"] = deepcopy(payload["recipe"])
    payload["recipe"]["compile_mode"] = COMPILE_MODE
    payload["recipe"]["loader_workers"] = int(args.workers)
    payload["recipe"]["loader_persistent_workers"] = (
        args.workers > 0 and os.environ.get("LNET_PERSISTENT_WORKERS", "1") == "1"
    )
    payload["recipe"]["cpu_affinity"] = os.environ.get(
        "LNET_CPU_AFFINITY_ACTIVE",
        os.environ.get("LNET_CPU_AFFINITY"),
    )
    payload["recipe"]["kernel"] = (
        "packed strict pole/re-excitation GEMMs, optimized associative D4 scan, "
        "and automatic packed/fused Phase-Gated kernels"
    )
    digest = ramp.heads.harness._digest
    payload["source_sha256"]["pac_reexcitation_pole_memory"] = digest(
        Path("src/lnet/pac_reexcitation_pole_memory.py")
    )
    payload["source_sha256"]["reexcitation_pole_runner"] = digest(Path(__file__))
    return payload


def main() -> None:
    base.joint.control.base.base._configure_ramp()
    ramp = base.joint.control.base.base.control.control.stemres.uniform.base
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
