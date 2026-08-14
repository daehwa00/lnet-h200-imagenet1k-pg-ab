#!/usr/bin/env python3
"""Train pole-scaled PGv2 transitions around the vector-input pole scan."""

from __future__ import annotations

# pyright: reportAny=false, reportExplicitAny=false
# pyright: reportImplicitRelativeImport=false, reportPrivateLocalImportUsage=false
# pyright: reportPrivateUsage=false
# ruff: noqa: SLF001
import os
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import run_a2d_deep4_calibrated_uniform_p96_phase_gated_h96_imagenet100 as control
import run_a2d_deep4_calibrated_uniform_p96_stemres_imagenet100 as stemres
import torch
from torch import nn
from torch.nn.utils import parametrize
from torch.nn.utils.parametrizations import orthogonal

from lnet.pac_pgv2_vector_pole_stage import PGv2VectorPoleStage

if TYPE_CHECKING:
    from argparse import Namespace

    import numpy as np
    from torch.utils.data import DataLoader

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig
    from lnet.pac_phase_gated_cffn import PhaseGatedComplexFFN
    from lnet.pac_product_scan_pipeline import ScanMemoryPolicy

VARIANT = "PGv2-CanonicalQ-FullPole-K64-64-128-128"
VARIANTS = (VARIANT,)
SEEDS = (501,)
STAGE_NAMES = ("stage1", "stage2", "stage3", "terminal")
EXCITATION_SCHEDULE = (64, 64, 128, 128)
POLE_SCHEDULE = EXCITATION_SCHEDULE
NEXT_EXCITATIONS = (*EXCITATION_SCHEDULE[1:], None)
STEM_INTERFACE_WIDTH = 2 * EXCITATION_SCHEDULE[0]
PATH_HIDDEN = 8
SCAN_MEMORY_POLICY: ScanMemoryPolicy = "retain"


def _configure_ramp() -> None:
    control._configure_ramp()
    ramp = control.control.control.stemres.uniform.base
    ramp.VARIANT = VARIANT
    ramp.VARIANTS = VARIANTS
    ramp.SEEDS = SEEDS


def _replace_real_interface(model: ComplexScanBackbone) -> None:
    if not isinstance(model.precomplex_fc, stemres.ResidualPreComplexMixer):
        message = "PGv2 shared-FC model requires the established residual real mixer"
        raise TypeError(message)
    modes = EXCITATION_SCHEDULE[0]
    width = 2 * modes
    model.stem = stemres.ModeScaledTwoConvStem(
        modes,
        strides=model.config.stem_strides,
    )
    model.precomplex_fc = stemres.ResidualPreComplexMixer(
        nn.Sequential(
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.Identity(),
        )
    )
    model.input_norm = nn.RMSNorm(width)
    analysis = nn.Linear(width, width, bias=False)
    nn.init.orthogonal_(analysis.weight)
    orthogonal(
        analysis,
        "weight",
        orthogonal_map="matrix_exp",
        use_trivialization=True,
    )
    model.analysis = analysis
    model.config = replace(
        model.config,
        stem_width=width,
        modes=EXCITATION_SCHEDULE[:3],
        augmented_widths=EXCITATION_SCHEDULE[1:3],
        quadrant_path_mode_cffn_widths=POLE_SCHEDULE[:2],
    )


def _head_standardizers(model: nn.Module) -> tuple[nn.BatchNorm1d, ...]:
    classifier = getattr(model, "classifier", None)
    if not isinstance(classifier, nn.Module):
        message = "Direct-Q model is missing its classifier"
        raise TypeError(message)
    modules = tuple(
        module
        for name, module in classifier.named_modules()
        if name.endswith("standardizer")
    )
    if len(modules) != 2 or not all(isinstance(module, nn.BatchNorm1d) for module in modules):
        message = "Direct-Q model requires the established two affine-free BatchNorm heads"
        raise TypeError(message)
    return cast("tuple[nn.BatchNorm1d, ...]", modules)


def _stage(
    index: int,
    config: ComplexScanConfig,
    memory_policy: ScanMemoryPolicy,
) -> PGv2VectorPoleStage:
    ramp = control.control.control.stemres.uniform.base
    modes = EXCITATION_SCHEDULE[index]
    poles = POLE_SCHEDULE[index]
    next_modes = NEXT_EXCITATIONS[index]
    terminal = index == len(STAGE_NAMES) - 1
    return PGv2VectorPoleStage(
        modes,
        poles,
        next_modes=next_modes,
        path_hidden=PATH_HIDDEN,
        post_hidden=None if terminal else 2 * cast("int", next_modes),
        stage_index=index,
        maximum_phase=ramp.canonical8.MAXIMUM_PHASES[index],
        frequency_scale=ramp.calibrated.FREQUENCY_SCALES[index],
        damping_scale=ramp.calibrated.DAMPING_SCALES[index],
        terminal=terminal,
        scan_memory_policy=memory_policy,
        damping_min=config.damping_min,
        damping_max=config.damping_max,
    )


def _assert_model(
    model: ComplexScanBackbone,
    *,
    memory_policy: ScanMemoryPolicy,
) -> None:
    if (
        not isinstance(model.stem, stemres.ModeScaledTwoConvStem)
        or model.stem.output_width != STEM_INTERFACE_WIDTH
        or not isinstance(model.precomplex_fc, stemres.ResidualPreComplexMixer)
        or model.precomplex_fc.width != STEM_INTERFACE_WIDTH
        or not isinstance(model.input_norm, nn.RMSNorm)
        or model.input_norm.normalized_shape != (STEM_INTERFACE_WIDTH,)
        or model.analysis is None
        or model.analysis.in_features != STEM_INTERFACE_WIDTH
        or model.analysis.out_features != STEM_INTERFACE_WIDTH
        or not parametrize.is_parametrized(model.analysis, "weight")
        or model.config.stem_width != STEM_INTERFACE_WIDTH
        or model.descriptor_dim != 4 * sum(POLE_SCHEDULE)
    ):
        message = "PGv2 vector-pole model changed its stem or Q1536 interface"
        raise RuntimeError(message)
    if any(module.affine for module in _head_standardizers(model)):
        message = "Direct-Q model requires affine-free BatchNorm heads"
        raise RuntimeError(message)
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
        terminal = index == len(STAGE_NAMES) - 1
        if (
            not isinstance(stage, PGv2VectorPoleStage)
            or stage.content_modes != modes
            or stage.poles != poles
            or stage.next_modes != next_modes
            or stage.output_modes != next_modes
            or stage.terminal != terminal
            or stage.scan_memory_policy != memory_policy
            or stage.pole_input.in_features != modes
            or stage.pole_input.out_features != poles
            or stage.pole_input.bias is not None
            or not parametrize.is_parametrized(stage.pole_input, "weight")
        ):
            message = f"{name} changed the vector-input scan contract"
            raise RuntimeError(message)
        if terminal:
            if any(
                block is not None
                for block in (
                    stage.transition,
                    stage.memory_adapter,
                    stage.carry_projection,
                    stage.post_norm,
                    stage.post_input,
                    stage.post_output,
                    stage.post_scale,
                )
            ):
                message = "terminal PGv2 vector-pole stage contains transition operators"
                raise RuntimeError(message)
            continue
        if (
            stage.transition is None
            or stage.transition.mode.hidden_modes != poles
            or stage.transition.path_input.input_paths != 4
            or stage.transition.path_input.output_paths != PATH_HIDDEN
            or stage.transition.path_output.input_paths != PATH_HIDDEN
            or stage.transition.path_output.output_paths != 1
            or stage.memory_adapter is None
            or stage.memory_adapter.input_modes != poles
            or stage.memory_adapter.output_modes != next_modes
            or stage.post_hidden != 2 * cast("int", next_modes)
            or stage.post_scale is None
            or tuple(stage.post_scale.shape) != (next_modes,)
        ):
            message = f"{name} changed the pole-scaled PGv2 transition contract"
            raise RuntimeError(message)


def _build_with_policy(
    variant: str,
    config: ComplexScanConfig,
    memory_policy: ScanMemoryPolicy,
) -> ComplexScanBackbone:
    if variant != VARIANT:
        message = f"unsupported PGv2 vector-pole variant: {variant}"
        raise ValueError(message)
    model = control._build(control.VARIANT, config)
    # Preserve the control's later shuffle/augmentation stream while giving
    # the replacement architecture deterministic learned initialization.
    with torch.random.fork_rng(devices=[]):
        _replace_real_interface(model)
        for index, name in enumerate(STAGE_NAMES):
            setattr(model, name, _stage(index, config, memory_policy))
    _configure_ramp()
    _assert_model(model, memory_policy=memory_policy)
    return model


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    return _build_with_policy(variant, config, SCAN_MEMORY_POLICY)


def _transition_modal_parameters(model: nn.Module) -> tuple[nn.Parameter, ...]:
    selected: dict[int, nn.Parameter] = {}
    for name in STAGE_NAMES:
        stage = getattr(model, name)
        if not isinstance(stage, PGv2VectorPoleStage):
            continue
        for module in (stage.pole_input, stage.memory_adapter, stage.carry_projection):
            if module is None:
                continue
            selected.update((id(parameter), parameter) for parameter in module.parameters())
        if stage.carry_logits is not None:
            selected[id(stage.carry_logits)] = stage.carry_logits
    return tuple(selected.values())


def _build_optimizer(model: nn.Module, recipe: dict[str, Any]) -> torch.optim.Optimizer:
    """Keep new basis/adapter parameters on the established modal schedule."""
    source = control.control.control.stemres.uniform.base.backbone.a2d_base.residuals
    optimizer = source.optimizer_source._build_optimizer(model, recipe)
    modal = _transition_modal_parameters(model)
    modal_ids = {id(parameter) for parameter in modal}
    removed = 0
    for group in optimizer.param_groups:
        retained = [parameter for parameter in group["params"] if id(parameter) not in modal_ids]
        removed += len(group["params"]) - len(retained)
        group["params"] = retained
    if removed != len(modal):
        message = "vector-pole modal optimizer ownership is incomplete"
        raise RuntimeError(message)
    optimizer.add_param_group(
        {
            "name": "vector_pole_modal",
            "params": modal,
            "lr": float(recipe["learning_rate"])
            * float(recipe["modal_learning_rate_multiplier"]),
            "weight_decay": 0.0,
        }
    )
    return optimizer


def _pg_blocks(stage: PGv2VectorPoleStage) -> tuple[tuple[str, PhaseGatedComplexFFN], ...]:
    if stage.transition is None:
        return ()
    return (("mode", stage.transition.mode),)


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
        if not isinstance(stage, PGv2VectorPoleStage):
            continue
        for metric, value in stage.diagnostic_metrics().items():
            metrics[f"pgv2_vector_pole/stage{index}/{metric}"] = value
        for axis, block in _pg_blocks(stage):
            for metric, value in block.diagnostic_metrics().items():
                metrics[f"phase_gated/stage{index}/{axis}_{metric}"] = value
    metrics.update(getattr(model, "_pgv2_vector_pole_gradient_metrics", {}))
    return metrics


def _capture_pg_gradient_metrics(model: nn.Module) -> None:
    metrics: dict[str, float] = {}
    for index, name in enumerate(STAGE_NAMES, start=1):
        stage = getattr(model, name)
        if not isinstance(stage, PGv2VectorPoleStage):
            continue
        for axis, block in _pg_blocks(stage):
            for metric, value in block.gradient_metrics().items():
                metrics[f"phase_gated/stage{index}/{axis}_{metric}"] = value
    cast("Any", model)._pgv2_vector_pole_gradient_metrics = metrics


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
    source = control.control.control.stemres.uniform.base
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


def _variant_config() -> dict[str, Any]:
    payload = deepcopy(control._variant_config())
    backbone = payload["backbone"]
    for stale_key in ("base_poles", "radial_levels", "canonical_orientations"):
        backbone.pop(stale_key, None)
    backbone.update(
        {
            "name": VARIANT,
            "excitation_schedule": list(EXCITATION_SCHEDULE),
            "pole_schedule": list(POLE_SCHEDULE),
            "augmented_widths": list(EXCITATION_SCHEDULE[1:]),
            "mode_cffn_widths": list(POLE_SCHEDULE[:-1]),
            "post_ffn_widths": [2 * width for width in EXCITATION_SCHEDULE[1:]],
            "stem_width": STEM_INTERFACE_WIDTH,
            "stem": {
                "rule": "real interface width equals 2*K1",
                "convolutions": "3-to-32 stride2 then 32-to-2K1 stride2",
                "normalization": "LayerNorm2d and GELU after each convolution",
                "precomplex_mixer": "residual Linear(2K1)-GELU-Linear(2K1)",
                "interface_norm": "RMSNorm(2K1)",
                "complex_projection": "orthogonal Linear(2K1,2K1) then real/imag split",
            },
            "pole_input": (
                "trainably semi-orthogonal bias-free real Linear K-to-P shared by real/imag; "
                "no pre-scan normalization"
            ),
            "scan": "P learned 2D poles with optimized associative D4 product scan",
            "descriptor": (
                "exact full-grid raw D4 per-pole energy returned by the fused scan; "
                "no endpoint recomputation or learned Q projection"
            ),
            "head_standardizer": "unchanged affine-free BatchNorm1d control",
            "stage_transition": (
                "PGv2 Mode PG P-to-2P-to-P (uP+vP), GWL path 4-to-8-to-1, "
                "modal-LR P-to-K_next adapter, simplex S2D carry, fixed-norm H=2K "
                "PostFusion with learned 0.1 residual scale"
            ),
        }
    )
    return payload


def _contract(args: Namespace) -> dict[str, Any]:
    payload = control._contract(args)
    ramp = control.control.control.stemres.uniform.base
    config = ramp.PoleModelConfig(output_dim=100, stem_strides=(2, 2))
    model = _build(VARIANT, config)
    payload["schema"] = "lnet.a2d.pgv2.canonical_q_full_pole.imagenet100.v1"
    payload["evidence_status"] = "Untrained canonical full-grid Q correction"
    payload["variants"] = list(VARIANTS)
    payload["seeds"] = list(SEEDS)
    payload["variant_configs"] = {VARIANT: _variant_config()}
    payload["parameter_counts"] = {
        VARIANT: sum(parameter.numel() for parameter in model.parameters())
    }
    payload["architecture"] = {
        VARIANT: (
            "The real stem interface is 2*K1. A trainably semi-orthogonal real-shared "
            "K-to-K map feeds raw excitation directly to the pole scan. Non-terminal Q is "
            "the exact full-grid raw directional descriptor returned by that same fused scan, not "
            "an endpoint reconstruction. Mode PG uses H=P; S2D carry stays on a learned "
            "simplex; PostFusion uses the established learned 0.1 residual scale. The "
            "existing affine-free BatchNorm heads, PG math, optimizer schedule, and common "
            "training recipe are retained."
        )
    }
    payload["recipe"] = deepcopy(payload["recipe"])
    payload["recipe"]["loader_workers"] = int(args.workers)
    payload["recipe"]["loader_persistent_workers"] = (
        args.workers > 0 and os.environ.get("LNET_PERSISTENT_WORKERS", "1") == "1"
    )
    digest = ramp.heads.harness._digest
    payload["source_sha256"]["pgv2_vector_pole_stage"] = digest(
        Path("src/lnet/pac_pgv2_vector_pole_stage.py")
    )
    payload["source_sha256"]["pgv2_vector_pole_runner"] = digest(Path(__file__))
    return payload


def main() -> None:
    _configure_ramp()
    ramp = control.control.control.stemres.uniform.base
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
            train_epoch=_train_epoch,
            evaluate=source.heads._evaluate,
            wandb_model_metrics=_wandb_model_metrics,
            summarize=source.heads._summarize,
        )
    )


if __name__ == "__main__":
    main()
