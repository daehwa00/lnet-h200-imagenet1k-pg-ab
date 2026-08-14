#!/usr/bin/env python3
"""Train the declarative PGv2-H192 full-state overnight architecture family."""

# pyright: reportArgumentType=false, reportExplicitAny=false
# pyright: reportImplicitRelativeImport=false, reportPrivateLocalImportUsage=false
# pyright: reportPrivateUsage=false
# ruff: noqa: SLF001

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import run_a2d_deep4_calibrated_uniform_p96_phase_gated_imagenet100 as base
import torch
from torch import Tensor, nn

from lnet.pac_factorized_stage_transition import FactorizedS2DPostFusionTransition
from lnet.pac_full_state_operators import GroupedPhaseGatedComplexFFN
from lnet.pac_full_state_overnight import (
    BASE_MODEL,
    EXPERIMENT_SPECS,
    SPECS_BY_VARIANT,
    FullStateExperimentSpec,
    StructuredFullStateTransition,
    experiment_manifest,
)
from lnet.pac_phase_gated_cffn import PhaseGatedComplexFFN
from lnet.pac_phase_gated_transition import (
    PhaseGatedModeResidualPathCollapse,
    PhaseGatedS2DPostFusionTransition,
)

if TYPE_CHECKING:
    from argparse import Namespace
    from collections.abc import Callable

    from torch.utils.data import DataLoader

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig, ComplexScanStage


VARIANTS = tuple(spec.variant for spec in EXPERIMENT_SPECS)
SEEDS = base.SEEDS
P = base.P
MODE_HIDDEN = base.MODE_HIDDEN
COMPILE_MODE = "max-autotune"


def _install_transition(
    stage: ComplexScanStage,
    spec: FullStateExperimentSpec,
) -> None:
    baseline = stage.quadrant_path_mode_combiner
    if not isinstance(baseline, PhaseGatedModeResidualPathCollapse):
        message = "full-state campaign requires the exact PGv2-H192 Base transition"
        raise TypeError(message)
    mixer = StructuredFullStateTransition(
        P,
        spec,
        mode_hidden=MODE_HIDDEN,
        gain_normalization=stage.product_gain_normalization,
    )
    mixer.copy_mode_from(baseline.mode)
    stage.quadrant_path_mode_combiner = mixer
    if spec.post_type == "phase_gated":
        post = stage.augmented
        if not isinstance(post, FactorizedS2DPostFusionTransition):
            message = "PGPost requires the exact Base factorized S2D transition"
            raise TypeError(message)
        replacement = PhaseGatedS2DPostFusionTransition(P, post_hidden=MODE_HIDDEN)
        replacement.copy_carry_from(post)
        cast("Any", stage).augmented = replacement


def _assert_candidate_stage(
    stage: ComplexScanStage,
    spec: FullStateExperimentSpec,
    name: str,
) -> None:
    mixer = stage.quadrant_path_mode_combiner
    if not isinstance(mixer, StructuredFullStateTransition):
        message = f"{name} is missing its structured full-state transition"
        raise TypeError(message)
    if mixer.spec != spec or mixer.mode.hidden_modes != MODE_HIDDEN:
        message = f"{name} changed the declared architecture signature"
        raise RuntimeError(message)
    if spec.post_type == "phase_gated":
        if not isinstance(stage.augmented, PhaseGatedS2DPostFusionTransition):
            message = f"{name} is missing its PGv2 post-fusion block"
            raise TypeError(message)
    elif type(stage.augmented) is not FactorizedS2DPostFusionTransition:
        message = f"{name} changed the Base post-fusion block"
        raise TypeError(message)


def _assert_base_stage(stage: ComplexScanStage, name: str) -> None:
    if not isinstance(stage.quadrant_path_mode_combiner, PhaseGatedModeResidualPathCollapse):
        message = f"{name} changed despite the disabled stage mask"
        raise TypeError(message)
    if type(stage.augmented) is not FactorizedS2DPostFusionTransition:
        message = f"{name} changed its Base post-fusion block"
        raise TypeError(message)


def _assert_model(model: ComplexScanBackbone, spec: FullStateExperimentSpec) -> None:
    base.control.stemres._assert_stem(model)
    for enabled, name in zip(spec.stage_mask, ("stage1", "stage2", "stage3"), strict=True):
        stage = getattr(model, name)
        if enabled:
            _assert_candidate_stage(stage, spec, name)
        else:
            _assert_base_stage(stage, name)
    if (
        model.terminal.output_modes is not None
        or model.terminal.quadrant_path_mode_combiner is not None
    ):
        message = "full-state campaign changed the terminal descriptor"
        raise RuntimeError(message)


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    try:
        spec = SPECS_BY_VARIANT[variant]
    except KeyError as error:
        message = f"unsupported full-state overnight variant: {variant}"
        raise ValueError(message) from error
    model = base._build(base.VARIANT, config)
    for enabled, name in zip(spec.stage_mask, ("stage1", "stage2", "stage3"), strict=True):
        if enabled:
            _install_transition(getattr(model, name), spec)
    _assert_model(model, spec)
    return model


def _variant_config(variant: str) -> dict[str, Any]:
    spec = SPECS_BY_VARIANT[variant]
    payload = deepcopy(base._variant_config())
    payload["backbone"]["name"] = f"A2D-{variant}"
    transition = payload["backbone"]["stage_transition"]
    transition.update(
        {
            "base_model": BASE_MODEL,
            "coarsening": "lossless direction-relative full 2x2 normalized product state",
            "architecture_signature": spec.signature(),
            "architecture_signature_sha256": spec.signature_hash(),
            "mode_pg_state": "exact stagewise copy from Pgv2-H192-All3e-3",
            "innovation_pole_gradient": "detached",
        }
    )
    return payload


def _owned_pg_no_decay_parameters(model: nn.Module) -> set[int]:
    owned: set[int] = set()
    for module in model.modules():
        if isinstance(module, PhaseGatedComplexFFN):
            owned.update((id(module.alpha), id(module.gamma), id(module.norm.weight)))
        elif isinstance(module, GroupedPhaseGatedComplexFFN):
            owned.update((id(module.alpha), id(module.gamma), id(module.norm_weight)))
    return owned


def _build_optimizer(
    model: nn.Module,
    recipe: dict[str, Any],
) -> torch.optim.Optimizer:
    """Preserve Base groups while classifying grouped PG scalars by ownership."""
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    modal_no_decay: list[nn.Parameter] = []
    geometry: list[nn.Parameter] = []
    pg_no_decay = _owned_pg_no_decay_parameters(model)
    for name, parameter in model.named_parameters():
        parameter_name = name.rsplit(".", maxsplit=1)[-1]
        if "damping_logits" in parameter_name or parameter_name.startswith("phase_"):
            geometry.append(parameter)
        elif (
            "analysis" in name
            or "widely_bridge" in name
            or "augmented.direction_mixer" in name
            or "augmented.output_projection" in name
        ):
            modal_no_decay.append(parameter)
        elif (
            id(parameter) in pg_no_decay
            or parameter.ndim < 2
            or "norm" in name
            or "initial_" in name
            or name.endswith(".bias")
        ):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    learning_rate = float(recipe["learning_rate"])
    groups = [
        {"params": decay, "lr": learning_rate, "weight_decay": recipe["weight_decay"]},
        {"params": no_decay, "lr": learning_rate, "weight_decay": 0.0},
        {
            "params": modal_no_decay,
            "lr": learning_rate * recipe["modal_learning_rate_multiplier"],
            "weight_decay": 0.0,
        },
        {
            "params": geometry,
            "lr": learning_rate * recipe["pole_geometry_learning_rate_multiplier"],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(groups, fused=bool(recipe.get("fused_optimizer", False)))
    grouped = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    if len(grouped) != len({id(parameter) for parameter in grouped}):
        message = "optimizer parameter groups overlap"
        raise RuntimeError(message)
    if {id(parameter) for parameter in grouped} != {
        id(parameter) for parameter in model.parameters()
    }:
        message = "optimizer parameter groups do not cover the model exactly"
        raise RuntimeError(message)
    return optimizer


def _set_memory_diagnostics(model: nn.Module, *, enabled: bool) -> None:
    for module in model.modules():
        if isinstance(module, StructuredFullStateTransition):
            module.set_diagnostics_enabled(enabled=enabled)


def _sampled_complex_rms(real: Tensor, imag: Tensor) -> Tensor:
    sampled_real = real.detach().reshape(-1, real.shape[-1])[:4096].float()
    sampled_imag = imag.detach().reshape(-1, imag.shape[-1])[:4096].float()
    return torch.sqrt((sampled_real.square() + sampled_imag.square()).mean())


def _merge_hook(
    metrics: dict[str, float],
    stage_index: int,
) -> Callable[[nn.Module, tuple[object, ...], object], None]:
    @torch.no_grad()
    def record(module: nn.Module, arguments: tuple[object, ...], output: object) -> None:
        if len(arguments) != 4 or not isinstance(output, tuple) or len(output) != 2:
            message = "post-fusion diagnostic hook received an invalid transition call"
            raise RuntimeError(message)
        if not all(isinstance(value, Tensor) for value in arguments) or not all(
            isinstance(value, Tensor) for value in output
        ):
            message = "post-fusion diagnostic hook requires tensor coordinates"
            raise TypeError(message)
        real, imag, carry_real, carry_imag = cast(
            "tuple[Tensor, Tensor, Tensor, Tensor]", arguments
        )
        output_real, output_imag = cast("tuple[Tensor, Tensor]", output)
        active = cast("Any", module)
        reduced_carry_real, reduced_carry_imag = active._carry(carry_real, carry_imag)
        merged_real = real + reduced_carry_real
        merged_imag = imag + reduced_carry_imag
        pole_rms = _sampled_complex_rms(real, imag)
        carry_rms = _sampled_complex_rms(reduced_carry_real, reduced_carry_imag)
        merged_rms = _sampled_complex_rms(merged_real, merged_imag)
        update_rms = _sampled_complex_rms(output_real - merged_real, output_imag - merged_imag)
        prefix = f"merge/stage{stage_index}"
        metrics[f"{prefix}/pole_rms"] = float(pole_rms)
        metrics[f"{prefix}/carry_rms"] = float(carry_rms)
        metrics[f"{prefix}/pole_over_carry"] = float(pole_rms / carry_rms.clamp_min(1.0e-12))
        metrics[f"{prefix}/post_update_over_merged"] = float(
            update_rms / merged_rms.clamp_min(1.0e-12)
        )

    return record


@torch.no_grad()
def _diagnostic_validation_batch(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    *,
    precision: str,
    channels_last: bool,
) -> None:
    inputs, _ = next(iter(loader))
    inputs = inputs.to(device, non_blocking=True)
    if channels_last:
        inputs = inputs.contiguous(memory_format=torch.channels_last)
    merge_metrics: dict[str, float] = {}
    handles = [
        getattr(model, name).augmented.register_forward_hook(_merge_hook(merge_metrics, index))
        for index, name in enumerate(("stage1", "stage2", "stage3"), start=1)
    ]
    _set_memory_diagnostics(model, enabled=True)
    try:
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=precision == "bfloat16",
        ):
            output = model(inputs)
    finally:
        _set_memory_diagnostics(model, enabled=False)
        for handle in handles:
            handle.remove()
    if not isinstance(output, tuple) or len(output) != 5:
        message = "full-state diagnostic pass requires the established five-output classifier"
        raise RuntimeError(message)
    descriptor = output[4].detach().float()
    if descriptor.shape[-1] != 1536:
        message = "full-state Q diagnostics require four 384-coordinate descriptors"
        raise RuntimeError(message)
    metrics: dict[str, float] = {}
    for index, part in enumerate(descriptor.split(384, dim=-1), start=1):
        metrics[f"q{index}_rms"] = float(torch.sqrt(part.square().mean()))
        metrics[f"q{index}_std"] = float(part.std(unbiased=False))
    metrics.update(merge_metrics)
    cast("Any", model)._latest_full_state_evaluation = metrics


def _evaluate(
    model: nn.Module,
    runtime: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    *,
    precision: str,
    channels_last: bool = False,
) -> dict[str, float]:
    result = base.control.stemres.uniform.base.heads._evaluate(
        model,
        runtime,
        loader,
        device,
        precision=precision,
        channels_last=channels_last,
    )
    _diagnostic_validation_batch(
        model,
        loader,
        device,
        precision=precision,
        channels_last=channels_last,
    )
    return result


def _wandb_model_metrics(model: nn.Module) -> dict[str, float]:
    metrics = {
        "model/parameters": float(sum(parameter.numel() for parameter in model.parameters())),
        "model/trainable_parameters": float(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
        "runtime/peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "runtime/peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    }
    metrics.update(
        {
            f"train/{name}": float(value)
            for name, value in getattr(model, "_latest_training_diagnostics", {}).items()
        }
    )
    metrics.update(
        {
            f"validation/{name}": float(value)
            for name, value in getattr(model, "_latest_full_state_evaluation", {}).items()
        }
    )
    for index, name in enumerate(("stage1", "stage2", "stage3"), start=1):
        stage = getattr(model, name)
        mixer = stage.quadrant_path_mode_combiner
        if isinstance(mixer, StructuredFullStateTransition):
            metrics.update(
                {
                    f"full_state/stage{index}/{metric}": value
                    for metric, value in mixer.diagnostic_metrics().items()
                }
            )
        post = stage.augmented
        if isinstance(post, PhaseGatedS2DPostFusionTransition):
            metrics.update(
                {
                    f"full_state/stage{index}/post_pg/{metric}": value
                    for metric, value in post.post.diagnostic_metrics().items()
                }
            )
        damping_x = stage.damping_min + (stage.damping_max - stage.damping_min) * torch.sigmoid(
            stage.damping_logits_x.detach().float()
        )
        damping_y = stage.damping_min + (stage.damping_max - stage.damping_min) * torch.sigmoid(
            stage.damping_logits_y.detach().float()
        )
        damping = torch.cat((damping_x, damping_y))
        phase = torch.cat((stage.phase_x.detach().float(), stage.phase_y.detach().float()))
        metrics.update(
            {
                f"pole/stage{index}/damping_min": float(damping.min()),
                f"pole/stage{index}/damping_mean": float(damping.mean()),
                f"pole/stage{index}/damping_max": float(damping.max()),
                f"pole/stage{index}/phase_mean": float(phase.mean()),
                f"pole/stage{index}/phase_std": float(phase.std(unbiased=False)),
            }
        )
    return metrics


def _contract(args: Namespace) -> dict[str, Any]:
    payload = base._contract(args)
    selected = tuple(args.variants)
    config = base.control.stemres.uniform.base.PoleModelConfig(
        output_dim=100,
        stem_strides=(2, 2),
    )
    models = {variant: _build(variant, config) for variant in selected}
    payload["schema"] = "lnet.a2d.pg_full_state_overnight.imagenet100.v2"
    payload["evidence_status"] = "15-way full-state memory transition overnight campaign"
    payload["variants"] = list(selected)
    payload["seeds"] = list(SEEDS)
    payload["variant_configs"] = {variant: _variant_config(variant) for variant in selected}
    payload["parameter_counts"] = {
        variant: sum(parameter.numel() for parameter in model.parameters())
        for variant, model in models.items()
    }
    payload["architecture_manifest"] = experiment_manifest()
    payload["recipe"]["compile_mode"] = COMPILE_MODE
    payload["recipe"]["oom_policy"] = {
        "preferred": {"batch_size": 128, "gradient_accumulation_steps": 1},
        "fallbacks": [
            {"batch_size": 64, "gradient_accumulation_steps": 2},
            {"batch_size": 32, "gradient_accumulation_steps": 4},
        ],
        "warning": (
            "fallback preserves effective batch but changes BatchNorm and MixUp "
            "microbatch semantics"
        ),
    }
    payload["recipe"]["compile_mode_note"] = (
        "Exact Pgv2-H192-All3e-3 max-autotune contract; capacity fallback is used only "
        "after an isolated CUDA OOM."
    )
    payload["optimizer_contract"] = {
        "PG_projection": {"lr": 0.003, "weight_decay": 0.05},
        "PG_alpha_gamma_norm": {"lr": 0.003, "weight_decay": 0.0},
        "compression_and_path_weights": {"lr": 0.003, "weight_decay": 0.05},
        "pole_geometry": {"lr": 0.0003, "weight_decay": 0.0},
        "selection": "module ownership for PG no-decay parameters; Base rules otherwise",
    }
    payload["deployment_source"] = {
        "git_commit": os.environ.get("LNET_SOURCE_COMMIT", "unknown"),
        "snapshot_sha256": os.environ.get("LNET_SOURCE_FINGERPRINT", "unknown"),
    }
    digest = base.control.stemres.uniform.base.heads.harness._digest
    for name in (
        "pac_full_state_operators.py",
        "pac_full_state_overnight.py",
        "pac_full_state_transition.py",
    ):
        payload["source_sha256"][name.removesuffix(".py")] = digest(Path("src/lnet") / name)
    payload["source_sha256"]["full_state_overnight_runner"] = digest(Path(__file__))
    payload.setdefault("runtime", {})["requested_compile_mode"] = os.environ.get(
        "LNET_COMPILE_MODE",
        COMPILE_MODE,
    )
    return payload


def main() -> None:
    base._configure_ramp()
    ramp = base.control.stemres.uniform.base
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
            evaluate=_evaluate,
            wandb_model_metrics=_wandb_model_metrics,
            summarize=source.heads._summarize,
        )
    )


if __name__ == "__main__":
    main()
