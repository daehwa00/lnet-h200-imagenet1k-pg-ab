#!/usr/bin/env python3
"""Run the controlled Complex Pole Vision Backbone overnight campaign."""

from __future__ import annotations

# pyright: reportAny=false, reportExplicitAny=false
# pyright: reportImplicitRelativeImport=false, reportPrivateLocalImportUsage=false
# pyright: reportPrivateUsage=false
# ruff: noqa: SLF001
import hashlib
import json
import math
import os
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import run_a2d_affine_qhead_imagenet100 as head_runner
import run_a2d_reexcitation_pole_scan_imagenet100 as control
import run_a2d_resaux1_deephead_imagenet100 as deephead
import torch
from torch import nn
from torch.nn.utils import parametrize
from torch.nn.utils.parametrizations import orthogonal

from lnet.complex_scan import ModalFusionHead
from lnet.image_layers import StandardizedAffineModalHead
from lnet.pac_complex_layers import PackedComplexLinear
from lnet.pac_phase_gated_cffn import PhaseGatedComplexFFN
from lnet.pac_pole_backbone_ablation import (
    CarryKind,
    DescriptorKind,
    MemoryAdapter,
    PoleBackboneAblationStage,
    RealSharedLinear,
)

if TYPE_CHECKING:
    from argparse import Namespace

    import numpy as np
    from torch.utils.data import DataLoader

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig
    from lnet.pac_product_scan_pipeline import ScanMemoryPolicy

HeadKind = Literal["fusion_aux", "affine"]

STAGE_NAMES = ("stage1", "stage2", "stage3", "terminal")
DEFAULT_EXCITATIONS = (64, 64, 128, 128)
DEFAULT_POLES = (32, 32, 64, 64)
DEFAULT_SEED = 501
SEED2 = 509
SEED3 = 521
SEEDS = (DEFAULT_SEED, SEED2, SEED3)
COMPILE_MODE = "reduce-overhead"
SCAN_MEMORY_POLICY: ScanMemoryPolicy = "retain"
DESCRIPTOR_PROJECTED_MODES = 96
STEM_HIDDEN_WIDTH = 32


@dataclass(frozen=True)
class BackboneExperimentSpec:
    """One controlled campaign arm; unspecified fields exactly match C0."""

    variant: str
    index: int
    purpose: str
    excitation_schedule: tuple[int, int, int, int] = DEFAULT_EXCITATIONS
    pole_schedule: tuple[int, int, int, int] = DEFAULT_POLES
    stem_width: int = 128
    head_kind: HeadKind = "fusion_aux"
    descriptor_kind: DescriptorKind = "direct"
    memory_adapter: MemoryAdapter = "complex"
    pole_pg: bool = True
    pole_pg_full_hidden: bool = False
    precarry_memory_pg: bool = False
    reexcitation_pg: bool = True
    carry_kind: CarryKind = "learned_s2d"
    seed: int = DEFAULT_SEED

    @property
    def descriptor_dim(self) -> int:
        if self.descriptor_kind == "projected":
            return 4 * len(STAGE_NAMES) * DESCRIPTOR_PROJECTED_MODES
        return 4 * sum(self.pole_schedule)

    def architecture_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("variant")
        payload.pop("index")
        payload.pop("purpose")
        payload.pop("seed")
        payload["descriptor_dim"] = self.descriptor_dim
        return payload

    def signature(self) -> str:
        return json.dumps(self.architecture_payload(), sort_keys=True, separators=(",", ":"))

    def signature_hash(self) -> str:
        return hashlib.sha256(self.signature().encode()).hexdigest()


C0 = BackboneExperimentSpec(
    variant="C01-Base-K64-64-128-P32-32-64",
    index=1,
    purpose="C0 baseline",
)

SPECS = (
    C0,
    replace(
        C0,
        variant="C02-AffineOnly",
        index=2,
        purpose="affine-only E2E head",
        head_kind="affine",
    ),
    replace(
        C0,
        variant="C03-ProjectedQ1536",
        index=3,
        purpose="legacy projected-Q 1536",
        descriptor_kind="projected",
    ),
    replace(
        C0,
        variant="C04-RealSharedAdapter",
        index=4,
        purpose="real-shared memory adapter",
        memory_adapter="real_shared",
    ),
    replace(
        C0,
        variant="C05-PreCarryMemoryPG",
        index=5,
        purpose="additional memory PG before carry merge",
        precarry_memory_pg=True,
    ),
    replace(
        C0,
        variant="C06-NoReExcitationPG",
        index=6,
        purpose="remove merge-time re-excitation PG",
        reexcitation_pg=False,
    ),
    replace(C0, variant="C07-NoPolePG", index=7, purpose="remove pole-memory PG", pole_pg=False),
    replace(
        C0,
        variant="C08-NoCarry",
        index=8,
        purpose="remove S2D residual carry",
        carry_kind="none",
    ),
    replace(
        C0,
        variant="C10-PolePGFullHidden",
        index=10,
        purpose="increase Pole PG hidden width from P/2 to P",
        pole_pg_full_hidden=True,
    ),
    replace(
        C0,
        variant="C11-P32-64-64-64",
        index=11,
        purpose="increase Stage-2 pole count",
        pole_schedule=(32, 64, 64, 64),
    ),
    replace(
        C0,
        variant="C12-P64-All",
        index=12,
        purpose="uniform P64 schedule",
        pole_schedule=(64, 64, 64, 64),
    ),
    replace(
        C0,
        variant="C13-CompactK64-64-96-128",
        index=13,
        purpose="compact width schedule",
        excitation_schedule=(64, 64, 96, 128),
        pole_schedule=(32, 32, 48, 64),
    ),
    replace(
        C0,
        variant="C14-SmoothK64-96-128-128",
        index=14,
        purpose="smooth width schedule",
        excitation_schedule=(64, 96, 128, 128),
        pole_schedule=(32, 48, 64, 64),
    ),
    replace(
        C0,
        variant="C15-Stem160",
        index=15,
        purpose="increase real stem width",
        stem_width=160,
    ),
    replace(
        C0,
        variant="C16-FixedAverageCarry",
        index=16,
        purpose="replace learned full-S2D carry with fixed positional average",
        carry_kind="fixed_average",
    ),
    replace(C0, variant="C17-Base-Seed509", index=17, purpose="C0 seed 2", seed=SEED2),
    replace(C0, variant="C18-Base-Seed521", index=18, purpose="C0 seed 3", seed=SEED3),
    replace(
        C0,
        variant="C19-RealSharedAdapter-Seed509",
        index=19,
        purpose="real-shared adapter seed 2",
        memory_adapter="real_shared",
        seed=SEED2,
    ),
    replace(
        C0,
        variant="C20-P32-64-64-64-Seed509",
        index=20,
        purpose="P=(32,64,64,64) seed 2",
        pole_schedule=(32, 64, 64, 64),
        seed=SEED2,
    ),
)
SPECS_BY_VARIANT = {spec.variant: spec for spec in SPECS}
VARIANTS = tuple(SPECS_BY_VARIANT)

GPU0_VARIANTS = tuple(spec.variant for spec in SPECS if spec.index % 2 == 1)
GPU1_VARIANTS = tuple(spec.variant for spec in SPECS if spec.index % 2 == 0)


def _replace_stem_and_analysis(model: ComplexScanBackbone, spec: BackboneExperimentSpec) -> None:
    if spec.stem_width % 2:
        message = "overnight stem width must be even"
        raise ValueError(message)
    stemres = control.base.joint.control.base.base.control.control.stemres
    model.stem = stemres.ModeScaledTwoConvStem(
        spec.stem_width // 2,
        model.config.stem_strides,
        hidden_width=STEM_HIDDEN_WIDTH,
    )
    source_mixer = nn.Sequential(
        nn.Linear(spec.stem_width, spec.stem_width),
        nn.GELU(),
        nn.Linear(spec.stem_width, spec.stem_width),
        nn.Identity(),
    )
    model.precomplex_fc = stemres.ResidualPreComplexMixer(source_mixer)
    model.input_norm = nn.RMSNorm(spec.stem_width)
    analysis = nn.Linear(spec.stem_width, 2 * spec.excitation_schedule[0], bias=False)
    nn.init.orthogonal_(analysis.weight)
    orthogonal(
        analysis,
        "weight",
        orthogonal_map="matrix_exp",
        use_trivialization=True,
    )
    model.analysis = analysis


def _replace_head(
    model: ComplexScanBackbone,
    spec: BackboneExperimentSpec,
    output_dim: int,
) -> None:
    affine = StandardizedAffineModalHead(spec.descriptor_dim, output_dim)
    if spec.head_kind == "affine":
        classifier = head_runner.A2DAffineQClassifier(
            spec.descriptor_dim,
            output_dim,
            main="affine",
            affine=affine,
            fusion=None,
            lrq=None,
            beta_lrq=None,
            affine_auxiliary_weight=0.0,
        )
    else:
        fusion_source = ModalFusionHead(spec.descriptor_dim, deephead.FIRST_WIDTH, output_dim)
        fusion = deephead.DeepModalFusionHead(fusion_source, output_dim)
        classifier = head_runner.A2DAffineQClassifier(
            spec.descriptor_dim,
            output_dim,
            main="fusion",
            affine=affine,
            fusion=fusion,
            lrq=None,
            beta_lrq=None,
            affine_auxiliary_weight=0.5,
        )
    model.descriptor_dim = spec.descriptor_dim
    cast("Any", model).classifier = classifier


def _stage(
    spec: BackboneExperimentSpec,
    index: int,
    config: ComplexScanConfig,
    memory_policy: ScanMemoryPolicy,
) -> PoleBackboneAblationStage:
    modes = spec.excitation_schedule[index]
    poles = spec.pole_schedule[index]
    terminal = index == len(STAGE_NAMES) - 1
    next_modes = None if terminal else spec.excitation_schedule[index + 1]
    pole_hidden = None
    if not terminal and spec.pole_pg:
        pole_hidden = poles if spec.pole_pg_full_hidden else max(1, poles // 2)
    precarry_hidden = (
        max(1, cast("int", next_modes) // 2)
        if not terminal and spec.precarry_memory_pg
        else None
    )
    reexcitation_hidden = (
        max(1, cast("int", next_modes) // 2)
        if not terminal and spec.reexcitation_pg
        else None
    )
    return PoleBackboneAblationStage(
        modes,
        poles,
        next_modes=next_modes,
        stage_index=index,
        maximum_phase=control.base.joint.canonical8.MAXIMUM_PHASES[index],
        frequency_scale=control.base.joint.calibrated.FREQUENCY_SCALES[index],
        damping_scale=control.base.joint.calibrated.DAMPING_SCALES[index],
        terminal=terminal,
        pole_pg_hidden=pole_hidden,
        memory_adapter=spec.memory_adapter,
        precarry_memory_pg_hidden=precarry_hidden,
        carry_kind="none" if terminal else spec.carry_kind,
        reexcitation_hidden=reexcitation_hidden,
        descriptor_kind=spec.descriptor_kind,
        descriptor_modes=DESCRIPTOR_PROJECTED_MODES,
        scan_memory_policy=memory_policy,
        damping_min=config.damping_min,
        damping_max=config.damping_max,
    )


def _expected_pole_hidden(spec: BackboneExperimentSpec, index: int) -> int | None:
    if index == len(STAGE_NAMES) - 1 or not spec.pole_pg:
        return None
    poles = spec.pole_schedule[index]
    return poles if spec.pole_pg_full_hidden else poles // 2


def _assert_transition_stage(
    stage: PoleBackboneAblationStage,
    spec: BackboneExperimentSpec,
    index: int,
) -> None:
    name = STAGE_NAMES[index]
    if stage.memory_adapter is None:
        message = f"{spec.variant}/{name} is missing its memory adapter"
        raise RuntimeError(message)
    if (spec.memory_adapter == "complex") != isinstance(
        stage.memory_adapter,
        PackedComplexLinear,
    ):
        message = f"{spec.variant}/{name} changed memory adapter algebra"
        raise TypeError(message)
    if (spec.memory_adapter == "real_shared") != isinstance(
        stage.memory_adapter,
        RealSharedLinear,
    ):
        message = f"{spec.variant}/{name} changed real-shared adapter algebra"
        raise TypeError(message)
    if (stage.precarry_memory_pg is not None) != spec.precarry_memory_pg:
        message = f"{spec.variant}/{name} changed pre-carry Memory PG"
        raise RuntimeError(message)
    if (stage.reexcitation_pg is not None) != spec.reexcitation_pg:
        message = f"{spec.variant}/{name} changed Re-Excitation PG"
        raise RuntimeError(message)


def _assert_stage(
    stage: object,
    spec: BackboneExperimentSpec,
    index: int,
    memory_policy: ScanMemoryPolicy,
) -> None:
    name = STAGE_NAMES[index]
    terminal = index == len(STAGE_NAMES) - 1
    next_modes = None if terminal else spec.excitation_schedule[index + 1]
    if (
        not isinstance(stage, PoleBackboneAblationStage)
        or stage.content_modes != spec.excitation_schedule[index]
        or stage.poles != spec.pole_schedule[index]
        or stage.next_modes != next_modes
        or stage.scan_memory_policy != memory_policy
        or stage.descriptor_kind != spec.descriptor_kind
        or stage.carry_kind != ("none" if terminal else spec.carry_kind)
    ):
        message = f"{spec.variant}/{name} changed its stage contract"
        raise RuntimeError(message)
    actual_hidden = (
        stage.memory_pole_pg.hidden_modes if stage.memory_pole_pg is not None else None
    )
    if actual_hidden != _expected_pole_hidden(spec, index):
        message = f"{spec.variant}/{name} changed its Pole PG width"
        raise RuntimeError(message)
    if not terminal:
        _assert_transition_stage(stage, spec, index)
        return
    terminal_blocks = (
        stage.memory_adapter,
        stage.precarry_memory_pg,
        stage.learned_carry,
        stage.fixed_carry_projection,
        stage.reexcitation_pg,
    )
    if any(block is not None for block in terminal_blocks):
        message = f"{spec.variant} terminal contains transition operators"
        raise RuntimeError(message)


def _assert_model(
    model: ComplexScanBackbone,
    spec: BackboneExperimentSpec,
    *,
    memory_policy: ScanMemoryPolicy,
) -> None:
    if (
        model.analysis is None
        or model.analysis.in_features != spec.stem_width
        or model.analysis.out_features != 2 * spec.excitation_schedule[0]
        or not parametrize.is_parametrized(model.analysis, "weight")
    ):
        message = f"{spec.variant} changed its stem-analysis contract"
        raise RuntimeError(message)
    for index, name in enumerate(STAGE_NAMES):
        _assert_stage(getattr(model, name), spec, index, memory_policy)
    classifier = model.classifier
    if (
        model.descriptor_dim != spec.descriptor_dim
        or not isinstance(classifier, head_runner.A2DAffineQClassifier)
        or classifier.main != ("affine" if spec.head_kind == "affine" else "fusion")
        or classifier.affine_auxiliary_weight != (0.0 if spec.head_kind == "affine" else 0.5)
    ):
        message = f"{spec.variant} changed its classifier contract"
        raise RuntimeError(message)


def _build_with_policy(
    variant: str,
    config: ComplexScanConfig,
    memory_policy: ScanMemoryPolicy,
) -> ComplexScanBackbone:
    try:
        spec = SPECS_BY_VARIANT[variant]
    except KeyError as error:
        message = f"unsupported overnight pole-backbone variant: {variant}"
        raise ValueError(message) from error
    model = control.base.joint._build(control.base.joint.VARIANT, config)
    _replace_stem_and_analysis(model, spec)
    for index, name in enumerate(STAGE_NAMES):
        setattr(model, name, _stage(spec, index, config, memory_policy))
    _replace_head(model, spec, config.output_dim)
    _assert_model(model, spec, memory_policy=memory_policy)
    return model


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    return _build_with_policy(variant, config, SCAN_MEMORY_POLICY)


def _pg_blocks(stage: PoleBackboneAblationStage) -> tuple[tuple[str, PhaseGatedComplexFFN], ...]:
    optional = (
        ("pole", stage.memory_pole_pg),
        ("memory", stage.precarry_memory_pg),
        ("reexcitation", stage.reexcitation_pg),
    )
    return tuple((name, block) for name, block in optional if block is not None)


def _capture_pg_gradient_metrics(model: nn.Module) -> None:
    metrics: dict[str, float] = {}
    for index, name in enumerate(STAGE_NAMES, start=1):
        stage = getattr(model, name)
        if not isinstance(stage, PoleBackboneAblationStage):
            continue
        for axis, block in _pg_blocks(stage):
            for metric, value in block.gradient_metrics().items():
                metrics[f"phase_gated/stage{index}/{axis}_{metric}"] = value
    cast("Any", model)._overnight_gradient_metrics = metrics


def _build_optimizer(
    model: nn.Module,
    recipe: dict[str, Any],
) -> torch.optim.Optimizer:
    """Build the shared optimizer and fail closed if PG grouping drifts."""
    source = control.base.joint.control.base.base.control.control.stemres.uniform.base
    optimizer = source.backbone.a2d_base.residuals.optimizer_source._build_optimizer(
        model,
        recipe,
    )
    groups = {
        id(parameter): (float(group["lr"]), float(group["weight_decay"]))
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    base_lr = float(recipe["learning_rate"])
    weight_decay = float(recipe["weight_decay"])

    def assert_group(parameter: nn.Parameter, expected: tuple[float, float], name: str) -> None:
        actual = groups.get(id(parameter))
        if actual is None or not all(
            math.isclose(left, right, rel_tol=0.0, abs_tol=1.0e-12)
            for left, right in zip(actual, expected, strict=True)
        ):
            message = f"optimizer group drifted for {name}: {actual} != {expected}"
            raise RuntimeError(message)

    for module_name, module in model.named_modules():
        if not isinstance(module, PhaseGatedComplexFFN):
            continue
        for name, parameter in (
            ("input_real", module.input_projection.weight_real),
            ("input_imag", module.input_projection.weight_imag),
            ("output_real", module.output_projection.weight_real),
            ("output_imag", module.output_projection.weight_imag),
        ):
            assert_group(parameter, (base_lr, weight_decay), f"{module_name}.{name}")
        for name, parameter in (
            ("alpha", module.alpha),
            ("gamma", module.gamma),
            ("norm", module.norm.weight),
        ):
            assert_group(parameter, (base_lr, 0.0), f"{module_name}.{name}")
    pole_lr = base_lr * float(recipe["pole_geometry_learning_rate_multiplier"])
    for module_name, module in model.named_modules():
        if not isinstance(module, PoleBackboneAblationStage):
            continue
        assert_group(module.damping_logits_x, (pole_lr, 0.0), f"{module_name}.damping_x")
        assert_group(module.damping_logits_y, (pole_lr, 0.0), f"{module_name}.damping_y")
    return optimizer


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
    metrics.update(getattr(model, "_overnight_performance_metrics", {}))
    metrics.update(getattr(model, "_overnight_gradient_metrics", {}))
    for index, name in enumerate(STAGE_NAMES, start=1):
        stage = getattr(model, name)
        if not isinstance(stage, PoleBackboneAblationStage):
            continue
        for metric, value in stage.diagnostic_metrics().items():
            metrics[f"pole_backbone/stage{index}/{metric}"] = value
        for axis, block in _pg_blocks(stage):
            for metric, value in block.diagnostic_metrics().items():
                metrics[f"phase_gated/stage{index}/{axis}_{metric}"] = value
    return metrics


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
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    source = control.base.joint.control.base.base.control.control.stemres.uniform.base
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
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    batch_size = int(loader.batch_size or 1)
    cast("Any", model)._overnight_performance_metrics = {
        "performance/images_per_second": len(loader) * batch_size / elapsed,
        "performance/training_epoch_seconds": elapsed,
        "performance/peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "performance/peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
    }
    _capture_pg_gradient_metrics(model)
    return result


def _variant_config(spec: BackboneExperimentSpec) -> dict[str, Any]:
    return {
        "backbone": {
            "name": spec.variant,
            "campaign_index": spec.index,
            "purpose": spec.purpose,
            "stem": f"3-to-{STEM_HIDDEN_WIDTH}-to-{spec.stem_width} Conv-LN-GELU",
            "excitation_schedule": list(spec.excitation_schedule),
            "pole_schedule": list(spec.pole_schedule),
            "pole_input": "CRMSNorm then packed strict ComplexLinear K-to-P",
            "scan": "optimized associative D4 product scan with endpoint 2x coarsening",
            "descriptor": (
                "direct raw directional pole energy"
                if spec.descriptor_kind == "direct"
                else "legacy shared strict P-to-96 projected energy per raw direction"
            ),
            "descriptor_dim": spec.descriptor_dim,
            "pole_pg": (
                "disabled"
                if not spec.pole_pg
                else ("hidden=P" if spec.pole_pg_full_hidden else "hidden=P/2")
            ),
            "memory_adapter": spec.memory_adapter,
            "precarry_memory_pg": spec.precarry_memory_pg,
            "carry": spec.carry_kind,
            "reexcitation_pg": spec.reexcitation_pg,
            "scan_memory_policy": SCAN_MEMORY_POLICY,
            "compile_mode": COMPILE_MODE,
            "signature_sha256": spec.signature_hash(),
        },
        "head": {
            "kind": spec.head_kind,
            "affine_auxiliary_weight": 0.0 if spec.head_kind == "affine" else 0.5,
        },
        "seed": spec.seed,
    }


def _selected_variants(args: Namespace) -> tuple[str, ...]:
    requested = tuple(getattr(args, "variants", ()))
    return requested or VARIANTS


def _contract(args: Namespace) -> dict[str, Any]:
    payload = control.base.joint._contract(args)
    selected = _selected_variants(args)
    ramp = control.base.joint.control.base.base.control.control.stemres.uniform.base
    config = ramp.PoleModelConfig(output_dim=100, stem_strides=(2, 2))
    models = {variant: _build(variant, config) for variant in selected}
    payload["schema"] = "lnet.a2d.pole_backbone_overnight.imagenet100.v1"
    payload["evidence_status"] = "controlled 20-arm overnight backbone campaign"
    payload["variants"] = list(selected)
    payload["seeds"] = sorted({SPECS_BY_VARIANT[variant].seed for variant in selected})
    payload["variant_configs"] = {
        variant: _variant_config(SPECS_BY_VARIANT[variant]) for variant in selected
    }
    payload["parameter_counts"] = {
        variant: sum(parameter.numel() for parameter in model.parameters())
        for variant, model in models.items()
    }
    payload["architecture"] = {
        variant: SPECS_BY_VARIANT[variant].purpose for variant in selected
    }
    payload["recipe"] = deepcopy(payload["recipe"])
    payload["recipe"].update(
        {
            "compile_mode": COMPILE_MODE,
            "loader_workers": int(args.workers),
            "loader_persistent_workers": (
                args.workers > 0 and os.environ.get("LNET_PERSISTENT_WORKERS", "1") == "1"
            ),
            "cpu_affinity": os.environ.get(
                "LNET_CPU_AFFINITY_ACTIVE",
                os.environ.get("LNET_CPU_AFFINITY"),
            ),
            "checkpointing_policy": "retain compact stage activations; epoch-boundary resume",
            "kernel": (
                "packed strict/real-shared adapters, associative D4 scan, and automatic "
                "packed/fused Phase-Gated kernels"
            ),
            "common_logging": (
                "PG ratios, adapter/carry RMS and correlation, Q distribution, params, "
                "peak VRAM, and images/sec"
            ),
            "phase_gated_optimizer_groups": {
                "projections": "base LR, configured weight decay",
                "alpha_gamma_crmsnorm": "base LR, zero weight decay",
                "pole_geometry": "0.1 times base LR, zero weight decay",
                "runtime_validation": "fail closed for every selected model",
            },
        }
    )
    digest = ramp.heads.harness._digest
    payload["source_sha256"]["pac_pole_backbone_ablation"] = digest(
        Path("src/lnet/pac_pole_backbone_ablation.py")
    )
    payload["source_sha256"]["pole_backbone_overnight_runner"] = digest(Path(__file__))
    return payload


def main() -> None:
    control.base.joint.control.base.base._configure_ramp()
    ramp = control.base.joint.control.base.base.control.control.stemres.uniform.base
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
