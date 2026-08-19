#!/usr/bin/env python3
"""Run the controlled PG versus NoPG ImageNet-1K experiment on one H200."""

from __future__ import annotations

# pyright: reportExplicitAny=false, reportImplicitRelativeImport=false
# pyright: reportPrivateUsage=false
# ruff: noqa: SLF001, T201
import hashlib
import json
import netrc
import os
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import run_a2d_deep4_calibrated_uniform_p96_phase_gated_h96_local_reader_imagenet100 as base
import torch

if TYPE_CHECKING:
    from argparse import Namespace

    from wandb.sdk.wandb_run import Run as WandbRun

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig


PG_VARIANT = base.RMS_MATCH_VARIANT
NO_PG_VARIANT = base.NO_PG_ALL_VARIANT
VARIANTS = (PG_VARIANT, NO_PG_VARIANT)
SEEDS = (501,)
NUM_CLASSES = 1000
EXPERIMENT = "h200-imagenet1k-k3-rmsmatch-pg-ab-v1"


def _configure() -> None:
    base._configure_ramp()
    ramp = base.control.control.control.stemres.uniform.base
    ramp.VARIANTS = VARIANTS
    ramp.SEEDS = SEEDS


def _assert_imagenet1k_phase_gated_model(model: ComplexScanBackbone) -> None:
    """Apply the frozen control assertions with a 1000-way classifier."""
    phase_gated = base.control.control
    phase_gated.control.stemres._assert_stem(model)
    for name in ("stage1", "stage2", "stage3"):
        stage = getattr(model, name)
        mixer = stage.quadrant_path_mode_combiner
        if not isinstance(mixer, phase_gated.PhaseGatedModeResidualPathCollapse):
            message = f"{name} is missing its Phase-Gated mode transition"
            raise TypeError(message)
        if mixer.mode.hidden_modes != phase_gated.MODE_HIDDEN:
            message = f"{name} has the wrong Phase-Gated hidden width"
            raise RuntimeError(message)
        if (
            mixer.mode.gate_redistribution != 0.5
            or mixer.mode.gamma.shape
            or hasattr(mixer.mode, "beta")
        ):
            message = f"{name} changed the Phase-Gated v2 parameterization"
            raise RuntimeError(message)
        if type(stage.augmented) is not phase_gated.FactorizedS2DPostFusionTransition:
            message = f"{name} changed the MPM8 post-fusion control"
            raise TypeError(message)
    if model.terminal.output_modes is not None:
        message = "Phase-Gated mode transition changed the terminal descriptor"
        raise RuntimeError(message)
    classifier = model.classifier
    if not isinstance(classifier, phase_gated.head_runner.A2DAffineQClassifier):
        message = "Phase-Gated model lost its main/auxiliary classifier"
        raise TypeError(message)
    fusion = classifier.fusion
    affine = classifier.affine
    if (
        model.descriptor_dim != 1536
        or not isinstance(fusion, phase_gated.deephead.DeepModalFusionHead)
        or not isinstance(fusion.standardizer, phase_gated.nn.BatchNorm1d)
        or fusion.standardizer.affine
        or fusion.fusion.in_features != 1536
        or fusion.fusion.out_features != 384
        or not isinstance(fusion.norm, phase_gated.nn.RMSNorm)
        or fusion.refinement.in_features != 384
        or fusion.refinement.out_features != 256
        or not isinstance(fusion.refinement_norm, phase_gated.nn.RMSNorm)
        or fusion.classifier.in_features != 256
        or fusion.classifier.out_features != NUM_CLASSES
    ):
        message = "ImageNet-1K Phase-Gated main classifier contract changed"
        raise RuntimeError(message)
    if (
        not isinstance(affine, phase_gated.StandardizedAffineModalHead)
        or not isinstance(affine.standardizer, phase_gated.nn.BatchNorm1d)
        or affine.standardizer.affine
        or affine.linear.in_features != 1536
        or affine.linear.out_features != NUM_CLASSES
        or classifier.affine_auxiliary_weight != phase_gated.AFFINE_AUXILIARY_WEIGHT
    ):
        message = "ImageNet-1K Phase-Gated affine auxiliary contract changed"
        raise RuntimeError(message)


def _build(variant: str, config: ComplexScanConfig) -> torch.nn.Module:
    if variant not in VARIANTS:
        message = f"unsupported H200 ImageNet-1K variant: {variant}"
        raise ValueError(message)
    if config.output_dim != NUM_CLASSES:
        message = f"ImageNet-1K requires {NUM_CLASSES} outputs, got {config.output_dim}"
        raise ValueError(message)
    phase_gated = base.control.control
    original_assertion = phase_gated._assert_model
    phase_gated._assert_model = _assert_imagenet1k_phase_gated_model
    try:
        return base._build(variant, config)
    finally:
        phase_gated._assert_model = original_assertion


def _contract(args: Namespace) -> dict[str, Any]:
    selected = tuple(args.variants)
    ramp = base.control.control.control.stemres.uniform.base
    config = ramp.PoleModelConfig(output_dim=NUM_CLASSES, stem_strides=(2, 2))
    models = {variant: _build(variant, config) for variant in selected}
    variant_configs = {
        variant: deepcopy(base._variant_config(variant)) for variant in selected
    }
    for variant_config in variant_configs.values():
        variant_config["task"] = {
            "dataset": "ImageNet-1K",
            "classes": NUM_CLASSES,
            "image_size": 224,
        }
    digest = ramp.heads.harness._digest
    payload = {
        "schema": "lnet.a2d.pgv2_h96.k3_rmsmatch.pg_ab.imagenet1k.h200.v2",
        "evidence_status": "controlled one-H200 ImageNet-1K PG contribution test",
        "variants": list(selected),
        "seeds": list(SEEDS),
        "priority": list(selected),
        "model": asdict(config),
        "variant_configs": variant_configs,
        "parameter_counts": {
            variant: sum(parameter.numel() for parameter in model.parameters())
            for variant, model in models.items()
        },
        "recipe": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
            "optimizer": "AdamW (fused, pole-aware parameter groups)",
            "fused_optimizer": True,
            "learning_rate": 3.0e-3,
            "modal_learning_rate_multiplier": 1.0 / 3.0,
            "pole_geometry_learning_rate_multiplier": 0.1,
            "weight_decay": 0.05,
            "warmup_epochs": 5,
            "schedule": "warmup plus cosine",
            "label_smoothing": 0.1,
            "mixup_alpha": 0.8,
            "precision": args.precision,
            "loader_prefetch_factor": ramp.heads.harness.PREFETCH_FACTOR,
            "device_prefetch_stream": True,
            "device_prefetch_scope": "copy_only",
            "fused_h2d_channels_last": False,
            "compile_mode": "default",
            "channels_last": True,
            "emulate_precision_casts": True,
            "augmentation": "matched ImageNet-1K 224px recipe",
            "selection": "fixed epoch 100; paired seed 501; no tuning",
            "resume": "epoch-boundary exact RNG restore",
            "kernel": "fused state-plus-stop-gradient-variance Triton recurrence",
            "runtime_bundle": (
                "conjugate-pole reuse + split vertical materialization + capture-safe "
                "orthogonal + channels-last + default torch.compile + fused AdamW"
            ),
            "matmul_precision": "high (TF32 enabled)",
            "compiled_training_preparation": True,
            "phase_gated_optimizer": {
                "alpha_gamma_crmsnorm_learning_rate": 3.0e-3,
                "alpha_gamma_crmsnorm_weight_decay": 0.0,
                "projection_learning_rate": 3.0e-3,
                "projection_weight_decay": 0.0,
                "selection": (
                    "PhaseGatedComplexFFN type-owned projections use base LR without "
                    "weight decay; this prevents gamma-gated branches from decaying "
                    "while task gradients are suppressed"
                ),
            },
        },
        "data": {
            "dataset": "ImageNet-1K",
            "classes": NUM_CLASSES,
            "layout": "ImageFolder train/val validated by h200/run.sh",
            "train_images": 1_281_167,
            "validation_images": 50_000,
        },
        "architecture": {
            variant: variant_configs[variant]["backbone"] for variant in selected
        },
        "comparison": {
            "controlled_factor": "Stage1-3 PhaseGated mode residual present versus absent",
            "fixed": (
                "seed, K3 RMSMatch reader, D4 scan, Q descriptor, path GWL, carry, head, "
                "optimizer, schedule, augmentation, precision, and batch size"
            ),
            "pg": PG_VARIANT,
            "no_pg": NO_PG_VARIANT,
        },
        "source_sha256": {
            "h200_entrypoint": digest(Path(__file__).parents[1] / "h200" / "run.sh"),
            "h200_imagenet1k_pg_ab_runner": digest(Path(__file__)),
            "pgv2_h96_local_reader_runner": digest(Path(base.__file__)),
        },
    }
    return json.loads(json.dumps(payload))


def _has_wandb_credentials() -> bool:
    if os.environ.get("WANDB_API_KEY"):
        return True
    try:
        authenticators = netrc.netrc().authenticators("api.wandb.ai")
    except (FileNotFoundError, netrc.NetrcParseError, OSError):
        return False
    return authenticators is not None


def _initialize_required_wandb_run(
    root: Path,
    contract: dict[str, Any],
    *,
    variant: str,
    seed: int,
    parameters: int,
) -> WandbRun:
    """Always create an online W&B run; use anonymous auth when no secret is mounted."""
    try:
        import wandb  # noqa: PLC0415
    except ModuleNotFoundError as error:
        message = "the H200 experiment requires the wandb package"
        raise RuntimeError(message) from error

    project = os.environ.get("WANDB_PROJECT", "alphabet2d-imagenet1k-h200")
    authenticated = _has_wandb_credentials()
    anonymous = "never" if authenticated else "must"
    entity = os.environ.get("WANDB_ENTITY") if authenticated else None
    run_key = f"{EXPERIMENT}::{variant}::seed{seed}"
    run_id = hashlib.sha256(run_key.encode()).hexdigest()[:16]
    tracking_root = root / "wandb"
    tracking_root.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        project=project,
        entity=entity,
        group=os.environ.get("WANDB_GROUP", EXPERIMENT),
        name=f"H200-I1K-{'PG' if variant == PG_VARIANT else 'NoPG'}-s{seed}",
        id=run_id,
        resume="allow",
        dir=str(tracking_root),
        mode="online",
        anonymous=anonymous,
        force=True,
        settings=wandb.Settings(
            init_timeout=float(os.environ.get("WANDB_INIT_TIMEOUT", "300")),
        ),
        tags=(
            "H200",
            "ImageNet-1K",
            "PG-ablation",
            "anonymous" if not authenticated else "authenticated",
        ),
        config={
            "experiment": EXPERIMENT,
            "variant": variant,
            "seed": seed,
            "parameters": parameters,
            "model": contract["variant_configs"][variant],
            "model_template": contract["model"],
            "recipe": contract["recipe"],
            "schema": contract["schema"],
            "wandb_auth": "account" if authenticated else "anonymous-claim-required",
        },
    )
    if run is None or not run.url:
        message = "W&B did not create an online run URL; refusing to spend H200 time"
        raise RuntimeError(message)
    print(f"WANDB_RUN_URL={run.url}", flush=True)
    return run


def _summarize(root: Path, _contract_payload: dict[str, Any]) -> dict[str, Any] | None:
    paths = {
        variant: root / "results" / f"{variant}__seed{SEEDS[0]}.json"
        for variant in VARIANTS
    }
    if not all(path.exists() for path in paths.values()):
        return None
    rows = {variant: json.loads(path.read_text()) for variant, path in paths.items()}
    pg_accuracy = float(rows[PG_VARIANT]["final_validation"]["accuracy"])
    no_pg_accuracy = float(rows[NO_PG_VARIANT]["final_validation"]["accuracy"])
    payload = {
        "schema": "lnet.a2d.pgv2_h96.k3_rmsmatch.pg_ab.imagenet1k.h200.summary.v1",
        "seed": SEEDS[0],
        "pg_final_accuracy": pg_accuracy,
        "no_pg_final_accuracy": no_pg_accuracy,
        "pg_minus_no_pg_percentage_points": 100.0 * (pg_accuracy - no_pg_accuracy),
        "results": rows,
    }
    harness = base.control.control.control.stemres.uniform.base.heads.harness
    harness._atomic_json(root / "summary.json", payload)
    return payload


def main() -> None:
    _configure()
    ramp = base.control.control.control.stemres.uniform.base
    source = ramp.canonical8.fair_init.backbone.deep4.baseline.baseline
    residuals = ramp.backbone.a2d_base.residuals
    harness = source.heads.harness
    harness._initialize_wandb_run = _initialize_required_wandb_run
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
            wandb_model_metrics=base._wandb_model_metrics,
            summarize=_summarize,
        )
    )


if __name__ == "__main__":
    main()
