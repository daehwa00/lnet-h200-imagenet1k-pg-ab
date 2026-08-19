#!/usr/bin/env python3
"""Run the controlled PG versus NoPG ImageNet-1K experiment on one H200."""

from __future__ import annotations

# pyright: reportExplicitAny=false, reportImplicitRelativeImport=false
# pyright: reportPrivateUsage=false
# ruff: noqa: ANN401, C901, PLR0915, SLF001, T201
import json
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
CAMPAIGN_RUNTIME_PATH = Path(__file__).parents[1] / "h200" / "campaign.runtime.json"


def _load_campaign_runtime() -> dict[str, Any]:
    payload = json.loads(CAMPAIGN_RUNTIME_PATH.read_text(encoding="utf-8"))
    required = {
        "campaign_id",
        "manifest_sha256",
        "dataset_train_images",
        "dataset_validation_images",
        "dataset_classes",
        "dataset_identity_algorithm",
        "wandb_sdk_version",
        "wandb_base_url",
        "wandb_app_url",
        "entity",
        "project",
        "group",
        "console",
        "relay_protocol_version",
        "pg_run_id",
        "pg_display_name",
        "pg_tags",
        "no_pg_run_id",
        "no_pg_display_name",
        "no_pg_tags",
    }
    missing = sorted(required.difference(payload))
    if payload.get("schema_version") != 3 or missing:
        message = f"invalid generated H200 campaign runtime; missing={missing}"
        raise RuntimeError(message)
    return payload


CAMPAIGN = _load_campaign_runtime()
EXPERIMENT = str(CAMPAIGN["campaign_id"])
RUN_METADATA = {
    PG_VARIANT: {
        "id": CAMPAIGN["pg_run_id"],
        "display_name": CAMPAIGN["pg_display_name"],
        "tags": tuple(CAMPAIGN["pg_tags"]),
    },
    NO_PG_VARIANT: {
        "id": CAMPAIGN["no_pg_run_id"],
        "display_name": CAMPAIGN["no_pg_display_name"],
        "tags": tuple(CAMPAIGN["no_pg_tags"]),
    },
}


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
    dataset_manifest_path = Path(os.environ["LNET_DATASET_MANIFEST_PATH"])
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    dataset_identity = str(dataset_manifest.get("identity_sha256", ""))
    if dataset_identity != os.environ.get("LNET_DATASET_IDENTITY_SHA256"):
        message = "ImageNet-1K dataset manifest does not match the frozen runtime identity"
        raise RuntimeError(message)
    expected_counts = (
        int(CAMPAIGN["dataset_train_images"]),
        int(CAMPAIGN["dataset_validation_images"]),
        int(CAMPAIGN["dataset_classes"]),
    )
    actual_counts = (
        int(dataset_manifest["splits"]["train"]["count"]),
        int(dataset_manifest["splits"]["val"]["count"]),
        int(dataset_manifest["classes"]["count"]),
    )
    if actual_counts != expected_counts:
        message = f"ImageNet-1K dataset counts changed: {actual_counts} != {expected_counts}"
        raise RuntimeError(message)
    payload = {
        "schema": "lnet.a2d.pgv2_h96.k3_rmsmatch.pg_ab.imagenet1k.h200.v3",
        "evidence_status": "controlled one-H200 ImageNet-1K PG contribution test",
        "campaign": {
            "id": EXPERIMENT,
            "manifest_sha256": CAMPAIGN["manifest_sha256"],
            "deployment_commit": os.environ.get("H200_EXPECTED_COMMIT"),
            "relay_protocol_version": CAMPAIGN["relay_protocol_version"],
            "wandb_sdk_version": CAMPAIGN["wandb_sdk_version"],
        },
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
            "resume": (
                "epoch-boundary RNG continuity with persistent workers forbidden; "
                "bitwise CUDA kernel determinism is not claimed"
            ),
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
            "identity_algorithm": CAMPAIGN["dataset_identity_algorithm"],
            "identity_sha256": dataset_identity,
            "class_names_sha256": dataset_manifest["classes"]["sha256"],
            "train_relpath_size_content_sha256": dataset_manifest["splits"]["train"][
                "relpath_size_content_sha256"
            ],
            "validation_relpath_size_content_sha256": dataset_manifest["splits"]["val"][
                "relpath_size_content_sha256"
            ],
        },
        "telemetry": {
            "authority": "durable local checkpoint, result JSON, and progress JSON",
            "wandb": "best-effort non-authoritative mirror through the scoped relay",
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
            "campaign_manifest": digest(
                Path(__file__).parents[1] / "h200" / "campaign.json"
            ),
            "campaign_runtime": digest(CAMPAIGN_RUNTIME_PATH),
            "environment_lock": digest(
                Path(__file__).parents[1] / "h200" / "requirements.lock"
            ),
            "uv_bootstrap_lock": digest(
                Path(__file__).parents[1] / "h200" / "uv-bootstrap.requirements.txt"
            ),
            "dataset_validator": digest(
                Path(__file__).parents[1] / "h200" / "validate_imagenet1k.py"
            ),
            "h200_imagenet1k_pg_ab_runner": digest(Path(__file__)),
            "pgv2_h96_local_reader_runner": digest(Path(base.__file__)),
        },
    }
    return json.loads(json.dumps(payload))


def _initialize_required_wandb_run(
    root: Path,
    contract: dict[str, Any],
    *,
    variant: str,
    seed: int,
    parameters: int,
) -> WandbRun:
    """Attempt a non-authoritative W&B mirror through the scoped secret relay."""
    try:
        import wandb  # noqa: PLC0415
    except ModuleNotFoundError as error:
        message = "the H200 experiment requires the wandb package"
        raise RuntimeError(message) from error

    expected_environment = {
        "WANDB_API_KEY": "0" * 40,
        "WANDB_APP_URL": CAMPAIGN["wandb_app_url"],
        "WANDB_BASE_URL": CAMPAIGN["wandb_base_url"],
        "WANDB_ENTITY": CAMPAIGN["entity"],
        "WANDB_PROJECT": CAMPAIGN["project"],
        "WANDB_GROUP": CAMPAIGN["group"],
        "WANDB_CONSOLE": CAMPAIGN["console"],
    }
    mismatches = {
        name: {"expected": value, "actual": os.environ.get(name)}
        for name, value in expected_environment.items()
        if os.environ.get(name) != value
    }
    if mismatches:
        message = f"frozen W&B relay environment changed: {sorted(mismatches)}"
        raise RuntimeError(message)
    metadata = RUN_METADATA[variant]
    tracking_root = root / "wandb"
    tracking_root.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        project=CAMPAIGN["project"],
        entity=CAMPAIGN["entity"],
        group=CAMPAIGN["group"],
        name=metadata["display_name"],
        id=metadata["id"],
        resume="allow",
        dir=str(tracking_root),
        mode="online",
        anonymous="never",
        force=True,
        settings=wandb.Settings(
            disable_code=True,
            console=CAMPAIGN["console"],
            disable_git=True,
            disable_job_creation=True,
            init_timeout=float(os.environ.get("WANDB_INIT_TIMEOUT", "30")),
            save_code=False,
            x_disable_meta=True,
            x_disable_stats=True,
            x_disable_viewer=True,
            x_extra_http_headers={
                "User-Agent": "Mozilla/5.0 lnet-h200-wandb-client/1"
            },
            x_save_requirements=False,
        ),
        tags=metadata["tags"],
        config={
            "experiment": EXPERIMENT,
            "variant": variant,
            "seed": seed,
            "parameters": parameters,
            "model": contract["variant_configs"][variant],
            "model_template": contract["model"],
            "recipe": contract["recipe"],
            "schema": contract["schema"],
            "campaign_manifest_sha256": CAMPAIGN["manifest_sha256"],
            "dataset_identity_sha256": contract["data"]["identity_sha256"],
            "telemetry_authority": "local-artifacts",
            "wandb_auth": "cloudflare-secret-relay",
        },
    )
    if run is None or not run.url:
        message = "W&B did not create an online mirror URL"
        raise RuntimeError(message)
    print(f"WANDB_RUN_URL={run.url}", flush=True)
    return run


def _streaming_qhead_evaluate(
    model: torch.nn.Module,
    runtime: torch.nn.Module,
    loader: Any,
    device: torch.device,
    *,
    precision: str,
    channels_last: bool = False,
) -> dict[str, float]:
    """Evaluate the five-output Q head without retaining ImageNet-1K logits."""
    phase_gated = base.control.control
    classifier = getattr(model, "classifier", None)
    if not isinstance(classifier, phase_gated.head_runner.A2DAffineQClassifier):
        message = "H200 streaming evaluation requires the frozen affine Q classifier"
        raise TypeError(message)

    harness = base.control.control.control.stemres.uniform.base.heads.harness
    model.eval()
    runtime.eval()
    total = 0
    joint_correct = 0
    joint_loss = 0.0
    affine_correct = 0
    affine_loss = 0.0
    fusion_correct = 0
    fusion_loss = 0.0
    lrq_correct = 0
    lrq_loss = 0.0
    without_lrq_correct = 0
    reconstruction_max_error = 0.0
    ece_bins = 15
    ece_counts = torch.zeros(ece_bins, dtype=torch.float64)
    ece_correct = torch.zeros(ece_bins, dtype=torch.float64)
    ece_confidence = torch.zeros(ece_bins, dtype=torch.float64)

    with torch.inference_mode():
        for inputs, targets in harness._device_batches(
            loader,
            device,
            channels_last=channels_last,
        ):
            harness._begin_cudagraph_step(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=precision == "bfloat16",
            ):
                output = runtime(inputs)
            if not isinstance(output, tuple) or len(output) != 5:
                message = "affine Q-head evaluation lost branch outputs"
                raise RuntimeError(message)
            joint, affine, fusion, lrq, descriptor = output
            batch_size = targets.numel()
            total += batch_size
            joint_loss += float(
                torch.nn.functional.cross_entropy(
                    joint.float(),
                    targets,
                    reduction="sum",
                )
            )
            joint_predictions = joint.argmax(dim=-1)
            joint_correct += int(joint_predictions.eq(targets).sum())

            probabilities = joint.float().softmax(dim=-1)
            confidence, predictions = probabilities.max(dim=-1)
            # This is equivalent to the reference intervals (lower, upper].
            indices = torch.ceil(confidence * ece_bins).to(torch.long).sub_(1)
            indices.clamp_(0, ece_bins - 1)
            ones = torch.ones_like(confidence, dtype=torch.float64)
            ece_counts += torch.bincount(
                indices,
                weights=ones,
                minlength=ece_bins,
            ).cpu()
            ece_correct += torch.bincount(
                indices,
                weights=predictions.eq(targets).to(torch.float64),
                minlength=ece_bins,
            ).cpu()
            ece_confidence += torch.bincount(
                indices,
                weights=confidence.to(torch.float64),
                minlength=ece_bins,
            ).cpu()

            if classifier.affine is not None:
                affine_correct += int(affine.argmax(dim=-1).eq(targets).sum())
                affine_loss += float(
                    torch.nn.functional.cross_entropy(
                        affine.float(),
                        targets,
                        reduction="sum",
                    )
                )
                standardized = classifier.affine.standardizer(descriptor)
                reconstructed = classifier.affine.linear(standardized)
                reconstruction_max_error = max(
                    reconstruction_max_error,
                    float((affine - reconstructed).abs().max()),
                )
            if classifier.fusion is not None:
                fusion_correct += int(fusion.argmax(dim=-1).eq(targets).sum())
                fusion_loss += float(
                    torch.nn.functional.cross_entropy(
                        fusion.float(),
                        targets,
                        reduction="sum",
                    )
                )
            if classifier.lrq is not None and classifier.beta_lrq is not None:
                lrq_correct += int(lrq.argmax(dim=-1).eq(targets).sum())
                lrq_loss += float(
                    torch.nn.functional.cross_entropy(
                        lrq.float(),
                        targets,
                        reduction="sum",
                    )
                )
                without_lrq = joint - float(classifier.beta_lrq.detach()) * lrq
                without_lrq_correct += int(
                    without_lrq.argmax(dim=-1).eq(targets).sum()
                )

    if total == 0:
        message = "ImageNet-1K validation loader produced no examples"
        raise RuntimeError(message)
    nonempty = ece_counts > 0
    ece = float(
        (
            (ece_counts[nonempty] / total)
            * (
                ece_correct[nonempty] / ece_counts[nonempty]
                - ece_confidence[nonempty] / ece_counts[nonempty]
            ).abs()
        ).sum()
    )
    result = {
        "accuracy": joint_correct / total,
        "cross_entropy": joint_loss / total,
        "nll": joint_loss / total,
        "ece": ece,
    }
    if classifier.affine is not None:
        result.update(
            {
                "affine_only_accuracy": affine_correct / total,
                "affine_only_nll": affine_loss / total,
                "affine_reconstruction_max_error": reconstruction_max_error,
            }
        )
    if classifier.fusion is not None:
        result.update(
            {
                "fusion_only_accuracy": fusion_correct / total,
                "fusion_only_nll": fusion_loss / total,
            }
        )
    if classifier.lrq is not None and classifier.beta_lrq is not None:
        without_lrq_accuracy = without_lrq_correct / total
        result.update(
            {
                "lrq_only_accuracy": lrq_correct / total,
                "lrq_only_nll": lrq_loss / total,
                "without_lrq_accuracy": without_lrq_accuracy,
                "lrq_removal_drop_pp": 100.0
                * (result["accuracy"] - without_lrq_accuracy),
            }
        )
    model._latest_evaluation_metrics = result  # pyright: ignore[reportAttributeAccessIssue]
    return result


def _summarize(root: Path, contract_payload: dict[str, Any]) -> dict[str, Any] | None:
    paths = {
        variant: root / "results" / f"{variant}__seed{SEEDS[0]}.json"
        for variant in VARIANTS
    }
    if not all(path.exists() for path in paths.values()):
        return None
    harness = base.control.control.control.stemres.uniform.base.heads.harness
    rows = {variant: json.loads(path.read_text()) for variant, path in paths.items()}
    pg_accuracy = float(rows[PG_VARIANT]["final_validation"]["accuracy"])
    no_pg_accuracy = float(rows[NO_PG_VARIANT]["final_validation"]["accuracy"])
    payload = {
        "schema": "lnet.a2d.pgv2_h96.k3_rmsmatch.pg_ab.imagenet1k.h200.summary.v2",
        "campaign": {
            "id": EXPERIMENT,
            "manifest_sha256": CAMPAIGN["manifest_sha256"],
        },
        "contract_sha256": harness._contract_sha256(contract_payload),
        "seed": SEEDS[0],
        "pg_final_accuracy": pg_accuracy,
        "no_pg_final_accuracy": no_pg_accuracy,
        "pg_minus_no_pg_percentage_points": 100.0 * (pg_accuracy - no_pg_accuracy),
        "results": rows,
    }
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
            evaluate=_streaming_qhead_evaluate,
            wandb_model_metrics=base._wandb_model_metrics,
            summarize=_summarize,
        )
    )


if __name__ == "__main__":
    main()
