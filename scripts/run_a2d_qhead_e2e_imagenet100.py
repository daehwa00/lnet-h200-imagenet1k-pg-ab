#!/usr/bin/env python3
"""End-to-end A2D Q-prototype screening on ImageNet-100."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import run_a2d_d4_pathmix_imagenet100 as a2d
import run_alphabet2d_imagenet100_nano as harness
import run_double_prefc_imagenet100 as a2d_base
import torch
from torch import Tensor, nn
from torch.nn import functional

from lnet.a2d_q_heads import (
    A2DPrototypeClassifier,
    EMAPrototypeHead,
    expected_calibration_error,
    grouped_logsumexp_logits,
)
from lnet.complex_scan import ComplexScanConfig

if TYPE_CHECKING:
    from argparse import Namespace

    from torch.utils.data import DataLoader


VARIANTS = (
    "E0-Current",
    "E1-ProtoK1",
    "E2-ProtoK2",
    "E3-StageProtoK2",
    "E4-StageProtoK2-LRQ",
    "E5-StageProtoK2-FusionLRQ",
    "E6-Current-ProtoAux",
)
SEEDS = (501, 509, 521)
STAGE_DIMS = (192, 192, 192)
_ORIGINAL_EVALUATE = harness._evaluate


def _head_protocol(variant: str) -> dict[str, Any]:
    protocols: dict[str, dict[str, Any]] = {
        "E0-Current": {
            "prototype": False,
            "fusion": True,
            "lrq": True,
            "prototype_main": False,
            "lambda_p": 0.0,
            "lambda_c": 0.0,
        },
        "E1-ProtoK1": {
            "components": 1,
            "rank": 0,
            "stagewise": False,
            "fusion": False,
            "lrq": False,
            "prototype_main": True,
            "lambda_p": 0.0,
            "lambda_c": 0.0,
        },
        "E2-ProtoK2": {
            "components": 2,
            "rank": 32,
            "stagewise": False,
            "fusion": False,
            "lrq": False,
            "prototype_main": True,
            "lambda_p": 0.0,
            "lambda_c": 0.0,
        },
        "E3-StageProtoK2": {
            "components": 2,
            "rank": 32,
            "stagewise": True,
            "fusion": False,
            "lrq": False,
            "prototype_main": True,
            "lambda_p": 0.0,
            "lambda_c": 0.0,
        },
        "E4-StageProtoK2-LRQ": {
            "components": 2,
            "rank": 32,
            "stagewise": True,
            "fusion": False,
            "lrq": True,
            "prototype_main": True,
            "beta_lrq": 0.1,
            "lambda_p": 0.2,
            "lambda_c": 0.01,
        },
        "E5-StageProtoK2-FusionLRQ": {
            "components": 2,
            "rank": 32,
            "stagewise": True,
            "fusion": True,
            "lrq": True,
            "prototype_main": True,
            "beta_fusion": 0.1,
            "beta_lrq": 0.1,
            "lambda_p": 0.5,
            "lambda_c": 0.01,
        },
        "E6-Current-ProtoAux": {
            "components": 2,
            "rank": 32,
            "stagewise": True,
            "fusion": True,
            "lrq": True,
            "prototype_main": False,
            "beta_lrq": 0.0,
            "lambda_p": 0.5,
            "lambda_c": 0.01,
        },
    }
    return protocols[variant]


def _build(variant: str, config: ComplexScanConfig) -> nn.Module:
    model = a2d._build(a2d.VARIANT, config)
    if variant == "E0-Current":
        return model
    protocol = _head_protocol(variant)
    prototype = EMAPrototypeHead(
        model.descriptor_dim,
        config.output_dim,
        components=int(protocol["components"]),
        rank=int(protocol["rank"]),
        stage_dims=STAGE_DIMS if bool(protocol["stagewise"]) else None,
        ema_momentum=0.05,
        delta_scale=0.05,
    )
    classifier = A2DPrototypeClassifier(
        model.descriptor_dim,
        config.output_dim,
        prototype=prototype,
        use_fusion=bool(protocol["fusion"]),
        use_lrq=bool(protocol["lrq"]),
        fusion_width=384,
        lrq_rank=64,
        prototype_main=bool(protocol["prototype_main"]),
        beta_fusion=float(protocol.get("beta_fusion", 0.1)),
        beta_lrq=float(protocol.get("beta_lrq", 0.1)),
        prototype_auxiliary=variant == "E6-Current-ProtoAux",
    )
    classifier.lambda_prototype = float(protocol["lambda_p"])
    classifier.lambda_compactness = float(protocol["lambda_c"])
    classifier.delta_penalty_weight = 1.0e-4
    model.classifier = classifier
    return model


def _contract(args: Namespace) -> dict[str, Any]:
    config = ComplexScanConfig(
        output_dim=100,
        stem_strides=(2, 2),
    )
    backbone_config = a2d._variant_config(a2d.VARIANT, config)
    models = {variant: _build(variant, config) for variant in VARIANTS}
    data_digest, train_count, validation_count = harness._dataset_digest(args.data_root)
    payload = {
        "schema": "lnet.a2d.qhead_e2e.imagenet100.v1",
        "evidence_status": "successive-halving end-to-end Q-head screen",
        "variants": list(VARIANTS),
        "seeds": list(SEEDS),
        "model": asdict(config),
        "backbone": {
            "name": "A2D-D4-PathMix",
            "config": asdict(backbone_config),
            "descriptor": "3 stages x 4 direction energies x 48 modes = 576 Q coordinates",
        },
        "variant_configs": {
            variant: {
                "backbone": asdict(backbone_config),
                "head": _head_protocol(variant),
            }
            for variant in VARIANTS
        },
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
            "loader_prefetch_factor": harness.PREFETCH_FACTOR,
            "device_prefetch_scope": "copy_only",
            "compile_mode": "default",
            "channels_last": True,
            "augmentation": ("RandomResizedCrop(224,bicubic)+HFlip+RandAugment(2,9)+RandomErasing"),
            "selection": "fixed final screening epoch; validation is not used within a run",
            "resume": "epoch-boundary exact RNG restore",
            "prototype": {
                "physical_estimator": "log1p(EMA(expm1(Q)))",
                "learnable_delta_scale": 0.05,
                "delta_penalty": 1.0e-4,
                "mixture_assignment": "nearest component with cold-start round robin",
            },
        },
        "data": {
            "manifest_sha256": data_digest,
            "train_images": train_count,
            "validation_images": validation_count,
        },
        "source_sha256": {
            "runner": harness._digest(Path(__file__)),
            "harness": harness._digest(Path("scripts/run_alphabet2d_imagenet100_nano.py")),
            "backbone_runner": harness._digest(Path("scripts/run_a2d_d4_pathmix_imagenet100.py")),
            "model": harness._digest(Path("src/lnet/complex_scan.py")),
            "q_heads": harness._digest(Path("src/lnet/a2d_q_heads.py")),
        },
    }
    return json.loads(json.dumps(payload))


def _mixed_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    permuted_targets: Tensor,
    mixing: float,
) -> Tensor:
    return mixing * functional.cross_entropy(
        logits,
        targets,
        label_smoothing=0.1,
    ) + (1.0 - mixing) * functional.cross_entropy(
        logits,
        permuted_targets,
        label_smoothing=0.1,
    )


def _training_objective(
    model: nn.Module,
    output: Tensor | tuple[Tensor, ...],
    targets: Tensor,
    permuted_targets: Tensor,
    mixing: float,
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    if isinstance(output, Tensor):
        return output, _mixed_cross_entropy(output, targets, permuted_targets, mixing), {}
    if len(output) != 5:
        message = "prototype model returned an invalid training tuple"
        raise RuntimeError(message)
    joint, prototype_logits, _fusion, _lrq, descriptor = output
    classifier = cast("A2DPrototypeClassifier", cast("Any", model).classifier)
    prototype = classifier.prototype
    if prototype is None:
        message = "structured A2D output is missing its prototype head"
        raise RuntimeError(message)
    joint_loss = _mixed_cross_entropy(joint, targets, permuted_targets, mixing)
    prototype_loss = (
        _mixed_cross_entropy(
            prototype_logits,
            targets,
            permuted_targets,
            mixing,
        )
        if classifier.lambda_prototype > 0.0
        else joint_loss.detach()
    )
    compactness = (
        mixing * prototype.compactness(descriptor, targets)
        + (1.0 - mixing) * prototype.compactness(descriptor, permuted_targets)
        if classifier.lambda_compactness > 0.0
        else descriptor.new_zeros(())
    )
    delta_penalty = prototype.delta_penalty()
    loss = (
        joint_loss
        + classifier.lambda_prototype * prototype_loss
        + classifier.lambda_compactness * compactness
        + classifier.delta_penalty_weight * delta_penalty
    )
    return (
        joint,
        loss,
        {
            "joint_ce": joint_loss,
            "prototype_ce": prototype_loss,
            "compactness": compactness,
            "delta_penalty": delta_penalty,
        },
    )


def _train_epoch(
    model: nn.Module,
    runtime: nn.Module,
    loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    mixup_generator: Any,
    mixup_alpha: float,
    precision: str,
    gradient_accumulation_steps: int = 1,
    channels_last: bool = False,
) -> dict[str, float]:
    """Matched harness loop with structured prototype objectives."""
    if gradient_accumulation_steps < 1:
        message = "gradient accumulation steps must be positive"
        raise ValueError(message)
    model.train()
    runtime.train()
    loss_terms: list[Tensor] = []
    correct_terms: list[Tensor] = []
    diagnostics: dict[str, list[Tensor]] = {}
    count = 0
    batch_count = len(loader)
    batches = harness._device_batches(loader, device, channels_last=channels_last)
    for batch_index, (inputs, targets) in enumerate(batches):
        group_offset = batch_index % gradient_accumulation_steps
        if group_offset == 0:
            optimizer.zero_grad(set_to_none=True)
        group_size = min(
            gradient_accumulation_steps,
            batch_count - (batch_index - group_offset),
        )
        permutation = torch.randperm(targets.numel(), device=device)
        mixing = float(mixup_generator.beta(mixup_alpha, mixup_alpha))
        mixed_inputs = mixing * inputs + (1.0 - mixing) * inputs[permutation]
        harness._begin_cudagraph_step(device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=precision == "bfloat16",
        ):
            output = runtime(mixed_inputs)
            logits, loss, active_diagnostics = _training_objective(
                model,
                output,
                targets,
                targets[permutation],
                mixing,
            )
        (loss / group_size).backward()
        _after_training_batch(
            model,
            output,
            targets,
            targets[permutation],
            mixing,
        )
        group_complete = group_offset + 1 == group_size or batch_index + 1 == batch_count
        if group_complete:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        loss_terms.append(loss.detach() * targets.numel())
        correct_terms.append(logits.argmax(dim=-1).eq(targets).sum())
        for name, value in active_diagnostics.items():
            diagnostics.setdefault(name, []).append(value.detach() * targets.numel())
        count += targets.numel()
    result = {
        "loss": float(torch.stack(loss_terms).double().sum()) / count,
        "mixed_accuracy": int(torch.stack(correct_terms).sum()) / count,
    }
    # The common harness records only the two canonical keys in history.  Keep
    # the latest auxiliary values on the model so W&B diagnostics can still
    # expose them without altering the shared/resumable harness contract.
    cast("Any", model)._latest_training_diagnostics = {
        name: float(torch.stack(values).double().sum()) / count
        for name, values in diagnostics.items()
    }
    return result


@torch.no_grad()
def _after_training_batch(
    model: nn.Module,
    output: Tensor | tuple[Tensor, ...],
    targets: Tensor,
    permuted_targets: Tensor,
    mixing: float,
) -> None:
    if isinstance(output, Tensor):
        return
    descriptor = output[4]
    classifier = cast("A2DPrototypeClassifier", cast("Any", model).classifier)
    prototype = classifier.prototype
    if prototype is None:
        return
    prototype.update_ema(
        descriptor,
        targets,
        weights=torch.full_like(targets, mixing, dtype=torch.float32),
    )
    prototype.update_ema(
        descriptor,
        permuted_targets,
        weights=torch.full_like(targets, 1.0 - mixing, dtype=torch.float32),
    )


def _flat_metrics(logits: Tensor, labels: Tensor, prefix: str) -> dict[str, float]:
    accuracy = float(logits.argmax(dim=-1).eq(labels).float().mean())
    return {
        f"{prefix}_accuracy": accuracy,
        f"{prefix}_nll": float(functional.cross_entropy(logits.float(), labels)),
    }


def _evaluate(
    model: nn.Module,
    runtime: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    *,
    precision: str,
    channels_last: bool = False,
) -> dict[str, float]:
    if not isinstance(cast("Any", model).classifier, A2DPrototypeClassifier):
        return _ORIGINAL_EVALUATE(
            model,
            runtime,
            loader,
            device,
            precision=precision,
            channels_last=channels_last,
        )
    model.eval()
    runtime.eval()
    outputs: list[list[Tensor]] = [[], [], [], [], []]
    labels: list[Tensor] = []
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
                message = "prototype evaluation did not return all branches"
                raise RuntimeError(message)
            for index, value in enumerate(output):
                outputs[index].append(value.detach().float().cpu())
            labels.append(targets.cpu())
    joint, prototype_logits, fusion, lrq, descriptor = [
        torch.cat(values, dim=0) for values in outputs
    ]
    target = torch.cat(labels)
    result = {
        "accuracy": float(joint.argmax(dim=-1).eq(target).float().mean()),
        "cross_entropy": float(functional.cross_entropy(joint, target)),
        "nll": float(functional.cross_entropy(joint, target)),
        "ece": expected_calibration_error(joint, target),
    }
    classifier = cast("A2DPrototypeClassifier", cast("Any", model).classifier)
    result.update(_flat_metrics(prototype_logits, target, "prototype_only"))
    if classifier.fusion is not None:
        result.update(_flat_metrics(fusion, target, "fusion_only"))
    if classifier.lrq is not None:
        result.update(_flat_metrics(lrq, target, "lrq_only"))
    if classifier.prototype_main:
        without_prototype = joint - prototype_logits
        without_residual = prototype_logits
        result["without_prototype_accuracy"] = float(
            without_prototype.argmax(dim=-1).eq(target).float().mean()
        )
        result["without_residual_accuracy"] = float(
            without_residual.argmax(dim=-1).eq(target).float().mean()
        )
        result["prototype_removal_drop_pp"] = 100.0 * (
            result["accuracy"] - result["without_prototype_accuracy"]
        )
        result["residual_removal_drop_pp"] = 100.0 * (
            result["accuracy"] - result["without_residual_accuracy"]
        )
    prototype_head = classifier.prototype
    if prototype_head is None:
        cast("Any", model)._latest_evaluation_metrics = result
        return result
    with torch.inference_mode():
        descriptor_gpu = descriptor.to(device)
        component_logits = prototype_head.component_logits(descriptor_gpu[:128])
        weight, bias = prototype_head.affine_component_parameters()
        reconstructed = descriptor_gpu[:128] @ weight.T + bias
        result["affine_component_reconstruction_max_error"] = float(
            (component_logits - reconstructed).abs().max()
        )
        if prototype_head.stage_dims:
            for index, stage_logits in enumerate(
                prototype_head.stage_component_logits(descriptor_gpu)
            ):
                stage_class_logits = grouped_logsumexp_logits(
                    stage_logits,
                    classes=prototype_head.classes,
                    components=prototype_head.components,
                    temperature=1.0,
                )
                result[f"stage{index + 1}_logit_abs_mean"] = float(stage_class_logits.abs().mean())
                result[f"stage{index + 1}_accuracy"] = float(
                    stage_class_logits.argmax(dim=-1).cpu().eq(target).float().mean()
                )
    physical = prototype_head.physical_prototypes().detach().cpu()
    class_centroids = physical.mean(dim=1)
    result["minimum_prototype_margin"] = float(torch.pdist(class_centroids).min())
    usage = prototype_head.component_usage().detach().cpu()
    for index, value in enumerate(usage):
        result[f"prototype_component{index}_usage"] = float(value)
    within = []
    for class_value in torch.unique(target):
        active = descriptor[target == class_value]
        within.append(active.var(dim=0, unbiased=False).mean())
    result["mean_within_class_q_variance"] = float(torch.stack(within).mean())
    cast("Any", model)._latest_evaluation_metrics = result
    return result


def _wandb_model_metrics(model: nn.Module) -> dict[str, float]:
    classifier = cast("Any", model).classifier
    if not isinstance(classifier, A2DPrototypeClassifier):
        return {}
    metrics: dict[str, float] = {}
    metrics.update(
        {
            f"train/{name}": float(value)
            for name, value in getattr(model, "_latest_training_diagnostics", {}).items()
        }
    )
    metrics.update(
        {
            f"validation/{name}": float(value)
            for name, value in getattr(model, "_latest_evaluation_metrics", {}).items()
            if name not in {"accuracy", "cross_entropy"}
        }
    )
    if classifier.beta_fusion is not None:
        metrics["head/beta_fusion"] = float(classifier.beta_fusion.detach())
    if classifier.beta_lrq is not None:
        metrics["head/beta_lrq"] = float(classifier.beta_lrq.detach())
    prototype = classifier.prototype
    if prototype is not None:
        metrics["prototype/initialized_fraction"] = float(prototype.ema_initialized.float().mean())
        metrics["prototype/delta_rms"] = float(prototype.delta.detach().square().mean().sqrt())
        for index, value in enumerate(prototype.component_usage().detach()):
            metrics[f"prototype/component{index}_usage"] = float(value)
        for index, value in enumerate(prototype.stage_log_scales.detach().exp()):
            metrics[f"prototype/stage{index + 1}_scale"] = float(value)
    return metrics


def _summarize(root: Path, contract: dict[str, Any]) -> dict[str, Any] | None:
    rows = []
    for variant in VARIANTS:
        for seed in SEEDS:
            path = root / "results" / f"{variant}__seed{seed}.json"
            if path.exists():
                rows.append(json.loads(path.read_text()))
    if not rows:
        return None
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["variant"], []).append(row)
    summary = {
        "schema": contract["schema"],
        "completed_runs": len(rows),
        "expected_runs": len(VARIANTS) * len(SEEDS),
        "variants": {
            variant: {
                "runs": len(active),
                "mean_accuracy": sum(row["final_validation"]["accuracy"] for row in active)
                / len(active),
                "mean_prototype_only_accuracy": sum(
                    row["final_validation"].get("prototype_only_accuracy", math.nan)
                    for row in active
                )
                / len(active),
            }
            for variant, active in grouped.items()
        },
    }
    harness._atomic_json(root / "summary.json", summary)
    return summary


def main() -> None:
    residuals = a2d_base.residuals
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    harness.main(
        harness.runner_bindings(
            variants=VARIANTS,
            seeds=SEEDS,
            model_config=ComplexScanConfig,
            build_model=_build,
            contract=_contract,
            build_optimizer=residuals.optimizer_source._build_optimizer,
            prepare_model=residuals.base._prepare_model,
            train_epoch=_train_epoch,
            evaluate=_evaluate,
            wandb_model_metrics=_wandb_model_metrics,
            summarize=_summarize,
        )
    )


if __name__ == "__main__":
    main()
