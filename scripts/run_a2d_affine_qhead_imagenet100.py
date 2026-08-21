#!/usr/bin/env python3
"""End-to-end affine/Q-head bakeoff on the matched A2D-D4-PathMix backbone."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import run_a2d_d4_pathmix_imagenet100 as a2d
import run_a2d_qhead_e2e_imagenet100 as structured
import run_alphabet2d_imagenet100_nano as harness
import run_double_prefc_imagenet100 as a2d_base
import torch
from torch import Tensor, nn
from torch.nn import functional

from lnet.a2d_q_heads import A2DAffineQClassifier, expected_calibration_error
from lnet.complex_scan import (
    ComplexScanConfig,
    ModalFusionHead,
    ParallelFusionLRQHead,
)
from lnet.image_layers import (
    LowRankQuadraticModalHead,
    StandardizedAffineModalHead,
)

if TYPE_CHECKING:
    from argparse import Namespace

    from torch.utils.data import DataLoader


VARIANTS = (
    "H0-Affine",
    "H1-Affine-LRQ64",
    "H2-Fusion384-LRQ64",
    "H3-Fusion384-LRQ64-AffAux",
    "H4-Fusion384-AffAux",
    "H5-Fusion384-AffAux05",
    "H6-Fusion384-AffAux10",
)
SEEDS = (501, 509, 521)
_ORIGINAL_EVALUATE = harness._evaluate


def _protocol(variant: str) -> dict[str, Any]:
    protocols: dict[str, dict[str, Any]] = {
        "H0-Affine": {
            "main": "affine",
            "affine": True,
            "fusion_width": None,
            "lrq_rank": None,
            "beta_lrq_initial": None,
            "affine_auxiliary_weight": 0.0,
        },
        "H1-Affine-LRQ64": {
            "main": "affine",
            "affine": True,
            "fusion_width": None,
            "lrq_rank": 64,
            "beta_lrq_initial": 0.1,
            "affine_auxiliary_weight": 0.2,
        },
        "H2-Fusion384-LRQ64": {
            "main": "fusion",
            "affine": False,
            "fusion_width": 384,
            "lrq_rank": 64,
            "beta_lrq_initial": 0.0,
            "affine_auxiliary_weight": 0.0,
        },
        "H3-Fusion384-LRQ64-AffAux": {
            "main": "fusion",
            "affine": True,
            "fusion_width": 384,
            "lrq_rank": 64,
            "beta_lrq_initial": 0.0,
            "affine_auxiliary_weight": 0.2,
        },
        "H4-Fusion384-AffAux": {
            "main": "fusion",
            "affine": True,
            "fusion_width": 384,
            "lrq_rank": None,
            "beta_lrq_initial": None,
            "affine_auxiliary_weight": 0.2,
        },
        "H5-Fusion384-AffAux05": {
            "main": "fusion",
            "affine": True,
            "fusion_width": 384,
            "lrq_rank": None,
            "beta_lrq_initial": None,
            "affine_auxiliary_weight": 0.5,
        },
        "H6-Fusion384-AffAux10": {
            "main": "fusion",
            "affine": True,
            "fusion_width": 384,
            "lrq_rank": None,
            "beta_lrq_initial": None,
            "affine_auxiliary_weight": 1.0,
        },
    }
    return protocols[variant]


def _build(variant: str, config: ComplexScanConfig) -> nn.Module:
    model = a2d._build(a2d.VARIANT, config)
    protocol = _protocol(variant)
    current = model.classifier
    affine = (
        StandardizedAffineModalHead(model.descriptor_dim, config.output_dim)
        if bool(protocol["affine"])
        else None
    )
    fusion: ModalFusionHead | None = None
    lrq: LowRankQuadraticModalHead | None = None
    beta_lrq: nn.Parameter | None = None
    if protocol["main"] == "fusion":
        if not isinstance(current, ParallelFusionLRQHead):
            raise RuntimeError("A2D baseline no longer exposes Fusion+LRQ")
        if current.fusion.hidden_dim != int(protocol["fusion_width"]):
            raise RuntimeError("A2D baseline Fusion width changed")
        fusion = current.fusion
        rank = protocol["lrq_rank"]
        if rank is not None:
            if current.quadratic.rank != int(rank):
                raise RuntimeError("A2D baseline LRQ rank changed")
            lrq = current.quadratic
            beta_lrq = current.beta
    else:
        rank = protocol["lrq_rank"]
        if rank is not None:
            lrq = LowRankQuadraticModalHead(
                model.descriptor_dim,
                config.output_dim,
                int(rank),
            )
            beta_lrq = nn.Parameter(torch.tensor(float(protocol["beta_lrq_initial"])))
    model.classifier = A2DAffineQClassifier(
        model.descriptor_dim,
        config.output_dim,
        main=cast("Literal['affine', 'fusion']", protocol["main"]),
        affine=affine,
        fusion=fusion,
        lrq=lrq,
        beta_lrq=beta_lrq,
        affine_auxiliary_weight=float(protocol["affine_auxiliary_weight"]),
    )
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
        "schema": "lnet.a2d.affine_qhead.imagenet100.v1",
        "evidence_status": "one-seed 30-epoch matched Q-head screening",
        "variants": list(VARIANTS),
        "seeds": list(SEEDS),
        "model": asdict(config),
        "backbone": {
            "name": "A2D-D4-PathMix",
            "config": asdict(backbone_config),
            "descriptor": "3 stages x 4 directions x 48 radial-log Q energies = 576",
        },
        "variant_configs": {
            variant: {
                "backbone": asdict(backbone_config),
                "head": _protocol(variant),
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
            "augmentation": "matched A2D ImageNet-100 public recipe",
            "selection": "fixed epoch 30; no within-run validation selection",
            "resume": "epoch-boundary exact RNG restore",
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
    if not isinstance(output, tuple) or len(output) != 5:
        raise RuntimeError("affine Q-head model returned an invalid training tuple")
    joint, affine, _fusion, _lrq, _descriptor = output
    classifier = cast("A2DAffineQClassifier", cast("Any", model).classifier)
    joint_loss = _mixed_cross_entropy(
        joint,
        targets,
        permuted_targets,
        mixing,
    )
    auxiliary = (
        _mixed_cross_entropy(
            affine,
            targets,
            permuted_targets,
            mixing,
        )
        if classifier.affine_auxiliary_weight > 0.0
        else joint_loss.detach()
    )
    loss = joint_loss + classifier.affine_auxiliary_weight * auxiliary
    return (
        joint,
        loss,
        {
            "joint_ce": joint_loss,
            "affine_aux_ce": auxiliary,
        },
    )


@torch.no_grad()
def _after_training_batch(
    _model: nn.Module,
    _output: Tensor | tuple[Tensor, ...],
    _targets: Tensor,
    _permuted_targets: Tensor,
    _mixing: float,
) -> None:
    return


def _flat_metrics(logits: Tensor, labels: Tensor, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_accuracy": float(logits.argmax(dim=-1).eq(labels).float().mean()),
        f"{prefix}_nll": float(functional.cross_entropy(logits.float(), labels)),
    }


def _evaluation_from_collected(
    model: nn.Module,
    outputs: list[list[Tensor]],
    target: Tensor,
    device: torch.device,
) -> dict[str, float]:
    joint, affine, fusion, lrq, descriptor = [torch.cat(values, dim=0) for values in outputs]
    classifier = cast("A2DAffineQClassifier", cast("Any", model).classifier)
    result = {
        "accuracy": float(joint.argmax(dim=-1).eq(target).float().mean()),
        "cross_entropy": float(functional.cross_entropy(joint, target)),
        "nll": float(functional.cross_entropy(joint, target)),
        "ece": expected_calibration_error(joint, target),
    }
    if classifier.affine is not None:
        result.update(_flat_metrics(affine, target, "affine_only"))
        with torch.inference_mode():
            active_descriptor = descriptor.to(device)
            standardized = classifier.affine.standardizer(active_descriptor)
            reconstructed = classifier.affine.linear(standardized).cpu()
        result["affine_reconstruction_max_error"] = float((affine - reconstructed).abs().max())
    if classifier.fusion is not None:
        result.update(_flat_metrics(fusion, target, "fusion_only"))
    if classifier.lrq is not None and classifier.beta_lrq is not None:
        result.update(_flat_metrics(lrq, target, "lrq_only"))
        without_lrq = joint - float(classifier.beta_lrq.detach()) * lrq
        result["without_lrq_accuracy"] = float(without_lrq.argmax(dim=-1).eq(target).float().mean())
        result["lrq_removal_drop_pp"] = 100.0 * (
            result["accuracy"] - result["without_lrq_accuracy"]
        )
    cast("Any", model)._latest_evaluation_metrics = result
    return result


def _evaluate(
    model: nn.Module,
    runtime: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    *,
    precision: str,
    channels_last: bool = False,
) -> dict[str, float]:
    classifier = cast("Any", model).classifier
    if not isinstance(classifier, A2DAffineQClassifier):
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
                raise RuntimeError("affine Q-head evaluation lost branch outputs")
            for index, value in enumerate(output):
                outputs[index].append(value.detach().float().cpu())
            labels.append(targets.cpu())
    target = torch.cat(labels)
    return _evaluation_from_collected(model, outputs, target, device)


def _wandb_model_metrics(model: nn.Module) -> dict[str, float]:
    classifier = cast("Any", model).classifier
    if not isinstance(classifier, A2DAffineQClassifier):
        return {}
    metrics = {
        f"train/{name}": float(value)
        for name, value in getattr(model, "_latest_training_diagnostics", {}).items()
    }
    metrics.update(
        {
            f"validation/{name}": float(value)
            for name, value in getattr(model, "_latest_evaluation_metrics", {}).items()
            if name not in {"accuracy", "cross_entropy"}
        }
    )
    if classifier.beta_lrq is not None:
        metrics["head/beta_lrq"] = float(classifier.beta_lrq.detach())
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
        "variants": {
            variant: {
                "runs": len(active),
                "mean_accuracy": sum(row["final_validation"]["accuracy"] for row in active)
                / len(active),
                "mean_affine_only_accuracy": sum(
                    row["final_validation"].get("affine_only_accuracy", math.nan) for row in active
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
    structured._training_objective = _training_objective
    structured._after_training_batch = _after_training_batch
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
            train_epoch=structured._train_epoch,
            evaluate=_evaluate,
            wandb_model_metrics=_wandb_model_metrics,
            summarize=_summarize,
        )
    )


if __name__ == "__main__":
    main()
