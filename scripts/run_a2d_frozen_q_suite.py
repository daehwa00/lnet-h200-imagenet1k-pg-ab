#!/usr/bin/env python3
"""Run the complete frozen-Q A2D head campaign from one descriptor cache."""

# ruff: noqa: T201

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
import traceback
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.nn import functional

from lnet.a2d_q_heads import (
    PrototypeMetricHead,
    PrototypeResidualHead,
    StagewisePrototypeMetricHead,
    energy_kmeans_prototypes,
    expected_calibration_error,
)
from lnet.a2d_spectral_prototype import (
    class_energy_prototypes,
    classification_metrics,
    diagonal_precision,
    pooled_within_class_variance,
    stratified_fit_calibration_split,
)
from lnet.complex_scan import ModalFusionHead
from lnet.image_layers import LowRankQuadraticModalHead

SEEDS = (501, 509, 521)
STAGE_DIMS = (192, 192, 192)


@dataclass(frozen=True, slots=True)
class HeadSpec:
    name: str
    family: str
    rank: int = 0
    components: int = 1
    width: int = 0


SPECS = (
    HeadSpec("F0-Linear", "linear"),
    HeadSpec("F1-MLP256", "mlp", width=256),
    HeadSpec("F2-LRQ16", "lrq", rank=16),
    HeadSpec("F2-LRQ32", "lrq", rank=32),
    HeadSpec("F2-LRQ64", "lrq", rank=64),
    HeadSpec("F3-ProtoK1-Diag", "prototype", components=1),
    HeadSpec("F4-ProtoK1-PSD16", "prototype", components=1, rank=16),
    HeadSpec("F4-ProtoK1-PSD32", "prototype", components=1, rank=32),
    HeadSpec("F4-ProtoK1-PSD64", "prototype", components=1, rank=64),
    HeadSpec("F5-ProtoK2-Diag", "prototype", components=2),
    HeadSpec("F5-ProtoK2-PSD32", "prototype", components=2, rank=32),
    HeadSpec("F5-ProtoK4-Diag", "prototype", components=4),
    HeadSpec("F5-ProtoK4-PSD32", "prototype", components=4, rank=32),
    HeadSpec("F6-StageProtoK1-Diag", "stagewise", components=1),
    HeadSpec("F6-StageProtoK2-PSD32", "stagewise", components=2, rank=32),
    HeadSpec("F7-ShrinkageLDA", "lda"),
    HeadSpec("F8-DiagonalQDA", "qda"),
    HeadSpec("F9-Proto-LRQ16", "proto_lrq", rank=16),
    HeadSpec("F9-Proto-LRQ32", "proto_lrq", rank=32),
    HeadSpec("F9-Proto-LRQ64", "proto_lrq", rank=64),
    HeadSpec("F10-Proto-Fusion128", "proto_fusion", width=128),
    HeadSpec("F10-Proto-Fusion256", "proto_fusion", width=256),
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--head-epochs", type=int, default=30)
    parser.add_argument("--head-batch-size", type=int, default=4096)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--retry-count", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "alphabet2d-imagenet100"))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", "daehwa"))
    return parser


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _metrics(logits: Tensor, labels: Tensor) -> dict[str, float]:
    output = classification_metrics(logits.float(), labels)
    output["nll"] = output.pop("cross_entropy")
    output["ece"] = expected_calibration_error(logits, labels)
    return output


def _load_cache(root: Path) -> tuple[Tensor, Tensor, Tensor, Tensor, dict[str, Any]]:
    train_path = root / "train-center-crop.pt"
    validation_path = root / "val-center-crop.pt"
    train = torch.load(train_path, map_location="cpu", weights_only=True)
    validation = torch.load(validation_path, map_location="cpu", weights_only=True)
    required = {"features", "labels", "checkpoint_sha256"}
    if not required <= train.keys() or not required <= validation.keys():
        raise RuntimeError("descriptor cache is incomplete")
    if train["checkpoint_sha256"] != validation["checkpoint_sha256"]:
        raise RuntimeError("train and validation caches come from different checkpoints")
    train_features = cast("Tensor", train["features"]).float()
    train_labels = cast("Tensor", train["labels"]).long()
    validation_features = cast("Tensor", validation["features"]).float()
    validation_labels = cast("Tensor", validation["labels"]).long()
    if train_features.shape[1] != 576 or validation_features.shape[1] != 576:
        raise RuntimeError("A2D frozen-Q suite requires 576-dimensional descriptors")
    if not bool(torch.isfinite(train_features).all() and torch.isfinite(validation_features).all()):
        raise RuntimeError("descriptor cache contains non-finite values")
    metadata = {
        "checkpoint_sha256": train["checkpoint_sha256"],
        "train_samples": train_features.shape[0],
        "validation_samples": validation_features.shape[0],
    }
    return train_features, train_labels, validation_features, validation_labels, metadata


def _synthetic_cache() -> tuple[Tensor, Tensor, Tensor, Tensor, dict[str, Any]]:
    generator = torch.Generator().manual_seed(7)
    classes = 4
    dimensions = 576
    train_labels = torch.arange(classes).repeat_interleave(16)
    validation_labels = torch.arange(classes).repeat_interleave(4)
    centers = 0.15 + 0.1 * torch.rand(classes, dimensions, generator=generator)
    train_features = centers[train_labels] + 0.02 * torch.rand(
        train_labels.numel(), dimensions, generator=generator
    )
    validation_features = centers[validation_labels] + 0.02 * torch.rand(
        validation_labels.numel(), dimensions, generator=generator
    )
    return train_features, train_labels, validation_features, validation_labels, {
        "checkpoint_sha256": "synthetic",
        "train_samples": train_features.shape[0],
        "validation_samples": validation_features.shape[0],
    }


@dataclass(slots=True)
class SeedData:
    fit_raw: Tensor
    fit: Tensor
    fit_labels: Tensor
    calibration_raw: Tensor
    calibration: Tensor
    calibration_labels: Tensor
    validation_raw: Tensor
    validation: Tensor
    validation_labels: Tensor
    mean: Tensor
    std: Tensor
    classes: Tensor
    raw_prototypes: Tensor
    prototypes: Tensor
    diagonal: Tensor
    mixture_prototypes: dict[int, Tensor] = field(default_factory=dict)


def _seed_data(
    train_features: Tensor,
    train_labels: Tensor,
    validation_features: Tensor,
    validation_labels: Tensor,
    *,
    seed: int,
    device: torch.device,
) -> SeedData:
    fit_indices, calibration_indices = stratified_fit_calibration_split(
        train_labels,
        calibration_fraction=0.1,
        seed=seed,
    )
    fit_raw = train_features[fit_indices].to(device)
    fit_labels = train_labels[fit_indices].to(device)
    calibration_raw = train_features[calibration_indices].to(device)
    calibration_labels = train_labels[calibration_indices].to(device)
    validation_raw = validation_features.to(device)
    validation_labels_gpu = validation_labels.to(device)
    mean = fit_raw.mean(dim=0)
    std = fit_raw.std(dim=0, unbiased=True).clamp_min_(1.0e-5)
    fit = (fit_raw - mean) / std
    calibration = (calibration_raw - mean) / std
    validation = (validation_raw - mean) / std
    classes, raw_prototypes = class_energy_prototypes(fit_raw, fit_labels)
    prototypes = (raw_prototypes - mean) / std
    variance = pooled_within_class_variance(fit, fit_labels, classes, prototypes)
    # Select the diagonal only on the calibration subset.
    candidates = []
    for shrinkage in (0.0, 0.1, 0.3, 0.5, 0.8, 1.0):
        diagonal = diagonal_precision(variance, shrinkage=shrinkage)
        logits = 2.0 * calibration @ (prototypes * diagonal).T - (
            prototypes.square() * diagonal
        ).sum(dim=1)
        candidates.append((float(functional.cross_entropy(logits, calibration_labels)), diagonal))
    diagonal = min(candidates, key=lambda candidate: candidate[0])[1]
    return SeedData(
        fit_raw,
        fit,
        fit_labels,
        calibration_raw,
        calibration,
        calibration_labels,
        validation_raw,
        validation,
        validation_labels_gpu,
        mean,
        std,
        classes,
        raw_prototypes,
        prototypes,
        diagonal,
    )


def _fit_model(
    model: nn.Module,
    data: SeedData,
    *,
    epochs: int,
    batch_size: int,
    seed: int,
    learning_rate: float = 3.0e-3,
    epoch_callback: Any = None,
) -> tuple[nn.Module, list[dict[str, float]]]:
    generator = torch.Generator(device=data.fit.device).manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1.0e-4)
    best_loss = float("inf")
    best_state = deepcopy(model.state_dict())
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(
            data.fit.shape[0], generator=generator, device=data.fit.device
        )
        for start in range(0, permutation.numel(), batch_size):
            indices = permutation[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = functional.cross_entropy(model(data.fit[indices]), data.fit_labels[indices])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            logits = model(data.calibration)
            calibration_loss = float(functional.cross_entropy(logits, data.calibration_labels))
            calibration_accuracy = float(
                logits.argmax(dim=-1).eq(data.calibration_labels).float().mean()
            )
        row = {
            "epoch": epoch + 1,
            "calibration_nll": calibration_loss,
            "calibration_accuracy": calibration_accuracy,
        }
        history.append(row)
        if epoch_callback is not None:
            epoch_callback(row)
        if calibration_loss < best_loss:
            best_loss = calibration_loss
            best_state = deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model.eval(), history


def _class_sample_means(data: SeedData) -> Tensor:
    return torch.stack([data.fit[data.fit_labels == value].mean(dim=0) for value in data.classes])


def _lda_logits(data: SeedData) -> tuple[Tensor, dict[str, Any]]:
    means = _class_sample_means(data)
    centered = data.fit - means[data.fit_labels]
    covariance = centered.T @ centered / max(1, data.fit.shape[0] - means.shape[0])
    target = covariance.diagonal().mean()
    identity = torch.eye(covariance.shape[0], device=covariance.device)
    best: tuple[float, float, Tensor, Tensor] | None = None
    for shrinkage in (0.01, 0.05, 0.1, 0.3, 0.5, 0.8):
        active = (1.0 - shrinkage) * covariance + shrinkage * target * identity
        precision_means = torch.linalg.solve(active, means.T).T
        calibration_logits = data.calibration @ precision_means.T - 0.5 * (
            means * precision_means
        ).sum(dim=1)
        loss = float(functional.cross_entropy(calibration_logits, data.calibration_labels))
        candidate = (loss, shrinkage, precision_means, means)
        if best is None or loss < best[0]:
            best = candidate
    if best is None:
        raise RuntimeError("LDA shrinkage selection failed")
    _, shrinkage, precision_means, means = best
    logits = data.validation @ precision_means.T - 0.5 * (
        means * precision_means
    ).sum(dim=1)
    return logits, {"selected_shrinkage": shrinkage}


def _qda_logits(data: SeedData) -> tuple[Tensor, dict[str, Any]]:
    means = _class_sample_means(data)
    variances = torch.stack(
        [
            data.fit[data.fit_labels == value].var(dim=0, unbiased=True)
            for value in data.classes
        ]
    )
    target = variances.mean()
    best: tuple[float, float, Tensor] | None = None
    for shrinkage in (0.0, 0.01, 0.05, 0.1, 0.3, 0.5, 0.8):
        active = ((1.0 - shrinkage) * variances + shrinkage * target).clamp_min(1.0e-5)
        calibration_logits = -0.5 * (
            ((data.calibration[:, None, :] - means[None, :, :]).square() / active).sum(dim=-1)
            + active.log().sum(dim=-1)[None, :]
        )
        loss = float(functional.cross_entropy(calibration_logits, data.calibration_labels))
        if best is None or loss < best[0]:
            best = (loss, shrinkage, active)
    if best is None:
        raise RuntimeError("QDA shrinkage selection failed")
    _, shrinkage, active = best
    logits = -0.5 * (
        ((data.validation[:, None, :] - means[None, :, :]).square() / active).sum(dim=-1)
        + active.log().sum(dim=-1)[None, :]
    )
    return logits, {"selected_shrinkage": shrinkage}


def _prototypes_for_components(
    data: SeedData,
    components: int,
) -> Tensor:
    if components == 1:
        return data.prototypes
    if components in data.mixture_prototypes:
        return data.mixture_prototypes[components]
    raw, _ = energy_kmeans_prototypes(
        data.fit,
        data.fit_raw,
        data.fit_labels,
        components=components,
    )
    standardized = (raw - data.mean) / data.std
    data.mixture_prototypes[components] = standardized
    return standardized


def _build_model(spec: HeadSpec, data: SeedData, classes: int) -> nn.Module:
    if spec.family == "linear":
        return nn.Linear(576, classes).to(data.fit.device)
    if spec.family == "mlp":
        return nn.Sequential(
            nn.Linear(576, spec.width),
            nn.GELU(),
            nn.RMSNorm(spec.width),
            nn.Linear(spec.width, classes),
        ).to(data.fit.device)
    if spec.family == "lrq":
        return LowRankQuadraticModalHead(576, classes, spec.rank).to(data.fit.device)
    prototypes = _prototypes_for_components(data, spec.components)
    if spec.family == "prototype":
        return PrototypeMetricHead(
            prototypes,
            classes=classes,
            components=spec.components,
            initial_diagonal=data.diagonal,
            rank=spec.rank,
            learn_temperature=True,
        ).to(data.fit.device)
    if spec.family == "stagewise":
        return StagewisePrototypeMetricHead(
            prototypes,
            classes=classes,
            components=spec.components,
            stage_dims=STAGE_DIMS,
            rank=spec.rank,
        ).to(data.fit.device)
    prototype = PrototypeMetricHead(
        data.prototypes,
        classes=classes,
        initial_diagonal=data.diagonal,
        rank=0,
    ).to(data.fit.device)
    if spec.family == "proto_lrq":
        residual = LowRankQuadraticModalHead(576, classes, spec.rank).to(data.fit.device)
    elif spec.family == "proto_fusion":
        residual = ModalFusionHead(576, spec.width, classes).to(data.fit.device)
    else:
        raise ValueError(f"unsupported learned head family: {spec.family}")
    return PrototypeResidualHead(prototype, residual, beta_initial=0.1).to(data.fit.device)


def _wandb_run(
    args: argparse.Namespace,
    spec: HeadSpec,
    seed: int,
    parameters: int,
) -> Any:
    if not args.wandb_project or os.environ.get("WANDB_MODE") == "disabled":
        return None
    try:
        import wandb
    except ModuleNotFoundError:
        return None
    key = f"{args.output_root.resolve()}::{spec.name}::{seed}"
    run_id = hashlib.sha256(key.encode()).hexdigest()[:16]
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group="A2D-FrozenQ",
        job_type="frozen-q-head",
        name=f"{spec.name}-s{seed}",
        id=run_id,
        resume="allow",
        dir=str(args.output_root / "wandb"),
        config={**asdict(spec), "seed": seed, "parameters": parameters, "backbone": "A2D-D4-PathMix"},
    )


def _run_one(
    spec: HeadSpec,
    seed: int,
    data: SeedData,
    metadata: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.perf_counter()
    classes = int(data.classes.numel())
    history: list[dict[str, float]] = []
    extra: dict[str, Any] = {}
    run = None
    if spec.family == "lda":
        logits, extra = _lda_logits(data)
        parameters = 0
    elif spec.family == "qda":
        logits, extra = _qda_logits(data)
        parameters = 0
    else:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = _build_model(spec, data, classes)
        parameters = sum(parameter.numel() for parameter in model.parameters())
        run = _wandb_run(args, spec, seed, parameters)

        def callback(row: dict[str, float]) -> None:
            if run is not None:
                run.log(
                    {
                        "epoch": row["epoch"],
                        "calibration/nll": row["calibration_nll"],
                        "calibration/accuracy": row["calibration_accuracy"],
                    },
                    step=int(row["epoch"]),
                )

        model, history = _fit_model(
            model,
            data,
            epochs=args.head_epochs,
            batch_size=args.head_batch_size,
            seed=seed,
            epoch_callback=callback,
        )
        with torch.inference_mode():
            logits = model(data.validation)
        if isinstance(model, PrototypeResidualHead):
            extra["residual_beta"] = float(model.beta.detach())
            with torch.inference_mode():
                extra["prototype_only"] = _metrics(
                    model.prototype(data.validation), data.validation_labels
                )
                extra["residual_only"] = _metrics(
                    model.residual(data.validation), data.validation_labels
                )
    metrics = _metrics(logits, data.validation_labels)
    pairwise = torch.pdist(data.prototypes)
    result = {
        "schema": "lnet.a2d.frozen_q_head.v1",
        "spec": asdict(spec),
        "seed": seed,
        "backbone": "A2D-D4-PathMix",
        "cache": metadata,
        "parameters": parameters,
        "validation": metrics,
        "prototype_geometry": {
            "minimum_pairwise_margin": float(pairwise.min()),
            "mean_pairwise_margin": float(pairwise.mean()),
            "mean_within_class_variance": float(
                pooled_within_class_variance(
                    data.fit,
                    data.fit_labels,
                    data.classes,
                    data.prototypes,
                ).mean()
            ),
        },
        "history": history,
        "extra": extra,
        "seconds": time.perf_counter() - started,
    }
    if run is None and spec.family in {"lda", "qda"}:
        run = _wandb_run(args, spec, seed, parameters)
    if run is not None:
        run.log(
            {
                "validation/accuracy": metrics["accuracy"],
                "validation/nll": metrics["nll"],
                "validation/ece": metrics["ece"],
                "validation/balanced_accuracy": metrics["balanced_accuracy"],
                "time/seconds": result["seconds"],
            }
        )
        run.summary.update(metrics)
        run.finish()
    return result


def _summary(root: Path) -> None:
    rows = []
    for path in sorted((root / "results").glob("*.json")):
        try:
            row = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if row.get("schema") == "lnet.a2d.frozen_q_head.v1":
            rows.append(row)
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(row["spec"]["name"], []).append(row["validation"]["accuracy"])
    payload = {
        "schema": "lnet.a2d.frozen_q_suite.v1",
        "completed_jobs": len(rows),
        "expected_jobs": len(SPECS) * len(SEEDS),
        "accuracy": {
            name: {
                "runs": len(values),
                "mean": sum(values) / len(values),
                "minimum": min(values),
                "maximum": max(values),
            }
            for name, values in sorted(grouped.items())
        },
    }
    _atomic_json(root / "summary.json", payload)


def main() -> None:
    args = _parser().parse_args()
    if args.worker_count <= 0 or not 0 <= args.worker_index < args.worker_count:
        raise ValueError("invalid frozen-Q worker partition")
    if args.head_epochs <= 0 or args.head_batch_size <= 0:
        raise ValueError("head epochs and batch size must be positive")
    random.seed(0)
    torch.set_float32_matmul_precision("high")
    if args.smoke_test:
        cache = _synthetic_cache()
        args.head_epochs = min(args.head_epochs, 1)
        args.head_batch_size = min(args.head_batch_size, 32)
        args.wandb_project = ""
    else:
        cache = _load_cache(args.cache_root)
    train_features, train_labels, validation_features, validation_labels, metadata = cache
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Frozen-Q CUDA worker cannot see a GPU")
    selected = [spec for spec in SPECS if not args.only or spec.name in set(args.only)]
    jobs = [
        (spec, seed)
        for spec in selected
        for seed in args.seeds
        if int(hashlib.sha256(f"{spec.name}:{seed}".encode()).hexdigest(), 16)
        % args.worker_count
        == args.worker_index
    ]
    args.output_root.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        args.output_root / f"worker-{args.worker_index}.contract.json",
        {
            "schema": "lnet.a2d.frozen_q_worker.v1",
            "worker_index": args.worker_index,
            "worker_count": args.worker_count,
            "jobs": [{"spec": asdict(spec), "seed": seed} for spec, seed in jobs],
            "metadata": metadata,
            "head_epochs": args.head_epochs,
            "head_batch_size": args.head_batch_size,
        },
    )
    seed_cache: dict[int, SeedData] = {}
    failures = 0
    for spec, seed in jobs:
        result_path = args.output_root / "results" / f"{spec.name}__s{seed}.json"
        if result_path.exists():
            continue
        for attempt in range(args.retry_count + 1):
            try:
                if seed not in seed_cache:
                    seed_cache[seed] = _seed_data(
                        train_features,
                        train_labels,
                        validation_features,
                        validation_labels,
                        seed=seed,
                        device=device,
                    )
                result = _run_one(spec, seed, seed_cache[seed], metadata, args)
                _atomic_json(result_path, result)
                print(
                    json.dumps(
                        {
                            "event": "job_complete",
                            "name": spec.name,
                            "seed": seed,
                            "accuracy": result["validation"]["accuracy"],
                        }
                    ),
                    flush=True,
                )
                break
            except Exception as error:  # noqa: BLE001
                failure = {
                    "event": "job_failure",
                    "name": spec.name,
                    "seed": seed,
                    "attempt": attempt,
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                }
                _atomic_json(
                    args.output_root / "failures" / f"{spec.name}__s{seed}__a{attempt}.json",
                    failure,
                )
                print(json.dumps(failure), flush=True)
                torch.cuda.empty_cache()
                if attempt == args.retry_count:
                    failures += 1
        _summary(args.output_root)
    _summary(args.output_root)
    if failures:
        raise RuntimeError(f"{failures} frozen-Q jobs exhausted all retries")


if __name__ == "__main__":
    main()
