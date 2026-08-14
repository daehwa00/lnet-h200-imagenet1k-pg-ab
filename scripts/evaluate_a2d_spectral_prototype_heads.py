"""Probe theorem-aligned heads on one frozen A2D ImageNet-100 descriptor cache."""

# ruff: noqa: PLR0915, SLF001, T201

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.nn import functional
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_a2d_d4_pathmix_imagenet100 as a2d

from lnet.a2d_spectral_prototype import (
    LearnedPSDPrototypeMetric,
    class_energy_prototypes,
    classification_metrics,
    diagonal_precision,
    grouped_max_logits,
    pooled_within_class_variance,
    prototype_logits,
    stratified_fit_calibration_split,
    two_prototypes_per_class,
)
from lnet.pac_capture_safe_orthogonal import prepare_capture_safe_orthogonal_
from lnet.complex_scan import ComplexScanConfig


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--head-epochs", type=int, default=30)
    parser.add_argument("--head-batch-size", type=int, default=4096)
    parser.add_argument("--metric-rank", type=int, default=16)
    parser.add_argument("--seed", type=int, default=501)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _evaluation_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
            ),
        ]
    )


def _model(checkpoint: dict[str, Any], device: torch.device) -> nn.Module:
    config = ComplexScanConfig(
        output_dim=100,
        stem_strides=(2, 2),
    )
    model = a2d._build(a2d.VARIANT, config)
    model.load_state_dict(checkpoint["model"], strict=True)
    prepare_capture_safe_orthogonal_(model)
    return model.eval().to(device=device, memory_format=torch.channels_last)


class _DescriptorOnly(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, inputs: Tensor) -> Tensor:
        return cast("Any", self.model).raw_descriptor(inputs)


def _descriptor_cache(
    *,
    split: str,
    model: nn.Module,
    runtime: nn.Module,
    data_root: Path,
    cache_path: Path,
    checkpoint_sha256: str,
    batch_size: int,
    workers: int,
    device: torch.device,
    maximum_samples: int | None,
) -> dict[str, Any]:
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        if cached["checkpoint_sha256"] != checkpoint_sha256 or cached["split"] != split:
            message = f"stale descriptor cache at {cache_path}"
            raise RuntimeError(message)
        return cached
    dataset = datasets.ImageFolder(data_root / split, transform=_evaluation_transform())
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
    )
    features: list[Tensor] = []
    labels: list[Tensor] = []
    count = 0
    started = time.perf_counter()
    model.eval()
    runtime.eval()
    with torch.inference_mode():
        for batch_inputs, batch_targets in loader:
            active_inputs = batch_inputs
            active_targets = batch_targets
            if maximum_samples is not None:
                remaining = maximum_samples - count
                if remaining <= 0:
                    break
                active_inputs = active_inputs[:remaining]
                active_targets = active_targets[:remaining]
            active_inputs = active_inputs.to(
                device=device,
                non_blocking=True,
                memory_format=torch.channels_last,
            )
            descriptor = runtime(active_inputs)
            if descriptor.shape[1] != 576 or not bool(torch.isfinite(descriptor).all()):
                message = "A2D descriptor extraction returned invalid values"
                raise RuntimeError(message)
            features.append(descriptor.float().cpu())
            labels.append(active_targets.to(torch.long))
            count += active_targets.numel()
            if count % 8192 < active_targets.numel():
                print(
                    json.dumps(
                        {"event": "descriptor_progress", "split": split, "count": count}
                    ),
                    flush=True,
                )
    payload: dict[str, Any] = {
        "schema": "lnet.a2d.spectral_prototype_descriptor.v1",
        "split": split,
        "checkpoint_sha256": checkpoint_sha256,
        "features": torch.cat(features),
        "labels": torch.cat(labels),
        "classes": dataset.classes,
        "seconds": time.perf_counter() - started,
    }
    _atomic_torch_save(cache_path, payload)
    return payload


def _head_metrics(logits: Tensor, labels: Tensor) -> dict[str, float]:
    return classification_metrics(logits.float(), labels)


def _fit_discriminative_head(
    model: nn.Module,
    fit_features: Tensor,
    fit_labels: Tensor,
    calibration_features: Tensor,
    calibration_labels: Tensor,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> nn.Module:
    generator = torch.Generator(device=fit_features.device).manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1.0e-4)
    best_loss = float("inf")
    best_state = deepcopy(model.state_dict())
    for _ in range(epochs):
        model.train()
        permutation = torch.randperm(
            fit_features.shape[0],
            generator=generator,
            device=fit_features.device,
        )
        for start in range(0, permutation.numel(), batch_size):
            indices = permutation[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = functional.cross_entropy(model(fit_features[indices]), fit_labels[indices])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            calibration_loss = float(
                functional.cross_entropy(model(calibration_features), calibration_labels)
            )
        if calibration_loss < best_loss:
            best_loss = calibration_loss
            best_state = deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model.eval()


def _fit_affine_residual(
    residual: nn.Linear,
    fit_features: Tensor,
    fit_base_logits: Tensor,
    fit_labels: Tensor,
    calibration_features: Tensor,
    calibration_base_logits: Tensor,
    calibration_labels: Tensor,
    *,
    epochs: int,
    batch_size: int,
    seed: int,
) -> nn.Linear:
    generator = torch.Generator(device=fit_features.device).manual_seed(seed)
    optimizer = torch.optim.AdamW(residual.parameters(), lr=3.0e-3, weight_decay=1.0e-3)
    best_loss = float("inf")
    best_state = deepcopy(residual.state_dict())
    for _ in range(epochs):
        residual.train()
        permutation = torch.randperm(
            fit_features.shape[0],
            generator=generator,
            device=fit_features.device,
        )
        for start in range(0, permutation.numel(), batch_size):
            indices = permutation[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = fit_base_logits[indices] + residual(fit_features[indices])
            loss = functional.cross_entropy(logits, fit_labels[indices])
            loss.backward()
            optimizer.step()
        residual.eval()
        with torch.inference_mode():
            calibration_loss = float(
                functional.cross_entropy(
                    calibration_base_logits + residual(calibration_features),
                    calibration_labels,
                )
            )
        if calibration_loss < best_loss:
            best_loss = calibration_loss
            best_state = deepcopy(residual.state_dict())
    residual.load_state_dict(best_state)
    return residual.eval()


def _select_diagonal(
    fit_features: Tensor,
    fit_labels: Tensor,
    prototypes: Tensor,
    classes: Tensor,
    calibration_features: Tensor,
    calibration_labels: Tensor,
) -> tuple[Tensor, float, dict[str, float]]:
    variance = pooled_within_class_variance(fit_features, fit_labels, classes, prototypes)
    best: tuple[float, Tensor, float, dict[str, float]] | None = None
    for shrinkage in (0.0, 0.1, 0.3, 0.5, 0.8, 1.0):
        diagonal = diagonal_precision(variance, shrinkage=shrinkage)
        logits = prototype_logits(calibration_features, prototypes, diagonal=diagonal)
        metrics = _head_metrics(logits, calibration_labels)
        candidate = (metrics["cross_entropy"], diagonal, shrinkage, metrics)
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None:
        message = "diagonal metric selection produced no candidate"
        raise RuntimeError(message)
    return best[1], best[2], best[3]


def main() -> None:
    args = _parser().parse_args()
    if not torch.cuda.is_available():
        message = "the frozen A2D descriptor probe requires CUDA"
        raise RuntimeError(message)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0")
    checkpoint_sha256 = _sha256(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = _model(checkpoint, device)
    descriptor_module = _DescriptorOnly(model)
    runtime: nn.Module = descriptor_module
    if not args.no_compile:
        runtime = cast(
            "nn.Module",
            torch.compile(descriptor_module, mode="default", fullgraph=False, dynamic=False),
        )

    train = _descriptor_cache(
        split="train",
        model=model,
        runtime=runtime,
        data_root=args.data_root,
        cache_path=args.output_root / "cache" / "train-center-crop.pt",
        checkpoint_sha256=checkpoint_sha256,
        batch_size=args.batch_size,
        workers=args.workers,
        device=device,
        maximum_samples=args.max_train_samples,
    )
    validation = _descriptor_cache(
        split="val",
        model=model,
        runtime=runtime,
        data_root=args.data_root,
        cache_path=args.output_root / "cache" / "val-center-crop.pt",
        checkpoint_sha256=checkpoint_sha256,
        batch_size=args.batch_size,
        workers=args.workers,
        device=device,
        maximum_samples=args.max_validation_samples,
    )
    del runtime, descriptor_module
    torch.cuda.empty_cache()

    train_features_cpu = cast("Tensor", train["features"]).float()
    train_labels_cpu = cast("Tensor", train["labels"]).long()
    validation_features = cast("Tensor", validation["features"]).to(device)
    validation_labels = cast("Tensor", validation["labels"]).to(device)
    fit_indices, calibration_indices = stratified_fit_calibration_split(
        train_labels_cpu,
        calibration_fraction=0.1,
        seed=args.seed,
    )
    fit_raw = train_features_cpu[fit_indices].to(device)
    fit_labels = train_labels_cpu[fit_indices].to(device)
    calibration_raw = train_features_cpu[calibration_indices].to(device)
    calibration_labels = train_labels_cpu[calibration_indices].to(device)

    feature_mean = fit_raw.mean(dim=0)
    feature_std = fit_raw.std(dim=0, unbiased=True).clamp_min_(1.0e-5)
    fit = (fit_raw - feature_mean) / feature_std
    calibration = (calibration_raw - feature_mean) / feature_std
    validation_standardized = (validation_features - feature_mean) / feature_std
    classes, raw_prototypes = class_energy_prototypes(fit_raw, fit_labels)
    prototypes = (raw_prototypes - feature_mean) / feature_std

    results: dict[str, Any] = {}
    raw_logits = prototype_logits(validation_features, raw_prototypes)
    results["single_prototype_raw"] = _head_metrics(raw_logits, validation_labels)
    standardized_logits = prototype_logits(validation_standardized, prototypes)
    results["single_prototype_standardized"] = _head_metrics(
        standardized_logits,
        validation_labels,
    )

    diagonal, shrinkage, calibration_diagonal_metrics = _select_diagonal(
        fit,
        fit_labels,
        prototypes,
        classes,
        calibration,
        calibration_labels,
    )
    diagonal_logits = prototype_logits(
        validation_standardized,
        prototypes,
        diagonal=diagonal,
    )
    results["diagonal_mahalanobis"] = {
        **_head_metrics(diagonal_logits, validation_labels),
        "selected_shrinkage": shrinkage,
        "calibration": calibration_diagonal_metrics,
    }

    affine = nn.Linear(576, 100).to(device)
    with torch.no_grad():
        affine.weight.copy_(2.0 * prototypes)
        affine.bias.copy_(-prototypes.square().sum(dim=1))
    affine = cast(
        "nn.Linear",
        _fit_discriminative_head(
            affine,
            fit,
            fit_labels,
            calibration,
            calibration_labels,
            epochs=args.head_epochs,
            batch_size=args.head_batch_size,
            learning_rate=1.0e-2,
            seed=args.seed,
        ),
    )
    with torch.inference_mode():
        results["standardized_affine"] = _head_metrics(
            affine(validation_standardized),
            validation_labels,
        )

    learned_metric = LearnedPSDPrototypeMetric(
        prototypes,
        diagonal,
        args.metric_rank,
    ).to(device)
    learned_metric = cast(
        "LearnedPSDPrototypeMetric",
        _fit_discriminative_head(
            learned_metric,
            fit,
            fit_labels,
            calibration,
            calibration_labels,
            epochs=args.head_epochs,
            batch_size=args.head_batch_size,
            learning_rate=1.0e-2,
            seed=args.seed + 1,
        ),
    )
    with torch.inference_mode():
        results[f"prototype_psd_rank{args.metric_rank}"] = {
            **_head_metrics(learned_metric(validation_standardized), validation_labels),
            "metric_rank": args.metric_rank,
            "effective_diagonal_min": float(learned_metric.diagonal().min()),
            "effective_diagonal_max": float(learned_metric.diagonal().max()),
        }

    multi_raw, multi_classes = two_prototypes_per_class(
        fit,
        fit_raw,
        fit_labels,
    )
    multi = (multi_raw - feature_mean) / feature_std
    multi_logits = prototype_logits(
        validation_standardized,
        multi,
        diagonal=diagonal,
    )
    results["two_prototype_diagonal"] = _head_metrics(
        grouped_max_logits(multi_logits, multi_classes, classes),
        validation_labels,
    )

    model.classifier.eval()
    with torch.no_grad():
        current_head = model.classifier
        current_logits = current_head(validation_features.float())
        current_fusion_logits = current_head.fusion(validation_features.float())
        current_standardized = current_head.quadratic.standardized(validation_features.float())
        current_lrq_linear = current_head.quadratic.linear(current_standardized)
        current_lrq_quadratic = current_head.quadratic.quadratic(
            current_head.quadratic.projection(current_standardized).square()
        )
        current_beta = current_head.beta.detach()
    results["trained_fusion384_lrq64"] = _head_metrics(current_logits, validation_labels)
    results["trained_fusion_only_beta0"] = _head_metrics(
        current_fusion_logits,
        validation_labels,
    )
    results["trained_without_lrq_quadratic"] = _head_metrics(
        current_fusion_logits + current_beta * current_lrq_linear,
        validation_labels,
    )
    results["trained_without_lrq_linear"] = _head_metrics(
        current_fusion_logits + current_beta * current_lrq_quadratic,
        validation_labels,
    )
    results["trained_lrq_contribution"] = {
        "beta": float(current_beta),
        "fusion_logit_abs_mean": float(current_fusion_logits.abs().mean()),
        "scaled_linear_logit_abs_mean": float((current_beta * current_lrq_linear).abs().mean()),
        "scaled_quadratic_logit_abs_mean": float(
            (current_beta * current_lrq_quadratic).abs().mean()
        ),
    }

    with torch.inference_mode():
        multi_psd_logits = prototype_logits(
            validation_standardized,
            multi,
            diagonal=learned_metric.diagonal(),
            low_rank=learned_metric.low_rank,
            logit_scale=learned_metric.logit_scale.clamp(-4.0, 4.0).exp(),
        )
    results[f"two_prototype_psd_rank{args.metric_rank}"] = _head_metrics(
        grouped_max_logits(multi_psd_logits, multi_classes, classes),
        validation_labels,
    )

    with torch.no_grad():
        fit_base_logits = current_head(fit_raw)
        calibration_base_logits = current_head(calibration_raw)
    affine_residual = nn.Linear(576, 100).to(device)
    nn.init.zeros_(affine_residual.weight)
    nn.init.zeros_(affine_residual.bias)
    affine_residual = _fit_affine_residual(
        affine_residual,
        fit,
        fit_base_logits,
        fit_labels,
        calibration,
        calibration_base_logits,
        calibration_labels,
        epochs=args.head_epochs,
        batch_size=args.head_batch_size,
        seed=args.seed + 2,
    )
    with torch.inference_mode():
        results["trained_joint_plus_affine_residual"] = _head_metrics(
            current_logits + affine_residual(validation_standardized),
            validation_labels,
        )

    pairwise_distances = torch.pdist(prototypes)
    summary = {
        "schema": "lnet.a2d.spectral_prototype_probe.v1",
        "protocol": {
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_variant": checkpoint["variant"],
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "backbone_frozen": True,
            "train_view": "deterministic Resize(256)+CenterCrop(224)",
            "validation_view": "deterministic Resize(256)+CenterCrop(224)",
            "validation_used_for_selection": False,
            "fit_samples": int(fit.shape[0]),
            "calibration_samples": int(calibration.shape[0]),
            "validation_samples": int(validation_features.shape[0]),
            "descriptor_dim": int(fit.shape[1]),
            "head_epochs": args.head_epochs,
        },
        "descriptor_extraction": {
            "train_seconds": float(train["seconds"]),
            "validation_seconds": float(validation["seconds"]),
        },
        "prototype_geometry": {
            "minimum_standardized_pair_distance": float(pairwise_distances.min()),
            "mean_standardized_pair_distance": float(pairwise_distances.mean()),
        },
        "heads": results,
    }
    _atomic_json(args.output_root / "results" / "summary.json", summary)
    _atomic_torch_save(
        args.output_root / "results" / "heads.pt",
        {
            "schema": summary["schema"],
            "feature_mean": feature_mean.cpu(),
            "feature_std": feature_std.cpu(),
            "classes": classes.cpu(),
            "raw_prototypes": raw_prototypes.cpu(),
            "diagonal": diagonal.cpu(),
            "affine": affine.state_dict(),
            "learned_metric": learned_metric.state_dict(),
            "joint_affine_residual": affine_residual.state_dict(),
            "multi_raw_prototypes": multi_raw.cpu(),
            "multi_prototype_classes": multi_classes.cpu(),
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
