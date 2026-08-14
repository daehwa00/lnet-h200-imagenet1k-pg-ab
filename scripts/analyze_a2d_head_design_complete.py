#!/usr/bin/env python3
"""Complete statistical and affine audits for the A2D frozen-Q head campaign."""

# ruff: noqa: ANN401, T201

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.nn import functional

from lnet.a2d_head_design import (
    SCREEN_SPECS,
    DescriptorStatistics,
    FixedAffineTransform,
    HeadDesignSpec,
    StageLogitHead,
    StageScalarTransform,
    build_head,
)
from lnet.a2d_spectral_prototype import stratified_fit_calibration_split

AFFINE_HEADS = {
    "N0-Raw",
    "N1-BNFixed",
    "N2-BNAffine",
    "N3-ZScore",
    "N4-RMSScale",
    "N5-StageScale",
    "N6-Whiten",
    "S1-StageLogits",
}
REFERENCE = "HN1-RMSAfter"
SEEDS = (501, 509, 521)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    return parser


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _dummy_statistics() -> DescriptorStatistics:
    return DescriptorStatistics(
        torch.zeros(576),
        torch.ones(576),
        torch.ones(576),
        torch.zeros(3),
        torch.ones(3),
        torch.eye(576),
    )


def _load_cache(root: Path) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    train = torch.load(root / "train-center-crop.pt", map_location="cpu", weights_only=True)
    validation = torch.load(root / "val-center-crop.pt", map_location="cpu", weights_only=True)
    return (
        cast("Tensor", train["features"]).float(),
        cast("Tensor", train["labels"]).long(),
        cast("Tensor", validation["features"]).float(),
        cast("Tensor", validation["labels"]).long(),
    )


def _specs() -> dict[str, HeadDesignSpec]:
    return {spec.name: spec for spec in SCREEN_SPECS}


def _artifact(root: Path, name: str, seed: int) -> dict[str, Any]:
    path = root / "predictions" / f"final__{name}__s{seed}.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=True)


def _result(root: Path, name: str, seed: int) -> dict[str, Any]:
    path = root / "results" / f"final__{name}__s{seed}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _bootstrap_delta(
    candidate: list[Tensor],
    reference: list[Tensor],
    labels: Tensor,
    *,
    draws: int,
    seed: int = 20_260_806,
) -> dict[str, float]:
    effects = torch.stack(
        [
            left.argmax(dim=-1).eq(labels).float() - right.argmax(dim=-1).eq(labels).float()
            for left, right in zip(candidate, reference, strict=True)
        ]
    )
    generator = torch.Generator().manual_seed(seed)
    chunks = []
    batch_draws = 100
    for start in range(0, draws, batch_draws):
        active = min(batch_draws, draws - start)
        seed_indices = torch.randint(len(candidate), (active, len(candidate)), generator=generator)
        sample_indices = torch.randint(
            labels.numel(), (active, labels.numel()), generator=generator
        )
        sampled = effects[seed_indices]
        sampled = sampled.gather(2, sample_indices[:, None, :].expand(-1, len(candidate), -1))
        chunks.append(sampled.mean(dim=(1, 2)))
    distribution = torch.cat(chunks)
    return {
        "mean_delta_pp": 100.0 * float(effects.mean()),
        "ci95_low_pp": 100.0 * float(torch.quantile(distribution, 0.025)),
        "ci95_high_pp": 100.0 * float(torch.quantile(distribution, 0.975)),
        "probability_positive": float(distribution.gt(0).float().mean()),
    }


def _condition(values: Tensor) -> dict[str, float | int]:
    centered = values.float() - values.float().mean(dim=0)
    covariance = centered.T @ centered / max(1, centered.shape[0] - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance.double()).float().clamp_min(0)
    threshold = eigenvalues.max() * 1.0e-8
    positive = eigenvalues[eigenvalues > threshold]
    return {
        "largest_eigenvalue": float(eigenvalues.max()),
        "smallest_effective_eigenvalue": float(positive.min()),
        "condition_number": float(positive.max() / positive.min()),
        "effective_rank": int(positive.numel()),
        "participation_rank": float(eigenvalues.sum().square() / eigenvalues.square().sum()),
    }


def _descriptor_geometry(train: Tensor, train_labels: Tensor) -> dict[str, Any]:
    fit, _ = stratified_fit_calibration_split(train_labels, calibration_fraction=0.1, seed=SEEDS[0])
    values = train[fit]
    mean = values.mean(dim=0)
    std = values.std(dim=0, unbiased=True).clamp_min(1.0e-5)
    standardized = (values - mean) / std
    return {
        "raw": _condition(values),
        "coordinate_zscore": _condition(standardized),
        "stages_raw": [_condition(stage) for stage in values.split(192, dim=1)],
        "stages_zscore": [_condition(stage) for stage in standardized.split(192, dim=1)],
        "coordinate_mean_range": [float(mean.min()), float(mean.max())],
        "coordinate_std_range": [float(std.min()), float(std.max())],
    }


def _base_affine(model: nn.Module) -> tuple[Tensor, Tensor]:
    if isinstance(model, StageLogitHead):
        weight = torch.cat([layer.weight.detach() for layer in model.weights], dim=1)
        return weight, model.bias.detach()
    classifier = cast("Any", model).classifier
    return classifier.weight.detach(), classifier.bias.detach()


def _fold_transform(model: nn.Module, weight: Tensor, bias: Tensor) -> tuple[Tensor, Tensor]:
    transform = cast("Any", model).transform
    if isinstance(transform, nn.Identity):
        return weight, bias
    if isinstance(transform, nn.BatchNorm1d):
        scale = torch.rsqrt(transform.running_var + transform.eps)
        shift = -transform.running_mean * scale
        if transform.affine:
            scale = scale * transform.weight
            shift = shift * transform.weight + transform.bias
        return weight * scale[None, :], bias + weight @ shift
    if isinstance(transform, FixedAffineTransform):
        active = transform.scale_or_matrix
        folded = weight / active[None, :] if active.ndim == 1 else weight @ active.T
        return folded, bias - folded @ transform.offset
    if isinstance(transform, StageScalarTransform):
        means = transform.means[:, None].expand(3, 192).reshape(-1)
        scales = transform.scales[:, None].expand(3, 192).reshape(-1)
        folded = weight / scales[None, :]
        return folded, bias - folded @ means
    message = f"non-affine transform cannot be folded: {type(transform).__name__}"
    raise TypeError(message)


def _load_affine_model(
    root: Path,
    spec: HeadDesignSpec,
    seed: int,
) -> nn.Module:
    artifact = _artifact(root, spec.name, seed)
    state = artifact.get("model_state")
    if state is None:
        message = f"{spec.name} seed {seed} lacks the complete model state"
        raise RuntimeError(message)
    model = build_head(spec, _dummy_statistics(), classes=100)
    model.load_state_dict(state, strict=True)
    return model.eval()


def _affine_audit(
    root: Path,
    validation: Tensor,
) -> tuple[dict[str, Any], dict[str, dict[int, Tensor]]]:
    specs = _specs()
    weights: dict[str, dict[int, Tensor]] = {}
    rows: dict[str, Any] = {}
    for name in sorted(AFFINE_HEADS):
        active: dict[int, Tensor] = {}
        fold_errors = []
        stage_mass = []
        direction_mass = []
        for seed in SEEDS:
            model = _load_affine_model(root, specs[name], seed)
            weight, bias = _base_affine(model)
            weight, bias = _fold_transform(model, weight, bias)
            centered = weight - weight.mean(dim=0, keepdim=True)
            active[seed] = centered
            with torch.inference_mode():
                direct = model(validation[:1024])
                folded = functional.linear(validation[:1024], weight, bias)
            fold_errors.append(float((direct - folded).abs().max()))
            shaped = centered.abs().reshape(100, 3, 4, 48)
            total = shaped.sum().clamp_min(1.0e-12)
            stage_mass.append((shaped.sum(dim=(0, 2, 3)) / total).tolist())
            direction_mass.append((shaped.sum(dim=(0, 1, 3)) / total).tolist())
        weights[name] = active
        pairs = {}
        seeds = sorted(active)
        for left_index, left_seed in enumerate(seeds):
            for right_seed in seeds[left_index + 1 :]:
                left = active[left_seed]
                right = active[right_seed]
                cosine = functional.cosine_similarity(left, right, dim=1)
                jaccard = []
                for class_index in range(left.shape[0]):
                    left_top = set(left[class_index].abs().topk(32).indices.tolist())
                    right_top = set(right[class_index].abs().topk(32).indices.tolist())
                    jaccard.append(len(left_top & right_top) / len(left_top | right_top))
                pairs[f"{left_seed}-{right_seed}"] = {
                    "class_weight_cosine_mean": float(cosine.mean()),
                    "top32_jaccard_mean": sum(jaccard) / len(jaccard),
                }
        rows[name] = {
            "fold_max_error": max(fold_errors),
            "seed_pairs": pairs,
            "stage_absolute_weight_fraction": torch.tensor(stage_mass).mean(dim=0).tolist(),
            "direction_absolute_weight_fraction": torch.tensor(direction_mass).mean(dim=0).tolist(),
        }
    return rows, weights


def _symmetric_least_squares(prototypes: Tensor, target: Tensor) -> Tensor:
    gram = prototypes.T @ prototypes
    right = prototypes.T @ target + target.T @ prototypes
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    rotated = eigenvectors.T @ right @ eigenvectors
    denominator = eigenvalues[:, None] + eigenvalues[None, :]
    threshold = eigenvalues.max().clamp_min(1.0) * 1.0e-7
    solution = torch.where(
        denominator > threshold,
        rotated / denominator.clamp_min(float(threshold)),
        torch.zeros_like(rotated),
    )
    return eigenvectors @ solution @ eigenvectors.T


def _fit_quality(reference: Tensor, candidate: Tensor) -> dict[str, float]:
    residual = (reference - candidate).square().sum()
    total = reference.square().sum().clamp_min(1.0e-12)
    return {
        "explained_frobenius": float(1.0 - residual / total),
        "relative_error": float((residual / total).sqrt()),
        "cosine": float(
            functional.cosine_similarity(reference.flatten(), candidate.flatten(), dim=0)
        ),
    }


def _prototype_projection(
    train: Tensor,
    train_labels: Tensor,
    weights: dict[int, Tensor],
) -> dict[str, Any]:
    rows = {}
    for seed, reference in weights.items():
        fit, _ = stratified_fit_calibration_split(train_labels, calibration_fraction=0.1, seed=seed)
        features = train[fit]
        labels = train_labels[fit]
        prototypes = torch.stack([features[labels == value].mean(dim=0) for value in range(100)])
        prototypes = prototypes - prototypes.mean(dim=0, keepdim=True)
        target = 0.5 * reference
        unconstrained = torch.linalg.pinv(prototypes, rtol=1.0e-6) @ target
        symmetric = _symmetric_least_squares(prototypes, target)
        eigenvalues, eigenvectors = torch.linalg.eigh(0.5 * (symmetric + symmetric.T))
        psd = (eigenvectors * eigenvalues.clamp_min(0)) @ eigenvectors.T
        rows[str(seed)] = {
            "prototype_rank": int(torch.linalg.matrix_rank(prototypes)),
            "unconstrained": _fit_quality(reference, 2.0 * prototypes @ unconstrained),
            "symmetric": _fit_quality(reference, 2.0 * prototypes @ symmetric),
            "psd": _fit_quality(reference, 2.0 * prototypes @ psd),
            "negative_eigenvalue_mass_fraction": float(
                eigenvalues.clamp_max(0).abs().sum() / eigenvalues.abs().sum().clamp_min(1.0e-12)
            ),
        }
    return rows


def main() -> None:
    args = _parser().parse_args()
    train, train_labels, validation, validation_labels = _load_cache(args.cache_root)
    specs = _specs()
    missing = [
        f"{spec.name}/s{seed}"
        for spec in specs.values()
        for seed in SEEDS
        if not (args.campaign_root / "results" / f"final__{spec.name}__s{seed}.json").exists()
    ]
    if missing:
        message = f"head campaign is incomplete: {missing[:5]}"
        raise RuntimeError(message)
    predictions = {
        name: [_artifact(args.campaign_root, name, seed)["logits"] for seed in SEEDS]
        for name in specs
    }
    bootstrap = {
        name: _bootstrap_delta(
            values,
            predictions[REFERENCE],
            validation_labels,
            draws=args.bootstrap_draws,
        )
        for name, values in predictions.items()
        if name != REFERENCE
    }
    affine, weights = _affine_audit(args.campaign_root, validation)
    summary = {}
    for name in specs:
        active = [_result(args.campaign_root, name, seed) for seed in SEEDS]
        accuracies = torch.tensor([row["validation"]["accuracy"] for row in active])
        summary[name] = {
            "accuracy_mean": float(accuracies.mean()),
            "accuracy_sd": float(accuracies.std(unbiased=True)),
            "nll_mean": sum(row["validation"]["nll"] for row in active) / len(active),
            "ece_mean": sum(row["validation"]["ece"] for row in active) / len(active),
            "generalization_gap_mean": sum(row["generalization_gap"] for row in active)
            / len(active),
            "parameters": active[0]["parameters"],
        }
    payload = {
        "schema": "lnet.a2d.head_design_complete_analysis.v1",
        "completed_heads": len(summary),
        "completed_runs": len(summary) * len(SEEDS),
        "summary": summary,
        "paired_object_and_seed_bootstrap_vs_RMS256": bootstrap,
        "descriptor_geometry": _descriptor_geometry(train, train_labels),
        "affine_folding_and_attribution": affine,
        "prototype_manifold_projection_N1": _prototype_projection(
            train, train_labels, weights["N1-BNFixed"]
        ),
        "notes": {
            "bootstrap": "seeds and validation objects resampled with replacement",
            "affine_coordinates": "3 stages x 4 directions x 48 modes",
            "prototype_projection": "free affine weights centered over classes before projection",
        },
    }
    _atomic_json(args.output, payload)
    print(json.dumps({"event": "analysis_complete", "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
