#!/usr/bin/env python3
"""Run the post-hoc A2D frozen-Q theory and attribution analyses."""

# ruff: noqa: ANN401, EM101, EM102, FBT003, I001, PLC0415, PLR0915, SLF001, T201, TRY003

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional

import run_a2d_frozen_q_suite as frozen
from lnet.a2d_q_heads import PrototypeMetricHead
from lnet.image_layers import LowRankQuadraticModalHead


STAGES = 3
DIRECTIONS = 4
MODES = 48
DESCRIPTOR_DIM = STAGES * DIRECTIONS * MODES


class FixedPrototypeResidual(nn.Module):
    """Train a residual while holding the prototype classifier fixed."""

    def __init__(self, prototype: nn.Module, residual: nn.Module) -> None:
        super().__init__()
        self.prototype = prototype
        for parameter in self.prototype.parameters():
            parameter.requires_grad_(requires_grad=False)
        self.residual = residual
        self.beta = nn.Parameter(torch.tensor(0.1))

    def forward(self, features: Tensor) -> Tensor:
        with torch.no_grad():
            prototype_logits = self.prototype(features)
        return prototype_logits + self.beta * self.residual(features)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--lowrank-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(frozen.SEEDS))
    parser.add_argument("--head-epochs", type=int, default=30)
    parser.add_argument("--head-batch-size", type=int, default=4096)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--wandb-project",
        default=os.environ.get("WANDB_PROJECT", "alphabet2d-imagenet100"),
    )
    parser.add_argument(
        "--wandb-entity",
        default=os.environ.get("WANDB_ENTITY", "daehwa"),
    )
    return parser


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    torch.save(payload, temporary)
    temporary.replace(path)


def _metrics(logits: Tensor, labels: Tensor) -> dict[str, float]:
    return frozen._metrics(logits, labels)


def _wandb_run(args: argparse.Namespace, seed: int) -> Any:
    if not args.wandb_project or os.environ.get("WANDB_MODE") == "disabled":
        return None
    try:
        import wandb
    except ModuleNotFoundError:
        return None
    key = f"{args.output_root.resolve()}::analysis::{seed}"
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group="A2D-QAnalysis",
        job_type="frozen-q-analysis",
        name=f"QAnalysis-s{seed}",
        id=hashlib.sha256(key.encode()).hexdigest()[:16],
        resume="allow",
        dir=str(args.output_root / "wandb"),
        config={
            "seed": seed,
            "backbone": "A2D-D4-PathMix",
            "descriptor_dim": DESCRIPTOR_DIM,
        },
    )


def _load_full_linear(
    lowrank_root: Path,
    seed: int,
    data: frozen.SeedData,
) -> nn.Linear:
    path = lowrank_root / "artifacts" / f"QFull__s{seed}.pt"
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if artifact.get("schema") != "lnet.a2d.frozen_q_lowrank_artifact.v1":
        raise RuntimeError(f"unexpected QFull artifact schema at {path}")
    if not torch.allclose(artifact["normalization_mean"], data.mean.cpu(), atol=1.0e-6):
        raise RuntimeError("QFull normalization mean does not match this seed split")
    if not torch.allclose(artifact["normalization_std"], data.std.cpu(), atol=1.0e-6):
        raise RuntimeError("QFull normalization std does not match this seed split")
    model = nn.Linear(DESCRIPTOR_DIM, int(data.classes.numel())).to(data.fit.device)
    model.load_state_dict(artifact["state_dict"], strict=True)
    return model.eval()


@torch.inference_mode(False)
def _fit_temperature(calibration_logits: Tensor, labels: Tensor) -> float:
    calibration_logits = calibration_logits.detach().clone()
    labels = labels.detach().clone()
    log_temperature = nn.Parameter(torch.zeros((), device=calibration_logits.device))
    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.25,
        max_iter=80,
        tolerance_grad=1.0e-9,
        tolerance_change=1.0e-10,
        line_search_fn="strong_wolfe",
    )

    def closure() -> Tensor:
        optimizer.zero_grad(set_to_none=True)
        temperature = log_temperature.clamp(-8.0, 8.0).exp()
        loss = functional.cross_entropy(calibration_logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().clamp(-8.0, 8.0).exp())


@torch.inference_mode(False)
def _fit_bias(base_logits: Tensor, labels: Tensor, initial: Tensor) -> Tensor:
    base_logits = base_logits.detach().clone()
    labels = labels.detach().clone()
    bias = nn.Parameter(initial.detach().clone())
    optimizer = torch.optim.LBFGS(
        [bias],
        lr=0.5,
        max_iter=100,
        tolerance_grad=1.0e-8,
        tolerance_change=1.0e-10,
        line_search_fn="strong_wolfe",
    )

    def closure() -> Tensor:
        optimizer.zero_grad(set_to_none=True)
        loss = functional.cross_entropy(base_logits + bias, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return bias.detach() - bias.detach().mean()


def _calibrated_metrics(
    calibration_logits: Tensor,
    calibration_labels: Tensor,
    validation_logits: Tensor,
    validation_labels: Tensor,
) -> dict[str, Any]:
    temperature = _fit_temperature(calibration_logits, calibration_labels)
    return {
        "temperature": temperature,
        "before": _metrics(validation_logits, validation_labels),
        "after": _metrics(validation_logits / temperature, validation_labels),
    }


def _weight_fit(reference: Tensor, candidate: Tensor) -> dict[str, float]:
    residual = (reference - candidate).square().sum()
    total = reference.square().sum().clamp_min(1.0e-12)
    cosine = functional.cosine_similarity(reference.flatten(), candidate.flatten(), dim=0)
    return {
        "explained_frobenius": float(1.0 - residual / total),
        "cosine": float(cosine),
        "relative_error": float((residual / total).sqrt()),
    }


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
    output = eigenvectors @ solution @ eigenvectors.T
    return 0.5 * (output + output.T)


def _psd_projection(matrix: Tensor) -> tuple[Tensor, dict[str, float]]:
    eigenvalues, eigenvectors = torch.linalg.eigh(0.5 * (matrix + matrix.T))
    positive = eigenvalues.clamp_min(0.0)
    projected = (eigenvectors * positive) @ eigenvectors.T
    return projected, {
        "negative_eigenvalue_count": int((eigenvalues < 0.0).sum()),
        "positive_eigenvalue_count": int((eigenvalues > 0.0).sum()),
        "negative_eigenvalue_mass_fraction": float(
            eigenvalues.clamp_max(0.0).abs().sum()
            / eigenvalues.abs().sum().clamp_min(1.0e-12)
        ),
        "minimum_eigenvalue": float(eigenvalues.min()),
        "maximum_eigenvalue": float(eigenvalues.max()),
    }


def _evaluate_affine_candidate(
    *,
    name: str,
    weight: Tensor,
    initial_bias: Tensor,
    tied_bias: Tensor | None,
    data: frozen.SeedData,
    reference_weight: Tensor,
) -> dict[str, Any]:
    calibration_base = data.calibration @ weight.T
    validation_base = data.validation @ weight.T
    free_bias = _fit_bias(calibration_base, data.calibration_labels, initial_bias)
    calibration_free = calibration_base + free_bias
    validation_free = validation_base + free_bias
    output: dict[str, Any] = {
        "name": name,
        "weight_fit": _weight_fit(reference_weight, weight),
        "free_bias": _calibrated_metrics(
            calibration_free,
            data.calibration_labels,
            validation_free,
            data.validation_labels,
        ),
        "free_bias_norm": float(free_bias.norm()),
    }
    if tied_bias is not None:
        tied_bias = tied_bias - tied_bias.mean()
        calibration_tied = calibration_base + tied_bias
        validation_tied = validation_base + tied_bias
        output["tied_bias"] = _calibrated_metrics(
            calibration_tied,
            data.calibration_labels,
            validation_tied,
            data.validation_labels,
        )
        output["tied_bias_norm"] = float(tied_bias.norm())
        output["radius_correction_norm"] = float((free_bias - tied_bias).norm())
    return output


def _affine_ladder(model: nn.Linear, data: frozen.SeedData) -> dict[str, Any]:
    with torch.inference_mode():
        # Softmax is invariant to a common class weight and bias. Removing them
        # makes the class-mean prototype matrix and W share the 99-D zero-sum space.
        reference_weight = model.weight.float() - model.weight.float().mean(dim=0, keepdim=True)
        reference_bias = model.bias.float() - model.bias.float().mean()
        prototypes = data.prototypes.float()
        target = 0.5 * reference_weight
        unconstrained = torch.linalg.pinv(prototypes, rtol=1.0e-6) @ target
        symmetric_projection = 0.5 * (unconstrained + unconstrained.T)
        symmetric_optimal = _symmetric_least_squares(prototypes, target)
        psd, psd_eigen = _psd_projection(symmetric_optimal)

        candidates = []
        for name, matrix, allow_tied in (
            ("unconstrained_A", unconstrained, False),
            ("symmetrized_A", symmetric_projection, True),
            ("symmetric_optimal_A", symmetric_optimal, True),
            ("projected_PSD_A", psd, True),
        ):
            weight = 2.0 * prototypes @ matrix
            tied_bias = -(prototypes @ matrix * prototypes).sum(dim=1) if allow_tied else None
            candidates.append(
                _evaluate_affine_candidate(
                    name=name,
                    weight=weight,
                    initial_bias=reference_bias,
                    tied_bias=tied_bias,
                    data=data,
                    reference_weight=reference_weight,
                )
            )
        calibration_logits = model(data.calibration)
        validation_logits = model(data.validation)
    return {
        "free_linear": _calibrated_metrics(
            calibration_logits,
            data.calibration_labels,
            validation_logits,
            data.validation_labels,
        ),
        "prototype_rank": int(torch.linalg.matrix_rank(prototypes).item()),
        "centered_linear_weight_rank": int(torch.linalg.matrix_rank(reference_weight).item()),
        "psd_eigenspectrum": psd_eigen,
        "candidates": candidates,
    }


def _coordinate(index: int) -> dict[str, int]:
    stage, remainder = divmod(index, DIRECTIONS * MODES)
    direction, mode = divmod(remainder, MODES)
    return {"index": index, "stage": stage + 1, "direction": direction, "mode": mode}


def _margin_audit(model: nn.Linear, data: frozen.SeedData) -> tuple[dict[str, Any], Tensor]:
    with torch.inference_mode():
        logits = model(data.validation)
        labels = data.validation_labels
        rows = torch.arange(labels.numel(), device=labels.device)
        competitors = logits.clone()
        competitors[rows, labels] = -torch.inf
        competitor = competitors.argmax(dim=1)
        delta_weight = model.weight[labels] - model.weight[competitor]
        coordinate_contribution = data.validation * delta_weight
        bias_contribution = model.bias[labels] - model.bias[competitor]
        margin = coordinate_contribution.sum(dim=1) + bias_contribution
        direct_margin = logits[rows, labels] - logits[rows, competitor]
        correct = logits.argmax(dim=1).eq(labels)
        shaped = coordinate_contribution.reshape(-1, STAGES, DIRECTIONS, MODES)
        absolute = coordinate_contribution.abs()
        total_absolute = absolute.sum(dim=1).clamp_min(1.0e-12)
        sorted_absolute = absolute.sort(dim=1, descending=True).values
        concentration = {
            str(count): float(sorted_absolute[:, :count].sum(dim=1).div(total_absolute).mean())
            for count in (1, 5, 10, 32, 64, 128)
        }
        supporting = coordinate_contribution.argmax(dim=1)
        opposing = coordinate_contribution.argmin(dim=1)
        support_counts = torch.bincount(supporting, minlength=DESCRIPTOR_DIM)
        oppose_counts = torch.bincount(opposing, minlength=DESCRIPTOR_DIM)
        top_support = support_counts.topk(20)
        top_oppose = oppose_counts.topk(20)
        stage_signed = shaped.sum(dim=(2, 3))
        direction_signed = shaped.sum(dim=3)
        mode_absolute = shaped.abs().sum(dim=(1, 2))
        class_accuracy = []
        class_margin = []
        for class_value in data.classes:
            active = labels == class_value
            class_accuracy.append(float(correct[active].float().mean()))
            class_margin.append(float(margin[active].median()))
        raw_weight = model.weight.float() / data.std[None, :]
        raw_weight = raw_weight - raw_weight.mean(dim=0, keepdim=True)
    result = {
        "validation": _metrics(logits, labels),
        "margin_reconstruction_max_error": float((margin - direct_margin).abs().max()),
        "margin": {
            "mean": float(margin.mean()),
            "median": float(margin.median()),
            "correct_mean": float(margin[correct].mean()),
            "incorrect_mean": float(margin[~correct].mean()),
            "positive_fraction": float((margin > 0).float().mean()),
        },
        "absolute_contribution_concentration": concentration,
        "stage_signed_mean": stage_signed.mean(dim=0).cpu().tolist(),
        "stage_signed_correct_mean": stage_signed[correct].mean(dim=0).cpu().tolist(),
        "stage_signed_incorrect_mean": stage_signed[~correct].mean(dim=0).cpu().tolist(),
        "direction_signed_mean": direction_signed.mean(dim=0).cpu().tolist(),
        "mode_absolute_mean": mode_absolute.mean(dim=0).cpu().tolist(),
        "top_supporting_coordinates": [
            {**_coordinate(int(index)), "count": int(count)}
            for count, index in zip(top_support.values, top_support.indices, strict=True)
        ],
        "top_opposing_coordinates": [
            {**_coordinate(int(index)), "count": int(count)}
            for count, index in zip(top_oppose.values, top_oppose.indices, strict=True)
        ],
        "per_class_accuracy": class_accuracy,
        "per_class_median_margin": class_margin,
    }
    return result, raw_weight.detach().cpu()


def _fisher_order(data: frozen.SeedData) -> Tensor:
    means = torch.stack(
        [data.fit[data.fit_labels == class_value].mean(dim=0) for class_value in data.classes]
    )
    within = torch.stack(
        [
            data.fit[data.fit_labels == class_value].var(dim=0, unbiased=True)
            for class_value in data.classes
        ]
    ).mean(dim=0)
    between = (means - means.mean(dim=0, keepdim=True)).square().mean(dim=0)
    return (between / within.clamp_min(1.0e-8)).argsort(descending=True)


def _ablation_indices(data: frozen.SeedData, *, smoke: bool) -> dict[str, Tensor]:
    device = data.fit.device
    stage = [torch.arange(s * 192, (s + 1) * 192, device=device) for s in range(STAGES)]
    direction = []
    for d in range(DIRECTIONS):
        pieces = [
            torch.arange(s * 192 + d * 48, s * 192 + (d + 1) * 48, device=device)
            for s in range(STAGES)
        ]
        direction.append(torch.cat(pieces))
    output: dict[str, Tensor] = {
        "S1": stage[0],
        "S2": stage[1],
        "S3": stage[2],
        "S12": torch.cat((stage[0], stage[1])),
        "S13": torch.cat((stage[0], stage[2])),
        "S23": torch.cat((stage[1], stage[2])),
    }
    for d in range(DIRECTIONS):
        output[f"D{d}"] = direction[d]
        output[f"NoD{d}"] = torch.cat(
            [direction[index] for index in range(DIRECTIONS) if index != d]
        )
    for s in range(STAGES):
        for d in range(DIRECTIONS):
            start = s * 192 + d * 48
            output[f"S{s + 1}D{d}"] = torch.arange(start, start + 48, device=device)
    fisher = _fisher_order(data)
    for count in (32, 64, 128, 256, 384):
        output[f"Fisher{count}"] = fisher[:count]
    if smoke:
        return {name: output[name] for name in ("S1", "S3", "D0", "NoD0", "Fisher32")}
    return output


def _fit_head(
    model: nn.Module,
    data: frozen.SeedData,
    args: argparse.Namespace,
    *,
    seed: int,
) -> tuple[nn.Module, list[dict[str, float]]]:
    return frozen._fit_model(
        model,
        data,
        epochs=args.head_epochs,
        batch_size=args.head_batch_size,
        seed=seed,
    )


def _linear_ablation(
    data: frozen.SeedData,
    args: argparse.Namespace,
    *,
    seed: int,
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for name, indices in _ablation_indices(data, smoke=args.smoke_test).items():
        active = replace(
            data,
            fit=data.fit[:, indices],
            calibration=data.calibration[:, indices],
            validation=data.validation[:, indices],
        )
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = nn.Linear(indices.numel(), int(data.classes.numel())).to(data.fit.device)
        started = time.perf_counter()
        model, history = _fit_head(model, active, args, seed=seed)
        with torch.inference_mode():
            logits = model(active.validation)
        stage_counts = torch.bincount(indices // 192, minlength=3)
        rows[name] = {
            "dimensions": int(indices.numel()),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "validation": _metrics(logits, active.validation_labels),
            "best_calibration_nll": min(row["calibration_nll"] for row in history),
            "stage_coordinate_counts": stage_counts.cpu().tolist(),
            "seconds": time.perf_counter() - started,
        }
        print(
            json.dumps(
                {
                    "event": "ablation_complete",
                    "seed": seed,
                    "name": name,
                    "accuracy": rows[name]["validation"]["accuracy"],
                }
            ),
            flush=True,
        )
    return rows


def _train_prototype(
    data: frozen.SeedData,
    args: argparse.Namespace,
    *,
    seed: int,
    components: int,
) -> tuple[PrototypeMetricHead, list[dict[str, float]]]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    prototypes = frozen._prototypes_for_components(data, components)
    model = PrototypeMetricHead(
        prototypes,
        classes=int(data.classes.numel()),
        components=components,
        initial_diagonal=data.diagonal,
        rank=32,
        learn_temperature=True,
    ).to(data.fit.device)
    trained, history = _fit_head(model, data, args, seed=seed)
    return trained, history


def _component_usage(model: PrototypeMetricHead, data: frozen.SeedData) -> dict[str, Any]:
    components = model.components
    classes = model.classes
    with torch.inference_mode():
        component_logits = model.component_logits(data.validation).reshape(-1, classes, components)
        labels = data.validation_labels
        rows = torch.arange(labels.numel(), device=labels.device)
        true_components = component_logits[rows, labels]
        temperature = (
            float(model.log_temperature.detach().clamp(-2.0, 2.0).exp())
            if model.log_temperature is not None
            else 1.0
        )
        responsibility = (true_components / temperature).softmax(dim=1)
        hard = responsibility.argmax(dim=1)
        usage = torch.zeros(classes, components, device=labels.device)
        soft_usage = torch.zeros_like(usage)
        entropy = torch.zeros(classes, device=labels.device)
        active_count = torch.zeros(classes, device=labels.device)
        for class_index in range(classes):
            active = labels == class_index
            counts = torch.bincount(hard[active], minlength=components).float()
            usage[class_index] = counts / counts.sum().clamp_min(1.0)
            soft_usage[class_index] = responsibility[active].mean(dim=0)
            probability = soft_usage[class_index].clamp_min(1.0e-12)
            entropy[class_index] = -(probability * probability.log()).sum() / math.log(components)
            active_count[class_index] = (usage[class_index] >= 0.05).sum()
        prototypes = model.prototypes.reshape(classes, components, -1)
        within_component_distance = torch.stack(
            [torch.pdist(prototypes[class_index]).mean() for class_index in range(classes)]
        )
    return {
        "temperature": temperature,
        "hard_usage": usage.cpu().tolist(),
        "soft_usage": soft_usage.cpu().tolist(),
        "normalized_entropy": entropy.cpu().tolist(),
        "entropy_mean": float(entropy.mean()),
        "entropy_minimum": float(entropy.min()),
        "entropy_maximum": float(entropy.max()),
        "active_components_mean": float(active_count.mean()),
        "collapsed_class_count": int((active_count <= 1).sum()),
        "within_class_component_distance_mean": float(within_component_distance.mean()),
        "within_class_component_distance_minimum": float(within_component_distance.min()),
    }


def _prototype_analysis(
    data: frozen.SeedData,
    args: argparse.Namespace,
    *,
    seed: int,
) -> tuple[dict[str, Any], PrototypeMetricHead, PrototypeMetricHead]:
    k1, k1_history = _train_prototype(data, args, seed=seed, components=1)
    k4, k4_history = _train_prototype(data, args, seed=seed, components=4)
    with torch.inference_mode():
        k1_calibration = k1(data.calibration)
        k1_validation = k1(data.validation)
        k4_calibration = k4(data.calibration)
        k4_validation = k4(data.validation)
        labels = data.validation_labels
        k1_prediction = k1_validation.argmax(dim=1)
        k4_prediction = k4_validation.argmax(dim=1)
        per_class_delta = []
        for class_value in data.classes:
            active = labels == class_value
            per_class_delta.append(
                float(
                    k4_prediction[active].eq(labels[active]).float().mean()
                    - k1_prediction[active].eq(labels[active]).float().mean()
                )
            )
    return (
        {
            "K1_PSD32": {
                "calibration": _calibrated_metrics(
                    k1_calibration,
                    data.calibration_labels,
                    k1_validation,
                    data.validation_labels,
                ),
                "best_calibration_nll": min(row["calibration_nll"] for row in k1_history),
            },
            "K4_PSD32": {
                "calibration": _calibrated_metrics(
                    k4_calibration,
                    data.calibration_labels,
                    k4_validation,
                    data.validation_labels,
                ),
                "best_calibration_nll": min(row["calibration_nll"] for row in k4_history),
                "usage": _component_usage(k4, data),
                "per_class_accuracy_delta_vs_K1": per_class_delta,
                "classes_improved": sum(delta > 0.0 for delta in per_class_delta),
                "classes_hurt": sum(delta < 0.0 for delta in per_class_delta),
            },
        },
        k1,
        k4,
    )


@torch.inference_mode(False)
def _fit_positive_ensemble(
    prototype_logits: Tensor,
    residual_logits: Tensor,
    labels: Tensor,
) -> Tensor:
    prototype_logits = prototype_logits.detach().clone()
    residual_logits = residual_logits.detach().clone()
    labels = labels.detach().clone()
    log_scales = nn.Parameter(torch.zeros(2, device=prototype_logits.device))
    optimizer = torch.optim.LBFGS(
        [log_scales],
        lr=0.25,
        max_iter=80,
        tolerance_grad=1.0e-9,
        tolerance_change=1.0e-10,
        line_search_fn="strong_wolfe",
    )

    def closure() -> Tensor:
        optimizer.zero_grad(set_to_none=True)
        scales = log_scales.clamp(-8.0, 8.0).exp()
        loss = functional.cross_entropy(
            scales[0] * prototype_logits + scales[1] * residual_logits,
            labels,
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    return log_scales.detach().clamp(-8.0, 8.0).exp()


def _residual_analysis(
    k4: PrototypeMetricHead,
    data: frozen.SeedData,
    args: argparse.Namespace,
    *,
    seed: int,
) -> dict[str, Any]:
    classes = int(data.classes.numel())
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    independent = LowRankQuadraticModalHead(DESCRIPTOR_DIM, classes, 16).to(data.fit.device)
    independent, _ = _fit_head(independent, data, args, seed=seed)
    with torch.inference_mode():
        proto_cal = k4(data.calibration)
        proto_val = k4(data.validation)
        independent_cal = independent(data.calibration)
        independent_val = independent(data.validation)
    scales = _fit_positive_ensemble(
        proto_cal,
        independent_cal,
        data.calibration_labels,
    )
    independent_joint = scales[0] * proto_val + scales[1] * independent_val

    fixed_prototype = deepcopy_module(k4)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    residual = LowRankQuadraticModalHead(DESCRIPTOR_DIM, classes, 16).to(data.fit.device)
    fixed_model = FixedPrototypeResidual(fixed_prototype, residual).to(data.fit.device)
    fixed_model, _ = _fit_head(fixed_model, data, args, seed=seed)
    with torch.inference_mode():
        fixed_proto = fixed_model.prototype(data.validation)
        fixed_residual = fixed_model.residual(data.validation)
        fixed_joint = fixed_proto + fixed_model.beta * fixed_residual
    return {
        "independently_trained_positive_ensemble": {
            "prototype_scale": float(scales[0]),
            "lrq_scale": float(scales[1]),
            "prototype_only": _metrics(proto_val, data.validation_labels),
            "lrq_only": _metrics(independent_val, data.validation_labels),
            "joint": _metrics(independent_joint, data.validation_labels),
        },
        "fixed_prototype_trained_residual": {
            "beta": float(fixed_model.beta.detach()),
            "prototype_only": _metrics(fixed_proto, data.validation_labels),
            "lrq_only": _metrics(fixed_residual, data.validation_labels),
            "joint": _metrics(fixed_joint, data.validation_labels),
        },
    }


def deepcopy_module(model: nn.Module) -> nn.Module:
    # State-dict round trips would require reconstructing every head type; deepcopy
    # is safe here because these frozen heads own no external runtime resources.
    import copy

    return copy.deepcopy(model)


def _calibrate_lowrank_artifacts(
    lowrank_root: Path,
    data: frozen.SeedData,
    *,
    seed: int,
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for path in sorted((lowrank_root / "artifacts").glob(f"Q*__s{seed}.pt")):
        artifact = torch.load(path, map_location="cpu", weights_only=True)
        state = artifact["state_dict"]
        name = path.stem.split("__", maxsplit=1)[0]
        if name == "QFull":
            model: nn.Module = nn.Linear(DESCRIPTOR_DIM, int(data.classes.numel()))
        else:
            rank = int(name.removeprefix("QRank"))
            from run_a2d_frozen_q_lowrank import FactorizedAffineHead

            model = FactorizedAffineHead(DESCRIPTOR_DIM, int(data.classes.numel()), rank)
        model.load_state_dict(state, strict=True)
        model = model.to(data.fit.device).eval()
        with torch.inference_mode():
            calibration_logits = model(data.calibration)
            validation_logits = model(data.validation)
        rows[name] = _calibrated_metrics(
            calibration_logits,
            data.calibration_labels,
            validation_logits,
            data.validation_labels,
        )
    return rows


def _cross_seed_stability(raw_weights: dict[int, Tensor]) -> dict[str, Any]:
    pairs: dict[str, Any] = {}
    seeds = sorted(raw_weights)
    for left_index, left_seed in enumerate(seeds):
        for right_seed in seeds[left_index + 1 :]:
            left = raw_weights[left_seed].float()
            right = raw_weights[right_seed].float()
            cosine = functional.cosine_similarity(left, right, dim=1)
            jaccard = []
            for class_index in range(left.shape[0]):
                left_top = set(left[class_index].abs().topk(32).indices.tolist())
                right_top = set(right[class_index].abs().topk(32).indices.tolist())
                jaccard.append(len(left_top & right_top) / len(left_top | right_top))
            pairs[f"{left_seed}-{right_seed}"] = {
                "class_weight_cosine_mean": float(cosine.mean()),
                "class_weight_cosine_median": float(cosine.median()),
                "class_weight_cosine_minimum": float(cosine.min()),
                "top32_coordinate_jaccard_mean": sum(jaccard) / len(jaccard),
            }
    return pairs


def _seed_summary(result: dict[str, Any]) -> dict[str, float]:
    prototype = result["prototype"]["K4_PSD32"]["calibration"]
    independent = result["residual"]["independently_trained_positive_ensemble"]
    fixed = result["residual"]["fixed_prototype_trained_residual"]
    output = {
        "affine/free_accuracy": result["affine_ladder"]["free_linear"]["before"]["accuracy"],
        "prototype/k4_accuracy": prototype["before"]["accuracy"],
        "prototype/k4_calibrated_nll": prototype["after"]["nll"],
        "residual/independent_joint_accuracy": independent["joint"]["accuracy"],
        "residual/fixed_joint_accuracy": fixed["joint"]["accuracy"],
    }
    for candidate in result["affine_ladder"]["candidates"]:
        free_key = f"affine/{candidate['name']}_free_bias_accuracy"
        output[free_key] = candidate["free_bias"]["before"]["accuracy"]
        if "tied_bias" in candidate:
            tied_key = f"affine/{candidate['name']}_tied_bias_accuracy"
            output[tied_key] = candidate["tied_bias"]["before"]["accuracy"]
    return output


def main() -> None:
    args = _parser().parse_args()
    if args.smoke_test:
        cache = frozen._synthetic_cache()
        args.seeds = [args.seeds[0]]
        args.head_epochs = 1
        args.head_batch_size = min(args.head_batch_size, 32)
        args.wandb_project = ""
    else:
        cache = frozen._load_cache(args.cache_root)
    train_features, train_labels, validation_features, validation_labels, metadata = cache
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("frozen-Q analysis cannot see a CUDA GPU")
    args.output_root.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        args.output_root / "contract.json",
        {
            "schema": "lnet.a2d.frozen_q_analysis_contract.v1",
            "seeds": args.seeds,
            "head_epochs": args.head_epochs,
            "head_batch_size": args.head_batch_size,
            "cache": metadata,
            "analyses": [
                "affine_constraint_ladder",
                "prototype_manifold_projection",
                "exact_margin_attribution",
                "linear_stage_direction_fisher_ablation",
                "K4_component_usage",
                "temperature_calibration",
                "fixed_vs_independent_residual",
            ],
        },
    )
    raw_weights: dict[int, Tensor] = {}
    failures = 0
    for seed in args.seeds:
        result_path = args.output_root / "results" / f"QAnalysis__s{seed}.json"
        weight_path = args.output_root / "artifacts" / f"QRawWeight__s{seed}.pt"
        if result_path.exists() and weight_path.exists():
            raw_weights[seed] = torch.load(weight_path, map_location="cpu", weights_only=True)[
                "raw_weight"
            ]
            continue
        try:
            data = frozen._seed_data(
                train_features,
                train_labels,
                validation_features,
                validation_labels,
                seed=seed,
                device=device,
            )
            full = _load_full_linear(args.lowrank_root, seed, data)
            run = _wandb_run(args, seed)
            started = time.perf_counter()
            margin, raw_weight = _margin_audit(full, data)
            raw_weights[seed] = raw_weight
            result = {
                "schema": "lnet.a2d.frozen_q_analysis.v1",
                "seed": seed,
                "cache": metadata,
                "affine_ladder": _affine_ladder(full, data),
                "margin_audit": margin,
                "linear_ablation": _linear_ablation(data, args, seed=seed),
            }
            prototype, _, k4 = _prototype_analysis(data, args, seed=seed)
            result["prototype"] = prototype
            result["lowrank_calibration"] = _calibrate_lowrank_artifacts(
                args.lowrank_root,
                data,
                seed=seed,
            )
            result["residual"] = _residual_analysis(k4, data, args, seed=seed)
            result["seconds"] = time.perf_counter() - started
            _atomic_json(result_path, result)
            _atomic_torch_save(weight_path, {"raw_weight": raw_weight, "seed": seed})
            if run is not None:
                run.log(_seed_summary(result))
                run.summary.update(_seed_summary(result))
                run.finish()
            print(
                json.dumps(
                    {
                        "event": "seed_complete",
                        "seed": seed,
                        "seconds": result["seconds"],
                    }
                ),
                flush=True,
            )
        except Exception as error:  # noqa: BLE001
            failures += 1
            failure = {
                "seed": seed,
                "error": repr(error),
                "traceback": traceback.format_exc(),
            }
            _atomic_json(args.output_root / "failures" / f"QAnalysis__s{seed}.json", failure)
            print(json.dumps({"event": "seed_failure", **failure}), flush=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
    if len(raw_weights) >= 2:
        _atomic_json(
            args.output_root / "cross_seed_stability.json",
            {
                "schema": "lnet.a2d.q_margin_stability.v1",
                "pairs": _cross_seed_stability(raw_weights),
            },
        )
    completed = len(list((args.output_root / "results").glob("QAnalysis__s*.json")))
    _atomic_json(
        args.output_root / "summary.json",
        {
            "schema": "lnet.a2d.frozen_q_analysis_summary.v1",
            "completed_seeds": completed,
            "expected_seeds": len(args.seeds),
            "failures": failures,
        },
    )
    if failures:
        raise RuntimeError(f"{failures} frozen-Q analysis seeds failed")


if __name__ == "__main__":
    main()
