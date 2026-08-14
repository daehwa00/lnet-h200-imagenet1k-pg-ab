#!/usr/bin/env python3
"""Probe gated cross-stage corrections on frozen D4-M64 ImageNet-100 Q."""

# Experimental runner intentionally uses private harness construction hooks and
# JSON-shaped payloads while emitting machine-readable progress to stdout.
# ruff: noqa: ANN401, EM101, EM102, PERF401, PLR0915, SLF001, T201, TRY003

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from copy import deepcopy
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

import run_a2d_deep4_backbone_variants_imagenet100 as m64_runner
import run_alphabet2d_imagenet100_nano as harness
import torch
from torch import Tensor, nn
from torch.nn import functional
from torch.utils.data import DataLoader
from torchvision import datasets

from lnet.a2d_gated_stage_head import (
    FixedAffineMain,
    FixedStandardizedMain,
    GatedCrossStageResidualHead,
)
from lnet.complex_scan import ComplexScanConfig

STAGES = 4
STAGE_DIM = 256
DESCRIPTOR_DIM = STAGES * STAGE_DIM


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--extract-batch-size", type=int, default=128)
    parser.add_argument("--head-batch-size", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=501)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--residual-width", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--residual-main", choices=("affine", "fusion"), default="affine")
    parser.add_argument("--candidate-names", nargs="*", default=[])
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _build_model(checkpoint: Path, device: torch.device) -> nn.Module:
    config = ComplexScanConfig(
        output_dim=100,
        stem_strides=(2, 2),
    )
    model = m64_runner._build(m64_runner.UNIFORM_M64, config)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("variant") != m64_runner.UNIFORM_M64:
        raise ValueError("checkpoint is not the trained D4-M64 variant")
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device).to(memory_format=torch.channels_last).eval()


def _loader(root: Path, split: str, batch_size: int, workers: int) -> DataLoader[Any]:
    _, evaluation_transform = harness._transforms()
    dataset = datasets.ImageFolder(root / split, evaluation_transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=harness.PREFETCH_FACTOR if workers > 0 else None,
    )


@torch.inference_mode()
def _extract(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    features: list[Tensor] = []
    labels: list[Tensor] = []
    started = time.perf_counter()
    for index, (inputs, targets) in enumerate(loader):
        device_inputs = inputs.to(device, non_blocking=True).contiguous(
            memory_format=torch.channels_last
        )
        descriptor = cast("Any", model).raw_descriptor(device_inputs)
        if descriptor.shape[-1] != DESCRIPTOR_DIM:
            raise RuntimeError("D4-M64 descriptor dimension changed")
        features.append(descriptor.float().cpu())
        labels.append(targets.long())
        if (index + 1) % 100 == 0:
            count = sum(batch.shape[0] for batch in labels)
            elapsed = time.perf_counter() - started
            print(
                json.dumps(
                    {
                        "event": "extract_progress",
                        "samples": count,
                        "images_per_second": count / elapsed,
                    }
                ),
                flush=True,
            )
    return torch.cat(features).contiguous(), torch.cat(labels).contiguous()


def _cached_split(
    model: nn.Module,
    *,
    data_root: Path,
    split: str,
    cache_path: Path,
    checkpoint_sha256: str,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> tuple[Tensor, Tensor]:
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        if payload.get("checkpoint_sha256") != checkpoint_sha256:
            raise RuntimeError(f"stale descriptor cache: {cache_path}")
        return payload["features"].float(), payload["labels"].long()
    loader = _loader(data_root, split, batch_size, workers)
    features, labels = _extract(model, loader, device)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".tmp.pt")
    torch.save(
        {
            "features": features,
            "labels": labels,
            "checkpoint_sha256": checkpoint_sha256,
            "split": split,
        },
        temporary,
    )
    temporary.replace(cache_path)
    return features, labels


def _fixed_affine(classifier: Any) -> FixedAffineMain:
    affine = classifier.affine
    standardizer = affine.standardizer
    if not isinstance(standardizer, nn.BatchNorm1d) or standardizer.affine:
        raise TypeError("D4-M64 affine branch no longer uses parameter-free BatchNorm")
    bias = affine.linear.bias
    if bias is None:
        raise TypeError("D4-M64 affine branch lost its bias")
    return FixedAffineMain(
        standardizer.running_mean,
        standardizer.running_var,
        affine.linear.weight,
        bias,
        eps=standardizer.eps,
    )


def _split_indices(labels: Tensor, seed: int) -> tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    fit: list[Tensor] = []
    calibration: list[Tensor] = []
    for label in labels.unique(sorted=True):
        indices = torch.nonzero(labels == label, as_tuple=False).flatten()
        indices = indices[torch.randperm(indices.numel(), generator=generator)]
        calibration_count = max(1, round(0.1 * indices.numel()))
        calibration.append(indices[:calibration_count])
        fit.append(indices[calibration_count:])
    return torch.cat(fit), torch.cat(calibration)


@torch.inference_mode()
def _logits(model: nn.Module, features: Tensor, device: torch.device, batch: int) -> Tensor:
    model.eval()
    output = []
    for start in range(0, features.shape[0], batch):
        output.append(model(features[start : start + batch].to(device)).float().cpu())
    return torch.cat(output)


def _metrics(logits: Tensor, labels: Tensor) -> dict[str, float]:
    probabilities = logits.softmax(dim=-1)
    confidence, prediction = probabilities.max(dim=-1)
    accuracy = prediction.eq(labels).float().mean()
    ece = torch.zeros(())
    edges = torch.linspace(0.0, 1.0, 16)
    for left, right in pairwise(edges):
        active = (confidence > left) & (confidence <= right)
        if active.any():
            bin_accuracy = prediction[active].eq(labels[active]).float().mean()
            ece += active.float().mean() * (bin_accuracy - confidence[active].mean()).abs()
    return {
        "accuracy": float(accuracy),
        "nll": float(functional.cross_entropy(logits, labels)),
        "ece": float(ece),
    }


def _fit_candidate(
    model: GatedCrossStageResidualHead,
    *,
    train_features: Tensor,
    train_labels: Tensor,
    calibration_features: Tensor,
    calibration_labels: Tensor,
    validation_features: Tensor,
    validation_labels: Tensor,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], Tensor]:
    model.to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        fused=device.type == "cuda",
    )
    generator = torch.Generator().manual_seed(args.seed)
    best_nll = float("inf")
    best_state = deepcopy(model.state_dict())
    history = []
    for epoch in range(args.epochs):
        model.train()
        permutation = torch.randperm(train_features.shape[0], generator=generator)
        for start in range(0, permutation.numel(), args.head_batch_size):
            indices = permutation[start : start + args.head_batch_size]
            inputs = train_features[indices].to(device, non_blocking=True)
            targets = train_labels[indices].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = functional.cross_entropy(model(inputs), targets)
            loss.backward()
            optimizer.step()
        calibration_logits = _logits(
            model, calibration_features, device, args.head_batch_size
        )
        calibration = _metrics(calibration_logits, calibration_labels)
        history.append({"epoch": epoch + 1, **calibration, "beta": float(model.beta.detach())})
        if calibration["nll"] < best_nll:
            best_nll = calibration["nll"]
            best_state = deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    validation_logits = _logits(model, validation_features, device, args.head_batch_size)
    with torch.inference_mode():
        gates = []
        corrections = []
        for start in range(0, validation_features.shape[0], args.head_batch_size):
            descriptor = validation_features[start : start + args.head_batch_size].to(device)
            _joint, _affine, residual, gate = model.components(descriptor)
            gates.append(gate.float().cpu())
            corrections.append((model.beta * gate * residual).float().cpu())
    gate_values = torch.cat(gates)
    correction = torch.cat(corrections)
    result = {
        "metrics": _metrics(validation_logits, validation_labels),
        "parameters_total": sum(parameter.numel() for parameter in model.parameters()),
        "parameters_trainable": sum(parameter.numel() for parameter in parameters),
        "beta": float(model.beta.detach()),
        "gate_mean": float(gate_values.mean()),
        "gate_std": float(gate_values.std(unbiased=False)),
        "correction_rms": float(correction.square().mean().sqrt()),
        "history": history,
    }
    return result, validation_logits


@torch.inference_mode()
def _main_margin_thresholds(
    main: FixedAffineMain | FixedStandardizedMain,
    features: Tensor,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, tuple[float, float]], dict[str, float]]:
    logits = _logits(main.to(device), features, device, batch_size)
    top_two = logits.topk(2, dim=-1).values
    margins = top_two[:, 0] - top_two[:, 1]
    quantile_levels = torch.tensor([0.05, 0.1, 0.15, 0.2, 0.25, 0.4, 0.75])
    quantiles = torch.quantile(margins, quantile_levels)
    scale = max(0.1, float(quantiles[6] - quantiles[4]) * 0.15)
    thresholds = {
        "margin_q05": (float(quantiles[0]), scale),
        "margin_q10": (float(quantiles[1]), scale),
        "margin_q15": (float(quantiles[2]), scale),
        "margin_q20": (float(quantiles[3]), scale),
        "margin_q40": (float(quantiles[5]), scale),
    }
    summary = {
        f"q{round(100 * float(level)):02d}": float(value)
        for level, value in zip(quantile_levels, quantiles, strict=True)
    }
    summary.update({
        "temperature": scale,
    })
    return thresholds, summary


def _intervention(
    base_logits: Tensor,
    candidate_logits: Tensor,
    labels: Tensor,
) -> dict[str, float | int]:
    base = base_logits.argmax(dim=-1)
    candidate = candidate_logits.argmax(dim=-1)
    changed = base.ne(candidate)
    base_correct = base.eq(labels)
    candidate_correct = candidate.eq(labels)
    return {
        "agreement": float((~changed).float().mean()),
        "changed": int(changed.sum()),
        "fixed": int((~base_correct & candidate_correct).sum()),
        "broken": int((base_correct & ~candidate_correct).sum()),
        "net_correct": int(candidate_correct.sum() - base_correct.sum()),
        "both_wrong_changed": int((changed & ~base_correct & ~candidate_correct).sum()),
    }


def main() -> None:
    args = _arguments()
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA was requested but is unavailable")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    args.output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_sha256 = _sha256(args.checkpoint)
    model = _build_model(args.checkpoint, device)
    classifier = cast("Any", model).classifier
    fixed_affine = _fixed_affine(classifier).cpu()
    fixed_fusion = FixedStandardizedMain(
        classifier.fusion,
        input_dim=DESCRIPTOR_DIM,
        output_dim=100,
    ).cpu()
    reference_fusion = deepcopy(classifier.fusion).cpu().eval()
    if args.smoke_test:
        generator = torch.Generator().manual_seed(args.seed)
        train_labels = torch.arange(4).repeat_interleave(32)
        validation_labels = torch.arange(4).repeat_interleave(8)
        centers = torch.randn(4, DESCRIPTOR_DIM, generator=generator)
        train_features = centers[train_labels] + 0.2 * torch.randn(
            train_labels.shape[0], DESCRIPTOR_DIM, generator=generator
        )
        validation_features = centers[validation_labels] + 0.2 * torch.randn(
            validation_labels.shape[0], DESCRIPTOR_DIM, generator=generator
        )
        # Synthetic mode only checks the training path; its random classifier
        # is intentionally unrelated to the generated labels.
        args.epochs = min(args.epochs, 2)
        args.head_batch_size = min(args.head_batch_size, 64)
    else:
        train_features, train_labels = _cached_split(
            model,
            data_root=args.data_root,
            split="train",
            cache_path=args.output_root / "cache" / "train-center-crop.pt",
            checkpoint_sha256=checkpoint_sha256,
            device=device,
            batch_size=args.extract_batch_size,
            workers=args.workers,
        )
        validation_features, validation_labels = _cached_split(
            model,
            data_root=args.data_root,
            split="val",
            cache_path=args.output_root / "cache" / "val-center-crop.pt",
            checkpoint_sha256=checkpoint_sha256,
            device=device,
            batch_size=args.extract_batch_size,
            workers=args.workers,
        )
    del model
    torch.cuda.empty_cache()
    fit_indices, calibration_indices = _split_indices(train_labels, args.seed)
    reference_affine_logits = _logits(
        fixed_affine.to(device), validation_features, device, args.head_batch_size
    )
    reference_fusion_logits = _logits(
        reference_fusion.to(device), validation_features, device, args.head_batch_size
    )
    fixed_main = fixed_affine if args.residual_main == "affine" else fixed_fusion
    results: dict[str, Any] = {
        "schema": "lnet.a2d.m64_gated_stage_head_probe.v2",
        "checkpoint_sha256": checkpoint_sha256,
        "residual_main": args.residual_main,
        "seed": args.seed,
        "epochs": args.epochs,
        "samples": {
            "fit": fit_indices.numel(),
            "calibration": calibration_indices.numel(),
            "validation": validation_labels.numel(),
        },
        "reference_affine": _metrics(reference_affine_logits, validation_labels),
        "reference_fusion": _metrics(reference_fusion_logits, validation_labels),
        "fusion_vs_affine": _intervention(
            reference_affine_logits, reference_fusion_logits, validation_labels
        ),
        "candidates": {},
    }
    margin_thresholds, margin_summary = _main_margin_thresholds(
        deepcopy(fixed_main),
        train_features[fit_indices],
        device,
        args.head_batch_size,
    )
    results["fit_main_margin"] = margin_summary
    candidate_specs = (
        ("cross_ungated", False, None),
        ("cross_learned_gate", True, None),
        ("cross_margin_q05", True, margin_thresholds["margin_q05"]),
        ("cross_margin_q10", True, margin_thresholds["margin_q10"]),
        ("cross_margin_q15", True, margin_thresholds["margin_q15"]),
        ("cross_margin_q20", True, margin_thresholds["margin_q20"]),
        ("cross_margin_q40", True, margin_thresholds["margin_q40"]),
    )
    requested_candidates = set(args.candidate_names)
    known_candidates = {name for name, _gated, _margin_gate in candidate_specs}
    unknown_candidates = requested_candidates - known_candidates
    if unknown_candidates:
        message = f"unknown candidates: {sorted(unknown_candidates)}"
        raise ValueError(message)
    for name, gated, margin_gate in candidate_specs:
        if requested_candidates and name not in requested_candidates:
            continue
        torch.manual_seed(args.seed)
        candidate = GatedCrossStageResidualHead(
            deepcopy(fixed_main),
            stage_count=STAGES,
            stage_dim=STAGE_DIM,
            embedding_dim=args.embedding_dim,
            residual_width=args.residual_width,
            gated=gated,
            affine_margin_threshold=None if margin_gate is None else margin_gate[0],
            affine_margin_temperature=1.0 if margin_gate is None else margin_gate[1],
        )
        result, logits = _fit_candidate(
            candidate,
            train_features=train_features[fit_indices],
            train_labels=train_labels[fit_indices],
            calibration_features=train_features[calibration_indices],
            calibration_labels=train_labels[calibration_indices],
            validation_features=validation_features,
            validation_labels=validation_labels,
            device=device,
            args=args,
        )
        result["vs_affine"] = _intervention(
            reference_affine_logits, logits, validation_labels
        )
        result["vs_fusion"] = _intervention(reference_fusion_logits, logits, validation_labels)
        results["candidates"][name] = result
        print(json.dumps({"event": "candidate_complete", "name": name, **result["metrics"]}))
    result_name = f"results-{args.residual_main}-main-s{args.seed}.json"
    result_path = args.output_root / result_name
    _atomic_json(result_path, results)
    print(json.dumps({"event": "probe_complete", "output": str(result_path)}))


if __name__ == "__main__":
    main()
