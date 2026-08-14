# ruff: noqa: SLF001, T201
# pyright: reportAny=false, reportExplicitAny=false, reportMissingImports=false
"""Diagnose a trained terminal PolePyramid with frozen feature probes and rank audits."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

import run_alphabet2d_cifar100_nano as harness
import run_polepyramid_a_tiny_cifar100 as base
import torch
from torch import Tensor, nn
from torch.nn import functional
from torch.utils.data import DataLoader, Subset, TensorDataset

from lnet.polepyramid_a_tiny import PolePyramidATerminalTiny


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=401)
    return parser.parse_args()


def _loader(
    path: Path,
    split: str,
    indices: list[int] | None,
    args: argparse.Namespace,
) -> DataLoader:
    _, transform = harness._transforms()
    dataset = base._PackedCifar100(path, split, transform)
    selected = Subset(dataset, indices) if indices is not None else dataset
    return DataLoader(
        selected,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )


@torch.inference_mode()
def _extract(
    model: PolePyramidATerminalTiny,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, Tensor], Tensor]:
    buckets: dict[str, list[Tensor]] = {"gap": [], "grid2": [], "energy": []}
    labels: list[Tensor] = []
    model.eval()
    for inputs, targets in loader:
        features, early = model.transport_features(inputs.to(device, non_blocking=True))
        terminal = model.terminal(features)
        buckets["gap"].append(features.float().mean((1, 2)).cpu())
        grid = functional.adaptive_avg_pool2d(
            features.permute(0, 3, 1, 2).float(), (2, 2)
        )
        buckets["grid2"].append(grid.flatten(1).cpu())
        buckets["energy"].append(torch.cat((early, terminal), dim=-1).float().cpu())
        labels.append(targets)
    return {name: torch.cat(values) for name, values in buckets.items()}, torch.cat(labels)


def _accuracy(
    model: nn.Module,
    features: Tensor,
    targets: Tensor,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    correct = 0
    loss = 0.0
    with torch.inference_mode():
        for start in range(0, targets.numel(), 2048):
            stop = start + 2048
            logits = model(features[start:stop].to(device))
            batch_targets = targets[start:stop].to(device)
            correct += int((logits.argmax(-1) == batch_targets).sum())
            loss += float(functional.cross_entropy(logits, batch_targets, reduction="sum"))
    return {"accuracy": correct / targets.numel(), "cross_entropy": loss / targets.numel()}


def _fit_probe(
    name: str,
    training: Tensor,
    train_targets: Tensor,
    validation: Tensor,
    validation_targets: Tensor,
    test: Tensor,
    test_targets: Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    mean = training.mean(0)
    scale = training.std(0).clamp_min(1.0e-5)
    training = (training - mean) / scale
    validation = (validation - mean) / scale
    test = (test - mean) / scale
    if name == "energy_mlp":
        head: nn.Module = nn.Sequential(
            nn.Linear(training.shape[1], 512), nn.GELU(), nn.Linear(512, 100)
        )
    else:
        head = nn.Linear(training.shape[1], 100)
    head.to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=3.0e-3, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda epoch: 0.5 * (1.0 + math.cos(math.pi * epoch / max(1, args.epochs))),
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        TensorDataset(training, train_targets),
        batch_size=1024,
        shuffle=True,
        generator=generator,
    )
    best_accuracy = -1.0
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    history = []
    for epoch in range(args.epochs):
        head.train()
        for batch_features, batch_targets in loader:
            logits = head(batch_features.to(device))
            loss = functional.cross_entropy(
                logits, batch_targets.to(device), label_smoothing=0.1
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        measured = _accuracy(head, validation, validation_targets, device)
        history.append({"epoch": epoch + 1, **measured})
        if measured["accuracy"] > best_accuracy:
            best_accuracy = measured["accuracy"]
            best_epoch = epoch + 1
            best_state = copy.deepcopy(head.state_dict())
        scheduler.step()
    if best_state is None:
        message = "probe produced no validation checkpoint"
        raise RuntimeError(message)
    head.load_state_dict(best_state)
    return {
        "input_dim": training.shape[1],
        "parameters": sum(p.numel() for p in head.parameters()),
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_accuracy,
        "test": _accuracy(head, test, test_targets, device),
        "history": history,
    }


def _audit(descriptor: Tensor, targets: Tensor, classifier_weight: Tensor) -> dict[str, Any]:
    values = descriptor.double()
    centered = values - values.mean(0)
    covariance = centered.T @ centered / max(1, values.shape[0] - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    effective_rank = float(eigenvalues.sum().square() / eigenvalues.square().sum().clamp_min(1e-20))
    total_trace = float(eigenvalues.sum())
    between_trace = 0.0
    global_mean = values.mean(0)
    for class_id in range(100):
        class_values = values[targets == class_id]
        delta = class_values.mean(0) - global_mean
        between_trace += class_values.shape[0] * float(delta.square().sum()) / values.shape[0]
    direction_correlations: dict[str, list[list[float]]] = {}
    offset = 0
    for stage, modes in enumerate((16, 24, 32), start=1):
        stage_values = values[:, offset : offset + 4 * modes].reshape(values.shape[0], 4, modes)
        flattened = stage_values.permute(1, 0, 2).reshape(4, -1)
        direction_correlations[f"stage{stage}"] = torch.corrcoef(flattened).tolist()
        offset += 4 * modes
    importance = classifier_weight.float().abs().amax(0)
    threshold = 0.01 * float(importance.max())
    return {
        "samples": values.shape[0],
        "coordinates": values.shape[1],
        "coordinate_variance": values.var(0, unbiased=True).tolist(),
        "effective_rank_participation_ratio": effective_rank,
        "numerical_rank_1e-6_relative": int((eigenvalues > eigenvalues.max() * 1.0e-6).sum()),
        "between_to_within_trace_ratio": between_trace / max(1.0e-20, total_trace - between_trace),
        "direction_pearson": direction_correlations,
        "classifier_coordinates_below_1pct_max": float((importance < threshold).float().mean()),
    }


def main() -> None:
    args = _arguments()
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = PolePyramidATerminalTiny()
    state = payload.get("best_state") or payload["model"]
    model.load_state_dict(state)
    model.to(device).eval()
    packed = args.data_root / "cifar100_packed.pt"
    targets = base._load_packed(packed)["train_targets"]
    train_indices, validation_indices = harness._stratified_indices(targets)
    train_features, train_targets = _extract(
        model, _loader(packed, "train", train_indices, args), device
    )
    validation_features, validation_targets = _extract(
        model, _loader(packed, "train", validation_indices, args), device
    )
    test_features, test_targets = _extract(model, _loader(packed, "test", None, args), device)
    probes = {
        "gap_linear": _fit_probe(
            "gap_linear", train_features["gap"], train_targets,
            validation_features["gap"], validation_targets,
            test_features["gap"], test_targets, args, device,
        ),
        "grid2_linear": _fit_probe(
            "grid2_linear", train_features["grid2"], train_targets,
            validation_features["grid2"], validation_targets,
            test_features["grid2"], test_targets, args, device,
        ),
        "energy_mlp": _fit_probe(
            "energy_mlp", train_features["energy"], train_targets,
            validation_features["energy"], validation_targets,
            test_features["energy"], test_targets, args, device,
        ),
    }
    result = {
        "schema": "lnet.polepyramid_a_terminal.cifar100.frozen_probes.v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": payload["epoch"],
        "checkpoint_best_epoch": payload["best_epoch"],
        "checkpoint_best_validation_accuracy": payload["best_accuracy"],
        "probes": probes,
        "descriptor_audit": _audit(
            validation_features["energy"], validation_targets, model.classifier.weight.cpu()
        ),
    }
    args.root.mkdir(parents=True, exist_ok=True)
    harness._atomic_json(args.root / "summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
