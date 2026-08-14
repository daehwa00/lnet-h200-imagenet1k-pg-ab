from __future__ import annotations

import json

import torch
from torch import Tensor, nn
from torch.nn import functional


@torch.no_grad()
def classification_diagnostics(
    model: nn.Module,
    inputs: Tensor,
    labels: Tensor,
    *,
    bins: int = 15,
    batch_size: int | None = None,
) -> dict[str, float | str]:
    was_training = model.training
    model.eval()
    try:
        logits = _batched_logits(model, inputs, batch_size=batch_size)
        return _diagnostics_from_logits(logits, labels.detach().cpu(), bins=bins)
    finally:
        model.train(was_training)


def corruption_suite(inputs: Tensor, seed: int) -> tuple[tuple[str, Tensor], ...]:
    generator = torch.Generator(device=inputs.device).manual_seed(seed + 41_003)
    noise = torch.randn(inputs.shape, generator=generator, device=inputs.device, dtype=inputs.dtype)
    missing = torch.rand(inputs.shape[:2], generator=generator, device=inputs.device)
    downsampled = functional.interpolate(
        inputs.transpose(1, 2),
        size=max(1, inputs.shape[1] // 2),
        mode="linear",
        align_corners=False,
    )
    restored = functional.interpolate(
        downsampled,
        size=inputs.shape[1],
        mode="linear",
        align_corners=False,
    ).transpose(1, 2)
    return (
        ("id", inputs),
        ("noise_std_0.1", inputs + 0.1 * noise),
        ("noise_std_0.2", inputs + 0.2 * noise),
        ("missing_rate_0.1", inputs.masked_fill((missing < 0.1).unsqueeze(-1), 0.0)),
        ("missing_rate_0.3", inputs.masked_fill((missing < 0.3).unsqueeze(-1), 0.0)),
        ("amplitude_0.5", 0.5 * inputs),
        ("amplitude_1.5", 1.5 * inputs),
        ("resample_half_restore", restored),
    )


@torch.no_grad()
def corruption_diagnostics(
    model: nn.Module,
    inputs: Tensor,
    labels: Tensor,
    seed: int,
    *,
    batch_size: int | None = None,
) -> str:
    rows: list[dict[str, float | str]] = []
    was_training = model.training
    model.eval()
    try:
        labels_cpu = labels.detach().cpu()
        for shift, shifted in corruption_suite(inputs, seed):
            logits = _batched_logits(model, shifted, batch_size=batch_size)
            diagnostics = _diagnostics_from_logits(logits, labels_cpu, bins=15)
            rows.append(
                {
                    "shift": shift,
                    "accuracy": float((logits.argmax(dim=-1) == labels_cpu).float().mean().item()),
                    "nll": float(diagnostics["nll"]),
                    "brier_score": float(diagnostics["brier_score"]),
                    "ece_15": float(diagnostics["ece_15"]),
                }
            )
    finally:
        model.train(was_training)
    id_accuracy = float(rows[0]["accuracy"])
    for row in rows:
        row["absolute_accuracy_drop"] = id_accuracy - float(row["accuracy"])
    return json.dumps(rows, separators=(",", ":"))


def _batched_logits(model: nn.Module, inputs: Tensor, *, batch_size: int | None) -> Tensor:
    chunk_size = inputs.shape[0] if batch_size is None else batch_size
    if chunk_size < 1:
        message = "batch_size must be positive"
        raise ValueError(message)
    return torch.cat(
        [model(batch).detach().cpu() for batch in inputs.split(chunk_size)],
        dim=0,
    )


def _diagnostics_from_logits(
    logits: Tensor,
    labels: Tensor,
    *,
    bins: int,
) -> dict[str, float | str]:
    probabilities = logits.softmax(dim=-1)
    predictions = probabilities.argmax(dim=-1)
    class_count = probabilities.shape[-1]
    one_hot = functional.one_hot(labels, num_classes=class_count).to(probabilities.dtype)
    confusion = torch.zeros(class_count, class_count, dtype=torch.long)
    confusion.index_put_((labels, predictions), torch.ones_like(labels), accumulate=True)
    return {
        "nll": float(functional.cross_entropy(logits, labels).item()),
        "brier_score": float((probabilities - one_hot).square().sum(dim=-1).mean().item()),
        "ece_15": _ece(probabilities, labels, bins),
        "mean_confidence": float(probabilities.max(dim=-1).values.mean().item()),
        "confusion_matrix_json": json.dumps(confusion.tolist(), separators=(",", ":")),
        "per_class_metrics_json": json.dumps(_per_class(confusion), separators=(",", ":")),
    }


def _ece(probabilities: Tensor, labels: Tensor, bins: int) -> float:
    confidence, predictions = probabilities.max(dim=-1)
    correct = predictions.eq(labels).to(probabilities.dtype)
    edges = torch.linspace(0.0, 1.0, bins + 1, device=probabilities.device)
    error = probabilities.new_zeros(())
    for index in range(bins):
        mask = confidence.ge(edges[index]) & (
            confidence.le(edges[index + 1])
            if index == bins - 1
            else confidence.lt(edges[index + 1])
        )
        if bool(mask.any()):
            error += mask.float().mean() * (confidence[mask].mean() - correct[mask].mean()).abs()
    return float(error.item())


def _per_class(confusion: Tensor) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for class_index in range(confusion.shape[0]):
        true_positive = int(confusion[class_index, class_index].item())
        support = int(confusion[class_index].sum().item())
        predicted = int(confusion[:, class_index].sum().item())
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "class": class_index,
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return rows
