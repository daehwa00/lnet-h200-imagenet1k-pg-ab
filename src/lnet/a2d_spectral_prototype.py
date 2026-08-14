"""Theory-aligned prototype heads for frozen A2D modal descriptors."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional


def class_energy_prototypes(features: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
    """Estimate ``log1p(E[energy | class])`` from log-energy descriptors."""
    if features.ndim != 2 or labels.ndim != 1 or features.shape[0] != labels.numel():
        message = "prototype inputs must be [samples, coordinates] and [samples]"
        raise ValueError(message)
    if not torch.isfinite(features).all() or not torch.isfinite(labels).all():
        message = "prototype inputs must be finite"
        raise ValueError(message)
    if bool((features < -1.0e-6).any()):
        message = "A2D log-energy descriptors cannot be negative"
        raise ValueError(message)
    classes = torch.unique(labels.to(torch.long), sorted=True)
    if classes.numel() < 2:
        message = "prototype fitting requires at least two classes"
        raise ValueError(message)
    raw_energy = torch.expm1(features.float()).clamp_min_(0.0)
    prototypes = torch.stack(
        [
            torch.log1p(raw_energy[labels == class_value].mean(dim=0))
            for class_value in classes
        ]
    )
    return classes, prototypes


def prototype_logits(
    features: Tensor,
    prototypes: Tensor,
    *,
    diagonal: Tensor | None = None,
    low_rank: Tensor | None = None,
    logit_scale: Tensor | float = 1.0,
) -> Tensor:
    """Return affine nearest-prototype logits under ``diag(d) + U U^T``."""
    if features.ndim != 2 or prototypes.ndim != 2:
        message = "prototype logits require two matrices"
        raise ValueError(message)
    if features.shape[1] != prototypes.shape[1]:
        message = "feature and prototype dimensions differ"
        raise ValueError(message)
    if diagonal is None:
        diagonal = torch.ones(
            features.shape[1],
            device=features.device,
            dtype=features.dtype,
        )
    if diagonal.shape != (features.shape[1],) or bool((diagonal <= 0).any()):
        message = "prototype diagonal metric must be positive and coordinate-wise"
        raise ValueError(message)
    weighted_prototypes = prototypes * diagonal
    logits = (
        2.0 * features @ weighted_prototypes.transpose(0, 1)
        - (prototypes * weighted_prototypes).sum(dim=1)
    )
    if low_rank is not None:
        if low_rank.ndim != 2 or low_rank.shape[0] != features.shape[1]:
            message = "prototype low-rank factor has an incompatible shape"
            raise ValueError(message)
        projected_features = features @ low_rank
        projected_prototypes = prototypes @ low_rank
        logits = logits + (
            2.0 * projected_features @ projected_prototypes.transpose(0, 1)
            - projected_prototypes.square().sum(dim=1)
        )
    return logits * torch.as_tensor(
        logit_scale,
        device=features.device,
        dtype=features.dtype,
    )


def pooled_within_class_variance(
    features: Tensor,
    labels: Tensor,
    classes: Tensor,
    prototypes: Tensor,
) -> Tensor:
    """Compute the shared coordinate-wise within-class variance."""
    if classes.shape[0] != prototypes.shape[0]:
        message = "classes and prototypes disagree"
        raise ValueError(message)
    residual_sum = torch.zeros(
        features.shape[1],
        device=features.device,
        dtype=features.dtype,
    )
    count = 0
    for index, class_value in enumerate(classes):
        active = features[labels == class_value]
        residual_sum.add_((active - prototypes[index]).square().sum(dim=0))
        count += active.shape[0]
    return residual_sum / max(1, count - classes.numel())


def diagonal_precision(
    within_variance: Tensor,
    *,
    shrinkage: float,
    minimum_ratio: float = 0.05,
    maximum_ratio: float = 20.0,
) -> Tensor:
    """Return a mean-one inverse-variance metric with scalar shrinkage."""
    if not 0.0 <= shrinkage <= 1.0:
        message = "diagonal shrinkage must lie in [0, 1]"
        raise ValueError(message)
    target = within_variance.mean()
    shrunk = (1.0 - shrinkage) * within_variance + shrinkage * target
    precision = shrunk.clamp_min(torch.finfo(shrunk.dtype).eps).reciprocal()
    precision = precision / precision.mean()
    return precision.clamp(minimum_ratio, maximum_ratio)


class LearnedPSDPrototypeMetric(nn.Module):
    """Learn a positive diagonal-plus-low-rank metric around fixed prototypes."""

    def __init__(
        self,
        prototypes: Tensor,
        initial_diagonal: Tensor,
        rank: int,
    ) -> None:
        super().__init__()
        if rank <= 0:
            message = "prototype metric rank must be positive"
            raise ValueError(message)
        if prototypes.ndim != 2 or initial_diagonal.shape != (prototypes.shape[1],):
            message = "prototype metric initialization has incompatible dimensions"
            raise ValueError(message)
        self.register_buffer("prototypes", prototypes.detach().clone())
        self.log_diagonal = nn.Parameter(initial_diagonal.log())
        self.low_rank = nn.Parameter(torch.empty(prototypes.shape[1], rank))
        nn.init.normal_(self.low_rank, std=1.0e-3)
        self.logit_scale = nn.Parameter(torch.tensor(0.0))

    def diagonal(self) -> Tensor:
        active = self.log_diagonal - torch.logsumexp(self.log_diagonal, dim=0)
        return torch.exp(active) * self.log_diagonal.numel()

    def forward(self, features: Tensor) -> Tensor:
        return prototype_logits(
            features,
            self.prototypes,
            diagonal=self.diagonal(),
            low_rank=self.low_rank,
            logit_scale=self.logit_scale.clamp(-4.0, 4.0).exp(),
        )


def stratified_fit_calibration_split(
    labels: Tensor,
    *,
    calibration_fraction: float,
    seed: int,
) -> tuple[Tensor, Tensor]:
    """Split every class deterministically without touching validation data."""
    if not 0.0 < calibration_fraction < 1.0:
        message = "calibration fraction must lie strictly between zero and one"
        raise ValueError(message)
    labels_cpu = labels.detach().cpu().to(torch.long)
    generator = torch.Generator().manual_seed(seed)
    fit: list[Tensor] = []
    calibration: list[Tensor] = []
    for class_value in torch.unique(labels_cpu, sorted=True):
        indices = torch.nonzero(labels_cpu == class_value, as_tuple=False).flatten()
        shuffled = indices[torch.randperm(indices.numel(), generator=generator)]
        count = max(1, min(indices.numel() - 1, math.ceil(indices.numel() * calibration_fraction)))
        calibration.append(shuffled[:count])
        fit.append(shuffled[count:])
    return torch.cat(fit).sort().values, torch.cat(calibration).sort().values


def two_prototypes_per_class(
    features: Tensor,
    raw_log_features: Tensor,
    labels: Tensor,
    *,
    iterations: int = 12,
) -> tuple[Tensor, Tensor]:
    """Fit deterministic two-means clusters, then estimate energy prototypes."""
    if iterations <= 0:
        message = "two-prototype fitting needs at least one iteration"
        raise ValueError(message)
    classes = torch.unique(labels, sorted=True)
    output: list[Tensor] = []
    output_classes: list[Tensor] = []
    for class_value in classes:
        mask = labels == class_value
        active = features[mask]
        active_raw_log = raw_log_features[mask]
        center = active.mean(dim=0, keepdim=True)
        first_index = (active - center).square().sum(dim=1).argmax()
        first = active[first_index]
        second_index = (active - first).square().sum(dim=1).argmax()
        centers = torch.stack((first, active[second_index]))
        assignment = torch.zeros(active.shape[0], device=active.device, dtype=torch.long)
        for _ in range(iterations):
            assignment = torch.cdist(active, centers).argmin(dim=1)
            if bool((torch.bincount(assignment, minlength=2) == 0).any()):
                assignment = torch.arange(active.shape[0], device=active.device) % 2
            centers = torch.stack([active[assignment == index].mean(dim=0) for index in range(2)])
        raw_energy = torch.expm1(active_raw_log).clamp_min_(0.0)
        output.extend(
            torch.log1p(raw_energy[assignment == index].mean(dim=0))
            for index in range(2)
        )
        output_classes.extend((class_value, class_value))
    return torch.stack(output), torch.stack(output_classes)


def grouped_max_logits(logits: Tensor, prototype_classes: Tensor, classes: Tensor) -> Tensor:
    """Reduce multiple affine prototype logits to one class logit by maximum."""
    return torch.stack(
        [logits[:, prototype_classes == class_value].amax(dim=1) for class_value in classes],
        dim=1,
    )


def classification_metrics(logits: Tensor, labels: Tensor) -> dict[str, float]:
    """Return accuracy, balanced accuracy, and cross entropy."""
    labels = labels.to(device=logits.device, dtype=torch.long)
    predictions = logits.argmax(dim=1)
    classes = torch.unique(labels, sorted=True)
    recalls = [
        (predictions[labels == class_value] == class_value).float().mean()
        for class_value in classes
    ]
    return {
        "accuracy": float((predictions == labels).float().mean()),
        "balanced_accuracy": float(torch.stack(recalls).mean()),
        "cross_entropy": float(functional.cross_entropy(logits, labels)),
    }


__all__ = [
    "LearnedPSDPrototypeMetric",
    "class_energy_prototypes",
    "classification_metrics",
    "diagonal_precision",
    "grouped_max_logits",
    "pooled_within_class_variance",
    "prototype_logits",
    "stratified_fit_calibration_split",
    "two_prototypes_per_class",
]
