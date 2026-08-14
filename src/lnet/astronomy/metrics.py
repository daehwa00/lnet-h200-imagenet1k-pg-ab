"""Streaming classification metrics shared by neural and feature baselines."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from lnet.astronomy.plasticc import PLASTICC_KNOWN_CLASS_WEIGHTS


@dataclass(slots=True)
class MetricAccumulator:
    classes: int
    confusion: Tensor
    class_loss_sum: Tensor
    class_count: Tensor
    calibration_count: Tensor
    calibration_confidence: Tensor
    calibration_correct: Tensor

    @classmethod
    def create(cls, classes: int) -> MetricAccumulator:
        return cls(
            classes=classes,
            confusion=torch.zeros(classes, classes, dtype=torch.long),
            class_loss_sum=torch.zeros(classes, dtype=torch.float64),
            class_count=torch.zeros(classes, dtype=torch.long),
            calibration_count=torch.zeros(15, dtype=torch.long),
            calibration_confidence=torch.zeros(15, dtype=torch.float64),
            calibration_correct=torch.zeros(15, dtype=torch.float64),
        )

    def update(self, probability: Tensor, target: Tensor) -> None:
        probability = probability.detach().cpu().clamp_min(
            torch.finfo(torch.float32).tiny
        )
        target = target.detach().cpu()
        prediction = probability.argmax(dim=-1)
        selected = probability[torch.arange(target.numel()), target]
        loss = -selected.log()
        confidence = probability.max(dim=-1).values
        bins = torch.clamp((confidence * 15).long(), max=14)
        confusion = torch.bincount(
            target * self.classes + prediction,
            minlength=self.classes * self.classes,
        ).reshape(self.classes, self.classes)
        self.confusion += confusion
        self.class_loss_sum.index_add_(0, target, loss.to(torch.float64))
        self.class_count += torch.bincount(target, minlength=self.classes)
        self.calibration_count += torch.bincount(bins, minlength=15)
        self.calibration_confidence.index_add_(0, bins, confidence.to(torch.float64))
        self.calibration_correct.index_add_(
            0,
            bins,
            target.eq(prediction).to(torch.float64),
        )

    def finalize(self) -> dict[str, float | int]:
        recall = self.confusion.diag() / self.confusion.sum(dim=1).clamp_min(1)
        precision = self.confusion.diag() / self.confusion.sum(dim=0).clamp_min(1)
        f1 = 2.0 * precision * recall / (precision + recall).clamp_min(
            torch.finfo(torch.float32).eps
        )
        weights = torch.tensor(PLASTICC_KNOWN_CLASS_WEIGHTS, dtype=torch.float64)
        present = self.class_count > 0
        weighted_log_loss = (
            weights[present]
            * self.class_loss_sum[present]
            / self.class_count[present]
        ).sum() / weights[present].sum()
        total = int(self.class_count.sum())
        ece = 0.0
        for index in range(15):
            count = int(self.calibration_count[index])
            if count:
                mean_confidence = float(self.calibration_confidence[index] / count)
                mean_accuracy = float(self.calibration_correct[index] / count)
                ece += count / total * abs(mean_accuracy - mean_confidence)
        return {
            "objects": total,
            "weighted_log_loss_known14": float(weighted_log_loss),
            "balanced_accuracy": float(recall[present].mean()),
            "macro_f1": float(f1[present].mean()),
            "ece": ece,
        }


def classification_metrics(
    probability: Tensor,
    target: Tensor,
) -> dict[str, float | int]:
    """Compute the shared known-class metric suite for one in-memory prediction."""
    accumulator = MetricAccumulator.create(probability.shape[-1])
    accumulator.update(probability, target)
    return accumulator.finalize()
