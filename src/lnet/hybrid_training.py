from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import TYPE_CHECKING

import torch
from torch.nn import functional

from lnet.advanced_experiments import RegressionOutcome

if TYPE_CHECKING:
    from torch import Tensor

    from lnet.advanced_experiments import SequenceRegressionTask
    from lnet.experiment import SyntheticLaplaceTask, TrainingConfig
    from lnet.hybrid import HybridModalPRLBlock


@dataclass(frozen=True, slots=True)
class HybridTrainingOptions:
    gate_entropy_weight: float = 0.0
    tap_entropy_weight: float = 0.0


def train_hybrid_regression_model(
    model: HybridModalPRLBlock,
    task: SequenceRegressionTask | SyntheticLaplaceTask,
    config: TrainingConfig,
    options: HybridTrainingOptions,
) -> RegressionOutcome:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    model.to(device=config.device)
    train_inputs = task.train_inputs.to(device=config.device)
    train_targets = task.train_targets.to(device=config.device)
    initial_loss = evaluate_hybrid_regression_loss(model, train_inputs, train_targets)
    model.train()
    for _ in range(config.epochs):
        optimizer.zero_grad(set_to_none=True)
        predictions = model(train_inputs)
        loss = functional.mse_loss(predictions, train_targets) + hybrid_regularization_loss(
            model,
            train_inputs,
            options,
        )
        loss.backward()
        optimizer.step()
    final_loss = evaluate_hybrid_regression_loss(model, train_inputs, train_targets)
    validation_loss = _evaluate_loss(model, task, config.device)
    pole_mae = float(torch.mean(torch.abs(model.temporal_mixer.continuous_poles())).item())
    return RegressionOutcome(
        initial_loss=initial_loss,
        final_loss=final_loss,
        validation_loss=validation_loss,
        pole_mae=pole_mae,
    )


def hybrid_regularization_loss(
    model: HybridModalPRLBlock,
    inputs: Tensor,
    options: HybridTrainingOptions,
) -> Tensor:
    loss = inputs.new_zeros(())
    projected = model.input_projection(inputs)
    if options.gate_entropy_weight > 0.0:
        weights = model.branch_weights(projected).clamp_min(1.0e-12)
        entropy = -(weights * torch.log(weights)).sum(dim=-1).mean() / log(weights.shape[-1])
        loss = loss + (options.gate_entropy_weight * entropy)
    if options.tap_entropy_weight > 0.0:
        taps = model.temporal_mixer.effective_tap_weights().abs().clamp_min(1.0e-12)
        probabilities = taps / taps.sum(dim=-1, keepdim=True)
        entropy = -(probabilities * torch.log(probabilities)).sum(dim=-1).mean() / log(
            probabilities.shape[-1],
        )
        loss = loss + (options.tap_entropy_weight * entropy.to(dtype=loss.dtype))
    return loss


def evaluate_hybrid_regression_loss(
    model: HybridModalPRLBlock,
    inputs: Tensor,
    targets: Tensor,
) -> float:
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            loss = functional.mse_loss(model(inputs), targets)
            return float(loss.item())
    finally:
        model.train(was_training)


def _evaluate_loss(
    model: HybridModalPRLBlock,
    task: SequenceRegressionTask | SyntheticLaplaceTask,
    device: str,
) -> float:
    validation_inputs = task.validation_inputs.to(device=device)
    validation_targets = task.validation_targets.to(device=device)
    return evaluate_hybrid_regression_loss(model, validation_inputs, validation_targets)
