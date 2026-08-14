from __future__ import annotations

from math import log
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn
from torch.nn import functional

from .advanced_experiments import RegressionOutcome
from .selective_tapped_prl import SelectiveTappedPRLBlock
from .selective_tapped_prl_metrics import pole_proxy

if TYPE_CHECKING:
    from .advanced_experiments import SequenceRegressionTask
    from .experiment import SyntheticLaplaceTask, TrainingConfig
    from .selective_tapped_prl_types import SelectiveExperimentConfig


def train_selective_regression_model(
    model: nn.Module,
    task: SequenceRegressionTask | SyntheticLaplaceTask,
    training: TrainingConfig,
    config: SelectiveExperimentConfig,
) -> RegressionOutcome:
    torch.manual_seed(config.seed)
    model.to(device=training.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )
    train_inputs = task.train_inputs.to(device=training.device)
    train_targets = task.train_targets.to(device=training.device)
    initial_loss = evaluate_regression_loss(model, train_inputs, train_targets)
    model.train()
    for _ in range(training.epochs):
        optimizer.zero_grad(set_to_none=True)
        predictions = model(train_inputs)
        loss = functional.mse_loss(predictions, train_targets)
        if isinstance(model, SelectiveTappedPRLBlock):
            loss = loss + selective_regularization(model, train_inputs, config)
        loss.backward()
        optimizer.step()
    validation_loss = evaluate_regression_loss(
        model,
        task.validation_inputs.to(device=training.device),
        task.validation_targets.to(device=training.device),
    )
    final_loss = evaluate_regression_loss(model, train_inputs, train_targets)
    return RegressionOutcome(initial_loss, final_loss, validation_loss, pole_proxy(model))


def evaluate_regression_loss(model: nn.Module, inputs: Tensor, targets: Tensor) -> float:
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            return float(functional.mse_loss(model(inputs), targets).item())
    finally:
        model.train(was_training)


def selective_regularization(
    model: SelectiveTappedPRLBlock,
    inputs: Tensor,
    config: SelectiveExperimentConfig,
) -> Tensor:
    projected = model.input_projection(inputs)
    loss = inputs.new_zeros(())
    if config.tap_entropy_weight > 0.0:
        taps = model.tap_selection_values(projected).clamp_min(1.0e-12)
        entropy = -(taps * torch.log(taps)).sum(dim=-1).mean() / log(model.tap_kernel_size)
        loss = loss + config.tap_entropy_weight * entropy
    if config.gate_entropy_weight > 0.0:
        gate = model.input_gate_values(projected).clamp(1.0e-6, 1.0 - 1.0e-6)
        entropy = -(gate * torch.log(gate) + (1.0 - gate) * torch.log(1.0 - gate)).mean()
        loss = loss + config.gate_entropy_weight * entropy
    return loss
