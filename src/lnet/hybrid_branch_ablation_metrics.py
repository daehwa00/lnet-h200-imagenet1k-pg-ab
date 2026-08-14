from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional

from .hybrid import BRANCH_ORDER, renormalized_branch_weights
from .hybrid_metrics import count_trainable_parameters, hybrid_gate_diagnostic
from .hybrid_training import evaluate_hybrid_regression_loss

if TYPE_CHECKING:
    from torch import Tensor

    from .advanced_experiments import SequenceRegressionTask
    from .experiment import SyntheticLaplaceTask
    from .hybrid import BranchName, HybridModalPRLBlock
    from .tapped_prl_followup_schema import JsonRow


def active_parameter_count(
    model: HybridModalPRLBlock,
    branches: tuple[BranchName, ...],
) -> int:
    active_indices = tuple(BRANCH_ORDER.index(branch) for branch in branches)
    modules: list[nn.Module] = [
        model.input_projection,
        model.output_projection,
        model.readout_projection,
    ]
    if "prl" in branches:
        modules.append(model.temporal_mixer)
    if "fir" in branches:
        modules.append(model.fir_branch)
    if "mlp" in branches:
        modules.append(model.mlp_branch)
    if model.branch_normalization == "layernorm":
        modules.extend(model.branch_layer_norms[index] for index in active_indices)
    parameters = tuple(parameter for module in modules for parameter in module.parameters())
    unique = {id(parameter): parameter for parameter in parameters if parameter.requires_grad}
    total = sum(parameter.numel() for parameter in unique.values())
    if model.branch_gate is not None:
        total += _active_gate_parameter_count(model.branch_gate, len(active_indices))
    if model.fixed_branch_logits is not None and model.fixed_branch_logits.requires_grad:
        total += len(active_indices)
    return total


def _active_gate_parameter_count(branch_gate: nn.Linear, active_branch_count: int) -> int:
    total = 0
    if branch_gate.weight.requires_grad:
        total += active_branch_count * branch_gate.in_features
    if branch_gate.bias is not None and branch_gate.bias.requires_grad:
        total += active_branch_count
    return total


def contribution_metrics(model: HybridModalPRLBlock, inputs: Tensor) -> JsonRow:
    diagnostic = hybrid_gate_diagnostic(model, inputs)
    return {
        "mean_prl_weight": diagnostic.mean_prl_weight,
        "mean_fir_weight": diagnostic.mean_fir_weight,
        "mean_mlp_weight": diagnostic.mean_mlp_weight,
        "gate_entropy": diagnostic.mean_gate_entropy,
        "prl_contribution_norm": diagnostic.prl_contribution_norm,
        "fir_contribution_norm": diagnostic.fir_contribution_norm,
        "mlp_contribution_norm": diagnostic.mlp_contribution_norm,
    }


def knockout_loss_rows(
    model: HybridModalPRLBlock,
    task: SequenceRegressionTask | SyntheticLaplaceTask,
    device: str,
) -> tuple[JsonRow, ...]:
    inputs = task.validation_inputs.to(device=device)
    targets = task.validation_targets.to(device=device)
    full_loss = evaluate_hybrid_regression_loss(model, inputs, targets)
    rows: list[JsonRow] = [
        {"knockout": "full", "validation_loss": full_loss, "delta": 0.0, "relative_delta": 0.0},
    ]
    for branch in BRANCH_ORDER:
        loss = _loss_with_branch_removed(model, inputs, targets, device, branch)
        rows.append(
            {
                "knockout": f"{branch}_off",
                "validation_loss": loss,
                "delta": loss - full_loss,
                "relative_delta": (loss - full_loss) / max(full_loss, 1.0e-12),
            },
        )
    return tuple(rows)


def total_parameter_count(model: nn.Module) -> int:
    return count_trainable_parameters(model)


def _loss_with_branch_removed(
    model: HybridModalPRLBlock,
    inputs: Tensor,
    targets: Tensor,
    device: str,
    branch: BranchName,
) -> float:
    was_training = model.training
    model.eval()
    keep = torch.tensor(
        [candidate != branch for candidate in BRANCH_ORDER],
        dtype=torch.bool,
        device=inputs.device,
    )
    try:
        with torch.no_grad():
            projected = model.input_projection(inputs.to(device=device))
            branches = model.normalize_branch_outputs(model.branch_outputs(projected))
            weights = renormalized_branch_weights(model.branch_weights(projected), keep)
            fused = torch.sum(weights.unsqueeze(-1) * torch.stack(branches, dim=2), dim=2)
            residual = projected + model.output_projection(model.activation(fused))
            predictions = model.readout_projection(residual)
            return float(functional.mse_loss(predictions, targets.to(device=device)).item())
    finally:
        model.train(was_training)
