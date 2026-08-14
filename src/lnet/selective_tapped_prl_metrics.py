from __future__ import annotations

from math import log
from typing import TYPE_CHECKING

import torch
from torch import nn

from .hybrid_metrics import count_trainable_parameters
from .selective_tapped_prl import SelectiveTappedPRLBlock

if TYPE_CHECKING:
    from .advanced_experiments import RegressionOutcome
    from .selective_tapped_prl_types import LabeledTask
    from .tapped_prl_followup_schema import JsonRow


def model_row(
    task: LabeledTask,
    model_label: str,
    model: nn.Module,
    outcome: RegressionOutcome,
) -> JsonRow:
    row: JsonRow = {
        "task": task.label,
        "model": model_label,
        "params": count_trainable_parameters(model),
        "validation_loss": outcome.validation_loss,
        "initial_loss": outcome.initial_loss,
        "final_loss": outcome.final_loss,
    }
    if isinstance(model, SelectiveTappedPRLBlock):
        row.update(selective_diagnostic_row(task, model))
    return row


def pole_proxy(model: nn.Module) -> float:
    if not isinstance(model, SelectiveTappedPRLBlock):
        return float("nan")
    return float(torch.mean(torch.abs(model.continuous_poles())).detach().cpu().item())


def selective_diagnostic_row(task: LabeledTask, model: SelectiveTappedPRLBlock) -> JsonRow:
    device = next(model.parameters()).device
    inputs = task.task.validation_inputs.to(device=device)
    projected = model.input_projection(inputs)
    with torch.no_grad():
        input_gate = model.input_gate_values(projected)
        read_gate = model.read_gate_values(projected)
        taps = model.tap_selection_values(projected).clamp_min(1.0e-12)
        mean_taps = taps.mean(dim=(0, 1, 2))
        tap_entropy = float(
            (-(taps * torch.log(taps)).sum(dim=-1).mean() / log(model.tap_kernel_size)).item(),
        )
    row: JsonRow = {
        "variant": model.variant,
        "mean_input_gate": float(input_gate.mean().item()),
        "mean_read_gate": float(read_gate.mean().item()),
        "tap_entropy": tap_entropy,
        "dominant_tap": int(torch.argmax(mean_taps).item()),
        "mean_abs_pole": pole_proxy(model),
    }
    true_delay = task.metadata.true_delay if task.metadata is not None else None
    if true_delay is not None:
        lower = max(0, true_delay - 1)
        upper = min(model.tap_kernel_size - 1, true_delay + 1)
        dominant_tap = int(torch.argmax(mean_taps).item())
        row["tap_mass_near_delay"] = float(mean_taps[lower : upper + 1].sum().item())
        row["tap_peak_error"] = abs(dominant_tap - true_delay)
    if task.label == "switching_teacher":
        midpoint = input_gate.shape[1] // 2
        row["first_half_input_gate"] = float(input_gate[:, :midpoint, :].mean().item())
        row["second_half_input_gate"] = float(input_gate[:, midpoint:, :].mean().item())
    return row
