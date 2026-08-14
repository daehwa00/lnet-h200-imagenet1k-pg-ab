from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from .selective_tapped_prl import SelectiveTappedPRLBlock

if TYPE_CHECKING:
    from .controlled_damping_types import ControlledDampingTask
    from .tapped_prl_followup_schema import JsonRow


@dataclass(frozen=True, slots=True)
class DampingDiagnostics:
    mean_effective_damping: float | None
    std_effective_damping: float | None
    min_effective_damping: float | None
    max_effective_damping: float | None
    max_abs_discrete_decay: float | None
    damping_saturation_fraction: float | None
    fast_damping_mean: float | None
    slow_damping_mean: float | None
    damping_regime_correlation: float | None
    damping_regime_auc: float | None
    frequency_input_invariant: bool | None


def diagnostic_row(model: nn.Module, task: ControlledDampingTask, device: str) -> JsonRow:
    diagnostics = damping_diagnostics(model, task, device)
    return {
        "mean_effective_damping": diagnostics.mean_effective_damping,
        "std_effective_damping": diagnostics.std_effective_damping,
        "min_effective_damping": diagnostics.min_effective_damping,
        "max_effective_damping": diagnostics.max_effective_damping,
        "max_abs_discrete_decay": diagnostics.max_abs_discrete_decay,
        "damping_saturation_fraction": diagnostics.damping_saturation_fraction,
        "fast_damping_mean": diagnostics.fast_damping_mean,
        "slow_damping_mean": diagnostics.slow_damping_mean,
        "damping_regime_correlation": diagnostics.damping_regime_correlation,
        "damping_regime_auc": diagnostics.damping_regime_auc,
        "frequency_input_invariant": diagnostics.frequency_input_invariant,
    }


def damping_diagnostics(
    model: nn.Module,
    task: ControlledDampingTask,
    device: str,
) -> DampingDiagnostics:
    if not isinstance(model, SelectiveTappedPRLBlock):
        return DampingDiagnostics(None, None, None, None, None, None, None, None, None, None, None)
    inputs = task.task.validation_inputs.to(device=device)
    with torch.no_grad():
        projected = model.input_projection(inputs)
        damping = model.effective_damping_values(projected).detach().cpu()
        decay = model.effective_discrete_decay(projected).detach().cpu()
        control = model.damping_control_values(projected).detach().cpu()
    score = damping.mean(dim=-1)
    regime = task.validation_fast_regime
    return DampingDiagnostics(
        mean_effective_damping=float(damping.mean().item()),
        std_effective_damping=float(damping.std(unbiased=False).item()),
        min_effective_damping=float(damping.min().item()),
        max_effective_damping=float(damping.max().item()),
        max_abs_discrete_decay=float(torch.abs(decay).max().item()),
        damping_saturation_fraction=float((control.abs() > 0.95).to(torch.float32).mean().item()),
        fast_damping_mean=_regime_mean(score, regime, fast=True),
        slow_damping_mean=_regime_mean(score, regime, fast=False),
        damping_regime_correlation=_binary_correlation(score, regime),
        damping_regime_auc=_binary_auc(score, regime),
        frequency_input_invariant=_frequency_input_invariant(decay),
    )


def _regime_mean(score: Tensor, regime: Tensor | None, *, fast: bool) -> float | None:
    if regime is None:
        return None
    mask = regime if fast else ~regime
    if not bool(mask.any()):
        return None
    return float(score[mask].mean().item())


def _binary_correlation(score: Tensor, regime: Tensor | None) -> float | None:
    if regime is None:
        return None
    labels = regime.to(dtype=torch.float64)
    values = score.to(dtype=torch.float64)
    if labels.numel() < 2 or float(labels.std(unbiased=False).item()) == 0.0:
        return None
    centered_values = values - values.mean()
    centered_labels = labels - labels.mean()
    denominator = centered_values.std(unbiased=False) * centered_labels.std(unbiased=False)
    return float((centered_values * centered_labels).mean().div(denominator).item())


def _binary_auc(score: Tensor, regime: Tensor | None) -> float | None:
    if regime is None:
        return None
    values = score.flatten()
    labels = regime.flatten()
    positive_count = int(labels.sum().item())
    negative_count = int((~labels).sum().item())
    if positive_count == 0 or negative_count == 0:
        return None
    order = torch.argsort(values)
    sorted_labels = labels[order]
    ranks = torch.arange(1, labels.numel() + 1, dtype=torch.float64)
    positive_rank_sum = ranks[sorted_labels].sum()
    numerator = positive_rank_sum - (positive_count * (positive_count + 1) / 2.0)
    return float((numerator / (positive_count * negative_count)).item())


def _frequency_input_invariant(decay: Tensor) -> bool:
    phase = torch.angle(decay)
    reference = phase[:1, :1, :]
    return bool(torch.allclose(phase, reference.expand_as(phase), atol=1.0e-6))
