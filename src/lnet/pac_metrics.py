from __future__ import annotations

from math import isfinite, sqrt
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from .pac_model import PACHybridPRLBlock

if TYPE_CHECKING:
    from .pac_types import PACRegressionTask
    from .tapped_prl_followup_schema import JsonRow


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def nrmse(mse: float, targets: Tensor) -> float:
    variance = float(targets.var(unbiased=False).clamp_min(1.0e-12).item())
    return sqrt(mse / variance)


def mechanism_row(model: nn.Module, task: PACRegressionTask, device: str) -> JsonRow:
    if not isinstance(model, PACHybridPRLBlock):
        return _empty_mechanism(task.label)
    inputs = task.validation_inputs.to(device=device)
    with torch.no_grad():
        projected = model.input_projection(inputs)
        prl_branch = model.require_prl_branch()
        damping = prl_branch.effective_damping_values(projected).detach().cpu()
        decay = prl_branch.effective_discrete_decay(projected).detach().cpu()
        taps = prl_branch.effective_tap_weights().detach().cpu()
        frequencies = prl_branch.frequency_values().detach().cpu()
    score = damping.mean(dim=-1)
    return {
        "task": task.label,
        "min_alpha_eff": _finite(damping.min()),
        "max_abs_discrete_decay": _finite(torch.abs(decay).max()),
        "tap_mass_near_true_delay": _tap_mass(taps, task.true_delay),
        "frequency_trend_correlation": _frequency_score(frequencies, task.true_frequency),
        "damping_corr": _damping_corr(score, task.validation_teacher_damping),
        "damping_r2": _damping_r2(score, task.validation_teacher_damping),
        "regime_auc": _regime_auc(score, task.validation_regime),
        "impulse_response_nmse_by_regime": _impulse_nmse(score, frequencies, task),
    }


def _empty_mechanism(task: str) -> JsonRow:
    return {
        "task": task,
        "min_alpha_eff": None,
        "max_abs_discrete_decay": None,
        "tap_mass_near_true_delay": None,
        "frequency_trend_correlation": None,
        "damping_corr": None,
        "damping_r2": None,
        "regime_auc": None,
        "impulse_response_nmse_by_regime": None,
    }


def _tap_mass(taps: Tensor, delay: int | None) -> float | None:
    if delay is None:
        return None
    left = max(delay - 1, 0)
    right = min(delay + 2, taps.shape[-1])
    return float(taps[:, left:right].sum(dim=-1).mean().item())


def _damping_corr(score: Tensor, teacher: Tensor | None) -> float | None:
    if teacher is None:
        return None
    values = score.flatten().to(dtype=torch.float64)
    labels = teacher.flatten().to(dtype=torch.float64)
    denominator = values.std(unbiased=False) * labels.std(unbiased=False)
    if float(denominator.item()) == 0.0:
        return None
    return float(
        ((values - values.mean()) * (labels - labels.mean())).mean().div(denominator).item()
    )


def _damping_r2(score: Tensor, teacher: Tensor | None) -> float | None:
    correlation = _damping_corr(score, teacher)
    return None if correlation is None else correlation * correlation


def _regime_auc(score: Tensor, regime: Tensor | None) -> float | None:
    if regime is None:
        return None
    labels = regime.flatten()
    values = score.flatten()
    positives = int(labels.sum().item())
    negatives = int((~labels).sum().item())
    if positives == 0 or negatives == 0:
        return None
    order = torch.argsort(values)
    ranks = torch.arange(1, labels.numel() + 1, dtype=torch.float64)
    positive_rank_sum = ranks[labels[order]].sum()
    return float(
        ((positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)).item()
    )


def _frequency_score(frequencies: Tensor, true_frequency: float | None) -> float | None:
    if true_frequency is None:
        return None
    closest_error = torch.min(torch.abs(frequencies - true_frequency))
    scale = max(abs(true_frequency), 1.0)
    return max(-1.0, 1.0 - float(closest_error.item()) / scale)


def _impulse_nmse(score: Tensor, frequencies: Tensor, task: PACRegressionTask) -> float | None:
    teacher = task.validation_teacher_damping
    regime = task.validation_regime
    if teacher is None or regime is None or task.true_frequency is None:
        return None
    true_frequency = task.true_frequency
    learned_frequency = _closest_frequency(frequencies, true_frequency)
    losses = [
        _regime_impulse_nmse(
            score,
            teacher,
            regime,
            regime_value=regime_value,
            learned_frequency=learned_frequency,
            true_frequency=true_frequency,
            task=task,
        )
        for regime_value in (False, True)
    ]
    valid_losses = [loss for loss in losses if loss is not None]
    return None if not valid_losses else float(sum(valid_losses) / len(valid_losses))


def _closest_frequency(frequencies: Tensor, true_frequency: float) -> float:
    index = torch.argmin(torch.abs(frequencies - true_frequency))
    return float(frequencies[index].item())


def _regime_impulse_nmse(
    score: Tensor,
    teacher: Tensor,
    regime: Tensor,
    *,
    regime_value: bool,
    learned_frequency: float,
    true_frequency: float,
    task: PACRegressionTask,
) -> float | None:
    mask = regime == regime_value
    if not bool(mask.any()):
        return None
    learned_alpha = float(score[mask].mean().item())
    teacher_alpha = float(teacher[mask].mean().item())
    horizon = min(32, task.validation_inputs.shape[1])
    time = torch.arange(horizon, dtype=torch.float64)
    learned = torch.exp(-learned_alpha * time) * torch.cos(learned_frequency * time)
    target = torch.exp(-teacher_alpha * time) * torch.cos(true_frequency * time)
    mse = torch.mean((learned - target).square())
    normalizer = torch.mean(target.square()).clamp_min(1.0e-12)
    return float((mse / normalizer).item())


def _finite(value: Tensor) -> float | None:
    scalar = float(value.item())
    return scalar if isfinite(scalar) else None
