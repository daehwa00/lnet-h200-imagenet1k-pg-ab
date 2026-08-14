from __future__ import annotations

from math import log
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from .pac_head_factorial_features import BlockContext, block_context
from .pac_metrics import count_parameters
from .pac_model import PACHybridPRLBlock
from .pac_training import evaluate_regression_loss

if TYPE_CHECKING:
    from .pac_types import PACRegressionTask
    from .tapped_prl_followup_schema import JsonRow


def pac_blocks(model: nn.Module) -> tuple[PACHybridPRLBlock, ...]:
    return tuple(block for module in model.modules() if (block := _as_block(module)) is not None)


def pole_rows(model: nn.Module, task: PACRegressionTask, seed: int) -> tuple[JsonRow, ...]:
    true_alpha = _true_alpha(task.label)
    rows: list[JsonRow] = []
    for block_index, block in enumerate(pac_blocks(model)):
        prl = block.require_prl_branch()
        alpha = prl.base_damping_values().detach().cpu()
        omega = prl.frequency_values().detach().cpu()
        best = _best_frequency_index(omega, task.true_frequency)
        rows.extend(
            (
                {
                    "task": task.label,
                    "seed": seed,
                    "model": model.__class__.__name__,
                    "block_index": block_index,
                    "mode": mode,
                    "base_alpha": float(alpha[mode].item()),
                    "base_omega": float(omega[mode].item()),
                    "true_alpha": true_alpha,
                    "true_frequency": task.true_frequency,
                    "frequency_abs_error": _abs_error(
                        float(omega[mode].item()), task.true_frequency
                    ),
                    "damping_abs_error": _abs_error(float(alpha[mode].item()), true_alpha),
                    "is_best_frequency_mode": mode == best,
                    "params_trainable": count_parameters(model),
                }
            )
            for mode in range(prl.modes)
        )
    return tuple(rows)


def tap_rows(model: nn.Module, task: PACRegressionTask, seed: int) -> tuple[JsonRow, ...]:
    rows: list[JsonRow] = []
    for block_index, block in enumerate(pac_blocks(model)):
        taps = block.require_prl_branch().effective_tap_weights().detach().cpu()
        for mode in range(taps.shape[0]):
            weights = taps[mode]
            peak = int(torch.argmax(weights).item())
            center = float((weights * torch.arange(weights.numel())).sum().item())
            rows.append(
                {
                    "task": task.label,
                    "seed": seed,
                    "model": model.__class__.__name__,
                    "block_index": block_index,
                    "mode": mode,
                    "tap_peak_index": peak,
                    "tap_center_of_mass": center,
                    "tap_entropy": _entropy(weights),
                    "true_delay": task.true_delay,
                    "tap_peak_error": None
                    if task.true_delay is None
                    else abs(peak - task.true_delay),
                    "tap_mass_near_true_delay": _tap_mass(weights, task.true_delay),
                }
            )
    return tuple(rows)


def damping_row(
    model: nn.Module, task: PACRegressionTask, seed: int, device: str
) -> JsonRow | None:
    contexts = _model_block_contexts(model, task.validation_inputs.to(device=device))
    if not contexts:
        return None
    context = contexts[-1]
    with torch.no_grad():
        damping = context.damping.detach().cpu()
        decay = context.decay_abs.detach().cpu()
    score = damping.mean(dim=-1)
    return {
        "task": task.label,
        "seed": seed,
        "model": model.__class__.__name__,
        "auc_mean": _auc(score, task.validation_regime),
        "corr_mean": _corr(score, task.validation_teacher_damping),
        "fast_damping_mean": _regime_mean(score, task.validation_regime, value=True),
        "slow_damping_mean": _regime_mean(score, task.validation_regime, value=False),
        "cohens_d": _cohens_d(score, task.validation_regime),
        "min_effective_damping": float(damping.min().item()),
        "max_effective_damping": float(damping.max().item()),
        "max_abs_discrete_decay": float(torch.abs(decay).max().item()),
    }


def mode_knockout_rows(
    model: nn.Module, task: PACRegressionTask, seed: int, full_loss: float, device: str
) -> tuple[JsonRow, ...]:
    rows: list[JsonRow] = []
    contexts = _model_block_contexts(model, task.test_inputs.to(device=device))
    for block_index, context in enumerate(contexts):
        block = context.block
        energy = _modal_energy(context).detach().cpu()
        prl = block.require_prl_branch()
        writer_norm = torch.linalg.vector_norm(
            torch.complex(prl.writer_real, prl.writer_imag).detach().cpu(), dim=-1
        )
        for mode in range(prl.modes):
            loss = _mode_knockout_loss(model, task, block, mode, device)
            rows.append(
                {
                    "task": task.label,
                    "seed": seed,
                    "block_index": block_index,
                    "mode": mode,
                    "full_test_loss": full_loss,
                    "knockout_loss": loss,
                    "absolute_delta": loss - full_loss,
                    "relative_delta": (loss - full_loss) / max(full_loss, 1.0e-12),
                    "writer_norm": float(writer_norm[mode].item()),
                    "modal_energy": float(energy[mode].item()),
                }
            )
    return tuple(rows)


def impulse_row(model: nn.Module, task: PACRegressionTask, seed: int) -> JsonRow | None:
    blocks = pac_blocks(model)
    if not blocks or task.true_frequency is None:
        return None
    prl = blocks[-1].require_prl_branch()
    alpha = prl.base_damping_values().detach().cpu()
    omega = prl.frequency_values().detach().cpu()
    index = _best_frequency_index(omega, task.true_frequency)
    if index is None:
        return None
    learned = _impulse(float(alpha[index].item()), float(omega[index].item()))
    target = _impulse(_true_alpha(task.label) or float(alpha[index].item()), task.true_frequency)
    nmse = torch.mean((learned - target).square()) / target.square().mean().clamp_min(1.0e-12)
    return {"task": task.label, "seed": seed, "impulse_response_nmse": float(nmse.item())}


def _as_block(module: nn.Module) -> PACHybridPRLBlock | None:
    match module:
        case PACHybridPRLBlock():
            return module
        case _:
            return None


def _model_block_contexts(model: nn.Module, inputs: Tensor) -> tuple[BlockContext, ...]:
    direct = _as_block(model)
    if direct is not None:
        return (block_context(direct, inputs),)
    modules = getattr(model, "blocks", ())
    features = inputs
    contexts: list[BlockContext] = []
    for module in modules:
        block = _as_block(module)
        if block is not None:
            context = block_context(block, features)
            contexts.append(context)
            features = context.output
    return tuple(contexts)


def _modal_energy(context: BlockContext) -> Tensor:
    return (context.states_real.square() + context.states_imag.square()).mean(dim=(0, 1))


def _mode_knockout_loss(
    model: nn.Module, task: PACRegressionTask, block: PACHybridPRLBlock, mode: int, device: str
) -> float:
    prl = block.require_prl_branch()
    reader = prl.reader.detach().clone()
    writer_real = prl.writer_real.detach().clone()
    writer_imag = prl.writer_imag.detach().clone()
    try:
        with torch.no_grad():
            prl.reader[mode].zero_()
            prl.writer_real[mode].zero_()
            prl.writer_imag[mode].zero_()
        return evaluate_regression_loss(
            model, task.test_inputs.to(device=device), task.test_targets.to(device=device)
        )
    finally:
        with torch.no_grad():
            prl.reader.copy_(reader)
            prl.writer_real.copy_(writer_real)
            prl.writer_imag.copy_(writer_imag)


def _best_frequency_index(values: Tensor, target: float | None) -> int | None:
    if target is None:
        return None
    return int(torch.argmin(torch.abs(values - target)).item())


def _true_alpha(task: str) -> float | None:
    return {
        "modal_teacher": 0.2,
        "delayed_exponential": 0.15,
        "delayed_oscillatory": 0.15,
        "multi_mode_delayed_resonance": 0.2,
        "active_damping_teacher": 0.8,
        "context_damped_exponential": -log(0.35),
        "delayed_context_damped_exponential": -log(0.35),
    }.get(task)


def _abs_error(value: float, target: float | None) -> float | None:
    return None if target is None else abs(value - target)


def _tap_mass(weights: Tensor, delay: int | None) -> float | None:
    if delay is None:
        return None
    left = max(delay - 1, 0)
    right = min(delay + 2, weights.numel())
    return float(weights[left:right].sum().item())


def _entropy(weights: Tensor) -> float:
    return float(-(weights * weights.clamp_min(1.0e-12).log()).sum().item())


def _corr(score: Tensor, teacher: Tensor | None) -> float | None:
    if teacher is None:
        return None
    values = score.flatten().to(dtype=torch.float64)
    labels = teacher.flatten().to(dtype=torch.float64)
    denominator = values.std(unbiased=False) * labels.std(unbiased=False)
    if float(denominator.item()) == 0.0:
        return None
    return float(((values - values.mean()) * (labels - labels.mean())).mean().div(denominator))


def _auc(score: Tensor, regime: Tensor | None) -> float | None:
    if regime is None:
        return None
    labels = regime.flatten()
    positives = int(labels.sum().item())
    negatives = int((~labels).sum().item())
    if positives == 0 or negatives == 0:
        return None
    values = score.flatten()
    order = torch.argsort(values)
    ranks = torch.arange(1, labels.numel() + 1, dtype=torch.float64)
    positive_rank_sum = ranks[labels[order]].sum()
    return float((positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))


def _regime_mean(score: Tensor, regime: Tensor | None, *, value: bool) -> float | None:
    if regime is None:
        return None
    mask = regime == value
    return float(score[mask].mean().item()) if bool(mask.any()) else None


def _cohens_d(score: Tensor, regime: Tensor | None) -> float | None:
    if regime is None:
        return None
    fast = score[regime]
    slow = score[~regime]
    pooled = torch.sqrt((fast.var(unbiased=False) + slow.var(unbiased=False)) / 2.0)
    return float(((fast.mean() - slow.mean()) / pooled.clamp_min(1.0e-12)).item())


def _impulse(alpha: float, omega: float) -> Tensor:
    time = torch.arange(32, dtype=torch.float64)
    return torch.exp(-alpha * time) * torch.cos(omega * time)
