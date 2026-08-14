from __future__ import annotations

import os
from dataclasses import replace
from typing import TYPE_CHECKING

import torch
from torch import nn

os.environ.setdefault("MPLBACKEND", "Agg")

from matplotlib import pyplot as plt

from .pac_builders import build_regression_model
from .pac_metrics import count_parameters
from .pac_model import PACHybridPRLBlock
from .pac_overnight_io import append_csv_row, read_csv
from .pac_tasks import make_ood_task, make_pac_synthetic_tasks
from .pac_training import evaluate_regression_loss, train_regression_model

if TYPE_CHECKING:
    from pathlib import Path

    from .pac_types import PACExperimentConfig, PACRegressionTask
    from .tapped_prl_followup_schema import JsonValue


def run_damping_diagnostics(
    output_root: Path,
    config: PACExperimentConfig,
    device: str,
    seeds: tuple[int, ...],
    betas: tuple[float, ...],
) -> None:
    path = output_root / "results" / "damping_diagnostics.csv"
    completed = _completed_diagnostics(path)
    for seed in seeds:
        for task in _diagnostic_tasks(config, seed):
            for beta in betas:
                key = (task.label, seed, beta)
                if key in completed:
                    continue
                run_config = replace(config, seeds=(seed,))
                model = build_regression_model("pac_full", run_config)
                _set_damping_range(model, beta)
                outcome = train_regression_model(model, task, run_config, device, seed)
                row = _diagnostic_row(model, task, outcome.test_loss, seed, beta, device)
                append_csv_row(path, row)
                completed.add(key)
                _save_damping_figures(output_root, model, task, seed, beta, device)
    _write_report(output_root)


def _completed_diagnostics(path: Path) -> set[tuple[str, int, float]]:
    completed: set[tuple[str, int, float]] = set()
    for row in read_csv(path):
        task = row.get("task")
        seed = row.get("seed")
        beta = row.get("damping_beta")
        if task is None or seed is None or beta is None:
            continue
        try:
            completed.add((task, int(seed), float(beta)))
        except ValueError:
            continue
    return completed


def _diagnostic_tasks(config: PACExperimentConfig, seed: int) -> tuple[PACRegressionTask, ...]:
    active = next(
        task
        for task in make_pac_synthetic_tasks(config, seed)
        if task.label == "active_damping_teacher"
    )
    context = make_ood_task(config, seed + 101, sequence_length=config.sequence_length, delay=0)
    delayed = make_ood_task(config, seed + 202, sequence_length=config.sequence_length, delay=6)
    return (
        active,
        replace(context, label="context_damped_exponential"),
        replace(delayed, label="delayed_context_damped_exponential"),
    )


def _diagnostic_row(
    model: nn.Module,
    task: PACRegressionTask,
    test_loss: float,
    seed: int,
    beta: float,
    device: str,
) -> dict[str, JsonValue]:
    encoder = _pac_block(model)
    if encoder is None:
        return {"task": task.label, "seed": seed, "model": "pac_full", "status": "not_pac"}
    projected = encoder.input_projection(task.validation_inputs.to(device=device))
    with torch.no_grad():
        prl_branch = encoder.require_prl_branch()
        damping = prl_branch.effective_damping_values(projected).detach().cpu()
        decay = prl_branch.effective_discrete_decay(projected).detach().cpu()
        control = prl_branch.damping_control_values(projected).detach().cpu()
    teacher = task.validation_teacher_damping
    regime = task.validation_regime
    mean_score = damping.mean(dim=-1)
    best_corr, best_auc = _best_mode_scores(damping, teacher, regime)
    weighted = _writer_weighted_score(encoder, damping)
    off_loss = _damping_off_loss(encoder, task, device)
    return {
        "task": task.label,
        "seed": seed,
        "model": "pac_full",
        "damping_beta": beta,
        "test_loss": test_loss,
        "inference_damping_off_loss": off_loss,
        "damping_off_delta": off_loss - test_loss,
        "auc_mean": _auc(mean_score, regime),
        "auc_best_mode": best_auc,
        "auc_writer_weighted": _auc(weighted, regime),
        "corr_mean": _corr(mean_score, teacher),
        "corr_best_mode": best_corr,
        "corr_writer_weighted": _corr(weighted, teacher),
        "fast_damping_mean": _regime_mean(mean_score, regime, value=True),
        "slow_damping_mean": _regime_mean(mean_score, regime, value=False),
        "fast_minus_slow": _fast_minus_slow(mean_score, regime),
        "cohens_d": _cohens_d(mean_score, regime),
        "mean_effective_damping": float(damping.mean().item()),
        "std_effective_damping": float(damping.std(unbiased=False).item()),
        "min_effective_damping": float(damping.min().item()),
        "max_effective_damping": float(damping.max().item()),
        "max_abs_discrete_decay": float(torch.abs(decay).max().item()),
        "fraction_abs_decay_ge_1": float((torch.abs(decay) >= 1.0).to(torch.float32).mean().item()),
        "damping_saturation_fraction": float(
            (torch.abs(control) >= beta * 0.95).to(torch.float32).mean().item()
        )
        if beta > 0
        else 0.0,
        "frequency_input_invariant": True,
        "params_trainable": count_parameters(model),
    }


def _pac_block(model: nn.Module) -> PACHybridPRLBlock | None:
    return model if isinstance(model, PACHybridPRLBlock) else None


def _set_damping_range(model: nn.Module, beta: float) -> None:
    encoder = _pac_block(model)
    if encoder is not None:
        encoder.require_prl_branch().damping_control_range = beta


def _damping_off_loss(encoder: PACHybridPRLBlock, task: PACRegressionTask, device: str) -> float:
    prl_branch = encoder.require_prl_branch()
    original = prl_branch.damping_control_range
    try:
        prl_branch.damping_control_range = 0.0
        return evaluate_regression_loss(
            encoder, task.test_inputs.to(device=device), task.test_targets.to(device=device)
        )
    finally:
        prl_branch.damping_control_range = original


def _best_mode_scores(
    damping: torch.Tensor,
    teacher: torch.Tensor | None,
    regime: torch.Tensor | None,
) -> tuple[float | None, float | None]:
    correlations = [_corr(damping[..., index], teacher) for index in range(damping.shape[-1])]
    aucs = [_auc(damping[..., index], regime) for index in range(damping.shape[-1])]
    return _best(correlations), _best(aucs)


def _writer_weighted_score(encoder: PACHybridPRLBlock, damping: torch.Tensor) -> torch.Tensor:
    weights = torch.linalg.vector_norm(
        encoder.require_prl_branch().writer_real.detach().cpu(), dim=-1
    )
    weights = weights / weights.sum().clamp_min(1.0e-12)
    return torch.sum(damping * weights.view(1, 1, -1), dim=-1)


def _corr(score: torch.Tensor, teacher: torch.Tensor | None) -> float | None:
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


def _auc(score: torch.Tensor, regime: torch.Tensor | None) -> float | None:
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


def _regime_mean(score: torch.Tensor, regime: torch.Tensor | None, *, value: bool) -> float | None:
    if regime is None:
        return None
    mask = regime == value
    return float(score[mask].mean().item()) if bool(mask.any()) else None


def _fast_minus_slow(score: torch.Tensor, regime: torch.Tensor | None) -> float | None:
    fast = _regime_mean(score, regime, value=True)
    slow = _regime_mean(score, regime, value=False)
    return None if fast is None or slow is None else fast - slow


def _cohens_d(score: torch.Tensor, regime: torch.Tensor | None) -> float | None:
    if regime is None:
        return None
    fast = score[regime]
    slow = score[~regime]
    pooled = torch.sqrt((fast.var(unbiased=False) + slow.var(unbiased=False)) / 2.0).clamp_min(
        1.0e-12
    )
    return float(((fast.mean() - slow.mean()) / pooled).item())


def _best(values: list[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    return None if not valid else max(valid)


def _save_damping_figures(
    output_root: Path,
    model: nn.Module,
    task: PACRegressionTask,
    seed: int,
    beta: float,
    device: str,
) -> None:
    encoder = _pac_block(model)
    if encoder is None or task.validation_regime is None:
        return
    projected = encoder.input_projection(task.validation_inputs[:1].to(device=device))
    damping = (
        encoder.require_prl_branch()
        .effective_damping_values(projected)
        .detach()
        .cpu()
        .mean(dim=-1)[0]
    )
    regime = task.validation_regime[0].to(dtype=torch.float32)
    fig_path = (
        output_root / "figures" / f"damping_vs_regime_{task.label}_seed_{seed}_beta_{beta:g}.png"
    )
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 3))
    plt.plot(damping.numpy(), label="mean alpha_eff")
    plt.plot(regime.numpy(), label="teacher regime")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()


def _write_report(output_root: Path) -> None:
    path = output_root / "reports" / "overnight_damping_diagnostics.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Damping Diagnostics\n\ndamping_diagnostics_status: mixed\n",
        encoding="utf-8",
    )
