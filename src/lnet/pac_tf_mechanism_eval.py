from __future__ import annotations

from functools import lru_cache
from itertools import permutations
from math import isfinite
from threading import Lock
from time import perf_counter
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor

from .hybrid_experiment_types import resolve_device
from .pac_metrics import count_parameters, nrmse
from .pac_tasks import make_pac_tf_mechanism_tasks
from .pac_tf_mechanism_models import PAC_TF_MODELS, build_mechanism_model
from .pac_training import train_regression_model

if TYPE_CHECKING:
    from .pac_tf_mechanism_types import MechanismJob
    from .pac_tight_frame_models import TightFrameSequenceRegressor
    from .pac_types import PACExperimentConfig, PACRegressionTask
    from .tapped_prl_followup_schema import JsonRow

_MODEL_INIT_LOCK = Lock()


def run_mechanism_job(config: PACExperimentConfig, job: MechanismJob) -> JsonRow:
    device = resolve_device(config.device)
    task = _task_for_job(config, job.seed, job.task)
    with _MODEL_INIT_LOCK, torch.random.fork_rng(devices=[]):
        torch.manual_seed(job.seed)
        model = build_mechanism_model(job.model, config)
    started = perf_counter()
    outcome = train_regression_model(model, task, config, device, job.seed)
    row: JsonRow = {
        "queue_key": job.key,
        "experiment_group": "synthetic_mechanism_recovery",
        "task": task.label,
        "mechanism_expectation": task.mechanism_expectation,
        "model": job.model,
        "seed": job.seed,
        "train_loss": outcome.train_loss,
        "validation_loss": outcome.validation_loss,
        "test_loss": outcome.test_loss,
        "test_nrmse": nrmse(outcome.test_loss, task.test_targets),
        "grad_norm": outcome.grad_norm,
        "elapsed_train_time": outcome.elapsed_time,
        "elapsed_total_time": perf_counter() - started,
        "params_trainable": count_parameters(model),
        "status": "done",
    }
    if job.model in PAC_TF_MODELS:
        row.update(pac_tf_recovery(cast("TightFrameSequenceRegressor", model), task, device))
    return row


@lru_cache(maxsize=32)
def _tasks_for_seed(config: PACExperimentConfig, seed: int) -> tuple[PACRegressionTask, ...]:
    return make_pac_tf_mechanism_tasks(config, seed)


def _task_for_job(config: PACExperimentConfig, seed: int, label: str) -> PACRegressionTask:
    for task in _tasks_for_seed(config, seed):
        if task.label == label:
            return task
    raise KeyError(label)


def pac_tf_recovery(
    model: TightFrameSequenceRegressor,
    task: PACRegressionTask,
    device: str,
) -> JsonRow:
    model.eval()
    with torch.no_grad():
        frequencies = model.frequency_values().detach().cpu()
        damping = model.damping_values().detach().cpu()
    frequency_error, matched_indices = frequency_recovery(frequencies, task.true_frequencies)
    mode_score = (
        damping[..., matched_indices].mean(dim=-1) if matched_indices else damping.mean(dim=-1)
    )
    damping_score = (
        mode_score.expand_as(task.validation_teacher_damping)
        if task.validation_teacher_damping is not None
        else mode_score
    )
    frequency_scale = (
        sum(abs(value) for value in task.true_frequencies) / len(task.true_frequencies)
        if task.true_frequencies
        else None
    )
    return {
        "frequency_recovery_mae": frequency_error,
        "frequency_recovery_relative_mae": (
            frequency_error / max(frequency_scale, 1.0e-12)
            if frequency_error is not None and frequency_scale is not None
            else None
        ),
        "damping_recovery_mae": damping_recovery(
            damping,
            task.true_dampings,
            matched_indices,
        ),
        "damping_teacher_mae": _teacher_mae(
            damping_score,
            task.validation_teacher_damping,
        ),
        "damping_correlation": _correlation(damping_score, task.validation_teacher_damping),
        "damping_regime_auc": _regime_auc(damping_score, task.validation_regime),
        "impulse_response_nmse": _impulse_nmse(model, task, device),
        "learned_frequency_count": int(frequencies.numel()),
    }


def frequency_recovery(
    learned: Tensor, truth: tuple[float, ...]
) -> tuple[float | None, tuple[int, ...]]:
    if not truth:
        return None, ()
    learned_values = learned.abs().flatten().to(dtype=torch.float64)
    truth_values = torch.tensor(tuple(abs(value) for value in truth), dtype=torch.float64)
    best_error = float("inf")
    best_indices: tuple[int, ...] = ()
    for indices in permutations(range(learned_values.numel()), len(truth)):
        selected = learned_values[list(indices)]
        error = float(torch.mean(torch.abs(selected - truth_values)).item())
        if error < best_error:
            best_error = error
            best_indices = tuple(indices)
    return (best_error if isfinite(best_error) else None), best_indices


def damping_recovery(
    learned: Tensor,
    truth: tuple[float, ...],
    matched_indices: tuple[int, ...],
) -> float | None:
    """Measure constant-mode damping recovery using the frequency assignment.

    Dynamic damping teachers are evaluated separately with token-wise MAE,
    correlation, and regime AUC.  A scalar recovery error is only well-defined
    when every teacher pole has one frequency-matched learned mode.
    """
    if not truth or len(truth) != len(matched_indices):
        return None
    per_mode = learned.reshape(-1, learned.shape[-1]).mean(dim=0).to(dtype=torch.float64)
    if any(index < 0 or index >= per_mode.numel() for index in matched_indices):
        return None
    selected = per_mode[list(matched_indices)]
    truth_values = torch.tensor(truth, dtype=torch.float64)
    return float(torch.mean(torch.abs(selected - truth_values)).item())


def _teacher_mae(values: Tensor, teacher: Tensor | None) -> float | None:
    if teacher is None:
        return None
    left = values.flatten().to(dtype=torch.float64)
    right = teacher.flatten().to(dtype=torch.float64)
    if left.numel() != right.numel():
        return None
    return float(torch.mean(torch.abs(left - right)).item())


def _correlation(values: Tensor, teacher: Tensor | None) -> float | None:
    if teacher is None:
        return None
    left = values.flatten().to(dtype=torch.float64)
    right = teacher.flatten().to(dtype=torch.float64)
    if left.numel() != right.numel():
        return None
    denominator = left.std(unbiased=False) * right.std(unbiased=False)
    if float(denominator.item()) <= 1.0e-12:
        return 0.0
    return float(((left - left.mean()) * (right - right.mean())).mean().div(denominator).item())


def _regime_auc(values: Tensor, regime: Tensor | None) -> float | None:
    if regime is None:
        return None
    labels = regime.flatten().to(dtype=torch.bool)
    scores = values.flatten()
    if scores.numel() != labels.numel():
        return None
    positives = int(labels.sum().item())
    negatives = int((~labels).sum().item())
    if positives == 0 or negatives == 0:
        return None
    order = torch.argsort(scores)
    ranks = torch.arange(1, labels.numel() + 1, dtype=torch.float64)
    positive_rank_sum = ranks[labels[order]].sum()
    return float(
        ((positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)).item()
    )


def _impulse_nmse(
    model: TightFrameSequenceRegressor, task: PACRegressionTask, device: str
) -> float | None:
    if task.diagnostic_inputs is None or task.diagnostic_targets is None:
        return None
    with torch.no_grad():
        prediction = model(task.diagnostic_inputs.to(device=device)).detach().cpu()
    target = task.diagnostic_targets
    variance = target.square().mean().clamp_min(1.0e-12)
    return float(((prediction - target).square().mean() / variance).item())
