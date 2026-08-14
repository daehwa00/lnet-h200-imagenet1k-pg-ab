from __future__ import annotations

from dataclasses import replace
from math import log
from typing import TYPE_CHECKING

import torch
from torch import nn

from .pac_model import PACHybridPRLBlock
from .pac_paper_queue_models import build_paper_regressor
from .pac_tasks import make_pac_synthetic_tasks
from .pac_training import evaluate_regression_loss, train_regression_model

if TYPE_CHECKING:
    from .pac_paper_queue_types import PaperJob
    from .pac_types import PACBranchName, PACExperimentConfig, PACRegressionTask
    from .tapped_prl_followup_schema import JsonRow


class PaperQueueModelError(TypeError):
    def __init__(self) -> None:
        super().__init__("pac_full must build PACHybridPRLBlock")


def counterfactual_row(config: PACExperimentConfig, device: str, job: PaperJob) -> JsonRow:
    task = _synthetic_task(config, job.seed, "active_damping_teacher")
    model = build_paper_regressor("pac_full", replace(config, seeds=(job.seed,)))
    outcome = train_regression_model(model, task, config, device, job.seed)
    if not isinstance(model, PACHybridPRLBlock):
        raise PaperQueueModelError
    low_loss, high_loss = _clamp_losses(model, task, device)
    before, after = _regime_flip_damping(model, task, device)
    return {
        "experiment_group": "damping_counterfactual",
        "model": "pac_full",
        "task": task.label,
        "seed": job.seed,
        "train_loss": outcome.train_loss,
        "validation_loss": outcome.validation_loss,
        "test_loss": outcome.test_loss,
        "clamp_low_loss": low_loss,
        "clamp_high_loss": high_loss,
        "tail_half_life_low": log(2.0) / 0.05,
        "tail_half_life_high": log(2.0) / 0.8,
        "regime_flip_alpha_before": before,
        "regime_flip_alpha_after": after,
        "regime_flip_delta": after - before,
    }


def role_ablation_row(config: PACExperimentConfig, device: str, job: PaperJob) -> JsonRow:
    task = _synthetic_task(config, job.seed, job.task)
    model = build_paper_regressor("pac_full", replace(config, seeds=(job.seed,)))
    train_regression_model(model, task, config, device, job.seed)
    if not isinstance(model, PACHybridPRLBlock):
        raise PaperQueueModelError
    full = _loss(model, task, device)
    loss = _knockout_loss(model, task, job.model, device)
    return {
        "experiment_group": "role_ablation",
        "task": task.label,
        "seed": job.seed,
        "model": "pac_full",
        "knockout_type": job.model,
        "full_test_loss": full,
        "knockout_loss": loss,
        "absolute_delta": loss - full,
        "relative_delta": (loss - full) / max(full, 1.0e-12),
    }


def _synthetic_task(config: PACExperimentConfig, seed: int, name: str) -> PACRegressionTask:
    return {task.label: task for task in make_pac_synthetic_tasks(config, seed)}[name]


def _clamp_losses(
    model: PACHybridPRLBlock, task: PACRegressionTask, device: str
) -> tuple[float, float]:
    return _clamped_loss(model, task, device, 0.05), _clamped_loss(model, task, device, 0.8)


def _clamped_loss(
    model: PACHybridPRLBlock, task: PACRegressionTask, device: str, alpha: float
) -> float:
    branch = model.require_prl_branch()
    original_decay = branch.raw_decay.detach().clone()
    original_range = branch.damping_control_range
    try:
        with torch.no_grad():
            target = torch.tensor(alpha - branch.min_decay).clamp_min(1.0e-6)
            branch.raw_decay.fill_(float(torch.log(torch.expm1(target)).item()))
            branch.damping_control_range = 0.0
        return _loss(model, task, device)
    finally:
        with torch.no_grad():
            branch.raw_decay.copy_(original_decay)
            branch.damping_control_range = original_range


def _regime_flip_damping(
    model: PACHybridPRLBlock, task: PACRegressionTask, device: str
) -> tuple[float, float]:
    inputs = task.validation_inputs.to(device=device)
    flipped = inputs.clone()
    if flipped.shape[-1] > 1:
        flipped[..., 1] = -flipped[..., 1]
    branch = model.require_prl_branch()
    with torch.no_grad():
        before = branch.effective_damping_values(model.input_projection(inputs)).mean()
        after = branch.effective_damping_values(model.input_projection(flipped)).mean()
    return float(before.item()), float(after.item())


def _knockout_loss(
    model: PACHybridPRLBlock, task: PACRegressionTask, knockout: str, device: str
) -> float:
    if knockout in {"prl_off", "fir_off", "mlp_off"}:
        return _loss(_BranchOff(model, knockout.removesuffix("_off")), task, device)
    if knockout == "direct_term_off":
        return _zero_parameter_loss(model, task, device, "direct_term")
    if knockout == "fir_pointwise_off":
        return _zero_module_loss(model, task, device, model.fir_pointwise)
    return _loss(model, task, device)


def _zero_parameter_loss(
    model: PACHybridPRLBlock, task: PACRegressionTask, device: str, name: str
) -> float:
    parameter = getattr(model.require_prl_branch(), name)
    original = parameter.detach().clone()
    try:
        with torch.no_grad():
            parameter.zero_()
        return _loss(model, task, device)
    finally:
        with torch.no_grad():
            parameter.copy_(original)


def _zero_module_loss(
    model: PACHybridPRLBlock, task: PACRegressionTask, device: str, module: nn.Module | None
) -> float:
    if module is None:
        return _loss(model, task, device)
    originals = [parameter.detach().clone() for parameter in module.parameters()]
    try:
        with torch.no_grad():
            for parameter in module.parameters():
                parameter.zero_()
        return _loss(model, task, device)
    finally:
        with torch.no_grad():
            for parameter, original in zip(module.parameters(), originals, strict=True):
                parameter.copy_(original)


def _loss(model: nn.Module, task: PACRegressionTask, device: str) -> float:
    return evaluate_regression_loss(
        model,
        task.test_inputs.to(device=device),
        task.test_targets.to(device=device),
    )


class _BranchOff(nn.Module):
    def __init__(self, model: PACHybridPRLBlock, branch: str) -> None:
        super().__init__()
        self.model = model
        self.branch: PACBranchName
        self.branch = _branch_name(branch)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model.forward_with_disabled(inputs, (self.branch,))


def _branch_name(branch: str) -> PACBranchName:
    match branch:
        case "prl" | "fir" | "mlp":
            return branch
        case _:
            message = f"unsupported branch knockout: {branch}"
            raise KeyError(message)
