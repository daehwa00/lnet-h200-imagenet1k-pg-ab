from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from .pac_builders import build_regression_model
from .pac_metrics import count_parameters, mechanism_row, nrmse
from .pac_model import PACHybridPRLBlock
from .pac_tasks import make_ood_task, make_pac_synthetic_tasks
from .pac_training import (
    evaluate_regression_loss,
    train_regression_model,
)

if TYPE_CHECKING:
    from .pac_types import (
        PACBranchName,
        PACExperimentConfig,
        PACModelName,
        PACRegressionTask,
        PACTrainOutcome,
    )
    from .tapped_prl_followup_schema import JsonRow


def main_synthetic(
    config: PACExperimentConfig,
    device: str,
    models: tuple[PACModelName, ...],
) -> tuple[list[JsonRow], list[JsonRow]]:
    rows: list[JsonRow] = []
    mechanisms: list[JsonRow] = []
    for seed in config.seeds:
        for task in make_pac_synthetic_tasks(config, seed):
            for model_name in models:
                model = build_regression_model(model_name, config)
                outcome = train_regression_model(model, task, config, device, seed)
                rows.append(regression_row("main", task, model_name, model, outcome))
                mechanism = mechanism_row(model, task, device)
                mechanism["model"] = model_name
                mechanism["seed"] = seed
                mechanisms.append(mechanism)
    return rows, mechanisms


def ablation(config: PACExperimentConfig, device: str) -> list[JsonRow]:
    models: tuple[PACModelName, ...] = (
        "pac_full",
        "controlled_tapped_prl_only",
        "tapped_prl_fixed",
        "fixed_prl",
        "fir_only",
        "mlp_only",
    )
    tasks = [
        task
        for task in make_pac_synthetic_tasks(config, config.seeds[0])
        if task.label in {"active_damping_teacher", "random_fir_teacher", "delayed_oscillatory"}
    ]
    rows: list[JsonRow] = []
    for task in tasks:
        for model_name in models:
            model = build_regression_model(model_name, config)
            outcome = train_regression_model(model, task, config, device, config.seeds[0])
            rows.append(regression_row("ablation", task, model_name, model, outcome))
    return rows


def knockout(config: PACExperimentConfig, device: str) -> list[JsonRow]:
    rows: list[JsonRow] = []
    tasks = [
        task
        for task in make_pac_synthetic_tasks(config, config.seeds[0])
        if task.label in {"active_damping_teacher", "random_fir_teacher"}
    ]
    for task in tasks:
        model = build_regression_model("pac_full", config)
        outcome = train_regression_model(model, task, config, device, config.seeds[0])
        if isinstance(model, PACHybridPRLBlock):
            rows.extend(_knockout_rows(model, task, outcome.test_loss, device))
    return rows


def ood(config: PACExperimentConfig, device: str) -> list[JsonRow]:
    rows: list[JsonRow] = []
    train_task = next(
        task
        for task in make_pac_synthetic_tasks(config, config.seeds[0])
        if task.label == "active_damping_teacher"
    )
    for model_name in ("pac_full", "gru"):
        model = build_regression_model(model_name, config)
        outcome = train_regression_model(model, train_task, config, device, config.seeds[0])
        rows.extend(_ood_rows(model, model_name, config, device, outcome.test_loss))
    return rows


class KnockoutWrapper(nn.Module):
    def __init__(self, model: PACHybridPRLBlock, disabled: tuple[PACBranchName, ...]) -> None:
        super().__init__()
        self.model = model
        self.disabled = disabled

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model.forward_with_disabled(inputs, self.disabled)


def regression_row(
    prefix: str,
    task: PACRegressionTask,
    model_name: PACModelName,
    model: nn.Module,
    outcome: PACTrainOutcome,
) -> JsonRow:
    return {
        "section": prefix,
        "task": task.label,
        "model": model_name,
        "params": count_parameters(model),
        "train_loss": outcome.train_loss,
        "validation_loss": outcome.validation_loss,
        "test_loss": outcome.test_loss,
        "validation_nrmse": nrmse(outcome.validation_loss, task.validation_targets),
        "grad_norm": outcome.grad_norm,
        "elapsed_time": outcome.elapsed_time,
    }


def _knockout_rows(
    model: PACHybridPRLBlock,
    task: PACRegressionTask,
    full_loss: float,
    device: str,
) -> list[JsonRow]:
    rows: list[JsonRow] = []
    for branch in ("prl", "fir", "mlp"):
        loss = evaluate_regression_loss(
            KnockoutWrapper(model, (branch,)),
            task.test_inputs.to(device=device),
            task.test_targets.to(device=device),
        )
        rows.append(
            {
                "task": task.label,
                "knockout": f"{branch}_off",
                "full_test_loss": full_loss,
                "knockout_loss": loss,
                "relative_delta": (loss - full_loss) / max(full_loss, 1.0e-12),
            }
        )
    return rows


def _ood_rows(
    model: nn.Module,
    model_name: str,
    config: PACExperimentConfig,
    device: str,
    baseline_loss: float,
) -> list[JsonRow]:
    rows: list[JsonRow] = []
    for length in _ood_lengths(config):
        task = make_ood_task(config, config.seeds[0], sequence_length=length, delay=4)
        loss = evaluate_regression_loss(
            model,
            task.test_inputs.to(device=device),
            task.test_targets.to(device=device),
        )
        rows.append(
            {
                "model": model_name,
                "ood_type": "length",
                "value": length,
                "train_test_loss": baseline_loss,
                "ood_loss": loss,
                "ood_train_ratio": loss / max(baseline_loss, 1.0e-12),
            }
        )
    return rows


def _ood_lengths(config: PACExperimentConfig) -> tuple[int, ...]:
    if config.sample_count <= 32:
        return config.sequence_length, config.sequence_length + 8
    return 64, 128, 256, 512
