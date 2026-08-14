from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Literal

import torch

from .pac_builders import build_regression_model
from .pac_model import PACHybridPRLBlock
from .pac_overnight_io import append_csv_row, read_csv
from .pac_tasks import make_ood_task, make_pac_synthetic_tasks
from .pac_training import evaluate_regression_loss, train_regression_model

if TYPE_CHECKING:
    from pathlib import Path

    from .pac_types import PACExperimentConfig, PACRegressionTask
    from .tapped_prl_followup_schema import JsonValue

KnockoutType = Literal[
    "prl_off",
    "fir_off",
    "mlp_off",
    "damping_control_off",
    "tap_off",
    "complex_frequency_off",
]


def run_expanded_knockout(
    output_root: Path,
    config: PACExperimentConfig,
    device: str,
    seeds: tuple[int, ...],
) -> None:
    path = output_root / "results" / "knockout_damping_off.csv"
    completed = _completed_knockouts(path)
    for seed in seeds:
        run_config = replace(config, seeds=(seed,))
        for task in _knockout_tasks(run_config, seed):
            model = build_regression_model("pac_full", run_config)
            train_regression_model(model, task, run_config, device, seed)
            full_loss = evaluate_regression_loss(
                model, task.test_inputs.to(device=device), task.test_targets.to(device=device)
            )
            if isinstance(model, PACHybridPRLBlock):
                for knockout in _knockouts():
                    key = (task.label, seed, knockout)
                    if key in completed:
                        continue
                    row = _knockout_row(model, task, seed, full_loss, knockout, device)
                    append_csv_row(path, row)
                    completed.add(key)
    _write_report(output_root)


def _completed_knockouts(path: Path) -> set[tuple[str, int, str]]:
    completed: set[tuple[str, int, str]] = set()
    for row in read_csv(path):
        task = row.get("task")
        seed = row.get("seed")
        knockout = row.get("knockout_type")
        if task is None or seed is None or knockout is None:
            continue
        try:
            completed.add((task, int(seed), knockout))
        except ValueError:
            continue
    return completed


def _knockout_tasks(config: PACExperimentConfig, seed: int) -> tuple[PACRegressionTask, ...]:
    tasks = make_pac_synthetic_tasks(config, seed)
    random_fir = next(task for task in tasks if task.label == "random_fir_teacher")
    active = next(task for task in tasks if task.label == "active_damping_teacher")
    oscillatory = next(task for task in tasks if task.label == "delayed_oscillatory")
    delayed = replace(
        make_ood_task(config, seed + 303, sequence_length=config.sequence_length, delay=6),
        label="delayed_context_damped_exponential",
    )
    return random_fir, active, oscillatory, delayed


def _knockout_row(
    model: PACHybridPRLBlock,
    task: PACRegressionTask,
    seed: int,
    full_loss: float,
    knockout: KnockoutType,
    device: str,
) -> dict[str, JsonValue]:
    loss = _knockout_loss(model, task, knockout, device)
    return {
        "task": task.label,
        "seed": seed,
        "full_test_loss": full_loss,
        "knockout_type": knockout,
        "knockout_loss": loss,
        "absolute_delta": loss - full_loss,
        "relative_delta": (loss - full_loss) / max(full_loss, 1.0e-12),
    }


def _knockout_loss(
    model: PACHybridPRLBlock,
    task: PACRegressionTask,
    knockout: KnockoutType,
    device: str,
) -> float:
    match knockout:
        case "prl_off" | "fir_off" | "mlp_off":
            branch = knockout.removesuffix("_off")
            return evaluate_regression_loss(
                _BranchOff(model, branch),
                task.test_inputs.to(device=device),
                task.test_targets.to(device=device),
            )
        case "damping_control_off":
            return _temporarily_zero_damping(model, task, device)
        case "tap_off":
            return _temporarily_current_tap(model, task, device)
        case "complex_frequency_off":
            return _temporarily_zero_frequency(model, task, device)


class _BranchOff(torch.nn.Module):
    def __init__(self, model: PACHybridPRLBlock, branch: str) -> None:
        super().__init__()
        self.model = model
        self.branch = branch

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.branch == "prl":
            return self.model.forward_with_disabled(inputs, ("prl",))
        if self.branch == "fir":
            return self.model.forward_with_disabled(inputs, ("fir",))
        return self.model.forward_with_disabled(inputs, ("mlp",))


def _temporarily_zero_damping(
    model: PACHybridPRLBlock, task: PACRegressionTask, device: str
) -> float:
    prl_branch = model.require_prl_branch()
    original = prl_branch.damping_control_range
    try:
        prl_branch.damping_control_range = 0.0
        return _loss(model, task, device)
    finally:
        prl_branch.damping_control_range = original


def _temporarily_current_tap(
    model: PACHybridPRLBlock, task: PACRegressionTask, device: str
) -> float:
    prl_branch = model.require_prl_branch()
    original = prl_branch.tap_logits.detach().clone()
    try:
        with torch.no_grad():
            prl_branch.tap_logits.fill_(-20.0)
            prl_branch.tap_logits[:, 0] = 20.0
        return _loss(model, task, device)
    finally:
        with torch.no_grad():
            prl_branch.tap_logits.copy_(original)


def _temporarily_zero_frequency(
    model: PACHybridPRLBlock, task: PACRegressionTask, device: str
) -> float:
    prl_branch = model.require_prl_branch()
    original = prl_branch.raw_frequency.detach().clone()
    try:
        with torch.no_grad():
            prl_branch.raw_frequency.zero_()
        return _loss(model, task, device)
    finally:
        with torch.no_grad():
            prl_branch.raw_frequency.copy_(original)


def _loss(model: PACHybridPRLBlock, task: PACRegressionTask, device: str) -> float:
    return evaluate_regression_loss(
        model, task.test_inputs.to(device=device), task.test_targets.to(device=device)
    )


def _knockouts() -> tuple[KnockoutType, ...]:
    return (
        "prl_off",
        "fir_off",
        "mlp_off",
        "damping_control_off",
        "tap_off",
        "complex_frequency_off",
    )


def _write_report(output_root: Path) -> None:
    path = output_root / "reports" / "overnight_knockout.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# Expanded Knockout\n\nknockout_status: mixed\n", encoding="utf-8")
