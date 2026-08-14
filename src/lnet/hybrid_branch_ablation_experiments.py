from __future__ import annotations

from statistics import mean
from time import perf_counter
from typing import TYPE_CHECKING, assert_never

from .experiment import TrainingConfig
from .hybrid import BRANCH_ORDER
from .hybrid_branch_ablation_metrics import (
    active_parameter_count,
    contribution_metrics,
    knockout_loss_rows,
    total_parameter_count,
)
from .hybrid_branch_ablation_tasks import make_seed_tasks
from .hybrid_branch_ablation_types import (
    BranchAblationMode,
    BranchSpec,
    HybridBranchAblationConfig,
)
from .hybrid_branch_ablation_verdicts import mlp_necessity_conclusion, task_mlp_delta_rows
from .hybrid_experiment_types import (
    HybridExperimentConfig,
    fit_hybrid,
    resolve_device,
)

if TYPE_CHECKING:
    from .advanced_experiments import SequenceRegressionTask
    from .experiment import SyntheticLaplaceTask
    from .hybrid import HybridModalPRLBlock
    from .hybrid_experiment_types import DeviceChoice, TrainedHybridModel
    from .tapped_prl_followup_schema import JsonRow, JsonValue


def config_for_mode(mode: BranchAblationMode, device: DeviceChoice) -> HybridBranchAblationConfig:
    match mode:
        case "smoke":
            return HybridBranchAblationConfig(
                sample_count=24,
                validation_count=8,
                sequence_length=24,
                epochs=3,
                seeds=(7,),
                device=device,
                task_names=("modal_teacher", "strict_delay_6"),
                branch_specs=(
                    BranchSpec(name="prl_fir", branches=("prl", "fir")),
                    BranchSpec(name="prl_fir_mlp", branches=BRANCH_ORDER),
                ),
            )
        case "full":
            return HybridBranchAblationConfig(device=device)
        case unreachable:
            assert_never(unreachable)


def run_hybrid_branch_ablation(
    config: HybridBranchAblationConfig,
) -> JsonRow:
    device = resolve_device(config.device)
    run_rows, knockout_rows = _run_rows(config, device)
    summary_rows = _summary_rows(run_rows)
    delta_rows = task_mlp_delta_rows(summary_rows)
    conclusion = mlp_necessity_conclusion(summary_rows, knockout_rows)
    sections: JsonRow = {
        "branch_ablation": _section("Hybrid Branch Ablation", run_rows),
        "task_summary": _section("Task Mean Summary", summary_rows + delta_rows),
        "branch_knockout": _section("Full Hybrid Branch Knockout", knockout_rows),
    }
    payload: JsonRow = {
        "schema_version": "hybrid_branch_ablation.v1",
        "device": device,
        "experiment_config": _config_row(config),
        "conclusion": conclusion,
        "sections": sections,
    }
    return payload


def _run_rows(
    config: HybridBranchAblationConfig,
    device: str,
) -> tuple[list[JsonRow], list[JsonRow]]:
    run_rows: list[JsonRow] = []
    knockout_rows: list[JsonRow] = []
    for seed in config.seeds:
        local_config = _hybrid_config(config, seed)
        training = _training_config(config, seed, device)
        for task in make_seed_tasks(config, seed):
            for spec in config.branch_specs:
                started_at = perf_counter()
                trained = fit_hybrid(task, local_config, training, spec.branches)
                elapsed = perf_counter() - started_at
                metrics = contribution_metrics(trained.model, task.task.validation_inputs)
                row = _result_row(seed, task.label, spec, trained, elapsed)
                row.update(metrics)
                run_rows.append(row)
                if spec.branches == BRANCH_ORDER:
                    knockout_rows.extend(
                        _task_knockout_rows(seed, task.label, trained.model, task.task, device),
                    )
    return run_rows, knockout_rows


def _summary_rows(rows: list[JsonRow]) -> list[JsonRow]:
    keys = sorted({(_as_str(row["task"]), _as_str(row["model"])) for row in rows})
    summaries: list[JsonRow] = []
    for task, model in keys:
        values = [
            _as_float(row["validation_loss"])
            for row in rows
            if row["task"] == task and row["model"] == model
        ]
        active_counts = [
            _as_int(row["active_params"])
            for row in rows
            if row["task"] == task and row["model"] == model
        ]
        summaries.append(
            {
                "task": task,
                "model": model,
                "mean_validation_loss": mean(values),
                "seed_count": len(values),
                "mean_active_params": mean(active_counts),
            },
        )
    return summaries


def _result_row(
    seed: int,
    task_label: str,
    spec: BranchSpec,
    trained: TrainedHybridModel,
    elapsed: float,
) -> JsonRow:
    return {
        "seed": seed,
        "task": task_label,
        "model": spec.name,
        "branches": "+".join(spec.branches),
        "total_params": total_parameter_count(trained.model),
        "active_params": active_parameter_count(trained.model, spec.branches),
        "validation_loss": trained.outcome.validation_loss,
        "train_loss": trained.outcome.final_loss,
        "elapsed_time": elapsed,
    }


def _task_knockout_rows(
    seed: int,
    task: str,
    model: HybridModalPRLBlock,
    regression_task: SequenceRegressionTask | SyntheticLaplaceTask,
    device: str,
) -> list[JsonRow]:
    rows: list[JsonRow] = []
    for row in knockout_loss_rows(model, regression_task, device):
        enriched: JsonRow = {"seed": seed, "task": task}
        enriched.update(row)
        rows.append(enriched)
    return rows


def _hybrid_config(config: HybridBranchAblationConfig, seed: int) -> HybridExperimentConfig:
    return HybridExperimentConfig(
        sample_count=config.sample_count,
        validation_count=config.validation_count,
        sequence_length=config.sequence_length,
        raw_input_dim=config.raw_input_dim,
        output_dim=config.output_dim,
        model_dim=config.model_dim,
        modes=config.modes,
        fir_kernel_size=config.fir_kernel_size,
        prl_tap_kernel_size=config.prl_tap_kernel_size,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        seed=seed,
        device=config.device,
    )


def _training_config(config: HybridBranchAblationConfig, seed: int, device: str) -> TrainingConfig:
    return TrainingConfig(
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        device=device,
        seed=seed,
    )


def _config_row(config: HybridBranchAblationConfig) -> JsonRow:
    return {
        "sample_count": config.sample_count,
        "validation_count": config.validation_count,
        "sequence_length": config.sequence_length,
        "model_dim": config.model_dim,
        "modes": config.modes,
        "fir_kernel_size": config.fir_kernel_size,
        "prl_tap_kernel_size": config.prl_tap_kernel_size,
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "seeds": list(config.seeds),
        "models": [spec.name for spec in config.branch_specs],
    }


def _section(title: str, rows: list[JsonRow]) -> JsonRow:
    return {"title": title, "rows": _rows_value(rows)}


def _rows_value(rows: list[JsonRow]) -> list[JsonValue]:
    values: list[JsonValue] = []
    for row in rows:
        value: JsonValue = dict(row)
        values.append(value)
    return values


def _as_str(value: JsonValue) -> str:
    if isinstance(value, str):
        return value
    message = f"expected string scalar, got {type(value).__name__}"
    raise TypeError(message)


def _as_float(value: JsonValue) -> float:
    if isinstance(value, int | float | str):
        return float(value)
    message = f"expected numeric scalar, got {type(value).__name__}"
    raise TypeError(message)


def _as_int(value: JsonValue) -> int:
    if isinstance(value, int | float | str):
        return int(value)
    message = f"expected integer scalar, got {type(value).__name__}"
    raise TypeError(message)
