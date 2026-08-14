from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .experiment import TrainingConfig
from .hybrid_delay_tasks import make_strict_delay_teacher_bundle
from .hybrid_experiment_types import resolve_device
from .selective_tapped_prl import SELECTIVE_VARIANTS, SelectiveVariant
from .selective_tapped_prl_metrics import model_row
from .selective_tapped_prl_tasks import (
    ablation_tasks,
    make_fir,
    make_gru,
    make_hybrid,
    make_selective,
    make_transformer,
    parameter_tasks,
    replace_tap_size,
    strict_delay_config,
)
from .selective_tapped_prl_training import train_selective_regression_model
from .selective_tapped_prl_types import (
    LabeledTask,
    SelectiveExperimentConfig,
    SelectiveMode,
    SelectiveRun,
    SelectiveSuite,
)
from .selective_tapped_prl_verdicts import (
    delay_verdict,
    parameter_verdict,
    selectivity_verdict,
)

if TYPE_CHECKING:
    from .tapped_prl_followup_schema import JsonRow, JsonValue


def run_selective_suite(*, suite: SelectiveSuite, mode: SelectiveMode) -> SelectiveRun:
    config = config_for_mode(mode)
    device = resolve_device(config.device)
    training = TrainingConfig(
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        device=device,
        seed=config.seed,
    )
    sections: dict[str, JsonRow] = {}
    if suite in {"all", "selectivity"}:
        sections["selectivity_ablation"] = selectivity_ablation_section(config, training, mode)
    if suite in {"all", "delay"}:
        sections["delay_kernel_sweep"] = delay_kernel_sweep_section(config, training, mode)
    if suite in {"all", "parameter"}:
        sections["parameter_matched"] = parameter_matched_section(config, training, mode)
    return SelectiveRun(
        mode=mode,
        suite=suite,
        device=device,
        sections=sections,
        experiment_config=_config_row(config),
        training_config=_training_row(training),
    )


def config_for_mode(mode: SelectiveMode) -> SelectiveExperimentConfig:
    if mode == "smoke":
        return SelectiveExperimentConfig(
            sample_count=16,
            validation_count=8,
            sequence_length=24,
            model_dim=4,
            modes=2,
            tap_kernel_size=5,
            epochs=3,
        )
    return SelectiveExperimentConfig(sample_count=96, validation_count=24, sequence_length=40)


def selectivity_ablation_section(
    config: SelectiveExperimentConfig,
    training: TrainingConfig,
    mode: SelectiveMode,
) -> JsonRow:
    rows: list[JsonRow] = []
    for task in ablation_tasks(config, mode):
        for variant in SELECTIVE_VARIANTS:
            torch.manual_seed(config.seed)
            model = make_selective(task, config, variant)
            outcome = train_selective_regression_model(model, task.task, training, config)
            rows.append(model_row(task, f"selective_{variant}", model, outcome))
        torch.manual_seed(config.seed)
        hybrid = make_hybrid(task, config)
        outcome = train_selective_regression_model(hybrid, task.task, training, config)
        rows.append(model_row(task, "hybrid_prl_fir_mlp", hybrid, outcome))
    return _section(
        "Selectivity Ablation",
        "Selective gates improve Fixed Tapped PRL.",
        rows,
        selectivity_verdict(rows),
    )


def delay_kernel_sweep_section(
    config: SelectiveExperimentConfig,
    training: TrainingConfig,
    mode: SelectiveMode,
) -> JsonRow:
    rows: list[JsonRow] = []
    delay_steps = (2, 6) if mode == "smoke" else (2, 4, 6, 10, 14)
    tap_sizes = (2, 5, 8) if mode == "smoke" else (1, 2, 4, 5, 8, 13, 17)
    variants: tuple[SelectiveVariant, ...] = ("fixed", "full")
    for delay in delay_steps:
        bundle = make_strict_delay_teacher_bundle(strict_delay_config(config, delay))
        task = LabeledTask(bundle.task.teacher_label, bundle.task, bundle.metadata)
        for tap_kernel_size in tap_sizes:
            adjusted = replace_tap_size(config, tap_kernel_size)
            for variant in variants:
                torch.manual_seed(adjusted.seed)
                model = make_selective(task, adjusted, variant)
                outcome = train_selective_regression_model(model, task.task, training, adjusted)
                row = model_row(task, f"selective_{variant}", model, outcome)
                row["true_delay"] = delay
                row["tap_kernel_size"] = tap_kernel_size
                row["horizon_satisfied"] = tap_kernel_size >= delay + 1
                rows.append(row)
    return _section(
        "Delay Tap Kernel Sweep",
        "K >= delay + 1 should improve delay tasks.",
        rows,
        delay_verdict(rows),
    )


def parameter_matched_section(
    config: SelectiveExperimentConfig,
    training: TrainingConfig,
    mode: SelectiveMode,
) -> JsonRow:
    rows: list[JsonRow] = []
    for task in parameter_tasks(config, mode):
        torch.manual_seed(config.seed)
        selective_fixed = make_selective(task, config, "fixed")
        outcome = train_selective_regression_model(selective_fixed, task.task, training, config)
        rows.append(model_row(task, "selective_fixed", selective_fixed, outcome))

        torch.manual_seed(config.seed)
        selective_full = make_selective(task, config, "full")
        outcome = train_selective_regression_model(selective_full, task.task, training, config)
        rows.append(model_row(task, "selective_full", selective_full, outcome))

        torch.manual_seed(config.seed)
        hybrid = make_hybrid(task, config)
        outcome = train_selective_regression_model(hybrid, task.task, training, config)
        rows.append(model_row(task, "hybrid_prl_fir_mlp", hybrid, outcome))

        torch.manual_seed(config.seed)
        fir = make_fir(task, config)
        outcome = train_selective_regression_model(fir, task.task, training, config)
        rows.append(model_row(task, "matched_fir", fir, outcome))

        torch.manual_seed(config.seed)
        gru = make_gru(task, config)
        outcome = train_selective_regression_model(gru, task.task, training, config)
        rows.append(model_row(task, "matched_gru", gru, outcome))

        torch.manual_seed(config.seed)
        transformer = make_transformer(task, config)
        outcome = train_selective_regression_model(transformer, task.task, training, config)
        rows.append(model_row(task, "matched_transformer", transformer, outcome))
    return _section(
        "Parameter-Matched Comparison",
        "Selective Tapped PRL is parameter-efficient.",
        rows,
        parameter_verdict(rows),
    )


def _section(title: str, hypothesis: str, rows: list[JsonRow], verdict: JsonRow) -> JsonRow:
    row_values: list[JsonValue] = [dict(row) for row in rows]
    payload: JsonRow = {
        "title": title,
        "hypothesis": hypothesis,
        "verdict": verdict,
        "rows": row_values,
    }
    return payload


def _config_row(config: SelectiveExperimentConfig) -> JsonRow:
    return {
        "sample_count": config.sample_count,
        "validation_count": config.validation_count,
        "sequence_length": config.sequence_length,
        "raw_input_dim": config.raw_input_dim,
        "output_dim": config.output_dim,
        "model_dim": config.model_dim,
        "modes": config.modes,
        "tap_kernel_size": config.tap_kernel_size,
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "seed": config.seed,
        "tap_entropy_weight": config.tap_entropy_weight,
        "gate_entropy_weight": config.gate_entropy_weight,
    }


def _training_row(training: TrainingConfig) -> JsonRow:
    return {
        "epochs": training.epochs,
        "learning_rate": training.learning_rate,
        "weight_decay": training.weight_decay,
        "device": training.device,
        "seed": training.seed,
    }
