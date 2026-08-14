from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING, assert_never

import torch
from torch import nn

from .controlled_damping_diagnostics import diagnostic_row
from .controlled_damping_tasks import make_controlled_damping_tasks
from .controlled_damping_types import (
    ControlledDampingConfig,
    ControlledDampingDevice,
    ControlledDampingMode,
    ControlledDampingTask,
    ControlledTrialSpec,
)
from .controlled_damping_verdicts import conclusion, summary_rows
from .experiment import TrainingConfig
from .hybrid import BRANCH_ORDER, HybridModalPRLBlock
from .hybrid_experiment_types import resolve_device
from .hybrid_metrics import count_trainable_parameters
from .models import FIRSequenceBaseline, GRUSequenceBaseline
from .selective_tapped_prl import SelectiveTappedPRLBlock, SelectiveVariant
from .selective_tapped_prl_training import train_selective_regression_model
from .selective_tapped_prl_types import SelectiveExperimentConfig

if TYPE_CHECKING:
    from .tapped_prl_followup_schema import JsonRow, JsonValue


def config_for_mode(
    mode: ControlledDampingMode,
    device: ControlledDampingDevice,
) -> ControlledDampingConfig:
    match mode:
        case "smoke":
            return ControlledDampingConfig(
                sample_count=16,
                validation_count=8,
                sequence_length=24,
                model_dim=4,
                modes=2,
                tap_kernel_size=5,
                epochs=3,
                seeds=(7,),
                beta_values=(0.0, 0.5),
                device=device,
            )
        case "full":
            return ControlledDampingConfig(
                sample_count=96,
                validation_count=24,
                sequence_length=40,
                device=device,
            )
        case unreachable:
            assert_never(unreachable)


def run_controlled_damping_suite(config: ControlledDampingConfig) -> JsonRow:
    device = resolve_device(config.device)
    runs = _run_rows(config, device)
    summary = summary_rows(runs)
    return {
        "schema_version": "controlled_damping_prl.v1",
        "device": device,
        "experiment_config": _config_row(config),
        "conclusion": conclusion(summary, runs),
        "sections": {
            "run_rows": _section("Controlled-Damping Runs", runs),
            "task_summary": _section("Task Mean Summary", summary),
        },
    }


def _run_rows(config: ControlledDampingConfig, device: str) -> list[JsonRow]:
    rows: list[JsonRow] = []
    for seed in config.seeds:
        training = _training_config(config, seed, device)
        selective_config = _selective_config(config, seed)
        for task in make_controlled_damping_tasks(config, seed):
            for spec in _trial_specs(config):
                torch.manual_seed(seed)
                model = _build_model(spec, task, config)
                started_at = perf_counter()
                outcome = train_selective_regression_model(
                    model,
                    task.task,
                    training,
                    selective_config,
                )
                row = _result_row(seed, task.label, spec, model, outcome.validation_loss)
                row["train_loss"] = outcome.final_loss
                row["elapsed_time"] = perf_counter() - started_at
                row.update(diagnostic_row(model, task, device))
                rows.append(row)
    return rows


def _trial_specs(config: ControlledDampingConfig) -> tuple[ControlledTrialSpec, ...]:
    specs: list[ControlledTrialSpec] = [
        ControlledTrialSpec("selective", "selective_fixed", "fixed"),
        ControlledTrialSpec("selective", "selective_full", "full"),
    ]
    for beta in config.beta_values:
        specs.append(ControlledTrialSpec("selective", "damping", "damping", beta))
        specs.append(ControlledTrialSpec("selective", "damping_full", "damping_full", beta))
    specs.extend(
        (
            ControlledTrialSpec("hybrid", "hybrid_prl_fir_mlp"),
            ControlledTrialSpec("fir", "matched_fir"),
            ControlledTrialSpec("gru", "matched_gru"),
        ),
    )
    return tuple(specs)


def _build_model(
    spec: ControlledTrialSpec,
    task: ControlledDampingTask,
    config: ControlledDampingConfig,
) -> nn.Module:
    raw_input_dim = task.task.train_inputs.shape[-1]
    output_dim = task.task.train_targets.shape[-1]
    match spec.kind:
        case "selective":
            if spec.variant is None:
                message = "selective trial requires a variant"
                raise RuntimeError(message)
            return SelectiveTappedPRLBlock(
                raw_input_dim=raw_input_dim,
                model_dim=config.model_dim,
                output_dim=output_dim,
                modes=config.modes,
                tap_kernel_size=config.tap_kernel_size,
                variant=_selective_variant(spec.variant),
                damping_beta=spec.damping_beta if spec.damping_beta is not None else 0.5,
            )
        case "hybrid":
            return HybridModalPRLBlock(
                raw_input_dim=raw_input_dim,
                model_dim=config.model_dim,
                output_dim=output_dim,
                modes=config.modes,
                fir_kernel_size=max(config.tap_kernel_size, 3),
                prl_tap_kernel_size=config.tap_kernel_size,
                active_branches=BRANCH_ORDER,
            )
        case "fir":
            return FIRSequenceBaseline(
                raw_input_dim=raw_input_dim,
                model_dim=config.model_dim,
                output_dim=output_dim,
                kernel_size=max(config.tap_kernel_size, 3),
            )
        case "gru":
            return GRUSequenceBaseline(
                raw_input_dim=raw_input_dim,
                model_dim=config.model_dim,
                output_dim=output_dim,
            )
        case unreachable:
            assert_never(unreachable)


def _selective_variant(value: str) -> SelectiveVariant:
    match value:
        case (
            "fixed"
            | "input_gate"
            | "tap_selective"
            | "input_tap"
            | "full"
            | "damping"
            | "damping_full"
        ):
            return value
        case _:
            message = f"unsupported selective variant: {value}"
            raise RuntimeError(message)


def _result_row(
    seed: int,
    task: str,
    spec: ControlledTrialSpec,
    model: nn.Module,
    validation_loss: float,
) -> JsonRow:
    return {
        "seed": seed,
        "task": task,
        "model": spec.name,
        "damping_beta": spec.damping_beta,
        "params": count_trainable_parameters(model),
        "validation_loss": validation_loss,
    }


def _training_config(config: ControlledDampingConfig, seed: int, device: str) -> TrainingConfig:
    return TrainingConfig(
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        device=device,
        seed=seed,
    )


def _selective_config(config: ControlledDampingConfig, seed: int) -> SelectiveExperimentConfig:
    return SelectiveExperimentConfig(
        sample_count=config.sample_count,
        validation_count=config.validation_count,
        sequence_length=config.sequence_length,
        raw_input_dim=config.raw_input_dim,
        output_dim=config.output_dim,
        model_dim=config.model_dim,
        modes=config.modes,
        tap_kernel_size=config.tap_kernel_size,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        seed=seed,
        device=config.device,
        tap_entropy_weight=config.tap_entropy_weight,
        gate_entropy_weight=config.gate_entropy_weight,
    )


def _config_row(config: ControlledDampingConfig) -> JsonRow:
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
        "seeds": list(config.seeds),
        "beta_values": list(config.beta_values),
        "tap_entropy_weight": config.tap_entropy_weight,
        "gate_entropy_weight": config.gate_entropy_weight,
    }


def _section(title: str, rows: list[JsonRow]) -> JsonRow:
    return {"title": title, "rows": _rows_value(rows)}


def _rows_value(rows: list[JsonRow]) -> list[JsonValue]:
    return [dict(row) for row in rows]
