# ruff: noqa: C901, E501, EM101, EM102, PLR0912, T201, TRY003
"""Restart-safe, parameter-matched variable-step endpoint campaign.

The campaign has a hard data-access boundary.  Learning-rate selection sees
only generated TRAIN and validation tensors.  OOD conditions are constructed
only by final workers after a selection artifact has been frozen.

"Causal" describes the estimand: every model predicts the state at the end of
an observed prefix and never receives a target or observation after that
endpoint.  It does not claim that every family uses a triangular internal
operator; the public family-native record encoders are retained.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import mean, stdev
from time import perf_counter
from typing import TYPE_CHECKING, Literal, Protocol, cast

import torch
from torch import Tensor, nn

from .alphabet import Alphabet
from .pac_external_benchmarks import _build_continuous_model  # pyright: ignore[reportPrivateUsage]
from .pac_matched_zoh_ood import (
    _build_case,  # pyright: ignore[reportPrivateUsage]
    _forcing_parameters,  # pyright: ignore[reportPrivateUsage]
    matched_zoh_conditions,
)
from .pac_metrics import count_parameters, nrmse
from .pac_types import PACDevice, PACExperimentConfig, PACRegressionTask

if TYPE_CHECKING:
    from .pac_external_benchmarks import ExternalModelFamily

ModelName = Literal["alphabet", "cnn1d", "tcn", "mamba", "gru", "lstm", "transformer"]
BaselineName = Literal["cnn1d", "tcn", "mamba", "gru", "lstm", "transformer"]
Regime = Literal["unit", "mixed_dt"]
Variant = Literal[
    "correct_dt_mask",
    "unit_dt_mask",
    "shuffled_dt_mask",
    "correct_dt_no_mask",
    "unit_dt_no_mask",
]

MODELS: tuple[ModelName, ...] = (
    "alphabet",
    "cnn1d",
    "tcn",
    "mamba",
    "gru",
    "lstm",
    "transformer",
)
REGIMES: tuple[Regime, ...] = ("unit", "mixed_dt")
VARIANTS: tuple[Variant, ...] = (
    "correct_dt_mask",
    "unit_dt_mask",
    "shuffled_dt_mask",
    "correct_dt_no_mask",
    "unit_dt_no_mask",
)
LEARNING_RATES = (1.0e-3, 3.0e-3, 1.0e-2)
SELECTION_SEEDS = (7, 11, 19)
FINAL_SEEDS = (23, 31, 43, 47, 59)
HELD_OUT_DELTAS = (0.25, 0.5, 2.0, 3.0)
MIXED_IRREGULARITY = 0.45
FACTORIAL_IRREGULARITY = 0.75
FACTORIAL_MISSING_RATE = 0.30
TARGET_PARAMS = 3_138
PARAMETER_TOLERANCE = 0.08
WIDTHS: dict[BaselineName, int] = {
    "cnn1d": 10,
    "tcn": 17,
    "mamba": 15,
    "gru": 29,
    "lstm": 25,
    "transformer": 18,
}
EXPECTED_PARAMS: dict[ModelName, int] = {
    "alphabet": 3_138,
    "cnn1d": 3_372,
    "tcn": 3_045,
    "mamba": 3_167,
    "gru": 3_105,
    "lstm": 3_152,
    "transformer": 3_260,
}
DEFAULT_ROOT = Path(
    "results/variable-step"
)
SOURCE_FILES = (
    "src/lnet/alphabet.py",
    "src/lnet/alphabet_backbone.py",
    "src/lnet/pac_tight_frame_models.py",
    "src/lnet/pac_efp16_exact_split_training.py",
    "src/lnet/pac_variable_step_causal_campaign.py",
    "src/lnet/pac_matched_zoh_ood.py",
)


@dataclass(frozen=True, slots=True)
class SelectionJob:
    regime: Regime
    model: ModelName
    learning_rate: float
    seed: int

    @property
    def key(self) -> str:
        return (
            f"variable_step_causal__selection__{self.regime}__{self.model}"
            f"__lr{self.learning_rate:g}__seed{self.seed}"
        )


@dataclass(frozen=True, slots=True)
class FinalJob:
    regime: Regime
    model: ModelName
    learning_rate: float
    seed: int

    @property
    def key(self) -> str:
        return f"variable_step_causal__final__{self.regime}__{self.model}__seed{self.seed}"


class _MetadataExactSplitRuntime(Protocol):
    def step(
        self,
        inputs: Tensor,
        targets: Tensor,
        *,
        time_delta: Tensor | None = None,
        observation_mask: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> Tensor: ...

    def close(self) -> None: ...

    def activate(self) -> None: ...

    def destroy(self) -> None: ...


class _PackedAlphabetExactSplitRuntime:
    """Preserve the packed campaign ABI around the metadata-native runtime."""

    def __init__(self, runtime: _MetadataExactSplitRuntime) -> None:
        self.runtime = runtime
        backend = getattr(runtime, "training_backend", "exact_split")
        self.training_backend = f"packed_metadata_{backend}"

    def step(self, inputs: Tensor, targets: Tensor) -> Tensor:
        return self.runtime.step(
            inputs[..., :2],
            targets,
            time_delta=inputs[..., 2:3],
            observation_mask=inputs[..., 3:4],
        )

    def close(self) -> None:
        self.runtime.close()

    def activate(self) -> None:
        self.runtime.activate()

    def destroy(self) -> None:
        self.runtime.destroy()


class _AlphabetEndpoint(nn.Module):
    """ALPHABET endpoint adapter with explicit metadata interventions."""

    def __init__(self, config: PACExperimentConfig) -> None:
        super().__init__()
        active = replace(config, raw_input_dim=2, model_dim=32, modes=16)
        self.model = Alphabet(active, active.output_dim, objective="regression")

    def forward(self, inputs: Tensor) -> Tensor:
        return self.forward_variant(inputs, "correct_dt_mask")

    def forward_variant(self, inputs: Tensor, variant: Variant) -> Tensor:
        delta = inputs[..., 2:3]
        mask: Tensor | None = inputs[..., 3:4]
        if variant in {"correct_dt_no_mask", "unit_dt_no_mask"}:
            mask = None
        return self.model(inputs[..., :2], time_delta=delta, observation_mask=mask)

    def post_optimizer_step(self) -> None:
        self.model.post_optimizer_step()

    def finalize_constraints(self) -> None:
        self.model.finalize_constraints()

    def prepare_external_exact_split_runtime(
        self,
        optimizer: torch.optim.AdamW,
        inputs: Tensor,
        targets: Tensor,
        *,
        objective: Literal["multiclass", "multilabel", "forecasting"],
        grad_clip_norm: float,
    ) -> _PackedAlphabetExactSplitRuntime:
        runtime = self.model.prepare_external_exact_split_runtime(
            optimizer,
            inputs[..., :2],
            targets,
            objective=objective,
            grad_clip_norm=grad_clip_norm,
            time_delta=inputs[..., 2:3],
            observation_mask=inputs[..., 3:4],
            metadata_prevalidated=True,
        )
        return _PackedAlphabetExactSplitRuntime(
            cast("_MetadataExactSplitRuntime", cast("object", runtime))
        )


def selection_jobs() -> tuple[SelectionJob, ...]:
    return tuple(
        SelectionJob(regime, model, learning_rate, seed)
        for regime in REGIMES
        for model in MODELS
        for learning_rate in LEARNING_RATES
        for seed in SELECTION_SEEDS
    )


def _config(root: Path, seed: int, learning_rate: float, *, smoke: bool) -> PACExperimentConfig:
    return PACExperimentConfig(
        sample_count=64 if smoke else 2048,
        validation_count=32 if smoke else 512,
        test_count=32 if smoke else 512,
        sequence_length=60,
        raw_input_dim=4,
        output_dim=2,
        model_dim=32,
        modes=16,
        epochs=1 if smoke else 100,
        batch_size=16 if smoke else 64,
        learning_rate=learning_rate,
        weight_decay=1.0e-4,
        grad_clip_norm=1.0,
        seeds=(seed,),
        device=cast("PACDevice", "cuda"),
        output_dir=root,
        compile_mode="none",
        precision="fp32",
    )


def _split(count: int, seed: int, regime: Regime) -> tuple[Tensor, Tensor]:
    if regime == "mixed_dt":
        return _build_case(
            _forcing_parameters(count, seed),
            irregularity=MIXED_IRREGULARITY,
            perturbation_seed=seed + 10_000,
        )
    return _build_case(_forcing_parameters(count, seed))


def _selection_task(config: PACExperimentConfig, seed: int, regime: Regime) -> PACRegressionTask:
    """Construct TRAIN/validation only; TEST is physically absent."""
    train_inputs, train_targets = _split(config.sample_count, seed + 101, regime)
    validation_inputs, validation_targets = _split(config.validation_count, seed + 211, regime)
    empty_inputs = torch.empty(0, config.sequence_length, 4)
    empty_targets = torch.empty(0, 2)
    return PACRegressionTask(
        label=f"variable_step_causal_selection_{regime}",
        train_inputs=train_inputs,
        train_targets=train_targets[:, -1],
        validation_inputs=validation_inputs,
        validation_targets=validation_targets[:, -1],
        test_inputs=empty_inputs,
        test_targets=empty_targets,
        true_delay=0,
        true_frequency=math.pi / 4,
        true_frequencies=(math.pi / 4,),
        true_dampings=(0.8,),
        mechanism_expectation="positive",
    )


def _final_task(config: PACExperimentConfig, seed: int, regime: Regime) -> PACRegressionTask:
    train_inputs, train_targets = _split(config.sample_count, seed + 101, regime)
    validation_inputs, validation_targets = _split(config.validation_count, seed + 211, regime)
    # Both training regimes share the same regular, fully observed ID endpoint.
    test_inputs, test_targets = _split(config.test_count, seed + 307, "unit")
    return PACRegressionTask(
        label=f"variable_step_causal_final_{regime}",
        train_inputs=train_inputs,
        train_targets=train_targets[:, -1],
        validation_inputs=validation_inputs,
        validation_targets=validation_targets[:, -1],
        test_inputs=test_inputs,
        test_targets=test_targets[:, -1],
        true_delay=0,
        true_frequency=math.pi / 4,
        true_frequencies=(math.pi / 4,),
        true_dampings=(0.8,),
        mechanism_expectation="positive",
    )


def _build_model(model: ModelName, config: PACExperimentConfig, seed: int) -> nn.Module:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if model == "alphabet":
            built: nn.Module = _AlphabetEndpoint(config)
        else:
            built = _build_continuous_model(
                cast("ExternalModelFamily", model),
                WIDTHS[model],
                4,
                2,
                config,
                "EFP16",
                objective="regression",
            )
    actual = count_parameters(built)
    if actual != EXPECTED_PARAMS[model]:
        raise RuntimeError(
            f"parameter lock changed for {model}: {actual} != {EXPECTED_PARAMS[model]}"
        )
    if abs(actual - TARGET_PARAMS) / TARGET_PARAMS > PARAMETER_TOLERANCE:
        raise RuntimeError(f"parameter match exceeds tolerance for {model}: {actual}")
    return built


def metadata_view(inputs: Tensor, variant: Variant, *, seed: int) -> Tensor:
    """Transform a complete condition before batching for invariant shuffling."""
    result = inputs.clone()
    if variant in {"unit_dt_mask", "unit_dt_no_mask"}:
        result[..., 2] = 1.0
    elif variant == "shuffled_dt_mask":
        generator = torch.Generator().manual_seed(seed)
        random = torch.rand(result.shape[0], result.shape[1], generator=generator)
        order = random.argsort(dim=1)
        result[..., 2] = result[..., 2].gather(1, order)
    if variant in {"correct_dt_no_mask", "unit_dt_no_mask"}:
        result[..., 3] = 1.0
    return result


def _callback(model: nn.Module, name: str) -> None:
    callback = getattr(model, name, None)
    if callable(callback):
        callback()


def _predict(
    model: nn.Module,
    model_name: ModelName,
    inputs: Tensor,
    *,
    variant: Variant,
    seed: int,
    device: str,
    batch_size: int,
) -> Tensor:
    viewed = metadata_view(inputs, variant, seed=seed).to(device)
    predictions: list[Tensor] = []
    model.eval()
    with torch.no_grad():
        for batch in viewed.split(batch_size):
            if model_name == "alphabet":
                output = cast("_AlphabetEndpoint", model).forward_variant(batch, variant)
            else:
                output = model(batch)
            if output.shape != (batch.shape[0], 2):
                raise RuntimeError(
                    f"invalid endpoint shape for {model_name}: {tuple(output.shape)}"
                )
            predictions.append(output.detach().cpu())
    return torch.cat(predictions)


def _loss(
    model: nn.Module,
    model_name: ModelName,
    inputs: Tensor,
    targets: Tensor,
    *,
    device: str,
    batch_size: int,
) -> float:
    predicted = _predict(
        model,
        model_name,
        inputs,
        variant="correct_dt_mask",
        seed=0,
        device=device,
        batch_size=batch_size,
    )
    return float(torch.nn.functional.mse_loss(predicted, targets.cpu()).item())


def _eager_training_step(
    model: nn.Module,
    model_name: ModelName,
    optimizer: torch.optim.AdamW,
    inputs: Tensor,
    targets: Tensor,
    *,
    grad_clip_norm: float,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    prediction = model(inputs)
    if prediction.shape != targets.shape:
        raise RuntimeError(f"invalid training endpoint shape for {model_name}")
    loss = torch.nn.functional.mse_loss(prediction, targets)
    if not torch.isfinite(loss):
        raise RuntimeError(f"nonfinite training loss for {model_name}")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
    optimizer.step()
    _callback(model, "post_optimizer_step")


def _exact_training_step(
    model: nn.Module,
    optimizer: torch.optim.AdamW,
    inputs: Tensor,
    targets: Tensor,
    runtime: _PackedAlphabetExactSplitRuntime | None,
    *,
    batch_size: int,
    grad_clip_norm: float,
) -> _PackedAlphabetExactSplitRuntime:
    if inputs.shape[0] != batch_size:
        message = "variable-step exact split requires complete batches"
        raise RuntimeError(message)
    if runtime is None:
        runtime = cast("_AlphabetEndpoint", model).prepare_external_exact_split_runtime(
            optimizer,
            inputs,
            targets,
            objective="forecasting",
            grad_clip_norm=grad_clip_norm,
        )
    runtime.step(inputs, targets)
    return runtime


def _fit_validation_only(
    model: nn.Module,
    model_name: ModelName,
    task: PACRegressionTask,
    config: PACExperimentConfig,
    *,
    device: str,
    seed: int,
) -> dict[str, float | int]:
    if task.test_inputs.numel() != 0 or task.test_targets.numel() != 0:
        raise RuntimeError("selection task must physically omit TEST tensors")
    model.to(device)
    use_exact_split = device.startswith("cuda") and model_name == "alphabet"
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        fused=use_exact_split,
        capturable=use_exact_split,
    )
    train_inputs = task.train_inputs.to(device)
    train_targets = task.train_targets.to(device)
    generator = torch.Generator(device=train_inputs.device).manual_seed(seed)
    best_loss = math.inf
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    runtime: _PackedAlphabetExactSplitRuntime | None = None
    exact_split_steps = 0
    started = perf_counter()
    try:
        for epoch in range(1, config.epochs + 1):
            model.train()
            order = torch.randperm(
                train_inputs.shape[0], generator=generator, device=train_inputs.device
            )
            for indices in order.split(config.batch_size):
                batch_inputs = train_inputs[indices]
                batch_targets = train_targets[indices]
                if use_exact_split:
                    runtime = _exact_training_step(
                        model,
                        optimizer,
                        batch_inputs,
                        batch_targets,
                        runtime,
                        batch_size=config.batch_size,
                        grad_clip_norm=config.grad_clip_norm,
                    )
                    exact_split_steps += 1
                else:
                    _eager_training_step(
                        model,
                        model_name,
                        optimizer,
                        batch_inputs,
                        batch_targets,
                        grad_clip_norm=config.grad_clip_norm,
                    )
            if runtime is not None:
                runtime.close()
            validation_loss = _loss(
                model,
                model_name,
                task.validation_inputs,
                task.validation_targets,
                device=device,
                batch_size=config.batch_size,
            )
            if not math.isfinite(validation_loss):
                raise RuntimeError(f"nonfinite validation loss for {model_name}")
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
    finally:
        if runtime is not None:
            runtime.destroy()
        model.__dict__["variable_step_exact_split_steps"] = exact_split_steps
    if best_state is None:
        raise RuntimeError("training produced no finite validation checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    _callback(model, "finalize_constraints")
    train_loss = _loss(
        model,
        model_name,
        task.train_inputs,
        task.train_targets,
        device=device,
        batch_size=config.batch_size,
    )
    return {
        "train_loss": train_loss,
        "validation_loss": best_loss,
        "best_epoch": best_epoch,
        "elapsed_seconds": perf_counter() - started,
    }


def run_selection_job(
    root: Path, job: SelectionJob, *, device: str, smoke: bool
) -> dict[str, object]:
    config = _config(root, job.seed, job.learning_rate, smoke=smoke)
    model = _build_model(job.model, config, job.seed)
    task = _selection_task(config, job.seed, job.regime)
    outcome = _fit_validation_only(model, job.model, task, config, device=device, seed=job.seed)
    return {
        "schema": "pac.variable_step_causal.selection.v1",
        "stage": "selection",
        "job_key": job.key,
        **asdict(job),
        "status": "done",
        "smoke": smoke,
        "params_trainable": count_parameters(model),
        "target_params": TARGET_PARAMS,
        "relative_parameter_error": abs(count_parameters(model) - TARGET_PARAMS) / TARGET_PARAMS,
        "test_or_ood_constructed": False,
        **outcome,
    }


def _metric_row(
    model: nn.Module,
    model_name: ModelName,
    inputs: Tensor,
    targets: Tensor,
    *,
    variant: Variant,
    seed: int,
    device: str,
    batch_size: int,
) -> tuple[float, float]:
    predicted = _predict(
        model,
        model_name,
        inputs,
        variant=variant,
        seed=seed,
        device=device,
        batch_size=batch_size,
    )
    mse = float(torch.nn.functional.mse_loss(predicted, targets.cpu()).item())
    return mse, nrmse(mse, targets)


def _factorial_cases(config: PACExperimentConfig, seed: int) -> list[tuple[str, Tensor, Tensor]]:
    parameters = _forcing_parameters(config.test_count, seed + 401)
    rows: list[tuple[str, Tensor, Tensor]] = []
    for timing, irregularity in (("regular", 0.0), ("irregular", FACTORIAL_IRREGULARITY)):
        for observation, missing_rate in (("observed", 0.0), ("missing", FACTORIAL_MISSING_RATE)):
            inputs, targets = _build_case(
                parameters,
                irregularity=irregularity,
                missing_rate=missing_rate,
                perturbation_seed=seed + 50_001,
            )
            rows.append((f"{timing}__{observation}", inputs, targets[:, -1]))
    return rows


def _final_evaluations(
    model: nn.Module,
    job: FinalJob,
    config: PACExperimentConfig,
    *,
    device: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    conditions = matched_zoh_conditions(config, job.seed)
    held_levels = {f"dt_{delta:g}" for delta in HELD_OUT_DELTAS}
    for index, condition in enumerate(conditions):
        paired_id_mse, paired_id_nrmse = _metric_row(
            model,
            job.model,
            condition.id_inputs,
            condition.id_targets[:, -1],
            variant="correct_dt_mask",
            seed=job.seed + 20_000 + index,
            device=device,
            batch_size=config.batch_size,
        )
        if condition.family == "sampling_rate" and condition.level in held_levels:
            mse, active_nrmse = _metric_row(
                model,
                job.model,
                condition.ood_inputs,
                condition.ood_targets[:, -1],
                variant="correct_dt_mask",
                seed=job.seed + 30_000 + index,
                device=device,
                batch_size=config.batch_size,
            )
            rows.append(
                {
                    "suite": "held_out_dt",
                    "family": condition.family,
                    "level": condition.level,
                    "variant": "correct_dt_mask",
                    "mse": mse,
                    "nrmse": active_nrmse,
                    "paired_id_mse": paired_id_mse,
                    "paired_id_nrmse": paired_id_nrmse,
                    "delta_nrmse": active_nrmse - paired_id_nrmse,
                }
            )
        for variant in VARIANTS:
            mse, active_nrmse = _metric_row(
                model,
                job.model,
                condition.ood_inputs,
                condition.ood_targets[:, -1],
                variant=variant,
                seed=job.seed + 40_000 + index,
                device=device,
                batch_size=config.batch_size,
            )
            rows.append(
                {
                    "suite": "profiles19",
                    "family": condition.family,
                    "level": condition.level,
                    "variant": variant,
                    "mse": mse,
                    "nrmse": active_nrmse,
                    "paired_id_mse": paired_id_mse,
                    "paired_id_nrmse": paired_id_nrmse,
                    "delta_nrmse": active_nrmse - paired_id_nrmse,
                }
            )
    factorial_values: dict[str, float] = {}
    factorial_rows: list[dict[str, object]] = []
    for index, (level, inputs, targets) in enumerate(_factorial_cases(config, job.seed)):
        mse, active_nrmse = _metric_row(
            model,
            job.model,
            inputs,
            targets,
            variant="correct_dt_mask",
            seed=job.seed + 60_000 + index,
            device=device,
            batch_size=config.batch_size,
        )
        factorial_values[level] = active_nrmse
        factorial_rows.append(
            {
                "suite": "factorial2x2",
                "family": "timing_x_missingness",
                "level": level,
                "variant": "correct_dt_mask",
                "mse": mse,
                "nrmse": active_nrmse,
            }
        )
    reference = factorial_values["regular__observed"]
    for row in factorial_rows:
        row["paired_id_nrmse"] = reference
        row["delta_nrmse"] = cast("float", row["nrmse"]) - reference
    rows.extend(factorial_rows)
    if len(rows) != 4 + 19 * len(VARIANTS) + 4:
        raise RuntimeError(f"final evaluation matrix has unexpected size: {len(rows)}")
    return rows


def run_final_job(root: Path, job: FinalJob, *, device: str, smoke: bool) -> dict[str, object]:
    selection_path = root / "selection.json"
    if not selection_path.is_file():
        raise RuntimeError("frozen selection artifact is required")
    config = _config(root, job.seed, job.learning_rate, smoke=smoke)
    model = _build_model(job.model, config, job.seed)
    task = _final_task(config, job.seed, job.regime)
    # Fit with validation checkpointing.  The local fitter's TEST guard is
    # satisfied by a view that physically omits the final task's ID tensors.
    selection_view = replace(
        task,
        test_inputs=torch.empty(0, config.sequence_length, 4),
        test_targets=torch.empty(0, 2),
    )
    outcome = _fit_validation_only(
        model, job.model, selection_view, config, device=device, seed=job.seed
    )
    id_mse = _loss(
        model,
        job.model,
        task.test_inputs,
        task.test_targets,
        device=device,
        batch_size=config.batch_size,
    )
    evaluations = _final_evaluations(model, job, config, device=device)
    return {
        "schema": "pac.variable_step_causal.final.v1",
        "stage": "final",
        "job_key": job.key,
        **asdict(job),
        "status": "done",
        "smoke": smoke,
        "params_trainable": count_parameters(model),
        "target_params": TARGET_PARAMS,
        "relative_parameter_error": abs(count_parameters(model) - TARGET_PARAMS) / TARGET_PARAMS,
        "selection_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
        "id_test_mse": id_mse,
        "id_test_nrmse": nrmse(id_mse, task.test_targets),
        **outcome,
        "evaluations": evaluations,
    }


def _contract(shards: int) -> dict[str, object]:
    repository = Path(__file__).resolve().parents[2]
    return {
        "schema": "pac.variable_step_causal.contract.v1",
        "alphabet_model_class": "lnet.alphabet.Alphabet",
        "alphabet_internal_spec": "radial_log_r_affine",
        "models": list(MODELS),
        "regimes": list(REGIMES),
        "learning_rates": list(LEARNING_RATES),
        "selection_seeds": list(SELECTION_SEEDS),
        "final_seeds": list(FINAL_SEEDS),
        "selection_jobs": len(selection_jobs()),
        "maximum_final_jobs": len(REGIMES) * len(MODELS) * len(FINAL_SEEDS),
        "target_params": TARGET_PARAMS,
        "parameter_tolerance": PARAMETER_TOLERANCE,
        "widths": WIDTHS,
        "expected_params": EXPECTED_PARAMS,
        "capacity_policy": "nearest real trainable width; no dummy, adapter, or inert parameters",
        "training": {
            "epochs": 100,
            "batch_size": 64,
            "weight_decay": 1.0e-4,
            "grad_clip_norm": 1.0,
            "checkpoint": "minimum validation MSE",
            "unit": "dt=1, fully observed, horizon=60",
            "mixed_dt": f"length-60 physical-time steps with irregularity={MIXED_IRREGULARITY}, fully observed, horizon=60",
        },
        "held_out_deltas": list(HELD_OUT_DELTAS),
        "factorial": {
            "timing": ["regular", "irregular"],
            "observation": ["observed", "missing"],
            "irregularity": FACTORIAL_IRREGULARITY,
            "missing_rate": FACTORIAL_MISSING_RATE,
        },
        "profiles": 19,
        "metadata_variants": list(VARIANTS),
        "selection_data_access": "TRAIN and validation only; TEST tensors empty; OOD constructors not called",
        "shards": shards,
        "source_sha256": {
            relative: hashlib.sha256((repository / relative).read_bytes()).hexdigest()
            for relative in SOURCE_FILES
        },
        "restart_safe": True,
        "locked_before_execution": True,
    }


def enqueue_selection(root: Path, *, shards: int = 14) -> dict[str, object]:
    if shards < 1:
        raise ValueError("shards must be positive")
    (root / "selection" / "completed").mkdir(parents=True, exist_ok=True)
    (root / "selection" / "failed").mkdir(parents=True, exist_ok=True)
    _write_locked_json(root / "contract.json", _contract(shards))
    _write_manifests(root / "selection" / "manifests", selection_jobs(), shards)
    return {"selection_jobs": len(selection_jobs()), "shards": shards}


def freeze_selection(root: Path, *, shards: int = 14) -> dict[str, object]:
    rows = _require_complete(root, "selection", {job.key for job in selection_jobs()})
    selected: dict[str, dict[str, object]] = {}
    final_jobs: list[FinalJob] = []
    for regime in REGIMES:
        for model in MODELS:
            candidates: list[tuple[float, float]] = []
            for learning_rate in LEARNING_RATES:
                active = [
                    row
                    for row in rows
                    if row["regime"] == regime
                    and row["model"] == model
                    and math.isclose(
                        float(cast("float | int", row["learning_rate"])), learning_rate
                    )
                ]
                if len(active) != len(SELECTION_SEEDS):
                    raise RuntimeError(
                        f"incomplete selection cell: {regime}/{model}/{learning_rate}"
                    )
                candidates.append(
                    (
                        learning_rate,
                        mean(float(cast("float | int", row["validation_loss"])) for row in active),
                    )
                )
            learning_rate, validation_loss = min(
                candidates, key=lambda item: (item[1], abs(item[0] - 3.0e-3), item[0])
            )
            key = f"{regime}:{model}"
            selected[key] = {
                "learning_rate": learning_rate,
                "mean_validation_loss": validation_loss,
                "selection_seeds": list(SELECTION_SEEDS),
                "test_or_ood_used": False,
            }
            final_jobs.extend(FinalJob(regime, model, learning_rate, seed) for seed in FINAL_SEEDS)
    payload: dict[str, object] = {
        "schema": "pac.variable_step_causal.selection_freeze.v1",
        "selected": selected,
        "selected_cells": len(selected),
        "source_rows": len(rows),
        "final_jobs": len(final_jobs),
        "test_or_ood_used": False,
    }
    _write_locked_json(root / "selection.json", payload)
    (root / "final" / "completed").mkdir(parents=True, exist_ok=True)
    (root / "final" / "failed").mkdir(parents=True, exist_ok=True)
    _write_manifests(root / "final" / "manifests", tuple(final_jobs), shards)
    return payload


def _write_manifests(
    directory: Path,
    jobs: tuple[SelectionJob, ...] | tuple[FinalJob, ...],
    shards: int,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for shard in range(shards):
        payload = "".join(
            json.dumps(asdict(job), sort_keys=True) + "\n"
            for index, job in enumerate(jobs)
            if index % shards == shard
        )
        _write_locked_text(directory / f"shard-{shard:02d}.jsonl", payload)


def worker(
    root: Path,
    stage: Literal["selection", "final"],
    shard: int,
    *,
    device: str,
    smoke: bool,
    max_jobs: int | None = None,
) -> int:
    manifest = root / stage / "manifests" / f"shard-{shard:02d}.jsonl"
    if not manifest.is_file():
        raise RuntimeError(f"missing manifest: {manifest}")
    completed_now = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        if stage == "selection":
            job: SelectionJob | FinalJob = SelectionJob(
                cast("Regime", payload["regime"]),
                cast("ModelName", payload["model"]),
                float(payload["learning_rate"]),
                int(payload["seed"]),
            )
        else:
            job = FinalJob(
                cast("Regime", payload["regime"]),
                cast("ModelName", payload["model"]),
                float(payload["learning_rate"]),
                int(payload["seed"]),
            )
        destination = root / stage / "completed" / f"{job.key}.json"
        if destination.exists():
            _validate_existing(destination, job.key, smoke=smoke)
            continue
        try:
            if stage == "selection":
                result = run_selection_job(
                    root, cast("SelectionJob", job), device=device, smoke=smoke
                )
            else:
                result = run_final_job(root, cast("FinalJob", job), device=device, smoke=smoke)
        except Exception as error:
            failure = {
                "schema": "pac.variable_step_causal.failure.v1",
                "stage": stage,
                "job_key": job.key,
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            }
            _atomic_json(root / stage / "failed" / f"{job.key}.json", failure)
            raise
        _atomic_json(destination, result)
        completed_now += 1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if max_jobs is not None and completed_now >= max_jobs:
            break
    return completed_now


def status(root: Path) -> dict[str, object]:
    payload: dict[str, object] = {"schema": "pac.variable_step_causal.status.v1"}
    for stage in ("selection", "final"):
        manifests = sorted((root / stage / "manifests").glob("*.jsonl"))
        expected = {
            (SelectionJob if stage == "selection" else FinalJob)(
                cast("Regime", row["regime"]),
                cast("ModelName", row["model"]),
                float(row["learning_rate"]),
                int(row["seed"]),
            ).key
            for path in manifests
            for row in map(json.loads, path.read_text(encoding="utf-8").splitlines())
        }
        completed = {path.stem for path in (root / stage / "completed").glob("*.json")}
        failed = {path.stem for path in (root / stage / "failed").glob("*.json")}
        active_failed = (expected & failed) - completed
        payload[stage] = {
            "expected": len(expected),
            "completed": len(expected & completed),
            "failed": len(active_failed),
            "remaining": len(expected - completed),
            "done": bool(expected) and expected <= completed and not active_failed,
        }
    return payload


def report(root: Path) -> dict[str, object]:
    selection_payload = json.loads((root / "selection.json").read_text(encoding="utf-8"))
    final_jobs = tuple(
        FinalJob(
            regime,
            model,
            float(
                cast(
                    "dict[str, dict[str, float | int]]",
                    selection_payload["selected"],
                )[f"{regime}:{model}"]["learning_rate"]
            ),
            seed,
        )
        for regime in REGIMES
        for model in MODELS
        for seed in FINAL_SEEDS
    )
    rows = _require_complete(root, "final", {job.key for job in final_jobs})
    for row in rows:
        model = cast("ModelName", row["model"])
        if int(cast("int", row["params_trainable"])) != EXPECTED_PARAMS[model]:
            raise RuntimeError(f"final parameter lock mismatch: {row['job_key']}")
        evaluations = cast("list[dict[str, object]]", row.get("evaluations"))
        if len(evaluations) != 103:
            raise RuntimeError(f"incomplete evaluation payload: {row['job_key']}")
        keys = {
            (
                str(item["suite"]),
                str(item["family"]),
                str(item["level"]),
                str(item["variant"]),
            )
            for item in evaluations
        }
        if len(keys) != 103:
            raise RuntimeError(f"duplicate evaluation cell: {row['job_key']}")
        if sum(key[0] == "held_out_dt" for key in keys) != 4:
            raise RuntimeError(f"held-out-dt coverage mismatch: {row['job_key']}")
        if sum(key[0] == "profiles19" for key in keys) != 95:
            raise RuntimeError(f"profile coverage mismatch: {row['job_key']}")
        if sum(key[0] == "factorial2x2" for key in keys) != 4:
            raise RuntimeError(f"factorial coverage mismatch: {row['job_key']}")
        for item in evaluations:
            for field in ("mse", "nrmse", "paired_id_nrmse", "delta_nrmse"):
                if not math.isfinite(float(cast("float | int", item[field]))):
                    raise RuntimeError(f"nonfinite {field} in final evaluation: {row['job_key']}")
    long_rows = [
        {
            "job_key": row["job_key"],
            "regime": row["regime"],
            "model": row["model"],
            "seed": row["seed"],
            **evaluation,
        }
        for row in rows
        for evaluation in cast("list[dict[str, object]]", row["evaluations"])
    ]
    if len(long_rows) != len(final_jobs) * 103:
        raise RuntimeError("final report matrix is incomplete")
    report_root = root / "reports"
    report_root.mkdir(exist_ok=True)
    with (report_root / "evaluations_long.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(long_rows[0]))
        writer.writeheader()
        writer.writerows(long_rows)
    grouped: dict[tuple[str, str, str, str, str], list[float]] = {}
    for row in long_rows:
        key = (
            str(row["regime"]),
            str(row["model"]),
            str(row["suite"]),
            str(row["family"]),
            f"{row['level']}:{row['variant']}",
        )
        grouped.setdefault(key, []).append(float(cast("float | int", row["nrmse"])))
    summary_rows: list[dict[str, object]] = []
    for key, values in sorted(grouped.items()):
        if len(values) != len(FINAL_SEEDS):
            raise RuntimeError(f"incomplete report cell: {key}")
        summary_rows.append(
            {
                "regime": key[0],
                "model": key[1],
                "suite": key[2],
                "family": key[3],
                "condition_variant": key[4],
                "mean_nrmse": mean(values),
                "sample_sd_nrmse": stdev(values),
                "seeds": len(values),
            }
        )
    aggregate_rows = _aggregate_final_rows(rows)
    summary: dict[str, object] = {
        "schema": "pac.variable_step_causal.report.v1",
        "complete": True,
        "selection_rows": len(selection_jobs()),
        "final_rows": len(rows),
        "evaluation_rows": len(long_rows),
        "summary": summary_rows,
        "aggregates": aggregate_rows,
    }
    _atomic_json(report_root / "summary.json", summary)
    _write_locked_text(root / "COMPLETE", "complete\n")
    return summary


def _aggregate_final_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Seed-level equal-family and factorial summaries, then five-seed moments."""
    seed_rows: list[dict[str, object]] = []
    for row in rows:
        evaluations = cast("list[dict[str, object]]", row["evaluations"])
        correct_profiles = [
            item
            for item in evaluations
            if item["suite"] == "profiles19" and item["variant"] == "correct_dt_mask"
        ]
        by_family: dict[str, list[float]] = {}
        for item in correct_profiles:
            by_family.setdefault(str(item["family"]), []).append(
                float(cast("float | int", item["delta_nrmse"]))
            )
        if len(by_family) != 7:
            raise RuntimeError(f"expected seven OOD families: {row['job_key']}")
        family_macro = mean(mean(values) for values in by_family.values())
        factorial = {
            str(item["level"]): float(cast("float | int", item["nrmse"]))
            for item in evaluations
            if item["suite"] == "factorial2x2"
        }
        if len(factorial) != 4:
            raise RuntimeError(f"incomplete factorial aggregate: {row['job_key']}")
        missing_effect = 0.5 * (
            factorial["regular__missing"]
            - factorial["regular__observed"]
            + factorial["irregular__missing"]
            - factorial["irregular__observed"]
        )
        irregular_effect = 0.5 * (
            factorial["irregular__observed"]
            - factorial["regular__observed"]
            + factorial["irregular__missing"]
            - factorial["regular__missing"]
        )
        interaction = (
            factorial["irregular__missing"]
            - factorial["irregular__observed"]
            - factorial["regular__missing"]
            + factorial["regular__observed"]
        )
        seed_rows.append(
            {
                "regime": row["regime"],
                "model": row["model"],
                "seed": row["seed"],
                "equal_family_macro_delta_nrmse": family_macro,
                "factorial_missing_effect": missing_effect,
                "factorial_irregular_effect": irregular_effect,
                "factorial_interaction": interaction,
            }
        )
    result: list[dict[str, object]] = []
    metric_fields = (
        "equal_family_macro_delta_nrmse",
        "factorial_missing_effect",
        "factorial_irregular_effect",
        "factorial_interaction",
    )
    for regime in REGIMES:
        for model in MODELS:
            active = [
                item for item in seed_rows if item["regime"] == regime and item["model"] == model
            ]
            if len(active) != len(FINAL_SEEDS):
                raise RuntimeError(f"incomplete seed aggregate: {regime}/{model}")
            payload: dict[str, object] = {
                "regime": regime,
                "model": model,
                "seeds": len(active),
            }
            for field in metric_fields:
                values = [float(cast("float | int", item[field])) for item in active]
                payload[f"mean_{field}"] = mean(values)
                payload[f"sample_sd_{field}"] = stdev(values)
            result.append(payload)
    return result


def _require_complete(root: Path, stage: str, expected: set[str]) -> list[dict[str, object]]:
    completed_stems = {path.stem for path in (root / stage / "completed").glob("*.json")}
    failed = {path.stem for path in (root / stage / "failed").glob("*.json")} - completed_stems
    if failed:
        raise RuntimeError(f"{stage} has {len(failed)} unresolved failure records")
    by_key: dict[str, dict[str, object]] = {}
    for path in sorted((root / stage / "completed").glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        key = str(row.get("job_key"))
        if key in by_key:
            raise RuntimeError(f"duplicate {stage} key: {key}")
        if row.get("status") != "done" or row.get("smoke"):
            raise RuntimeError(f"invalid paper-facing {stage} row: {path}")
        if not math.isfinite(float(row["validation_loss"])):
            raise RuntimeError(f"nonfinite {stage} validation loss: {key}")
        if stage == "selection":
            if row.get("test_or_ood_constructed") is not False:
                raise RuntimeError(f"selection data-access flag failed: {key}")
            model = cast("ModelName", row["model"])
            if int(cast("int", row["params_trainable"])) != EXPECTED_PARAMS[model]:
                raise RuntimeError(f"selection parameter lock mismatch: {key}")
        by_key[key] = row
    missing = expected - set(by_key)
    extra = set(by_key) - expected
    if missing or extra:
        raise RuntimeError(f"incomplete {stage}: missing={len(missing)}, extra={len(extra)}")
    return [by_key[key] for key in sorted(expected)]


def _validate_existing(path: Path, job_key: str, *, smoke: bool) -> None:
    row = json.loads(path.read_text(encoding="utf-8"))
    if (
        row.get("job_key") != job_key
        or row.get("status") != "done"
        or bool(row.get("smoke")) != smoke
    ):
        raise RuntimeError(f"existing result does not satisfy immutable job: {path}")


def _write_locked_json(path: Path, payload: object) -> None:
    _write_locked_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_locked_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise RuntimeError(f"immutable artifact differs: {path}")
        return
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=(
            "enqueue-selection",
            "selection-worker",
            "freeze",
            "final-worker",
            "status",
            "report",
        ),
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--shards", type=int, default=14)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-jobs", type=int)
    args = parser.parse_args()
    if args.stage == "enqueue-selection":
        payload = enqueue_selection(args.root, shards=args.shards)
    elif args.stage == "selection-worker":
        payload = {
            "completed_now": worker(
                args.root,
                "selection",
                args.shard,
                device=args.device,
                smoke=args.smoke,
                max_jobs=args.max_jobs,
            )
        }
    elif args.stage == "freeze":
        payload = freeze_selection(args.root, shards=args.shards)
    elif args.stage == "final-worker":
        payload = {
            "completed_now": worker(
                args.root,
                "final",
                args.shard,
                device=args.device,
                smoke=args.smoke,
                max_jobs=args.max_jobs,
            )
        }
    elif args.stage == "status":
        payload = status(args.root)
    else:
        payload = report(args.root)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
