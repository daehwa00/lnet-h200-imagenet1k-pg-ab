# ruff: noqa: EM101, EM102, T201, TRY003
"""Sealed, parameter-matched synthetic OOD campaign for compact ALPHABET.

This is a new evidence stream; it deliberately does not mutate or reinterpret
the historical EFP16 synthetic-OOD artifacts.  Every family is trained with
the same fixed recipe on the regular exact-ZOH teacher.  Validation chooses
the checkpoint, while ID TEST and the 19 OOD profiles are unavailable to the
fitter and are constructed only after fitting has finished.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from pathlib import Path
from statistics import mean, stdev
from typing import TYPE_CHECKING, Final, Literal, TypedDict, cast

import torch
from torch import Tensor, nn

from .pac_alphabet_synthetic_ood import (
    SEEDS as _SEEDS,
)
from .pac_alphabet_synthetic_ood import (
    VARIANTS as _VARIANTS,
)
from .pac_alphabet_synthetic_ood import (
    _permute_delta,  # pyright: ignore[reportPrivateUsage]
)
from .pac_alphabet_synthetic_ood_baselines import (
    _average_ranks,  # pyright: ignore[reportPrivateUsage]
)
from .pac_efp_writer_reader import CompactEFPHOnlyTerminalPAC
from .pac_external_benchmarks import (
    _build_continuous_model,  # pyright: ignore[reportPrivateUsage]
)
from .pac_matched_zoh_ood import (
    _regular_split,  # pyright: ignore[reportPrivateUsage]
    matched_zoh_conditions,
    matched_zoh_training_task,
)
from .pac_metrics import count_parameters, nrmse
from .pac_types import PACDevice, PACExperimentConfig, PACRegressionTask
from .pac_variable_step_causal_campaign import (
    _fit_validation_only,  # pyright: ignore[reportPrivateUsage]
)

if TYPE_CHECKING:
    from .pac_external_benchmarks import ExternalModelFamily

ModelName = Literal[
    "compact_h_only",
    "cnn1d",
    "tcn",
    "mamba",
    "gru",
    "lstm",
    "transformer",
]
BaselineName = Literal["cnn1d", "tcn", "mamba", "gru", "lstm", "transformer"]


class ParameterLock(TypedDict):
    target_params: int
    widths: dict[str, int]
    expected_params: dict[str, int]
    relative_parameter_errors: dict[str, float]


SEEDS: Final = tuple(_SEEDS)
VARIANTS: Final = tuple(_VARIANTS)

MODELS: Final[tuple[ModelName, ...]] = (
    "compact_h_only",
    "cnn1d",
    "tcn",
    "mamba",
    "gru",
    "lstm",
    "transformer",
)
BASELINES: Final[tuple[BaselineName, ...]] = (
    "cnn1d",
    "tcn",
    "mamba",
    "gru",
    "lstm",
    "transformer",
)
MODEL_DIM: Final = 32
MODES: Final = 16
MAX_BASELINE_WIDTH: Final = 128
MAX_RELATIVE_PARAMETER_ERROR: Final = 0.10
CONDITIONS_PER_JOB: Final = 19
FROZEN_PROFILES: Final[tuple[tuple[str, str], ...]] = (
    ("sampling_rate", "dt_0.25"),
    ("sampling_rate", "dt_0.5"),
    ("sampling_rate", "dt_0.75"),
    ("sampling_rate", "dt_1.25"),
    ("sampling_rate", "dt_1.5"),
    ("sampling_rate", "dt_2"),
    ("sampling_rate", "dt_3"),
    ("irregular_timestamps_missingness", "moderate"),
    ("irregular_timestamps_missingness", "hard"),
    ("physical_horizon", "120"),
    ("physical_horizon", "240"),
    ("delay", "4"),
    ("delay", "8"),
    ("additive_noise", "0.05"),
    ("additive_noise", "0.1"),
    ("damping", "1.2"),
    ("damping", "1.6"),
    ("frequency", "pi_over_8"),
    ("frequency", "pi_over_2"),
)
DEFAULT_ROOT: Final = Path(".omx/results/pac-compact-h-only-synthetic-ood-20260719")
DEFAULT_UCR_SELECTION: Final = Path(
    ".omx/results/pac-efp-compact-equal-search-20260719/stage2/selection.json"
)
DEFAULT_EXTERNAL_SELECTION: Final = Path(
    ".omx/results/pac-efp-compact-external-equal-search-20260719/stage2/selection.json"
)
DEFAULT_COMPARISON: Final = (
    Path(".omx/results/pac-efp-compact-external-equal-search-20260719")
    / "reports/combined_30_task_comparison.json"
)


@dataclass(frozen=True, slots=True)
class SyntheticOODJob:
    model: ModelName
    seed: int

    @property
    def key(self) -> str:
        return f"compact_h_only_synthetic_ood__{self.model}__seed{self.seed}"


def jobs() -> tuple[SyntheticOODJob, ...]:
    return tuple(SyntheticOODJob(model, seed) for model in MODELS for seed in SEEDS)


class _CompactEndpoint(nn.Module):
    """Expose CompactEFPHOnlyTerminalPAC through the synthetic metadata API."""

    def __init__(self, config: PACExperimentConfig) -> None:
        super().__init__()
        active = replace(
            config,
            raw_input_dim=2,
            model_dim=MODEL_DIM,
            modes=MODES,
        )
        self.model = CompactEFPHOnlyTerminalPAC(
            active,
            active.output_dim,
            objective="regression",
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.forward_variant(inputs, "correct_dt_mask", seed=0)

    def forward_variant(self, inputs: Tensor, variant: str, *, seed: int) -> Tensor:
        if inputs.ndim != 3 or inputs.shape[-1] != 4:
            raise ValueError("compact synthetic OOD inputs must have [value2, dt, mask]")
        if variant not in VARIANTS:
            raise ValueError(f"unknown metadata variant: {variant}")
        delta = inputs[..., 2:3]
        mask: Tensor | None = inputs[..., 3:4]
        if variant in {"unit_dt_mask", "unit_dt_no_mask"}:
            delta = torch.ones_like(delta)
        elif variant == "shuffled_dt_mask":
            delta = _permute_delta(delta, seed)
        if variant in {"correct_dt_no_mask", "unit_dt_no_mask"}:
            mask = None
        return self.model(
            inputs[..., :2],
            time_delta=delta,
            observation_mask=mask,
        )

    def post_optimizer_step(self) -> None:
        self.model.post_optimizer_step()

    def finalize_constraints(self) -> None:
        self.model.finalize_constraints()


def _config(root: Path, seed: int, *, smoke: bool) -> PACExperimentConfig:
    return PACExperimentConfig(
        sample_count=64 if smoke else 2048,
        validation_count=32 if smoke else 512,
        test_count=32 if smoke else 512,
        sequence_length=60,
        raw_input_dim=4,
        output_dim=2,
        model_dim=MODEL_DIM,
        modes=MODES,
        epochs=1 if smoke else 100,
        batch_size=16 if smoke else 64,
        learning_rate=3.0e-3,
        weight_decay=1.0e-4,
        grad_clip_norm=1.0,
        seeds=(seed,),
        device=cast("PACDevice", "cuda"),
        output_dir=root,
        compile_mode="none",
        precision="fp32",
    )


def _sealed_training_task(config: PACExperimentConfig, seed: int) -> PACRegressionTask:
    """Build only TRAIN/validation; ID TEST and OOD do not exist yet."""
    train_inputs, train_targets = _regular_split(config.sample_count, seed + 101)
    validation_inputs, validation_targets = _regular_split(
        config.validation_count,
        seed + 211,
    )
    return PACRegressionTask(
        label="compact_h_only_exact_zoh_train_validation_only",
        train_inputs=train_inputs,
        train_targets=train_targets[:, -1],
        validation_inputs=validation_inputs,
        validation_targets=validation_targets[:, -1],
        test_inputs=torch.empty(0, config.sequence_length, 4),
        test_targets=torch.empty(0, 2),
        true_delay=0,
        true_frequency=math.pi / 4,
        true_frequencies=(math.pi / 4,),
        true_dampings=(0.8,),
        mechanism_expectation="positive",
    )


def _endpoint_test_task(config: PACExperimentConfig, seed: int) -> PACRegressionTask:
    task = matched_zoh_training_task(config, seed)
    return replace(
        task,
        label="compact_h_only_exact_zoh_endpoint_test",
        train_targets=task.train_targets[:, -1],
        validation_targets=task.validation_targets[:, -1],
        test_targets=task.test_targets[:, -1],
    )


@lru_cache(maxsize=1)
def parameter_lock() -> ParameterLock:
    """Recompute the target and nearest realizable baseline widths."""
    config = _config(Path(), SEEDS[0], smoke=True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        target_model = _CompactEndpoint(config)
    target = count_parameters(target_model)
    widths: dict[str, int] = {}
    expected: dict[str, int] = {"compact_h_only": target}
    relative_errors: dict[str, float] = {"compact_h_only": 0.0}
    for model in BASELINES:
        candidates: list[tuple[int, int, int]] = []
        for width in range(1, MAX_BASELINE_WIDTH + 1):
            try:
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(0)
                    built = _build_continuous_model(
                        cast("ExternalModelFamily", model),
                        width,
                        4,
                        2,
                        config,
                        "",
                        objective="regression",
                    )
            except (AssertionError, RuntimeError, ValueError):
                continue
            actual = count_parameters(built)
            candidates.append((abs(actual - target), width, actual))
        if not candidates:
            raise RuntimeError(f"no realizable width found for {model}")
        _, width, actual = min(candidates)
        error = abs(actual - target) / target
        if error > MAX_RELATIVE_PARAMETER_ERROR:
            raise RuntimeError(
                f"nearest width for {model} exceeds the parameter tolerance: {error:.4f}"
            )
        widths[model] = width
        expected[model] = actual
        relative_errors[model] = error
    return {
        "target_params": target,
        "widths": widths,
        "expected_params": expected,
        "relative_parameter_errors": relative_errors,
    }


def _build_model(model: ModelName, config: PACExperimentConfig, seed: int) -> nn.Module:
    lock = parameter_lock()
    expected = lock["expected_params"]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if model == "compact_h_only":
            built: nn.Module = _CompactEndpoint(config)
        else:
            widths = lock["widths"]
            built = _build_continuous_model(
                cast("ExternalModelFamily", model),
                widths[model],
                4,
                2,
                config,
                "",
                objective="regression",
            )
    actual = count_parameters(built)
    if actual != expected[model]:
        raise RuntimeError(f"parameter lock changed for {model}: {actual} != {expected[model]}")
    return built


def _baseline_metadata_view(inputs: Tensor, variant: str, *, seed: int) -> Tensor:
    result = inputs.clone()
    if variant in {"unit_dt_mask", "unit_dt_no_mask"}:
        result[..., 2] = 1.0
    elif variant == "shuffled_dt_mask":
        result[..., 2:3] = _permute_delta(result[..., 2:3], seed)
    if variant in {"correct_dt_no_mask", "unit_dt_no_mask"}:
        result[..., 3] = 1.0
    return result


@torch.no_grad()
def _predict(
    model: nn.Module,
    model_name: ModelName,
    inputs: Tensor,
    *,
    variant: str,
    seed: int,
    device: str,
    batch_size: int,
) -> Tensor:
    model.eval()
    if model_name == "compact_h_only":
        batches = inputs.to(device).split(batch_size)
        predicted = [
            cast("_CompactEndpoint", model)
            .forward_variant(batch, variant, seed=seed)
            .detach()
            .cpu()
            for batch in batches
        ]
    else:
        viewed = _baseline_metadata_view(inputs, variant, seed=seed).to(device)
        predicted = [model(batch).detach().cpu() for batch in viewed.split(batch_size)]
    return torch.cat(predicted)


def _metric(
    model: nn.Module,
    model_name: ModelName,
    inputs: Tensor,
    targets: Tensor,
    *,
    variant: str,
    seed: int,
    device: str,
    batch_size: int,
) -> tuple[float, float]:
    prediction = _predict(
        model,
        model_name,
        inputs,
        variant=variant,
        seed=seed,
        device=device,
        batch_size=batch_size,
    )
    target = targets.detach().cpu()
    mse = float(torch.nn.functional.mse_loss(prediction, target).item())
    return mse, nrmse(mse, target)


def _frozen_architecture_source(
    ucr_selection: Path,
    external_selection: Path,
    comparison: Path,
) -> dict[str, object]:
    paths = (ucr_selection, external_selection, comparison)
    if not all(path.is_file() for path in paths):
        missing = [str(path) for path in paths if not path.is_file()]
        raise RuntimeError(f"frozen compact selection evidence is missing: {missing}")
    ucr = json.loads(ucr_selection.read_text(encoding="utf-8"))
    external = json.loads(external_selection.read_text(encoding="utf-8"))
    combined = json.loads(comparison.read_text(encoding="utf-8"))
    if (
        ucr.get("official_test_accessed") is not False
        or external.get("official_test_accessed") is not False
        or combined.get("official_test_accessed") is not False
        or combined.get("provisional_champion") != "compact_h_only"
        or int(cast("int | str", combined.get("tasks", 0))) != 30
    ):
        raise RuntimeError("frozen comparison does not seal compact_h_only on 30 validation tasks")
    selected = {
        **cast("dict[str, dict[str, object]]", ucr["selected"]),
        **cast("dict[str, dict[str, object]]", external["selected"]),
    }
    compact = {
        key: value
        for key, value in selected.items()
        if key.endswith(":compact_h_only")
    }
    if len(compact) != 30:
        raise RuntimeError("frozen comparison does not contain 30 compact_h_only cells")
    reference_cells = sorted(
        key
        for key, value in compact.items()
        if (
            int(cast("int | str", value["model_dim"])),
            int(cast("int | str", value["modes"])),
        )
        == (MODEL_DIM, MODES)
    )
    if not reference_cells:
        raise RuntimeError("fixed d32-m16 synthetic reference is absent from frozen selection")
    return {
        "architecture": "compact_h_only",
        "model_dim": MODEL_DIM,
        "modes": MODES,
        "configuration_policy": (
            "fixed d32-m16 synthetic reference, attested by the frozen 30-task selection; "
            "task-specific D/M and optimizer choices are not transferred to this synthetic task"
        ),
        "reference_cells": reference_cells,
        "selection_is_train_derived": True,
        "official_test_accessed": False,
        "source_files": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
        },
    }


def _contract(
    shards: int,
    *,
    ucr_selection: Path,
    external_selection: Path,
    comparison: Path,
) -> dict[str, object]:
    lock = parameter_lock()
    return {
        "schema": "pac.compact_h_only.synthetic_ood.contract.v1",
        "models": list(MODELS),
        "seeds": list(SEEDS),
        "jobs": len(jobs()),
        "shards": shards,
        "variants": list(VARIANTS),
        "conditions_per_job": CONDITIONS_PER_JOB,
        "frozen_profiles": [list(profile) for profile in FROZEN_PROFILES],
        "teacher": "matched exact-ZOH physical-time endpoint",
        "training_distribution": "regular dt=1, fully observed, physical horizon 60",
        "recipe": {
            "epochs": 100,
            "learning_rate": 3.0e-3,
            "weight_decay": 1.0e-4,
            "batch_size": 64,
            "grad_clip_norm": 1.0,
            "checkpoint": "minimum validation MSE",
            "precision": "fp32",
        },
        **lock,
        "max_relative_parameter_error": MAX_RELATIVE_PARAMETER_ERROR,
        "capacity_policy": "nearest realizable trainable width; no dummy or inert parameters",
        "frozen_architecture_source": _frozen_architecture_source(
            ucr_selection,
            external_selection,
            comparison,
        ),
        "data_access": {
            "fitting": "TRAIN and validation only; TEST tensors physically empty",
            "id_test": "constructed after the validation checkpoint is restored",
            "ood": "19 profiles constructed after fitting; never used for selection",
        },
        "same_weight_metadata_attribution": True,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "restart_safe": True,
        "locked_before_execution": True,
    }


def enqueue(
    root: Path = DEFAULT_ROOT,
    *,
    shards: int = 8,
    ucr_selection: Path = DEFAULT_UCR_SELECTION,
    external_selection: Path = DEFAULT_EXTERNAL_SELECTION,
    comparison: Path = DEFAULT_COMPARISON,
) -> dict[str, object]:
    if shards < 1:
        raise ValueError("shards must be positive")
    (root / "completed").mkdir(parents=True, exist_ok=True)
    (root / "failed").mkdir(parents=True, exist_ok=True)
    _write_locked_json(
        root / "contract.json",
        _contract(
            shards,
            ucr_selection=ucr_selection,
            external_selection=external_selection,
            comparison=comparison,
        ),
    )
    active_jobs = jobs()
    manifests = root / "manifests"
    manifests.mkdir(exist_ok=True)
    for shard in range(shards):
        payload = "".join(
            json.dumps(asdict(job), sort_keys=True) + "\n"
            for index, job in enumerate(active_jobs)
            if index % shards == shard
        )
        _write_locked_text(manifests / f"shard-{shard:02d}.jsonl", payload)
    return {"jobs": len(active_jobs), "shards": shards}


def run_job(
    root: Path,
    job: SyntheticOODJob,
    *,
    device: str,
    smoke: bool,
) -> dict[str, object]:
    contract_path = root / "contract.json"
    if not contract_path.is_file():
        raise RuntimeError("locked campaign contract is required")
    config = _config(root, job.seed, smoke=smoke)
    model = _build_model(job.model, config, job.seed)
    training_task = _sealed_training_task(config, job.seed)
    outcome = _fit_validation_only(
        model,
        job.model,  # pyright: ignore[reportArgumentType]
        training_task,
        config,
        device=device,
        seed=job.seed,
    )

    # This boundary is intentionally after fitting and checkpoint restoration.
    test_task = _endpoint_test_task(config, job.seed)
    id_mse, id_nrmse = _metric(
        model,
        job.model,
        test_task.test_inputs,
        test_task.test_targets,
        variant="correct_dt_mask",
        seed=job.seed,
        device=device,
        batch_size=config.batch_size,
    )
    conditions = matched_zoh_conditions(config, job.seed)
    active_profiles = {(condition.family, condition.level) for condition in conditions}
    if len(conditions) != CONDITIONS_PER_JOB or active_profiles != set(FROZEN_PROFILES):
        raise RuntimeError("exact-ZOH generator no longer matches the frozen 19 OOD profiles")
    rows: list[dict[str, object]] = []
    for condition_index, condition in enumerate(conditions):
        for variant in VARIANTS:
            mse, active_nrmse = _metric(
                model,
                job.model,
                condition.ood_inputs,
                condition.ood_targets[:, -1],
                variant=variant,
                seed=job.seed + 10_000 + condition_index,
                device=device,
                batch_size=config.batch_size,
            )
            rows.append(
                {
                    "family": condition.family,
                    "level": condition.level,
                    "variant": variant,
                    "mse": mse,
                    "nrmse": active_nrmse,
                }
            )
    lock = parameter_lock()
    target = lock["target_params"]
    actual = count_parameters(model)
    return {
        "schema": "pac.compact_h_only.synthetic_ood.result.v1",
        "job_key": job.key,
        "model": job.model,
        "seed": job.seed,
        "status": "done",
        "smoke": smoke,
        "params_trainable": actual,
        "target_params": target,
        "relative_parameter_error": abs(actual - target) / target,
        "contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
        "fit_test_tensors_empty": True,
        "ood_constructed_after_fit": True,
        **outcome,
        "id_test_mse": id_mse,
        "id_test_nrmse": id_nrmse,
        "conditions": rows,
    }


def worker(
    root: Path,
    shard: int,
    *,
    device: str,
    smoke: bool,
    max_jobs: int | None = None,
) -> int:
    manifest = root / "manifests" / f"shard-{shard:02d}.jsonl"
    if not manifest.is_file():
        raise RuntimeError(f"missing manifest: {manifest}")
    completed_now = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        job = SyntheticOODJob(cast("ModelName", payload["model"]), int(payload["seed"]))
        destination = root / "completed" / f"{job.key}.json"
        if destination.exists():
            _validate_existing(
                destination,
                job.key,
                smoke=smoke,
                contract_sha=hashlib.sha256((root / "contract.json").read_bytes()).hexdigest(),
            )
            continue
        try:
            result = run_job(root, job, device=device, smoke=smoke)
        except Exception as error:
            _atomic_json(
                root / "failed" / f"{job.key}.json",
                {
                    "schema": "pac.compact_h_only.synthetic_ood.failure.v1",
                    "job_key": job.key,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                },
            )
            raise
        _atomic_json(destination, result)
        completed_now += 1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if max_jobs is not None and completed_now >= max_jobs:
            break
    return completed_now


def _manifest_keys(root: Path) -> set[str]:
    return {
        SyntheticOODJob(cast("ModelName", row["model"]), int(row["seed"])).key
        for path in sorted((root / "manifests").glob("*.jsonl"))
        for row in map(json.loads, path.read_text(encoding="utf-8").splitlines())
    }


def status(root: Path) -> dict[str, object]:
    expected = _manifest_keys(root)
    completed = {path.stem for path in (root / "completed").glob("*.json")}
    failed = {path.stem for path in (root / "failed").glob("*.json")} - completed
    return {
        "schema": "pac.compact_h_only.synthetic_ood.status.v1",
        "expected": len(expected),
        "completed": len(expected & completed),
        "failed": len(expected & failed),
        "remaining": len(expected - completed),
        "done": bool(expected) and expected <= completed and not (expected & failed),
    }


def _require_complete(root: Path) -> list[dict[str, object]]:
    expected = _manifest_keys(root)
    if expected != {job.key for job in jobs()}:
        raise RuntimeError("manifest does not contain the exact 35-job matrix")
    by_key: dict[str, dict[str, object]] = {}
    contract_sha = hashlib.sha256((root / "contract.json").read_bytes()).hexdigest()
    lock = parameter_lock()
    expected_params = lock["expected_params"]
    expected_condition_keys = {
        (family, level, variant)
        for family, level in FROZEN_PROFILES
        for variant in VARIANTS
    }
    for path in sorted((root / "completed").glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        key = str(row.get("job_key", ""))
        if key in by_key:
            raise RuntimeError(f"duplicate completed key: {key}")
        model = str(row.get("model", ""))
        conditions = cast("list[dict[str, object]]", row.get("conditions", []))
        condition_keys = {
            (str(item["family"]), str(item["level"]), str(item["variant"]))
            for item in conditions
        }
        if (
            key not in expected
            or row.get("status") != "done"
            or row.get("smoke")
            or row.get("contract_sha256") != contract_sha
            or row.get("fit_test_tensors_empty") is not True
            or row.get("ood_constructed_after_fit") is not True
            or model not in MODELS
            or int(cast("int | str", row.get("params_trainable", -1)))
            != expected_params[model]
            or len(conditions) != CONDITIONS_PER_JOB * len(VARIANTS)
            or condition_keys != expected_condition_keys
        ):
            raise RuntimeError(f"invalid final result: {path}")
        if any(
            not math.isfinite(float(cast("float | int | str", item[field])))
            for item in conditions
            for field in ("mse", "nrmse")
        ):
            raise RuntimeError(f"nonfinite OOD result: {path}")
        by_key[key] = row
    missing = expected - set(by_key)
    extra = set(by_key) - expected
    if missing or extra:
        raise RuntimeError(f"incomplete final matrix: missing={len(missing)}, extra={len(extra)}")
    return [by_key[key] for key in sorted(expected)]


def report(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    payloads = _require_complete(root)
    long_rows = [
        {"model": payload["model"], "seed": payload["seed"], **condition}
        for payload in payloads
        for condition in cast("list[dict[str, object]]", payload["conditions"])
    ]
    profile_means: dict[tuple[str, str, str, str], list[float]] = {}
    for row in long_rows:
        key = (str(row["family"]), str(row["level"]), str(row["variant"]), str(row["model"]))
        profile_means.setdefault(key, []).append(
            float(cast("float | int | str", row["nrmse"]))
        )
    variants: dict[str, object] = {}
    for variant in VARIANTS:
        rank_values = {model: [] for model in MODELS}
        top = dict.fromkeys(MODELS, 0)
        profiles: list[dict[str, object]] = []
        family_levels = sorted(
            {(str(row["family"]), str(row["level"])) for row in long_rows}
        )
        if len(family_levels) != CONDITIONS_PER_JOB:
            raise RuntimeError("report does not contain the frozen 19-profile grid")
        for family, level in family_levels:
            scores = {
                model: mean(profile_means[(family, level, variant, model)])
                for model in MODELS
            }
            ranks = _average_ranks(scores)
            best = min(scores.values())
            winners = [
                model
                for model in MODELS
                if math.isclose(scores[model], best, rel_tol=1.0e-5, abs_tol=1.0e-8)
            ]
            for model in MODELS:
                rank_values[model].append(ranks[model])
                top[model] += model in winners
            profiles.append(
                {
                    "family": family,
                    "level": level,
                    "mean_nrmse": scores,
                    "average_tie_rank": ranks,
                    "joint_top1": winners,
                }
            )
        macro: dict[str, float] = {}
        for model in MODELS:
            family_means = [
                mean(
                    float(cast("float | int | str", row["nrmse"]))
                    for row in long_rows
                    if row["model"] == model
                    and row["variant"] == variant
                    and row["family"] == family
                )
                for family in sorted({str(row["family"]) for row in long_rows})
            ]
            macro[model] = mean(family_means)
        variants[variant] = {
            "models": {
                model: {
                    "equal_family_mean_nrmse": macro[model],
                    "mean_rank_19": mean(rank_values[model]),
                    "top1_19": top[model],
                }
                for model in MODELS
            },
            "profiles": profiles,
        }
    id_nrmse = {
        model: {
            "mean": mean(
                float(cast("float | int | str", row["id_test_nrmse"]))
                for row in payloads
                if row["model"] == model
            ),
            "sample_sd": stdev(
                float(cast("float | int | str", row["id_test_nrmse"]))
                for row in payloads
                if row["model"] == model
            ),
        }
        for model in MODELS
    }
    summary: dict[str, object] = {
        "schema": "pac.compact_h_only.synthetic_ood.report.v1",
        "complete": True,
        "models": list(MODELS),
        "seeds": list(SEEDS),
        "jobs": len(payloads),
        "condition_rows": len(long_rows),
        "parameters": parameter_lock(),
        "id_test_nrmse": id_nrmse,
        "variants": variants,
        "claim_boundary": (
            "same-recipe nearest-parameter diagnostic under one synthetic exact-ZOH teacher; "
            "not universal OOD superiority evidence"
        ),
    }
    reports = root / "reports"
    reports.mkdir(exist_ok=True)
    with (reports / "synthetic_ood_long.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(long_rows[0]))
        writer.writeheader()
        writer.writerows(long_rows)
    _atomic_json(reports / "summary.json", summary)
    _write_locked_text(root / "COMPLETE", "complete\n")
    return summary


def _validate_existing(
    path: Path,
    job_key: str,
    *,
    smoke: bool,
    contract_sha: str,
) -> None:
    row = json.loads(path.read_text(encoding="utf-8"))
    if (
        row.get("job_key") != job_key
        or row.get("status") != "done"
        or bool(row.get("smoke")) != smoke
        or row.get("contract_sha256") != contract_sha
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
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("enqueue", "worker", "status", "report"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument("--ucr-selection", type=Path, default=DEFAULT_UCR_SELECTION)
    parser.add_argument("--external-selection", type=Path, default=DEFAULT_EXTERNAL_SELECTION)
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    args = parser.parse_args()
    if args.stage == "enqueue":
        payload = enqueue(
            args.root,
            shards=args.shards,
            ucr_selection=args.ucr_selection,
            external_selection=args.external_selection,
            comparison=args.comparison,
        )
    elif args.stage == "worker":
        payload = {
            "completed_now": worker(
                args.root,
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
