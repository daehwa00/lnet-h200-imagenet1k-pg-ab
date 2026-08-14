# ruff: noqa: BLE001, EM101, EM102, T201, TRY003
# pyright: reportExplicitAny=false
"""Leakage-controlled PhysioNet/CinC 2012 mortality campaign.

Set A is split once, by a locked stratified RecordID hash, into development
and validation folds.  Hyperparameters are selected using that validation
fold only.  After selection is locked, each model is retrained on all of Set A
and evaluated once on the public official final-ranking Set C outcomes.  Event timestamps
are not binned: measurements sharing an original minute timestamp form one
event, and natural feature-level missingness is retained explicitly.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import tarfile
import tempfile
import urllib.request
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import mean, median, stdev
from typing import TYPE_CHECKING, Any, Final, Literal, Protocol, cast

import torch
from torch import Tensor, nn
from torch.nn import functional

from .alphabet import Alphabet
from .pac_external_benchmarks import (
    _build_continuous_model,  # pyright: ignore[reportPrivateUsage]
)
from .pac_irregular_models import GRUDClassifier
from .pac_metrics import count_parameters
from .pac_time_normalization import (
    fit_characteristic_time_scale,
    normalize_time_delta,
)
from .pac_types import PACDevice, PACExperimentConfig

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .pac_external_benchmarks import ExternalModelFamily

ModelName = Literal[
    "alphabet", "grud", "cnn1d", "tcn", "mamba", "gru", "lstm", "transformer"
]
BaselineName = Literal["grud", "cnn1d", "tcn", "mamba", "gru", "lstm", "transformer"]
Json = dict[str, Any]

MODELS: Final[tuple[ModelName, ...]] = (
    "alphabet",
    "grud",
    "cnn1d",
    "tcn",
    "mamba",
    "gru",
    "lstm",
    "transformer",
)
BASELINES: Final[tuple[BaselineName, ...]] = (
    "grud",
    "cnn1d",
    "tcn",
    "mamba",
    "gru",
    "lstm",
    "transformer",
)
SELECTION_SEEDS: Final = (7, 11, 19)
FINAL_SEEDS: Final = (23, 31, 43, 47, 59)
SPLIT_SALT: Final = "alphabet-physionet2012-validation-v1"
VALIDATION_FRACTION: Final = 0.20
MAX_SEQUENCE_LENGTH: Final = 208
PARAMETER_TOLERANCE: Final = 0.062
DEFAULT_DATA_ROOT: Final = Path("data/physionet2012")
DEFAULT_OUTPUT_ROOT: Final = Path("results/physionet2012")
SOURCE_BASE: Final = "https://physionet.org/files/challenge-2012/1.0.0"

# Version 1.0.0 variables, excluding RecordID.  Their order is part of the
# experiment contract and never inferred from the test set.
FEATURES: Final = (
    "ALP",
    "ALT",
    "AST",
    "Age",
    "Albumin",
    "BUN",
    "Bilirubin",
    "Cholesterol",
    "Creatinine",
    "DiasABP",
    "FiO2",
    "GCS",
    "Gender",
    "Glucose",
    "HCO3",
    "HCT",
    "HR",
    "Height",
    "ICUType",
    "K",
    "Lactate",
    "MAP",
    "MechVent",
    "Mg",
    "NIDiasABP",
    "NIMAP",
    "NISysABP",
    "Na",
    "PaCO2",
    "PaO2",
    "Platelets",
    "RespRate",
    "SaO2",
    "SysABP",
    "Temp",
    "TroponinI",
    "TroponinT",
    "Urine",
    "WBC",
    "Weight",
    "pH",
)
FEATURE_INDEX: Final = {name: index for index, name in enumerate(FEATURES)}
PACKED_INPUT_DIM: Final = 2 * len(FEATURES) + 2
ALPHABET_SIGNAL_DIM: Final = len(FEATURES)

DOWNLOADS: Final = (
    "set-a.tar.gz",
    "set-c.tar.gz",
    "Outcomes-a.txt",
    "Outcomes-c.txt",
)
MAX_DOWNLOAD_BYTES: Final = 512 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class Recipe:
    trial: int
    learning_rate: float
    weight_decay: float
    batch_size: int
    grad_clip_norm: float
    max_epochs: int = 60
    patience: int = 8


RECIPES: Final = (
    Recipe(1, 1.0e-3, 1.0e-5, 32, 0.5),
    Recipe(2, 1.0e-3, 1.0e-4, 64, 0.5),
    Recipe(3, 3.0e-3, 1.0e-5, 32, 1.0),
    Recipe(4, 3.0e-3, 1.0e-4, 64, 1.0),
    Recipe(5, 1.0e-2, 1.0e-5, 32, 2.0),
    Recipe(6, 1.0e-2, 1.0e-4, 64, 2.0),
)


@dataclass(frozen=True, slots=True)
class SelectionJob:
    model: ModelName
    trial: int
    seed: int

    @property
    def key(self) -> str:
        return f"physionet2012__selection__{self.model}__trial{self.trial}__seed{self.seed}"


@dataclass(frozen=True, slots=True)
class FinalJob:
    model: ModelName
    seed: int

    @property
    def key(self) -> str:
        return f"physionet2012__final__{self.model}__seed{self.seed}"


@dataclass(slots=True)
class RawCohort:
    record_ids: list[int]
    values: Tensor
    observed: Tensor
    delta_hours: Tensor
    valid: Tensor
    labels: Tensor


@dataclass(frozen=True, slots=True)
class BinaryMetrics:
    auroc: float
    auprc: float
    balanced_accuracy: float
    threshold: float


def selection_jobs() -> tuple[SelectionJob, ...]:
    return tuple(
        SelectionJob(model, recipe.trial, seed)
        for model in MODELS
        for recipe in RECIPES
        for seed in SELECTION_SEEDS
    )


def final_jobs() -> tuple[FinalJob, ...]:
    return tuple(FinalJob(model, seed) for model in MODELS for seed in FINAL_SEEDS)


def download_dataset(data_root: Path = DEFAULT_DATA_ROOT) -> Json:
    """Download and safely extract the four public version-locked artifacts."""
    raw = data_root / "raw"
    extracted = data_root / "extracted"
    raw.mkdir(parents=True, exist_ok=True)
    extracted.mkdir(parents=True, exist_ok=True)
    for name in DOWNLOADS:
        destination = raw / name
        if destination.is_file() and destination.stat().st_size > 0:
            continue
        temporary = destination.with_suffix(destination.suffix + f".tmp-{os.getpid()}")
        temporary.unlink(missing_ok=True)
        request = urllib.request.Request(
            f"{SOURCE_BASE}/{name}",
            headers={"User-Agent": "ALPHABET-reproducibility/1.0"},
        )
        with urllib.request.urlopen(request, timeout=60.0) as response:  # noqa: S310
            declared_length = response.headers.get("Content-Length")
            if declared_length is not None and int(declared_length) > MAX_DOWNLOAD_BYTES:
                raise RuntimeError(f"download exceeds size limit for {name}")
            with temporary.open("wb") as handle:
                total = 0
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > MAX_DOWNLOAD_BYTES:
                        raise RuntimeError(f"download exceeds size limit for {name}")
                    handle.write(chunk)
        temporary.replace(destination)
    for cohort in ("a", "c"):
        destination = extracted / f"set-{cohort}"
        if len(tuple(destination.glob("*.txt"))) == 4_000:
            continue
        staging = extracted / f".set-{cohort}.tmp-{os.getpid()}"
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        _safe_extract(raw / f"set-{cohort}.tar.gz", staging)
        source = staging / f"set-{cohort}"
        if len(tuple(source.glob("*.txt"))) != 4_000:
            raise RuntimeError(f"set-{cohort} archive did not contain 4000 records")
        shutil.rmtree(destination, ignore_errors=True)
        source.replace(destination)
        shutil.rmtree(staging, ignore_errors=True)
    manifest = dataset_audit(data_root)
    _atomic_json(data_root / "download-manifest.json", manifest)
    return manifest


def dataset_audit(data_root: Path = DEFAULT_DATA_ROOT) -> Json:
    raw = data_root / "raw"
    rows: Json = {}
    directory = data_root / "extracted" / "set-a"
    outcomes = _read_outcomes(raw / "Outcomes-a.txt")
    files = tuple(sorted(directory.glob("*.txt")))
    if len(files) != 4_000 or len(outcomes) != 4_000:
        raise RuntimeError("set-a is incomplete")
    record_ids = {int(path.stem) for path in files}
    if record_ids != set(outcomes):
        raise RuntimeError("set-a records and outcomes differ")
    max_steps = max(_record_step_count(path) for path in files)
    if max_steps > MAX_SEQUENCE_LENGTH:
        raise RuntimeError(f"set-a needs {max_steps} steps, lock is {MAX_SEQUENCE_LENGTH}")
    rows["set_a"] = {
        "records": len(files),
        "deaths": sum(outcomes.values()),
        "survivors": len(outcomes) - sum(outcomes.values()),
        "maximum_original_timestamp_groups": max_steps,
        "outcomes_file": "Outcomes-a.txt",
        "archive_file": "set-a.tar.gz",
    }
    # Before selection is locked, Set C remains sealed.  Only public artifact
    # identity is recorded; no Set-C record or outcome content is parsed.
    for name in ("set-c.tar.gz", "Outcomes-c.txt"):
        if not (raw / name).is_file() or (raw / name).stat().st_size <= 0:
            raise RuntimeError(f"sealed artifact is missing or empty: {name}")
    rows["set_c"] = {
        "sealed_until_final": True,
        "archive_file": "set-c.tar.gz",
        "outcomes_file": "Outcomes-c.txt",
        "outcomes_parsed_during_selection": False,
    }
    return {
        "schema": "alphabet.physionet2012.dataset_manifest.v1",
        "source": SOURCE_BASE,
        "license": "Open Data Commons Attribution License v1.0",
        "features": list(FEATURES),
        "sequence_encoding": "original minute timestamp groups; no temporal binning",
        "missingness": "natural feature-level observation indicators; no forward fill",
        "cohorts": rows,
    }


def load_cohort(data_root: Path, cohort: Literal["a", "c"]) -> RawCohort:
    directory = data_root / "extracted" / f"set-{cohort}"
    outcomes = _read_outcomes(data_root / "raw" / f"Outcomes-{cohort}.txt")
    record_ids = sorted(outcomes)
    values = torch.zeros((len(record_ids), MAX_SEQUENCE_LENGTH, len(FEATURES)))
    observed = torch.zeros_like(values, dtype=torch.uint8)
    delta = torch.zeros((len(record_ids), MAX_SEQUENCE_LENGTH, 1))
    valid = torch.zeros_like(delta, dtype=torch.uint8)
    for row_index, record_id in enumerate(record_ids):
        event_values, event_observed, event_delta = _parse_record(directory / f"{record_id}.txt")
        length = event_values.shape[0]
        values[row_index, :length] = event_values
        observed[row_index, :length] = event_observed
        delta[row_index, :length] = event_delta
        valid[row_index, :length] = 1
    labels = torch.tensor([outcomes[record_id] for record_id in record_ids], dtype=torch.long)
    return RawCohort(record_ids, values, observed, delta, valid, labels)


def stratified_split(record_ids: Sequence[int], labels: Tensor) -> tuple[Tensor, Tensor]:
    """Locked, seed-independent 80/20 split derived only from Set A."""
    validation: list[int] = []
    training: list[int] = []
    for class_index in (0, 1):
        members = [index for index, label in enumerate(labels.tolist()) if label == class_index]
        ordered = sorted(members, key=lambda index: record_ids[index])
        validation_count = round(len(ordered) * VALIDATION_FRACTION)
        validation.extend(ordered[:validation_count])
        training.extend(ordered[validation_count:])
    return torch.tensor(sorted(training)), torch.tensor(sorted(validation))


def pack_cohort(cohort: RawCohort, fit_indices: Tensor) -> Tensor:
    """Standardize observed values using only ``fit_indices`` and pack equal information."""
    fit_values = cohort.values[fit_indices]
    fit_observed = cohort.observed[fit_indices].bool()
    means = torch.zeros(len(FEATURES))
    scales = torch.ones(len(FEATURES))
    for feature in range(len(FEATURES)):
        selected = fit_values[..., feature][fit_observed[..., feature]]
        if selected.numel():
            means[feature] = selected.mean()
            scale = selected.std(unbiased=False)
            scales[feature] = scale if scale > 1.0e-6 else 1.0
    standardized = (cohort.values - means) / scales
    standardized = standardized * cohort.observed.to(standardized.dtype)
    # Fit one dataset-level clock from TRAIN transitions only.  The pole
    # dynamics, physical-time local operators, and radial-log lags then share
    # the same dimensionless time coordinate.
    time_scale = fit_characteristic_time_scale(
        cohort.delta_hours[fit_indices],
        cohort.valid[fit_indices],
    )
    delta = normalize_time_delta(
        cohort.delta_hours,
        time_scale,
        cohort.valid,
    )
    return torch.cat(
        (standardized, cohort.observed.to(torch.float32), delta, cohort.valid.to(torch.float32)),
        dim=-1,
    )


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

    def destroy(self) -> None: ...


class _PackedClinicalExactSplitRuntime:
    """Keep the clinical packed-input contract outside the captured graph."""

    def __init__(self, runtime: _MetadataExactSplitRuntime) -> None:
        self.runtime = runtime
        backend = getattr(runtime, "training_backend", "exact_split")
        self.training_backend = f"packed_clinical_metadata_{backend}"

    def step(self, packed: Tensor, labels: Tensor) -> Tensor:
        return self.runtime.step(
            packed[..., :ALPHABET_SIGNAL_DIM],
            labels,
            time_delta=packed[..., -2:-1],
            observation_mask=packed[..., ALPHABET_SIGNAL_DIM : 2 * ALPHABET_SIGNAL_DIM],
            valid_mask=packed[..., -1:],
        )

    def close(self) -> None:
        self.runtime.close()

    def destroy(self) -> None:
        self.runtime.destroy()


class ClinicalAlphabet(nn.Module):
    def __init__(self, config: PACExperimentConfig) -> None:
        super().__init__()
        self.core = Alphabet(
            replace(config, raw_input_dim=ALPHABET_SIGNAL_DIM),
            2,
            objective="classification",
        )

    def forward(self, packed: Tensor) -> Tensor:
        signals = packed[..., :ALPHABET_SIGNAL_DIM]
        observed = packed[..., ALPHABET_SIGNAL_DIM : 2 * ALPHABET_SIGNAL_DIM]
        delta = packed[..., -2:-1]
        valid = packed[..., -1:]
        return self.core(
            signals,
            time_delta=delta,
            observation_mask=observed,
            valid_mask=valid,
        )

    def post_optimizer_step(self) -> None:
        self.core.post_optimizer_step()

    def finalize_constraints(self) -> None:
        self.core.finalize_constraints()

    def prepare_external_exact_split_runtime(
        self,
        optimizer: torch.optim.AdamW,
        packed: Tensor,
        labels: Tensor,
        *,
        loss_weight: Tensor,
        grad_clip_norm: float,
    ) -> _PackedClinicalExactSplitRuntime:
        runtime = self.core.prepare_external_exact_split_runtime(
            optimizer,
            packed[..., :ALPHABET_SIGNAL_DIM],
            labels,
            objective="multiclass",
            grad_clip_norm=grad_clip_norm,
            time_delta=packed[..., -2:-1],
            observation_mask=packed[..., ALPHABET_SIGNAL_DIM : 2 * ALPHABET_SIGNAL_DIM],
            valid_mask=packed[..., -1:],
            loss_weight=loss_weight,
            metadata_prevalidated=True,
        )
        return _PackedClinicalExactSplitRuntime(
            cast("_MetadataExactSplitRuntime", cast("object", runtime))
        )


class ClinicalBaseline(nn.Module):
    def __init__(self, core: nn.Module) -> None:
        super().__init__()
        self.core = core

    def forward(self, packed: Tensor) -> Tensor:
        return self.core(packed)


class ClinicalGRUD(nn.Module):
    def __init__(self, width: int, *, depth: int = 1) -> None:
        super().__init__()
        self.core = GRUDClassifier(ALPHABET_SIGNAL_DIM, width, 2, depth=depth)

    def forward(self, packed: Tensor) -> Tensor:
        signals = packed[..., :ALPHABET_SIGNAL_DIM]
        observed = packed[..., ALPHABET_SIGNAL_DIM : 2 * ALPHABET_SIGNAL_DIM].bool()
        valid = packed[..., -1:].bool()
        feature_delta = torch.zeros_like(signals)
        elapsed = packed[..., -2:-1]
        for step in range(1, signals.shape[1]):
            feature_delta[:, step] = torch.where(
                observed[:, step - 1],
                elapsed[:, step],
                feature_delta[:, step - 1] + elapsed[:, step],
            )
        return self.core(signals, observed, feature_delta, valid)


def build_model(
    model: ModelName,
    width: int | None,
    config: PACExperimentConfig,
    *,
    seed: int,
) -> nn.Module:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if model == "alphabet":
            return ClinicalAlphabet(config)
        if width is None:
            raise ValueError(f"baseline {model} requires a matched width")
        if model == "grud":
            return ClinicalGRUD(width)
        core = _build_continuous_model(
            cast("ExternalModelFamily", model),
            width,
            PACKED_INPUT_DIM,
            2,
            config,
            "",
            objective="classification",
        )
        return ClinicalBaseline(core)


def parameter_matches(
    config: PACExperimentConfig,
) -> dict[ModelName, dict[str, int | float | None]]:
    target = count_parameters(build_model("alphabet", None, config, seed=0))
    matches: dict[ModelName, dict[str, int | float | None]] = {
        "alphabet": {"width": 32, "parameters": target, "relative_error": 0.0}
    }
    for family in BASELINES:
        candidates: list[tuple[int, int, int]] = []
        for width in range(1, 257):
            parameters = count_parameters(build_model(family, width, config, seed=0))
            candidates.append((abs(parameters - target), width, parameters))
        _, width, parameters = min(candidates)
        relative_error = abs(parameters - target) / target
        if relative_error > PARAMETER_TOLERANCE:
            raise RuntimeError(f"{family} parameter error {relative_error:.4%} exceeds tolerance")
        matches[family] = {
            "width": width,
            "parameters": parameters,
            "relative_error": relative_error,
        }
    return matches


def binary_metrics(
    probabilities: Tensor, labels: Tensor, threshold: float | None = None
) -> BinaryMetrics:
    probabilities = probabilities.detach().cpu().to(torch.float64)
    labels = labels.detach().cpu().bool()
    if threshold is None:
        candidates = torch.unique(probabilities).tolist()
        candidates.extend((0.0, 0.5, 1.0))
        threshold = min(
            candidates,
            key=lambda value: (
                -_balanced_accuracy(probabilities, labels, float(value)),
                abs(value - 0.5),
            ),
        )
    return BinaryMetrics(
        auroc=_binary_auroc(probabilities, labels),
        auprc=_binary_auprc(probabilities, labels),
        balanced_accuracy=_balanced_accuracy(probabilities, labels, threshold),
        threshold=float(threshold),
    )


def enqueue(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
    *,
    shards: int = 8,
    smoke: bool = False,
) -> Json:
    if shards < 1:
        raise ValueError("shards must be positive")
    audit = dataset_audit(data_root)
    output_root.mkdir(parents=True, exist_ok=True)
    config = _experiment_config(output_root, smoke=smoke)
    matches = parameter_matches(config)
    jobs = selection_jobs()
    contract: Json = {
        "schema": "alphabet.physionet2012.contract.v1",
        "source_manifest": ["src/lnet/pac_physionet2012.py"],
        "dataset_manifest": audit,
        "protocol": {
            "selection": "fixed Set-A TRAIN-derived stratified validation only",
            "final": "retrain all Set A after lock; evaluate untouched official Set C once",
            "selection_seeds": list(SELECTION_SEEDS),
            "final_seeds": list(FINAL_SEEDS),
            "split_salt": SPLIT_SALT,
            "validation_fraction": VALIDATION_FRACTION,
            "test_access_during_selection": False,
        },
        "information_contract": {
            "common": "values, feature observation indicators, elapsed time, padding indicator",
            "alphabet": (
                "values and feature indicators as signal channels; elapsed time and "
                "timestep padding through native time_delta, observation_mask, valid_mask routes"
            ),
            "baselines": "same four parts concatenated as channels",
            "temporal_binning": False,
        },
        "recipes": [asdict(recipe) for recipe in RECIPES],
        "parameter_matches": matches,
        "parameter_tolerance": PARAMETER_TOLERANCE,
        "selection_jobs": len(jobs),
        "final_jobs": len(final_jobs()),
        "shards": shards,
        "smoke": smoke,
    }
    _write_locked(output_root / "contract.json", contract)
    _write_manifests(output_root / "selection" / "manifests", jobs, shards)
    return contract


def run_selection_worker(
    output_root: Path,
    data_root: Path,
    *,
    shard: int,
    device: str,
    max_jobs: int | None = None,
) -> int:
    contract = _read_contract(output_root)
    jobs = _read_jobs(
        output_root / "selection" / "manifests" / f"shard-{shard:02d}.jsonl", SelectionJob
    )
    cohort = load_cohort(data_root, "a")
    train_indices, validation_indices = stratified_split(cohort.record_ids, cohort.labels)
    if bool(contract["smoke"]):
        train_indices = _balanced_subset(train_indices, cohort.labels, 32)
        validation_indices = _balanced_subset(validation_indices, cohort.labels, 16)
    packed = pack_cohort(cohort, train_indices)
    completed = 0
    for job in jobs:
        path = _result_path(output_root, "selection", job.key)
        if _done(path):
            continue
        try:
            payload = _run_selection_job(
                job,
                packed,
                cohort.labels,
                train_indices,
                validation_indices,
                contract,
                output_root,
                device,
            )
        except Exception as error:
            _atomic_json(path, {"job_key": job.key, "status": "failed", "error": repr(error)})
        else:
            _atomic_json(path, payload)
        completed += 1
        if max_jobs is not None and completed >= max_jobs:
            break
    return completed


def select(output_root: Path, data_root: Path, *, shards: int = 8) -> Json:
    _read_contract(output_root)
    rows = _strict_results(output_root, "selection", [job.key for job in selection_jobs()])
    if any(
        row.get("official_test_accessed") is not False
        or row.get("data_scope") != "set_a_train_and_fixed_validation_only"
        for row in rows
    ):
        raise RuntimeError("selection result violated the sealed Set-C data scope")
    selections: Json = {}
    for model in MODELS:
        candidates: list[tuple[float, float, int, list[Json]]] = []
        for recipe in RECIPES:
            selected = [
                row for row in rows if row["model"] == model and row["trial"] == recipe.trial
            ]
            candidates.append(
                (
                    -mean(float(row["validation"]["auprc"]) for row in selected),
                    -mean(float(row["validation"]["auroc"]) for row in selected),
                    recipe.trial,
                    selected,
                )
            )
        _, _, trial, selected_rows = min(candidates)
        selections[model] = {
            "trial": trial,
            "recipe": asdict(_recipe(trial)),
            "final_epochs": max(1, round(median(int(row["best_epoch"]) for row in selected_rows))),
            "threshold": float(
                median(float(row["validation"]["threshold"]) for row in selected_rows)
            ),
            "mean_validation_auprc": mean(
                float(row["validation"]["auprc"]) for row in selected_rows
            ),
            "mean_validation_auroc": mean(
                float(row["validation"]["auroc"]) for row in selected_rows
            ),
        }
    payload: Json = {
        "schema": "alphabet.physionet2012.selection.v1",
        "selection_primary": "mean validation AUPRC; AUROC then lower trial tie-break",
        "official_test_accessed": False,
        "models": selections,
    }
    _write_locked(output_root / "selection.json", payload)
    # Only after the selection artifact is immutable do we expose final jobs.
    # and expose final jobs.
    dataset_audit(data_root)
    _write_manifests(output_root / "final" / "manifests", final_jobs(), shards)
    return payload


def run_final_worker(
    output_root: Path,
    data_root: Path,
    *,
    shard: int,
    device: str,
    max_jobs: int | None = None,
) -> int:
    contract = _read_contract(output_root)
    if bool(contract["smoke"]):
        raise RuntimeError(
            "smoke campaigns cannot unseal official Set C; use a non-smoke locked selection"
        )
    selection = json.loads((output_root / "selection.json").read_text(encoding="utf-8"))
    jobs = _read_jobs(output_root / "final" / "manifests" / f"shard-{shard:02d}.jsonl", FinalJob)
    train = load_cohort(data_root, "a")
    test = load_cohort(data_root, "c")
    all_train = torch.arange(len(train.record_ids))
    train_packed = pack_cohort(train, all_train)
    # Use Set-A statistics for Set C.  Reconstruct them by temporarily joining
    # tensors but never include Set-C rows in fit_indices.
    joined = RawCohort(
        train.record_ids + test.record_ids,
        torch.cat((train.values, test.values)),
        torch.cat((train.observed, test.observed)),
        torch.cat((train.delta_hours, test.delta_hours)),
        torch.cat((train.valid, test.valid)),
        torch.cat((train.labels, test.labels)),
    )
    test_packed = pack_cohort(joined, all_train)[len(train.record_ids) :]
    train_labels = train.labels
    test_labels = test.labels
    completed = 0
    for job in jobs:
        path = _result_path(output_root, "final", job.key)
        if _done(path):
            continue
        try:
            payload = _run_final_job(
                job,
                train_packed,
                train_labels,
                test_packed,
                test_labels,
                selection,
                contract,
                output_root,
                device,
            )
        except Exception as error:
            _atomic_json(path, {"job_key": job.key, "status": "failed", "error": repr(error)})
        else:
            _atomic_json(path, payload)
        completed += 1
        if max_jobs is not None and completed >= max_jobs:
            break
    return completed


def status(output_root: Path) -> Json:
    report: Json = {}
    for stage, jobs in (("selection", selection_jobs()), ("final", final_jobs())):
        if not (output_root / stage / "manifests").exists():
            report[stage] = {"expected": len(jobs), "done": 0, "failed": 0, "pending": len(jobs)}
            continue
        paths = [_result_path(output_root, stage, job.key) for job in jobs]
        done = sum(_done(path) for path in paths)
        failed = sum(path.is_file() and not _done(path) for path in paths)
        report[stage] = {
            "expected": len(jobs),
            "done": done,
            "failed": failed,
            "pending": len(jobs) - done - failed,
        }
    return report


def report(output_root: Path) -> Json:
    rows = _strict_results(output_root, "final", [job.key for job in final_jobs()])
    summaries: list[Json] = []
    for model in MODELS:
        selected = [row for row in rows if row["model"] == model]
        summary: Json = {
            "model": model,
            "width": selected[0]["width"],
            "parameters": selected[0]["parameters"],
            "relative_parameter_error": selected[0]["relative_parameter_error"],
        }
        for metric in ("auroc", "auprc", "balanced_accuracy"):
            values = [float(row["official_test"][metric]) for row in selected]
            summary[f"mean_{metric}"] = mean(values)
            summary[f"sample_sd_{metric}"] = stdev(values)
        summaries.append(summary)
    for metric in ("auroc", "auprc", "balanced_accuracy"):
        ordered = sorted(
            summaries, key=lambda row: (-float(row[f"mean_{metric}"]), str(row["model"]))
        )
        for rank, row in enumerate(ordered, start=1):
            row[f"rank_{metric}"] = rank
    payload: Json = {
        "schema": "alphabet.physionet2012.report.v1",
        "dataset": "PhysioNet/CinC Challenge 2012",
        "endpoint": "In-hospital_death",
        "official_test_set": "Set C (official final-ranking cohort)",
        "seed_count": len(FINAL_SEEDS),
        "models": summaries,
    }
    _atomic_json(output_root / "report.json", payload)
    lines = [
        "# PhysioNet 2012 mortality comparison",
        "",
        (
            "Set-A validation-only selection; full Set-A retraining; "
            "one untouched official Set-C evaluation."
        ),
        "",
        "| Model | Params | AUROC | AUPRC | Balanced accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    lines.extend(
        (
            f"| {row['model']} | {row['parameters']} | "
            f"{float(row['mean_auroc']):.4f} +/- {float(row['sample_sd_auroc']):.4f} | "
            f"{float(row['mean_auprc']):.4f} +/- {float(row['sample_sd_auprc']):.4f} | "
            f"{float(row['mean_balanced_accuracy']):.4f} +/- "
            f"{float(row['sample_sd_balanced_accuracy']):.4f} |"
        )
        for row in summaries
    )
    _atomic_text(output_root / "report.md", "\n".join(lines) + "\n")
    return payload


def _run_selection_job(
    job: SelectionJob,
    packed: Tensor,
    labels: Tensor,
    train_indices: Tensor,
    validation_indices: Tensor,
    contract: Json,
    output_root: Path,
    device: str,
) -> Json:
    recipe = _recipe(job.trial)
    if bool(contract["smoke"]):
        recipe = replace(recipe, batch_size=8, max_epochs=1, patience=0)
    config = _experiment_config(output_root, smoke=bool(contract["smoke"]), recipe=recipe)
    match = contract["parameter_matches"][job.model]
    model = build_model(job.model, _optional_int(match["width"]), config, seed=job.seed)
    fitted = _fit(
        model,
        packed[train_indices],
        labels[train_indices],
        packed[validation_indices],
        labels[validation_indices],
        recipe,
        job.seed,
        device,
    )
    return {
        "schema": "alphabet.physionet2012.selection_result.v1",
        "job_key": job.key,
        "status": "done",
        "model": job.model,
        "trial": job.trial,
        "seed": job.seed,
        "width": match["width"],
        "parameters": count_parameters(model),
        "best_epoch": fitted["best_epoch"],
        "validation": fitted["validation"],
        "data_scope": "set_a_train_and_fixed_validation_only",
        "official_test_accessed": False,
    }


def _run_final_job(
    job: FinalJob,
    train_inputs: Tensor,
    train_labels: Tensor,
    test_inputs: Tensor,
    test_labels: Tensor,
    selection: Json,
    contract: Json,
    output_root: Path,
    device: str,
) -> Json:
    selected = selection["models"][job.model]
    recipe = replace(
        _recipe(int(selected["trial"])), max_epochs=int(selected["final_epochs"]), patience=0
    )
    if bool(contract["smoke"]):
        recipe = replace(recipe, batch_size=8, max_epochs=1)
    config = _experiment_config(output_root, smoke=bool(contract["smoke"]), recipe=recipe)
    match = contract["parameter_matches"][job.model]
    model = build_model(job.model, _optional_int(match["width"]), config, seed=job.seed)
    fitted = _fit(model, train_inputs, train_labels, None, None, recipe, job.seed, device)
    probabilities = _predict_probabilities(model, test_inputs, recipe.batch_size, device)
    metrics = binary_metrics(probabilities, test_labels, float(selected["threshold"]))
    return {
        "schema": "alphabet.physionet2012.final_result.v1",
        "job_key": job.key,
        "status": "done",
        "model": job.model,
        "seed": job.seed,
        "trial": selected["trial"],
        "epochs": recipe.max_epochs,
        "threshold_locked_from_set_a_validation": selected["threshold"],
        "width": match["width"],
        "parameters": count_parameters(model),
        "target_parameters": contract["parameter_matches"]["alphabet"]["parameters"],
        "relative_parameter_error": match["relative_error"],
        "train_loss": fitted["train_loss"],
        "official_test": asdict(metrics),
    }


def _training_batch(
    model: nn.Module,
    optimizer: torch.optim.AdamW,
    batch_inputs: Tensor,
    batch_labels: Tensor,
    weights: Tensor,
    recipe: Recipe,
    runtime: _PackedClinicalExactSplitRuntime | None,
    captured_batch_size: int | None,
    *,
    use_exact_split: bool,
) -> tuple[_PackedClinicalExactSplitRuntime | None, int | None, bool]:
    use_runtime = use_exact_split and (
        captured_batch_size is None or batch_inputs.shape[0] == captured_batch_size
    )
    if use_runtime and runtime is None:
        runtime = cast("ClinicalAlphabet", model).prepare_external_exact_split_runtime(
            optimizer,
            batch_inputs,
            batch_labels,
            loss_weight=weights,
            grad_clip_norm=recipe.grad_clip_norm,
        )
        captured_batch_size = int(batch_inputs.shape[0])
    if use_runtime and runtime is not None:
        runtime.step(batch_inputs, batch_labels)
        return runtime, captured_batch_size, True
    if runtime is not None:
        runtime.close()
    optimizer.zero_grad(set_to_none=runtime is None)
    logits = model(batch_inputs)
    loss = functional.cross_entropy(logits, batch_labels, weight=weights)
    auxiliary = getattr(model, "auxiliary_loss", None)
    if isinstance(auxiliary, Tensor):
        loss = loss + auxiliary
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), recipe.grad_clip_norm)
    optimizer.step()
    callback = getattr(model, "post_optimizer_step", None)
    if callable(callback):
        callback()
    return runtime, captured_batch_size, False


def _validation_checkpoint(
    model: nn.Module,
    inputs: Tensor,
    labels: Tensor,
    recipe: Recipe,
    device: str,
    epoch: int,
    best_score: float,
    stale: int,
) -> tuple[float, int, int, dict[str, Tensor] | None, bool]:
    probabilities = _predict_probabilities(model, inputs, recipe.batch_size, device)
    metrics = binary_metrics(probabilities, labels)
    if metrics.auprc > best_score + 1.0e-12:
        state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        return metrics.auprc, epoch, 0, state, False
    stale += 1
    should_stop = bool(recipe.patience and stale >= recipe.patience)
    return best_score, epoch, stale, None, should_stop


def _close_exact_split_runtime(runtime: _PackedClinicalExactSplitRuntime | None) -> None:
    if runtime is not None:
        runtime.close()


def _destroy_exact_split_runtime(runtime: _PackedClinicalExactSplitRuntime | None) -> None:
    if runtime is not None:
        runtime.destroy()


def _fit(
    model: nn.Module,
    train_inputs: Tensor,
    train_labels: Tensor,
    validation_inputs: Tensor | None,
    validation_labels: Tensor | None,
    recipe: Recipe,
    seed: int,
    device: str,
    *,
    enable_exact_split: bool | None = None,
) -> Json:
    torch.manual_seed(seed)
    model.to(device)
    exact_split_available = device.startswith("cuda") and isinstance(
        model, ClinicalAlphabet
    )
    use_exact_split = (
        exact_split_available
        if enable_exact_split is None
        else enable_exact_split and exact_split_available
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=recipe.learning_rate,
        weight_decay=recipe.weight_decay,
        fused=use_exact_split,
        capturable=use_exact_split,
    )
    counts = torch.bincount(train_labels, minlength=2).to(torch.float32)
    weights = (counts.sum() / (2.0 * counts.clamp_min(1))).to(device)
    generator = torch.Generator().manual_seed(seed)
    best_score = -math.inf
    best_epoch = recipe.max_epochs
    best_state: dict[str, Tensor] | None = None
    stale = 0
    runtime: _PackedClinicalExactSplitRuntime | None = None
    captured_batch_size: int | None = None
    exact_split_steps = 0
    eager_fallback_steps = 0
    try:
        for epoch in range(1, recipe.max_epochs + 1):
            model.train()
            order = torch.randperm(train_inputs.shape[0], generator=generator)
            for indices in order.split(recipe.batch_size):
                batch_inputs = train_inputs[indices].to(device)
                batch_labels = train_labels[indices].to(device)
                runtime, captured_batch_size, used_runtime = _training_batch(
                    model,
                    optimizer,
                    batch_inputs,
                    batch_labels,
                    weights,
                    recipe,
                    runtime,
                    captured_batch_size,
                    use_exact_split=use_exact_split,
                )
                exact_split_steps += int(used_runtime)
                eager_fallback_steps += int(use_exact_split and not used_runtime)
            _close_exact_split_runtime(runtime)
            if validation_inputs is None or validation_labels is None:
                continue
            (
                best_score,
                updated_epoch,
                stale,
                updated_state,
                should_stop,
            ) = _validation_checkpoint(
                model,
                validation_inputs,
                validation_labels,
                recipe,
                device,
                epoch,
                best_score,
                stale,
            )
            if updated_state is not None:
                best_epoch = updated_epoch
                best_state = updated_state
            if should_stop:
                break
    finally:
        _destroy_exact_split_runtime(runtime)
        model.__dict__["physionet_exact_split_steps"] = exact_split_steps
        model.__dict__["physionet_exact_split_fallback_steps"] = eager_fallback_steps
    if best_state is not None:
        model.load_state_dict(best_state)
    callback = getattr(model, "finalize_constraints", None)
    if callable(callback):
        callback()
    train_loss = _classification_loss(model, train_inputs, train_labels, recipe.batch_size, device)
    result: Json = {"best_epoch": best_epoch, "train_loss": train_loss}
    if validation_inputs is not None and validation_labels is not None:
        probabilities = _predict_probabilities(model, validation_inputs, recipe.batch_size, device)
        result["validation"] = asdict(binary_metrics(probabilities, validation_labels))
    return result


def _predict_probabilities(
    model: nn.Module, inputs: Tensor, batch_size: int, device: str
) -> Tensor:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        rows = [
            torch.softmax(model(batch.to(device)), dim=-1)[:, 1].cpu()
            for batch in inputs.split(batch_size)
        ]
    model.train(was_training)
    return torch.cat(rows)


def _classification_loss(
    model: nn.Module, inputs: Tensor, labels: Tensor, batch_size: int, device: str
) -> float:
    was_training = model.training
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch, target in zip(inputs.split(batch_size), labels.split(batch_size), strict=True):
            total += float(
                functional.cross_entropy(
                    model(batch.to(device)), target.to(device), reduction="sum"
                ).item()
            )
    model.train(was_training)
    return total / len(labels)


def _binary_auroc(scores: Tensor, labels: Tensor) -> float:
    positives = int(labels.sum())
    negatives = labels.numel() - positives
    if not positives or not negatives:
        return math.nan
    order = torch.argsort(scores, descending=True, stable=True)
    scores = scores[order]
    labels = labels[order]
    tp = fp = 0.0
    previous_tp = previous_fp = area = 0.0
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and scores[end] == scores[start]:
            end += 1
        group = labels[start:end]
        tp += float(group.sum())
        fp += float((~group).sum())
        area += (fp - previous_fp) * (tp + previous_tp) / 2.0
        previous_tp, previous_fp = tp, fp
        start = end
    return area / (positives * negatives)


def _binary_auprc(scores: Tensor, labels: Tensor) -> float:
    positives = int(labels.sum())
    if not positives:
        return math.nan
    order = torch.argsort(scores, descending=True, stable=True)
    scores = scores[order]
    labels = labels[order]
    tp = fp = previous_recall = area = 0.0
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and scores[end] == scores[start]:
            end += 1
        group = labels[start:end]
        tp += float(group.sum())
        fp += float((~group).sum())
        recall = tp / positives
        precision = tp / max(tp + fp, 1.0)
        area += (recall - previous_recall) * precision
        previous_recall = recall
        start = end
    return area


def _balanced_accuracy(scores: Tensor, labels: Tensor, threshold: float) -> float:
    predictions = scores >= threshold
    positive_recall = float((predictions & labels).sum()) / max(float(labels.sum()), 1.0)
    negative = ~labels
    negative_recall = float((~predictions & negative).sum()) / max(float(negative.sum()), 1.0)
    return (positive_recall + negative_recall) / 2.0


def _parse_record(path: Path) -> tuple[Tensor, Tensor, Tensor]:
    grouped: dict[int, dict[int, list[float]]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            parameter = row["Parameter"]
            if parameter == "RecordID":
                continue
            # Official Set C contains a small number of malformed rows whose
            # variable name is empty.  Their values cannot be assigned to any
            # feature, so the post-selection protocol amendment drops only
            # those rows.  Unknown non-empty variables still fail closed.
            if not parameter.strip():
                continue
            if parameter not in FEATURE_INDEX:
                raise RuntimeError(f"unknown PhysioNet variable {parameter!r}")
            hour, minute = (int(value) for value in row["Time"].split(":"))
            timestamp = 60 * hour + minute
            feature = FEATURE_INDEX[parameter]
            grouped.setdefault(timestamp, {}).setdefault(feature, []).append(float(row["Value"]))
    timestamps = sorted(grouped)
    if len(timestamps) > MAX_SEQUENCE_LENGTH:
        # Selection fixes a 208-step input geometry around the 203-step Set-A
        # maximum.  One official Set-C record exceeds it, so the post-selection amendment
        # applies the same deterministic right-truncation boundary to every
        # family instead of changing the selected input geometry.
        timestamps = timestamps[:MAX_SEQUENCE_LENGTH]
    values = torch.zeros((len(timestamps), len(FEATURES)))
    observed = torch.zeros_like(values, dtype=torch.uint8)
    delta = torch.zeros((len(timestamps), 1))
    previous = 0
    for row_index, timestamp in enumerate(timestamps):
        delta[row_index, 0] = (timestamp - previous) / 60.0
        previous = timestamp
        for feature, candidates in grouped[timestamp].items():
            # -1 is the official unknown sentinel, not an observed value.
            valid_candidates = [value for value in candidates if value >= 0.0]
            if valid_candidates:
                values[row_index, feature] = mean(valid_candidates)
                observed[row_index, feature] = 1
    return values, observed, delta


def _record_step_count(path: Path) -> int:
    with path.open(encoding="utf-8", newline="") as handle:
        return len({row["Time"] for row in csv.DictReader(handle)})


def _balanced_subset(indices: Tensor, labels: Tensor, count: int) -> Tensor:
    per_class = count // 2
    selected: list[int] = []
    members = indices.tolist()
    for class_index in (0, 1):
        class_members = [index for index in members if int(labels[index]) == class_index]
        selected.extend(class_members[:per_class])
    if len(selected) < count:
        selected_set = set(selected)
        selected.extend(index for index in members if index not in selected_set)
    return torch.tensor(sorted(selected[:count]))


def _read_outcomes(path: Path) -> dict[int, int]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            int(row["RecordID"]): int(row["In-hospital_death"]) for row in csv.DictReader(handle)
        }


def _safe_extract(archive: Path, destination: Path) -> None:
    with tarfile.open(archive) as handle:
        for member in handle.getmembers():
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination.resolve()) or member.issym() or member.islnk():
                raise RuntimeError(f"unsafe archive member: {member.name}")
        handle.extractall(destination, filter="data")


def _experiment_config(
    root: Path, *, smoke: bool, recipe: Recipe | None = None
) -> PACExperimentConfig:
    active = RECIPES[0] if recipe is None else recipe
    return PACExperimentConfig(
        sample_count=32 if smoke else 3_200,
        validation_count=16 if smoke else 800,
        test_count=16 if smoke else 4_000,
        sequence_length=MAX_SEQUENCE_LENGTH,
        raw_input_dim=len(FEATURES),
        output_dim=2,
        model_dim=32,
        modes=16,
        epochs=1 if smoke else active.max_epochs,
        batch_size=8 if smoke else active.batch_size,
        learning_rate=active.learning_rate,
        weight_decay=active.weight_decay,
        grad_clip_norm=active.grad_clip_norm,
        device=cast("PACDevice", "cuda"),
        output_dir=root,
        precision="fp32",
    )


def _recipe(trial: int) -> Recipe:
    return next(recipe for recipe in RECIPES if recipe.trial == trial)


def _write_manifests(
    directory: Path, jobs: Sequence[SelectionJob] | Sequence[FinalJob], shards: int
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for shard in range(shards):
        rows = [asdict(job) for index, job in enumerate(jobs) if index % shards == shard]
        text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        _write_locked_text(directory / f"shard-{shard:02d}.jsonl", text)


def _read_jobs(path: Path, job_type: type[Any]) -> list[Any]:
    return [job_type(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines()]


def _strict_results(output_root: Path, stage: str, keys: Sequence[str]) -> list[Json]:
    expected = set(keys)
    directory = output_root / stage / "results"
    files = tuple(directory.glob("*.json")) if directory.exists() else ()
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    observed = {str(row.get("job_key")) for row in rows if row.get("status") == "done"}
    if observed != expected or len(rows) != len(expected):
        message = "{} incomplete: done={}/{}, extra={}".format(  # noqa: UP032
            stage,
            len(observed),
            len(expected),
            sorted(observed - expected),
        )
        raise RuntimeError(message)
    return rows


def _result_path(root: Path, stage: str, key: str) -> Path:
    return root / stage / "results" / f"{key}.json"


def _done(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "done"
    except (json.JSONDecodeError, OSError):
        return False


def _read_contract(root: Path) -> Json:
    path = root / "contract.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "alphabet.physionet2012.contract.v1":
        raise RuntimeError("unexpected PhysioNet contract")
    source_manifest = payload.get("source_manifest")
    if not isinstance(source_manifest, list) or not all(
        isinstance(path, str) and path for path in source_manifest
    ):
        raise RuntimeError("PhysioNet source manifest is invalid")
    return payload


def _write_locked(path: Path, payload: Json) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    _write_locked_text(path, text)


def _write_locked_text(path: Path, text: str) -> None:
    if path.is_file():
        if path.read_text(encoding="utf-8") != text:
            raise RuntimeError(f"locked artifact differs: {path}")
        return
    _atomic_text(path, text)


def _atomic_json(path: Path, payload: Json) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.replace(path)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int):
        raise TypeError(f"expected integer width, got {type(value).__name__}")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "download",
            "enqueue",
            "worker-selection",
            "select",
            "worker-final",
            "status",
            "report",
        ),
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "download":
        payload = download_dataset(args.data_root)
    elif args.command == "enqueue":
        payload = enqueue(args.output_root, args.data_root, shards=args.shards, smoke=args.smoke)
    elif args.command == "worker-selection":
        payload = {
            "processed": run_selection_worker(
                args.output_root,
                args.data_root,
                shard=args.shard,
                device=args.device,
                max_jobs=args.max_jobs,
            )
        }
    elif args.command == "select":
        payload = select(args.output_root, args.data_root, shards=args.shards)
    elif args.command == "worker-final":
        payload = {
            "processed": run_final_worker(
                args.output_root,
                args.data_root,
                shard=args.shard,
                device=args.device,
                max_jobs=args.max_jobs,
            )
        }
    elif args.command == "status":
        payload = status(args.output_root)
    else:
        payload = report(args.output_root)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
