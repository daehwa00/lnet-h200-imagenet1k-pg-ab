from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch

from .pac_confirmatory_baselines import (
    confirmatory_implementation_metadata,
    confirmatory_trial_spec,
)
from .pac_overnight_io import read_csv
from .pac_stiefel_variants import capacity_for_model
from .pac_tf_p1p2_types import P1P2Config, P1P2Job

if TYPE_CHECKING:
    from .pac_confirmatory_baselines import ConfirmatoryFamily

EVIDENCE_MODELS = (
    "pac_tf",
    "tcn",
    "cnn1d",
    "gru",
    "lstm",
    "transformer",
    "mamba",
    "s4d",
    "inception_time",
)


def build_p1p2_jobs(config: P1P2Config) -> tuple[P1P2Job, ...]:
    protocol_bytes = config.protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    _validate_protocol(protocol)
    protocol_sha256 = hashlib.sha256(protocol_bytes).hexdigest()
    selected, reference_model = _read_locked_selection(config, protocol_sha256)
    unknown_models = set(config.models) - set(EVIDENCE_MODELS)
    if unknown_models:
        message = f"unsupported P1/P2 models: {sorted(unknown_models)}"
        raise ValueError(message)
    seeds = tuple(int(seed) for seed in protocol["seeds"])
    final_datasets = tuple(str(name) for name in protocol["untouched_final_datasets"])
    ratios = tuple(float(value) for value in protocol["low_data_ratios"])
    p0_sources = _read_p0_final_sources(
        config,
        seeds=seeds,
        datasets=final_datasets,
        selected=selected,
        reference_model=reference_model,
    )
    jobs = [
        _classification_evidence_job(
            p0_sources,
            selected,
            reference_model=reference_model,
            protocol_sha256=protocol_sha256,
            model=model,
            dataset=dataset,
            seed=seed,
            ratio=ratio,
        )
        for seed in seeds
        for dataset in final_datasets
        for ratio in ratios
        for model in config.models
    ]
    efficiency = protocol["efficiency"]
    jobs.extend(
        P1P2Job(
            key=f"pac_tf__efficiency__{model}__{runtime}__n{length}__b{batch_size}",
            package="efficiency",
            seed=seeds[0],
            model=model,
            reference_model=reference_model,
            length=int(length),
            batch_size=int(batch_size),
            runtime=runtime,
            slots=16 if runtime == "compiled" else 2,
            selection_trial=int(selected[model]["trial"]),
            architecture_metadata_json=_architecture_json(model, int(selected[model]["trial"])),
            refit_epochs=int(selected[model]["refit_epochs"]),
            learning_rate=float(selected[model]["learning_rate"]),
            weight_decay=float(selected[model]["weight_decay"]),
            protocol_sha256=protocol_sha256,
        )
        for length in efficiency["sequence_lengths"]
        for batch_size in efficiency["batch_sizes"]
        for runtime in ("train", "eager", "compiled")
        for model in config.models
    )
    jobs.extend(
        P1P2Job(
            key=f"pac_tf__synthetic_ood_suite__{model}__seed{seed}",
            package="synthetic_ood",
            seed=seed,
            model=model,
            reference_model=reference_model,
            slots=2,
            selection_trial=int(selected[model]["trial"]),
            architecture_metadata_json=_architecture_json(model, int(selected[model]["trial"])),
            refit_epochs=int(selected[model]["refit_epochs"]),
            learning_rate=float(selected[model]["learning_rate"]),
            weight_decay=float(selected[model]["weight_decay"]),
            protocol_sha256=protocol_sha256,
            synthetic_estimand=config.synthetic_estimand,
            synthetic_target_params=config.synthetic_target_params,
        )
        for seed in seeds
        for model in config.models
    )
    jobs.extend(
        P1P2Job(
            key=f"pac_tf__real_domain_ood__mit_bih__{model}__seed{seed}",
            package="real_domain_ood",
            seed=seed,
            model=model,
            reference_model=reference_model,
            dataset="mit-bih-ds1-ds2",
            slots=2,
            selection_trial=int(selected[model]["trial"]),
            architecture_metadata_json=_architecture_json(model, int(selected[model]["trial"])),
            refit_epochs=int(selected[model]["refit_epochs"]),
            learning_rate=float(selected[model]["learning_rate"]),
            weight_decay=float(selected[model]["weight_decay"]),
            protocol_sha256=protocol_sha256,
        )
        for seed in seeds
        for model in config.models
    )
    return tuple(job for job in jobs if job.package in config.packages)


def _classification_evidence_job(
    sources: dict[str, dict[str, str]],
    selected: dict[str, dict[str, float | int]],
    *,
    reference_model: str,
    protocol_sha256: str,
    model: str,
    dataset: str,
    seed: int,
    ratio: float,
) -> P1P2Job:
    source = _p0_job_fields(sources, model, dataset, seed, ratio)
    return P1P2Job(
        key=f"pac_tf__low_data__{model}__{dataset}__ratio{ratio:g}__seed{seed}",
        package="real_diagnostics" if ratio == 1.0 else "low_data",
        seed=seed,
        model=model,
        reference_model=reference_model,
        dataset=dataset,
        ratio=ratio,
        selection_trial=int(selected[model]["trial"]),
        architecture_metadata_json=_architecture_json(model, int(selected[model]["trial"])),
        refit_epochs=int(selected[model]["refit_epochs"]),
        learning_rate=float(selected[model]["learning_rate"]),
        weight_decay=float(selected[model]["weight_decay"]),
        protocol_sha256=protocol_sha256,
        p0_job_key=source[0],
        p0_checkpoint_path=source[1],
        p0_checkpoint_sha256=source[2],
        p0_metrics_json=source[3],
    )


def _architecture_json(family: str, trial: int) -> str:
    return json.dumps(
        confirmatory_implementation_metadata(cast("ConfirmatoryFamily", family), trial),
        sort_keys=True,
        separators=(",", ":"),
    )


def write_manifest(config: P1P2Config) -> Path:
    config.output_root.mkdir(parents=True, exist_ok=True)
    jobs = build_p1p2_jobs(config)
    protocol_sha256 = jobs[0].protocol_sha256
    path = config.output_root / "queue_manifest.jsonl"
    path.write_text(
        "".join(json.dumps(asdict(job), sort_keys=True) + "\n" for job in jobs),
        encoding="utf-8",
    )
    metadata = {
        "reference_model": jobs[0].reference_model,
        "protocol": str(config.protocol_path),
        "protocol_sha256": protocol_sha256,
        "selection": str(config.selection_path),
        "unseen_root": str(config.unseen_root),
        "p0_checkpoint_policy": "exact frozen full-TRAIN checkpoint reused at ratio=1",
        "test_policy": "locked protocol; no job result may alter models, hyperparameters, or jobs",
        "real_ood_scope": (
            "UCR corruption shifts plus MIT-BIH patient-disjoint DS1-to-DS2 domain OOD"
        ),
        "packages": list(config.packages),
        "synthetic_estimand": config.synthetic_estimand,
        "synthetic_target_params": config.synthetic_target_params,
    }
    (config.output_root / "manifest_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def _validate_protocol(protocol: dict[str, object]) -> None:
    if protocol.get("locked_before_final_evaluation") is not True:
        message = "P1/P2 final evidence requires a locked protocol"
        raise ValueError(message)
    if protocol.get("protocol_id") != "pac-tf-confirmatory-20260711-v1":
        message = "unexpected confirmatory protocol id"
        raise ValueError(message)
    if protocol.get("low_data_ratios") != [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]:
        message = "low-data ratios differ from the locked contract"
        raise ValueError(message)
    efficiency = protocol.get("efficiency")
    if not isinstance(efficiency, dict):
        message = "efficiency contract must be an object"
        raise TypeError(message)


def _read_locked_selection(  # noqa: C901, PLR0912 - explicit fail-closed validation
    config: P1P2Config, protocol_sha256: str
) -> tuple[dict[str, dict[str, float | int]], str]:
    if not config.selection_path.exists():
        message = f"P1/P2 enqueue refused: missing P0 tuning artifact {config.selection_path}"
        raise FileNotFoundError(message)
    payload = json.loads(config.selection_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "pac_confirmatory_baseline_selection.v1":
        message = "P1/P2 enqueue refused: unexpected P0 tuning schema"
        raise ValueError(message)
    if payload.get("status") != "complete":
        message = "P1/P2 enqueue refused: P0 tuning is not complete"
        raise ValueError(message)
    if payload.get("protocol_sha256") != protocol_sha256:
        message = "P1/P2 enqueue refused: tuning/protocol SHA-256 mismatch"
        raise ValueError(message)
    reference_model = payload.get("reference_model")
    if not isinstance(reference_model, str) or capacity_for_model(reference_model) is None:
        message = "P1/P2 enqueue refused: missing or unsupported selected reference_model"
        raise ValueError(message)
    selected = payload.get("selected_trials")
    if not isinstance(selected, dict) or set(selected) != set(EVIDENCE_MODELS):
        message = "P1/P2 enqueue refused: tuning must cover exactly the locked nine families"
        raise ValueError(message)
    locked: dict[str, dict[str, float | int]] = {}
    for family in EVIDENCE_MODELS:
        row = selected.get(family)
        if not isinstance(row, dict) or set(row) < {
            "trial",
            "refit_epochs",
            "learning_rate",
            "weight_decay",
        }:
            message = f"P1/P2 enqueue refused: malformed selected trial for {family}"
            raise ValueError(message)
        if int(row["trial"]) not in range(1, 7):
            message = f"P1/P2 enqueue refused: invalid trial index for {family}"
            raise ValueError(message)
        if int(row["refit_epochs"]) < 1:
            message = f"P1/P2 enqueue refused: invalid refit_epochs for {family}"
            raise ValueError(message)
        if float(row["learning_rate"]) <= 0.0 or float(row["weight_decay"]) < 0.0:
            message = f"P1/P2 enqueue refused: invalid optimizer values for {family}"
            raise ValueError(message)
        spec = confirmatory_trial_spec(family, int(row["trial"]))
        if (
            float(row["learning_rate"]) != spec.learning_rate
            or float(row["weight_decay"]) != spec.weight_decay
        ):
            message = f"P1/P2 enqueue refused: optimizer differs from trial contract for {family}"
            raise ValueError(message)
        if row.get("architecture") != confirmatory_implementation_metadata(
            family, int(row["trial"])
        ):
            message = f"P1/P2 enqueue refused: architecture differs from trial for {family}"
            raise ValueError(message)
        locked[family] = {
            "trial": int(row["trial"]),
            "refit_epochs": int(row["refit_epochs"]),
            "learning_rate": float(row["learning_rate"]),
            "weight_decay": float(row["weight_decay"]),
        }
    return locked, reference_model


_P0_METRICS = (
    "train_loss",
    "test_loss",
    "test_accuracy",
    "macro_f1",
    "weighted_f1",
    "balanced_accuracy",
)


def _read_p0_final_sources(
    config: P1P2Config,
    *,
    seeds: tuple[int, ...],
    datasets: tuple[str, ...],
    selected: dict[str, dict[str, float | int]],
    reference_model: str,
) -> dict[str, dict[str, str]]:
    result_path = config.unseen_root / "results" / "low_data_recommended_real.csv"
    if not result_path.exists():
        message = f"P1/P2 enqueue refused: missing P0 unseen-final results {result_path}"
        raise FileNotFoundError(message)
    expected = {
        f"low_data:unseen_final:{seed}:{family}:{dataset}:1.0"
        for seed in seeds
        for dataset in datasets
        for family in config.models
    }
    sources: dict[str, dict[str, str]] = {}
    for row in read_csv(result_path):
        key = row.get("job_key", "")
        if key not in expected or row.get("status") != "done":
            continue
        if key in sources:
            message = f"P1/P2 enqueue refused: duplicate terminal P0 row {key}"
            raise ValueError(message)
        _validate_p0_source_row(
            config,
            row,
            selected=selected,
            reference_model=reference_model,
        )
        sources[key] = row
    missing = sorted(expected - set(sources))
    if missing:
        message = (
            "P1/P2 enqueue refused: P0 unseen-final checkpoint set is incomplete; "
            f"missing={len(missing)} first={missing[0]}"
        )
        raise ValueError(message)
    return sources


def _validate_p0_source_row(  # noqa: C901, PLR0912, PLR0915 - provenance firewall
    config: P1P2Config,
    row: dict[str, str],
    *,
    selected: dict[str, dict[str, float | int]],
    reference_model: str,
) -> None:
    family = row.get("model", "")
    if family not in selected or row.get("reference_model") != reference_model:
        message = "P1/P2 enqueue refused: P0 family/reference provenance mismatch"
        raise ValueError(message)
    trial = int(row.get("validation_trial", "0") or 0)
    refit_epochs = int(row.get("refit_epochs", "0") or 0)
    if trial != int(selected[family]["trial"]) or refit_epochs != int(
        selected[family]["refit_epochs"]
    ):
        message = "P1/P2 enqueue refused: P0 trial/refit provenance mismatch"
        raise ValueError(message)
    if float(row.get("learning_rate", "nan")) != float(selected[family]["learning_rate"]) or float(
        row.get("weight_decay", "nan")
    ) != float(selected[family]["weight_decay"]):
        message = "P1/P2 enqueue refused: P0 optimizer provenance mismatch"
        raise ValueError(message)
    path = Path(row.get("checkpoint_path", ""))
    if not path.is_file():
        message = f"P1/P2 enqueue refused: missing P0 checkpoint {path}"
        raise FileNotFoundError(message)
    checkpoint_root = (config.unseen_root / "checkpoints").resolve()
    if not path.resolve().is_relative_to(checkpoint_root):
        message = f"P1/P2 enqueue refused: checkpoint escapes unseen root {path}"
        raise ValueError(message)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != row.get("checkpoint_sha256"):
        message = f"P1/P2 enqueue refused: checkpoint SHA-256 mismatch {path}"
        raise ValueError(message)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != (
        "pac_unseen_final_checkpoint.v1"
    ):
        message = f"P1/P2 enqueue refused: unsupported checkpoint payload {path}"
        raise ValueError(message)
    required = {
        "job_key": row.get("job_key"),
        "dataset": row.get("dataset_or_task"),
        "seed": int(row["seed"]),
        "family": family,
        "reference_model": reference_model,
        "validation_trial": trial,
        "refit_epochs": refit_epochs,
    }
    if any(payload.get(name) != value for name, value in required.items()):
        message = f"P1/P2 enqueue refused: checkpoint identity mismatch {path}"
        raise ValueError(message)
    if payload.get("architecture_metadata_json") != _architecture_json(family, trial):
        message = f"P1/P2 enqueue refused: checkpoint architecture mismatch {path}"
        raise ValueError(message)
    normalization = payload.get("normalization")
    if not isinstance(normalization, dict) or normalization.get("fit_split") != (
        "all_official_train"
    ):
        message = f"P1/P2 enqueue refused: invalid checkpoint normalization {path}"
        raise ValueError(message)
    if not isinstance(normalization.get("mean"), int | float) or not isinstance(
        normalization.get("std"), int | float
    ):
        message = f"P1/P2 enqueue refused: missing normalization moments {path}"
        raise TypeError(message)
    experiment = payload.get("experiment_config")
    if (
        not isinstance(experiment, dict)
        or experiment.get("epochs") != refit_epochs
        or float(payload.get("learning_rate", "nan")) != float(selected[family]["learning_rate"])
        or float(payload.get("weight_decay", "nan")) != float(selected[family]["weight_decay"])
    ):
        message = f"P1/P2 enqueue refused: checkpoint training config mismatch {path}"
        raise ValueError(message)
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict) or not state_dict:
        message = f"P1/P2 enqueue refused: checkpoint has no state_dict {path}"
        raise ValueError(message)
    for metric in _P0_METRICS:
        value = row.get(metric, "")
        if not value:
            message = f"P1/P2 enqueue refused: P0 row missing metric {metric}"
            raise ValueError(message)
    checkpoint_metrics = payload.get("p0_metrics")
    if not isinstance(checkpoint_metrics, dict) or any(
        float(checkpoint_metrics.get(metric, "nan")) != float(row[metric]) for metric in _P0_METRICS
    ):
        message = f"P1/P2 enqueue refused: checkpoint/result metric mismatch {path}"
        raise ValueError(message)


def _p0_job_fields(
    sources: dict[str, dict[str, str]],
    family: str,
    dataset: str,
    seed: int,
    ratio: float,
) -> tuple[str, str, str, str]:
    if ratio != 1.0:
        return "", "", "", ""
    key = f"low_data:unseen_final:{seed}:{family}:{dataset}:1.0"
    row = sources[key]
    metrics = {name: float(row[name]) for name in _P0_METRICS}
    return (
        key,
        row["checkpoint_path"],
        row["checkpoint_sha256"],
        json.dumps(metrics, sort_keys=True, separators=(",", ":")),
    )
