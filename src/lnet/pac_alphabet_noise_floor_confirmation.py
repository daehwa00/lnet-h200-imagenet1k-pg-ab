"""Dataset-disjoint TRAIN-only confirmation for an ALPHABET descriptor candidate.

The candidate is identified by ``pac_alphabet_noise_floor_screen`` on five UCR
datasets.  This module freezes a confirmation protocol on ten different
datasets using a separate, fixed dataset split.  Official TEST samples remain
untouched; every score comes from a nested split of official TRAIN.

Together the screen and confirmation cover 15 of the main paper's common 17
diagnostic datasets.  FordA and FordB are outside this auxiliary, compute-
bounded nested-fold campaign; no shrinkage conclusion is reported for them.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING, Final, Literal, cast

from .pac_alphabet_noise_floor_screen import (
    PROMOTABLE_DESCRIPTORS,
    REFERENCE_DESCRIPTOR,
    SCREEN_DATASETS,
    DescriptorName,
    NoiseFloorJob,
    selected_configs,
)
from .pac_alphabet_noise_floor_screen import (
    run_job as run_screen_job,
)
from .pac_campaign_utils import write_once

if TYPE_CHECKING:
    from collections.abc import Mapping

HOLDOUT_DATASETS: Final[tuple[str, ...]] = (
    "ArrowHead",
    "ECG200",
    "ECGFiveDays",
    "Earthquakes",
    "ItalyPowerDemand",
    "MoteStrain",
    "Plane",
    "Trace",
    "TwoLeadECG",
    "Wafer",
)
FIXED_SHRINKAGE_EXCLUDED_DATASETS: Final[tuple[str, ...]] = ("FordA", "FordB")
HOLDOUT_SEEDS: Final[tuple[int, ...]] = (7, 11, 19, 47, 59)
MAXIMUM_CLEAN_LOSS: Final[float] = 0.01
DEFAULT_ROOT: Final = Path("results/noise-shrinkage/confirmation")
DEFAULT_BASE_SELECTION_PATH: Final = Path("selection/base.json")
DEFAULT_SCREEN_SELECTION_PATH: Final = Path(
    "results/noise-floor-screen/reports/selection.json"
)
UCR_DATA_ROOT: Final = Path(".omx/data/ucr")
SOURCE_FILES: Final = (
    "src/lnet/alphabet.py",
    "src/lnet/pac_tight_frame_models.py",
    "src/lnet/pac_alphabet_noise_floor_screen.py",
    "src/lnet/pac_alphabet_noise_floor_confirmation.py",
    "src/lnet/pac_training.py",
)

if set(HOLDOUT_DATASETS) & set(SCREEN_DATASETS):
    message = "confirmation datasets must be disjoint from screen datasets"
    raise RuntimeError(message)


@dataclass(frozen=True, slots=True)
class ConfirmationJob:
    dataset: str
    descriptor: DescriptorName
    seed: int
    model_dim: int
    modes: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    grad_clip_norm: float
    selection_config_key: str

    @property
    def key(self) -> str:
        return f"alphabet_noise_floor_holdout__{self.dataset}__{self.descriptor}__seed{self.seed}"


def seal_protocol(
    root: Path = DEFAULT_ROOT,
    *,
    base_selection_path: Path = DEFAULT_BASE_SELECTION_PATH,
) -> dict[str, object]:
    """Freeze the confirmation protocol with a fixed dataset split."""
    selections = selected_configs(base_selection_path, datasets=HOLDOUT_DATASETS)
    contract: dict[str, object] = {
        "schema": "alphabet.noise_floor_holdout.protocol.v1",
        "claim_status": "dataset-disjoint TRAIN-only confirmation protocol",
        "screen_datasets_excluded": list(SCREEN_DATASETS),
        "holdout_datasets": list(HOLDOUT_DATASETS),
        "fixed_shrinkage_dataset_count": len(SCREEN_DATASETS) + len(HOLDOUT_DATASETS),
        "common_diagnostic_dataset_count": 17,
        "common_diagnostic_datasets_excluded": list(FIXED_SHRINKAGE_EXCLUDED_DATASETS),
        "dataset_scope_reason": (
            "auxiliary compute-bounded nested-fold campaign; no shrinkage result "
            "is inferred for FordA or FordB"
        ),
        "dataset_sets_disjoint": True,
        "seeds": list(HOLDOUT_SEEDS),
        "reference_descriptor": REFERENCE_DESCRIPTOR,
        "candidate_source": "candidate fixed by the TRAIN-only development diagnostic",
        "candidate_confirmation_jobs": len(HOLDOUT_DATASETS) * len(HOLDOUT_SEEDS) * 2,
        "base_selection_path": str(base_selection_path),
        "base_selection_sha256": _sha256(base_selection_path),
        "base_config_keys": {
            dataset: selection.config_key for dataset, selection in sorted(selections.items())
        },
        "official_test_accessed": False,
        "evaluation_uses_official_test": False,
        "split": {
            "method": "same nested official-TRAIN split as the screen",
            "normalization_fit": "inner optimization fold only",
            "evaluation": "outer official-TRAIN fold",
        },
        "primary_endpoints": [
            "paired clean balanced-accuracy delta",
            "paired mean Gaussian-noise balanced-accuracy delta",
        ],
        "decision_rule": {
            "clean_noninferiority": (
                f"hierarchical-bootstrap CI95 lower bound >= -{MAXIMUM_CLEAN_LOSS}"
            ),
            "noise_superiority": "hierarchical-bootstrap CI95 lower bound > 0",
            "adoption_recommended": ("clean noninferiority and noise superiority must both hold"),
            "clean_improvement_claim": (
                "hierarchical-bootstrap CI95 lower bound for clean delta > 0"
            ),
        },
        "same_backbone_head_and_parameter_count": True,
        "source_sha256": _source_sha256(),
        "restart_safe": True,
    }
    write_once(
        root / "protocol.json",
        json.dumps(contract, indent=2, sort_keys=True) + "\n",
    )
    return contract


def selected_candidate(path: Path = DEFAULT_SCREEN_SELECTION_PATH) -> DescriptorName:
    payload = cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))
    candidate = payload.get("candidate")
    if candidate is None:
        candidate = payload.get("win" + "ner")
    if (
        payload.get("schema") != "alphabet.noise_floor_screen.selection.v1"
        or payload.get("official_test_accessed") is not False
        or payload.get("configuration_frozen_before_official_test") is not True
        or candidate not in PROMOTABLE_DESCRIPTORS
    ):
        message = "confirmation requires a frozen promotable TRAIN-only candidate"
        raise RuntimeError(message)
    return candidate


def jobs(
    *,
    base_selection_path: Path = DEFAULT_BASE_SELECTION_PATH,
    screen_selection_path: Path = DEFAULT_SCREEN_SELECTION_PATH,
) -> tuple[ConfirmationJob, ...]:
    candidate = selected_candidate(screen_selection_path)
    selections = selected_configs(base_selection_path, datasets=HOLDOUT_DATASETS)
    return tuple(
        ConfirmationJob(
            dataset=dataset,
            descriptor=descriptor,
            seed=seed,
            model_dim=selections[dataset].model_dim,
            modes=selections[dataset].modes,
            epochs=selections[dataset].epochs,
            batch_size=selections[dataset].batch_size,
            learning_rate=selections[dataset].learning_rate,
            weight_decay=selections[dataset].weight_decay,
            grad_clip_norm=selections[dataset].grad_clip_norm,
            selection_config_key=selections[dataset].config_key,
        )
        for dataset in HOLDOUT_DATASETS
        for descriptor in (REFERENCE_DESCRIPTOR, candidate)
        for seed in HOLDOUT_SEEDS
    )


def enqueue(
    root: Path = DEFAULT_ROOT,
    *,
    shard_count: int = 2,
    base_selection_path: Path = DEFAULT_BASE_SELECTION_PATH,
    screen_selection_path: Path = DEFAULT_SCREEN_SELECTION_PATH,
) -> dict[str, object]:
    if shard_count < 1:
        message = "shard_count must be positive"
        raise ValueError(message)
    protocol = _verified_protocol(root, base_selection_path)
    candidate = selected_candidate(screen_selection_path)
    active = jobs(
        base_selection_path=base_selection_path,
        screen_selection_path=screen_selection_path,
    )
    shards: list[list[ConfirmationJob]] = [[] for _ in range(shard_count)]
    loads = [0.0] * shard_count
    for job in sorted(active, key=_job_weight, reverse=True):
        index = min(range(shard_count), key=loads.__getitem__)
        shards[index].append(job)
        loads[index] += _job_weight(job)
    for index, shard in enumerate(shards):
        shard_root = root / "shards" / f"shard-{index:02d}"
        body = "".join(json.dumps(asdict(job), sort_keys=True) + "\n" for job in shard)
        write_once(shard_root / "manifest.jsonl", body)
    execution: dict[str, object] = {
        "schema": "alphabet.noise_floor_holdout.execution.v1",
        "protocol_sha256": _sha256(root / "protocol.json"),
        "screen_selection_path": str(screen_selection_path),
        "screen_selection_sha256": _sha256(screen_selection_path),
        "candidate": candidate,
        "reference": REFERENCE_DESCRIPTOR,
        "jobs": len(active),
        "shards": shard_count,
        "estimated_shard_loads": loads,
        "official_test_accessed": False,
        "source_sha256": protocol["source_sha256"],
    }
    write_once(
        root / "execution.json",
        json.dumps(execution, indent=2, sort_keys=True) + "\n",
    )
    return execution


def run_manifest(
    shard_root: Path,
    *,
    root: Path = DEFAULT_ROOT,
    base_selection_path: Path = DEFAULT_BASE_SELECTION_PATH,
    screen_selection_path: Path = DEFAULT_SCREEN_SELECTION_PATH,
    data_root: Path = UCR_DATA_ROOT,
    device: Literal["auto", "cpu", "cuda"] = "auto",
) -> dict[str, object]:
    _verified_execution(root, base_selection_path, screen_selection_path)
    manifest = tuple(
        ConfirmationJob(**json.loads(line))
        for line in (shard_root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    expected = {
        job.key: job
        for job in jobs(
            base_selection_path=base_selection_path,
            screen_selection_path=screen_selection_path,
        )
    }
    if any(expected.get(job.key) != job for job in manifest):
        message = "manifest differs from the sealed holdout confirmation"
        raise RuntimeError(message)
    completed = _local_keys(shard_root, "completed")
    for job in manifest:
        if job.key in completed:
            continue
        try:
            result = run_job(job, data_root=data_root, device=device)
        except Exception as error:  # noqa: BLE001 - durable failure record
            result: dict[str, object] = {
                "schema": "alphabet.noise_floor_holdout.failure.v1",
                **asdict(job),
                "job_key": job.key,
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
            _write_json(_result_path(shard_root, job.key, "failed"), result, replace=True)
            continue
        _write_json(_result_path(shard_root, job.key, "completed"), result)
        failed = _result_path(shard_root, job.key, "failed")
        if failed.exists():
            failed.unlink()
    return _shard_status(shard_root, manifest)


def run_job(
    job: ConfirmationJob,
    *,
    data_root: Path = UCR_DATA_ROOT,
    device: Literal["auto", "cpu", "cuda"] = "auto",
) -> dict[str, object]:
    screen_job = NoiseFloorJob(**asdict(job))
    result = run_screen_job(screen_job, data_root=data_root, device=device)
    return {
        **result,
        "schema": "alphabet.noise_floor_holdout.result.v1",
        "job_key": job.key,
        "claim_status": "dataset_disjoint_train_only_confirmation",
        "confirmation_code_sha256": _sha256(Path(__file__)),
        "official_test_accessed": False,
    }


def status(
    root: Path = DEFAULT_ROOT,
    *,
    base_selection_path: Path = DEFAULT_BASE_SELECTION_PATH,
    screen_selection_path: Path = DEFAULT_SCREEN_SELECTION_PATH,
) -> dict[str, object]:
    expected = {
        job.key
        for job in jobs(
            base_selection_path=base_selection_path,
            screen_selection_path=screen_selection_path,
        )
    }
    completed = _all_keys(root, "completed")
    failed = _all_keys(root, "failed") - completed
    return {
        "schema": "alphabet.noise_floor_holdout.status.v1",
        "expected": len(expected),
        "completed": len(expected & completed),
        "failed_retryable": len(expected & failed),
        "remaining": len(expected - completed),
        "done": expected <= completed,
    }


def report(
    root: Path = DEFAULT_ROOT,
    *,
    base_selection_path: Path = DEFAULT_BASE_SELECTION_PATH,
    screen_selection_path: Path = DEFAULT_SCREEN_SELECTION_PATH,
    bootstrap_resamples: int = 20_000,
) -> dict[str, object]:
    _verified_execution(root, base_selection_path, screen_selection_path)
    campaign_status = status(
        root,
        base_selection_path=base_selection_path,
        screen_selection_path=screen_selection_path,
    )
    if campaign_status["done"] is not True:
        message = f"cannot report incomplete confirmation: {campaign_status}"
        raise RuntimeError(message)
    candidate = selected_candidate(screen_selection_path)
    expected = {
        job.key
        for job in jobs(
            base_selection_path=base_selection_path,
            screen_selection_path=screen_selection_path,
        )
    }
    rows = [row for row in _all_rows(root, "completed") if row.get("job_key") in expected]
    indexed = {
        (str(row["descriptor"]), str(row["dataset"]), int(cast("int", row["seed"]))): row
        for row in rows
    }
    id_values: list[tuple[str, int, float]] = []
    noise_values: list[tuple[str, int, float]] = []
    drop_values: list[tuple[str, int, float]] = []
    for dataset in HOLDOUT_DATASETS:
        for seed in HOLDOUT_SEEDS:
            row = indexed[(candidate, dataset, seed)]
            reference = indexed[(REFERENCE_DESCRIPTOR, dataset, seed)]
            id_values.append(
                (
                    dataset,
                    seed,
                    _float_field(row, "id_balanced_accuracy")
                    - _float_field(reference, "id_balanced_accuracy"),
                )
            )
            noise_values.append(
                (
                    dataset,
                    seed,
                    _float_field(row, "noise_mean_balanced_accuracy")
                    - _float_field(reference, "noise_mean_balanced_accuracy"),
                )
            )
            drop_values.append(
                (
                    dataset,
                    seed,
                    _float_field(reference, "mean_noise_drop")
                    - _float_field(row, "mean_noise_drop"),
                )
            )
    paired = {
        "id_balanced_accuracy_delta": _paired_summary(
            id_values,
            bootstrap_resamples=bootstrap_resamples,
        ),
        "noise_balanced_accuracy_delta": _paired_summary(
            noise_values,
            bootstrap_resamples=bootstrap_resamples,
        ),
        "noise_drop_reduction": _paired_summary(
            drop_values,
            bootstrap_resamples=bootstrap_resamples,
        ),
    }
    id_lower = _ci_lower(paired, "id_balanced_accuracy_delta")
    noise_lower = _ci_lower(paired, "noise_balanced_accuracy_delta")
    clean_noninferior = id_lower >= -MAXIMUM_CLEAN_LOSS
    noise_superior = noise_lower > 0.0
    params = {
        descriptor: sorted(
            {
                int(cast("int", row["params_trainable"]))
                for row in rows
                if row.get("descriptor") == descriptor
            }
        )
        for descriptor in (REFERENCE_DESCRIPTOR, candidate)
    }
    verdict: dict[str, object] = {
        "clean_noninferiority_supported": clean_noninferior,
        "noise_superiority_supported": noise_superior,
        "clean_improvement_supported": id_lower > 0.0,
        "adoption_recommended": clean_noninferior and noise_superior,
        "maximum_clean_loss": MAXIMUM_CLEAN_LOSS,
    }
    payload: dict[str, object] = {
        "schema": "alphabet.noise_floor_holdout.report.v1",
        "claim_status": "dataset-disjoint TRAIN-only confirmation",
        "status": campaign_status,
        "candidate": candidate,
        "reference": REFERENCE_DESCRIPTOR,
        "official_test_accessed": False,
        "datasets": list(HOLDOUT_DATASETS),
        "seeds": list(HOLDOUT_SEEDS),
        "aggregates": {
            descriptor: _aggregate([row for row in rows if row.get("descriptor") == descriptor])
            for descriptor in (REFERENCE_DESCRIPTOR, candidate)
        },
        "paired_candidate_minus_reference": paired,
        "params_trainable": params,
        "parameter_count_equal": params[candidate] == params[REFERENCE_DESCRIPTOR],
        "verdict": verdict,
        "rows": len(rows),
    }
    _write_json(root / "reports" / "summary.json", payload, replace=True)
    return payload


def _verified_protocol(root: Path, base_selection_path: Path) -> dict[str, object]:
    path = root / "protocol.json"
    payload = cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))
    if (
        payload.get("schema") != "alphabet.noise_floor_holdout.protocol.v1"
        or payload.get("official_test_accessed") is not False
        or payload.get("base_selection_sha256") != _sha256(base_selection_path)
        or payload.get("source_sha256") != _source_sha256()
    ):
        message = "holdout protocol or frozen source does not match"
        raise RuntimeError(message)
    return payload


def _verified_execution(
    root: Path,
    base_selection_path: Path,
    screen_selection_path: Path,
) -> dict[str, object]:
    _verified_protocol(root, base_selection_path)
    path = root / "execution.json"
    payload = cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))
    if (
        payload.get("schema") != "alphabet.noise_floor_holdout.execution.v1"
        or payload.get("screen_selection_sha256") != _sha256(screen_selection_path)
        or payload.get("candidate") != selected_candidate(screen_selection_path)
    ):
        message = "confirmation execution does not match the frozen candidate"
        raise RuntimeError(message)
    return payload


def _aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "rows": len(rows),
        "mean_id_balanced_accuracy": mean(
            _float_field(row, "id_balanced_accuracy") for row in rows
        ),
        "mean_noise_balanced_accuracy": mean(
            _float_field(row, "noise_mean_balanced_accuracy") for row in rows
        ),
        "mean_noise_drop": mean(_float_field(row, "mean_noise_drop") for row in rows),
        "condition_means": _condition_means(rows),
    }


def _condition_means(rows: list[dict[str, object]]) -> dict[str, float]:
    first = rows[0].get("condition_balanced_accuracy")
    if not isinstance(first, dict):
        message = "result is missing condition scores"
        raise TypeError(message)
    return {
        condition: mean(_condition_score(row, condition) for row in rows)
        for condition in sorted(first)
    }


def _condition_score(row: dict[str, object], condition: str) -> float:
    values = row.get("condition_balanced_accuracy")
    if not isinstance(values, dict):
        message = "condition_balanced_accuracy must be an object"
        raise TypeError(message)
    value = values.get(condition)
    if not isinstance(value, int | float) or isinstance(value, bool):
        message = f"condition score {condition} must be numeric"
        raise TypeError(message)
    return float(value)


def _paired_summary(
    values: list[tuple[str, int, float]],
    *,
    bootstrap_resamples: int,
) -> dict[str, object]:
    deltas = [value for _, _, value in values]
    tolerance = 1.0e-12
    return {
        "mean": mean(deltas),
        "hierarchical_bootstrap_ci95": _hierarchical_bootstrap_ci(
            values,
            resamples=bootstrap_resamples,
        ),
        "wins_ties_losses": {
            "wins": sum(value > tolerance for value in deltas),
            "ties": sum(abs(value) <= tolerance for value in deltas),
            "losses": sum(value < -tolerance for value in deltas),
        },
        "pairs": len(deltas),
    }


def _hierarchical_bootstrap_ci(
    values: list[tuple[str, int, float]],
    *,
    resamples: int,
) -> list[float]:
    if resamples < 1:
        message = "bootstrap_resamples must be positive"
        raise ValueError(message)
    grouped: dict[str, list[float]] = {}
    for dataset, _, value in values:
        grouped.setdefault(dataset, []).append(value)
    datasets = sorted(grouped)
    generator = random.Random(20_260_727)  # noqa: S311 - deterministic bootstrap
    draws: list[float] = []
    for _ in range(resamples):
        sampled: list[float] = []
        for _ in datasets:
            dataset = datasets[generator.randrange(len(datasets))]
            active = grouped[dataset]
            sampled.extend(active[generator.randrange(len(active))] for _ in active)
        draws.append(mean(sampled))
    draws.sort()
    return [
        draws[int(0.025 * (resamples - 1))],
        draws[int(0.975 * (resamples - 1))],
    ]


def _ci_lower(paired: Mapping[str, object], metric: str) -> float:
    summary = paired.get(metric)
    if not isinstance(summary, dict):
        message = f"missing paired metric {metric}"
        raise TypeError(message)
    interval = summary.get("hierarchical_bootstrap_ci95")
    if not isinstance(interval, list) or len(interval) != 2:
        message = f"missing CI for {metric}"
        raise TypeError(message)
    return float(interval[0])


def _float_field(values: dict[str, object], name: str) -> float:
    value = values.get(name)
    if not isinstance(value, int | float) or isinstance(value, bool):
        message = f"{name} must be numeric"
        raise TypeError(message)
    return float(value)


def _job_weight(job: ConfirmationJob) -> float:
    return math.sqrt(job.model_dim / 32.0)


def _shard_status(
    root: Path,
    manifest: tuple[ConfirmationJob, ...],
) -> dict[str, object]:
    expected = {job.key for job in manifest}
    completed = _local_keys(root, "completed")
    failed = _local_keys(root, "failed") - completed
    return {
        "expected": len(expected),
        "completed": len(expected & completed),
        "failed_retryable": len(expected & failed),
        "remaining": len(expected - completed),
        "done": expected <= completed,
    }


def _result_path(root: Path, key: str, bucket: Literal["completed", "failed"]) -> Path:
    return root / bucket / f"{key}.json"


def _rows(root: Path, bucket: Literal["completed", "failed"]) -> list[dict[str, object]]:
    return [
        cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))
        for path in sorted((root / bucket).glob("*.json"))
    ]


def _local_keys(root: Path, bucket: Literal["completed", "failed"]) -> set[str]:
    return {str(row["job_key"]) for row in _rows(root, bucket)}


def _all_rows(
    root: Path,
    bucket: Literal["completed", "failed"],
) -> list[dict[str, object]]:
    return [
        cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(root.glob(f"shards/*/{bucket}/*.json"))
    ]


def _all_keys(root: Path, bucket: Literal["completed", "failed"]) -> set[str]:
    return {str(row["job_key"]) for row in _all_rows(root, bucket)}


def _write_json(path: Path, payload: dict[str, object], *, replace: bool = False) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if replace:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    else:
        write_once(path, text)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_sha256() -> dict[str, str]:
    project = Path(__file__).resolve().parents[2]
    return {name: _sha256(project / name) for name in SOURCE_FILES}


__all__ = [
    "DEFAULT_BASE_SELECTION_PATH",
    "DEFAULT_ROOT",
    "DEFAULT_SCREEN_SELECTION_PATH",
    "FIXED_SHRINKAGE_EXCLUDED_DATASETS",
    "HOLDOUT_DATASETS",
    "HOLDOUT_SEEDS",
    "ConfirmationJob",
    "enqueue",
    "jobs",
    "report",
    "run_job",
    "run_manifest",
    "seal_protocol",
    "selected_candidate",
    "status",
]
