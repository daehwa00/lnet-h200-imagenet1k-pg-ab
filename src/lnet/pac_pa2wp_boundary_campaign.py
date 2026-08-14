from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from statistics import mean, stdev
from typing import Final, cast

from .pac_tf_p1p2_types import P1P2Job

DEFAULT_ROOT: Final = Path(".omx/results/pac-pa2wp-boundary-20260713")
WP_REFERENCE_ROOT: Final = Path(
    ".omx/results/pac-wp-evidence-stability-rerun-20260712/p1p2-training-shards"
)
MODEL: Final = "pa2wp_pac"
REFERENCE_MODEL: Final = "pac_headroom_phase_augmented_ensemble_wp_d64_m16"
SEEDS: Final = (7, 11, 19, 23, 31)
LOW_DATASETS: Final = (
    "CinCECGTorso",
    "CricketX",
    "Earthquakes",
    "Phoneme",
    "StarLightCurves",
)
LOW_RATIOS: Final = (0.01, 0.05, 0.10, 0.25, 0.50)


def pa2wp_boundary_jobs() -> tuple[P1P2Job, ...]:
    jobs = [
        P1P2Job(
            key=f"pa2wp_boundary:low_data:{dataset}:ratio{ratio:g}:seed{seed}",
            package="low_data",
            seed=seed,
            model=MODEL,
            reference_model=REFERENCE_MODEL,
            dataset=dataset,
            ratio=ratio,
            slots=2,
            learning_rate=3.0e-3,
            weight_decay=1.0e-4,
            ratio_one_fit_policy="optimization_fold_validation",
        )
        for dataset in LOW_DATASETS
        for ratio in LOW_RATIOS
        for seed in SEEDS
    ]
    jobs.extend(
        P1P2Job(
            key=f"pa2wp_boundary:real_diagnostics:{dataset}:seed{seed}",
            package="real_diagnostics",
            seed=seed,
            model=MODEL,
            reference_model=REFERENCE_MODEL,
            dataset=dataset,
            ratio=1.0,
            slots=2,
            learning_rate=3.0e-3,
            weight_decay=1.0e-4,
            ratio_one_fit_policy="optimization_fold_validation",
        )
        for dataset in LOW_DATASETS
        for seed in SEEDS
    )
    jobs.extend(
        P1P2Job(
            key=f"pa2wp_boundary:synthetic_ood:seed{seed}",
            package="synthetic_ood",
            seed=seed,
            model=MODEL,
            reference_model=REFERENCE_MODEL,
            slots=2,
            learning_rate=3.0e-3,
            weight_decay=1.0e-4,
            ratio_one_fit_policy="optimization_fold_validation",
        )
        for seed in SEEDS
    )
    jobs.extend(
        P1P2Job(
            key=f"pa2wp_boundary:real_domain_ood:mit_bih:seed{seed}",
            package="real_domain_ood",
            seed=seed,
            model=MODEL,
            reference_model=REFERENCE_MODEL,
            dataset="mit-bih-ds1-ds2",
            slots=2,
            learning_rate=3.0e-3,
            weight_decay=1.0e-4,
            ratio_one_fit_policy="optimization_fold_validation",
        )
        for seed in SEEDS
    )
    return tuple(jobs)


def enqueue_pa2wp_boundary(
    root: Path = DEFAULT_ROOT,
    *,
    shard_count: int = 6,
) -> dict[str, object]:
    if shard_count < 1:
        message = "shard_count must be positive"
        raise ValueError(message)
    jobs = pa2wp_boundary_jobs()
    shards: list[list[P1P2Job]] = [[] for _ in range(shard_count)]
    loads = [0.0] * shard_count
    for job in sorted(jobs, key=_job_weight, reverse=True):
        index = min(range(shard_count), key=loads.__getitem__)
        shards[index].append(job)
        loads[index] += _job_weight(job)
    shard_root = root / "shards"
    for index, shard in enumerate(shards):
        active = shard_root / f"shard-{index:02d}"
        active.mkdir(parents=True, exist_ok=True)
        (active / "p1p2_manifest.jsonl").write_text(
            "".join(json.dumps(asdict(job), sort_keys=True) + "\n" for job in shard),
            encoding="utf-8",
        )
    contract: dict[str, object] = {
        "schema": "pac_pa2wp_boundary.v1",
        "model": MODEL,
        "reference_model": REFERENCE_MODEL,
        "selection_source": "split-aligned five-seed UCR-18 validation",
        "selection_objective": "top-ranked dataset count",
        "architecture_frozen": True,
        "ratio_one_fit_policy": "optimization_fold_validation",
        "seeds": list(SEEDS),
        "low_data_datasets": list(LOW_DATASETS),
        "low_data_ratios": list(LOW_RATIOS),
        "low_data_jobs": sum(job.package == "low_data" for job in jobs),
        "real_corruption_ood_jobs": sum(job.package == "real_diagnostics" for job in jobs),
        "synthetic_ood_jobs": sum(job.package == "synthetic_ood" for job in jobs),
        "patient_disjoint_ood_jobs": sum(job.package == "real_domain_ood" for job in jobs),
        "jobs": len(jobs),
        "shards": shard_count,
        "estimated_shard_loads": loads,
        "official_test_accessed_at_enqueue": False,
        "test_policy": (
            "fixed PA2WP evaluated once per prespecified seed/ratio; no TEST-driven tuning"
        ),
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return contract


def pa2wp_boundary_status(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    expected = {job.key for job in pa2wp_boundary_jobs()}
    rows = _campaign_rows(root)
    done = {str(row["job_key"]) for row in rows if row.get("status") == "done"}
    failed = {str(row["job_key"]) for row in rows if row.get("status") == "failed"} - done
    return {
        "expected": len(expected),
        "completed": len(expected & done),
        "failed": len(expected & failed),
        "remaining": len(expected - done - failed),
        "done": expected <= done,
    }


def write_pa2wp_boundary_report(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    rows = [row for row in _campaign_rows(root) if row.get("status") == "done"]
    reference = [row for row in _reference_rows() if row.get("status") == "done"]
    low_data = _low_data_summary(rows, reference)
    corruption = _corruption_summary(rows, reference)
    domain = _patient_disjoint_summary(rows, reference)
    synthetic = _synthetic_summary(rows, reference)
    payload: dict[str, object] = {
        "schema": "pac_pa2wp_boundary_report.v1",
        "status": pa2wp_boundary_status(root),
        "low_data": low_data,
        "real_corruption_ood": corruption,
        "patient_disjoint_ood": domain,
        "synthetic_ood": synthetic,
        "comparison_reference": str(WP_REFERENCE_ROOT),
    }
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "PA2WP_BOUNDARY.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def _low_data_summary(
    rows: list[dict[str, str]], reference: list[dict[str, str]]
) -> dict[str, object]:
    cells: dict[str, object] = {}
    deltas: list[float] = []
    wins = ties = losses = 0
    for dataset in LOW_DATASETS:
        for ratio in LOW_RATIOS:
            active = _values(rows, "low_data", "balanced_accuracy", dataset, ratio)
            baseline = _values(reference, "low_data", "balanced_accuracy", dataset, ratio)
            active_mean = mean(active) if active else None
            baseline_mean = mean(baseline) if baseline else None
            delta = (
                None
                if active_mean is None or baseline_mean is None
                else active_mean - baseline_mean
            )
            if delta is not None:
                deltas.append(delta)
                if delta > 1.0e-12:
                    wins += 1
                elif delta < -1.0e-12:
                    losses += 1
                else:
                    ties += 1
            cells[f"{dataset}:ratio{ratio:g}"] = {
                "pa2wp_mean": active_mean,
                "wp_mean": baseline_mean,
                "delta": delta,
                "pa2wp_seeds": len(active),
                "wp_seeds": len(baseline),
            }
    by_ratio: dict[str, object] = {}
    for ratio in LOW_RATIOS:
        active = [
            value
            for dataset in LOW_DATASETS
            for value in _values(rows, "low_data", "balanced_accuracy", dataset, ratio)
        ]
        baseline = [
            value
            for dataset in LOW_DATASETS
            for value in _values(reference, "low_data", "balanced_accuracy", dataset, ratio)
        ]
        by_ratio[f"{ratio:g}"] = _paired_descriptive(active, baseline, direction="max")
    return {
        "cells": cells,
        "by_ratio": by_ratio,
        "mean_cell_delta": mean(deltas) if deltas else None,
        "wins": wins,
        "ties": ties,
        "losses": losses,
    }


def _corruption_summary(
    rows: list[dict[str, str]],
    reference: list[dict[str, str]],
) -> dict[str, object]:
    active = _json_metric_by_condition(
        rows,
        package="real_diagnostics",
        field="real_corruption_ood_json",
        condition_fields=("shift",),
        metric="absolute_accuracy_drop",
    )
    baseline = _json_metric_by_condition(
        reference,
        package="real_diagnostics",
        field="real_corruption_ood_json",
        condition_fields=("shift",),
        metric="absolute_accuracy_drop",
    )
    conditions = {
        condition: _paired_descriptive(
            active.get(condition, []),
            baseline.get(condition, []),
            direction="min",
        )
        for condition in sorted(set(active) | set(baseline))
    }
    deltas: list[float] = []
    for condition, row in conditions.items():
        delta = row["delta"]
        if condition != "id" and isinstance(delta, int | float):
            deltas.append(float(delta))
    return {
        "metric": "absolute_accuracy_drop",
        "direction": "min",
        "conditions": conditions,
        "mean_condition_delta_excluding_id": mean(deltas) if deltas else None,
    }


def _patient_disjoint_summary(
    rows: list[dict[str, str]], reference: list[dict[str, str]]
) -> dict[str, object]:
    metrics = (
        ("id_common_accuracy", "max"),
        ("ood_common_accuracy", "max"),
        ("id_common_balanced_accuracy", "max"),
        ("ood_common_balanced_accuracy", "max"),
        ("absolute_common_accuracy_drop", "min"),
        ("absolute_common_balanced_accuracy_drop", "min"),
        ("ood_full_5class_balanced_accuracy", "max"),
    )
    return {
        "metrics": {
            metric: _paired_descriptive(
                _package_metric(rows, "real_domain_ood", metric),
                _package_metric(reference, "real_domain_ood", metric),
                direction=direction,
            )
            for metric, direction in metrics
        }
    }


def _synthetic_summary(
    rows: list[dict[str, str]], reference: list[dict[str, str]]
) -> dict[str, object]:
    active_by_condition = _json_metric_by_condition(
        rows,
        package="synthetic_ood",
        field="ood_sweep_json",
        condition_fields=("family", "level"),
        metric="absolute_nrmse_increase",
    )
    baseline_by_condition = _json_metric_by_condition(
        reference,
        package="synthetic_ood",
        field="ood_sweep_json",
        condition_fields=("family", "level"),
        metric="absolute_nrmse_increase",
    )
    active = [value for values in active_by_condition.values() for value in values]
    baseline = [value for values in baseline_by_condition.values() for value in values]
    active_mean = mean(active) if active else None
    baseline_mean = mean(baseline) if baseline else None
    return {
        "metric": "mean absolute_nrmse_increase over OOD conditions",
        "direction": "min",
        "pa2wp_mean": active_mean,
        "wp_mean": baseline_mean,
        "delta": (
            None if active_mean is None or baseline_mean is None else active_mean - baseline_mean
        ),
        "pa2wp_conditions": len(active),
        "wp_conditions": len(baseline),
        "conditions": {
            condition: _paired_descriptive(
                active_by_condition.get(condition, []),
                baseline_by_condition.get(condition, []),
                direction="min",
            )
            for condition in sorted(set(active_by_condition) | set(baseline_by_condition))
        },
    }


def _values(
    rows: list[dict[str, str]],
    package: str,
    metric: str,
    dataset: str,
    ratio: float,
) -> list[float]:
    return [
        float(row[metric])
        for row in rows
        if row.get("package") == package
        and row.get("dataset_or_task") == dataset
        and abs(float(row.get("data_ratio", "nan")) - ratio) < 1.0e-12
        and row.get(metric) not in {None, ""}
    ]


def _package_metric(rows: list[dict[str, str]], package: str, metric: str) -> list[float]:
    return [
        float(row[metric])
        for row in rows
        if row.get("package") == package and row.get(metric) not in {None, ""}
    ]


def _json_metric_by_condition(
    rows: list[dict[str, str]],
    *,
    package: str,
    field: str,
    condition_fields: tuple[str, ...],
    metric: str,
) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        if row.get("package") != package or not row.get(field):
            continue
        items = cast("list[dict[str, object]]", json.loads(row[field]))
        for item in items:
            value = item.get(metric)
            parts = [item.get(name) for name in condition_fields]
            if not isinstance(value, int | float) or not all(
                isinstance(part, str) for part in parts
            ):
                continue
            condition = "=".join(cast("list[str]", parts))
            grouped.setdefault(condition, []).append(float(value))
    return grouped


def _paired_descriptive(
    active: list[float], baseline: list[float], *, direction: str
) -> dict[str, object]:
    active_mean = mean(active) if active else None
    baseline_mean = mean(baseline) if baseline else None
    return {
        "direction": direction,
        "pa2wp_mean": active_mean,
        "pa2wp_sample_sd": stdev(active) if len(active) > 1 else None,
        "pa2wp_rows": len(active),
        "wp_mean": baseline_mean,
        "wp_sample_sd": stdev(baseline) if len(baseline) > 1 else None,
        "wp_rows": len(baseline),
        "delta": (
            None if active_mean is None or baseline_mean is None else active_mean - baseline_mean
        ),
    }


def _campaign_rows(root: Path) -> list[dict[str, str]]:
    return _csv_rows(root / "shards")


def _reference_rows() -> list[dict[str, str]]:
    return _csv_rows(WP_REFERENCE_ROOT)


def _csv_rows(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not root.exists():
        return rows
    for path in root.glob("**/results/*.csv"):
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(dict(row) for row in csv.DictReader(handle))
    return rows


def _job_weight(job: P1P2Job) -> float:
    if job.package == "real_domain_ood":
        return 12.0
    if job.package == "synthetic_ood":
        return 8.0
    dataset_weight = {
        "StarLightCurves": 5.0,
        "CricketX": 3.0,
        "CinCECGTorso": 2.0,
        "Phoneme": 2.0,
    }.get(job.dataset, 1.0)
    if job.package == "real_diagnostics":
        return 2.0 * dataset_weight
    return dataset_weight * max(0.25, job.ratio)
