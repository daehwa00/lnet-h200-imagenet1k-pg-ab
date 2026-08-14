#!/usr/bin/env python3
# ruff: noqa: C901, EM101, EM102, PLR0912, PLR0915, T201, TC003, TRY003
"""Independently recompute the frozen ALPHABET secondary evidence.

This deliberately does not import any paper table generator.  It reads the
completed result ledgers, reconstructs the reported estimands, and fails if a
paper summary differs from the raw measurements or has incomplete coverage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from statistics import geometric_mean, mean, median, stdev
from typing import Any

RTOL = 1.0e-5
ATOL = 1.0e-8
MODELS = (
    "compact_h_only",
    "cnn1d",
    "tcn",
    "mamba",
    "gru",
    "lstm",
    "transformer",
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=1.0e-11, abs_tol=1.0e-12):
        raise RuntimeError(f"{label}: {actual!r} != {expected!r}")


def _unique(rows: Iterable[dict[str, Any]], keys: tuple[str, ...], label: str) -> None:
    materialized = list(rows)
    identities = [tuple(row[key] for key in keys) for row in materialized]
    if len(identities) != len(set(identities)):
        raise RuntimeError(f"{label}: duplicate result identity")


def _average_ranks(scores: dict[str, float], *, lower: bool) -> dict[str, float]:
    ordered = sorted(scores, key=lambda key: scores[key], reverse=not lower)
    result: dict[str, float] = {}
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and math.isclose(
            scores[ordered[cursor]], scores[ordered[end]], rel_tol=RTOL, abs_tol=ATOL
        ):
            end += 1
        rank = ((cursor + 1) + end) / 2.0
        for key in ordered[cursor:end]:
            result[key] = rank
        cursor = end
    return result


def _top(scores: dict[str, float], *, lower: bool) -> set[str]:
    best = min(scores.values()) if lower else max(scores.values())
    return {
        key
        for key, value in scores.items()
        if math.isclose(value, best, rel_tol=RTOL, abs_tol=ATOL)
    }


def _audit_ablation(repo: Path) -> dict[str, Any]:
    root = repo / ".omx/results/pac-compact-h-only-ablation-20260719"
    summary_path = repo / "paper/generated/alphabet_ablation_summary.json"
    summary = _load(summary_path)
    rows = [_load(path) for path in sorted((root / "completed").glob("*.json"))]
    datasets = tuple(summary["datasets"])
    variants = tuple(summary["variants"])
    seeds = tuple(summary["seeds"])
    expected = {(d, v, s) for d in datasets for v in variants for s in seeds}
    actual = {(row["dataset"], row["variant"], row["train_seed"]) for row in rows}
    if actual != expected or len(rows) != 630:
        raise RuntimeError(f"ablation coverage mismatch: {len(actual)}/{len(expected)}")
    _unique(rows, ("dataset", "variant", "train_seed"), "ablation")
    for row in rows:
        if (
            row.get("status") != "done"
            or row.get("evaluation_split") != "validation"
            or row.get("official_test_accessed") is not False
            or row.get("test_evaluated") is not False
            or int(row.get("test_count", -1)) != 0
        ):
            raise RuntimeError(f"ablation split/status contract failed: {row.get('key')}")

    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    params: dict[str, list[int]] = defaultdict(list)
    deltas: dict[str, list[float]] = defaultdict(list)
    by_identity = {(r["dataset"], r["train_seed"], r["variant"]): r for r in rows}
    for row in rows:
        values[(row["dataset"], row["variant"])].append(float(row["validation_balanced_accuracy"]))
        params[row["variant"]].append(int(row["params_trainable"]))
        full = by_identity[(row["dataset"], row["train_seed"], "full")]
        deltas[row["variant"]].append(
            float(row["validation_balanced_accuracy"]) - float(full["validation_balanced_accuracy"])
        )

    ranks: dict[str, list[float]] = defaultdict(list)
    tops: dict[str, int] = defaultdict(int)
    for dataset in datasets:
        scores = {variant: mean(values[(dataset, variant)]) for variant in variants}
        cell_ranks = _average_ranks(scores, lower=False)
        cell_top = _top(scores, lower=False)
        published = next(cell for cell in summary["cells"] if cell["dataset"] == dataset)
        for variant in variants:
            _close(scores[variant], published["mean"][variant], f"ablation {dataset}/{variant}")
            _close(
                cell_ranks[variant],
                published["average_tie_rank"][variant],
                f"ablation rank {dataset}/{variant}",
            )
            ranks[variant].append(cell_ranks[variant])
            tops[variant] += int(variant in cell_top)
        if cell_top != set(published["joint_top1"]):
            raise RuntimeError(f"ablation top mismatch: {dataset}")

    for variant in variants:
        all_values = [value for d in datasets for value in values[(d, variant)]]
        row = summary["aggregate"][variant]
        checks = {
            "mean_balanced_accuracy": mean(all_values),
            "sample_sd_balanced_accuracy": stdev(all_values),
            "mean_paired_delta_vs_full": mean(deltas[variant]),
            "sample_sd_paired_delta_vs_full": stdev(deltas[variant]),
            "median_params_trainable": median(params[variant]),
            "min_params_trainable": min(params[variant]),
            "max_params_trainable": max(params[variant]),
            "mean_rank_18": mean(ranks[variant]),
            "top1_18": tops[variant],
        }
        for key, value in checks.items():
            _close(value, row[key], f"ablation aggregate {variant}/{key}")
    return {
        "status": "PASS",
        "raw_rows": len(rows),
        "summary_sha256": _sha256(summary_path),
        "headline": {
            "full_mean_bacc": summary["aggregate"]["full"]["mean_balanced_accuracy"],
            "no_terminal_mean_bacc": summary["aggregate"]["no_terminal_reader"][
                "mean_balanced_accuracy"
            ],
        },
    }


def _audit_boundary(repo: Path) -> dict[str, Any]:
    root = repo / ".omx/results/pac-compact-h-only-boundary-20260719"
    summary_path = repo / "paper/generated/boundary_final_summary.json"
    summary = _load(summary_path)
    paths = sorted((root / "shards").glob("shard-*/completed/*.json"))
    rows = [_load(path) for path in paths]
    _unique(rows, ("key",), "boundary")
    datasets = tuple(summary["audit"]["datasets"])
    ratios = tuple(float(x) for x in summary["audit"]["ratios"])
    seeds = tuple(int(x) for x in summary["audit"]["seeds"])
    shifts = tuple(summary["audit"]["shifts"])
    low = [row for row in rows if row["package"] == "low_data"]
    diagnostic = [row for row in rows if row["package"] == "real_diagnostics"]
    low_expected = {(d, r, m, s) for d in datasets for r in ratios for m in MODELS for s in seeds}
    low_actual = {(r["dataset"], float(r["ratio"]), r["model"], int(r["seed"])) for r in low}
    diagnostic_expected = {(d, m, s) for d in datasets for m in MODELS for s in seeds}
    diagnostic_actual = {(r["dataset"], r["model"], int(r["seed"])) for r in diagnostic}
    if low_actual != low_expected or diagnostic_actual != diagnostic_expected:
        raise RuntimeError("boundary Cartesian coverage mismatch")
    for row in rows:
        if (
            row.get("status") != "done"
            or row.get("selection_test_evidence_used") is not False
            or row.get("evaluation_split") != "official_test"
        ):
            raise RuntimeError(f"boundary contract failed: {row.get('key')}")

    low_values: dict[tuple[str, float, str], list[float]] = defaultdict(list)
    for row in low:
        low_values[(row["dataset"], float(row["ratio"]), row["model"])].append(
            float(row["balanced_accuracy"])
        )
    low_ranks: dict[str, list[float]] = defaultdict(list)
    low_top: dict[str, int] = defaultdict(int)
    low_all: dict[str, list[float]] = defaultdict(list)
    for dataset in datasets:
        for ratio in ratios:
            scores = {model: mean(low_values[(dataset, ratio, model)]) for model in MODELS}
            ranks = _average_ranks(scores, lower=False)
            top = _top(scores, lower=False)
            for model in MODELS:
                low_ranks[model].append(ranks[model])
                low_top[model] += int(model in top)
                low_all[model].extend(low_values[(dataset, ratio, model)])
    for model in MODELS:
        published = summary["low_data"]["aggregate"][model]
        for key, value in {
            "mean_score": mean(low_all[model]),
            "mean_rank": mean(low_ranks[model]),
            "top1_count": low_top[model],
            "observations": len(low_all[model]),
            "analysis_cells": len(low_ranks[model]),
        }.items():
            _close(value, published[key], f"boundary low {model}/{key}")

    corrupt_values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    corrupt_drops: dict[str, list[float]] = defaultdict(list)
    for row in diagnostic:
        parsed = json.loads(row["real_corruption_ood_json"])
        if {item["shift"] for item in parsed} != set(shifts):
            raise RuntimeError(f"boundary shift coverage failed: {row['key']}")
        for item in parsed:
            if item["shift"] == "id":
                continue
            corrupt_values[(row["dataset"], item["shift"], row["model"])].append(
                float(item["accuracy"])
            )
            corrupt_drops[row["model"]].append(float(item["absolute_accuracy_drop"]))
    corrupt_ranks: dict[str, list[float]] = defaultdict(list)
    corrupt_top: dict[str, int] = defaultdict(int)
    corrupt_all: dict[str, list[float]] = defaultdict(list)
    for dataset in datasets:
        for shift in shifts:
            if shift == "id":
                continue
            scores = {model: mean(corrupt_values[(dataset, shift, model)]) for model in MODELS}
            ranks = _average_ranks(scores, lower=False)
            top = _top(scores, lower=False)
            for model in MODELS:
                corrupt_ranks[model].append(ranks[model])
                corrupt_top[model] += int(model in top)
                corrupt_all[model].extend(corrupt_values[(dataset, shift, model)])
    for model in MODELS:
        published = summary["corruption"]["aggregate_non_id"][model]
        for key, value in {
            "mean_score": mean(corrupt_all[model]),
            "mean_absolute_accuracy_drop": mean(corrupt_drops[model]),
            "absolute_accuracy_drop_sample_sd": stdev(corrupt_drops[model]),
            "mean_rank": mean(corrupt_ranks[model]),
            "top1_count": corrupt_top[model],
            "observations": len(corrupt_all[model]),
            "analysis_cells": len(corrupt_ranks[model]),
        }.items():
            _close(value, published[key], f"boundary corruption {model}/{key}")
    return {
        "status": "PASS",
        "raw_rows": len(rows),
        "low_data_rows": len(low),
        "diagnostic_rows": len(diagnostic),
        "derived_corruption_rows": sum(len(v) for v in corrupt_all.values()),
        "summary_sha256": _sha256(summary_path),
        "headline": {
            "alphabet_low_data_rank": summary["low_data"]["aggregate"]["compact_h_only"][
                "mean_rank"
            ],
            "alphabet_corruption_rank": summary["corruption"]["aggregate_non_id"]["compact_h_only"][
                "mean_rank"
            ],
        },
    }


def _audit_synthetic_ood(repo: Path) -> dict[str, Any]:
    root = repo / ".omx/results/pac-compact-h-only-synthetic-ood-20260719"
    summary_path = repo / "paper/generated/synthetic_ood_baseline_summary.json"
    summary = _load(summary_path)
    rows = [_load(path) for path in sorted((root / "completed").glob("*.json"))]
    seeds = tuple(int(x) for x in summary["seeds"])
    expected = {(model, seed) for model in MODELS for seed in seeds}
    actual = {(row["model"], int(row["seed"])) for row in rows}
    if actual != expected or len(rows) != 35:
        raise RuntimeError("synthetic OOD fit coverage mismatch")
    _unique(rows, ("model", "seed"), "synthetic OOD")
    conditions: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    seed_conditions: dict[tuple[str, int, str, str, str], float] = {}
    id_values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if (
            row.get("status") != "done"
            or row.get("fit_test_tensors_empty") is not True
            or row.get("ood_constructed_after_fit") is not True
            or len(row["conditions"]) != 95
        ):
            raise RuntimeError(f"synthetic OOD contract failed: {row.get('job_key')}")
        id_values[row["model"]].append(float(row["id_test_nrmse"]))
        for item in row["conditions"]:
            conditions[(item["variant"], item["family"], str(item["level"]), row["model"])].append(
                float(item["nrmse"])
            )
            seed_key = (
                row["model"],
                int(row["seed"]),
                item["variant"],
                item["family"],
                str(item["level"]),
            )
            if seed_key in seed_conditions:
                raise RuntimeError(f"duplicate synthetic OOD seed condition: {seed_key}")
            seed_conditions[seed_key] = float(item["nrmse"])
    if sum(len(row["conditions"]) for row in rows) != 3325:
        raise RuntimeError("synthetic OOD condition count mismatch")
    for model in MODELS:
        _close(
            mean(id_values[model]),
            summary["id_test_nrmse"][model]["mean"],
            f"OOD ID {model}",
        )
        _close(
            stdev(id_values[model]),
            summary["id_test_nrmse"][model]["sample_sd"],
            f"OOD ID SD {model}",
        )

    for variant, published_variant in summary["variants"].items():
        profiles = published_variant["profiles"]
        rank_values: dict[str, list[float]] = defaultdict(list)
        top_counts: dict[str, int] = defaultdict(int)
        family_means: dict[tuple[str, str], list[float]] = defaultdict(list)
        for profile in profiles:
            family, level = profile["family"], str(profile["level"])
            scores = {model: mean(conditions[(variant, family, level, model)]) for model in MODELS}
            ranks = _average_ranks(scores, lower=True)
            top = _top(scores, lower=True)
            if top != set(profile["joint_top1"]):
                raise RuntimeError(f"synthetic OOD top mismatch: {variant}/{family}/{level}")
            for model in MODELS:
                _close(
                    scores[model],
                    profile["mean_nrmse"][model],
                    f"OOD profile {variant}/{family}/{level}/{model}",
                )
                _close(
                    ranks[model],
                    profile["average_tie_rank"][model],
                    f"OOD rank {variant}/{family}/{level}/{model}",
                )
                rank_values[model].append(ranks[model])
                top_counts[model] += int(model in top)
                family_means[(family, model)].append(scores[model])
        families = sorted({profile["family"] for profile in profiles})
        for model in MODELS:
            equal_family = mean(mean(family_means[(family, model)]) for family in families)
            published = published_variant["models"][model]
            _close(
                equal_family,
                published["equal_family_mean_nrmse"],
                f"OOD family mean {variant}/{model}",
            )
            _close(
                mean(rank_values[model]),
                published["mean_rank_19"],
                f"OOD mean rank {variant}/{model}",
            )
            _close(top_counts[model], published["top1_19"], f"OOD top count {variant}/{model}")
            seed_macros = [
                mean(
                    mean(
                        seed_conditions[(model, seed, variant, family, str(profile["level"]))]
                        for profile in profiles
                        if profile["family"] == family
                    )
                    for family in families
                )
                for seed in (23, 31, 43, 47, 59)
            ]
            _close(
                stdev(seed_macros),
                summary["variant_equal_family_sample_sd"][variant][model],
                f"OOD equal-family SD {variant}/{model}",
            )
            if variant == "correct_dt_mask":
                for family in families:
                    _close(
                        mean(family_means[(family, model)]),
                        summary["primary_by_family"][family][model],
                        f"OOD primary family {family}/{model}",
                    )
                _close(
                    stdev(seed_macros),
                    summary["primary_equal_family_sample_sd"][model],
                    f"OOD equal-family SD {model}",
                )
                for profile in profiles:
                    family, level = profile["family"], str(profile["level"])
                    _close(
                        stdev(
                            seed_conditions[(model, seed, variant, family, level)]
                            for seed in (23, 31, 43, 47, 59)
                        ),
                        summary["primary_profile_sample_sd"][family][level][model],
                        f"OOD profile SD {family}/{level}/{model}",
                    )
    return {
        "status": "PASS",
        "fit_rows": len(rows),
        "condition_rows": sum(len(row["conditions"]) for row in rows),
        "summary_sha256": _sha256(summary_path),
        "headline": summary["variants"]["correct_dt_mask"]["models"]["compact_h_only"],
    }


def _factorial_effects(evaluations: list[dict[str, Any]]) -> dict[str, float]:
    cells = {
        item["level"]: float(item["delta_nrmse"])
        for item in evaluations
        if item["suite"] == "factorial2x2" and item["variant"] == "correct_dt_mask"
    }
    required = {
        "regular__observed",
        "regular__missing",
        "irregular__observed",
        "irregular__missing",
    }
    if set(cells) != required:
        raise RuntimeError(f"factorial cells mismatch: {sorted(cells)}")
    return {
        "factorial_missing_effect": 0.5
        * (
            cells["regular__missing"]
            - cells["regular__observed"]
            + cells["irregular__missing"]
            - cells["irregular__observed"]
        ),
        "factorial_irregular_effect": 0.5
        * (
            cells["irregular__observed"]
            - cells["regular__observed"]
            + cells["irregular__missing"]
            - cells["regular__missing"]
        ),
        "factorial_interaction": cells["irregular__missing"]
        - cells["irregular__observed"]
        - cells["regular__missing"]
        + cells["regular__observed"],
    }


def _equal_family_delta(evaluations: list[dict[str, Any]]) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for item in evaluations:
        if item["suite"] == "profiles19" and item["variant"] == "correct_dt_mask":
            grouped[item["family"]].append(float(item["delta_nrmse"]))
    if len(grouped) != 7 or sum(map(len, grouped.values())) != 19:
        raise RuntimeError("variable-step 19-profile coverage mismatch")
    return mean(mean(values) for values in grouped.values())


def _audit_variable_step(repo: Path) -> dict[str, Any]:
    root = repo / ".omx/results/pac-compact-h-only-variable-step-20260719"
    summary_path = repo / "paper/generated/variable_step_causal_summary.json"
    summary = _load(summary_path)
    selection = [_load(path) for path in sorted((root / "selection/completed").glob("*.json"))]
    final = [_load(path) for path in sorted((root / "final/completed").glob("*.json"))]
    if len(selection) != 126 or len(final) != 70:
        raise RuntimeError(f"variable-step row mismatch: {len(selection)}/{len(final)}")
    _unique(selection, ("model", "regime", "learning_rate", "seed"), "variable selection")
    _unique(final, ("model", "regime", "seed"), "variable final")
    for row in selection:
        if row.get("status") != "done" or row.get("test_or_ood_constructed") is not False:
            raise RuntimeError(f"variable-step selection leakage: {row.get('job_key')}")
    for row in final:
        if row.get("status") != "done" or len(row["evaluations"]) != 103:
            raise RuntimeError(f"variable-step final contract failed: {row.get('job_key')}")
    values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    seed_values: dict[tuple[str, str, int, str], float] = {}
    for row in final:
        model, regime, seed = row["model"], row["regime"], int(row["seed"])
        row_values = {
            "equal_family_macro_delta_nrmse": _equal_family_delta(row["evaluations"]),
            **_factorial_effects(row["evaluations"]),
        }
        for key, value in row_values.items():
            values[(model, regime, key)].append(value)
            seed_key = (model, regime, seed, key)
            if seed_key in seed_values:
                raise RuntimeError(f"duplicate variable-step seed value: {seed_key}")
            seed_values[seed_key] = value
    for model in MODELS:
        for regime in ("unit", "mixed_dt"):
            published = summary["models"][model][regime]
            for key in (
                "equal_family_macro_delta_nrmse",
                "factorial_missing_effect",
                "factorial_irregular_effect",
                "factorial_interaction",
            ):
                active = values[(model, regime, key)]
                _close(
                    mean(active), published[f"mean_{key}"], f"variable {model}/{regime}/{key} mean"
                )
                _close(
                    stdev(active),
                    published[f"sample_sd_{key}"],
                    f"variable {model}/{regime}/{key} sd",
                )
        benefit = mean(values[(model, "unit", "equal_family_macro_delta_nrmse")]) - mean(
            values[(model, "mixed_dt", "equal_family_macro_delta_nrmse")]
        )
        _close(
            benefit, summary["models"][model]["mixed_training_benefit"], f"variable benefit {model}"
        )
        paired_benefits = [
            seed_values[(model, "unit", seed, "equal_family_macro_delta_nrmse")]
            - seed_values[(model, "mixed_dt", seed, "equal_family_macro_delta_nrmse")]
            for seed in (23, 31, 43, 47, 59)
        ]
        _close(
            stdev(paired_benefits),
            summary["models"][model]["mixed_training_benefit_sample_sd"],
            f"variable benefit SD {model}",
        )
    return {
        "status": "PASS",
        "selection_rows": len(selection),
        "final_rows": len(final),
        "evaluation_rows": sum(len(row["evaluations"]) for row in final),
        "summary_sha256": _sha256(summary_path),
        "headline": summary["models"]["compact_h_only"],
    }


def _audit_physionet(repo: Path) -> dict[str, Any]:
    root = repo / ".omx/results/pac-compact-h-only-physionet2012-20260719"
    summary_path = repo / "paper/generated/physionet2012_summary.json"
    summary = _load(summary_path)
    selection = [_load(path) for path in sorted((root / "selection/results").glob("*.json"))]
    final = [_load(path) for path in sorted((root / "final/results").glob("*.json"))]
    if len(selection) != 126 or len(final) != 35:
        raise RuntimeError(f"PhysioNet row mismatch: {len(selection)}/{len(final)}")
    _unique(selection, ("model", "trial", "seed"), "PhysioNet selection")
    _unique(final, ("model", "seed"), "PhysioNet final")
    if any(row.get("official_test_accessed") is not False for row in selection):
        raise RuntimeError("PhysioNet selection accessed TEST")
    metrics = ("auprc", "auroc", "balanced_accuracy")
    aggregates: dict[str, dict[str, float]] = {}
    for model in MODELS:
        rows = [row for row in final if row["model"] == model]
        if len(rows) != 5 or any(row.get("status") != "done" for row in rows):
            raise RuntimeError(f"PhysioNet final coverage failed: {model}")
        aggregates[model] = {}
        for metric in metrics:
            values = [float(row["official_test"][metric]) for row in rows]
            aggregates[model][f"mean_{metric}"] = mean(values)
            aggregates[model][f"sample_sd_{metric}"] = stdev(values)
    published = {row["model"]: row for row in summary["models"]}
    for metric in metrics:
        order = sorted(MODELS, key=lambda model: aggregates[model][f"mean_{metric}"], reverse=True)
        for rank, model in enumerate(order, 1):
            _close(
                aggregates[model][f"mean_{metric}"],
                published[model][f"mean_{metric}"],
                f"PhysioNet {model}/{metric}",
            )
            _close(
                aggregates[model][f"sample_sd_{metric}"],
                published[model][f"sample_sd_{metric}"],
                f"PhysioNet sd {model}/{metric}",
            )
            _close(rank, published[model][f"rank_{metric}"], f"PhysioNet rank {model}/{metric}")
    return {
        "status": "PASS",
        "selection_rows": len(selection),
        "final_rows": len(final),
        "summary_sha256": _sha256(summary_path),
        "headline": published["compact_h_only"],
    }


def _audit_systems(repo: Path) -> dict[str, Any]:
    root = repo / ".omx/results/pac-compact-h-only-systems-20260719/reports"
    benchmark_path = root / "benchmark.json"
    evaluation_path = root / "evaluation.json"
    summary_path = repo / "paper/generated/compact_systems_summary.json"
    benchmark = _load(benchmark_path)
    evaluation = _load(evaluation_path)
    summary = _load(summary_path)
    if evaluation.get("status") != "PASS":
        raise RuntimeError("systems evaluation is not PASS")
    if len(benchmark["inference_rows"]) != 168 or len(benchmark["training_rows"]) != 126:
        raise RuntimeError("systems raw row count mismatch")
    computed: dict[str, dict[str, float]] = {}
    for model in MODELS:
        inference = [
            row
            for row in benchmark["inference_rows"]
            if row["model"] == model and row["runtime"] == "best_exact_fp32"
        ]
        training = [
            row
            for row in benchmark["training_rows"]
            if row["model"] == model and row["runtime"] == "best_exact_train_step_fp32"
        ]
        if len(inference) != 6 or len(training) != 6:
            raise RuntimeError(f"systems selected-cell coverage failed: {model}")
        computed[model] = {
            "inference": geometric_mean(float(row["latency_ms"]) for row in inference),
            "training": geometric_mean(
                float(row["full_train_step_wall_ms"]) for row in training
            ),
        }
    published = {row["model"]: row for row in summary["models"]}
    for metric in ("inference", "training"):
        order = sorted(MODELS, key=lambda model: computed[model][metric])
        for rank, model in enumerate(order, 1):
            _close(
                computed[model][metric],
                published[model][f"{metric}_geometric_mean_ms"],
                f"systems {model}/{metric}",
            )
            _close(rank, published[model][f"{metric}_rank"], f"systems rank {model}/{metric}")
    if _sha256(benchmark_path) != summary["benchmark_sha256"]:
        raise RuntimeError("systems benchmark hash mismatch")
    if _sha256(evaluation_path) != summary["evaluation_sha256"]:
        raise RuntimeError("systems evaluation hash mismatch")
    return {
        "status": "PASS",
        "inference_rows": len(benchmark["inference_rows"]),
        "training_rows": len(benchmark["training_rows"]),
        "summary_sha256": _sha256(summary_path),
        "headline": published["compact_h_only"],
    }


def audit_all(repo: Path) -> dict[str, Any]:
    sections = {
        "component_ablation": _audit_ablation(repo),
        "boundary": _audit_boundary(repo),
        "synthetic_ood": _audit_synthetic_ood(repo),
        "variable_step": _audit_variable_step(repo),
        "physionet2012": _audit_physionet(repo),
        "systems": _audit_systems(repo),
    }
    return {
        "schema": "alphabet.secondary_evidence_independent_audit.v1",
        "status": "PASS",
        "public_model": "ALPHABET",
        "internal_model": "compact_h_only",
        "independence_contract": (
            "computed directly from completed result ledgers without importing paper generators"
        ),
        "sections": sections,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/generated/alphabet_secondary_evidence_independent_audit.json"),
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.output if args.output.is_absolute() else repo / args.output
    payload = audit_all(repo)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
