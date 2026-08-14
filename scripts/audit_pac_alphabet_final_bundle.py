#!/usr/bin/env python3
# ruff: noqa: C901, EM101, EM102, FBT001, PLR0912, PLR0915, T201, TRY003
"""Independently audit the final ALPHABET Q1/Q2 paper bundle.

This checker intentionally does not call the paper generators.  It recomputes
the published ranks and Top-1 counts from immutable final result rows, then
compares them with the generated JSON summaries.  It is expected to fail until
the replacement campaign has a verified completion marker and the paper assets
have been regenerated from that campaign.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

MODELS = (
    "compact_h_only",
    "cnn1d",
    "tcn",
    "mamba",
    "gru",
    "lstm",
    "transformer",
)
SEEDS = (23, 31, 43, 47, 59)
BUDGETS = (0.5, 1.0, 2.0, 4.0)
RTOL = 1.0e-5
ATOL = 1.0e-8
PARAMETER_TOLERANCE = 0.062
UCR_TASKS = (
    "ArrowHead",
    "CinCECGTorso",
    "CricketX",
    "ECG200",
    "ECG5000",
    "ECGFiveDays",
    "Earthquakes",
    "FordA",
    "FordB",
    "GunPoint",
    "ItalyPowerDemand",
    "MoteStrain",
    "Phoneme",
    "Plane",
    "StarLightCurves",
    "Trace",
    "TwoLeadECG",
    "Wafer",
)
EXTERNAL_TASKS = (
    "audioset-balanced",
    "cwru",
    "electricity",
    "ettm1",
    "ettm2",
    "mit-bih",
    "permuted-mnist",
    "ptb-xl",
    "sequential-cifar",
    "sequential-mnist",
    "speech-commands",
    "weather",
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metric(suite: str, dataset: str) -> tuple[str, bool]:
    if suite == "ucr":
        return "balanced_accuracy", False
    if dataset in {"electricity", "ettm1", "ettm2", "weather"}:
        return "mse", True
    if dataset == "audioset-balanced":
        return "macro_auprc", False
    if dataset == "ptb-xl":
        return "macro_auroc", False
    return "accuracy", False


def _average_tie_ranks(scores: dict[str, float], lower: bool) -> dict[str, float]:
    ordered = sorted(scores, key=scores.__getitem__, reverse=not lower)
    ranks: dict[str, float] = {}
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and math.isclose(
            scores[ordered[cursor]], scores[ordered[end]], rel_tol=RTOL, abs_tol=ATOL
        ):
            end += 1
        rank = ((cursor + 1) + end) / 2.0
        for model in ordered[cursor:end]:
            ranks[model] = rank
        cursor = end
    return ranks


def _primary_value(row: dict[str, Any]) -> float:
    field, _ = _metric(str(row["suite"]), str(row["dataset"]))
    value = float(row[field])
    if not math.isfinite(value):
        raise RuntimeError(f"nonfinite {field}: {row['job_key']}")
    return value


def _load_rows(root: Path, stage: str) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    completed_dir = root / stage / "completed"
    for path in sorted(completed_dir.glob("*.json")):
        row = _read(path)
        if row.get("model") not in MODELS:
            continue
        if row.get("status") != "done":
            raise RuntimeError(f"non-done row: {path}")
        if row.get("evaluation_split") != "test" or not row.get("test_evaluated"):
            raise RuntimeError(f"non-TEST row: {path}")
        if not row.get("official_test_accessed"):
            raise RuntimeError(f"unsealed row: {path}")
        key = str(row["job_key"])
        if key in by_key:
            raise RuntimeError(f"duplicate result key: {key}")
        by_key[key] = row
    completed_files = {path.name for path in completed_dir.glob("*.json")}
    unresolved_failures = [
        path.name
        for path in (root / stage / "failed").glob("*.json")
        if path.name not in completed_files
    ]
    if unresolved_failures:
        raise RuntimeError(
            f"{stage} has {len(unresolved_failures)} unresolved failed keys; "
            f"first={unresolved_failures[0]}"
        )
    return list(by_key.values())


def _audit_attempt_events(
    root: Path, stage: str, *, allow_unfinished: bool = False
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    by_attempt: dict[str, set[str]] = defaultdict(set)
    failed_exception_types: Counter[str] = Counter()
    for path in sorted((root / stage / "attempts").rglob("*.json")):
        row = _read(path)
        if row.get("schema") != "pac_baseline_fairness_attempt_event.v1":
            raise RuntimeError(f"unexpected attempt-event schema: {path}")
        event = str(row.get("event"))
        if event not in {"started", "failed", "succeeded", "abandoned"}:
            raise RuntimeError(f"unexpected attempt event {event!r}: {path}")
        attempt_id = str(row["attempt_id"])
        counts[event] += 1
        by_attempt[attempt_id].add(event)
        if event == "failed":
            error = str(row.get("error", "UnknownError"))
            failed_exception_types[error.split(":", maxsplit=1)[0]] += 1
    unfinished = sum(
        "started" in events and not ({"failed", "succeeded", "abandoned"} & events)
        for events in by_attempt.values()
    )
    result = {
        "started": counts["started"],
        "failed": counts["failed"],
        "succeeded": counts["succeeded"],
        "abandoned": counts["abandoned"],
        "unfinished": unfinished,
        "failed_exception_types": dict(sorted(failed_exception_types.items())),
    }
    if unfinished and not allow_unfinished:
        raise RuntimeError(f"{stage} has {unfinished} unfinished attempts")
    expected_started = (
        result["failed"] + result["succeeded"] + result["abandoned"] + result["unfinished"]
    )
    if result["started"] != expected_started:
        raise RuntimeError(f"{stage} attempt-event accounting is not closed")
    return result


def _budget_token(budget: float) -> str:
    return str(int(budget)) if budget.is_integer() else f"{budget:g}"


def _failure_class(error: str) -> str:
    if error.startswith("ExternalDatasetError:") and "requires a prepared task at " in error:
        return "missing_prepared_external_task"
    if error.startswith("ExternalDatasetError:") and "missing required files:" in error:
        return "missing_external_source_files"
    if "out of memory" in error.lower():
        return "out_of_memory"
    return "unclassified"


def _audit_q2_partial_failures(
    root: Path,
    *,
    completed_job_keys: set[str],
    selected: dict[str, Any],
) -> dict[str, Any]:
    """Account for transient Q2 failure rows without declaring them terminal."""
    failure_job_keys: set[str] = set()
    unresolved_exception_types: Counter[str] = Counter()
    unresolved_failure_classes: Counter[str] = Counter()
    unresolved_datasets: Counter[str] = Counter()
    superseded_rows = 0
    failure_files = sorted((root / "q2_final/failed").glob("*.json"))
    for path in failure_files:
        row = _read(path)
        if row.get("schema") != "pac_baseline_fairness_failure.v1":
            raise RuntimeError(f"partial Q2 unexpected failure schema: {path}")
        if row.get("status") != "failed" or row.get("stage") != "q2_final":
            raise RuntimeError(f"partial Q2 malformed failure row: {path}")
        model = str(row.get("model"))
        if model not in MODELS:
            raise RuntimeError(f"partial Q2 failure has unexpected model: {model}")
        suite = str(row["suite"])
        dataset = str(row["dataset"])
        budget = float(row["budget_multiplier"])
        selection_key = f"{suite}:{dataset}:{model}:budget{_budget_token(budget)}"
        if selection_key not in selected:
            raise RuntimeError(f"partial Q2 failure is absent from selection: {selection_key}")
        seed = int(row["train_seed"])
        if seed not in SEEDS or int(row["split_seed"]) != seed:
            raise RuntimeError(f"partial Q2 failure seed mismatch: {path}")
        job_key = str(row["job_key"])
        if job_key in failure_job_keys:
            raise RuntimeError(f"partial Q2 duplicate failure key: {job_key}")
        failure_job_keys.add(job_key)
        if job_key in completed_job_keys:
            superseded_rows += 1
            continue
        error = str(row.get("error", "UnknownError"))
        error_type = error.split(":", maxsplit=1)[0] or "UnknownError"
        unresolved_exception_types[error_type] += 1
        unresolved_failure_classes[_failure_class(error)] += 1
        unresolved_datasets[f"{suite}:{dataset}"] += 1

    unresolved_rows = len(failure_job_keys - completed_job_keys)
    known_preparation_retryable_rows = sum(
        unresolved_failure_classes[name]
        for name in ("missing_prepared_external_task", "missing_external_source_files")
    )
    return {
        "failure_files": len(failure_files),
        "unresolved_rows": unresolved_rows,
        "superseded_rows": superseded_rows,
        "known_preparation_retryable_rows": known_preparation_retryable_rows,
        "unclassified_unresolved_rows": unresolved_failure_classes["unclassified"],
        "unresolved_exception_types": dict(sorted(unresolved_exception_types.items())),
        "unresolved_failure_classes": dict(sorted(unresolved_failure_classes.items())),
        "unresolved_datasets": dict(sorted(unresolved_datasets.items())),
        "terminal_status_inferred": False,
    }


def _audit_q2_execution_identity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Bind every Q2 result to one frozen runner source and selection artifact."""
    if not rows:
        return {
            "code_sha256": None,
            "selection_artifact_sha256": None,
            "manifest_sha256_count": 0,
        }
    code_hashes = {str(row.get("code_sha256", "")) for row in rows}
    if "" in code_hashes or len(code_hashes) != 1:
        raise RuntimeError(
            f"Q2 result rows do not share one frozen code hash: {sorted(code_hashes)}"
        )
    selection_hashes = {str(row.get("selection_artifact_sha256", "")) for row in rows}
    if "" in selection_hashes or len(selection_hashes) != 1:
        raise RuntimeError(
            f"Q2 result rows do not share one selection-artifact hash: {sorted(selection_hashes)}"
        )
    manifest_hashes = {str(row.get("manifest_sha256", "")) for row in rows}
    if "" in manifest_hashes:
        raise RuntimeError("Q2 result row is missing its execution-manifest hash")
    return {
        "code_sha256": next(iter(code_hashes)),
        "selection_artifact_sha256": next(iter(selection_hashes)),
        "manifest_sha256_count": len(manifest_hashes),
    }


def _audit_q2_partial(root: Path) -> dict[str, Any]:
    """Validate completed Q2 rows without interpreting an incomplete score matrix."""
    selection_path = root / "q2_calibration/selection.json"
    selection = _read(selection_path)
    selected = selection["selected"]
    if [int(seed) for seed in selection["final_seeds"]] != list(SEEDS):
        raise RuntimeError(f"Q2 selection final-seed mismatch: {selection['final_seeds']}")
    selection_sha256 = _sha256(selection_path)
    completed_dir = root / "q2_final/completed"
    rows = [_read(path) for path in sorted(completed_dir.glob("*.json"))]
    logical_keys: set[str] = set()
    cell_seeds: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        if row.get("status") != "done":
            raise RuntimeError(f"partial Q2 non-done row: {row.get('job_key')}")
        if row.get("evaluation_split") != "test" or row.get("test_evaluated") is not True:
            raise RuntimeError(f"partial Q2 non-TEST row: {row.get('job_key')}")
        if row.get("official_test_accessed") is not True:
            raise RuntimeError(f"partial Q2 unsealed row: {row.get('job_key')}")
        model = str(row.get("model"))
        if model not in MODELS:
            raise RuntimeError(f"partial Q2 unexpected model: {model}")
        suite = str(row["suite"])
        dataset = str(row["dataset"])
        budget = float(row["budget_multiplier"])
        selection_key = f"{suite}:{dataset}:{model}:budget{_budget_token(budget)}"
        chosen = selected.get(selection_key)
        if chosen is None:
            raise RuntimeError(f"partial Q2 row is absent from selection: {selection_key}")
        key = str(row["job_key"])
        if key in logical_keys:
            raise RuntimeError(f"partial Q2 duplicate result key: {key}")
        logical_keys.add(key)
        seed = int(row["train_seed"])
        if seed not in SEEDS or int(row["split_seed"]) != seed:
            raise RuntimeError(f"partial Q2 seed mismatch: {key}")
        if seed in cell_seeds[selection_key]:
            raise RuntimeError(f"partial Q2 duplicate cell seed: {selection_key}/{seed}")
        cell_seeds[selection_key].add(seed)
        for field in ("params_trainable", "target_parameters", "trial", "width"):
            if int(row[field]) != int(chosen[field]):
                raise RuntimeError(f"partial Q2 selected-{field} mismatch: {key}")
        if str(row["config_key"]) != str(chosen["config_key"]):
            raise RuntimeError(f"partial Q2 selected-config mismatch: {key}")
        for field in ("learning_rate", "lr_multiplier", "relative_parameter_error"):
            if not math.isclose(
                float(row[field]), float(chosen[field]), rel_tol=0.0, abs_tol=1e-12
            ):
                raise RuntimeError(f"partial Q2 selected-{field} mismatch: {key}")
        if str(row.get("selection_artifact_sha256")) != selection_sha256:
            raise RuntimeError(f"partial Q2 selection-artifact hash mismatch: {key}")
        if float(row["relative_parameter_error"]) > PARAMETER_TOLERANCE + 1e-12:
            raise RuntimeError(f"partial Q2 parameter tolerance exceeded: {key}")
        _primary_value(row)
    failure_ledger = _audit_q2_partial_failures(
        root,
        completed_job_keys=logical_keys,
        selected=selected,
    )
    attempt_events = _audit_attempt_events(root, "q2_final", allow_unfinished=True)
    execution_identity = _audit_q2_execution_identity(rows)
    return {
        "status": "PASS",
        "interpret_scores": False,
        "completed_rows": len(rows),
        "started_cells": len(cell_seeds),
        "declared_selected_cells": int(selection["selected_realizable_cells"]),
        "declared_final_rows": int(selection["final_jobs"]),
        "execution_identity": execution_identity,
        "failure_ledger": failure_ledger,
        "attempt_events": attempt_events,
    }


def _profile_summary(
    cell_means: dict[tuple[str, str, str], float],
    profiles: list[tuple[str, str]],
) -> dict[str, Any]:
    rank_rows: list[dict[str, float]] = []
    top = dict.fromkeys(MODELS, 0)
    sole = dict.fromkeys(MODELS, 0)
    for suite, dataset in profiles:
        scores = {model: cell_means[(suite, dataset, model)] for model in MODELS}
        _, lower = _metric(suite, dataset)
        ranks = _average_tie_ranks(scores, lower)
        best = min(scores.values()) if lower else max(scores.values())
        winners = [
            model
            for model, value in scores.items()
            if math.isclose(value, best, rel_tol=RTOL, abs_tol=ATOL)
        ]
        for model in winners:
            top[model] += 1
        if len(winners) == 1:
            sole[winners[0]] += 1
        rank_rows.append(ranks)
    return {
        "profiles": len(profiles),
        "mean_rank": {model: mean(row[model] for row in rank_rows) for model in MODELS},
        "top_count": top,
        "sole_top_count": sole,
    }


def _compute_q1(root: Path) -> dict[str, Any]:
    rows = _load_rows(root, "final")
    expected_rows = (len(UCR_TASKS) + len(EXTERNAL_TASKS)) * len(MODELS) * len(SEEDS)
    if len(rows) != expected_rows:
        raise RuntimeError(f"Q1 has {len(rows)}/{expected_rows} rows")
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["suite"]), str(row["dataset"]), str(row["model"]))].append(row)
    expected_cells = {
        (suite, task, model)
        for suite, tasks in (("ucr", UCR_TASKS), ("external", EXTERNAL_TASKS))
        for task in tasks
        for model in MODELS
    }
    if set(grouped) != expected_cells:
        missing = sorted(expected_cells - set(grouped))
        extra = sorted(set(grouped) - expected_cells)
        raise RuntimeError(f"Q1 cell coverage mismatch: missing={missing[:1]}, extra={extra[:1]}")
    cell_means: dict[tuple[str, str, str], float] = {}
    for key, items in grouped.items():
        seeds = sorted(int(row["train_seed"]) for row in items)
        if seeds != list(SEEDS):
            raise RuntimeError(f"Q1 seed mismatch for {key}: {seeds}")
        if any(int(row["split_seed"]) != int(row["train_seed"]) for row in items):
            raise RuntimeError(f"Q1 split/train seed mismatch for {key}")
        if len({str(row["config_key"]) for row in items}) != 1:
            raise RuntimeError(f"Q1 config drift for {key}")
        if len({int(row["params_trainable"]) for row in items}) != 1:
            raise RuntimeError(f"Q1 parameter drift for {key}")
        cell_means[key] = mean(_primary_value(row) for row in items)
    ucr = [("ucr", task) for task in UCR_TASKS]
    external = [("external", task) for task in EXTERNAL_TASKS]
    recomputed = {
        "all30": _profile_summary(cell_means, ucr + external),
        "ucr18": _profile_summary(cell_means, ucr),
        "external12": _profile_summary(cell_means, external),
    }
    return {
        "rows": len(rows),
        "cells": len(grouped),
        "subsets": recomputed,
        "retained_attempt_events": {
            "scope": (
                "append-only retry and recovery events retained by the final campaign; "
                "the complete 1,050-row result matrix is the coverage authority"
            ),
            "counts": _audit_attempt_events(root, "final"),
        },
    }


def _audit_q1(root: Path, summary: dict[str, Any]) -> dict[str, Any]:
    audit = _compute_q1(root)
    recomputed = audit["subsets"]
    for subset, values in recomputed.items():
        for field in ("mean_rank", "top_count", "sole_top_count"):
            published = summary[subset][field]
            for model in MODELS:
                actual = values[field][model]
                expected = published[model]
                if isinstance(actual, float):
                    if not math.isclose(actual, float(expected), rel_tol=0.0, abs_tol=1e-12):
                        raise RuntimeError(
                            f"Q1 {subset}/{field}/{model}: raw={actual}, summary={expected}"
                        )
                elif actual != expected:
                    raise RuntimeError(
                        f"Q1 {subset}/{field}/{model}: raw={actual}, summary={expected}"
                    )
    return audit


def _audit_q2(root: Path, summary: dict[str, Any]) -> dict[str, Any]:
    selection_path = root / "q2_calibration/selection.json"
    selection = _read(selection_path)
    selected = selection["selected"]
    rows = _load_rows(root, "q2_final")
    expected_cells = int(selection["selected_realizable_cells"])
    expected_rows = expected_cells * len(SEEDS)
    if len(selected) != expected_cells:
        raise RuntimeError(f"Q2 selection has {len(selected)}/{expected_cells} selected cells")
    if [int(seed) for seed in selection["final_seeds"]] != list(SEEDS):
        raise RuntimeError(f"Q2 selection final-seed mismatch: {selection['final_seeds']}")
    if int(selection["final_jobs"]) != expected_rows:
        raise RuntimeError(
            f"Q2 selection declares {selection['final_jobs']}/{expected_rows} final jobs"
        )
    if len(rows) != expected_rows:
        raise RuntimeError(f"Q2 has {len(rows)}/{expected_rows} rows")
    grouped: dict[tuple[str, str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["suite"]),
                str(row["dataset"]),
                str(row["model"]),
                float(row["budget_multiplier"]),
            )
        ].append(row)
    if len(grouped) != expected_cells:
        raise RuntimeError(f"Q2 has {len(grouped)}/{expected_cells} cells")
    selected_cell_keys = {
        (
            *selection_key.split(":", maxsplit=3)[:3],
            float(selection_key.rsplit("budget", maxsplit=1)[1]),
        )
        for selection_key in selected
    }
    if set(grouped) != selected_cell_keys:
        missing = sorted(selected_cell_keys - set(grouped))
        extra = sorted(set(grouped) - selected_cell_keys)
        raise RuntimeError(
            f"Q2 final cells differ from selection: missing={missing[:1]}, extra={extra[:1]}"
        )
    selection_sha256 = _sha256(selection_path)
    means: dict[tuple[str, str, str, float], float] = {}
    for key, items in grouped.items():
        suite, dataset, model, budget = key
        selection_key = f"{suite}:{dataset}:{model}:budget{_budget_token(budget)}"
        chosen = selected[selection_key]
        seeds = sorted(int(row["train_seed"]) for row in items)
        if seeds != list(SEEDS):
            raise RuntimeError(f"Q2 seed mismatch for {key}: {seeds}")
        if any(int(row["split_seed"]) != int(row["train_seed"]) for row in items):
            raise RuntimeError(f"Q2 split/train seed mismatch for {key}")
        if len({int(row["params_trainable"]) for row in items}) != 1:
            raise RuntimeError(f"Q2 parameter drift for {key}")
        if len({str(row["config_key"]) for row in items}) != 1:
            raise RuntimeError(f"Q2 config drift for {key}")
        first = items[0]
        if str(first["config_key"]) != str(chosen["config_key"]):
            raise RuntimeError(f"Q2 selected-config mismatch for {key}")
        for field in ("params_trainable", "target_parameters", "trial", "width"):
            if int(first[field]) != int(chosen[field]):
                raise RuntimeError(f"Q2 selected-{field} mismatch for {key}")
        for field in ("learning_rate", "lr_multiplier", "relative_parameter_error"):
            if not math.isclose(
                float(first[field]), float(chosen[field]), rel_tol=0.0, abs_tol=1e-12
            ):
                raise RuntimeError(f"Q2 selected-{field} mismatch for {key}")
        if any(str(row.get("selection_artifact_sha256")) != selection_sha256 for row in items):
            raise RuntimeError(f"Q2 selection-artifact hash mismatch for {key}")
        if float(first["relative_parameter_error"]) > PARAMETER_TOLERANCE + 1e-12:
            raise RuntimeError(f"Q2 parameter tolerance exceeded for {key}")
        means[key] = mean(_primary_value(row) for row in items)

    by_budget: dict[str, Any] = {}
    pooled_rank_rows: list[dict[str, float]] = []
    pooled_winners: list[list[str]] = []
    for budget in BUDGETS:
        profiles = [
            (suite, task)
            for suite, tasks in (("ucr", UCR_TASKS), ("external", EXTERNAL_TASKS))
            for task in tasks
            if all((suite, task, model, budget) in means for model in MODELS)
        ]
        rank_rows: list[dict[str, float]] = []
        top = dict.fromkeys(MODELS, 0)
        for suite, task in profiles:
            scores = {model: means[(suite, task, model, budget)] for model in MODELS}
            _, lower = _metric(suite, task)
            ranks = _average_tie_ranks(scores, lower)
            best = min(scores.values()) if lower else max(scores.values())
            winners = [
                model
                for model, value in scores.items()
                if math.isclose(value, best, rel_tol=RTOL, abs_tol=ATOL)
            ]
            for model in winners:
                top[model] += 1
            rank_rows.append(ranks)
            pooled_rank_rows.append(ranks)
            pooled_winners.append(winners)
        key = {0.5: "0.5x", 1.0: "1x", 2.0: "2x", 4.0: "4x"}[budget]
        by_budget[key] = {
            "common_profiles": len(profiles),
            "mean_rank": {model: mean(row[model] for row in rank_rows) for model in MODELS},
            "top_count": top,
        }
    pooled = {
        "common_profiles": len(pooled_rank_rows),
        "mean_rank": {model: mean(row[model] for row in pooled_rank_rows) for model in MODELS},
        "top_count": {
            model: sum(model in winners for winners in pooled_winners) for model in MODELS
        },
    }
    by_budget["pooled_descriptive"] = pooled
    for key, values in by_budget.items():
        published = summary["rankings"][key]
        if values["common_profiles"] != int(published["common_profiles"]):
            raise RuntimeError(f"Q2 {key} profile count mismatch")
        for field in ("mean_rank", "top_count"):
            for model in MODELS:
                actual = values[field][model]
                expected = published[field][model]
                if isinstance(actual, float):
                    if not math.isclose(actual, float(expected), rel_tol=0.0, abs_tol=1e-12):
                        raise RuntimeError(
                            f"Q2 {key}/{field}/{model}: raw={actual}, summary={expected}"
                        )
                elif actual != expected:
                    raise RuntimeError(
                        f"Q2 {key}/{field}/{model}: raw={actual}, summary={expected}"
                    )
    attempts = _audit_attempt_events(root, "q2_final")
    if attempts != summary["protocol"]["final_attempt_events"]:
        raise RuntimeError("Q2 raw attempt ledger disagrees with the generated summary")
    if attempts["succeeded"] != len(rows):
        raise RuntimeError("Q2 successful-attempt count differs from the result matrix")
    execution_identity = _audit_q2_execution_identity(rows)
    if summary.get("execution_identity") != execution_identity:
        raise RuntimeError("Q2 generated summary execution identity disagrees with raw rows")
    return {
        "rows": len(rows),
        "cells": len(grouped),
        "execution_identity": execution_identity,
        "rankings": by_budget,
        "attempt_events": attempts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--campaign-root",
        type=Path,
        default=Path(".omx/results/pac-alphabet-q1q2-final-20260719"),
    )
    parser.add_argument(
        "--q1-summary", type=Path, default=Path("paper/generated/q1_final_summary.json")
    )
    parser.add_argument(
        "--q2-summary", type=Path, default=Path("paper/generated/q2_final_summary.json")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/generated/alphabet_final_independent_audit.json"),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--partial-q2", action="store_true")
    mode.add_argument("--q1-only", action="store_true")
    parser.add_argument("--partial-q2-output", type=Path)
    parser.add_argument("--q1-only-output", type=Path)
    args = parser.parse_args()

    if args.partial_q2:
        audit = _audit_q2_partial(args.campaign_root)
        output = args.partial_q2_output or (args.campaign_root / "audit/q2_partial_progress.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(audit, indent=2, sort_keys=True))
        return
    if args.q1_only:
        audit = {
            "schema": "alphabet.q1_independent_pre_activation_audit.v1",
            "chosen_internal_model": "compact_h_only",
            "interpret_scores": True,
            "q1": _compute_q1(args.campaign_root),
            "status": "PASS",
        }
        output = args.q1_only_output or (
            args.campaign_root / "audit/q1_independent_pre_activation.json"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(audit, indent=2, sort_keys=True))
        return

    marker_path = args.campaign_root / "pipeline_complete.json"
    if not marker_path.is_file():
        raise RuntimeError("replacement campaign is not complete; refusing a partial audit")
    marker = _read(marker_path)
    if marker.get("schema") != "pac_alphabet_q1_q2_pipeline_complete.v2" or not marker.get(
        "verified"
    ):
        raise RuntimeError("replacement completion marker is not verified")
    if marker.get("chosen_internal_model") != "compact_h_only":
        raise RuntimeError("completion marker does not select compact_h_only")

    q1_summary = _read(args.q1_summary)
    q2_summary = _read(args.q2_summary)
    audit = {
        "schema": "alphabet.final_independent_audit.v1",
        "campaign": str(args.campaign_root),
        "chosen_internal_model": "compact_h_only",
        "sources_sha256": {
            "pipeline_complete": _sha256(marker_path),
            "q1_summary": _sha256(args.q1_summary),
            "q2_summary": _sha256(args.q2_summary),
        },
        "q1": _audit_q1(args.campaign_root, q1_summary),
        "q2": _audit_q2(args.campaign_root, q2_summary),
        "status": "PASS",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "PASS: independently recomputed "
        f"Q1 {audit['q1']['rows']} rows and "
        f"Q2 {audit['q2']['rows']} rows/{audit['q2']['cells']} cells"
    )


if __name__ == "__main__":
    main()
