"""Independently audit and summarize learned two-tap ALPHABET Q1-final."""

# ruff: noqa: C901, EM101, EM102, T201, TRY003
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Final, Literal

from scipy.stats import friedmanchisquare, wilcoxon

CANDIDATE: Final = "two_tap_h_only"
MODELS: Final = (
    CANDIDATE,
    "cnn1d",
    "tcn",
    "mamba",
    "gru",
    "lstm",
    "transformer",
)
SEEDS: Final = (23, 31, 43, 47, 59)
SELECTION_SEEDS: Final = (7, 11, 19)
RTOL: Final = 1.0e-5
ATOL: Final = 1.0e-8
CONFIGS: Final = tuple(
    f"d{model_dim}-m{modes}-t{trial}"
    for model_dim, modes in ((16, 8), (32, 8), (32, 16), (64, 8), (64, 16), (64, 32))
    for trial in (2, 4, 6)
)
UCR_TASKS: Final = (
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
EXTERNAL_TASKS: Final = (
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


def _expected_tasks() -> set[tuple[str, str]]:
    return {
        *(("ucr", task) for task in UCR_TASKS),
        *(("external", task) for task in EXTERNAL_TASKS),
    }


def _manifest_hashes(root: Path, stage: str) -> set[str]:
    return {_sha256(path) for path in (root / stage / "manifests").glob("*.jsonl")}


def _audit_source_manifest(root: Path) -> tuple[str, dict[str, str]]:
    path = root / "reports/source_manifest.json"
    manifest = _read(path)
    claimed = str(manifest.pop("sha256", ""))
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    recomputed = hashlib.sha256(canonical).hexdigest()
    source_hashes = manifest.get("source_sha256")
    if (
        manifest.get("schema") != "pac_two_tap_q1_source_manifest.v1"
        or manifest.get("candidate") != CANDIDATE
        or not isinstance(source_hashes, dict)
        or claimed != recomputed
    ):
        raise RuntimeError("invalid learned-two-tap source manifest")
    source_root = Path(__file__).resolve().parents[1] / "src/lnet"
    for name, expected in source_hashes.items():
        source = source_root / str(name)
        if not source.is_file() or _sha256(source) != str(expected):
            raise RuntimeError(f"current source differs from frozen campaign: {name}")
    return claimed, {str(name): str(value) for name, value in source_hashes.items()}


def _average_tie_ranks(scores: dict[str, float], *, lower: bool) -> dict[str, float]:
    ordered = sorted(scores, key=scores.__getitem__, reverse=not lower)
    result: dict[str, float] = {}
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and math.isclose(
            scores[ordered[cursor]], scores[ordered[end]], rel_tol=RTOL, abs_tol=ATOL
        ):
            end += 1
        rank = ((cursor + 1) + end) / 2.0
        for model in ordered[cursor:end]:
            result[model] = rank
        cursor = end
    return result


def _holm(raw: dict[str, float]) -> dict[str, float]:
    ordered = sorted(raw, key=lambda model: (raw[model], model))
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for index, model in enumerate(ordered):
        running = max(running, min(1.0, (count - index) * raw[model]))
        adjusted[model] = running
    return adjusted


def _audit_scope(root: Path) -> dict[str, Any]:
    decision = _read(root / "study_scope_decision.json")
    if (
        decision.get("schema") != "pac_two_tap_q1_scope_decision.v1"
        or decision.get("chosen_internal_model") != CANDIDATE
        or decision.get("architecture_status") != "final"
        or decision.get("architecture_choice_uses_test_evidence") is not False
        or "Q2-final TEST evaluation" not in decision.get("excluded_scope", [])
    ):
        raise RuntimeError("study scope does not freeze learned two-tap and exclude Q2")
    return decision


def _load_validation_rows(
    root: Path,
    stage: Literal["stage1", "stage2"],
    expected_count: int,
) -> list[dict[str, Any]]:
    completed = root / stage / "completed"
    rows = [_read(path) for path in sorted(completed.glob("*.json"))]
    unresolved = {path.name for path in (root / stage / "failed").glob("*.json")} - {
        path.name for path in completed.glob("*.json")
    }
    if unresolved or len(rows) != expected_count:
        raise RuntimeError(
            f"{stage} ledger is incomplete: rows={len(rows)}/{expected_count}, "
            f"failures={len(unresolved)}"
        )
    return rows


def _validate_validation_row(
    row: dict[str, Any],
    *,
    stage: Literal["stage1", "stage2"],
    expected_seeds: set[int],
    manifest_hashes: set[str],
) -> tuple[str, str, int]:
    suite = str(row.get("suite"))
    dataset = str(row.get("dataset"))
    config = str(row.get("config_key"))
    seed = int(row.get("train_seed", -1))
    score = row.get("selection_score")
    if (
        (suite, dataset) not in _expected_tasks()
        or row.get("stage") != stage
        or row.get("model") != CANDIDATE
        or config not in CONFIGS
        or seed not in expected_seeds
        or int(row.get("split_seed", -1)) != seed
        or row.get("status") != "done"
        or row.get("evaluation_split") != "validation"
        or row.get("official_test_accessed") is not False
        or row.get("test_evaluated") is not False
        or not isinstance(score, (int, float))
        or not math.isfinite(float(score))
        or str(row.get("manifest_sha256", "")) not in manifest_hashes
    ):
        raise RuntimeError(f"invalid {stage} validation row: {row.get('job_key')}")
    return f"{suite}:{dataset}:{CANDIDATE}", config, seed


def _audit_validation_ledgers(root: Path) -> dict[str, Any]:
    stage1_selection_path = root / "stage1/selection.json"
    stage1_selection = _read(stage1_selection_path)
    selected = stage1_selection.get("selected")
    if not isinstance(selected, dict):
        raise TypeError("Stage 1 selection is missing its selected grid")

    stage1 = _load_validation_rows(root, "stage1", 540)
    stage2 = _load_validation_rows(root, "stage2", 360)
    stage1_manifests = _manifest_hashes(root, "stage1")
    stage2_manifests = _manifest_hashes(root, "stage2")
    stage1_keys = {
        _validate_validation_row(
            row,
            stage="stage1",
            expected_seeds={7},
            manifest_hashes=stage1_manifests,
        )
        for row in stage1
    }
    expected_stage1 = {
        (f"{suite}:{dataset}:{CANDIDATE}", config, 7)
        for suite, dataset in _expected_tasks()
        for config in CONFIGS
    }
    if stage1_keys != expected_stage1:
        raise RuntimeError("Stage 1 does not form the exact 30 x 18 x 1 validation product")

    stage1_sha256 = _sha256(stage1_selection_path)
    stage2_keys: set[tuple[str, str, int]] = set()
    for row in stage2:
        key = _validate_validation_row(
            row,
            stage="stage2",
            expected_seeds={11, 19},
            manifest_hashes=stage2_manifests,
        )
        if str(row.get("selection_artifact_sha256")) != stage1_sha256:
            raise RuntimeError(f"Stage 2 selection hash drift: {row.get('job_key')}")
        stage2_keys.add(key)
    expected_stage2 = {
        (cell_key, str(config), seed)
        for cell_key, configs in selected.items()
        for config in configs
        for seed in (11, 19)
    }
    if stage2_keys != expected_stage2:
        raise RuntimeError("Stage 2 does not form the exact selected 30 x 6 x 2 product")

    runner_hashes = {str(row.get("code_sha256", "")) for row in [*stage1, *stage2]}
    if len(runner_hashes) != 1 or "" in runner_hashes:
        raise RuntimeError("validation ledgers mix runner implementations")
    return {
        "stage1_rows": len(stage1),
        "stage2_rows": len(stage2),
        "stage1_manifest_hashes": len(stage1_manifests),
        "stage2_manifest_hashes": len(stage2_manifests),
        "runner_code_sha256": next(iter(runner_hashes)),
    }


def _audit_selection(root: Path) -> tuple[dict[str, dict[str, Any]], str]:
    stage1 = _read(root / "stage1/selection.json")
    stage2_path = root / "stage2/selection.json"
    stage2 = _read(stage2_path)
    if (
        stage1.get("schema") != "pac_two_tap_q1_stage1_selection.v1"
        or stage1.get("official_test_accessed") is not False
        or stage2.get("schema") != "pac_two_tap_q1_stage2_selection.v1"
        or stage2.get("official_test_accessed") is not False
        or stage2.get("candidate") != CANDIDATE
    ):
        raise RuntimeError("selection artifacts are incomplete or TEST-contaminated")
    selected = stage2.get("selected")
    expected = {f"{suite}:{dataset}:{CANDIDATE}" for suite, dataset in _expected_tasks()}
    if not isinstance(selected, dict) or set(selected) != expected:
        raise RuntimeError(f"selected task grid is not exact: {len(selected or {})}/30")
    for key, row in selected.items():
        if (
            tuple(row.get("selection_seeds", ())) != SELECTION_SEEDS
            or int(row["model_dim"]) not in {16, 32, 64}
            or int(row["modes"]) not in {8, 16, 32}
            or 2 * int(row["modes"]) > int(row["model_dim"])
            or int(row["trial"]) not in {2, 4, 6}
        ):
            raise RuntimeError(f"invalid frozen task configuration: {key}")
    return selected, _sha256(stage2_path)


def _audit_freeze_and_contract(
    root: Path,
    selected: dict[str, dict[str, Any]],
    source_manifest_sha256: str,
) -> dict[str, str]:
    freeze_path = root / "architecture_freeze.json"
    contract_path = root / "final/contract.json"
    freeze = _read(freeze_path)
    contract = _read(contract_path)
    if (
        freeze.get("schema") != "pac_alphabet_q1_final_freeze.v1"
        or freeze.get("chosen_internal_model") != CANDIDATE
        or not isinstance(freeze.get("public_models"), list)
        or len(freeze["public_models"]) != len(MODELS)
        or set(freeze["public_models"]) != set(MODELS)
        or freeze.get("selection_seeds") != list(SELECTION_SEEDS)
        or freeze.get("final_seeds") != list(SEEDS)
        or freeze.get("selected") != selected
        or freeze.get("test_evidence_used_for_architecture_choice") is not False
        or freeze.get("source_manifest_sha256") != source_manifest_sha256
    ):
        raise RuntimeError("architecture freeze does not match the audited Q1 selection")
    if (
        contract.get("schema") != "pac_two_tap_q1_final_contract.v1"
        or contract.get("public_model") != "ALPHABET"
        or contract.get("chosen_internal_model") != CANDIDATE
        or int(contract.get("tasks", 0)) != 30
        or int(contract.get("models", 0)) != 7
        or contract.get("final_seeds") != list(SEEDS)
        or int(contract.get("jobs", 0)) != 1050
        or int(contract.get("reused_frozen_baseline_rows", 0)) != 900
        or int(contract.get("new_alphabet_jobs", 0)) != 150
    ):
        raise RuntimeError("Q1-final contract is incomplete or inconsistent")
    return {
        "architecture_freeze_sha256": _sha256(freeze_path),
        "final_contract_sha256": _sha256(contract_path),
    }


def _load_final_rows(root: Path) -> list[dict[str, Any]]:
    completed = root / "final/completed"
    rows = [_read(path) for path in sorted(completed.glob("*.json"))]
    failures = {path.name for path in (root / "final/failed").glob("*.json")} - {
        path.name for path in completed.glob("*.json")
    }
    if failures:
        raise RuntimeError(f"Q1-final has {len(failures)} unresolved failures")
    expected_count = len(_expected_tasks()) * len(MODELS) * len(SEEDS)
    if len(rows) != expected_count:
        raise RuntimeError(f"Q1-final coverage is {len(rows)}/{expected_count}")
    return rows


def _audit_provenance(
    root: Path,
    row: dict[str, Any],
    audited: set[str],
) -> None:
    provenance_sha256 = str(row.get("provenance_sha256", ""))
    if provenance_sha256 in audited:
        return
    provenance_path = root / "provenance" / f"{provenance_sha256}.json"
    if len(provenance_sha256) != 64 or not provenance_path.is_file():
        raise RuntimeError(f"missing provenance: {row['job_key']}")
    provenance = _read(provenance_path)
    claimed = str(provenance.pop("provenance_sha256", ""))
    canonical = json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode()
    recomputed = hashlib.sha256(canonical).hexdigest()
    if claimed != provenance_sha256 or recomputed != provenance_sha256:
        raise RuntimeError(f"invalid provenance digest: {provenance_path}")
    audited.add(provenance_sha256)


def _audit_rows(
    root: Path,
    rows: list[dict[str, Any]],
    selected: dict[str, dict[str, Any]],
    selection_sha256: str,
) -> tuple[dict[tuple[str, str, str], list[dict[str, Any]]], set[str], str]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    logical: set[tuple[str, str, str, int]] = set()
    candidate_code_hashes: set[str] = set()
    audited_provenance: set[str] = set()
    final_manifest_hashes = _manifest_hashes(root, "final")
    for row in rows:
        suite = str(row.get("suite"))
        dataset = str(row.get("dataset"))
        model = str(row.get("model"))
        seed = int(row.get("train_seed", -1))
        cell = (suite, dataset, model, seed)
        if (
            (suite, dataset) not in _expected_tasks()
            or model not in MODELS
            or seed not in SEEDS
            or int(row.get("split_seed", -1)) != seed
            or row.get("stage") != "final"
            or row.get("status") != "done"
            or row.get("evaluation_split") != "test"
            or row.get("test_evaluated") is not True
            or row.get("official_test_accessed") is not True
            or cell in logical
        ):
            raise RuntimeError(f"invalid or duplicate Q1-final row: {row.get('job_key')}")
        logical.add(cell)
        metric, _lower = _metric(suite, dataset)
        value = float(row[metric])
        if not math.isfinite(value):
            raise RuntimeError(f"nonfinite {metric}: {row['job_key']}")
        _audit_provenance(root, row, audited_provenance)
        if model == CANDIDATE:
            frozen = selected[f"{suite}:{dataset}:{CANDIDATE}"]
            for actual, expected in (
                (int(row["width"]), int(frozen["model_dim"])),
                (int(row["modes"]), int(frozen["modes"])),
                (int(row["trial"]), int(frozen["trial"])),
            ):
                if actual != expected:
                    raise RuntimeError(f"candidate configuration drift: {row['job_key']}")
            if str(row["config_key"]) != str(frozen["config_key"]):
                raise RuntimeError(f"candidate config-key drift: {row['job_key']}")
            if str(row.get("selection_artifact_sha256")) != selection_sha256:
                raise RuntimeError(f"candidate selection hash drift: {row['job_key']}")
            if str(row.get("manifest_sha256", "")) not in final_manifest_hashes:
                raise RuntimeError(f"candidate manifest hash drift: {row['job_key']}")
            candidate_code_hashes.add(str(row.get("code_sha256", "")))
        grouped[(suite, dataset, model)].append(row)
    expected_logical = {
        (suite, dataset, model, seed)
        for suite, dataset in _expected_tasks()
        for model in MODELS
        for seed in SEEDS
    }
    if logical != expected_logical:
        raise RuntimeError("Q1-final rows do not form the exact task/model/seed product")
    if len(candidate_code_hashes) != 1 or "" in candidate_code_hashes:
        raise RuntimeError("candidate TEST rows mix runner implementations")
    if any(len(cell_rows) != len(SEEDS) for cell_rows in grouped.values()):
        raise RuntimeError("Q1-final contains an incomplete five-seed cell")
    return grouped, audited_provenance, next(iter(candidate_code_hashes))


def _attempt_audit(root: Path) -> dict[str, int]:
    events: dict[str, set[str]] = defaultdict(set)
    counts: dict[str, int] = defaultdict(int)
    for path in (root / "final/attempts").rglob("*.json"):
        row = _read(path)
        event = str(row.get("event"))
        if event not in {"started", "failed", "succeeded", "abandoned"}:
            raise RuntimeError(f"invalid final attempt event: {path}")
        counts[event] += 1
        events[str(row["attempt_id"])].add(event)
    unfinished = sum(
        "started" in states and not ({"failed", "succeeded", "abandoned"} & states)
        for states in events.values()
    )
    if unfinished:
        raise RuntimeError(f"Q1-final has {unfinished} unfinished attempts")
    return {**counts, "unfinished": unfinished}


def summarize(root: Path) -> dict[str, Any]:
    scope = _audit_scope(root)
    source_manifest_sha256, source_hashes = _audit_source_manifest(root)
    selected, selection_sha256 = _audit_selection(root)
    validation_ledgers = _audit_validation_ledgers(root)
    contract_hashes = _audit_freeze_and_contract(root, selected, source_manifest_sha256)
    rows = _load_final_rows(root)
    grouped, audited_provenance, candidate_code_sha256 = _audit_rows(
        root, rows, selected, selection_sha256
    )
    expected_runner_sha256 = source_hashes.get("pac_baseline_fairness_maximal.py")
    if (
        validation_ledgers["runner_code_sha256"] != expected_runner_sha256
        or candidate_code_sha256 != expected_runner_sha256
    ):
        raise RuntimeError("result runner hash differs from the frozen source manifest")

    per_task: dict[str, dict[str, Any]] = {}
    rank_vectors: dict[str, list[float]] = {model: [] for model in MODELS}
    top_counts = dict.fromkeys(MODELS, 0)
    sole_top_counts = dict.fromkeys(MODELS, 0)
    parameters: dict[str, list[int]] = {model: [] for model in MODELS}
    for suite, dataset in sorted(_expected_tasks()):
        metric, lower = _metric(suite, dataset)
        means: dict[str, float] = {}
        sample_sds: dict[str, float] = {}
        for model in MODELS:
            cell = grouped[(suite, dataset, model)]
            values = [float(row[metric]) for row in cell]
            means[model] = mean(values)
            sample_sds[model] = stdev(values)
            cell_parameters = {int(row["params_trainable"]) for row in cell}
            if len(cell_parameters) != 1:
                raise RuntimeError(
                    f"parameter count varies across final seeds: {suite}/{dataset}/{model}"
                )
            parameters[model].append(next(iter(cell_parameters)))
        ranks = _average_tie_ranks(means, lower=lower)
        best = min(means.values()) if lower else max(means.values())
        winners = [
            model
            for model, value in means.items()
            if math.isclose(value, best, rel_tol=RTOL, abs_tol=ATOL)
        ]
        for model in MODELS:
            rank_vectors[model].append(ranks[model])
            if model in winners:
                top_counts[model] += 1
                sole_top_counts[model] += int(len(winners) == 1)
        per_task[f"{suite}:{dataset}"] = {
            "metric": metric,
            "lower_is_better": lower,
            "means": means,
            "sample_sds": sample_sds,
            "ranks": ranks,
            "joint_top_models": winners,
        }

    mean_ranks = {model: mean(values) for model, values in rank_vectors.items()}
    friedman = friedmanchisquare(*(rank_vectors[model] for model in MODELS))
    raw_p: dict[str, float] = {}
    for model in MODELS[1:]:
        result = wilcoxon(
            rank_vectors[CANDIDATE],
            rank_vectors[model],
            alternative="two-sided",
            zero_method="pratt",
        )
        raw_p[model] = float(result.pvalue)
    adjusted = _holm(raw_p)

    aggregate = {
        model: {
            "mean_rank": mean_ranks[model],
            "joint_top1": top_counts[model],
            "sole_top1": sole_top_counts[model],
            "median_params": median(parameters[model]),
            "min_params": min(parameters[model]),
            "max_params": max(parameters[model]),
        }
        for model in MODELS
    }
    return {
        "schema": "pac_two_tap_q1_final_audit.v1",
        "status": "complete",
        "public_model": "ALPHABET",
        "chosen_internal_model": CANDIDATE,
        "scope_decision_sha256": _sha256(root / "study_scope_decision.json"),
        "source_manifest_sha256": source_manifest_sha256,
        "selection_artifact_sha256": selection_sha256,
        **contract_hashes,
        "q2_excluded": "Q2-final TEST evaluation" in scope["excluded_scope"],
        "rows": len(rows),
        "tasks": len(_expected_tasks()),
        "models": len(MODELS),
        "seeds": list(SEEDS),
        "provenance_records": len(audited_provenance),
        "validation_ledgers": validation_ledgers,
        "tie_tolerance": {"rtol": RTOL, "atol": ATOL},
        "aggregate": aggregate,
        "friedman": {"statistic": float(friedman.statistic), "pvalue": float(friedman.pvalue)},
        "pairwise_rank_wilcoxon": {
            model: {"raw_pvalue": raw_p[model], "holm_adjusted_pvalue": adjusted[model]}
            for model in MODELS[1:]
        },
        "attempts": _attempt_audit(root),
        "per_task": per_task,
    }


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-root",
        type=Path,
        default=Path(".omx/results/pac-two-tap-q1-final-20260720"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = summarize(args.campaign_root)
    output = args.output or args.campaign_root / "audit/q1_final_audit.json"
    _atomic_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
