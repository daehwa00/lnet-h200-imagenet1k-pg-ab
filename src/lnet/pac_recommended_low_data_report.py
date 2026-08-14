from __future__ import annotations

import json
import math
from collections import defaultdict
from statistics import median
from typing import TYPE_CHECKING, cast

from .pac_overnight_io import read_csv

if TYPE_CHECKING:
    from pathlib import Path

    from .pac_confirmatory_baselines import ConfirmatoryFamily


def write_low_data_report(root: Path) -> None:
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    rows = read_csv(root / "results" / "low_data_recommended_real.csv")
    lines = ["# PAC Recommended Low-Data Summary", "", _queue_status(root), ""]
    lines.extend(("# Rows", "", f"- completed: {sum(row.get('status') == 'done' for row in rows)}"))
    lines.extend(("", "## Mean Accuracy By Ratio", ""))
    lines.extend(_ratio_table(rows))
    lines.extend(("", "## Low-Data AUC", ""))
    lines.extend(_auc_table(rows))
    (reports / "low_data_recommended_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    _write_validation_capacity_selection(root, rows)
    _write_confirmatory_baseline_selection(root, rows)


def _write_confirmatory_baseline_selection(
    root: Path,
    rows: list[dict[str, str]],
) -> None:
    relevant = [
        row
        for row in rows
        if row.get("status") == "done"
        and row.get("evaluation_collection") == "unseen_final_validation"
        and row.get("validation_balanced_accuracy", "").strip()
    ]
    if not relevant:
        return
    expected = _manifest_keys_for_collection(root, "unseen_final_validation")
    completed = {row.get("job_key", "") for row in relevant}
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    best_epochs: dict[tuple[str, int], list[int]] = defaultdict(list)
    trial_contract: dict[tuple[str, int], tuple[float, float]] = {}
    for row in relevant:
        family = row.get("baseline_family", "")
        trial = int(row["validation_trial"])
        grouped[(family, trial)].append(float(row["validation_balanced_accuracy"]))
        epoch = int(row.get("best_epoch", "0") or 0)
        if epoch > 0:
            best_epochs[(family, trial)].append(epoch)
        trial_contract[(family, trial)] = (
            float(row["learning_rate"]),
            float(row["weight_decay"]),
        )
    selected: dict[str, dict[str, object]] = {}
    families = sorted({family for family, _ in grouped})
    reference_models = {
        value for row in relevant if (value := row.get("reference_model", "").strip())
    }
    for family in families:
        candidates = [
            (sum(values) / len(values), trial)
            for (candidate_family, trial), values in grouped.items()
            if candidate_family == family
        ]
        _, winner_trial = max(candidates, key=lambda item: (item[0], -item[1]))
        learning_rate, weight_decay = trial_contract[(family, winner_trial)]
        winner_epochs = best_epochs[(family, winner_trial)]
        complete_epoch_grid = len(winner_epochs) == len(grouped[(family, winner_trial)])
        selected[family] = {
            "trial": winner_trial,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "refit_epochs": (int(median(winner_epochs) + 0.5) if complete_epoch_grid else 0),
            "architecture": _confirmatory_architecture(family, winner_trial),
        }
    complete = (
        bool(expected)
        and expected.issubset(completed)
        and all(
            len({trial for candidate_family, trial in grouped if candidate_family == family}) == 6
            for family in families
        )
        and len(reference_models) == 1
        and all(
            isinstance(epoch := row["refit_epochs"], int) and epoch > 0 for row in selected.values()
        )
    )
    protocol_sha256 = _validation_protocol_sha256(root)
    payload = {
        "schema_version": "pac_confirmatory_baseline_selection.v1",
        "status": "complete" if complete else "partial",
        "selection_split": "official_train_stratified_validation",
        "primary_score": "macro_mean_validation_balanced_accuracy",
        "refit_epoch_policy": (
            "per-family median best_epoch over the winning trial's dataset-by-seed runs; "
            "round half upward"
        ),
        "expected_jobs": len(expected),
        "completed_jobs": len(expected & completed),
        "families": families,
        "trials_per_family": 6,
        "trial_policy": (
            "six predeclared family-specific architecture/training candidates; "
            "selected trial is reproduced unchanged for full-TRAIN refit and P1/P2"
        ),
        "reference_model": next(iter(reference_models)) if len(reference_models) == 1 else None,
        "selected_trials": selected if complete else {},
    }
    if protocol_sha256 is not None:
        payload["protocol_sha256"] = protocol_sha256
    reports = root / "reports"
    (reports / "confirmatory_baseline_selection.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _confirmatory_architecture(family: str, trial: int) -> dict[str, object]:
    from .pac_confirmatory_baselines import (  # noqa: PLC0415
        confirmatory_implementation_metadata,
    )

    return confirmatory_implementation_metadata(
        cast("ConfirmatoryFamily", family),
        trial,
    )


def _manifest_keys_for_collection(root: Path, collection: str) -> set[str]:
    candidates = (
        root / "queue_manifest.jsonl",
        root / "reports" / f"{collection}_manifest.jsonl",
    )
    for manifest in candidates:
        if not manifest.exists():
            continue
        keys = {
            str(row["key"])
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
            for row in (json.loads(line),)
            if row.get("evaluation_collection") == collection
        }
        if keys:
            return keys
    return set()


def _validation_protocol_sha256(root: Path) -> str | None:
    lock_path = root / "reports" / "unseen_final_validation_collection_lock.json"
    if not lock_path.exists():
        return None
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    value = payload.get("protocol_sha256")
    return value if isinstance(value, str) and len(value) == 64 else None


def _write_validation_capacity_selection(
    root: Path,
    rows: list[dict[str, str]],
) -> None:
    selection_rows = [
        row
        for row in rows
        if row.get("status") == "done"
        and row.get("evaluation_split") == "validation"
        and row.get("validation_balanced_accuracy", "").strip()
    ]
    if not selection_rows:
        return
    grouped: dict[str, list[tuple[float, int]]] = defaultdict(list)
    completed_keys: set[str] = set()
    for row in selection_rows:
        try:
            accuracy = float(row["validation_balanced_accuracy"])
            parameters = int(row["params_trainable"])
        except (KeyError, ValueError):
            continue
        model = row.get("model", "").strip()
        if not model:
            continue
        grouped[model].append((accuracy, parameters))
        completed_keys.add(row.get("job_key", ""))
    candidates = [
        {
            "model": model,
            "mean_validation_balanced_accuracy": sum(score for score, _ in values) / len(values),
            "params_trainable": round(sum(params for _, params in values) / len(values)),
            "rows": len(values),
        }
        for model, values in grouped.items()
    ]
    candidates.sort(
        key=lambda row: (
            -float(row["mean_validation_balanced_accuracy"]),
            str(row["model"]),
        )
    )
    expected_keys = _selection_manifest_keys(root)
    complete = bool(expected_keys) and expected_keys.issubset(completed_keys)
    selected_model = None
    if complete and candidates:
        best_score = float(candidates[0]["mean_validation_balanced_accuracy"])
        near_best = [
            row
            for row in candidates
            if best_score - float(row["mean_validation_balanced_accuracy"]) <= 0.005
        ]
        winner = min(
            near_best,
            key=lambda row: (int(row["params_trainable"]), str(row["model"])),
        )
        selected_model = str(winner["model"])
    payload = {
        "schema_version": "pac_validation_capacity_selection.v1",
        "selection_split": "validation",
        "official_test_observed": False,
        "data_protocol": "stratified_split_before_normalization",
        "normalization_fit": "optimization_fold_only",
        "checkpoint_policy": "best_validation_loss",
        "status": "complete" if complete else "partial",
        "expected_jobs": len(expected_keys),
        "completed_jobs": len(expected_keys & completed_keys),
        "selected_model": selected_model,
        "primary_score": "macro_mean_validation_balanced_accuracy",
        "tie_margin_absolute": 0.005,
        "tie_breaks": ["within_0.005_choose_params_asc", "model_name_asc"],
        "candidates": candidates,
    }
    reports = root / "reports"
    (reports / "stiefel_validation_capacity_selection.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Validation-Only Capacity Selection",
        "",
        f"- status: {payload['status']}",
        "- selection split: official TRAIN-derived validation",
        "- official TEST observed: no",
        f"- selected model: {selected_model or '[not locked: queue incomplete]'}",
        "",
        "| Rank | Model | Mean Validation Balanced Accuracy | Parameters | Rows |",
        "|---:|---|---:|---:|---:|",
    ]
    lines.extend(
        " | ".join(
            (
                "",
                str(rank),
                str(row["model"]),
                f"{float(row['mean_validation_balanced_accuracy']):.4f}",
                str(int(row["params_trainable"])),
                str(int(row["rows"])),
                "",
            )
        )
        for rank, row in enumerate(candidates, start=1)
    )
    (reports / "stiefel_validation_capacity_selection.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _selection_manifest_keys(root: Path) -> set[str]:
    manifest = root / "queue_manifest.jsonl"
    if not manifest.exists():
        return set()
    keys: set[str] = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("evaluation_split") == "validation":
            keys.add(str(row.get("key")))
    return keys


def _ratio_table(rows: list[dict[str, str]]) -> list[str]:
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        parsed = _parse_done_row(row)
        if parsed is None:
            continue
        model, ratio, accuracy = parsed
        groups[(model, ratio)].append(accuracy)
    lines = ["| Model | Ratio | Mean Accuracy | Rows |", "|---|---:|---:|---:|"]
    ordered = sorted(groups.items(), key=lambda item: (item[0][0], float(item[0][1])))
    for (model, ratio), values in ordered:
        mean = sum(values) / len(values)
        lines.append(f"| {model} | {float(ratio):.2f} | {mean:.4f} | {len(values)} |")
    return lines


def _auc_table(rows: list[dict[str, str]]) -> list[str]:
    per_model: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        parsed = _parse_done_row(row)
        if parsed is None:
            continue
        model, ratio, accuracy = parsed
        per_model[model][float(ratio)].append(accuracy)
    scored: list[tuple[float, str, int]] = []
    for model, ratios in per_model.items():
        points = sorted((ratio, sum(values) / len(values)) for ratio, values in ratios.items())
        if len(points) >= 2:
            scored.append((_log_auc(points), model, sum(len(values) for values in ratios.values())))
    lines = ["| Model | Log-Ratio AUC | Rows |", "|---|---:|---:|"]
    for auc, model, count in sorted(scored, reverse=True):
        lines.append(f"| {model} | {auc:.4f} | {count} |")
    return lines


def _log_auc(points: list[tuple[float, float]]) -> float:
    xs = [math.log(max(ratio, 1.0e-12)) for ratio, _ in points]
    ys = [value for _, value in points]
    area = sum((xs[i] - xs[i - 1]) * (ys[i] + ys[i - 1]) / 2 for i in range(1, len(xs)))
    width = xs[-1] - xs[0]
    return area / max(width, 1.0e-12)


def _parse_done_row(row: dict[str, str]) -> tuple[str, str, float] | None:
    if row.get("status") != "done":
        return None
    model = row.get("model", "").strip()
    ratio = row.get("data_ratio", "").strip()
    accuracy = row.get("test_accuracy", "").strip()
    if not model or not ratio or not accuracy:
        return None
    try:
        parsed_accuracy = float(accuracy)
        float(ratio)
    except ValueError:
        return None
    return model, ratio, parsed_accuracy


def _queue_status(root: Path) -> str:
    manifest = root / "queue_manifest.jsonl"
    state = root / "queue_state.jsonl"
    if not manifest.exists() or not state.exists():
        return "overall_status: partial"
    latest: dict[str, str] = {}
    for line in state.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            latest[str(row.get("key"))] = str(row.get("status"))
    statuses = [
        latest.get(str(json.loads(line)["key"]), "pending")
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if statuses and all(status == "done" for status in statuses):
        return "overall_status: complete"
    if any(status == "failed" for status in statuses):
        return "overall_status: partial_failed"
    return "overall_status: running_or_pending"
