from __future__ import annotations

import json
import math
import shutil
from dataclasses import replace
from pathlib import Path
from statistics import median
from typing import Final, cast

from .pac_campaign_utils import canonical_json_sha256, write_once
from .pac_baseline_fairness_maximal import (
    BASELINES,
    FINAL_SEEDS,
    PARAMETER_TOLERANCE,
    Q2_BUDGET_MULTIPLIERS,
    Q2_LR_MULTIPLIERS,
    SEARCH_SEED,
    FairnessJob,
    ResourceLane,
    _job_from_result,  # pyright: ignore[reportPrivateUsage]
    _manifest_job_keys,  # pyright: ignore[reportPrivateUsage]
    _write_manifests,  # pyright: ignore[reportPrivateUsage]
    campaign_status,
    enqueue_q2_calibration,
    select_q2_calibration,
)
from .pac_efp16_final_campaign import EXTERNAL_DATASETS, UCR_DATASETS
from .pac_efp_compact_equal_search import (
    COMPATIBLE_SELECTION_RUNNER_SHA256,
    default_lanes,
)
from .pac_efp_compact_external_equal_search import SOURCE_MANIFEST_FILES

DEFAULT_ROOT: Final = Path(".omx/results/pac-alphabet-q1q2-final-20260719")
DEFAULT_UCR_SEARCH_ROOT: Final = Path(".omx/results/pac-efp-compact-equal-search-20260719")
DEFAULT_EXTERNAL_SEARCH_ROOT: Final = Path(
    ".omx/results/pac-efp-compact-external-equal-search-20260719"
)
DEFAULT_BASELINE_ROOT: Final = Path(".omx/results/pac-baseline-fairness-maximal-20260714")
STAGES: Final = ("final", "q2_calibration", "q2_final")
CANDIDATES: Final = ("compact_h_only", "efp_tuned")
ARCHITECTURE_SELECTION_RULE: Final = "global top count, then mean rank"
PAPER_TIE_TOLERANCE: Final = {"rtol": 1.0e-5, "atol": 1.0e-8}


def _completed_rows(root: Path, stage: str) -> list[dict[str, object]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((root / stage / "completed").glob("*.json"))
    ]


def _safe_result_name(job_key: str) -> str:
    return job_key.replace(":", "__").replace("/", "_") + ".json"


def _validate_architecture_comparison(comparison: dict[str, object]) -> str:
    """Return the candidate only after recomputing the declared 30-task rule."""
    candidate = str(comparison.get("provisional_champion"))
    summaries = cast("dict[str, dict[str, object]]", comparison.get("global_summary"))
    pairwise = cast("dict[str, object]", comparison.get("pairwise"))
    if (
        comparison.get("schema") != "pac_efp_compact_equal_search_30_task_comparison.v1"
        or candidate not in CANDIDATES
        or bool(comparison.get("official_test_accessed"))
        or int(cast("int", comparison.get("tasks", 0))) != 30
        or int(cast("int", comparison.get("ucr_tasks", 0))) != len(UCR_DATASETS)
        or int(cast("int", comparison.get("external_tasks", 0))) != len(EXTERNAL_DATASETS)
        or comparison.get("selection_rule") != ARCHITECTURE_SELECTION_RULE
        or comparison.get("tie_tolerance") != PAPER_TIE_TOLERANCE
        or not isinstance(summaries, dict)
        or set(summaries) != set(CANDIDATES)
        or not isinstance(pairwise, dict)
        or sum(int(pairwise.get(key, -31)) for key in ("compact_wins", "ties", "efp_wins"))
        != 30
    ):
        message = "architecture source is not the sealed 30-task comparison"
        raise RuntimeError(message)
    for model in CANDIDATES:
        summary = summaries[model]
        top_count = int(summary["global_top_count"])
        mean_rank = float(summary["mean_rank_vs_six_baselines"])
        if not 0 <= top_count <= 30 or not math.isfinite(mean_rank):
            message = f"invalid architecture summary for {model}"
            raise RuntimeError(message)
    recomputed = min(
        CANDIDATES,
        key=lambda model: (
            -int(summaries[model]["global_top_count"]),
            float(summaries[model]["mean_rank_vs_six_baselines"]),
            model,
        ),
    )
    if recomputed != candidate:
        message = (
            f"declared architecture champion {candidate} "
            f"disagrees with rule winner {recomputed}"
        )
        raise RuntimeError(message)
    return candidate


def _validate_source_manifest(
    comparison: dict[str, object],
    manifest_path: Path,
) -> str:
    """Validate the sealed source ledger referenced by the 30-task comparison."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_hashes = manifest.get("source_sha256")
    compatibility_scope = manifest.get("runner_compatibility_scope")
    if (
        manifest.get("schema") != "pac_efp_compact_30_task_source_manifest.v2"
        or not isinstance(source_hashes, dict)
        or set(source_hashes) != set(SOURCE_MANIFEST_FILES)
        or manifest.get("compatible_selection_runner_sha256")
        != sorted(COMPATIBLE_SELECTION_RUNNER_SHA256)
        or not isinstance(compatibility_scope, dict)
        or compatibility_scope.get("changed_functions")
        != ["_public_models", "_build_model_from_metadata"]
        or compatibility_scope.get("equal_search_stage1_or_stage2_training_path_affected")
        is not False
        or compatibility_scope.get("pre_extension_hash_reconstructed_from_recorded_patch")
        is not True
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in source_hashes.values()
        )
    ):
        message = "architecture source manifest is incomplete or malformed"
        raise RuntimeError(message)
    body = {key: value for key, value in manifest.items() if key != "sha256"}
    digest = canonical_json_sha256(body)
    if (
        manifest.get("sha256") != digest
        or comparison.get("source_manifest_sha256") != digest
    ):
        message = "architecture comparison and source manifest digests disagree"
        raise RuntimeError(message)
    return digest


def _candidate_base_rows(
    candidate: str,
    ucr_root: Path,
    external_root: Path,
) -> tuple[dict[str, FairnessJob], dict[str, dict[str, object]]]:
    frozen: dict[str, FairnessJob] = {}
    selected_metadata: dict[str, dict[str, object]] = {}
    for suite, datasets, root in (
        ("ucr", UCR_DATASETS, ucr_root),
        ("external", EXTERNAL_DATASETS, external_root),
    ):
        selection = json.loads((root / "stage2" / "selection.json").read_text(encoding="utf-8"))
        selected = cast("dict[str, dict[str, object]]", selection["selected"])
        expected_search_keys = {
            f"{suite}:{dataset}:{model}" for dataset in datasets for model in CANDIDATES
        }
        if bool(selection.get("official_test_accessed")) or set(selected) != expected_search_keys:
            message = f"{suite} candidate selection is incomplete or TEST-accessed"
            raise RuntimeError(message)
        rows = _completed_rows(root, "stage1") + _completed_rows(root, "stage2")
        by_cell_config: dict[tuple[str, str], list[dict[str, object]]] = {}
        for row in rows:
            by_cell_config.setdefault((str(row["cell_key"]), str(row["config_key"])), []).append(
                row
            )
        for dataset in datasets:
            cell_key = f"{suite}:{dataset}:{candidate}"
            config_key = str(selected[cell_key]["config_key"])
            config_rows = by_cell_config[(cell_key, config_key)]
            if (
                len(config_rows) != 3
                or {int(row["train_seed"]) for row in config_rows} != {7, 11, 19}
                or any(
                    row.get("status") != "done"
                    or row.get("evaluation_split") != "validation"
                    or row.get("official_test_accessed") is not False
                    for row in config_rows
                )
            ):
                message = f"incomplete frozen selection rows for {cell_key}/{config_key}"
                raise RuntimeError(message)
            base_row = min(config_rows, key=lambda row: int(row["train_seed"]))
            base = _job_from_result(base_row)
            best_epochs = [
                int(row["best_epoch"]) for row in config_rows if row.get("best_epoch") is not None
            ]
            refit_epochs = (
                max(1, round(median(best_epochs)))
                if suite == "ucr" and best_epochs
                else base.epochs
            )
            frozen[cell_key] = replace(base, epochs=refit_epochs)
            selected_metadata[cell_key] = {
                **selected[cell_key],
                "width": base.width,
                "width_tier": base.width_tier,
                "modes": base.modes,
                "ucr_refit_epochs": refit_epochs if suite == "ucr" else None,
            }
    return frozen, selected_metadata


def _copy_baseline_results(source_root: Path, target_root: Path) -> list[FairnessJob]:
    jobs_and_paths: list[tuple[FairnessJob, Path]] = []
    seen: set[str] = set()
    cells: set[tuple[str, str, str, int]] = set()
    expected_tasks = _expected_task_keys()
    expected_cells = _expected_q1_cells(set(BASELINES))
    for path in sorted((source_root / "final" / "completed").glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("model") not in BASELINES:
            continue
        cell = _validate_test_row(
            row,
            public_models=set(BASELINES),
            expected_tasks=expected_tasks,
            stage="reused Q1 baseline",
        )
        job = _job_from_result(row)
        if job.key in seen:
            message = f"duplicate baseline final key: {job.key}"
            raise RuntimeError(message)
        seen.add(job.key)
        cells.add(cell)
        jobs_and_paths.append((job, path))
    if cells != expected_cells or len(jobs_and_paths) != len(expected_cells):
        message = (
            "baseline final ledger is not the exact Cartesian grid: "
            f"rows={len(jobs_and_paths)}/{len(expected_cells)}, "
            f"missing={len(expected_cells - cells)}, extra={len(cells - expected_cells)}"
        )
        raise RuntimeError(message)
    destination = target_root / "final" / "completed"
    destination.mkdir(parents=True, exist_ok=True)
    for job, path in jobs_and_paths:
        target = destination / _safe_result_name(job.key)
        if not target.exists():
            shutil.copy2(path, target)
    return [job for job, _path in jobs_and_paths]


def enqueue_q1_final(
    root: Path = DEFAULT_ROOT,
    *,
    ucr_search_root: Path = DEFAULT_UCR_SEARCH_ROOT,
    external_search_root: Path = DEFAULT_EXTERNAL_SEARCH_ROOT,
    baseline_root: Path = DEFAULT_BASELINE_ROOT,
    lanes: tuple[ResourceLane, ...] | None = None,
) -> dict[str, object]:
    combined_path = external_search_root / "reports" / "combined_30_task_comparison.json"
    combined = json.loads(combined_path.read_text(encoding="utf-8"))
    candidate = _validate_architecture_comparison(combined)
    source_manifest_path = external_search_root / "reports" / "source_manifest.json"
    source_manifest_sha256 = _validate_source_manifest(combined, source_manifest_path)
    frozen, candidate_selection = _candidate_base_rows(
        candidate,
        ucr_search_root,
        external_search_root,
    )
    candidate_jobs = [
        replace(
            base,
            stage="final",
            split_seed=seed,
            train_seed=seed,
            evaluation_split="test",
        )
        for base in frozen.values()
        for seed in FINAL_SEEDS
    ]
    baseline_jobs = _copy_baseline_results(baseline_root, root)
    all_jobs = baseline_jobs + candidate_jobs
    active_lanes = default_lanes() if lanes is None else lanes
    loads = _write_manifests(root, "final", all_jobs, active_lanes)

    baseline_selection = json.loads(
        (baseline_root / "stage2" / "selection.json").read_text(encoding="utf-8")
    )
    old_selected = cast("dict[str, dict[str, object]]", baseline_selection["selected"])
    selected = {
        key: value for key, value in old_selected.items() if key.rsplit(":", 1)[-1] in BASELINES
    }
    selected.update(candidate_selection)
    selection_payload = {
        "schema": "pac_alphabet_q1_final_freeze.v1",
        "chosen_internal_model": candidate,
        "public_models": [candidate, *BASELINES],
        "selection_seeds": [7, 11, 19],
        "final_seeds": list(FINAL_SEEDS),
        "selected": selected,
        "test_evidence_used_for_architecture_choice": False,
        "source_comparison": str(combined_path),
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": source_manifest_sha256,
    }
    write_once(
        root / "stage2" / "selection.json",
        json.dumps(selection_payload, indent=2, sort_keys=True) + "\n",
    )
    decision = {
        "schema": "pac_alphabet_q1_architecture_decision.v1",
        "chosen_internal_model": candidate,
        "public_model": "ALPHABET",
        "rule": "30-task validation global Top-1 count, then mean rank",
        "test_evidence_used": False,
        "source_comparison": str(combined_path),
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": source_manifest_sha256,
        "global_summary": combined["global_summary"],
    }
    write_once(
        root / "architecture_decision.json",
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
    )
    contract: dict[str, object] = {
        "schema": "pac_alphabet_q1_final_contract.v1",
        "chosen_internal_model": candidate,
        "public_models": [candidate, *BASELINES],
        "tasks": 30,
        "final_seeds": list(FINAL_SEEDS),
        "jobs": len(all_jobs),
        "reused_frozen_baseline_rows": len(baseline_jobs),
        "new_alphabet_jobs": len(candidate_jobs),
        "official_test_access": "allowed only after the recorded validation freeze",
        "source_manifest_sha256": source_manifest_sha256,
        "estimated_normalized_lane_seconds": loads,
    }
    write_once(
        root / "final" / "contract.json",
        json.dumps(contract, indent=2, sort_keys=True) + "\n",
    )
    return contract


def enqueue_q2(
    root: Path = DEFAULT_ROOT,
    *,
    lanes: tuple[ResourceLane, ...] | None = None,
) -> dict[str, object]:
    candidate = _verify_architecture_freeze(root)
    _verify_q1_final(root, candidate)
    return enqueue_q2_calibration(root, lanes=lanes)


def select_q2(
    root: Path = DEFAULT_ROOT,
    *,
    lanes: tuple[ResourceLane, ...] | None = None,
) -> dict[str, object]:
    candidate = _verify_architecture_freeze(root)
    _verify_q1_final(root, candidate)
    _validate_q2_calibration_rows(
        _completed_rows(root, "q2_calibration"),
        candidate,
        expected_job_keys=_manifest_job_keys(root / "q2_calibration/manifests"),
    )
    return select_q2_calibration(root, lanes=lanes)


def _verify_terminal_stages(root: Path) -> None:
    full_status = campaign_status(root)
    for stage in STAGES:
        stage_status = cast("dict[str, object]", full_status[stage])
        if not bool(stage_status["done"]):
            message = f"cannot finalize incomplete {stage}: {stage_status}"
            raise RuntimeError(message)


def _verify_architecture_freeze(root: Path) -> str:
    decision = json.loads((root / "architecture_decision.json").read_text(encoding="utf-8"))
    candidate = str(decision["chosen_internal_model"])
    if (
        decision.get("schema") != "pac_alphabet_q1_architecture_decision.v1"
        or decision.get("public_model") != "ALPHABET"
        or decision.get("rule") != "30-task validation global Top-1 count, then mean rank"
        or candidate not in CANDIDATES
    ):
        message = f"unexpected frozen ALPHABET candidate: {candidate}"
        raise RuntimeError(message)
    if bool(decision["test_evidence_used"]):
        message = "Q1 architecture decision used TEST evidence"
        raise RuntimeError(message)
    source = Path(str(decision["source_comparison"]))
    comparison = json.loads(source.read_text(encoding="utf-8"))
    if _validate_architecture_comparison(comparison) != candidate:
        message = "architecture freeze disagrees with its sealed source comparison"
        raise RuntimeError(message)
    source_manifest = Path(str(decision.get("source_manifest", "")))
    source_manifest_sha256 = _validate_source_manifest(comparison, source_manifest)
    if decision.get("source_manifest_sha256") != source_manifest_sha256:
        message = "architecture freeze disagrees with its source manifest"
        raise RuntimeError(message)
    selection = json.loads((root / "stage2/selection.json").read_text(encoding="utf-8"))
    selected = cast("dict[str, dict[str, object]]", selection["selected"])
    public_models = {candidate, *BASELINES}
    expected_selection_keys = {
        f"{task}:{model}" for task in _expected_task_keys() for model in public_models
    }
    if (
        selection["chosen_internal_model"] != candidate
        or selection.get("schema") != "pac_alphabet_q1_final_freeze.v1"
        or selection.get("public_models") != [candidate, *BASELINES]
        or selection.get("selection_seeds") != [7, 11, 19]
        or selection.get("final_seeds") != list(FINAL_SEEDS)
        or bool(selection["test_evidence_used_for_architecture_choice"])
        or selection.get("source_manifest") != str(source_manifest)
        or selection.get("source_manifest_sha256") != source_manifest_sha256
        or set(selected) != expected_selection_keys
    ):
        message = "Q1 public selection does not match the architecture freeze"
        raise RuntimeError(message)
    return candidate


def _expected_task_keys() -> set[str]:
    return {
        *(f"ucr:{dataset}" for dataset in UCR_DATASETS),
        *(f"external:{dataset}" for dataset in EXTERNAL_DATASETS),
    }


def _expected_q1_cells(public_models: set[str]) -> set[tuple[str, str, str, int]]:
    return {
        (suite, dataset, model, seed)
        for suite, datasets in (("ucr", UCR_DATASETS), ("external", EXTERNAL_DATASETS))
        for dataset in datasets
        for model in public_models
        for seed in FINAL_SEEDS
    }


def _expected_q2_cells(public_models: set[str]) -> set[tuple[str, str, str, float]]:
    return {
        (suite, dataset, model, multiplier)
        for suite, datasets in (("ucr", UCR_DATASETS), ("external", EXTERNAL_DATASETS))
        for dataset in datasets
        for model in public_models
        for multiplier in (0.5, 1.0, 2.0, 4.0)
    }


def _validate_q2_calibration_rows(
    rows: list[dict[str, object]],
    candidate: str,
    *,
    expected_job_keys: set[str] | None = None,
) -> set[tuple[str, str, str, float, float]]:
    """Reject contaminated, duplicated, or out-of-contract Q2 selection rows."""
    public_models = {candidate, *BASELINES}
    expected_tasks = _expected_task_keys()
    allowed_budgets = set(Q2_BUDGET_MULTIPLIERS)
    allowed_learning_rates = set(Q2_LR_MULTIPLIERS)
    job_keys: set[str] = set()
    cells: set[tuple[str, str, str, float, float]] = set()
    for row in rows:
        key = str(row["job_key"])
        if key in job_keys:
            message = f"duplicate Q2 calibration key: {key}"
            raise RuntimeError(message)
        job_keys.add(key)
        suite = str(row["suite"])
        dataset = str(row["dataset"])
        model = str(row["model"])
        budget = float(row["budget_multiplier"])
        lr_multiplier = float(row["lr_multiplier"])
        if (
            row.get("status") != "done"
            or row.get("stage") != "q2_calibration"
            or row.get("evaluation_split") != "validation"
            or row.get("test_evaluated") is not False
            or row.get("official_test_accessed") is not False
            or model not in public_models
            or f"{suite}:{dataset}" not in expected_tasks
            or budget not in allowed_budgets
            or lr_multiplier not in allowed_learning_rates
            or int(cast("int", row["split_seed"])) != SEARCH_SEED
            or int(cast("int", row["train_seed"])) != SEARCH_SEED
        ):
            message = f"invalid or TEST-contaminated Q2 calibration row: {key}"
            raise RuntimeError(message)
        target = int(cast("int", row["target_parameters"]))
        parameters = int(cast("int", row["params_trainable"]))
        reported_error = float(row["relative_parameter_error"])
        recomputed_error = abs(parameters - target) / max(target, 1)
        if (
            target <= 0
            or not math.isclose(
                reported_error,
                recomputed_error,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or reported_error > PARAMETER_TOLERANCE + 1.0e-12
        ):
            message = f"invalid Q2 parameter match in calibration row: {key}"
            raise RuntimeError(message)
        cell = (suite, dataset, model, budget, lr_multiplier)
        if cell in cells:
            message = f"duplicate Q2 calibration model-budget-LR cell: {cell}"
            raise RuntimeError(message)
        cells.add(cell)
    if expected_job_keys is not None and not job_keys <= expected_job_keys:
        unexpected = sorted(job_keys - expected_job_keys)
        message = f"Q2 calibration rows are absent from the sealed manifest: {unexpected[:3]}"
        raise RuntimeError(message)
    return cells


def _parse_q2_selection_cell(selection_key: str) -> tuple[str, str, str, float]:
    cell_key, budget = selection_key.rsplit(":budget", 1)
    suite, dataset, model = cell_key.split(":", 2)
    return suite, dataset, model, float(budget)


def _validate_q2_seed_groups(
    groups: dict[tuple[str, str, str, float], set[int]],
    selected_cells: set[tuple[str, str, str, float]],
) -> None:
    if set(groups) != selected_cells or any(
        seeds != set(FINAL_SEEDS) for seeds in groups.values()
    ):
        message = "Q2 final selected-cell seed coverage is incomplete"
        raise RuntimeError(message)


def _validate_test_row(
    row: dict[str, object],
    *,
    public_models: set[str],
    expected_tasks: set[str],
    stage: str,
) -> tuple[str, str, str, int]:
    key = str(row["job_key"])
    if (
        row.get("status") != "done"
        or row.get("evaluation_split") != "test"
        or row.get("test_evaluated") is not True
        or row.get("official_test_accessed") is not True
    ):
        message = f"invalid {stage} TEST row: {key}"
        raise RuntimeError(message)
    suite = str(row["suite"])
    dataset = str(row["dataset"])
    model = str(row["model"])
    seed = int(cast("int", row["train_seed"]))
    if (
        model not in public_models
        or f"{suite}:{dataset}" not in expected_tasks
        or seed not in FINAL_SEEDS
    ):
        message = f"out-of-contract {stage} row: {key}"
        raise RuntimeError(message)
    return suite, dataset, model, seed


def _verify_q1_final(root: Path, candidate: str) -> int:
    public_models = {candidate, *BASELINES}
    expected_tasks = _expected_task_keys()
    expected_cells = _expected_q1_cells(public_models)
    q1_contract = json.loads((root / "final/contract.json").read_text(encoding="utf-8"))
    q1_rows = _completed_rows(root, "final")
    expected_q1 = len(expected_cells)
    if int(q1_contract["jobs"]) != expected_q1 or len(q1_rows) != expected_q1:
        message = f"Q1 final coverage is {len(q1_rows)}/{expected_q1}"
        raise RuntimeError(message)
    q1_keys: set[str] = set()
    q1_cells: set[tuple[str, str, str, int]] = set()
    for row in q1_rows:
        key = str(row["job_key"])
        if key in q1_keys:
            message = f"duplicate Q1 final key: {key}"
            raise RuntimeError(message)
        q1_keys.add(key)
        q1_cells.add(
            _validate_test_row(
                row,
                public_models=public_models,
                expected_tasks=expected_tasks,
                stage="Q1",
            )
        )
    if q1_cells != expected_cells:
        message = "Q1 final Cartesian grid is incomplete"
        raise RuntimeError(message)
    return len(q1_rows)


def _verify_q2_final(root: Path, candidate: str) -> tuple[int, int, float, dict[str, object]]:
    public_models = {candidate, *BASELINES}
    expected_tasks = _expected_task_keys()
    expected_cells = _expected_q2_cells(public_models)
    q2_contract = json.loads((root / "q2_calibration/contract.json").read_text(encoding="utf-8"))
    q2_selection = json.loads((root / "q2_calibration/selection.json").read_text(encoding="utf-8"))
    calibration_rows = _completed_rows(root, "q2_calibration")
    _validate_q2_calibration_rows(
        calibration_rows,
        candidate,
        expected_job_keys=_manifest_job_keys(root / "q2_calibration/manifests"),
    )
    selected = cast("dict[str, dict[str, object]]", q2_selection["selected"])
    unavailable_rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((root / "q2_calibration/unavailable").glob("*.json"))
    ]
    selected_cells = {_parse_q2_selection_cell(selection_key) for selection_key in selected}
    unavailable_cells = {
        (
            str(row["suite"]),
            str(row["dataset"]),
            str(row["model"]),
            float(row["budget_multiplier"]),
        )
        for row in unavailable_rows
    }
    terminally_unavailable_cells = {
        (
            *str(row["cell_key"]).split(":"),
            float(row["budget_multiplier"]),
        )
        for row in cast("list[dict[str, object]]", q2_selection["terminally_unavailable_cells"])
    }
    audited_cells = selected_cells | unavailable_cells | terminally_unavailable_cells
    if (
        selected_cells & unavailable_cells
        or selected_cells & terminally_unavailable_cells
        or unavailable_cells & terminally_unavailable_cells
        or audited_cells != expected_cells
    ):
        missing = sorted(expected_cells - audited_cells)
        extra = sorted(audited_cells - expected_cells)
        message = (
            "Q2 does not audit the complete 30-task model-budget grid: "
            f"missing={missing[:3]} ({len(missing)}), extra={extra[:3]} ({len(extra)})"
        )
        raise RuntimeError(message)
    expected_maximum_jobs = len(expected_cells) * 3
    if (
        int(q2_contract["maximum_jobs"]) != expected_maximum_jobs
        or int(q2_contract["jobs"]) + 3 * len(unavailable_cells) != expected_maximum_jobs
        or int(q2_contract["not_realizable"]) != len(unavailable_cells)
        or int(q2_selection["source_rows"])
        + int(q2_selection["terminal_failure_rows"])
        != int(q2_contract["jobs"])
        or int(q2_selection["source_rows"]) != len(calibration_rows)
    ):
        message = "Q2 calibration accounting does not cover the complete 30-task grid"
        raise RuntimeError(message)
    expected_q2_rows = len(selected) * len(FINAL_SEEDS)
    q2_rows = _completed_rows(root, "q2_final")
    if int(q2_selection["final_jobs"]) != expected_q2_rows or len(q2_rows) != expected_q2_rows:
        message = f"Q2 final coverage is {len(q2_rows)}/{expected_q2_rows}"
        raise RuntimeError(message)
    tolerance = float(q2_contract["parameter_tolerance"])
    q2_keys: set[str] = set()
    q2_groups: dict[tuple[str, str, str, float], set[int]] = {}
    for row in q2_rows:
        key = str(row["job_key"])
        if key in q2_keys:
            message = f"duplicate Q2 final key: {key}"
            raise RuntimeError(message)
        q2_keys.add(key)
        if float(row["relative_parameter_error"]) > tolerance + 1.0e-12:
            message = f"Q2 parameter mismatch exceeds tolerance: {key}"
            raise RuntimeError(message)
        suite, dataset, model, seed = _validate_test_row(
            row,
            public_models=public_models,
            expected_tasks=expected_tasks,
            stage="Q2",
        )
        group = (
            suite,
            dataset,
            model,
            float(row["budget_multiplier"]),
        )
        q2_groups.setdefault(group, set()).add(seed)
    _validate_q2_seed_groups(q2_groups, selected_cells)
    return len(q2_rows), len(selected), tolerance, q2_selection


def finalize_pipeline(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    """Fail closed unless the frozen Q1 and matched Q2 ledgers are complete."""
    _verify_terminal_stages(root)
    candidate = _verify_architecture_freeze(root)
    q1_rows = _verify_q1_final(root, candidate)
    q2_rows, q2_cells, tolerance, q2_selection = _verify_q2_final(root, candidate)

    payload: dict[str, object] = {
        "schema": "pac_alphabet_q1_q2_pipeline_complete.v2",
        "chosen_internal_model": candidate,
        "q1_final_rows": q1_rows,
        "q2_calibration_successful_rows": int(q2_selection["source_rows"]),
        "q2_calibration_terminal_failures": int(q2_selection["terminal_failure_rows"]),
        "q2_final_rows": q2_rows,
        "q2_realizable_cells": q2_cells,
        "q2_audited_tasks": len(_expected_task_keys()),
        "q2_total_model_budget_cells": len(
            _expected_q2_cells({candidate, *BASELINES})
        ),
        "parameter_tolerance": tolerance,
        "test_evidence_used_for_architecture_choice": False,
        "verified": True,
    }
    write_once(
        root / "pipeline_complete.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    return payload


def status(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    full = campaign_status(root)
    return {
        "schema": "pac_alphabet_q1_q2_final_status.v1",
        **{stage: full[stage] for stage in STAGES},
    }
