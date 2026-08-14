from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .pac_tf_analysis_registry import (
    DEFAULT_REGISTRY_PATH,
    AnalysisFamilyRegistry,
    load_analysis_family_registry,
)
from .pac_tf_confirmatory_stats import (
    bh_fdr,
    bootstrap_summary,
    equivalence_test,
    hierarchical_bootstrap_summary,
    paired_test_summary,
)
from .pac_tf_evidence_queue import (
    DEFAULT_ROOT as DEFAULT_SELECTED_EVIDENCE_ROOT,
)
from .pac_tf_evidence_queue import (
    validate_selected_evidence_root,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

CsvRow = dict[str, str]

DEFAULT_PROTOCOL = Path(".omx/protocols/pac_tf_confirmatory_20260711.json")
DEFAULT_UNSEEN_ROOT = Path(".omx/results/pac-tf-confirmatory-unseen-20260711")
DEFAULT_P1P2_ROOT = Path(".omx/results/pac-tf-p1p2-confirmatory-20260711")
DEFAULT_EVIDENCE_ROOT = DEFAULT_SELECTED_EVIDENCE_ROOT
DEFAULT_OUTPUT_ROOT = Path(".omx/results/pac-tf-confirmatory-supervisor-20260711/reports")

_P1P2_FILES = {
    "low_data": "pac_tf_low_data.csv",
    "synthetic_ood": "pac_tf_synthetic_ood.csv",
    "real_diagnostics": "pac_tf_real_diagnostics.csv",
    "real_domain_ood": "pac_tf_real_domain_ood.csv",
    "efficiency": "pac_tf_final_efficiency.csv",
}

_STATIC_SENSITIVITY_REFERENCE = {
    "stem_kernel": "9",
    "local_kernel": "5",
    "stride": "2",
    "depth": "2",
    "moment_lags": "1,4",
    "pooling_scales": "1,2,4",
    "alpha_min": "0.001",
    "alpha_max": "2.0",
    "gate_range": "0,2",
}


@dataclass(frozen=True, slots=True)
class ConfirmatoryReportConfig:
    protocol_path: Path = DEFAULT_PROTOCOL
    unseen_root: Path = DEFAULT_UNSEEN_ROOT
    p1p2_root: Path = DEFAULT_P1P2_ROOT
    evidence_root: Path = DEFAULT_EVIDENCE_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT
    analysis_registry_path: Path | None = DEFAULT_REGISTRY_PATH
    strict_provenance: bool = True
    bootstrap_seed: int = 20_260_711
    bootstrap_iterations: int = 2_000

    def __post_init__(self) -> None:
        if self.bootstrap_iterations < 100:
            message = "bootstrap-iterations must be at least 100"
            raise ValueError(message)


def write_confirmatory_report(config: ConfirmatoryReportConfig) -> tuple[Path, Path]:
    config.output_root.mkdir(parents=True, exist_ok=True)
    report = build_confirmatory_report(config)
    json_path = config.output_root / "confirmatory_analysis.json"
    markdown_path = config.output_root / "confirmatory_analysis.md"
    _atomic_write_json(json_path, report)
    _atomic_write_text(markdown_path, render_markdown(report))
    return json_path, markdown_path


def build_confirmatory_report(config: ConfirmatoryReportConfig) -> dict[str, object]:
    protocol = _load_protocol(config.protocol_path)
    protocol_sha256 = hashlib.sha256(config.protocol_path.read_bytes()).hexdigest()
    expected_seeds = _protocol_int_tuple(protocol, "seeds")
    locked_ratios = _protocol_float_tuple(protocol, "low_data_ratios")
    registry = (
        load_analysis_family_registry(
            config.analysis_registry_path,
            protocol_path=config.protocol_path,
        )
        if config.analysis_registry_path is not None
        else None
    )
    _validate_selected_contract(config, protocol_sha256)

    unseen_rows = _done_rows(
        _read_csv(config.unseen_root / "results" / "low_data_recommended_real.csv")
    )
    unseen_candidates = [
        row
        for row in unseen_rows
        if row.get("evaluation_collection") == "unseen_final_ucr"
        and row.get("evaluation_split") == "test"
    ]
    unseen_final, unseen_source = _manifest_bound_source(
        config.unseen_root / "queue_manifest.jsonl",
        unseen_candidates,
        result_key="job_key",
        manifest_filter=lambda row: row.get("evaluation_collection") == "unseen_final_ucr",
        protocol_id=str(protocol.get("protocol_id", "")),
        protocol_sha256=protocol_sha256,
        strict_provenance=config.strict_provenance,
    )
    if config.strict_provenance and unseen_candidates:
        _validate_unseen_collection_lock(config, protocol_sha256)

    p1_manifest = config.p1p2_root / "queue_manifest.jsonl"
    p1p2_rows: dict[str, list[CsvRow]] = {}
    p1_sources: dict[str, dict[str, object]] = {}
    for package, filename in _P1P2_FILES.items():
        accepted, source = _manifest_bound_source(
            p1_manifest,
            _done_rows(_read_csv(config.p1p2_root / "results" / filename)),
            result_key="job_key",
            manifest_filter=lambda row, package=package: row.get("package") == package,
            protocol_id=str(protocol.get("protocol_id", "")),
            protocol_sha256=protocol_sha256,
            strict_provenance=config.strict_provenance,
        )
        p1p2_rows[package] = accepted
        p1_sources[f"p1p2_{package}"] = source

    evidence_rows: dict[str, list[CsvRow]] = {}
    evidence_sources: dict[str, dict[str, object]] = {}
    for kind in (
        "core_ablation",
        "mechanism_checkpoint",
        "interpretability",
        "sensitivity",
    ):
        accepted, source = _manifest_bound_source(
            config.evidence_root / f"{kind}_manifest.jsonl",
            _done_rows(_read_csv(config.evidence_root / "results" / f"{kind}.csv")),
            result_key="queue_key",
            protocol_id=str(protocol.get("protocol_id", "")),
            protocol_sha256=protocol_sha256,
            strict_provenance=config.strict_provenance,
        )
        evidence_rows[kind] = accepted
        evidence_sources[f"evidence_{kind}"] = source

    sources = {"unseen_final": unseen_source, **p1_sources, **evidence_sources}
    final_evaluation = _final_evaluation(
        unseen_final,
        protocol,
        expected_seeds,
        config,
        registry,
    )
    low_data = _low_data(
        [*p1p2_rows["low_data"], *p1p2_rows["real_diagnostics"]],
        locked_ratios,
        expected_seeds,
        config,
        registry,
        locked_models=tuple(str(value) for value in _object_list(protocol, "baseline_families")),
        locked_datasets=tuple(
            str(value) for value in _object_list(protocol, "untouched_final_datasets")
        ),
    )
    ood = {
        "synthetic": _synthetic_ood(
            p1p2_rows["synthetic_ood"],
            config,
            expected_families=tuple(
                str(value) for value in _object_list(protocol, "synthetic_ood_families")
            ),
        ),
        "real_domain": _real_domain_ood(p1p2_rows["real_domain_ood"], config),
        "real_corruption_diagnostics": _real_corruption_ood(p1p2_rows["real_diagnostics"], config),
    }
    calibration = _calibration(
        p1p2_rows["real_diagnostics"],
        p1p2_rows["real_domain_ood"],
        config,
    )
    accuracy_by_model = _final_accuracy_by_model(final_evaluation)
    efficiency = _efficiency(
        p1p2_rows["efficiency"],
        accuracy_by_model,
        config,
        expected_models=tuple(str(value) for value in _object_list(protocol, "baseline_families")),
    )
    core_ablation = _core_ablation(evidence_rows["core_ablation"], config, registry)
    sensitivity = _sensitivity(
        evidence_rows["sensitivity"],
        config,
        registry,
        level_order=_sensitivity_level_order(
            config.evidence_root / "sensitivity_manifest.jsonl"
        ),
    )
    mechanism = _mechanism_recovery(evidence_rows["mechanism_checkpoint"], config, registry)
    interventions = _interventions(evidence_rows["interpretability"], config, registry)

    _apply_execution_status(final_evaluation, sources["unseen_final"])
    _apply_execution_status(
        low_data,
        sources["p1p2_low_data"],
        sources["p1p2_real_diagnostics"],
    )
    synthetic_ood = ood["synthetic"]
    real_domain_ood = ood["real_domain"]
    corruption_ood = ood["real_corruption_diagnostics"]
    _apply_execution_status(synthetic_ood, sources["p1p2_synthetic_ood"])
    _apply_execution_status(real_domain_ood, sources["p1p2_real_domain_ood"])
    _apply_execution_status(corruption_ood, sources["p1p2_real_diagnostics"])
    _apply_execution_status(
        calibration,
        sources["p1p2_real_diagnostics"],
        sources["p1p2_real_domain_ood"],
    )
    _apply_execution_status(efficiency, sources["p1p2_efficiency"])
    _apply_execution_status(core_ablation, sources["evidence_core_ablation"])
    _apply_execution_status(sensitivity, sources["evidence_sensitivity"])
    _apply_execution_status(mechanism, sources["evidence_mechanism_checkpoint"])
    _apply_execution_status(interventions, sources["evidence_interpretability"])

    sections = (
        final_evaluation,
        low_data,
        synthetic_ood,
        real_domain_ood,
        corruption_ood,
        calibration,
        efficiency,
        core_ablation,
        sensitivity,
        mechanism,
        interventions,
    )
    complete = bool(sources) and all(
        str(source.get("status")) == "complete" for source in sources.values()
    ) and all(
        str(section.get("analysis_status")) == "complete" for section in sections
    )
    return {
        "schema_version": "pac_tf_confirmatory_analysis.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "report_status": "complete" if complete else "partial",
        "protocol_id": protocol.get("protocol_id"),
        "protocol_path": str(config.protocol_path),
        "protocol_sha256": protocol_sha256,
        "analysis_family_registry": _registry_metadata(config, registry),
        "bootstrap": {
            "seed": config.bootstrap_seed,
            "iterations": config.bootstrap_iterations,
            "interval": "percentile_95",
            "hierarchy": "datasets_or_tasks_then_paired_seed_runs",
        },
        "evidence_policy": {
            "completed_rows_only": True,
            "missing_cells_are_never_imputed": True,
            "partial_sections_are_labeled": True,
            "primary_final_metric": "balanced_accuracy",
            "paired_difference_direction": "PAC-TF minus baseline",
        },
        "sources": sources,
        "equivalence_policy": {
            "protocol_statement": (
                protocol.get("statistics", {})
                if isinstance(protocol.get("statistics"), dict)
                else {}
            ),
            "locked_margin": _equivalence_margin(protocol),
            "default": equivalence_test([], _equivalence_margin(protocol)),
            "claim": "No equivalence claim is made without a separately locked margin.",
        },
        "final_evaluation": final_evaluation,
        "low_data": low_data,
        "ood": ood,
        "calibration_and_error_analysis": calibration,
        "efficiency": efficiency,
        "core_ablation": core_ablation,
        "architecture_sensitivity": sensitivity,
        "mechanism_recovery": mechanism,
        "interpretability_interventions": interventions,
    }


def _final_evaluation(
    rows: Sequence[CsvRow],
    protocol: Mapping[str, object],
    expected_seeds: Sequence[int],
    config: ConfirmatoryReportConfig,
    registry: AnalysisFamilyRegistry | None,
) -> dict[str, object]:
    metrics = ("balanced_accuracy", "test_accuracy", "macro_f1")
    cells: list[dict[str, object]] = []
    by_cell: dict[tuple[str, str], list[CsvRow]] = defaultdict(list)
    for row in rows:
        by_cell[(row.get("dataset_or_task", ""), _model(row))].append(row)
    for (dataset, model), cell_rows in sorted(by_cell.items()):
        metric_summaries = {
            metric: bootstrap_summary(
                _float_values(cell_rows, metric),
                iterations=config.bootstrap_iterations,
                seed=config.bootstrap_seed,
                label=f"final:{dataset}:{model}:{metric}",
            )
            for metric in metrics
        }
        observed_seeds = sorted(_int_values(cell_rows, "seed"))
        cells.append(
            {
                "dataset": dataset,
                "model": model,
                "metrics": metric_summaries,
                "observed_seeds": observed_seeds,
                "expected_seeds": list(expected_seeds),
                "cell_status": (
                    "complete" if set(observed_seeds) == set(expected_seeds) else "partial"
                ),
                "ci_scope": "seed_bootstrap_within_dataset",
            }
        )

    aggregate: list[dict[str, object]] = []
    models = sorted({_model(row) for row in rows})
    for model in models:
        model_rows = [row for row in rows if _model(row) == model]
        summaries: dict[str, object] = {}
        for metric in metrics:
            grouped = _group_values(model_rows, "dataset_or_task", metric)
            summaries[metric] = hierarchical_bootstrap_summary(
                grouped,
                iterations=config.bootstrap_iterations,
                seed=config.bootstrap_seed,
                label=f"final-aggregate:{model}:{metric}",
            )
        aggregate.append({"model": model, "metrics": summaries})

    families = [str(value) for value in _object_list(protocol, "baseline_families")]
    comparisons = []
    for baseline in families:
        if baseline == "pac_tf":
            continue
        differences = _paired_differences(
            rows,
            reference="pac_tf",
            comparator=baseline,
            metric="balanced_accuracy",
            group_field="dataset_or_task",
        )
        group_means = [float(np.mean(values)) for values in differences.values() if values]
        summary = paired_test_summary(
            differences,
            iterations=config.bootstrap_iterations,
            seed=config.bootstrap_seed,
            label=f"final-paired:pac_tf:{baseline}",
        )
        summary.update(
            {
                "reference": "pac_tf",
                "baseline": baseline,
                "hypothesis_id": f"pac_tf_vs_{baseline}",
                "difference": "pac_tf_minus_baseline_balanced_accuracy",
                "equivalence": equivalence_test(group_means, _equivalence_margin(protocol)),
            }
        )
        comparisons.append(summary)
    comparisons = _registered_bh(
        comparisons,
        registry,
        "final_pac_tf_vs_locked_baselines",
    )
    analysis_status = _family_analysis_status(
        rows,
        comparisons,
        required_fields=("balanced_accuracy",),
    )
    return {
        "status": "available" if rows else "missing",
        "analysis_status": analysis_status,
        "per_dataset_model": cells,
        "model_aggregate": aggregate,
        "paired_pac_tf_vs_baselines": comparisons,
        "paired_test_unit": "dataset-level mean across exactly matched seeds",
        "fdr_family": "all locked PAC-TF-versus-baseline final comparisons",
        "unseen_datasets_expected": protocol.get("untouched_final_datasets", []),
    }


def _low_data(  # noqa: C901, PLR0912, PLR0915 - explicit complete-curve contract
    rows: Sequence[CsvRow],
    locked_ratios: Sequence[float],
    expected_seeds: Sequence[int],
    config: ConfirmatoryReportConfig,
    registry: AnalysisFamilyRegistry | None,
    *,
    locked_models: Sequence[str],
    locked_datasets: Sequence[str],
) -> dict[str, object]:
    curves: list[dict[str, object]] = []
    by_point: dict[tuple[str, str, float], list[CsvRow]] = defaultdict(list)
    for row in rows:
        ratio = _float(row.get("requested_ratio") or row.get("data_ratio"))
        if ratio is not None:
            by_point[(_model(row), row.get("dataset_or_task", ""), ratio)].append(row)
    for (model, dataset, ratio), point_rows in sorted(by_point.items()):
        curves.append(
            {
                "model": model,
                "dataset": dataset,
                "requested_ratio": ratio,
                "balanced_accuracy": bootstrap_summary(
                    _float_values(point_rows, "balanced_accuracy"),
                    iterations=config.bootstrap_iterations,
                    seed=config.bootstrap_seed,
                    label=f"low-data:{model}:{dataset}:{ratio}",
                ),
                "realized_ratio_mean": _mean_or_none(_float_values(point_rows, "realized_ratio")),
                "realized_count_mean": _mean_or_none(_float_values(point_rows, "realized_count")),
                "min_class_count_min": _min_or_none(_float_values(point_rows, "min_class_count")),
                "observed_seeds": sorted(_int_values(point_rows, "seed")),
            }
        )

    auc_rows: list[dict[str, object]] = []
    grouped_runs: dict[tuple[str, str, int], list[CsvRow]] = defaultdict(list)
    for row in rows:
        seed = _int(row.get("seed"))
        if seed is not None:
            grouped_runs[(_model(row), row.get("dataset_or_task", ""), seed)].append(row)
    by_model_dataset: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    auc_by_run: dict[tuple[str, str, int], float] = {}
    grid_by_run: dict[tuple[str, str, int], tuple[float, ...]] = {}
    invalid_curve_runs: list[str] = []
    for (model, dataset, seed), run_rows in grouped_runs.items():
        by_ratio: dict[float, list[CsvRow]] = defaultdict(list)
        for row in run_rows:
            ratio = _float(row.get("requested_ratio") or row.get("data_ratio"))
            if ratio is not None:
                by_ratio[ratio].append(row)
        run_label = f"{model}/{dataset}/seed{seed}"
        if set(by_ratio) != set(locked_ratios) or any(
            len(by_ratio[ratio]) != 1 for ratio in locked_ratios
        ):
            invalid_curve_runs.append(run_label)
            continue
        valid: list[tuple[float, float]] = []
        for ratio in locked_ratios:
            row = by_ratio[ratio][0]
            realized = _float(row.get("realized_ratio"))
            accuracy = _float(row.get("balanced_accuracy"))
            if realized is None or accuracy is None or realized <= 0.0:
                valid = []
                break
            valid.append((realized, accuracy))
        if len(valid) != len(locked_ratios):
            invalid_curve_runs.append(run_label)
            continue
        if any(right[0] < left[0] for left, right in pairwise(valid)):
            invalid_curve_runs.append(run_label)
            continue
        x_values = np.log(np.asarray([point[0] for point in valid], dtype=np.float64))
        y_values = np.asarray([point[1] for point in valid], dtype=np.float64)
        span = float(x_values[-1] - x_values[0])
        if span <= 0.0:
            continue
        auc = float(np.trapezoid(y_values, x_values) / span)
        by_model_dataset[(model, dataset)].append((seed, auc))
        auc_by_run[(model, dataset, seed)] = auc
        grid_by_run[(model, dataset, seed)] = tuple(point[0] for point in valid)
    for (model, dataset), values in sorted(by_model_dataset.items()):
        auc_rows.append(
            {
                "model": model,
                "dataset": dataset,
                "normalized_log_ratio_auc": bootstrap_summary(
                    [value for _, value in values],
                    iterations=config.bootstrap_iterations,
                    seed=config.bootstrap_seed,
                    label=f"low-data-auc:{model}:{dataset}",
                ),
                "eligible_complete_curve_seeds": sorted(seed for seed, _ in values),
                "realized_ratio_grid_by_seed": {
                    str(seed): list(grid_by_run[(model, dataset, seed)]) for seed, _ in values
                },
                "log_ratio_span_by_seed": {
                    str(seed): math.log(grid_by_run[(model, dataset, seed)][-1])
                    - math.log(grid_by_run[(model, dataset, seed)][0])
                    for seed, _ in values
                },
                "expected_seeds": list(expected_seeds),
                "curve_status": (
                    "complete" if {seed for seed, _ in values} == set(expected_seeds) else "partial"
                ),
            }
        )
    models = list(locked_models)
    ratio_comparisons = []
    for ratio in locked_ratios:
        ratio_rows = [
            row
            for row in rows
            if _float(row.get("requested_ratio") or row.get("data_ratio")) == ratio
        ]
        for baseline in models:
            if baseline == "pac_tf":
                continue
            differences = _paired_differences(
                ratio_rows,
                reference="pac_tf",
                comparator=baseline,
                metric="balanced_accuracy",
                group_field="dataset_or_task",
            )
            summary = paired_test_summary(
                differences,
                iterations=config.bootstrap_iterations,
                seed=config.bootstrap_seed,
                label=f"low-data-paired:{ratio}:{baseline}",
            )
            summary.update(
                {
                    "requested_ratio": ratio,
                    "reference": "pac_tf",
                    "baseline": baseline,
                    "hypothesis_id": (
                        f"ratio_{_ratio_hypothesis_label(ratio)}__pac_tf_vs_{baseline}"
                    ),
                    "difference": "pac_tf_minus_baseline_balanced_accuracy",
                }
            )
            ratio_comparisons.append(summary)
    auc_comparisons = []
    mismatched_auc_grids: list[str] = []
    for baseline in models:
        if baseline == "pac_tf":
            continue
        differences: dict[str, list[float]] = defaultdict(list)
        for dataset in locked_datasets:
            for seed in expected_seeds:
                pac = auc_by_run.get(("pac_tf", dataset, seed))
                other = auc_by_run.get((baseline, dataset, seed))
                if pac is not None and other is not None:
                    pac_grid = grid_by_run[("pac_tf", dataset, seed)]
                    other_grid = grid_by_run[(baseline, dataset, seed)]
                    if not _same_float_grid(pac_grid, other_grid):
                        mismatched_auc_grids.append(f"{baseline}/{dataset}/seed{seed}")
                        continue
                    differences[dataset].append(pac - other)
        summary = paired_test_summary(
            differences,
            iterations=config.bootstrap_iterations,
            seed=config.bootstrap_seed,
            label=f"low-data-auc-paired:{baseline}",
        )
        summary.update(
            {
                "reference": "pac_tf",
                "baseline": baseline,
                "hypothesis_id": f"auc__pac_tf_vs_{baseline}",
                "difference": "pac_tf_minus_baseline_normalized_log_ratio_auc",
            }
        )
        auc_comparisons.append(summary)

    ratio_comparisons = _registered_bh(
        ratio_comparisons,
        registry,
        "low_data_ratio_by_locked_baseline",
    )
    auc_comparisons = _registered_bh(
        auc_comparisons,
        registry,
        "low_data_log_ratio_auc_by_locked_baseline",
    )
    expected_cells = len(locked_models) * len(locked_datasets)
    complete_cells = sum(row["curve_status"] == "complete" for row in auc_rows)
    analysis_complete = (
        bool(rows)
        and not invalid_curve_runs
        and not mismatched_auc_grids
        and len(auc_rows) == expected_cells
        and complete_cells == expected_cells
        and all(
            row.get("fdr_family_status") != "incomplete_not_adjusted"
            for row in (*ratio_comparisons, *auc_comparisons)
        )
    )
    return {
        "status": "available" if rows else "missing",
        "analysis_status": (
            "complete" if analysis_complete else "partial" if rows else "missing"
        ),
        "locked_requested_ratios": list(locked_ratios),
        "curve_points": curves,
        "log_ratio_auc": auc_rows,
        "paired_pac_tf_vs_baselines_by_ratio": ratio_comparisons,
        "paired_pac_tf_vs_baselines_log_ratio_auc": auc_comparisons,
        "auc_definition": (
            "trapezoidal balanced-accuracy area over log(realized ratio), normalized by "
            "the observed log-ratio span; only complete locked ratio grids are eligible"
        ),
        "complete_curve_cells": complete_cells,
        "partial_curve_cells": len(auc_rows) - complete_cells,
        "expected_model_dataset_cells": expected_cells,
        "invalid_or_duplicate_curve_runs": sorted(invalid_curve_runs),
        "mismatched_paired_auc_grids": sorted(set(mismatched_auc_grids)),
    }


def _synthetic_ood(
    rows: Sequence[CsvRow],
    config: ConfirmatoryReportConfig,
    *,
    expected_families: Sequence[str],
) -> dict[str, object]:
    expanded: dict[tuple[str, str, str], list[dict[str, float]]] = defaultdict(list)
    malformed = 0
    incomplete_family_rows: list[str] = []
    for row in rows:
        sweep = _json_list(row.get("ood_sweep_json", ""))
        if sweep is None:
            malformed += 1
            continue
        observed_families = {str(item.get("family", "")) for item in sweep}
        if observed_families != set(expected_families):
            incomplete_family_rows.append(row.get("job_key", ""))
        for item in sweep:
            family = str(item.get("family", ""))
            level = str(item.get("level", ""))
            id_loss = _object_float(item.get("id_test_loss"))
            ood_loss = _object_float(item.get("ood_test_loss"))
            id_nrmse = _object_float(item.get("id_nrmse"))
            ood_nrmse = _object_float(item.get("ood_nrmse"))
            if (
                not family
                or not level
                or id_loss is None
                or ood_loss is None
                or id_nrmse is None
                or ood_nrmse is None
            ):
                malformed += 1
                continue
            expanded[(_model(row), family, level)].append(
                {
                    "id_loss": id_loss,
                    "ood_loss": ood_loss,
                    "id_nrmse": id_nrmse,
                    "ood_nrmse": ood_nrmse,
                    "relative_loss_increase": (ood_loss - id_loss) / max(abs(id_loss), 1.0e-12),
                    "absolute_nrmse_increase": ood_nrmse - id_nrmse,
                    "relative_nrmse_increase": (
                        (ood_nrmse - id_nrmse) / id_nrmse
                        if id_nrmse > 0.0
                        else math.nan
                    ),
                }
            )
    summaries = []
    for (model, family, level), values in sorted(expanded.items()):
        summaries.append(
            {
                "model": model,
                "family": family,
                "level": level,
                "id_loss": _bootstrap_field(
                    values, "id_loss", config, f"syn:{model}:{family}:{level}:id"
                ),
                "ood_loss": _bootstrap_field(
                    values, "ood_loss", config, f"syn:{model}:{family}:{level}:ood"
                ),
                "id_nrmse": _bootstrap_field(
                    values, "id_nrmse", config, f"syn:{model}:{family}:{level}:id-nrmse"
                ),
                "ood_nrmse": _bootstrap_field(
                    values, "ood_nrmse", config, f"syn:{model}:{family}:{level}:nrmse"
                ),
                "relative_loss_increase": _bootstrap_field(
                    values,
                    "relative_loss_increase",
                    config,
                    f"syn:{model}:{family}:{level}:drop",
                ),
                "absolute_nrmse_increase": _bootstrap_field(
                    values,
                    "absolute_nrmse_increase",
                    config,
                    f"syn:{model}:{family}:{level}:absolute-nrmse",
                ),
                "relative_nrmse_increase": _bootstrap_field(
                    values,
                    "relative_nrmse_increase",
                    config,
                    f"syn:{model}:{family}:{level}:relative-nrmse",
                ),
            }
        )
    analysis_complete = (
        bool(rows) and bool(summaries) and malformed == 0 and not incomplete_family_rows
    )
    return {
        "status": "available" if rows else "missing",
        "analysis_status": (
            "complete" if analysis_complete else "partial" if rows else "missing"
        ),
        "metric_direction": "lower loss/NRMSE is better; positive increase is degradation",
        "conditions": summaries,
        "malformed_completed_rows": malformed,
        "rows_missing_locked_families": sorted(incomplete_family_rows),
    }


def _real_domain_ood(rows: Sequence[CsvRow], config: ConfirmatoryReportConfig) -> dict[str, object]:
    summaries = []
    by_model: dict[str, list[CsvRow]] = defaultdict(list)
    for row in rows:
        by_model[_model(row)].append(row)
    for model, model_rows in sorted(by_model.items()):
        summaries.append(
            {
                "model": model,
                "dataset": model_rows[0].get("dataset_or_task", ""),
                "id_common_balanced_accuracy": _bootstrap_rows(
                    model_rows,
                    "id_common_balanced_accuracy",
                    config,
                    f"real-ood:{model}:id-common-ba",
                ),
                "ood_common_balanced_accuracy": _bootstrap_rows(
                    model_rows,
                    "ood_common_balanced_accuracy",
                    config,
                    f"real-ood:{model}:ood-common-ba",
                ),
                "relative_common_balanced_accuracy_drop": _bootstrap_rows(
                    model_rows,
                    "relative_common_balanced_accuracy_drop",
                    config,
                    f"real-ood:{model}:drop-common-ba",
                ),
                "id_common_macro_f1": _bootstrap_rows(
                    model_rows,
                    "id_common_macro_f1",
                    config,
                    f"real-ood:{model}:id-common-f1",
                ),
                "ood_common_macro_f1": _bootstrap_rows(
                    model_rows,
                    "ood_common_macro_f1",
                    config,
                    f"real-ood:{model}:ood-common-f1",
                ),
                "id_common_accuracy": _bootstrap_rows(
                    model_rows,
                    "id_common_accuracy",
                    config,
                    f"real-ood:{model}:id-common-acc",
                ),
                "ood_common_accuracy": _bootstrap_rows(
                    model_rows,
                    "ood_common_accuracy",
                    config,
                    f"real-ood:{model}:ood-common-acc",
                ),
                "relative_common_accuracy_drop": _bootstrap_rows(
                    model_rows,
                    "relative_common_accuracy_drop",
                    config,
                    f"real-ood:{model}:drop-common-acc",
                ),
                "ood_full_5class_balanced_accuracy": _bootstrap_rows(
                    model_rows,
                    "ood_full_5class_balanced_accuracy",
                    config,
                    f"real-ood:{model}:ood-full-ba",
                ),
                "ood_full_5class_macro_f1": _bootstrap_rows(
                    model_rows,
                    "ood_full_5class_macro_f1",
                    config,
                    f"real-ood:{model}:ood-full-f1",
                ),
                "ood_full_5class_accuracy": _bootstrap_rows(
                    model_rows,
                    "ood_full_5class_accuracy",
                    config,
                    f"real-ood:{model}:ood-full-acc",
                ),
            }
        )
    required_fields = (
        "id_common_balanced_accuracy",
        "ood_common_balanced_accuracy",
        "relative_common_balanced_accuracy_drop",
        "id_common_macro_f1",
        "ood_common_macro_f1",
    )
    return {
        "status": "available" if rows else "missing",
        "analysis_status": _required_rows_status(rows, required_fields),
        "domain_contract": "MIT-BIH held-out DS1 ID versus patient-disjoint DS2 OOD",
        "comparison_population": "classes with non-zero support in both ID and OOD domains",
        "full_5class_ood_is_reported_without_an_incomparable_id_drop": True,
        "models": summaries,
    }


def _real_corruption_ood(
    rows: Sequence[CsvRow], config: ConfirmatoryReportConfig
) -> dict[str, object]:
    expanded: dict[tuple[str, str, str], list[dict[str, float]]] = defaultdict(list)
    malformed = 0
    for row in rows:
        sweep = _json_list(row.get("real_corruption_ood_json", ""))
        if sweep is None:
            malformed += 1
            continue
        for item in sweep:
            shift = str(item.get("shift", ""))
            accuracy = _object_float(item.get("accuracy"))
            drop = _object_float(item.get("absolute_accuracy_drop"))
            nll = _object_float(item.get("nll"))
            brier = _object_float(item.get("brier_score"))
            ece = _object_float(item.get("ece_15"))
            if (
                not shift
                or accuracy is None
                or drop is None
                or nll is None
                or brier is None
                or ece is None
            ):
                malformed += 1
                continue
            expanded[(_model(row), row.get("dataset_or_task", ""), shift)].append(
                {
                    "accuracy": accuracy,
                    "absolute_accuracy_drop": drop,
                    "nll": nll,
                    "brier_score": brier,
                    "ece_15": ece,
                }
            )
    summaries = []
    for (model, dataset, shift), values in sorted(expanded.items()):
        summaries.append(
            {
                "model": model,
                "dataset": dataset,
                "shift": shift,
                "accuracy": _bootstrap_field(
                    values, "accuracy", config, f"corruption:{model}:{dataset}:{shift}:acc"
                ),
                "absolute_accuracy_drop": _bootstrap_field(
                    values,
                    "absolute_accuracy_drop",
                    config,
                    f"corruption:{model}:{dataset}:{shift}:drop",
                ),
                "nll": _bootstrap_field(
                    values, "nll", config, f"corruption:{model}:{dataset}:{shift}:nll"
                ),
                "brier_score": _bootstrap_field(
                    values,
                    "brier_score",
                    config,
                    f"corruption:{model}:{dataset}:{shift}:brier",
                ),
                "ece_15": _bootstrap_field(
                    values, "ece_15", config, f"corruption:{model}:{dataset}:{shift}:ece"
                ),
            }
        )
    analysis_complete = bool(rows) and bool(summaries) and malformed == 0
    return {
        "status": "available" if rows else "missing",
        "analysis_status": (
            "complete" if analysis_complete else "partial" if rows else "missing"
        ),
        "scope": "prespecified corruption shifts; not a real domain-generalization claim",
        "conditions": summaries,
        "malformed_completed_rows": malformed,
    }


def _calibration(
    diagnostic_rows: Sequence[CsvRow],
    real_ood_rows: Sequence[CsvRow],
    config: ConfirmatoryReportConfig,
) -> dict[str, object]:
    artifact_root = config.output_root / "artifacts" / "calibration"
    records: list[dict[str, object]] = []
    aggregates: dict[tuple[str, str, str], list[dict[str, float]]] = defaultdict(list)
    malformed = 0
    for row in diagnostic_rows:
        metrics = _diagnostic_metrics(row)
        artifact = _write_diagnostic_artifact(
            artifact_root,
            scope=row.get("dataset_or_task", ""),
            model=_model(row),
            seed=row.get("seed", ""),
            domain="test",
            diagnostics=row,
        )
        if artifact is None:
            malformed += 1
        else:
            records.append(
                {
                    "dataset": row.get("dataset_or_task", ""),
                    "model": _model(row),
                    "seed": _int(row.get("seed")),
                    "domain": "test",
                    "artifact": str(artifact.relative_to(config.output_root)),
                }
            )
        aggregates[(row.get("dataset_or_task", ""), _model(row), "test")].append(metrics)
    for row in real_ood_rows:
        for domain in ("id", "ood"):
            parsed = _json_dict(row.get(f"{domain}_diagnostics_json", ""))
            if parsed is None:
                malformed += 1
                continue
            metrics = {
                key: value
                for key in ("nll", "ece_15", "brier_score")
                if (value := _object_float(parsed.get(key))) is not None
            }
            artifact = _write_diagnostic_artifact(
                artifact_root,
                scope=row.get("dataset_or_task", "mit-bih-ds1-to-ds2"),
                model=_model(row),
                seed=row.get("seed", ""),
                domain=domain,
                diagnostics={
                    key: json.dumps(value) if isinstance(value, list) else str(value)
                    for key, value in parsed.items()
                },
            )
            if artifact is not None:
                records.append(
                    {
                        "dataset": row.get("dataset_or_task", ""),
                        "model": _model(row),
                        "seed": _int(row.get("seed")),
                        "domain": domain,
                        "artifact": str(artifact.relative_to(config.output_root)),
                    }
                )
            aggregates[(row.get("dataset_or_task", ""), _model(row), domain)].append(metrics)
    summaries = []
    for (dataset, model, domain), values in sorted(aggregates.items()):
        summaries.append(
            {
                "dataset": dataset,
                "model": model,
                "domain": domain,
                "nll": _bootstrap_field(
                    values, "nll", config, f"cal:{dataset}:{model}:{domain}:nll"
                ),
                "ece_15": _bootstrap_field(
                    values, "ece_15", config, f"cal:{dataset}:{model}:{domain}:ece"
                ),
                "brier_score": _bootstrap_field(
                    values,
                    "brier_score",
                    config,
                    f"cal:{dataset}:{model}:{domain}:brier",
                ),
            }
        )
    return {
        "status": "available" if diagnostic_rows or real_ood_rows else "missing",
        "analysis_status": (
            "complete"
            if (diagnostic_rows or real_ood_rows) and malformed == 0 and bool(summaries)
            else "partial"
            if diagnostic_rows or real_ood_rows
            else "missing"
        ),
        "summary": summaries,
        "per_run_confusion_and_per_class_artifacts": records,
        "malformed_completed_rows": malformed,
    }


def _efficiency(
    rows: Sequence[CsvRow],
    accuracy_by_model: Mapping[str, float],
    config: ConfirmatoryReportConfig,
    *,
    expected_models: Sequence[str],
) -> dict[str, object]:
    grouped: dict[tuple[str, int, int, str], list[CsvRow]] = defaultdict(list)
    for row in rows:
        length = _int(row.get("sequence_length"))
        batch = _int(row.get("batch_size"))
        if length is not None and batch is not None:
            grouped[(_model(row), length, batch, row.get("runtime", ""))].append(row)
    cells: list[dict[str, object]] = []
    for (model, length, batch, runtime), cell_rows in sorted(grouped.items()):
        outcome_counts = Counter(row.get("outcome_status", "unknown") for row in cell_rows)
        reasons = sorted(
            {
                row.get("resource_limit_reason", "")
                for row in cell_rows
                if row.get("resource_limit_reason")
            }
        )
        measured = [row for row in cell_rows if row.get("outcome_status") == "measured"]
        cells.append(
            {
                "model": model,
                "sequence_length": length,
                "batch_size": batch,
                "runtime": runtime,
                "outcome_counts": dict(sorted(outcome_counts.items())),
                "resource_or_compile_reasons": reasons,
                "cell_status": (
                    "measured"
                    if measured and len(measured) == len(cell_rows)
                    else "mixed_censored"
                    if measured
                    else "censored"
                ),
                "accuracy_joined_from_final_balanced_accuracy": accuracy_by_model.get(model),
                "latency_ms": _bootstrap_rows(
                    measured,
                    "latency_ms",
                    config,
                    f"eff:{model}:{length}:{batch}:{runtime}:latency",
                ),
                "tokens_per_second": _bootstrap_rows(
                    measured,
                    "tokens_per_second",
                    config,
                    f"eff:{model}:{length}:{batch}:{runtime}:throughput",
                ),
                "peak_memory_mb": _bootstrap_rows(
                    measured,
                    "peak_memory_mb",
                    config,
                    f"eff:{model}:{length}:{batch}:{runtime}:memory",
                ),
                "params_trainable": _mean_or_none(_float_values(cell_rows, "params_trainable")),
                "pareto_nondominated": None,
                "pareto_dimensions": [],
                "pareto_status": "not_evaluated",
            }
        )
    _mark_pareto(cells, expected_models=expected_models)
    analysis_complete = bool(rows) and all(
        cell.get("pareto_status") in {"evaluated", "censored"} for cell in cells
    )
    return {
        "status": "available" if rows else "missing",
        "analysis_status": (
            "complete" if analysis_complete else "partial" if rows else "missing"
        ),
        "cells": cells,
        "pareto_scope": (
            "models are compared only within the same sequence_length, batch_size, and runtime; "
            "accuracy is joined from the held-out final aggregate"
        ),
        "censored_cells_are_retained": True,
    }


def _core_ablation(
    rows: Sequence[CsvRow],
    config: ConfirmatoryReportConfig,
    registry: AnalysisFamilyRegistry | None,
) -> dict[str, object]:
    reference = {
        (row.get("dataset_or_task", ""), row.get("seed", "")): _float(
            row.get("validation_balanced_accuracy")
        )
        for row in rows
        if row.get("intervention") == "reference"
    }
    comparisons = []
    interventions = sorted(
        {row.get("intervention", "") for row in rows if row.get("intervention") != "reference"}
    )
    for intervention in interventions:
        grouped: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            if row.get("intervention") != intervention:
                continue
            key = (row.get("dataset_or_task", ""), row.get("seed", ""))
            value = _float(row.get("validation_balanced_accuracy"))
            base = reference.get(key)
            if value is not None and base is not None:
                grouped[key[0]].append(base - value)
        summary = paired_test_summary(
            grouped,
            iterations=config.bootstrap_iterations,
            seed=config.bootstrap_seed,
            label=f"ablation:{intervention}",
        )
        summary.update(
            {
                "intervention": intervention,
                "hypothesis_id": f"reference_vs_{intervention}",
                "difference": "reference_minus_ablation_balanced_accuracy",
            }
        )
        comparisons.append(summary)
    comparisons = _registered_bh(comparisons, registry, "core_component_ablation")
    return {
        "status": "available" if rows else "missing",
        "analysis_status": _family_analysis_status(
            rows,
            comparisons,
            required_fields=("validation_balanced_accuracy",),
        ),
        "comparisons": comparisons,
        "positive_difference_means_reference_is_better": True,
    }


def _sensitivity(
    rows: Sequence[CsvRow],
    config: ConfirmatoryReportConfig,
    registry: AnalysisFamilyRegistry | None,
    *,
    level_order: Mapping[str, Sequence[str]],
) -> dict[str, object]:
    comparisons = []
    references = _sensitivity_references(rows)
    for factor in sorted({row.get("intervention", "") for row in rows}):
        reference_level = references.get(factor)
        if reference_level is None:
            continue
        factor_rows = [row for row in rows if row.get("intervention") == factor]
        reference = {
            (row.get("dataset_or_task", ""), row.get("seed", "")): _float(
                row.get("validation_balanced_accuracy")
            )
            for row in factor_rows
            if row.get("level") == reference_level
        }
        observed_candidates = sorted(
            {row.get("level", "") for row in factor_rows} - {reference_level}
        )
        candidates = tuple(level_order.get(factor, ()))
        if not candidates and not config.strict_provenance:
            candidates = tuple(observed_candidates)
        candidate_index_by_level = {
            level: index for index, level in enumerate(candidates, start=1)
        }
        for level in observed_candidates:
            candidate_index = candidate_index_by_level.get(level)
            if candidate_index is None:
                message = f"sensitivity level is not bound to the locked manifest: {factor}={level}"
                raise ValueError(message)
            grouped: dict[str, list[float]] = defaultdict(list)
            for row in factor_rows:
                if row.get("level") != level:
                    continue
                key = (row.get("dataset_or_task", ""), row.get("seed", ""))
                value = _float(row.get("validation_balanced_accuracy"))
                base = reference.get(key)
                if value is not None and base is not None:
                    grouped[key[0]].append(value - base)
            summary = paired_test_summary(
                grouped,
                iterations=config.bootstrap_iterations,
                seed=config.bootstrap_seed,
                label=f"sensitivity:{factor}:{level}",
            )
            summary.update(
                {
                    "factor": factor,
                    "level": level,
                    "reference_level": reference_level,
                    "hypothesis_id": (
                        f"{factor}__candidate_{candidate_index}_vs_reference"
                    ),
                    "difference": "level_minus_locked_reference_balanced_accuracy",
                }
            )
            comparisons.append(summary)
    comparisons = _registered_bh(
        comparisons,
        registry,
        "architecture_one_factor_sensitivity",
    )
    return {
        "status": "available" if rows else "missing",
        "analysis_status": _family_analysis_status(
            rows,
            comparisons,
            required_fields=("validation_balanced_accuracy",),
        ),
        "comparisons": comparisons,
        "reference_levels": references,
    }


def _sensitivity_references(rows: Sequence[CsvRow]) -> dict[str, str]:
    references = dict(_STATIC_SENSITIVITY_REFERENCE)
    for row in rows:
        if row.get("reference_level", "").lower() == "true":
            factor = row.get("intervention", "")
            level = row.get("level", "")
            if factor and level:
                references[factor] = level
    return references


def _mechanism_recovery(
    rows: Sequence[CsvRow],
    config: ConfirmatoryReportConfig,
    registry: AnalysisFamilyRegistry | None,
) -> dict[str, object]:
    metric_names = (
        "frequency_recovery_mae",
        "damping_correlation",
        "damping_regime_auc",
        "impulse_response_nmse",
        "test_loss",
    )
    grouped: dict[tuple[str, str], list[CsvRow]] = defaultdict(list)
    for row in rows:
        grouped[(_model(row), row.get("dataset_or_task", ""))].append(row)
    summaries = []
    for (model, task), task_rows in sorted(grouped.items()):
        summaries.append(
            {
                "model": model,
                "task": task,
                "metrics": {
                    metric: _bootstrap_rows(
                        task_rows,
                        metric,
                        config,
                        f"mechanism:{model}:{task}:{metric}",
                    )
                    for metric in metric_names
                },
            }
        )
    metric_direction = {
        "frequency_recovery_mae": "lower",
        "damping_correlation": "higher",
        "damping_regime_auc": "higher",
        "impulse_response_nmse": "lower",
        "test_loss": "lower",
    }
    paired = []
    for metric in metric_names:
        differences = _paired_differences(
            rows,
            reference="pac_tf",
            comparator="pac_tf_fixed_damping",
            metric=metric,
            group_field="dataset_or_task",
        )
        summary = paired_test_summary(
            differences,
            iterations=config.bootstrap_iterations,
            seed=config.bootstrap_seed,
            label=f"mechanism-paired:{metric}",
        )
        summary.update(
            {
                "metric": metric,
                "hypothesis_id": f"pac_tf_vs_fixed_damping__{metric}",
                "difference": "pac_tf_minus_pac_tf_fixed_damping",
                "preferred_direction": metric_direction[metric],
            }
        )
        paired.append(summary)
    paired = _registered_bh(
        paired,
        registry,
        "mechanism_recovery_pac_tf_vs_fixed_damping",
    )
    return {
        "status": "available" if rows else "missing",
        "analysis_status": _family_analysis_status(
            rows,
            paired,
            required_fields=("test_loss",),
        ),
        "task_model_summaries": summaries,
        "pac_tf_vs_fixed_damping": paired,
    }


def _interventions(  # noqa: C901, PLR0912, PLR0915 - heterogeneous surfaces remain explicit
    rows: Sequence[CsvRow],
    config: ConfirmatoryReportConfig,
    registry: AnalysisFamilyRegistry | None,
) -> dict[str, object]:
    grouped: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        delta = _float(row.get("metric_delta"))
        if delta is not None:
            key = (
                row.get("architecture_surface", ""),
                _model(row),
                row.get("intervention", ""),
            )
            grouped[key][row.get("dataset_or_task", "")].append(delta)
    summaries = []
    for (surface, model, intervention), task_values in sorted(grouped.items()):
        summary = hierarchical_bootstrap_summary(
            task_values,
            iterations=config.bootstrap_iterations,
            seed=config.bootstrap_seed,
            label=f"intervention:{surface}:{model}:{intervention}",
        )
        if surface == "tight_frame_classifier":
            effect = "intervention_minus_baseline_balanced_accuracy; negative is harmful"
        else:
            effect = "intervention_minus_baseline_loss; positive is harmful"
        summaries.append(
            {
                "architecture_surface": surface,
                "model": model,
                "intervention": intervention,
                "metric_delta": summary,
                "interpretation": effect,
            }
        )
    index: dict[tuple[str, str, str, int, str, str], float] = {}
    for row in rows:
        seed = _int(row.get("seed"))
        delta = _float(row.get("metric_delta"))
        if seed is None or delta is None:
            continue
        index[
            (
                row.get("architecture_surface", ""),
                _model(row),
                row.get("dataset_or_task", ""),
                seed,
                row.get("intervention", ""),
                row.get("teacher_mode_index", ""),
            )
        ] = delta
    specificity = []
    for surface, model in sorted({(key[0], key[1]) for key in index}):
        differences: dict[str, list[float]] = defaultdict(list)
        tasks = sorted({key[2] for key in index if key[:2] == (surface, model)})
        seeds = sorted({key[3] for key in index if key[:2] == (surface, model)})
        for task in tasks:
            for seed in seeds:
                teacher_modes = sorted(
                    {
                        key[5]
                        for key in index
                        if key[:4] == (surface, model, task, seed)
                        and key[4] in {"matched_mode_knockout", "random_mode_knockout"}
                    }
                )
                for teacher_mode in teacher_modes:
                    matched = index.get(
                        (surface, model, task, seed, "matched_mode_knockout", teacher_mode)
                    )
                    random = index.get(
                        (surface, model, task, seed, "random_mode_knockout", teacher_mode)
                    )
                    if matched is not None and random is not None:
                        differences[f"{task}/teacher_mode_{teacher_mode}"].append(
                            matched - random
                        )
        if not differences:
            continue
        summary = paired_test_summary(
            differences,
            iterations=config.bootstrap_iterations,
            seed=config.bootstrap_seed,
            label=f"mode-specificity:{surface}:{model}",
        )
        summary.update(
            {
                "architecture_surface": surface,
                "model": model,
                "hypothesis_id": (
                    f"{model}__matched_mode_knockout_vs_random_mode_knockout"
                ),
                "difference": "matched_mode_delta_minus_random_mode_delta",
                "interpretation": (
                    "positive means teacher-matched knockout raises loss more than random knockout"
                ),
            }
        )
        specificity.append(summary)
    specificity = _registered_bh(
        specificity,
        registry,
        "interpretability_mode_specificity",
    )

    mechanism_interventions = []
    for model in ("pac_tf", "pac_tf_fixed_damping"):
        grouped_deltas = _group_intervention_deltas(
            rows,
            architecture_surface="causal_tight_frame_sequence_regressor",
            model=model,
            intervention="frame_subspace_perturbation",
        )
        summary = paired_test_summary(
            grouped_deltas,
            iterations=config.bootstrap_iterations,
            seed=config.bootstrap_seed,
            label=f"mechanism-intervention:{model}:frame",
        )
        summary.update(
            {
                "model": model,
                "intervention": "frame_subspace_perturbation",
                "hypothesis_id": f"{model}__frame_subspace_perturbation_vs_zero_delta",
                "difference": "intervention_minus_baseline_test_loss",
            }
        )
        mechanism_interventions.append(summary)
    mechanism_interventions = _registered_bh(
        mechanism_interventions,
        registry,
        "interpretability_mechanism_interventions",
    )

    classifier_hypotheses = {
        "moment_head_intervention": "moment_head_removal_vs_zero_delta",
        "forward_direction_removal": "forward_direction_removal_vs_zero_delta",
        "backward_direction_removal": "backward_direction_removal_vs_zero_delta",
        "lag1_intervention": "lag1_removal_vs_zero_delta",
        "lag4_intervention": "lag4_removal_vs_zero_delta",
    }
    classifier_interventions = []
    for intervention, hypothesis_id in classifier_hypotheses.items():
        grouped_deltas = _group_intervention_deltas(
            rows,
            architecture_surface="tight_frame_classifier",
            model=None,
            intervention=intervention,
        )
        summary = paired_test_summary(
            grouped_deltas,
            iterations=config.bootstrap_iterations,
            seed=config.bootstrap_seed,
            label=f"classifier-intervention:{intervention}",
        )
        summary.update(
            {
                "intervention": intervention,
                "hypothesis_id": hypothesis_id,
                "difference": "intervention_minus_baseline_validation_balanced_accuracy",
            }
        )
        classifier_interventions.append(summary)
    classifier_interventions = _registered_bh(
        classifier_interventions,
        registry,
        "interpretability_classifier_interventions",
    )
    recovery_controls = []
    for intervention in (
        "untrained_initialization_recovery",
        "random_grid_recovery",
    ):
        control_rows = [row for row in rows if row.get("intervention") == intervention]
        if not control_rows:
            continue
        recovery_controls.append(
            {
                "intervention": intervention,
                "frequency_mae_improvement": _bootstrap_rows(
                    control_rows,
                    "frequency_recovery_improvement_from_control",
                    config,
                    f"recovery-control:{intervention}:frequency",
                ),
                "damping_mae_improvement": _bootstrap_rows(
                    control_rows,
                    "damping_recovery_improvement_from_control",
                    config,
                    f"recovery-control:{intervention}:damping",
                ),
                "positive_means_training_improved_recovery": True,
            }
        )
    family_rows = [*specificity, *mechanism_interventions, *classifier_interventions]
    return {
        "status": "available" if rows else "missing",
        "analysis_status": _family_analysis_status(
            rows,
            family_rows,
            required_fields=("metric_delta",),
        ),
        "summaries": summaries,
        "matched_vs_random_mode_specificity": specificity,
        "mechanism_intervention_tests": mechanism_interventions,
        "classifier_intervention_tests": classifier_interventions,
        "recovery_improvement_from_controls": recovery_controls,
    }


def _manifest_bound_source(  # noqa: C901, PLR0912 - explicit provenance quarantine
    manifest: Path,
    rows: Sequence[CsvRow],
    *,
    result_key: str,
    manifest_filter: Callable[[dict[str, object]], bool] | None = None,
    protocol_id: str,
    protocol_sha256: str,
    strict_provenance: bool,
) -> tuple[list[CsvRow], dict[str, object]]:
    manifest_rows = _read_jsonl_strict(manifest)
    if manifest_filter is not None:
        manifest_rows = [row for row in manifest_rows if manifest_filter(row)]
    expected_rows: dict[str, dict[str, object]] = {}
    for row in manifest_rows:
        key = str(row.get("key", ""))
        if not key or key in expected_rows:
            message = f"locked manifest has an empty or duplicate key: {manifest}"
            raise ValueError(message)
        expected_rows[key] = row
    observed_rows: dict[str, list[CsvRow]] = defaultdict(list)
    for row in rows:
        key = row.get(result_key, "")
        if key:
            observed_rows[key].append(row)
    expected = set(expected_rows)
    observed = set(observed_rows)
    unexpected = sorted(observed - expected)
    duplicate = sorted(key for key, values in observed_rows.items() if len(values) != 1)
    mismatched: dict[str, list[str]] = {}
    accepted: list[CsvRow] = []
    for key in sorted(expected & observed):
        if key in duplicate:
            continue
        row = observed_rows[key][0]
        reasons = _provenance_mismatches(
            expected_rows[key],
            row,
            protocol_id=protocol_id,
            protocol_sha256=protocol_sha256,
            strict=strict_provenance,
        )
        if reasons:
            mismatched[key] = reasons
            continue
        accepted.append(row)
    by_analysis_identity: dict[tuple[str, ...], list[CsvRow]] = defaultdict(list)
    for row in accepted:
        by_analysis_identity[_analysis_identity(row)].append(row)
    duplicate_analysis_cells = sorted(
        "/".join(identity)
        for identity, values in by_analysis_identity.items()
        if len(values) > 1
    )
    if duplicate_analysis_cells:
        accepted = [
            row
            for row in accepted
            if len(by_analysis_identity[_analysis_identity(row)]) == 1
        ]
    accepted_keys = {row[result_key] for row in accepted}
    integrity_clean = (
        not unexpected and not duplicate and not duplicate_analysis_cells and not mismatched
    )
    if not manifest.exists():
        status = "missing"
    elif expected and accepted_keys == expected and integrity_clean:
        status = "complete"
    elif observed or unexpected or duplicate or mismatched:
        status = "partial"
    else:
        status = "pending"
    return accepted, {
        "status": status,
        "execution_status": (
            "complete"
            if expected and expected <= observed
            else "partial"
            if observed
            else "pending"
            if manifest.exists()
            else "missing"
        ),
        "integrity_status": "clean" if integrity_clean else "quarantined_rows_present",
        "manifest": str(manifest),
        "expected_rows": len(expected),
        "completed_rows": len(accepted_keys),
        "missing_rows": len(expected - accepted_keys),
        "unexpected_completed_rows": len(unexpected),
        "unexpected_keys": unexpected,
        "duplicate_result_keys": duplicate,
        "duplicate_analysis_cells": duplicate_analysis_cells,
        "provenance_mismatches": mismatched,
    }


_IDENTITY_FIELDS = (
    "protocol_id",
    "protocol_sha256",
    "capacity_artifact_sha256",
    "selected_model",
    "selected_model_dim",
    "selected_modes",
    "seed",
    "model",
    "package",
    "reference_model",
    "selection_trial",
    "validation_trial",
    "refit_epochs",
    "learning_rate",
    "weight_decay",
    "runtime",
    "batch_size",
    "baseline_family",
    "evaluation_collection",
    "evaluation_split",
    "intervention",
    "level",
)

_IDENTITY_ALIASES = (
    ("dataset", ("dataset_or_task",)),
    ("scope", ("dataset_or_task",)),
    ("ratio", ("requested_ratio", "data_ratio")),
    ("length", ("sequence_length",)),
)


def _provenance_mismatches(  # noqa: C901, PLR0912 - explicit provenance audit
    manifest: Mapping[str, object],
    result: Mapping[str, str],
    *,
    protocol_id: str,
    protocol_sha256: str,
    strict: bool,
) -> list[str]:
    reasons: list[str] = []
    if strict:
        manifest_protocol_sha = manifest.get("protocol_sha256")
        if manifest_protocol_sha is not None and str(manifest_protocol_sha) != protocol_sha256:
            reasons.append("manifest.protocol_sha256")
        manifest_protocol_id = manifest.get("protocol_id")
        if manifest_protocol_id is not None and str(manifest_protocol_id) != protocol_id:
            reasons.append("manifest.protocol_id")
        result_protocol_sha = result.get("protocol_sha256")
        if result_protocol_sha and result_protocol_sha != protocol_sha256:
            reasons.append("result.protocol_sha256")
        result_protocol_id = result.get("protocol_id")
        if result_protocol_id and result_protocol_id != protocol_id:
            reasons.append("result.protocol_id")
    critical = {
        "protocol_id",
        "protocol_sha256",
        "capacity_artifact_sha256",
        "selected_model",
        "selected_model_dim",
        "selected_modes",
    }
    for field in _IDENTITY_FIELDS:
        if field not in manifest or manifest.get(field) is None:
            continue
        result_value = result.get(field)
        if result_value in (None, ""):
            if strict and field in critical:
                reasons.append(f"missing_result.{field}")
            continue
        if not _identity_equal(manifest[field], result_value):
            reasons.append(field)
    for manifest_field, result_fields in _IDENTITY_ALIASES:
        if manifest_field not in manifest or manifest.get(manifest_field) is None:
            continue
        result_value = next(
            (result[field] for field in result_fields if result.get(field) not in (None, "")),
            None,
        )
        if result_value is not None and not _identity_equal(manifest[manifest_field], result_value):
            reasons.append(f"{manifest_field}->{result_fields[0]}")
    return sorted(set(reasons))


def _identity_equal(expected: object, observed: str) -> bool:
    expected_number = _object_float(expected)
    observed_number = _object_float(observed)
    if expected_number is not None and observed_number is not None:
        return math.isclose(expected_number, observed_number, rel_tol=1.0e-12, abs_tol=1.0e-12)
    if isinstance(expected, bool):
        return observed.lower() == str(expected).lower()
    return str(expected) == observed


def _analysis_identity(row: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(
        row.get(field, "")
        for field in (
            "dataset_or_task",
            "seed",
            "baseline_family",
            "model",
            "requested_ratio",
            "data_ratio",
            "intervention",
            "level",
            "architecture_surface",
            "teacher_mode_index",
            "runtime",
            "sequence_length",
            "batch_size",
        )
    )


def _validate_selected_contract(config: ConfirmatoryReportConfig, protocol_sha256: str) -> None:
    if not config.strict_provenance:
        return
    contract = config.evidence_root / "evidence_contract.json"
    has_material = contract.exists() or any(config.evidence_root.glob("*_manifest.jsonl")) or any(
        (config.evidence_root / "results").glob("*.csv")
    )
    if not has_material:
        return
    if not contract.is_file():
        message = "selected evidence material exists without evidence_contract.json"
        raise ValueError(message)
    binding = validate_selected_evidence_root(config.evidence_root)
    if binding.protocol_sha256 != protocol_sha256:
        message = "selected evidence contract is bound to a different active protocol"
        raise ValueError(message)


def _validate_unseen_collection_lock(
    config: ConfirmatoryReportConfig, protocol_sha256: str
) -> None:
    path = config.unseen_root / "reports" / "unseen_final_collection_lock.json"
    if not path.is_file():
        message = "unseen-final rows exist without their collection lock"
        raise ValueError(message)
    payload = _json_object_from_path(path, "unseen-final collection lock")
    if payload.get("schema_version") != "pac_unseen_final_collection.v1":
        message = "unsupported unseen-final collection lock schema"
        raise ValueError(message)
    if payload.get("protocol_sha256") != protocol_sha256:
        message = "unseen-final collection lock is bound to a different protocol"
        raise ValueError(message)


def _done_rows(rows: Sequence[CsvRow]) -> list[CsvRow]:
    return [row for row in rows if row.get("status") == "done"]


def _read_jsonl_strict(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            message = f"invalid JSONL in locked manifest {path}:{line_number}"
            raise ValueError(message) from error
        if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
            message = f"locked manifest row is not an object: {path}:{line_number}"
            raise TypeError(message)
        rows.append(value)
    return rows


def _registered_bh(
    rows: Sequence[dict[str, object]],
    registry: AnalysisFamilyRegistry | None,
    family_id: str,
) -> list[dict[str, object]]:
    output = [dict(row) for row in rows]
    if registry is None:
        return bh_fdr(output)
    family = registry.family(family_id)
    observed = [str(row.get("hypothesis_id", "")) for row in output]
    if any(not hypothesis for hypothesis in observed) or len(set(observed)) != len(observed):
        message = f"BH family has empty or duplicate hypothesis ids: {family_id}"
        raise ValueError(message)
    extras = sorted(set(observed) - set(family.hypotheses))
    if extras:
        message = f"BH family contains unregistered hypotheses: {family_id}: {extras}"
        raise ValueError(message)
    complete = set(observed) == set(family.hypotheses) and all(
        _object_float(row.get("wilcoxon_p")) is not None for row in output
    )
    for row in output:
        row["fdr_family_id"] = family_id
        row["fdr_family_size"] = family.expected_hypothesis_count
        row["fdr_family_status"] = "complete" if complete else "incomplete_not_adjusted"
    return bh_fdr(output, alpha=family.alpha) if complete else output


def _required_rows_status(rows: Sequence[CsvRow], fields: Sequence[str]) -> str:
    if not rows:
        return "missing"
    return (
        "complete"
        if all(_float(row.get(field)) is not None for row in rows for field in fields)
        else "partial"
    )


def _family_analysis_status(
    rows: Sequence[CsvRow],
    comparisons: Sequence[Mapping[str, object]],
    *,
    required_fields: Sequence[str],
) -> str:
    if not rows:
        return "missing"
    if _required_rows_status(rows, required_fields) != "complete":
        return "partial"
    if not comparisons or any(
        row.get("fdr_family_status") == "incomplete_not_adjusted" for row in comparisons
    ):
        return "partial"
    return "complete"


def _apply_execution_status(
    section: dict[str, object], *sources: Mapping[str, object]
) -> None:
    execution = _combined_source_status(*sources)
    analysis = str(section.get("analysis_status", "missing"))
    section["execution_status"] = execution
    if execution == "complete" and analysis == "complete":
        section["status"] = "complete"
    elif execution == "missing" and analysis == "missing":
        section["status"] = "missing"
    elif execution == "pending" and analysis == "missing":
        section["status"] = "pending"
    else:
        section["status"] = "partial"


def _registry_metadata(
    config: ConfirmatoryReportConfig, registry: AnalysisFamilyRegistry | None
) -> dict[str, object]:
    if registry is None or config.analysis_registry_path is None:
        return {"status": "not_configured"}
    return {
        "status": "validated",
        "path": str(config.analysis_registry_path),
        "sha256": hashlib.sha256(config.analysis_registry_path.read_bytes()).hexdigest(),
        "registry_id": registry.registry_id,
        "family_ids": [family.family_id for family in registry.families],
        "hypothesis_count": sum(
            family.expected_hypothesis_count for family in registry.families
        ),
    }


def _ratio_hypothesis_label(ratio: float) -> str:
    return f"{ratio:.2f}".replace(".", "p")


def _same_float_grid(left: Sequence[float], right: Sequence[float]) -> bool:
    return len(left) == len(right) and all(
        math.isclose(a, b, rel_tol=1.0e-12, abs_tol=1.0e-12)
        for a, b in zip(left, right, strict=True)
    )


def _sensitivity_level_order(manifest: Path) -> dict[str, tuple[str, ...]]:
    levels: dict[str, list[str]] = defaultdict(list)
    references: dict[str, str] = {}
    for row in _read_jsonl_strict(manifest):
        factor = str(row.get("intervention", ""))
        level = str(row.get("level", ""))
        if not factor or not level:
            continue
        if row.get("reference_level") is True:
            references[factor] = level
        elif level not in levels[factor]:
            levels[factor].append(level)
    return {
        factor: tuple(sorted(level for level in values if level != references.get(factor)))
        for factor, values in levels.items()
    }


def _group_intervention_deltas(
    rows: Sequence[CsvRow],
    *,
    architecture_surface: str,
    model: str | None,
    intervention: str,
) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("architecture_surface") != architecture_surface:
            continue
        if row.get("intervention") != intervention:
            continue
        if model is not None and _model(row) != model:
            continue
        value = _float(row.get("metric_delta"))
        if value is not None:
            grouped[row.get("dataset_or_task", "")].append(value)
    return grouped


def _json_object_from_path(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        message = f"{label} is not valid JSON: {path}"
        raise ValueError(message) from error
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        message = f"{label} must be a JSON object: {path}"
        raise TypeError(message)
    return value


def _combined_source_status(*sources: Mapping[str, object]) -> str:
    statuses = {str(source.get("status", "missing")) for source in sources}
    if statuses == {"complete"}:
        return "complete"
    completed = sum(_object_int(source.get("completed_rows")) or 0 for source in sources)
    if completed:
        return "partial"
    if "pending" in statuses or "partial" in statuses or "complete" in statuses:
        return "pending"
    return "missing"


def _paired_differences(
    rows: Sequence[CsvRow],
    *,
    reference: str,
    comparator: str,
    metric: str,
    group_field: str,
) -> dict[str, list[float]]:
    indexed: dict[tuple[str, int, str], float] = {}
    for row in rows:
        seed = _int(row.get("seed"))
        value = _float(row.get(metric))
        if seed is not None and value is not None:
            key = (row.get(group_field, ""), seed, _model(row))
            if key in indexed:
                message = f"ambiguous duplicate paired cell: {key}"
                raise ValueError(message)
            indexed[key] = value
    grouped: dict[str, list[float]] = defaultdict(list)
    groups = sorted({key[0] for key in indexed})
    seeds = sorted({key[1] for key in indexed})
    for group in groups:
        for seed in seeds:
            base = indexed.get((group, seed, reference))
            other = indexed.get((group, seed, comparator))
            if base is not None and other is not None:
                grouped[group].append(base - other)
    return grouped


def _mark_pareto(
    cells: list[dict[str, object]], *, expected_models: Sequence[str]
) -> None:
    slices: dict[tuple[int, int, str], list[dict[str, object]]] = defaultdict(list)
    for cell in cells:
        length = _object_int(cell.get("sequence_length"))
        batch = _object_int(cell.get("batch_size"))
        if length is None or batch is None:
            continue
        slices[
            (
                length,
                batch,
                str(cell["runtime"]),
            )
        ].append(cell)
    for slice_cells in slices.values():
        observed_models = {str(cell.get("model", "")) for cell in slice_cells}
        slice_complete = observed_models == set(expected_models) and all(
            cell.get("cell_status") in {"measured", "censored"} for cell in slice_cells
        )
        eligible = [
            cell
            for cell in slice_cells
            if slice_complete
            and cell.get("cell_status") == "measured"
            and cell.get("accuracy_joined_from_final_balanced_accuracy") is not None
            and _summary_mean(cell.get("latency_ms")) is not None
            and cell.get("params_trainable") is not None
        ]
        measured_count = sum(cell.get("cell_status") == "measured" for cell in slice_cells)
        slice_complete = slice_complete and len(eligible) == measured_count
        use_memory = bool(eligible) and all(
            _summary_mean(cell.get("peak_memory_mb")) is not None for cell in eligible
        )
        dimensions = ["accuracy(max)", "latency_ms(min)", "params(min)"]
        if use_memory:
            dimensions.append("peak_memory_mb(min)")
        for cell in slice_cells:
            cell["pareto_dimensions"] = dimensions
            if not slice_complete:
                cell["pareto_status"] = "provisional_incomplete_slice"
                cell["pareto_nondominated"] = None
                continue
            if cell.get("cell_status") == "censored":
                cell["pareto_status"] = "censored"
                cell["pareto_nondominated"] = None
                continue
            if cell not in eligible:
                cell["pareto_nondominated"] = None
                continue
            dominated = any(
                _dominates(other, cell, use_memory=use_memory)
                for other in eligible
                if other is not cell
            )
            cell["pareto_nondominated"] = not dominated
            cell["pareto_status"] = "evaluated"


def _dominates(
    candidate: Mapping[str, object], target: Mapping[str, object], *, use_memory: bool
) -> bool:
    candidate_accuracy = _object_float(
        candidate.get("accuracy_joined_from_final_balanced_accuracy")
    )
    candidate_latency = _summary_mean(candidate.get("latency_ms"))
    candidate_params = _object_float(candidate.get("params_trainable"))
    target_accuracy = _object_float(target.get("accuracy_joined_from_final_balanced_accuracy"))
    target_latency = _summary_mean(target.get("latency_ms"))
    target_params = _object_float(target.get("params_trainable"))
    if (
        candidate_accuracy is None
        or candidate_latency is None
        or candidate_params is None
        or target_accuracy is None
        or target_latency is None
        or target_params is None
    ):
        return False
    candidate_values = (
        -candidate_accuracy,
        candidate_latency,
        candidate_params,
    )
    target_values = (
        -target_accuracy,
        target_latency,
        target_params,
    )
    if use_memory:
        candidate_memory = _summary_mean(candidate.get("peak_memory_mb"))
        target_memory = _summary_mean(target.get("peak_memory_mb"))
        if candidate_memory is None or target_memory is None:
            return False
        candidate_values += (candidate_memory,)
        target_values += (target_memory,)
    weakly_better = all(
        left <= right for left, right in zip(candidate_values, target_values, strict=True)
    )
    strictly_better = any(
        left < right for left, right in zip(candidate_values, target_values, strict=True)
    )
    return weakly_better and strictly_better


def _final_accuracy_by_model(final: Mapping[str, object]) -> dict[str, float]:
    output: dict[str, float] = {}
    aggregate = final.get("model_aggregate")
    if not isinstance(aggregate, list):
        return output
    for row in aggregate:
        if not isinstance(row, dict):
            continue
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            continue
        balanced = metrics.get("balanced_accuracy")
        mean = _summary_mean(balanced)
        if mean is not None:
            output[str(row.get("model", ""))] = mean
    return output


def _write_diagnostic_artifact(
    root: Path,
    *,
    scope: str,
    model: str,
    seed: str,
    domain: str,
    diagnostics: Mapping[str, str],
) -> Path | None:
    confusion = _json_value(diagnostics.get("confusion_matrix_json", ""))
    per_class = _json_value(diagnostics.get("per_class_metrics_json", ""))
    if confusion is None or per_class is None:
        return None
    path = root / (f"{_slug(scope)}__{_slug(model)}__seed{_slug(seed)}__{_slug(domain)}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(
        path,
        {
            "schema_version": "pac_tf_classification_diagnostics.v1",
            "dataset_or_task": scope,
            "model": model,
            "seed": _int(seed),
            "domain": domain,
            "confusion_matrix": confusion,
            "per_class_metrics": per_class,
        },
    )
    return path


def _diagnostic_metrics(row: Mapping[str, str]) -> dict[str, float]:
    output: dict[str, float] = {}
    for key in ("nll", "ece_15", "brier_score"):
        value = _float(row.get(key))
        if value is not None:
            output[key] = value
    return output


def _bootstrap_rows(
    rows: Sequence[CsvRow],
    field: str,
    config: ConfirmatoryReportConfig,
    label: str,
) -> dict[str, object]:
    return bootstrap_summary(
        _float_values(rows, field),
        iterations=config.bootstrap_iterations,
        seed=config.bootstrap_seed,
        label=label,
    )


def _bootstrap_field(
    rows: Sequence[Mapping[str, float]],
    field: str,
    config: ConfirmatoryReportConfig,
    label: str,
) -> dict[str, object]:
    return bootstrap_summary(
        [row[field] for row in rows if field in row],
        iterations=config.bootstrap_iterations,
        seed=config.bootstrap_seed,
        label=label,
    )


def _group_values(
    rows: Sequence[CsvRow], group_field: str, value_field: str
) -> dict[str, list[float]]:
    output: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = _float(row.get(value_field))
        if value is not None:
            output[row.get(group_field, "")].append(value)
    return output


def _float_values(rows: Sequence[Mapping[str, str]], field: str) -> list[float]:
    return [value for row in rows if (value := _float(row.get(field))) is not None]


def _int_values(rows: Sequence[Mapping[str, str]], field: str) -> set[int]:
    return {value for row in rows if (value := _int(row.get(field))) is not None}


def _mean_or_none(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _min_or_none(values: Sequence[float]) -> float | None:
    return float(min(values)) if values else None


def _summary_mean(value: object) -> float | None:
    if not isinstance(value, dict):
        return None
    return _object_float(value.get("mean"))


def _model(row: Mapping[str, str]) -> str:
    return row.get("baseline_family") or row.get("model", "")


def _float(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _object_float(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if not isinstance(value, (int, float, str)):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _object_int(value: object) -> int | None:
    parsed = _object_float(value)
    return int(parsed) if parsed is not None else None


def _object_list(payload: Mapping[str, object], key: str) -> list[object]:
    value = payload.get(key)
    return value if isinstance(value, list) else []


def _protocol_int_tuple(payload: Mapping[str, object], key: str) -> tuple[int, ...]:
    output: list[int] = []
    for item in _object_list(payload, key):
        value = _object_int(item)
        if value is not None:
            output.append(value)
    return tuple(output)


def _protocol_float_tuple(payload: Mapping[str, object], key: str) -> tuple[float, ...]:
    output: list[float] = []
    for item in _object_list(payload, key):
        value = _object_float(item)
        if value is not None:
            output.append(value)
    return tuple(output)


def _int(value: str | None) -> int | None:
    parsed = _float(value)
    return int(parsed) if parsed is not None else None


def _json_value(value: str) -> object | None:
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _json_list(value: str) -> list[dict[str, object]] | None:
    parsed = _json_value(value)
    if not isinstance(parsed, list) or not all(isinstance(item, dict) for item in parsed):
        return None
    return parsed


def _json_dict(value: str) -> dict[str, object] | None:
    parsed = _json_value(value)
    return parsed if isinstance(parsed, dict) else None


def _read_csv(path: Path) -> list[CsvRow]:
    if not path.exists():
        return []
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except (csv.Error, OSError, UnicodeDecodeError):
        return []


def _load_protocol(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        message = "confirmatory protocol must be a JSON object"
        raise TypeError(message)
    if payload.get("locked_before_final_evaluation") is not True:
        message = "confirmatory report refused: protocol is not locked"
        raise ValueError(message)
    return payload


def _equivalence_margin(protocol: Mapping[str, object]) -> float | None:
    value = protocol.get("equivalence_margin")
    return _object_float(value)


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip()).strip("-.")
    return cleaned or "unknown"


def _atomic_write_json(path: Path, payload: object) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary).replace(path)
    finally:
        temporary_path = Path(temporary)
        if temporary_path.exists():
            temporary_path.unlink()


def render_markdown(  # noqa: C901, PLR0912 - report sections mirror the evidence contract
    report: Mapping[str, object],
) -> str:
    boundary_prefix = "This report contains only completed rows from locked manifests. "
    boundary_suffix = (
        "Empty, pending, failed, malformed, or unmatched cells are not converted into "
        "numerical evidence."
    )
    lines = [
        "# PAC-TF Confirmatory Analysis",
        "",
        f"- report status: **{report.get('report_status', 'partial')}**",
        f"- protocol: `{report.get('protocol_id', '')}`",
        "- missing cells are not imputed; all incomplete sections remain explicitly partial.",
        "- paired differences are PAC-TF minus baseline unless a section states otherwise.",
        "- no equivalence claim is made because no separately locked margin is present.",
        "",
        "## Source completeness",
        "",
        "| source | status | complete / expected | missing |",
        "|---|---:|---:|---:|",
    ]
    sources = report.get("sources", {})
    if isinstance(sources, dict):
        for name, value in sorted(sources.items()):
            source = value if isinstance(value, dict) else {}
            lines.append(
                "".join(
                    (
                        f"| {name} | {source.get('status', 'missing')} | ",
                        f"{source.get('completed_rows', 0)} / ",
                        f"{source.get('expected_rows', 0)} | ",
                        f"{source.get('missing_rows', 0)} |",
                    )
                )
            )
    lines.extend(("", "## Held-out final evaluation", ""))
    final = report.get("final_evaluation", {})
    if isinstance(final, dict):
        aggregate = final.get("model_aggregate", [])
        lines.extend(("| model | balanced accuracy mean | 95% CI |", "|---|---:|---:|"))
        if isinstance(aggregate, list):
            for row in aggregate:
                if not isinstance(row, dict):
                    continue
                metrics = row.get("metrics", {})
                balanced = metrics.get("balanced_accuracy", {}) if isinstance(metrics, dict) else {}
                lines.append(
                    "".join(
                        (
                            f"| {row.get('model', '')} | ",
                            f"{_fmt_summary_mean(balanced)} | ",
                            f"{_fmt_summary_ci(balanced)} |",
                        )
                    )
                )
        lines.extend(("", "### PAC-TF paired comparisons", ""))
        lines.extend(
            (
                "| baseline | mean difference | 95% CI | Wilcoxon p | BH q |",
                "|---|---:|---:|---:|---:|",
            )
        )
        comparisons = final.get("paired_pac_tf_vs_baselines", [])
        if isinstance(comparisons, list):
            lines.extend(
                (
                    f"| {row.get('baseline', '')} | {_fmt(row.get('mean'))} | "
                    f"[{_fmt(row.get('ci95_low'))}, {_fmt(row.get('ci95_high'))}] | "
                    f"{_fmt(row.get('wilcoxon_p'))} | {_fmt(row.get('fdr_q'))} |"
                )
                for row in comparisons
                if isinstance(row, dict)
            )
    lines.extend(("", "## Low-data log-ratio AUC", ""))
    low_data = report.get("low_data", {})
    lines.extend(("| dataset | model | normalized AUC | curve status |", "|---|---|---:|---:|"))
    if isinstance(low_data, dict) and isinstance(low_data.get("log_ratio_auc"), list):
        lines.extend(
            (
                f"| {row.get('dataset', '')} | {row.get('model', '')} | "
                f"{_fmt_summary_mean(row.get('normalized_log_ratio_auc'))} | "
                f"{row.get('curve_status', 'partial')} |"
            )
            for row in low_data["log_ratio_auc"]
            if isinstance(row, dict)
        )
    lines.extend(("", "## Core ablation", ""))
    lines.extend(
        (
            "| removed component | reference - ablation | 95% CI | BH q |",
            "|---|---:|---:|---:|",
        )
    )
    core = report.get("core_ablation", {})
    if isinstance(core, dict) and isinstance(core.get("comparisons"), list):
        lines.extend(
            (
                f"| {row.get('intervention', '')} | {_fmt(row.get('mean'))} | "
                f"[{_fmt(row.get('ci95_low'))}, {_fmt(row.get('ci95_high'))}] | "
                f"{_fmt(row.get('fdr_q'))} |"
            )
            for row in core["comparisons"]
            if isinstance(row, dict)
        )
    calibration = report.get("calibration_and_error_analysis", {})
    records = (
        calibration.get("per_run_confusion_and_per_class_artifacts", [])
        if isinstance(calibration, dict)
        else []
    )
    lines.extend(("", "## Diagnostic artifacts", ""))
    if isinstance(records, list) and records:
        for row in records:
            if isinstance(row, dict):
                path = str(row.get("artifact", ""))
                lines.append(
                    "".join(
                        (
                            f"- {row.get('dataset', '')} / {row.get('model', '')} / ",
                            f"seed {row.get('seed', '')} / {row.get('domain', '')}: ",
                            f"[{path}]({path})",
                        )
                    )
                )
    else:
        lines.append("- [missing] No completed calibration/confusion rows are available yet.")
    efficiency = report.get("efficiency", {})
    cells = efficiency.get("cells", []) if isinstance(efficiency, dict) else []
    measured = sum(
        isinstance(cell, dict) and cell.get("cell_status") == "measured"
        for cell in cells
        if isinstance(cells, list)
    )
    censored = sum(
        isinstance(cell, dict) and cell.get("cell_status") in {"censored", "mixed_censored"}
        for cell in cells
        if isinstance(cells, list)
    )
    lines.extend(
        (
            "",
            "## Efficiency and Pareto inventory",
            "",
            f"- measured cells: {measured}",
            f"- censored/resource-limit/compile-unsupported cells retained: {censored}",
            "- full cell-level Pareto flags are in `confirmatory_analysis.json`.",
            "",
            "## Evidence boundary",
            "",
            f"{boundary_prefix}{boundary_suffix}",
        )
    )
    return "\n".join(lines) + "\n"


def _fmt_summary_mean(value: object) -> str:
    return _fmt(_summary_mean(value))


def _fmt_summary_ci(value: object) -> str:
    if not isinstance(value, dict):
        return "—"
    return f"[{_fmt(value.get('ci95_low'))}, {_fmt(value.get('ci95_high'))}]"


def _fmt(value: object) -> str:
    number = _object_float(value)
    return "—" if number is None else f"{number:.4f}"
