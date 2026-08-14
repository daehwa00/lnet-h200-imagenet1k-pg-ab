from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

SEEDS = (7, 11, 19, 23, 31)


def _number(row: dict[str, object], key: str) -> float:
    value = row[key]
    if not isinstance(value, (int, float)):
        message = f"{key} must be numeric"
        raise TypeError(message)
    return float(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _balanced_accuracy(directory: Path) -> dict[str, float]:
    values = [
        float(
            cast(
                "dict[str, dict[str, float]]",
                json.loads((directory / f"alphabet-seed{seed}.json").read_text()),
            )["test"]["balanced_accuracy"]
        )
        for seed in SEEDS
    ]
    return {
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _group_curve_metrics(
    rows: list[dict[str, object]],
    independent: str,
) -> list[dict[str, object]]:
    groups: dict[tuple[str, float], list[float]] = {}
    for row in rows:
        key = (str(row["model"]), _number(row, independent))
        groups.setdefault(key, []).append(_number(row, "balanced_accuracy"))
    return [
        {
            "model": model,
            independent: value,
            "balanced_accuracy_mean": statistics.fmean(scores),
            "balanced_accuracy_sample_std": statistics.stdev(scores),
        }
        for (model, value), scores in sorted(groups.items())
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.results_dir

    pole_rows: list[dict[str, object]] = []
    correlations: list[dict[str, object]] = []
    artifacts: dict[str, dict[str, object]] = {}
    for seed in SEEDS:
        path = root / f"pole-audit-seed{seed}.json"
        audit = cast("dict[str, object]", json.loads(path.read_text()))
        correlations.append(
            {
                "seed": seed,
                **cast("dict[str, float]", audit["rr_spearman"]),
            }
        )
        pole_rows.extend(
            cast("list[dict[str, object]]", audit["objects"])
        )
        artifacts[str(path)] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    rr_rows = [
        row
        for row in pole_rows
        if _number(row, "target") == 2
        and math.isfinite(_number(row, "attributed_period_days"))
        and math.isfinite(_number(row, "lomb_scargle_period_days"))
    ]
    harmonic_errors = [
        min(
            abs(
                math.log(
                    _number(row, "attributed_period_days")
                    / (_number(row, "lomb_scargle_period_days") * harmonic)
                )
            )
            for harmonic in (0.5, 1.0, 2.0)
        )
        for row in rr_rows
    ]
    significant_positive = sum(
        math.isfinite(_number(row, "statistic"))
        and _number(row, "statistic") > 0.0
        and _number(row, "pvalue") < 0.05
        for row in correlations
    )

    curves_path = root / "phase2-curves.json"
    curves = cast("dict[str, list[dict[str, object]]]", json.loads(curves_path.read_text()))
    artifacts[str(curves_path)] = {
        "bytes": curves_path.stat().st_size,
        "sha256": _sha256(curves_path),
    }
    gap_path = root / "gap-quadrature.json"
    gap = cast("dict[str, object]", json.loads(gap_path.read_text()))
    gap_rows = cast("list[dict[str, float]]", gap["rows"])
    artifacts[str(gap_path)] = {
        "bytes": gap_path.stat().st_size,
        "sha256": _sha256(gap_path),
    }

    intervention_summary: list[dict[str, object]] = []
    for path in sorted(root.glob("pole-interventions-seed*.json")):
        row = cast("dict[str, object]", json.loads(path.read_text()))
        intervention_summary.append(
            {
                "seed": row["seed"],
                "full_test_balanced_accuracy": row["full_test_balanced_accuracy"],
                "train_test_importance_rank_correlation": (
                    row["train_test_importance_rank_correlation"]
                ),
                "max_logit_reconstruction_residual": (
                    row["max_logit_reconstruction_residual"]
                ),
                "max_margin_decomposition_residual": (
                    row["max_margin_decomposition_residual"]
                ),
            }
        )
        artifacts[str(path)] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }

    ablation_root = root / "ablations"
    phase0_path = root / "phase0-summary.json"
    phase0 = cast("dict[str, object]", json.loads(phase0_path.read_text()))
    artifacts[str(phase0_path)] = {
        "bytes": phase0_path.stat().st_size,
        "sha256": _sha256(phase0_path),
    }
    for directory in ("a2-unit", "a3-token", "a3-energy"):
        for path in sorted((ablation_root / directory).iterdir()):
            if path.is_file():
                artifacts[str(path)] = {
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
    phase0_metrics = cast("dict[str, object]", phase0["metrics"])
    alphabet_metrics = cast("dict[str, object]", phase0_metrics["alphabet"])
    alphabet_test = cast("dict[str, object]", alphabet_metrics["test"])
    physical_accuracy = cast(
        "dict[str, float]",
        alphabet_test["balanced_accuracy"],
    )
    readout_controls = {
        "actual_intervals_physical_lag": physical_accuracy,
        "unit_intervals_physical_lag": _balanced_accuracy(ablation_root / "a2-unit"),
        "actual_intervals_token_lag": _balanced_accuracy(ablation_root / "a3-token"),
        "actual_intervals_energy_only": _balanced_accuracy(ablation_root / "a3-energy"),
    }
    payload = {
        "schema": "lnet.astronomy.mechanism_summary.v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "g4_pole_period_tracking": {
            "decision": "FAIL",
            "criterion": "at least 3/5 seeds with positive Spearman and p < 0.05",
            "significant_positive_seeds": significant_positive,
            "per_seed_rr_spearman": correlations,
            "harmonic_aware": {
                "allowed_multipliers": [0.5, 1.0, 2.0],
                "rr_object_seed_pairs": len(harmonic_errors),
                "within_10_percent": (
                    sum(error <= math.log(1.1) for error in harmonic_errors)
                    / len(harmonic_errors)
                ),
                "within_20_percent": (
                    sum(error <= math.log(1.2) for error in harmonic_errors)
                    / len(harmonic_errors)
                ),
                "median_absolute_log_ratio": statistics.median(harmonic_errors),
            },
            "scientific_claim_allowed": False,
        },
        "readout_and_time_controls": readout_controls,
        "early_classification": _group_curve_metrics(
            curves["early_classification"],
            "days",
        ),
        "seasonal_gap": _group_curve_metrics(curves["seasonal_gap"], "gap_days"),
        "long_gap_quadrature": {
            "maximum_endpoint_energy_relative_error": max(
                abs(row["endpoint_energy_relative_error"]) for row in gap_rows
            ),
            "maximum_interpolated_previous_log10_ratio": max(
                abs(row["interpolated_previous_log10_ratio"]) for row in gap_rows
            ),
            "interpretation": (
                "The fixed physical-time interpolation readout is not an exact "
                "continuous-time moment over long gaps."
            ),
        },
        "causal_pole_interventions": intervention_summary,
        "phase3_extensions_authorized": False,
        "artifacts": artifacts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")


if __name__ == "__main__":
    main()
