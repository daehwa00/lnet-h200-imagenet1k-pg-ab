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
MODELS = ("alphabet", "gru")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metric_summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runs: dict[str, list[dict[str, object]]] = {}
    artifacts: dict[str, dict[str, object]] = {}
    finite_training = True
    convergent_training = True
    for model in MODELS:
        model_runs: list[dict[str, object]] = []
        for seed in SEEDS:
            result_path = args.results_dir / f"{model}-seed{seed}.json"
            checkpoint_path = args.results_dir / f"{model}-seed{seed}.pt"
            result = cast("dict[str, object]", json.loads(result_path.read_text()))
            history = cast("list[dict[str, float | int]]", result["history"])
            finite_training = finite_training and all(
                math.isfinite(float(value))
                for row in history
                for key, value in row.items()
                if key != "epoch"
            )
            validation_losses = [
                float(row["validation_loss"]) for row in history
            ]
            convergent_training = convergent_training and (
                min(validation_losses) < validation_losses[0]
            )
            model_runs.append(result)
            artifacts[result_path.name] = {
                "bytes": result_path.stat().st_size,
                "sha256": _sha256(result_path),
            }
            artifacts[checkpoint_path.name] = {
                "bytes": checkpoint_path.stat().st_size,
                "sha256": _sha256(checkpoint_path),
            }
        runs[model] = model_runs

    metrics: dict[str, object] = {}
    for model, model_runs in runs.items():
        metrics[model] = {
            "seeds": SEEDS,
            "parameter_count": sorted(
                {int(cast("int", result["parameter_count"])) for result in model_runs}
            ),
            "best_epochs": [int(cast("int", result["best_epoch"])) for result in model_runs],
            "test": {
                metric: _metric_summary(
                    [
                        float(cast("dict[str, float]", result["test"])[metric])
                        for result in model_runs
                    ]
                )
                for metric in (
                    "loss",
                    "weighted_log_loss",
                    "balanced_accuracy",
                    "macro_f1",
                    "expected_calibration_error",
                )
                if all(
                    metric in cast("dict[str, float]", result["test"])
                    for result in model_runs
                )
            },
        }

    alphabet_accuracy = statistics.fmean(
        float(cast("dict[str, float]", result["test"])["balanced_accuracy"])
        for result in runs["alphabet"]
    )
    gru_accuracy = statistics.fmean(
        float(cast("dict[str, float]", result["test"])["balanced_accuracy"])
        for result in runs["gru"]
    )
    gate_decisions: dict[str, dict[str, object]] = {
        "G1_finite_convergent_training": {
            "threshold": (
                "all recorded metrics finite and each run improves validation loss"
            ),
            "finite": finite_training,
            "improved": convergent_training,
            "pass": finite_training and convergent_training,
        },
        "G2_alphabet_balanced_accuracy": {
            "threshold": "> 0.85 five-seed mean",
            "value": alphabet_accuracy,
            "pass": alphabet_accuracy > 0.85,
        },
        "G3_alphabet_not_worse_than_gru": {
            "threshold": "ALPHABET mean >= GRU mean - 0.03",
            "alphabet": alphabet_accuracy,
            "gru": gru_accuracy,
            "gap_percentage_points": 100.0 * (alphabet_accuracy - gru_accuracy),
            "pass": alphabet_accuracy >= gru_accuracy - 0.03,
        },
    }
    pole_results = [
        cast(
            "dict[str, object]",
            json.loads(
                (args.results_dir / f"pole-audit-seed{seed}.json").read_text()
            ),
        )
        for seed in SEEDS
    ]
    pole_correlations = [
        cast("dict[str, float]", result["rr_spearman"])
        for result in pole_results
    ]
    significant_positive = sum(
        math.isfinite(row["statistic"])
        and row["statistic"] > 0.0
        and row["pvalue"] < 0.05
        for row in pole_correlations
    )
    gate_decisions["G4_period_pole_tracks_lomb_scargle"] = {
        "threshold": "positive significant object-level correlation in a majority of seeds",
        "significant_positive_seeds": significant_positive,
        "statistics": pole_correlations,
        "pass": significant_positive >= 3,
    }
    split_manifest = args.results_dir / "split-manifest.json"
    artifacts[split_manifest.name] = {
        "bytes": split_manifest.stat().st_size,
        "sha256": _sha256(split_manifest),
    }
    payload = {
        "schema": "lnet.astronomy.phase0_summary.v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "results_dir": str(args.results_dir),
        "preregistered_configuration": {
            "seeds": SEEDS,
            "models": MODELS,
            "model_dim": 64,
            "modes": 16,
            "epochs_max": 50,
            "split_seed": 20260729,
            "time_mode": "actual",
            "lag_mode": "physical",
        },
        "metrics": metrics,
        "gate_decisions": gate_decisions,
        "learning_go_G1_to_G3": all(
            bool(gate_decisions[key]["pass"])
            for key in (
                "G1_finite_convergent_training",
                "G2_alphabet_balanced_accuracy",
                "G3_alphabet_not_worse_than_gru",
            )
        ),
        "all_G1_to_G4_pass": all(
            bool(gate["pass"]) for gate in gate_decisions.values()
        ),
        "artifacts": artifacts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
