from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import cast

SEEDS = (7, 11, 19, 23, 31)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, object]:
    return cast("dict[str, object]", json.loads(path.read_text()))


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _neural_validation(directory: Path, model: str) -> dict[str, object]:
    rows = [_read(directory / f"{model}-seed{seed}.json") for seed in SEEDS]
    metric_names = (
        "weighted_log_loss",
        "balanced_accuracy",
        "macro_f1",
        "expected_calibration_error",
    )
    return {
        "objects": 784,
        "seeds": SEEDS,
        "parameter_count": sorted(
            {int(cast("int", row["parameter_count"])) for row in rows}
        ),
        "checkpoint_selection_metric": sorted(
            {str(row.get("checkpoint_selection_metric", "legacy mean CE")) for row in rows}
        ),
        "metrics": {
            metric: _summary(
                [
                    float(cast("dict[str, float]", row["test"])[metric])
                    for row in rows
                ]
            )
            for metric in metric_names
        },
    }


def _rf_validation(directory: Path) -> dict[str, object]:
    rows = cast(
        "list[dict[str, float | int]]",
        json.loads((directory / "validation.json").read_text()),
    )
    return {
        "objects": 784,
        "seeds": [int(row["seed"]) for row in rows],
        "metrics": {
            metric: _summary([float(row[metric]) for row in rows])
            for metric in (
                "weighted_log_loss",
                "balanced_accuracy",
                "macro_f1",
                "expected_calibration_error",
            )
            if all(metric in row for row in rows)
        },
    }


def _official(directory: Path) -> dict[str, object] | None:
    path = directory / "official-test-known14.json"
    return _read(path) if path.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-dir", type=Path, required=True)
    parser.add_argument("--gru-dir", type=Path, required=True)
    parser.add_argument("--rf-dir", type=Path, required=True)
    parser.add_argument("--supernnova", type=Path, required=True)
    parser.add_argument("--parsnip", type=Path, required=True)
    parser.add_argument("--throughput", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    token_validation = _neural_validation(args.token_dir, "alphabet")
    gru_validation = _neural_validation(args.gru_dir, "gru")
    rf_validation = _rf_validation(args.rf_dir)
    supernnova = _read(args.supernnova)
    parsnip = _read(args.parsnip)

    payload = {
        "schema": "lnet.astronomy.phase1_summary.v3",
        "selection_protocol": {
            "split": "fixed class-stratified 7064/784 train/validation",
            "split_seed": 20260729,
            "checkpoint_metric": "known-14 weighted log-loss",
            "official_metric": "weighted_log_loss_known14",
            "exact_15_class_competition_metric": False,
        },
        "validation": {
            "alphabet_token": token_validation,
            "delta_time_gru": gru_validation,
            "compact_statistical_rf": rf_validation,
            "supernnova": supernnova,
            "parsnip_subset_only": parsnip,
        },
        "official_known14": {
            "alphabet_token": _official(args.token_dir),
            "delta_time_gru": _official(args.gru_dir),
            "compact_statistical_rf": _official(args.rf_dir),
        },
        "preregistered_decisions": {
            "C1": {
                "status": "NOT_ADJUDICABLE",
                "reason": "The durable preregistration does not preserve a numerical C1 threshold.",
            },
            "C2": {
                "status": "NOT_ADJUDICABLE",
                "reason": "The durable preregistration does not preserve a numerical C2 threshold.",
            },
            "C3": {
                "status": "NOT_ADJUDICABLE",
                "reason": "The durable preregistration does not preserve a numerical C3 threshold.",
            },
        },
        "external_artifacts": {
            "supernnova": {
                "path": str(args.supernnova),
                "sha256": _sha256(args.supernnova),
                "execution_host": "local_gpu",
            },
            "parsnip": {
                "path": str(args.parsnip),
                "sha256": _sha256(args.parsnip),
                "execution_host": "local_gpu",
            },
            "throughput": {
                "path": str(args.throughput),
                "sha256": _sha256(args.throughput),
                "execution_host": "secondary_gpu",
            },
        },
        "limitations": [
            "Official scores exclude true unknown targets 991-994 and class 99.",
            "The compact RF is not a faithful ALeRCE or Avocado reproduction.",
            "ParSNIP covers only 515 supernova-like validation objects.",
            "SuperNNova uses its native unweighted training loss.",
            "Throughput measures in-memory batched model forward, not Kafka or batch-1 latency.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
