"""UCR-16 screen for exact-ZOH and strict-past fully lag-free resolvents."""

# pyright: reportPrivateUsage=false, reportArgumentType=false
# ruff: noqa: SLF001, T201

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import TYPE_CHECKING, cast

from scipy.stats import t as student_t
from scipy.stats import ttest_1samp

from lnet.pac_final_validation import UCR_SECONDS
from lnet.pac_metrics import count_parameters
from optimization.learned_two_tap_strict_resolvent import (
    ExactZOHResolventALPHABET,
    LagReaderALPHABET,
    StrictPastResolventALPHABET,
)
from scripts import pac_pole_attention_ucr16_fast as base

if TYPE_CHECKING:
    from lnet.pac_types import PACDevice

ROOT = Path(".omx/results/pac-strict-resolvent-ucr16-fast-20260720")
LAG_REFERENCE_ROOT = Path(".omx/results/pac-resolvent-reader-ucr16-fast-20260720")
VARIANTS = ("exact_zoh", "strict_past")


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    paths = (
        Path("optimization/learned_two_tap_alphabet.py"),
        Path("optimization/learned_two_tap_resolvent_reader.py"),
        Path("optimization/learned_two_tap_strict_resolvent.py"),
        Path("optimization/masked_modal_moments.py"),
        Path("scripts/pac_pole_attention_ucr16_fast.py"),
        Path("scripts/pac_strict_resolvent_ucr16_fast.py"),
    )
    return {str(path): hashlib.sha256((root / path).read_bytes()).hexdigest() for path in paths}


def _design() -> dict[str, object]:
    baseline = LagReaderALPHABET(1, base.MODEL_DIM, base.MODES, 5)
    exact = ExactZOHResolventALPHABET(1, base.MODEL_DIM, base.MODES, 5)
    strict = StrictPastResolventALPHABET(1, base.MODEL_DIM, base.MODES, 5)
    parameter_counts = {
        "lag_reader": count_parameters(baseline),
        "exact_zoh": count_parameters(exact),
        "strict_past": count_parameters(strict),
    }
    return {
        "schema": "pac_strict_resolvent_ucr16_fast_contract.v1",
        "purpose": "decisive validation-only fully lag-free resolvent control",
        "official_test_accessed": False,
        "datasets": list(base.DATASETS),
        "excluded_slow_datasets": ["FordA", "FordB"],
        "seeds": list(base.SEEDS),
        "trained_variants": list(VARIANTS),
        "lag_reference_root": str(LAG_REFERENCE_ROOT),
        "model_dim": base.MODEL_DIM,
        "modes": base.MODES,
        "epochs": base.EPOCHS,
        "batch_size": base.BATCH_SIZE,
        "learning_rate": base.LEARNING_RATE,
        "weight_decay": base.WEIGHT_DECAY,
        "grad_clip_norm": base.GRAD_CLIP_NORM,
        "hyperparameter_tuning": False,
        "controlled_difference": (
            "writer and reader lag-1/lag-4 moments are both replaced; no mode attention, "
            "last-state query, or new parameters are introduced"
        ),
        "variants": {
            "exact_zoh": (
                "exact augmented ZOH for z'=lambda*z+u, p'=lambda*p+alpha*z, "
                "q'=lambda*q+alpha*p"
            ),
            "strict_past": (
                "p_n is driven only by z_(n-1), and q_n only by p_(n-1), "
                "using the same alpha-scaled pole gain"
            ),
        },
        "descriptor": (
            "per-mode [log1p(E_z), Re/Im NCorr(z,p), Re/Im NCorr(z,q)], "
            "flattened to 5M for both writer and reader"
        ),
        "parameter_control": parameter_counts,
        "source_sha256": _source_hashes(),
    }


def design_sha256() -> str:
    encoded = json.dumps(_design(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def jobs() -> list[base.Job]:
    digest = design_sha256()
    return [
        base.Job(
            key=f"strict_resolvent_ucr16_fast:{dataset}:{variant}:seed{seed}",
            dataset=dataset,
            variant=variant,
            split_seed=seed,
            train_seed=seed,
            model_dim=base.MODEL_DIM,
            modes=base.MODES,
            heads=0,
            epochs=base.EPOCHS,
            batch_size=base.BATCH_SIZE,
            learning_rate=base.LEARNING_RATE,
            weight_decay=base.WEIGHT_DECAY,
            grad_clip_norm=base.GRAD_CLIP_NORM,
            evaluation_split="validation",
            estimated_seconds=UCR_SECONDS[dataset],
            design_sha256=digest,
        )
        for dataset in base.DATASETS
        for seed in base.SEEDS
        for variant in VARIANTS
    ]


def _build_model(job: base.Job, input_dim: int, output_dim: int):  # noqa: ANN202
    if job.variant == "exact_zoh":
        return ExactZOHResolventALPHABET(input_dim, job.model_dim, job.modes, output_dim)
    return StrictPastResolventALPHABET(input_dim, job.model_dim, job.modes, output_dim)


def _load_rows(root: Path) -> list[dict[str, object]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((root / "completed").glob("*.json"))
    ]


def _lag_reference_rows() -> list[dict[str, object]]:
    contract = json.loads((LAG_REFERENCE_ROOT / "contract.json").read_text(encoding="utf-8"))
    current_base_hash = _source_hashes()["optimization/learned_two_tap_alphabet.py"]
    recorded_base_hash = contract["source_sha256"]["optimization/learned_two_tap_alphabet.py"]
    if current_base_hash != recorded_base_hash:
        message = "lag reference used a different learned-two-tap source"
        raise RuntimeError(message)
    rows = [
        row
        for row in _load_rows(LAG_REFERENCE_ROOT)
        if row.get("variant") == "lag_reader"
    ]
    if len(rows) != len(base.DATASETS) * len(base.SEEDS):
        message = "lag reference is incomplete"
        raise RuntimeError(message)
    if any(
        row.get("official_test_accessed") is not False
        or row.get("test_evaluated") is not False
        or row.get("design_sha256") != contract["design_sha256"]
        for row in rows
    ):
        message = "lag reference failed contamination or source validation"
        raise RuntimeError(message)
    return rows


def _average_ranks(scores: dict[str, float]) -> dict[str, float]:
    ordered = sorted(scores, key=scores.__getitem__, reverse=True)
    ranks: dict[str, float] = {}
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and math.isclose(
            scores[ordered[index]], scores[ordered[end]], rel_tol=1.0e-5, abs_tol=1.0e-8
        ):
            end += 1
        average_rank = (index + 1 + end) / 2.0
        for variant in ordered[index:end]:
            ranks[variant] = average_rank
        index = end
    return ranks


def _paired_summary(deltas: list[float]) -> dict[str, object]:
    average = mean(deltas)
    sem = stdev(deltas) / math.sqrt(len(deltas))
    critical = float(student_t.ppf(0.975, len(deltas) - 1))
    test = ttest_1samp(deltas, popmean=0.0)
    return {
        "candidate_minus_lag_mean": average,
        "ci95": [average - critical * sem, average + critical * sem],
        "two_sided_t_pvalue": float(cast("float", test[1])),
        "wins_ties_losses": {
            "wins": sum(delta > 1.0e-8 for delta in deltas),
            "ties": sum(abs(delta) <= 1.0e-8 for delta in deltas),
            "losses": sum(delta < -1.0e-8 for delta in deltas),
        },
    }


def report(root: Path) -> dict[str, object]:
    campaign_status = base.status(root)
    if campaign_status["done"] is not True:
        message = f"refusing to report incomplete results: {campaign_status}"
        raise RuntimeError(message)
    candidate_rows = _load_rows(root)
    lag_rows = _lag_reference_rows()
    rows = lag_rows + candidate_rows
    variants = ("lag_reader", *VARIANTS)
    rank_sums = dict.fromkeys(variants, 0.0)
    top_counts = dict.fromkeys(variants, 0)
    dataset_deltas = {variant: [] for variant in VARIANTS}
    seed_deltas = {variant: [] for variant in VARIANTS}
    datasets: list[dict[str, object]] = []
    for dataset in base.DATASETS:
        scores: dict[str, float] = {}
        sample_sds: dict[str, float] = {}
        seed_values: dict[str, list[float]] = {}
        for variant in variants:
            values = [
                float(row["validation_balanced_accuracy"])
                for row in rows
                if row["dataset"] == dataset and row["variant"] == variant
            ]
            if len(values) != len(base.SEEDS):
                message = f"incomplete cell: {dataset}/{variant}"
                raise RuntimeError(message)
            seed_values[variant] = values
            scores[variant] = mean(values)
            sample_sds[variant] = stdev(values)
        ranks = _average_ranks(scores)
        best = max(scores.values())
        for variant in variants:
            rank_sums[variant] += ranks[variant]
            top_counts[variant] += int(
                math.isclose(scores[variant], best, rel_tol=1.0e-5, abs_tol=1.0e-8)
            )
        deltas = {variant: scores[variant] - scores["lag_reader"] for variant in VARIANTS}
        for variant in VARIANTS:
            dataset_deltas[variant].append(deltas[variant])
            seed_deltas[variant].extend(
                candidate - lag
                for candidate, lag in zip(
                    seed_values[variant], seed_values["lag_reader"], strict=True
                )
            )
        datasets.append(
            {
                "dataset": dataset,
                "means": scores,
                "sample_sds": sample_sds,
                "ranks": ranks,
                "candidate_minus_lag": deltas,
            }
        )
    aggregate: dict[str, object] = {}
    for variant in variants:
        variant_rows = [row for row in rows if row["variant"] == variant]
        aggregate[variant] = {
            "row_mean_balanced_accuracy": mean(
                float(row["validation_balanced_accuracy"]) for row in variant_rows
            ),
            "mean_rank": rank_sums[variant] / len(base.DATASETS),
            "joint_top1": top_counts[variant],
            "params_trainable": sorted({int(row["params_trainable"]) for row in variant_rows}),
            "mean_train_seconds": mean(float(row["train_seconds"]) for row in variant_rows),
        }
    payload: dict[str, object] = {
        "schema": "pac_strict_resolvent_ucr16_fast_report.v1",
        "status": campaign_status,
        "official_test_accessed": False,
        "trained_rows": len(candidate_rows),
        "reused_lag_reference_rows": len(lag_rows),
        "aggregate": aggregate,
        "dataset_level_paired_inference": {
            variant: _paired_summary(dataset_deltas[variant]) for variant in VARIANTS
        },
        "seed_pair_wins_ties_losses": {
            variant: {
                "wins": sum(delta > 1.0e-8 for delta in seed_deltas[variant]),
                "ties": sum(abs(delta) <= 1.0e-8 for delta in seed_deltas[variant]),
                "losses": sum(delta < -1.0e-8 for delta in seed_deltas[variant]),
            }
            for variant in VARIANTS
        },
        "datasets": datasets,
    }
    base._atomic_json(root / "reports/summary.json", payload)
    return payload


def _configure_harness() -> None:
    base._design = _design
    base.design_sha256 = design_sha256
    base.jobs = jobs
    base._build_model = _build_model


def main() -> None:
    _configure_harness()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("enqueue", "worker", "status", "report"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--data-root", type=Path, default=base.DATA_ROOT)
    args = parser.parse_args()
    if args.command == "enqueue":
        payload = base.enqueue(args.root, args.workers)
    elif args.command == "worker":
        if args.manifest is None:
            parser.error("worker requires --manifest")
        base.run_manifest(
            args.root,
            args.manifest,
            device=cast("PACDevice", args.device),
            data_root=args.data_root,
        )
        payload = base.status(args.root)
    elif args.command == "report":
        payload = report(args.root)
    else:
        payload = base.status(args.root)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
