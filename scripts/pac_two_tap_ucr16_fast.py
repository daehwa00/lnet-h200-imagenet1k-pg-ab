# ruff: noqa: C901, EM101, EM102, T201, TRY003
"""Restart-safe UCR-16 screen of fixed versus learned two-tap ALPHABET input maps."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from statistics import mean, stdev
from typing import TYPE_CHECKING, Final, cast

from scipy.stats import t as student_t
from scipy.stats import ttest_1samp

from lnet.pac_compact_h_only_ablation import (
    CompactAblationJob,
    CompactAblationVariant,
    run_manifest,
)
from lnet.pac_final_validation import UCR_SECONDS

if TYPE_CHECKING:
    from lnet.pac_types import PACDevice

ROOT: Final = Path(".omx/results/pac-two-tap-ucr16-fast-20260720")
DATASETS: Final = (
    "ArrowHead",
    "CinCECGTorso",
    "CricketX",
    "ECG200",
    "ECG5000",
    "ECGFiveDays",
    "Earthquakes",
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
SEEDS: Final = (7, 11, 19, 23, 31)
VARIANTS: Final[tuple[CompactAblationVariant, ...]] = (
    "full",
    "unconstrained_two_tap",
)
MODEL_DIM: Final = 32
MODES: Final = 16
EPOCHS: Final = 100
BATCH_SIZE: Final = 64
LEARNING_RATE: Final = 0.003
WEIGHT_DECAY: Final = 0.0001
GRAD_CLIP_NORM: Final = 1.0


def _design() -> dict[str, object]:
    return {
        "schema": "pac_two_tap_ucr16_fast_contract.v1",
        "purpose": "fixed-D/M candidate screen before any public architecture replacement",
        "endpoint": "official-TRAIN-derived validation balanced accuracy",
        "official_test_accessed": False,
        "datasets": list(DATASETS),
        "excluded_slow_datasets": ["FordA", "FordB"],
        "seeds": list(SEEDS),
        "variants": list(VARIANTS),
        "model_dim": MODEL_DIM,
        "modes": MODES,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip_norm": GRAD_CLIP_NORM,
        "restore_best_validation": True,
        "hyperparameter_tuning": False,
        "jobs": len(DATASETS) * len(SEEDS) * len(VARIANTS),
        "controlled_difference": (
            "the canonical degree-normalized complementary path-edge map and "
            "semi-orthogonal joint projection are replaced by an equal-size learned "
            "overlapping Conv1d kernel of width two; writer, reader, head, D, M, and "
            "training recipe remain unchanged"
        ),
        "selection_rule": (
            "prefer learned two-tap only if it improves mean dataset balanced accuracy "
            "and mean rank without a concentrated catastrophic regression"
        ),
        "restart_safe": True,
    }


def design_sha256() -> str:
    encoded = json.dumps(_design(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def jobs() -> list[CompactAblationJob]:
    design_hash = design_sha256()
    return [
        CompactAblationJob(
            key=f"two_tap_ucr16_fast:{dataset}:{variant}:seed{seed}",
            dataset=dataset,
            variant=variant,
            split_seed=seed,
            train_seed=seed,
            model_dim=MODEL_DIM,
            modes=MODES,
            trial=1,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            grad_clip_norm=GRAD_CLIP_NORM,
            width_tier=0,
            selected_config_key="fixed-d32-m16-no-tuning",
            selection_sha256=design_hash,
            evaluation_split="validation",
            estimated_seconds=UCR_SECONDS[dataset],
        )
        for dataset in DATASETS
        for seed in SEEDS
        for variant in VARIANTS
    ]


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _result_keys(directory: Path) -> set[str]:
    if not directory.exists():
        return set()
    return {
        str(json.loads(path.read_text(encoding="utf-8"))["job_key"])
        for path in directory.glob("*.json")
    }


def enqueue(root: Path, workers: int) -> dict[str, object]:
    if not 1 <= workers <= 32:
        raise ValueError("workers must be between 1 and 32")
    expected = jobs()
    completed = _result_keys(root / "completed")
    pending = [job for job in expected if job.key not in completed]
    shards: list[list[CompactAblationJob]] = [[] for _ in range(workers)]
    loads = [0.0] * workers
    for job in sorted(pending, key=lambda item: item.estimated_seconds, reverse=True):
        index = min(range(workers), key=loads.__getitem__)
        shards[index].append(job)
        loads[index] += job.estimated_seconds
    manifests = root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    for stale in manifests.glob("worker-*.jsonl"):
        stale.unlink()
    for index, shard in enumerate(shards):
        (manifests / f"worker-{index:02d}.jsonl").write_text(
            "".join(json.dumps(asdict(job), sort_keys=True) + "\n" for job in shard),
            encoding="utf-8",
        )
    contract = {**_design(), "design_sha256": design_sha256(), "workers": workers}
    _atomic_json(root / "contract.json", contract)
    return {
        "jobs": len(expected),
        "pending": len(pending),
        "workers": workers,
        "estimated_worker_seconds": loads,
    }


def status(root: Path) -> dict[str, object]:
    expected = {job.key for job in jobs()}
    completed = _result_keys(root / "completed")
    failed = _result_keys(root / "failed") - completed
    return {
        "expected": len(expected),
        "completed": len(expected & completed),
        "failed": len(expected & failed),
        "remaining": len(expected - completed - failed),
        "unexpected": len((completed | failed) - expected),
        "done": expected == completed and not failed and not (completed - expected),
    }


def _average_ranks(scores: dict[str, float]) -> dict[str, float]:
    first, second = VARIANTS
    if math.isclose(scores[first], scores[second], rel_tol=1.0e-5, abs_tol=1.0e-8):
        return {first: 1.5, second: 1.5}
    winner = max(VARIANTS, key=scores.__getitem__)
    loser = min(VARIANTS, key=scores.__getitem__)
    return {winner: 1.0, loser: 2.0}


def report(root: Path) -> dict[str, object]:
    campaign_status = status(root)
    if campaign_status["done"] is not True:
        raise RuntimeError(f"refusing to report an incomplete screen: {campaign_status}")
    expected = {job.key: job for job in jobs()}
    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((root / "completed").glob("*.json"))
    ]
    if len(rows) != len(expected):
        raise RuntimeError("completed row count disagrees with the sealed screen")
    for row in rows:
        job = expected.get(str(row.get("job_key")))
        if (
            job is None
            or row.get("schema") != "pac_compact_h_only_ablation_result.v1"
            or row.get("official_test_accessed") is not False
            or row.get("test_evaluated") is not False
            or row.get("test_count") != 0
            or row.get("selection_sha256") != design_sha256()
            or int(row.get("model_dim", -1)) != MODEL_DIM
            or int(row.get("modes", -1)) != MODES
        ):
            raise RuntimeError(f"invalid or contaminated result: {row.get('job_key')}")

    dataset_rows: list[dict[str, object]] = []
    rank_sums = dict.fromkeys(VARIANTS, 0.0)
    top_counts = dict.fromkeys(VARIANTS, 0)
    deltas: list[float] = []
    for dataset in DATASETS:
        scores: dict[str, float] = {}
        sample_sds: dict[str, float] = {}
        for variant in VARIANTS:
            values = [
                float(row["validation_balanced_accuracy"])
                for row in rows
                if row["dataset"] == dataset and row["variant"] == variant
            ]
            if len(values) != len(SEEDS):
                raise RuntimeError(f"incomplete dataset/variant cell: {dataset}/{variant}")
            scores[variant] = mean(values)
            sample_sds[variant] = stdev(values)
        ranks = _average_ranks(scores)
        for variant in VARIANTS:
            rank_sums[variant] += ranks[variant]
            if ranks[variant] == 1.0 or ranks[variant] == 1.5:
                top_counts[variant] += 1
        delta = scores["unconstrained_two_tap"] - scores["full"]
        deltas.append(delta)
        dataset_rows.append(
            {
                "dataset": dataset,
                "means": scores,
                "sample_sds": sample_sds,
                "ranks": ranks,
                "two_tap_minus_full": delta,
            }
        )

    aggregate: dict[str, object] = {}
    for variant in VARIANTS:
        values = [
            float(row["validation_balanced_accuracy"]) for row in rows if row["variant"] == variant
        ]
        aggregate[variant] = {
            "row_mean_balanced_accuracy": mean(values),
            "row_sample_sd_balanced_accuracy": stdev(values),
            "mean_rank": rank_sums[variant] / len(DATASETS),
            "joint_top1": top_counts[variant],
            "mean_params_trainable": mean(
                int(row["params_trainable"]) for row in rows if row["variant"] == variant
            ),
        }
    paired = ttest_1samp(deltas, popmean=0.0)
    sem = stdev(deltas) / math.sqrt(len(deltas))
    critical = float(student_t.ppf(0.975, len(deltas) - 1))
    inference = {
        "unit": "dataset-level five-seed mean",
        "two_tap_minus_full_mean": mean(deltas),
        "sample_sd": stdev(deltas),
        "ci95": [mean(deltas) - critical * sem, mean(deltas) + critical * sem],
        "two_sided_t_pvalue": float(paired.pvalue),
        "wins_ties_losses": {
            "wins": sum(delta > 1.0e-8 for delta in deltas),
            "ties": sum(abs(delta) <= 1.0e-8 for delta in deltas),
            "losses": sum(delta < -1.0e-8 for delta in deltas),
        },
    }
    payload = {
        "schema": "pac_two_tap_ucr16_fast_report.v1",
        "status": campaign_status,
        "contract_sha256": hashlib.sha256((root / "contract.json").read_bytes()).hexdigest(),
        "official_test_accessed": False,
        "rows": len(rows),
        "aggregate": aggregate,
        "paired_inference": inference,
        "datasets": dataset_rows,
    }
    _atomic_json(root / "reports/summary.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("enqueue", "worker", "status", "report"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--data-root", type=Path, default=Path(".omx/data/ucr"))
    args = parser.parse_args()
    if args.command == "enqueue":
        payload = enqueue(args.root, args.workers)
    elif args.command == "worker":
        if args.manifest is None:
            parser.error("worker requires --manifest")
        run_manifest(
            args.root,
            args.manifest,
            device=cast("PACDevice", args.device),
            data_root=args.data_root,
        )
        payload = status(args.root)
    elif args.command == "status":
        payload = status(args.root)
    else:
        payload = report(args.root)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
