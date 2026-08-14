"""Fast UCR-16 validation screen for the V5 versus V5.1 input stem."""

# pyright: reportPrivateUsage=false
# ruff: noqa: EM101, EM102, SLF001, TRY003

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import cast

from scipy.stats import t as student_t
from scipy.stats import ttest_1samp

from lnet.pac_final_validation import UCR_SECONDS
from lnet.pac_metrics import count_parameters
from optimization.direct_stem_three_stage_log_signature import (
    FullyPoleNativeDirectStemThreeStageALPHABET,
)
from optimization.three_stage_causal_log_signature import (
    FullyPoleNativeThreeStageCausalALPHABET,
)
from scripts import pac_pole_attention_ucr16_fast as base

ROOT = Path(".omx/results/pac-logsig-v51-ucr16-fast-20260721")
VARIANTS = ("v5_pointwise_dwconv", "v51_direct_conv")


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    paths = (
        Path("optimization/direct_stem_three_stage_log_signature.py"),
        Path("optimization/three_stage_causal_log_signature.py"),
        Path("optimization/pointwise_causal_log_signature.py"),
        Path("optimization/learned_two_tap_log_signature.py"),
        Path("optimization/learned_two_tap_alphabet.py"),
        Path("optimization/masked_modal_moments.py"),
        Path("optimization/stage2_recurrence.py"),
        Path("optimization/stage2_tail_metadata.py"),
        Path("src/lnet/pac_recurrence.py"),
        Path("src/lnet/pac_triton_log_signature.py"),
        Path("src/lnet/pac_triton_log_signature_training.py"),
        Path("scripts/pac_pole_attention_ucr16_fast.py"),
        Path("scripts/pac_logsig_v51_ucr16_fast.py"),
    )
    return {str(path): hashlib.sha256((root / path).read_bytes()).hexdigest() for path in paths}


def _design() -> dict[str, object]:
    baseline = FullyPoleNativeThreeStageCausalALPHABET(1, base.MODEL_DIM, base.MODES, 5)
    candidate = FullyPoleNativeDirectStemThreeStageALPHABET(
        1,
        base.MODEL_DIM,
        base.MODES,
        5,
    )
    return {
        "schema": "pac_logsig_v51_ucr16_fast_contract.v1",
        "purpose": "validation-only architecture screen; no automatic Q1 promotion",
        "official_test_accessed": False,
        "datasets": list(base.DATASETS),
        "excluded_slow_datasets": ["FordA", "FordB"],
        "seeds": list(base.SEEDS),
        "variants": {
            VARIANTS[0]: "V5 pointwise projection followed by causal depthwise convolution",
            VARIANTS[1]: "V5.1 direct causal Conv1d from raw input to model width",
        },
        "model_dim": base.MODEL_DIM,
        "modes": base.MODES,
        "epochs": base.EPOCHS,
        "batch_size": base.BATCH_SIZE,
        "learning_rate": base.LEARNING_RATE,
        "weight_decay": base.WEIGHT_DECAY,
        "grad_clip_norm": base.GRAD_CLIP_NORM,
        "hyperparameter_tuning": False,
        "automatic_q1_promotion": False,
        "controlled_difference": (
            "only the first local lift changes: V5 factorizes raw-to-D pointwise mixing "
            "and a k=5,d=1 depthwise convolution, whereas V5.1 uses one direct "
            "raw-to-D causal Conv1d(k=5,d=1); writer, reader, LogSig descriptors, head, "
            "D, M, optimizer, epochs, splits, and seeds are fixed"
        ),
        "parameter_control": {
            VARIANTS[0]: count_parameters(baseline),
            VARIANTS[1]: count_parameters(candidate),
            "candidate_minus_baseline": count_parameters(candidate) - count_parameters(baseline),
        },
        "source_sha256": _source_hashes(),
    }


def design_sha256() -> str:
    encoded = json.dumps(_design(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def jobs() -> list[base.Job]:
    digest = design_sha256()
    return [
        base.Job(
            key=f"logsig_v51_ucr16_fast:{dataset}:{variant}:seed{seed}",
            dataset=dataset,
            variant=cast("base.ReaderVariant", cast("object", variant)),
            split_seed=seed,
            train_seed=seed,
            model_dim=base.MODEL_DIM,
            modes=base.MODES,
            heads=1,
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
    if job.variant == VARIANTS[0]:
        return FullyPoleNativeThreeStageCausalALPHABET(
            input_dim,
            job.model_dim,
            job.modes,
            output_dim,
        )
    if job.variant == VARIANTS[1]:
        return FullyPoleNativeDirectStemThreeStageALPHABET(
            input_dim,
            job.model_dim,
            job.modes,
            output_dim,
        )
    raise ValueError(f"unknown V5.1 fast-screen variant: {job.variant}")


def _ranks(scores: dict[str, float]) -> dict[str, float]:
    if math.isclose(scores[VARIANTS[0]], scores[VARIANTS[1]], rel_tol=1.0e-5, abs_tol=1.0e-8):
        return {VARIANTS[0]: 1.5, VARIANTS[1]: 1.5}
    winner = max(VARIANTS, key=scores.__getitem__)
    return {variant: 1.0 if variant == winner else 2.0 for variant in VARIANTS}


def report(root: Path) -> dict[str, object]:
    campaign_status = base.status(root)
    if campaign_status["done"] is not True:
        raise RuntimeError(f"refusing to report incomplete results: {campaign_status}")
    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((root / "completed").glob("*.json"))
    ]
    expected = {job.key: job for job in jobs()}
    if len(rows) != len(expected):
        raise RuntimeError("completed row count disagrees with the sealed design")
    for row in rows:
        job = expected.get(str(row.get("job_key")))
        if (
            job is None
            or row.get("schema") != "pac_pole_attention_ucr16_fast_result.v1"
            or row.get("official_test_accessed") is not False
            or row.get("test_evaluated") is not False
            or row.get("test_count") != 0
            or row.get("design_sha256") != design_sha256()
        ):
            raise RuntimeError(f"invalid or contaminated row: {row.get('job_key')}")

    rank_sums = dict.fromkeys(VARIANTS, 0.0)
    top_counts = dict.fromkeys(VARIANTS, 0)
    deltas: list[float] = []
    datasets: list[dict[str, object]] = []
    for dataset in base.DATASETS:
        scores: dict[str, float] = {}
        sample_sds: dict[str, float] = {}
        for variant in VARIANTS:
            values = [
                float(row["validation_balanced_accuracy"])
                for row in rows
                if row["dataset"] == dataset and row["variant"] == variant
            ]
            if len(values) != len(base.SEEDS):
                raise RuntimeError(f"incomplete cell: {dataset}/{variant}")
            scores[variant] = mean(values)
            sample_sds[variant] = stdev(values)
        ranks = _ranks(scores)
        for variant in VARIANTS:
            rank_sums[variant] += ranks[variant]
            top_counts[variant] += int(ranks[variant] <= 1.5)
        delta = scores[VARIANTS[1]] - scores[VARIANTS[0]]
        deltas.append(delta)
        datasets.append(
            {
                "dataset": dataset,
                "means": scores,
                "sample_sds": sample_sds,
                "ranks": ranks,
                "v51_minus_v5": delta,
            }
        )

    aggregate: dict[str, object] = {}
    for variant in VARIANTS:
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
    paired = ttest_1samp(deltas, popmean=0.0)
    sem = stdev(deltas) / math.sqrt(len(deltas))
    critical = float(student_t.ppf(0.975, len(deltas) - 1))
    delta_mean = mean(deltas)
    payload: dict[str, object] = {
        "schema": "pac_logsig_v51_ucr16_fast_report.v1",
        "status": campaign_status,
        "official_test_accessed": False,
        "rows": len(rows),
        "aggregate": aggregate,
        "paired_inference": {
            "unit": "dataset-level five-seed mean",
            "v51_minus_v5_mean": delta_mean,
            "ci95": [delta_mean - critical * sem, delta_mean + critical * sem],
            "two_sided_t_pvalue": float(cast("float", paired[1])),
            "wins_ties_losses": {
                "wins": sum(delta > 1.0e-8 for delta in deltas),
                "ties": sum(abs(delta) <= 1.0e-8 for delta in deltas),
                "losses": sum(delta < -1.0e-8 for delta in deltas),
            },
        },
        "datasets": datasets,
    }
    base._atomic_json(root / "reports/summary.json", payload)
    return payload


def main() -> None:
    base.ROOT = ROOT
    base.VARIANTS = cast(
        "tuple[base.ReaderVariant, ...]",
        cast("object", VARIANTS),
    )
    base._design = _design
    base.design_sha256 = design_sha256
    base.jobs = jobs
    base._build_model = _build_model
    base.report = report
    base.main()


if __name__ == "__main__":
    main()
