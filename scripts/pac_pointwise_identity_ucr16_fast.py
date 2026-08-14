"""Paired UCR-16 validation screen of edge versus raw-pointwise Identity ALPHABET."""

# pyright: reportPrivateUsage=false
# ruff: noqa: EM101, EM102, SLF001, TRY003

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import cast

from lnet.alphabet_backbone import BenchmarkAlphabetBackbone
from lnet.pac_campaign_utils import file_sha256
from lnet.pac_final_validation import UCR_SECONDS
from lnet.pac_metrics import count_parameters
from optimization.pointwise_identity_alphabet import PointwiseBenchmarkAlphabetBackbone
from scripts import pac_pole_attention_ucr16_fast as base

ROOT = Path(".omx/results/pac-pointwise-identity-ucr16-fast-20260722")
EDGE_VARIANT = "identity_edge"
POINTWISE_VARIANT = "identity_pointwise"
VARIANTS = (EDGE_VARIANT, POINTWISE_VARIANT)


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    paths = (
        Path("src/lnet/alphabet_backbone.py"),
        Path("optimization/pointwise_identity_alphabet.py"),
        Path("src/lnet/pac_h_compact_lag124_tied.py"),
        Path("src/lnet/pac_h_compact_lag124.py"),
        Path("src/lnet/pac_headroom_efficient_models.py"),
        Path("src/lnet/pac_tight_frame_models.py"),
        Path("src/lnet/pac_efp16_exact_split_training.py"),
        Path("src/lnet/pac_triton_parallel_static_recurrence_lag124_training.py"),
        Path("src/lnet/pac_triton_parallel_static_recurrence.py"),
        Path("src/lnet/pac_triton_edge_frame_stem_training.py"),
        Path("src/lnet/pac_triton_terminal_reader_local_training.py"),
        Path("src/lnet/pac_cuda_outer_graph.py"),
        Path("src/lnet/pac_cuda_conditional_matrix_exp.py"),
        Path("src/lnet/pac_native_matrix_exp_vjp.py"),
        Path("src/lnet/pac_recurrence.py"),
        Path("src/lnet/pac_triton_recurrence_op.py"),
        Path("src/lnet/pac_cuda_fused_optimizer.py"),
        Path("src/lnet/pac_cuda_fused_optimizer_runtime.py"),
        Path("src/lnet/pac_triton_skew_matrix_exp_vjp.py"),
        Path("scripts/pac_pole_attention_ucr16_fast.py"),
        Path("scripts/pac_pointwise_identity_ucr16_fast.py"),
    )
    return {str(path): file_sha256(root / path) for path in paths}


def _design() -> dict[str, object]:
    edge = BenchmarkAlphabetBackbone(1, base.MODEL_DIM, base.MODES, 5)
    pointwise = PointwiseBenchmarkAlphabetBackbone(1, base.MODEL_DIM, base.MODES, 5)
    return {
        "schema": "pac_pointwise_identity_ucr16_fast_contract.v1",
        "purpose": "validation-only controlled input-stem screen",
        "official_test_accessed": False,
        "datasets": list(base.DATASETS),
        "excluded_slow_datasets": ["FordA", "FordB"],
        "seeds": list(base.SEEDS),
        "variants": list(VARIANTS),
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
            "replace only degree-normalized level/detail edge construction and its "
            "2C-to-D projection with a semi-orthogonal pointwise C-to-D Linear; "
            "retain DWConv(k=5,d=4), SiLU, tied writer, identity terminal projection, "
            "terminal local analyzer, lag-(1,2,4) moments, and linear task head"
        ),
        "token_grids": {EDGE_VARIANT: "N-1 edges", POINTWISE_VARIANT: "N raw nodes"},
        "params_trainable_for_five_classes": {
            EDGE_VARIANT: count_parameters(edge),
            POINTWISE_VARIANT: count_parameters(pointwise),
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
            key=f"pointwise_identity_ucr16_fast:{dataset}:{variant}:seed{seed}",
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
    if job.variant == EDGE_VARIANT:
        return BenchmarkAlphabetBackbone(input_dim, job.model_dim, job.modes, output_dim)
    if job.variant == POINTWISE_VARIANT:
        return PointwiseBenchmarkAlphabetBackbone(input_dim, job.model_dim, job.modes, output_dim)
    raise ValueError(f"unknown pointwise Identity variant: {job.variant}")


def report(root: Path) -> dict[str, object]:
    # Reporting is intentionally decoupled from workers so lean CUDA training
    # environments do not need SciPy installed.
    from scipy.stats import t as student_t  # noqa: PLC0415
    from scipy.stats import ttest_1samp  # noqa: PLC0415

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
    digest = design_sha256()
    for row in rows:
        if (
            str(row.get("job_key")) not in expected
            or row.get("official_test_accessed") is not False
            or row.get("test_evaluated") is not False
            or row.get("test_count") != 0
            or row.get("design_sha256") != digest
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
        tied = math.isclose(
            scores[EDGE_VARIANT],
            scores[POINTWISE_VARIANT],
            rel_tol=1.0e-5,
            abs_tol=1.0e-8,
        )
        ranks = (
            {EDGE_VARIANT: 1.5, POINTWISE_VARIANT: 1.5}
            if tied
            else {
                EDGE_VARIANT: 1.0 if scores[EDGE_VARIANT] > scores[POINTWISE_VARIANT] else 2.0,
                POINTWISE_VARIANT: (
                    1.0 if scores[POINTWISE_VARIANT] > scores[EDGE_VARIANT] else 2.0
                ),
            }
        )
        for variant in VARIANTS:
            rank_sums[variant] += ranks[variant]
            top_counts[variant] += int(ranks[variant] <= 1.5)
        delta = scores[POINTWISE_VARIANT] - scores[EDGE_VARIANT]
        deltas.append(delta)
        datasets.append(
            {
                "dataset": dataset,
                "means": scores,
                "sample_sds": sample_sds,
                "ranks": ranks,
                "pointwise_minus_edge": delta,
            }
        )

    aggregate: dict[str, object] = {}
    for variant in VARIANTS:
        active = [row for row in rows if row["variant"] == variant]
        aggregate[variant] = {
            "row_mean_balanced_accuracy": mean(
                float(row["validation_balanced_accuracy"]) for row in active
            ),
            "mean_rank": rank_sums[variant] / len(base.DATASETS),
            "joint_top1": top_counts[variant],
            "params_trainable": sorted({int(row["params_trainable"]) for row in active}),
            "mean_train_seconds": mean(float(row["train_seconds"]) for row in active),
        }
    paired = ttest_1samp(deltas, popmean=0.0)
    sem = stdev(deltas) / math.sqrt(len(deltas))
    critical = float(student_t.ppf(0.975, len(deltas) - 1))
    delta_mean = mean(deltas)
    payload: dict[str, object] = {
        "schema": "pac_pointwise_identity_ucr16_fast_report.v1",
        "status": campaign_status,
        "official_test_accessed": False,
        "rows": len(rows),
        "aggregate": aggregate,
        "paired_inference": {
            "unit": "dataset-level five-seed mean",
            "pointwise_minus_edge_mean": delta_mean,
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
    base.VARIANTS = cast("tuple[base.ReaderVariant, ...]", cast("object", VARIANTS))
    base._design = _design
    base.design_sha256 = design_sha256
    base.jobs = jobs
    base._build_model = _build_model
    base.report = report
    base.main()


if __name__ == "__main__":
    main()
