"""Candidate-only Fast UCR-16 screen for the single-writer LogSignature V6."""

# pyright: reportExplicitAny=false, reportPrivateUsage=false
# ruff: noqa: EM101, EM102, SLF001, TRY003

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any, cast

from scipy.stats import t as student_t
from scipy.stats import ttest_1samp

from lnet.pac_campaign_utils import file_sha256
from lnet.pac_final_validation import UCR_SECONDS
from lnet.pac_metrics import count_parameters
from optimization.single_writer_log_signature import SingleWriterLogSignatureALPHABET
from scripts import pac_pole_attention_ucr16_fast as base

ROOT = Path(".omx/results/pac-logsig-v6-ucr16-fast-20260721")
H_COMPACT_ROOT = Path(".omx/results/pac-two-tap-ucr16-fast-20260720")
VARIANT = "v6_single_writer"
H_VARIANT = "full"
REPORT_SCHEMA = "pac_logsig_v6_ucr16_fast_report.v1"
DELTA_FIELD = "v6_minus_h_compact"


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    paths = (
        Path("optimization/single_writer_log_signature.py"),
        Path("optimization/learned_two_tap_log_signature.py"),
        Path("optimization/learned_two_tap_alphabet.py"),
        Path("optimization/masked_modal_moments.py"),
        Path("optimization/stage2_recurrence.py"),
        Path("optimization/stage2_tail_metadata.py"),
        Path("src/lnet/pac_recurrence.py"),
        Path("src/lnet/pac_triton_log_signature.py"),
        Path("src/lnet/pac_triton_log_signature_training.py"),
        Path("scripts/pac_pole_attention_ucr16_fast.py"),
        Path("scripts/pac_logsig_v6_ucr16_fast.py"),
    )
    return {str(path): file_sha256(root / path) for path in paths}


def _design() -> dict[str, object]:
    candidate = SingleWriterLogSignatureALPHABET(1, base.MODEL_DIM, base.MODES, 5)
    h_contract = H_COMPACT_ROOT / "contract.json"
    h_summary = H_COMPACT_ROOT / "reports/summary.json"
    return {
        "schema": "pac_logsig_v6_ucr16_fast_contract.v1",
        "purpose": "validation-only V6 screen; no automatic Q1 promotion",
        "official_test_accessed": False,
        "datasets": list(base.DATASETS),
        "excluded_slow_datasets": ["FordA", "FordB"],
        "seeds": list(base.SEEDS),
        "variant": VARIANT,
        "reference": {
            "model": "H-compact",
            "variant": H_VARIANT,
            "root": str(H_COMPACT_ROOT),
            "contract_sha256": file_sha256(h_contract),
            "summary_sha256": file_sha256(h_summary),
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
        "model_contract": {
            "stem": "learned Conv1d(k=2) transition then centered DWConv1d(k=5,d=1) and SiLU",
            "modal_core": "one RMS-normalized exact-pole writer scan",
            "content_readout": "RMSNorm and masked mean of the synthesized real trajectory",
            "dynamics_readout": "writer pole-motion degree-two LogSignature projected to 5M",
            "head": "single linear map from D+5M",
            "reader": False,
            "second_scan": False,
            "fixed_lags": False,
        },
        "params_trainable_for_five_classes": count_parameters(candidate),
        "source_sha256": _source_hashes(),
    }


def design_sha256() -> str:
    encoded = json.dumps(_design(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def jobs() -> list[base.Job]:
    digest = design_sha256()
    return [
        base.Job(
            key=f"logsig_v6_ucr16_fast:{dataset}:{VARIANT}:seed{seed}",
            dataset=dataset,
            variant=cast("base.ReaderVariant", cast("object", VARIANT)),
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
            estimated_seconds=UCR_SECONDS[dataset] * 0.6,
            design_sha256=digest,
        )
        for dataset in base.DATASETS
        for seed in base.SEEDS
    ]


def _build_model(job: base.Job, input_dim: int, output_dim: int):  # noqa: ANN202
    if job.variant != VARIANT:
        raise ValueError(f"unknown V6 fast-screen variant: {job.variant}")
    return SingleWriterLogSignatureALPHABET(
        input_dim,
        job.model_dim,
        job.modes,
        output_dim,
    )


def _rows(root: Path) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((root / "completed").glob("*.json"))
    ]


def report(root: Path) -> dict[str, object]:
    campaign_status = base.status(root)
    if campaign_status["done"] is not True:
        raise RuntimeError(f"refusing to report incomplete results: {campaign_status}")
    candidate_rows = _rows(root)
    reference_rows = [
        row
        for row in _rows(H_COMPACT_ROOT)
        if row.get("variant") == H_VARIANT
    ]
    if len(candidate_rows) != 80 or len(reference_rows) != 80:
        raise RuntimeError("V6 or H-compact Fast UCR-16 cell is incomplete")

    deltas: list[float] = []
    datasets: list[dict[str, object]] = []
    rank_sums = {"h_compact": 0.0, VARIANT: 0.0}
    top_counts = {"h_compact": 0, VARIANT: 0}
    for dataset in base.DATASETS:
        h_values = [
            float(row["validation_balanced_accuracy"])
            for row in reference_rows
            if row["dataset"] == dataset
        ]
        v6_values = [
            float(row["validation_balanced_accuracy"])
            for row in candidate_rows
            if row["dataset"] == dataset
        ]
        if len(h_values) != 5 or len(v6_values) != 5:
            raise RuntimeError(f"incomplete dataset: {dataset}")
        h_score, v6_score = mean(h_values), mean(v6_values)
        delta = v6_score - h_score
        deltas.append(delta)
        tied = math.isclose(h_score, v6_score, rel_tol=1.0e-5, abs_tol=1.0e-8)
        ranks = (
            {"h_compact": 1.5, VARIANT: 1.5}
            if tied
            else {
                "h_compact": 1.0 if h_score > v6_score else 2.0,
                VARIANT: 1.0 if v6_score > h_score else 2.0,
            }
        )
        for model, model_rank in ranks.items():
            rank_sums[model] += model_rank
            top_counts[model] += int(model_rank <= 1.5)
        datasets.append(
            {
                "dataset": dataset,
                "means": {"h_compact": h_score, VARIANT: v6_score},
                "sample_sds": {
                    "h_compact": stdev(h_values),
                    VARIANT: stdev(v6_values),
                },
                "ranks": ranks,
                DELTA_FIELD: delta,
            }
        )

    delta_mean = mean(deltas)
    sem = stdev(deltas) / math.sqrt(len(deltas))
    critical = float(student_t.ppf(0.975, len(deltas) - 1))
    paired = ttest_1samp(deltas, popmean=0.0)
    payload: dict[str, object] = {
        "schema": REPORT_SCHEMA,
        "status": campaign_status,
        "official_test_accessed": False,
        "aggregate": {
            "h_compact": {
                "mean_balanced_accuracy": mean(
                    float(row["validation_balanced_accuracy"]) for row in reference_rows
                ),
                "mean_rank": rank_sums["h_compact"] / len(base.DATASETS),
                "joint_top1": top_counts["h_compact"],
                "mean_params_trainable": mean(
                    int(row["params_trainable"]) for row in reference_rows
                ),
            },
            VARIANT: {
                "mean_balanced_accuracy": mean(
                    float(row["validation_balanced_accuracy"]) for row in candidate_rows
                ),
                "mean_rank": rank_sums[VARIANT] / len(base.DATASETS),
                "joint_top1": top_counts[VARIANT],
                "mean_params_trainable": mean(
                    int(row["params_trainable"]) for row in candidate_rows
                ),
                "mean_train_seconds": mean(float(row["train_seconds"]) for row in candidate_rows),
            },
        },
        "paired_inference": {
            "unit": "dataset-level five-seed mean",
            f"{DELTA_FIELD}_mean": delta_mean,
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
        cast("object", (VARIANT,)),
    )
    base._design = _design
    base.design_sha256 = design_sha256
    base.jobs = jobs
    base._build_model = _build_model
    base.report = report
    base.main()


if __name__ == "__main__":
    main()
