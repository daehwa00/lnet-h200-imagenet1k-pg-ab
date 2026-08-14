"""Paired UCR-6 validation screen of fixed-zero versus learned pole initial states."""

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
from optimization.learnable_initial_state_alphabet import LearnableInitialStateALPHABET
from scripts import pac_pole_attention_ucr16_fast as base

ROOT = Path(".omx/results/pac-learnable-initial-state-ucr6-fast-20260724")
DATASETS = (
    "ItalyPowerDemand",
    "TwoLeadECG",
    "MoteStrain",
    "ECG200",
    "ECGFiveDays",
    "GunPoint",
)
FIXED_ZERO = "fixed_zero"
LEARNED_ZERO_INIT = "learned_zero_init"
VARIANTS = (FIXED_ZERO, LEARNED_ZERO_INIT)


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    paths = (
        Path("src/lnet/alphabet_backbone.py"),
        Path("optimization/learnable_initial_state_alphabet.py"),
        Path("src/lnet/pac_tight_frame_models.py"),
        Path("src/lnet/pac_h_compact_lag124.py"),
        Path("src/lnet/pac_h_compact_lag124_tied.py"),
        Path("src/lnet/pac_recurrence.py"),
        Path("src/lnet/pac_triton_recurrence_lag124_training.py"),
        Path("scripts/pac_pole_attention_ucr16_fast.py"),
        Path("scripts/pac_learnable_initial_state_ucr6_fast.py"),
    )
    return {str(path): file_sha256(root / path) for path in paths}


def _disable_whole_step_runtime(model: BenchmarkAlphabetBackbone) -> None:
    model.use_efp16_exact_split_training = False
    model.require_external_exact_split_training = False
    for block in (model.forward_block, model.backward_block):
        block.fused_excitation_recurrence_training = False
        block.parallel_static_excitation_recurrence_training = False


def _design() -> dict[str, object]:
    fixed = BenchmarkAlphabetBackbone(1, base.MODEL_DIM, base.MODES, 5)
    learned = LearnableInitialStateALPHABET(1, base.MODEL_DIM, base.MODES, 5)
    return {
        "schema": "pac_learnable_initial_state_ucr6_fast_contract.v1",
        "purpose": "validation-only learned complex pole initial-state screen",
        "official_test_accessed": False,
        "datasets": list(DATASETS),
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
            "replace fixed z0=0 in both writer and terminal pole analyzers with "
            "zero-initialized learned real/imaginary mode vectors; all forward "
            "operations, D, M, optimizer, split, seed, and head remain fixed"
        ),
        "initial_state_contract": {
            "parameterization": "two real vectors per analyzer representing one complex z0",
            "initialization": "exact zero",
            "recurrence_equivalence": "fold A*z0 into the first traversal drive",
            "padding": "fixed-length unpadded UCR screen only",
        },
        "params_trainable_for_five_classes": {
            FIXED_ZERO: count_parameters(fixed),
            LEARNED_ZERO_INIT: count_parameters(learned),
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
            key=f"learnable_initial_state_ucr6_fast:{dataset}:{variant}:seed{seed}",
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
        for dataset in DATASETS
        for seed in base.SEEDS
        for variant in VARIANTS
    ]


def _build_model(job: base.Job, input_dim: int, output_dim: int):  # noqa: ANN202
    if job.variant == FIXED_ZERO:
        model = BenchmarkAlphabetBackbone(input_dim, job.model_dim, job.modes, output_dim)
        _disable_whole_step_runtime(model)
        return model
    if job.variant == LEARNED_ZERO_INIT:
        return LearnableInitialStateALPHABET(
            input_dim,
            job.model_dim,
            job.modes,
            output_dim,
        )
    raise ValueError(f"unknown initial-state variant: {job.variant}")


def _ranks(scores: dict[str, float]) -> dict[str, float]:
    if math.isclose(scores[FIXED_ZERO], scores[LEARNED_ZERO_INIT], rel_tol=1.0e-5, abs_tol=1.0e-8):
        return {FIXED_ZERO: 1.5, LEARNED_ZERO_INIT: 1.5}
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

    datasets: list[dict[str, object]] = []
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
            if len(values) != len(base.SEEDS):
                raise RuntimeError(f"incomplete cell: {dataset}/{variant}")
            scores[variant] = mean(values)
            sample_sds[variant] = stdev(values)
        ranks = _ranks(scores)
        for variant in VARIANTS:
            rank_sums[variant] += ranks[variant]
            top_counts[variant] += int(ranks[variant] <= 1.5)
        delta = scores[LEARNED_ZERO_INIT] - scores[FIXED_ZERO]
        deltas.append(delta)
        datasets.append(
            {
                "dataset": dataset,
                "means": scores,
                "sample_sds": sample_sds,
                "ranks": ranks,
                "learned_minus_fixed": delta,
            }
        )

    aggregate: dict[str, object] = {}
    for variant in VARIANTS:
        active = [row for row in rows if row["variant"] == variant]
        aggregate[variant] = {
            "row_mean_balanced_accuracy": mean(
                float(row["validation_balanced_accuracy"]) for row in active
            ),
            "mean_rank": rank_sums[variant] / len(DATASETS),
            "joint_top1": top_counts[variant],
            "params_trainable": sorted({int(row["params_trainable"]) for row in active}),
            "mean_train_seconds": mean(float(row["train_seconds"]) for row in active),
        }
    payload: dict[str, object] = {
        "schema": "pac_learnable_initial_state_ucr6_fast_report.v1",
        "status": campaign_status,
        "official_test_accessed": False,
        "rows": len(rows),
        "aggregate": aggregate,
        "paired_summary": {
            "unit": "dataset-level five-seed mean",
            "learned_minus_fixed_mean": mean(deltas),
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
    base.DATASETS = cast("tuple[str, ...]", DATASETS)
    base.VARIANTS = cast("tuple[base.ReaderVariant, ...]", cast("object", VARIANTS))
    base._design = _design
    base.design_sha256 = design_sha256
    base.jobs = jobs
    base._build_model = _build_model
    base._ranks = _ranks
    base.report = report
    base.main()


if __name__ == "__main__":
    main()
