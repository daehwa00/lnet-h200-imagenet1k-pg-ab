"""Fast matched UCR-16 test of removing ALPHABET's reader DWConv."""

# pyright: reportExplicitAny=false, reportPrivateUsage=false
# ruff: noqa: SLF001

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import numpy as np

from lnet.pac_metrics import count_parameters
from lnet.pac_reader_local_removal_ablation import (
    VARIANTS,
    ReaderLocalRemovalAlphabet,
    ReaderLocalVariant,
)
from lnet.pac_types import PACExperimentConfig
from scripts import pac_pole_attention_ucr16_fast as harness

ROOT = Path(".omx/results/pac-reader-local-removal-ucr16-fast-kau-20260726")
MODEL_DIM = 64
MODES = 16
BOOTSTRAP_DRAWS = 20_000
_SHARED_MAIN = harness.main


def _model_config(input_dim: int, output_dim: int) -> PACExperimentConfig:
    return PACExperimentConfig(
        1,
        1,
        0,
        2,
        raw_input_dim=input_dim,
        output_dim=output_dim,
        model_dim=MODEL_DIM,
        modes=MODES,
    )


def _build_named_model(
    variant: str,
    input_dim: int,
    output_dim: int,
) -> ReaderLocalRemovalAlphabet:
    return ReaderLocalRemovalAlphabet(
        _model_config(input_dim, output_dim),
        output_dim,
        variant=cast("ReaderLocalVariant", variant),
    )


def _source_hashes() -> dict[str, str]:
    project = Path(__file__).resolve().parents[1]
    names = (
        "src/lnet/alphabet.py",
        "src/lnet/alphabet_backbone.py",
        "src/lnet/pac_reader_local_removal_ablation.py",
        "src/lnet/pac_training.py",
        "scripts/pac_pole_attention_ucr16_fast.py",
        "scripts/pac_reader_local_removal_ucr16_fast.py",
    )
    return {
        name: hashlib.sha256((project / name).read_bytes()).hexdigest()
        for name in names
    }


def _design() -> dict[str, object]:
    models = {
        variant: _build_named_model(variant, 1, 5)
        for variant in VARIANTS
    }
    return {
        "schema": "pac_reader_local_removal_ucr16_fast_contract.v1",
        "purpose": (
            "validation-only initialization-matched test of whether ALPHABET "
            "needs a reader-local DWConv when the input stem uses contiguous d1"
        ),
        "claim_status": "exploratory validation-only architecture development",
        "official_test_accessed": False,
        "automatic_promotion": False,
        "datasets": list(harness.DATASETS),
        "excluded_slow_datasets": ["FordA", "FordB"],
        "seeds": list(harness.SEEDS),
        "variants": list(VARIANTS),
        "model_dim": MODEL_DIM,
        "modes": MODES,
        "epochs": harness.EPOCHS,
        "batch_size": harness.BATCH_SIZE,
        "learning_rate": harness.LEARNING_RATE,
        "weight_decay": harness.WEIGHT_DECAY,
        "grad_clip_norm": harness.GRAD_CLIP_NORM,
        "hyperparameter_tuning": False,
        "fixed_structure": (
            "final radial-log R0/R1/R2/R4 descriptor, affine head, RMSNorms, "
            "writer-reader pole banks, recurrence, synthesis, D, M, optimizer, "
            "split, seed, training budget, and input DWConv5 dilation 1"
        ),
        "controlled_cells": {
            "stem_d1_reader_d1": {
                "stem": "centered depthwise kernel-5 dilation-1 padding-2 plus SiLU",
                "reader": "centered depthwise kernel-5 dilation-1 padding-2 plus SiLU",
            },
            "stem_d1_reader_none": {
                "stem": "centered depthwise kernel-5 dilation-1 padding-2 plus SiLU",
                "reader": "parameter-free identity plus the existing SiLU",
            },
        },
        "initialization_match": (
            "both cells initialize the complete canonical model under the same seed; "
            "the no-reader cell then replaces second_local with a parameter-free identity"
        ),
        "primary_estimand": (
            "stem_d1_reader_none - stem_d1_reader_d1 validation balanced accuracy"
        ),
        "five_class_trainable_parameters": {
            variant: count_parameters(model)
            for variant, model in models.items()
        },
        "source_sha256": _source_hashes(),
    }


def design_sha256() -> str:
    encoded = json.dumps(_design(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def jobs() -> list[harness.Job]:
    digest = design_sha256()
    return [
        harness.Job(
            key=f"reader_local_removal_ucr16_fast:{dataset}:{variant}:seed{seed}",
            dataset=dataset,
            variant=cast("harness.ReaderVariant", cast("object", variant)),
            split_seed=seed,
            train_seed=seed,
            model_dim=MODEL_DIM,
            modes=MODES,
            heads=1,
            epochs=harness.EPOCHS,
            batch_size=harness.BATCH_SIZE,
            learning_rate=harness.LEARNING_RATE,
            weight_decay=harness.WEIGHT_DECAY,
            grad_clip_norm=harness.GRAD_CLIP_NORM,
            evaluation_split="validation",
            estimated_seconds=harness.UCR_SECONDS[dataset] * 2.0,
            design_sha256=digest,
        )
        for dataset in harness.DATASETS
        for seed in harness.SEEDS
        for variant in VARIANTS
    ]


def _build_model(
    job: harness.Job,
    input_dim: int,
    output_dim: int,
) -> ReaderLocalRemovalAlphabet:
    return _build_named_model(str(job.variant), input_dim, output_dim)


def _completed_rows(root: Path) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((root / "completed").glob("*.json"))
    ]


def _ranks(scores: dict[str, float]) -> dict[str, float]:
    first, second = VARIANTS
    if abs(scores[first] - scores[second]) <= 1.0e-8:
        return {first: 1.5, second: 1.5}
    winner = max(VARIANTS, key=scores.__getitem__)
    return {variant: 1.0 if variant == winner else 2.0 for variant in VARIANTS}


def _bootstrap_effect(effect: np.ndarray) -> dict[str, object]:
    task_effect = effect.mean(axis=1)
    task_count, seed_count = effect.shape
    rng = np.random.default_rng(20260726)
    bootstrap = np.empty(BOOTSTRAP_DRAWS)
    for draw in range(BOOTSTRAP_DRAWS):
        sampled_tasks = rng.integers(task_count, size=task_count)
        sampled_effects = np.empty(task_count)
        for slot, task_index in enumerate(sampled_tasks):
            sampled_seeds = rng.integers(seed_count, size=seed_count)
            sampled_effects[slot] = effect[task_index, sampled_seeds].mean()
        bootstrap[draw] = sampled_effects.mean()
    return {
        "mean_effect": float(task_effect.mean()),
        "hierarchical_paired_bootstrap_ci95": np.quantile(
            bootstrap,
            (0.025, 0.975),
        ).tolist(),
        "bootstrap_resamples_positive_fraction": float((bootstrap > 0).mean()),
        "wins_ties_losses": {
            "wins": int((task_effect > 1.0e-8).sum()),
            "ties": int((np.abs(task_effect) <= 1.0e-8).sum()),
            "losses": int((task_effect < -1.0e-8).sum()),
        },
    }


def report(root: Path) -> dict[str, object]:
    status = harness.status(root)
    if status["done"] is not True:
        message = f"refusing to report incomplete results: {status}"
        raise RuntimeError(message)
    rows = _completed_rows(root)
    expected = {job.key for job in jobs()}
    digest = design_sha256()
    if len(rows) != len(expected):
        message = "completed row count disagrees with sealed reader-local design"
        raise RuntimeError(message)
    for row in rows:
        if (
            row.get("job_key") not in expected
            or row.get("design_sha256") != digest
            or row.get("official_test_accessed") is not False
            or row.get("test_evaluated") is not False
            or row.get("test_count") != 0
        ):
            message = f"invalid or contaminated reader-local row: {row.get('job_key')}"
            raise RuntimeError(message)

    tasks = list(harness.DATASETS)
    seeds = list(harness.SEEDS)
    values: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    for row in rows:
        values[(str(row["dataset"]), str(row["variant"]))][int(row["train_seed"])] = float(
            row["validation_balanced_accuracy"]
        )
    matrices = {
        variant: np.asarray(
            [[values[(task, variant)][seed] for seed in seeds] for task in tasks],
            dtype=np.float64,
        )
        for variant in VARIANTS
    }
    effect = matrices["stem_d1_reader_none"] - matrices["stem_d1_reader_d1"]
    rank_sums = dict.fromkeys(VARIANTS, 0.0)
    datasets = []
    for task_index, task in enumerate(tasks):
        means = {
            variant: float(matrices[variant][task_index].mean())
            for variant in VARIANTS
        }
        ranks = _ranks(means)
        for variant in VARIANTS:
            rank_sums[variant] += ranks[variant]
        datasets.append(
            {
                "dataset": task,
                "means": means,
                "ranks": ranks,
                "reader_none_minus_reader_d1": float(effect[task_index].mean()),
            }
        )

    payload: dict[str, object] = {
        "schema": "pac_reader_local_removal_ucr16_fast_report.v1",
        "status": status,
        "claim_status": "exploratory validation-only architecture development",
        "official_test_accessed": False,
        "rows": len(rows),
        "mean_task_balanced_accuracy": {
            variant: float(matrix.mean(axis=1).mean())
            for variant, matrix in matrices.items()
        },
        "mean_rank": {
            variant: rank_sums[variant] / len(tasks)
            for variant in VARIANTS
        },
        "reader_none_minus_reader_d1": _bootstrap_effect(effect),
        "params_trainable": {
            variant: sorted(
                {
                    int(row["params_trainable"])
                    for row in rows
                    if row["variant"] == variant
                }
            )
            for variant in VARIANTS
        },
        "mean_train_seconds": {
            variant: float(
                np.mean(
                    [
                        float(row["train_seconds"])
                        for row in rows
                        if row["variant"] == variant
                    ]
                )
            )
            for variant in VARIANTS
        },
        "datasets": datasets,
    }
    harness._atomic_json(root / "reports" / "summary.json", payload)
    return payload


def configure_harness() -> None:
    harness.ROOT = ROOT
    harness.VARIANTS = cast("tuple[harness.ReaderVariant, ...]", VARIANTS)
    harness.MODEL_DIM = MODEL_DIM
    harness.MODES = MODES
    harness._design = _design
    harness.design_sha256 = design_sha256
    harness.jobs = jobs
    harness._build_model = _build_model
    harness.report = report


def main() -> None:
    configure_harness()
    _SHARED_MAIN()


if __name__ == "__main__":
    main()
