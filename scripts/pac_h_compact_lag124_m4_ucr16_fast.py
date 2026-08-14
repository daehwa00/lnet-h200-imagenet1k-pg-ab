"""Fast UCR-16 validation screen for adding M=4 to H-compact lag-(1,2,4)."""

# pyright: reportExplicitAny=false, reportPrivateUsage=false
# ruff: noqa: EM101, EM102, SLF001, TRY003

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, cast

from lnet.pac_final_validation import UCR_SECONDS
from lnet.pac_h_compact_lag124 import HCompactLag124ALPHABET
from lnet.pac_metrics import count_parameters
from scripts import pac_pole_attention_ucr16_fast as base

ROOT = Path(".omx/results/pac-h-compact-lag124-m4-ucr16-fast-v3-20260721")
REFERENCE_ROOT = Path(".omx/results/pac-h-compact-lag124-ucr16-fast-20260721")
REFERENCE_VARIANT = "h_compact_lags_1_2_4"
MODEL_DIMS = (16, 32, 64)
MODES = 4
VARIANTS = tuple(f"h_compact_lag124_d{model_dim}_m4" for model_dim in MODEL_DIMS)


def _design() -> dict[str, object]:
    source = Path(__file__).resolve()
    model_source = source.parents[1] / "src/lnet/pac_h_compact_lag124.py"
    return {
        "schema": "pac_h_compact_lag124_m4_ucr16_fast_contract.v1",
        "purpose": "validation-only M=4 boundary screen; no Q1 result merging",
        "official_test_accessed": False,
        "datasets": list(base.DATASETS),
        "excluded_slow_datasets": ["FordA", "FordB"],
        "seeds": list(base.SEEDS),
        "model_dims": list(MODEL_DIMS),
        "modes": MODES,
        "epochs": base.EPOCHS,
        "batch_size": base.BATCH_SIZE,
        "learning_rate": base.LEARNING_RATE,
        "weight_decay": base.WEIGHT_DECAY,
        "grad_clip_norm": base.GRAD_CLIP_NORM,
        "hyperparameter_tuning": False,
        "automatic_q1_promotion": False,
        "params_trainable_for_five_classes": {
            str(model_dim): count_parameters(
                HCompactLag124ALPHABET(1, model_dim, MODES, 5)
            )
            for model_dim in MODEL_DIMS
        },
        "source_sha256": {
            str(source.relative_to(source.parents[1])): hashlib.sha256(
                source.read_bytes()
            ).hexdigest(),
            str(model_source.relative_to(source.parents[1])): hashlib.sha256(
                model_source.read_bytes()
            ).hexdigest(),
        },
    }


def design_sha256() -> str:
    encoded = json.dumps(_design(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def jobs() -> list[base.Job]:
    digest = design_sha256()
    return [
        base.Job(
            key=f"h_compact_lag124_m4_ucr16_fast:{dataset}:{variant}:seed{seed}",
            dataset=dataset,
            variant=cast("base.ReaderVariant", cast("object", variant)),
            split_seed=seed,
            train_seed=seed,
            model_dim=model_dim,
            modes=MODES,
            heads=1,
            epochs=base.EPOCHS,
            batch_size=base.BATCH_SIZE,
            learning_rate=base.LEARNING_RATE,
            weight_decay=base.WEIGHT_DECAY,
            grad_clip_norm=base.GRAD_CLIP_NORM,
            evaluation_split="validation",
            estimated_seconds=UCR_SECONDS[dataset] * max(0.5, model_dim / 32),
            design_sha256=digest,
        )
        for model_dim, variant in zip(MODEL_DIMS, VARIANTS, strict=True)
        for dataset in base.DATASETS
        for seed in base.SEEDS
    ]


def _build_model(job: base.Job, input_dim: int, output_dim: int):  # noqa: ANN202
    if job.variant not in VARIANTS or job.modes != MODES:
        raise ValueError(f"unknown M=4 variant: {job.variant}")
    return HCompactLag124ALPHABET(input_dim, job.model_dim, job.modes, output_dim)


def _rows(root: Path) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((root / "completed").glob("*.json"))
    ]


def report(root: Path) -> dict[str, object]:
    campaign_status = base.status(root)
    if campaign_status["done"] is not True:
        raise RuntimeError(f"refusing to report incomplete results: {campaign_status}")
    rows = _rows(root)
    reference_rows = _rows(REFERENCE_ROOT)
    if len(rows) != len(jobs()):
        raise RuntimeError("M=4 completed rows do not match the sealed job set")
    datasets: list[dict[str, object]] = []
    selected_dims: dict[int, int] = dict.fromkeys(MODEL_DIMS, 0)
    deltas: list[float] = []
    for dataset in base.DATASETS:
        reference_values = [
            float(row["validation_balanced_accuracy"])
            for row in reference_rows
            if row.get("dataset") == dataset and row.get("variant") == REFERENCE_VARIANT
        ]
        scores: dict[str, float] = {}
        for variant in VARIANTS:
            values = [
                float(row["validation_balanced_accuracy"])
                for row in rows
                if row.get("dataset") == dataset and row.get("variant") == variant
            ]
            if len(values) != len(base.SEEDS):
                raise RuntimeError(f"incomplete M=4 cell: {dataset}/{variant}")
            scores[variant] = mean(values)
        best_variant = max(scores, key=scores.__getitem__)
        best_dim = int(best_variant.split("_d", 1)[1].split("_", 1)[0])
        selected_dims[best_dim] += 1
        reference_score = mean(reference_values)
        best_score = scores[best_variant]
        delta = best_score - reference_score
        deltas.append(delta)
        datasets.append(
            {
                "dataset": dataset,
                "scores": scores,
                "best_m4_variant": best_variant,
                "best_m4_score": best_score,
                "lag124_d32_m16_score": reference_score,
                "best_m4_minus_lag124_d32_m16": delta,
                "m4_wins": delta > 0 and not math.isclose(delta, 0.0, abs_tol=1.0e-8),
            }
        )
    payload: dict[str, object] = {
        "schema": "pac_h_compact_lag124_m4_ucr16_fast_report.v1",
        "official_test_accessed": False,
        "jobs": len(rows),
        "selected_model_dim_counts": selected_dims,
        "best_m4_mean_balanced_accuracy": mean(
            float(row["best_m4_score"]) for row in datasets
        ),
        "lag124_d32_m16_mean_balanced_accuracy": mean(
            float(row["lag124_d32_m16_score"]) for row in datasets
        ),
        "mean_delta": mean(deltas),
        "win_tie_loss": {
            "wins": sum(delta > 1.0e-8 for delta in deltas),
            "ties": sum(abs(delta) <= 1.0e-8 for delta in deltas),
            "losses": sum(delta < -1.0e-8 for delta in deltas),
        },
        "datasets": datasets,
    }
    output = root / "reports/summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    base.ROOT = ROOT
    base._design = _design
    base.design_sha256 = design_sha256
    base.jobs = jobs
    base._build_model = _build_model
    base.report = report
    base.main()


if __name__ == "__main__":
    main()
