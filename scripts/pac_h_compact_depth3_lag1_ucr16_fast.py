"""Fast UCR-16 screen of a three-stage, lag-one H-style ALPHABET."""

# pyright: reportPrivateUsage=false
# ruff: noqa: SLF001

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

from lnet.pac_final_validation import UCR_SECONDS
from lnet.pac_h_compact_depth3_lag1 import HCompactDepth3Lag1ALPHABET
from lnet.pac_metrics import count_parameters
from scripts import pac_logsig_v6_ucr16_fast as harness
from scripts import pac_pole_attention_ucr16_fast as base

ROOT = Path(".omx/results/pac-h-compact-depth3-lag1-ucr16-fast-20260721")
VARIANT = "h_compact_depth3_lag1"
_SHARED_REPORT = harness.report


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    paths = (
        Path("src/lnet/pac_h_compact_depth3_lag1.py"),
        Path("src/lnet/pac_efp_writer_reader.py"),
        Path("src/lnet/pac_laplace_native_input.py"),
        Path("src/lnet/pac_headroom_efficient_models.py"),
        Path("src/lnet/pac_headroom_models.py"),
        Path("src/lnet/pac_raw_efficiency_candidates.py"),
        Path("src/lnet/pac_tight_frame_models.py"),
        Path("scripts/pac_pole_attention_ucr16_fast.py"),
        Path("scripts/pac_logsig_v6_ucr16_fast.py"),
        Path("scripts/pac_h_compact_depth3_lag1_ucr16_fast.py"),
    )
    return {
        str(path): hashlib.sha256((root / path).read_bytes()).hexdigest()
        for path in paths
    }


def _design() -> dict[str, object]:
    candidate = HCompactDepth3Lag1ALPHABET(1, base.MODEL_DIM, base.MODES, 5)
    h_contract = harness.H_COMPACT_ROOT / "contract.json"
    h_summary = harness.H_COMPACT_ROOT / "reports/summary.json"
    return {
        "schema": "pac_h_compact_depth3_lag1_ucr16_fast_contract.v1",
        "purpose": "validation-only depth-three lag-one screen; no automatic Q1 promotion",
        "official_test_accessed": False,
        "datasets": list(base.DATASETS),
        "excluded_slow_datasets": ["FordA", "FordB"],
        "seeds": list(base.SEEDS),
        "variant": VARIANT,
        "reference": {
            "model": "H-compact",
            "variant": harness.H_VARIANT,
            "root": str(harness.H_COMPACT_ROOT),
            "contract_sha256": hashlib.sha256(h_contract.read_bytes()).hexdigest(),
            "summary_sha256": hashlib.sha256(h_summary.read_bytes()).hexdigest(),
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
            "replace the canonical full-writer plus read-only-reader depth with two "
            "full H-style writers and one read-only terminal analyzer; every stage "
            "uses only normalized complex lag-one correlations"
        ),
        "stage_roles": ["full_writer", "full_writer", "read_only_terminal_analyzer"],
        "moment_lags_per_stage": [[1], [1], [1]],
        "descriptor_dim": base.MODEL_DIM + 9 * base.MODES,
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
            key=f"h_compact_depth3_lag1_ucr16_fast:{dataset}:{VARIANT}:seed{seed}",
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
            estimated_seconds=UCR_SECONDS[dataset] * 1.35,
            design_sha256=digest,
        )
        for dataset in base.DATASETS
        for seed in base.SEEDS
    ]


def _build_model(job: base.Job, input_dim: int, output_dim: int):  # noqa: ANN202
    if job.variant != VARIANT:
        message = f"unknown H depth-three lag-one variant: {job.variant}"
        raise ValueError(message)
    return HCompactDepth3Lag1ALPHABET(
        input_dim,
        job.model_dim,
        job.modes,
        output_dim,
    )


def report(root: Path) -> dict[str, object]:
    campaign_status = base.status(root)
    if campaign_status["done"] is not True:
        message = f"refusing to report incomplete depth-three results: {campaign_status}"
        raise RuntimeError(message)
    rows = cast("list[dict[str, object]]", harness._rows(root))
    expected = {job.key for job in jobs()}
    if len(rows) != len(expected) or {str(row.get("job_key")) for row in rows} != expected:
        message = "depth-three completed rows do not match the sealed job set"
        raise RuntimeError(message)
    digest = design_sha256()
    for row in rows:
        if (
            row.get("design_sha256") != digest
            or row.get("official_test_accessed") is not False
            or row.get("test_evaluated") is not False
            or row.get("test_count") != 0
        ):
            message = f"invalid, stale, or contaminated depth-three row: {row.get('job_key')}"
            raise RuntimeError(message)
    return _SHARED_REPORT(root)


def main() -> None:
    harness.ROOT = ROOT
    harness.VARIANT = VARIANT
    harness.REPORT_SCHEMA = "pac_h_compact_depth3_lag1_ucr16_fast_report.v1"
    harness.DELTA_FIELD = "depth3_lag1_minus_h_compact"
    harness._design = _design
    harness.design_sha256 = design_sha256
    harness.jobs = jobs
    harness._build_model = _build_model
    harness.report = report
    harness.main()


if __name__ == "__main__":
    main()
