"""Fast UCR-16 screen for the corrected re-entry-aware V9.7 model."""

# pyright: reportPrivateUsage=false
# ruff: noqa: SLF001

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

from lnet.pac_final_validation import UCR_SECONDS
from lnet.pac_metrics import count_parameters
from optimization.reentry_pole_pooling_v97 import ReentryPolePoolingV97ALPHABET
from scripts import pac_logsig_v6_ucr16_fast as harness
from scripts import pac_pole_attention_ucr16_fast as base

ROOT = Path(".omx/results/pac-logsig-v97-ucr16-fast-20260721")
VARIANT = "v97_reentry_state_only_two_layer_head"
_SHARED_REPORT = harness.report


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    paths = (
        Path("optimization/reentry_pole_pooling_v97.py"),
        Path("optimization/state_only_pole_pooling_v96.py"),
        Path("optimization/state_only_pole_pooling_v95.py"),
        Path("optimization/attentive_pole_pooling_v94.py"),
        Path("optimization/attentive_pole_pooling.py"),
        Path("optimization/legendre_pole_trajectory.py"),
        Path("optimization/direct_pole_log_signature.py"),
        Path("optimization/physical_single_writer_log_signature.py"),
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
        Path("scripts/pac_logsig_v97_ucr16_fast.py"),
    )
    return {
        str(path): hashlib.sha256((root / path).read_bytes()).hexdigest()
        for path in paths
    }


def _design() -> dict[str, object]:
    candidate = ReentryPolePoolingV97ALPHABET(
        1,
        base.MODEL_DIM,
        base.MODES,
        5,
    )
    h_contract = harness.H_COMPACT_ROOT / "contract.json"
    h_summary = harness.H_COMPACT_ROOT / "reports/summary.json"
    return {
        "schema": "pac_logsig_v97_ucr16_fast_contract.v1",
        "purpose": "validation-only V9.7 screen; no automatic Q1 promotion",
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
        "reference_time_scale": 1.0,
        "hyperparameter_tuning": False,
        "automatic_q1_promotion": False,
        "controlled_difference_from_v96": (
            "normalize intervals by a fixed reference scale, mask raw values before projection, "
            "add a re-entry-only level branch, and remove the index-based DWConv"
        ),
        "model_contract": {
            "edge_lift": (
                "safe re-entry-aware level/velocity lift; consecutive observations use level "
                "and dimensionless velocity; re-entries use current level only"
            ),
            "raw_masking": "torch.where removes missing and padding values before projection",
            "time": "all intervals divided by frozen h_ref=1; valid intervals strictly positive",
            "temporal_operators": "one ExactPoleScan only; no DWConv or other temporal mixer",
            "normalization": "parameter-free amplitude-preserving bounded map",
            "modal_analysis": "invertible square orthogonal analysis with D=2M",
            "modal_core": (
                "one exact-ZOH dimensionless-time Laplace scan; missing intervals decay without "
                "drive; padding intervals freeze"
            ),
            "pole_trajectory": (
                "all interleaved real/imaginary states divided by learned positive per-mode "
                "shared scales and stabilized with elementwise asinh"
            ),
            "physical_mean": "dimensionless-time-weighted discrete right-endpoint state mean",
            "attention": (
                "state-only per-time Linear(2M,32)-SiLU-Linear(32,1,bias=False) relevance; "
                "dimensionless-time base measure; output pools original 2M pole coordinates"
            ),
            "attention_initialization": "zero query gives deterministic physical mean",
            "terminal_readout": "last-valid 2M pole vector, including valid missing-time decay",
            "explicit_relative_time": False,
            "duration_descriptor": False,
            "descriptor_dim": 6 * base.MODES,
            "head": "Linear(96,128)-SiLU-Linear(128,K)",
            "head_hidden_dim": 128,
            "initial_pole_scale": 1.0,
            "minimum_pole_scale": 1.0e-3,
            "coverage_scalar": False,
            "pole_cnn": False,
            "legendre_wls": False,
            "synthesis": False,
            "reader": False,
            "second_scan": False,
            "log_signature": False,
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
            key=f"logsig_v97_ucr16_fast:{dataset}:{VARIANT}:seed{seed}",
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
            estimated_seconds=UCR_SECONDS[dataset] * 0.55,
            design_sha256=digest,
        )
        for dataset in base.DATASETS
        for seed in base.SEEDS
    ]


def _build_model(job: base.Job, input_dim: int, output_dim: int):  # noqa: ANN202
    if job.variant != VARIANT:
        message = f"unknown V9.7 fast-screen variant: {job.variant}"
        raise ValueError(message)
    return ReentryPolePoolingV97ALPHABET(
        input_dim,
        job.model_dim,
        job.modes,
        output_dim,
        reference_time_scale=1.0,
    )


def report(root: Path) -> dict[str, object]:
    """Reject stale or TEST-contaminated rows before shared aggregation."""
    campaign_status = base.status(root)
    if campaign_status["done"] is not True:
        message = f"refusing to report incomplete V9.7 results: {campaign_status}"
        raise RuntimeError(message)
    rows = cast("list[dict[str, object]]", harness._rows(root))
    expected = {job.key for job in jobs()}
    if len(rows) != len(expected) or {str(row.get("job_key")) for row in rows} != expected:
        message = "V9.7 completed rows do not match the sealed job set"
        raise RuntimeError(message)
    digest = design_sha256()
    for row in rows:
        if (
            row.get("design_sha256") != digest
            or row.get("official_test_accessed") is not False
            or row.get("test_evaluated") is not False
            or row.get("test_count") != 0
        ):
            message = f"invalid, stale, or contaminated V9.7 row: {row.get('job_key')}"
            raise RuntimeError(message)
    return _SHARED_REPORT(root)


def main() -> None:
    harness.ROOT = ROOT
    harness.VARIANT = VARIANT
    harness.REPORT_SCHEMA = "pac_logsig_v97_ucr16_fast_report.v1"
    harness.DELTA_FIELD = "v97_minus_h_compact"
    harness._design = _design
    harness.design_sha256 = design_sha256
    harness.jobs = jobs
    harness._build_model = _build_model
    harness.report = report
    harness.main()


if __name__ == "__main__":
    main()
