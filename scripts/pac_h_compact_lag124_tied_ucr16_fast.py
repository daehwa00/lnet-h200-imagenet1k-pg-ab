"""Fast UCR-16 screen of tied versus untied H-compact lag-(1,2,4)."""

# pyright: reportPrivateUsage=false
# ruff: noqa: EM102, SLF001, TRY003

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

from lnet.pac_campaign_utils import file_sha256
from lnet.pac_final_validation import UCR_SECONDS
from lnet.pac_h_compact_lag124_tied import HCompactLag124TiedALPHABET
from lnet.pac_metrics import count_parameters
from scripts import pac_logsig_v6_ucr16_fast as harness
from scripts import pac_pole_attention_ucr16_fast as base

ROOT = Path(".omx/results/pac-h-compact-lag124-tied-ucr16-fast-20260721")
REFERENCE_ROOT = Path(".omx/results/pac-h-compact-lag124-ucr16-fast-20260721")
VARIANT = "h_compact_lags_1_2_4_tied_rs"
REFERENCE_VARIANT = "h_compact_lags_1_2_4"


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    paths = (
        Path("src/lnet/pac_h_compact_lag124_tied.py"),
        Path("src/lnet/pac_h_compact_lag124.py"),
        Path("src/lnet/pac_efp_writer_reader.py"),
        Path("src/lnet/pac_tight_frame_models.py"),
        Path("scripts/pac_pole_attention_ucr16_fast.py"),
        Path("scripts/pac_logsig_v6_ucr16_fast.py"),
        Path("scripts/pac_h_compact_lag124_tied_ucr16_fast.py"),
    )
    return {str(path): file_sha256(root / path) for path in paths}


def _design() -> dict[str, object]:
    candidate = HCompactLag124TiedALPHABET(1, base.MODEL_DIM, base.MODES, 5)
    reference_contract = REFERENCE_ROOT / "contract.json"
    reference_summary = REFERENCE_ROOT / "reports/summary.json"
    return {
        "schema": "pac_h_compact_lag124_tied_ucr16_fast_contract.v1",
        "purpose": "validation-only writer analysis/synthesis tying screen",
        "official_test_accessed": False,
        "datasets": list(base.DATASETS),
        "excluded_slow_datasets": ["FordA", "FordB"],
        "seeds": list(base.SEEDS),
        "variant": VARIANT,
        "reference": {
            "model": "H-compact lag-(1,2,4), untied writer frames",
            "variant": REFERENCE_VARIANT,
            "root": str(REFERENCE_ROOT),
            "contract_sha256": file_sha256(reference_contract),
            "summary_sha256": file_sha256(reference_summary),
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
            "remove only the writer independent synthesis frame and reuse its "
            "orthogonal analysis frame for synthesis; retain lag-(1,2,4), the "
            "terminal analyzer, head, optimizer, split, and seeds"
        ),
        "writer_analysis_synthesis_tied": True,
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
            key=f"h_compact_lag124_tied_ucr16_fast:{dataset}:{VARIANT}:seed{seed}",
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
            estimated_seconds=UCR_SECONDS[dataset],
            design_sha256=digest,
        )
        for dataset in base.DATASETS
        for seed in base.SEEDS
    ]


def _build_model(job: base.Job, input_dim: int, output_dim: int):  # noqa: ANN202
    if job.variant != VARIANT:
        raise ValueError(f"unknown tied H-compact variant: {job.variant}")
    return HCompactLag124TiedALPHABET(
        input_dim,
        job.model_dim,
        job.modes,
        output_dim,
    )


def main() -> None:
    harness.ROOT = ROOT
    harness.H_COMPACT_ROOT = REFERENCE_ROOT
    harness.H_VARIANT = REFERENCE_VARIANT
    harness.VARIANT = VARIANT
    harness.REPORT_SCHEMA = "pac_h_compact_lag124_tied_ucr16_fast_report.v1"
    harness.DELTA_FIELD = "tied_minus_untied"
    harness._design = _design
    harness.design_sha256 = design_sha256
    harness.jobs = jobs
    harness._build_model = _build_model
    harness.main()


if __name__ == "__main__":
    main()
