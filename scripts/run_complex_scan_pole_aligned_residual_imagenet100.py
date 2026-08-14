# ruff: noqa: SLF001
# pyright: reportAny=false, reportExplicitAny=false, reportImplicitRelativeImport=false
# pyright: reportPrivateLocalImportUsage=false, reportPrivateUsage=false
"""Run the pole-aligned stage-residual complex scan backbone on ImageNet-100."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import run_complex_scan_zero_init_imagenet100 as base

if TYPE_CHECKING:
    from argparse import Namespace

    from lnet.complex_scan import ComplexScanConfig

VARIANT = "pole_aligned_residual_dual_fusion256_lrq64"
SEEDS = (501,)
FUSION_WIDTH = 256
STAGE_RESIDUAL_SCALE_INITIAL = 0.1
_BASE_VARIANT_CONFIG = base._variant_config
_BASE_CONTRACT = base._contract


def _variant_config(config: ComplexScanConfig) -> ComplexScanConfig:
    return replace(
        _BASE_VARIANT_CONFIG(config),
        carry_bases=("none", "none"),
        use_pole_aligned_shortcuts=True,
        stage_residual_scale_initial=STAGE_RESIDUAL_SCALE_INITIAL,
        fusion_width=FUSION_WIDTH,
        dual_fusion_lrq_head=True,
    )


def _contract(args: Namespace) -> dict[str, object]:
    payload = _BASE_CONTRACT(args)
    payload["schema"] = "lnet.complex_scan.pole_aligned_residual.imagenet100.v1"
    payload["evidence_status"] = "single-seed 100-epoch architecture screen"
    architecture = payload["architecture"]
    if not isinstance(architecture, dict):
        message = "pole-aligned residual contract architecture is malformed"
        raise TypeError(message)
    architecture.update(
        {
            "backbone": (
                "pole-aligned complex low-pass shortcut plus scaled "
                "PoleStage/PoleDown/Augmented-FFN residual"
            ),
            "stage_shortcut": (
                "per-mode conjugate pole-phase alignment, fixed 3x3 binomial "
                "low-pass, stride 2, and coarse-grid carrier retention"
            ),
            "stage_residual_scale_initial": STAGE_RESIDUAL_SCALE_INITIAL,
            "legacy_stage_carry": "disabled",
            "head": "Fusion256 + initially disabled parallel LRQ64",
        }
    )
    sources = payload["source_sha256"]
    if not isinstance(sources, dict):
        message = "pole-aligned residual contract source payload is malformed"
        raise TypeError(message)
    sources["runner"] = base.harness._digest(Path(__file__))
    return json.loads(json.dumps(payload))


def _summarize(root: Path, contract: dict[str, Any]) -> dict[str, Any] | None:
    path = root / "results" / f"{VARIANT}__seed{SEEDS[0]}.json"
    if not path.exists():
        return None
    row = json.loads(path.read_text())
    accuracy = float(row["final_validation"]["accuracy"])
    payload = {
        "schema": contract["schema"],
        "variant": VARIANT,
        "seed": SEEDS[0],
        "parameters": int(row["parameters"]),
        "final_accuracy": accuracy,
        "final_cross_entropy": float(row["final_validation"]["cross_entropy"]),
        "training_examples_per_second": float(
            row["complete_training_examples_per_second"]
        ),
        "delta_to_existing_baselines_pp": {
            name: 100.0 * (accuracy - float(reference["accuracy"]))
            for name, reference in base.EXISTING_BASELINES.items()
        },
    }
    base.harness._atomic_json(root / "summary.json", payload)
    return payload


def main() -> None:
    base.VARIANTS = (VARIANT,)
    base.SEEDS = SEEDS
    base._variant_config = _variant_config
    base._build = lambda _variant, config: base.ComplexScanBackbone(
        _variant_config(config)
    )
    base._contract = _contract
    base._summarize = _summarize
    base.main()


if __name__ == "__main__":
    main()
