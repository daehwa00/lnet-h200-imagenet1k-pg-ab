# ruff: noqa: SLF001
# pyright: reportAny=false, reportExplicitAny=false, reportImplicitRelativeImport=false
# pyright: reportPrivateLocalImportUsage=false, reportPrivateUsage=false
"""Run the stride-1 RGB fusion and local-complex-Fourier complex scan backbone."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import run_complex_scan_zero_init_imagenet100 as base

if TYPE_CHECKING:
    from argparse import Namespace

    import torch

    from lnet.complex_scan import ComplexScanConfig

VARIANT = "local_fourier_residual_dual_fusion256_lrq64"
SEEDS = (501,)
_BASE_VARIANT_CONFIG = base._variant_config
_BASE_CONTRACT = base._contract
_BASE_PREPARE_MODEL = base._BASE_PREPARE_MODEL


def _variant_config(config: ComplexScanConfig) -> ComplexScanConfig:
    return replace(
        _BASE_VARIANT_CONFIG(config),
        stem="local_fourier",
        stem_strides=(1, 4),
        carry_bases=("none", "none"),
        use_pole_aligned_shortcuts=True,
        stage_residual_scale_initial=0.1,
        fusion_width=256,
        dual_fusion_lrq_head=True,
    )


def _prepare_model(model: torch.nn.Module, recipe: dict[str, Any]) -> torch.nn.Module:
    # The local Fourier stem directly produces complex modes, so the old real
    # analysis matrix (and its capture-safe orthogonal parametrization) is absent.
    return _BASE_PREPARE_MODEL(model, recipe)


def _contract(args: Namespace) -> dict[str, object]:
    payload = _BASE_CONTRACT(args)
    payload["schema"] = "lnet.complex_scan.local_fourier.imagenet100.v1"
    payload["evidence_status"] = "single-seed 100-epoch architecture screen"
    architecture = payload["architecture"]
    if not isinstance(architecture, dict):
        message = "local Fourier contract architecture is malformed"
        raise TypeError(message)
    architecture.update(
        {
            "stem": (
                "3x3 stride-1 RGB fusion, LayerNorm2d, GELU, then selected "
                "8x8 Hann-windowed complex Fourier modes at hop 4"
            ),
            "stem_output": "32 complex modes on a 56x56 grid",
            "stage_shortcut": "pole-aligned complex residual downsample",
            "legacy_stage_carry": "disabled",
            "head": "Fusion256 + initially disabled parallel LRQ64",
        }
    )
    sources = payload["source_sha256"]
    if not isinstance(sources, dict):
        message = "local Fourier contract source payload is malformed"
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
    base._prepare_model = _prepare_model
    base._contract = _contract
    base._summarize = _summarize
    base.main()


if __name__ == "__main__":
    main()
