#!/usr/bin/env python3
# pyright: reportAny=false, reportExplicitAny=false, reportImplicitRelativeImport=false
# pyright: reportPrivateLocalImportUsage=false, reportPrivateUsage=false
"""Train A2D-D4 with a learned four-product-path combiner."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import run_double_prefc_imagenet100 as baseline

from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig

if TYPE_CHECKING:
    from argparse import Namespace

VARIANT = "a2d_d4_pathmix"


def _variant_config(
    variant: str,
    config: ComplexScanConfig,
) -> ComplexScanConfig:
    if variant != VARIANT:
        return baseline._variant_config(variant, config)
    return replace(
        baseline._variant_config(baseline.A2D, config),
        quadrant_path_mode_cffn_widths=(96, 96),
        quadrant_path_cffn_widths=(16, 16),
    )


def _build(
    variant: str,
    config: ComplexScanConfig,
) -> ComplexScanBackbone:
    model = ComplexScanBackbone(_variant_config(variant, config))
    return baseline._remove_final_precomplex_gelu(model)


def _contract(args: Namespace) -> dict[str, Any]:
    payload = baseline._contract(args)
    base_config = ComplexScanConfig(
        output_dim=100,
        stem_strides=(2, 2),
    )
    active = _variant_config(VARIANT, base_config)
    model = _build(VARIANT, base_config)
    payload["variant_configs"][VARIANT] = asdict(active)
    payload["variant_configs"][VARIANT]["precomplex_fc_widths"] = [96, 96, 96]
    payload["variant_configs"][VARIANT]["precomplex_fc_activations"] = [
        "gelu",
        "identity",
    ]
    payload["variant_configs"][VARIANT]["stage1_ffn_norm"] = True
    payload["parameter_counts"][VARIANT] = sum(
        parameter.numel() for parameter in model.parameters()
    )
    payload["architecture"][VARIANT] = (
        "A2D-D4 with the four x-to-y product paths processed by a shared "
        "residual Cartesian-SiLU ModeCFFN 48-96-48 and PathCFFN 4-16-4, "
        "followed by learned complex mode-wise four-path quadrant "
        "synthesis; the existing raw directional descriptor, augmented transition, S2D carry, "
        "normalizations, descriptors, and dual head are retained"
    )
    payload["source_sha256"]["a2d_d4_pathmix_runner"] = baseline.harness._digest(Path(__file__))
    # The immutable contract is read back from JSON on resume.  Normalize
    # dataclass tuples now so an unchanged run compares equal after reload.
    return json.loads(json.dumps(payload))


def main() -> None:
    residuals = baseline.residuals
    residuals.main(
        variants=(VARIANT,),
        build_model=_build,
        contract=_contract,
    )


if __name__ == "__main__":
    main()
