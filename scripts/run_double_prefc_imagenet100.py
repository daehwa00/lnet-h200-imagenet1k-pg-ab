#!/usr/bin/env python3
# pyright: reportAny=false, reportExplicitAny=false, reportImplicitRelativeImport=false
# pyright: reportPrivateLocalImportUsage=false, reportPrivateUsage=false
"""Run the retained Double-PreFC ImageNet-100 variants."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import complex_scan_imagenet100_runtime as residuals
import run_alphabet2d_imagenet100_nano as harness
from torch import Tensor, nn

from lnet.complex_scan import (
    ComplexField,
    ComplexScanBackbone,
    ComplexScanConfig,
)

if TYPE_CHECKING:
    from argparse import Namespace


DOUBLE_PREFC = "double_prefc"
NO_NORM = "double_prefc_no_norm"
A2D = "a2d"
A2D_NO_NORM = "a2d_no_norm"
VARIANTS = (DOUBLE_PREFC, NO_NORM, A2D, A2D_NO_NORM)
REFERENCE_VARIANT = residuals.REFERENCE_VARIANT

_reference_variant_config = residuals._variant_config
_reference_contract = residuals._contract


class ComplexIdentity(nn.Module):
    """Two-input identity matching the complex normalization interface."""

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        return real, imag


def _variant_config(
    variant: str,
    config: ComplexScanConfig,
) -> ComplexScanConfig:
    if variant not in VARIANTS:
        return _reference_variant_config(variant, config)
    return _reference_variant_config(REFERENCE_VARIANT, config)


def _remove_stage1_ffn_norm(model: ComplexScanBackbone) -> ComplexScanBackbone:
    if model.stage1.augmented is None:
        message = "Double-PreFC stage 1 is missing its augmented transition"
        raise RuntimeError(message)
    cast("Any", model.stage1.augmented).ffn_norm = ComplexIdentity()
    return model


def _remove_final_precomplex_gelu(model: ComplexScanBackbone) -> ComplexScanBackbone:
    if not isinstance(model.precomplex_fc, nn.Sequential):
        message = "Double-PreFC is missing its pre-complex projection"
        raise TypeError(message)
    layers = list(model.precomplex_fc.children())
    if (
        len(layers) != 4
        or not isinstance(layers[0], nn.Linear)
        or not isinstance(layers[1], nn.GELU)
        or not isinstance(layers[2], nn.Linear)
        or not isinstance(layers[3], nn.GELU)
    ):
        message = "Double-PreFC does not have the expected Linear-GELU-Linear-GELU layout"
        raise RuntimeError(message)
    model.precomplex_fc = nn.Sequential(layers[0], layers[1], layers[2], nn.Identity())
    return model


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    model = ComplexScanBackbone(_variant_config(variant, config))
    if variant in {NO_NORM, A2D_NO_NORM}:
        _remove_stage1_ffn_norm(model)
    if variant in {A2D, A2D_NO_NORM}:
        _remove_final_precomplex_gelu(model)
    return model


def _contract(args: Namespace) -> dict[str, Any]:
    payload = _reference_contract(args)
    config = ComplexScanConfig(
        output_dim=100,
        stem_strides=(2, 2),
    )
    architecture = {
        DOUBLE_PREFC: "the matched associative D4 Double-PreFC model",
        NO_NORM: "Double-PreFC without the first augmented-transition ffn_norm",
        A2D: "A2D",
        A2D_NO_NORM: "A2D without the first augmented-transition ffn_norm",
    }
    for variant in VARIANTS:
        active = _variant_config(variant, config)
        model = _build(variant, config)
        payload["variant_configs"][variant] = asdict(active)
        if variant in {NO_NORM, A2D_NO_NORM}:
            payload["variant_configs"][variant]["stage1_ffn_norm"] = False
        if variant in {A2D, A2D_NO_NORM}:
            payload["variant_configs"][variant]["precomplex_fc_widths"] = [96, 96, 96]
            payload["variant_configs"][variant]["precomplex_fc_activations"] = [
                "gelu",
                "identity",
            ]
        payload["parameter_counts"][variant] = sum(
            parameter.numel() for parameter in model.parameters()
        )
        payload["architecture"][variant] = architecture[variant]
    payload["source_sha256"]["double_prefc_runner"] = harness._digest(Path(__file__))
    return payload


def main() -> None:
    residuals.main(
        variants=VARIANTS,
        build_model=_build,
        contract=_contract,
    )


if __name__ == "__main__":
    main()
