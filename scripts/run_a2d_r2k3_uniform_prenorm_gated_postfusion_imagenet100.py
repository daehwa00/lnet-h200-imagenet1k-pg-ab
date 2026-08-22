#!/usr/bin/env python3
"""Train uniform-width R2K3 controls with pre-norm readers and gated PostFusion."""

from __future__ import annotations

# pyright: reportArgumentType=false, reportExplicitAny=false
# pyright: reportImplicitRelativeImport=false, reportPrivateUsage=false
# pyright: reportImplicitStringConcatenation=false
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import a2d_r2k3_capacity_factory as capacity
import a2d_r2k3_runtime as runtime

from lnet.pac_factorized_complex_scan_reader import FactorizedComplexConv2dReader
from lnet.pac_gated_post_fusion import (
    GatedComplexPostFusion,
    GatedPoleExcitationS2DTransition,
)

if TYPE_CHECKING:
    from argparse import Namespace

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig


WIDTHS = (64, 96, 128)
VARIANTS = tuple(f"R2K3-K{width}-P{width}x4-PreNorm-GatedPostFusion" for width in WIDTHS)
SEEDS = runtime.DEFAULT_SEEDS
STAGE_NAMES = runtime.STAGE_NAMES
SPECS = {
    variant: capacity.CapacitySpec(
        excitation_modes=(width,) * 4,
        pole_modes=(width,) * 4,
        post_hidden_ratio=2.0,
    )
    for variant, width in zip(VARIANTS, WIDTHS, strict=True)
}


def _pre_norm_reader(
    source: FactorizedComplexConv2dReader,
) -> FactorizedComplexConv2dReader:
    reader = FactorizedComplexConv2dReader(
        source.input_modes,
        source.output_modes,
        rank=source.rank,
        kernel_size=source.kernel_size,
        variance_epsilon=source.variance_epsilon,
        normalize_input=True,
        match_input_rms=False,
    )
    incompatible = reader.load_state_dict(source.state_dict(), strict=False)
    if incompatible.missing_keys != ["input_norm.weight"] or incompatible.unexpected_keys:
        message = "pre-norm reader failed to preserve the initialized R2K3 weights"
        raise RuntimeError(message)
    return reader


def _install_combined_transition(model: ComplexScanBackbone) -> None:
    for name in STAGE_NAMES[:3]:
        stage = getattr(model, name)
        reader = stage.pole_input_projection
        if type(reader) is not FactorizedComplexConv2dReader:
            message = f"{name} does not expose the established R2K3 reader"
            raise TypeError(message)
        stage.pole_input_projection = _pre_norm_reader(reader)

        source = stage.augmented
        if source is None:
            message = f"{name} does not expose the established transition"
            raise TypeError(message)
        stage.augmented = GatedPoleExcitationS2DTransition(
            source.input_modes,
            source.excitation_modes,
            source.output_modes,
            post_hidden=source.post_hidden,
        )


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    try:
        spec = SPECS[variant]
    except KeyError as error:
        message = f"unsupported uniform combined variant: {variant}"
        raise ValueError(message) from error

    model = capacity._build_spec(spec, config)
    _install_combined_transition(model)
    _assert_model(model, variant)
    return model


def _assert_model(model: ComplexScanBackbone, variant: str) -> None:
    spec = SPECS[variant]
    if model.descriptor_dim != spec.descriptor_dim:
        message = f"{variant} changed its raw descriptor width"
        raise RuntimeError(message)

    for index, name in enumerate(STAGE_NAMES):
        width = spec.excitation_modes[index]
        stage = getattr(model, name)
        reader = stage.pole_input_projection
        if (
            type(reader) is not FactorizedComplexConv2dReader
            or reader.input_modes != width
            or reader.output_modes != width
        ):
            message = f"{variant}/{name} changed its uniform reader width"
            raise RuntimeError(message)

        if name == "terminal":
            if (
                reader.normalize_input
                or reader.input_norm is not None
                or not reader.match_input_rms
            ):
                message = f"{variant}/terminal changed the established reader contract"
                raise RuntimeError(message)
            continue

        if not reader.normalize_input or reader.input_norm is None or reader.match_input_rms:
            message = f"{variant}/{name} lost its pre-norm reader contract"
            raise RuntimeError(message)
        transition = stage.augmented
        if (
            type(transition) is not GatedPoleExcitationS2DTransition
            or transition.input_modes != width
            or transition.excitation_modes != width
            or transition.output_modes != width
            or transition.post_hidden != 2 * width
            or type(transition.post_fusion) is not GatedComplexPostFusion
        ):
            message = f"{variant}/{name} lost its gated PostFusion contract"
            raise RuntimeError(message)


def _variant_config(variant: str) -> dict[str, Any]:
    spec = SPECS[variant]
    payload = deepcopy(capacity._variant_config_for_spec(variant, spec))
    payload["backbone"]["name"] = f"A2D-{variant}"
    payload["backbone"]["pole_input"]["normalization"] = (
        "stage1-3 learned CRMSNorm before unit-row-energy R2K3; no RMSMatch; "
        "terminal unit-row-energy R2K3 plus token RMSMatch unchanged"
    )
    payload["backbone"]["transition"]["post_fusion"] = (
        "residual CRMSNorm; bias-free WL value K-to-2K; bias-free real Linear "
        "gate [real,imag]-to-2K with SiLU; bias-free WL output 2K-to-K"
    )
    return payload


def _selected_variants(args: Namespace) -> tuple[str, ...]:
    selected = getattr(args, "variants", None)
    if not selected:
        return VARIANTS
    if isinstance(selected, str):
        return (selected,)
    return tuple(cast("list[str] | tuple[str, ...]", selected))


def _contract(args: Namespace) -> dict[str, Any]:
    payload = runtime.base_contract(args)
    selected = _selected_variants(args)
    config = runtime.model_config()
    payload["schema"] = "lnet.a2d.r2k3.uniform_prenorm_gated_postfusion.v1"
    payload["evidence_status"] = "candidate-specific CPU and compiled CUDA smoke required"
    payload["variants"] = list(selected)
    payload["seeds"] = list(SEEDS)
    payload["variant_configs"] = {variant: _variant_config(variant) for variant in selected}
    payload["parameter_counts"] = {
        variant: sum(parameter.numel() for parameter in _build(variant, config).parameters())
        for variant in selected
    }
    payload["architecture"] = dict.fromkeys(
        selected,
        (
            "Uniform K=P R2K3 Raw-Q Orth NoPG capacity control with learned CRMSNorm "
            "before each nonterminal reader and real-SiLU-gated complex PostFusion; "
            "terminal reader, Q4 affine head, optimizer, and recipe remain unchanged."
        ),
    )
    payload["source_sha256"]["uniform_combined_runner"] = runtime.digest(Path(__file__))
    payload["source_sha256"]["r2k3_runtime"] = runtime.digest(Path("scripts/a2d_r2k3_runtime.py"))
    payload["source_sha256"]["r2k3_capacity_factory"] = runtime.digest(
        Path("scripts/a2d_r2k3_capacity_factory.py")
    )
    payload["source_sha256"]["gated_post_fusion"] = runtime.digest(
        Path("src/lnet/pac_gated_post_fusion.py")
    )
    payload["source_sha256"]["factorized_complex_scan_reader"] = runtime.digest(
        Path("src/lnet/pac_factorized_complex_scan_reader.py")
    )
    return payload


def main() -> None:
    runtime.run(
        variants=VARIANTS,
        seeds=SEEDS,
        build_model=_build,
        contract=_contract,
    )


if __name__ == "__main__":
    main()
