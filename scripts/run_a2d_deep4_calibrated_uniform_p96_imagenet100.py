#!/usr/bin/env python3
"""Train the calibrated product-only Deep4 model with 96 modes per stage."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import run_a2d_deep4_calibrated_pole_ramp_imagenet100 as base

if TYPE_CHECKING:
    from argparse import Namespace
    from collections.abc import Iterator


VARIANT = "D4-Cal-U96"
VARIANTS = (VARIANT,)
SEEDS = (501,)
MODES = 96
STAGE_MODES = (MODES, MODES, MODES, MODES)
SPEC = base.backbone.Deep4BackboneSpec(
    modes=STAGE_MODES,
    stem_width=2 * MODES,
    mode_cffn_widths=(2 * MODES, 2 * MODES, 2 * MODES),
    augmented_widths=(2 * MODES, 2 * MODES, 2 * MODES),
    post_ffn_widths=(2 * MODES, 2 * MODES, 2 * MODES),
)
_ORIGINAL_CONTRACT = base._contract


def _configure_base() -> None:
    """Point the established calibrated runner at the uniform-96 spec."""
    base.VARIANT = VARIANT
    base.VARIANTS = VARIANTS
    base.SEEDS = SEEDS
    base.BASE_POLES = MODES
    base.STAGE_MODES = STAGE_MODES
    base.SPEC = SPEC


@contextmanager
def _temporary_base_config() -> Iterator[None]:
    previous = {
        "VARIANT": base.VARIANT,
        "VARIANTS": base.VARIANTS,
        "SEEDS": base.SEEDS,
        "BASE_POLES": base.BASE_POLES,
        "STAGE_MODES": base.STAGE_MODES,
        "SPEC": base.SPEC,
    }
    _configure_base()
    try:
        yield
    finally:
        for name, value in previous.items():
            setattr(base, name, value)


def _build(variant: str, config: base.PoleModelConfig) -> base.nn.Module:
    with _temporary_base_config():
        return base._build(variant, config)


def _variant_config() -> dict[str, Any]:
    with _temporary_base_config():
        payload = base._variant_config()
    payload["backbone"]["name"] = "A2D-Calibrated-Product4-Uniform96-FullOpt"
    return payload


def _contract(args: Namespace) -> dict[str, Any]:
    with _temporary_base_config():
        payload = _ORIGINAL_CONTRACT(args)
        variant_config = base._variant_config()
    variant_config["backbone"]["name"] = "A2D-Calibrated-Product4-Uniform96-FullOpt"
    payload["schema"] = "lnet.a2d.deep4_calibrated_uniform_p96.imagenet100.v1"
    payload["evidence_status"] = "untrained calibrated uniform-96 FullOpt candidate"
    payload["variant_configs"] = {VARIANT: variant_config}
    payload["architecture"] = {
        VARIANT: (
            "The calibrated product-only Deep4 model with 96 pole modes at all "
            "four stages, canonical eight-angle radial groups, stage-matched "
            "mode CFFNs, and the existing width-generic optimized kernels."
        )
    }
    payload["source_sha256"]["a2d_deep4_calibrated_uniform_p96_runner"] = (
        base.heads.harness._digest(Path(__file__))
    )
    return payload


def main() -> None:
    _configure_base()
    base._contract = _contract
    base.main()


if __name__ == "__main__":
    main()
