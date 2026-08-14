#!/usr/bin/env python3
"""Run the Fusion256 ResAux model with affine auxiliary weight 0.5."""

# ruff: noqa: SLF001

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import run_a2d_resaux1_f256_imagenet100 as source
import run_alphabet2d_imagenet100_nano as harness

if TYPE_CHECKING:
    from argparse import Namespace


VARIANT = "A2D-ResAux05-F256"
AFFINE_AUXILIARY_WEIGHT = 0.5
_SOURCE_CONTRACT = source._contract


def _contract(args: Namespace) -> dict[str, Any]:
    payload = _SOURCE_CONTRACT(args)
    payload["schema"] = "lnet.a2d.resaux05_f256.imagenet100.v1"
    payload["evidence_status"] = "one-seed 100-epoch auxiliary-weight comparison"
    payload["variant_configs"][VARIANT]["head"]["affine_auxiliary_weight"] = (
        AFFINE_AUXILIARY_WEIGHT
    )
    payload["architecture"][VARIANT] = (
        "A2D-ResAux1 Fusion256 with affine Q auxiliary CE weight 0.5; "
        "all backbone, path-combination, residual, and optimizer settings unchanged."
    )
    payload["source_sha256"]["resaux05_f256_runner"] = (
        harness._digest(Path(__file__))
    )
    return json.loads(json.dumps(payload))


source.VARIANT = VARIANT
source.AFFINE_AUXILIARY_WEIGHT = AFFINE_AUXILIARY_WEIGHT
source._contract = _contract
_build = source._build


def main() -> None:
    source.main()


if __name__ == "__main__":
    main()
