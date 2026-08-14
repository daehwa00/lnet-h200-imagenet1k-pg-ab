# ruff: noqa: SLF001
# pyright: reportImplicitRelativeImport=false
# pyright: reportPrivateLocalImportUsage=false
# pyright: reportPrivateUsage=false
"""Run the 256-wide global descriptor-fusion complex scan backbone on ImageNet-100."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import run_complex_scan_zero_init_imagenet100 as base

if TYPE_CHECKING:
    from argparse import Namespace

    from lnet.complex_scan import ComplexScanConfig

FUSION_WIDTH = 256
_BASE_VARIANT_CONFIG = base._variant_config
_BASE_CONTRACT = base._contract


def _variant_config(config: ComplexScanConfig) -> ComplexScanConfig:
    return replace(_BASE_VARIANT_CONFIG(config), fusion_width=FUSION_WIDTH)


def _contract(args: Namespace) -> dict[str, object]:
    payload = _BASE_CONTRACT(args)
    payload["schema"] = "lnet.complex_scan.fusion.imagenet100.optimized.v1"
    payload["evidence_status"] = "queued 100-epoch three-seed fusion-head comparison"
    model = payload["model"]
    if not isinstance(model, dict):
        message = "fusion contract model payload is malformed"
        raise TypeError(message)
    model["fusion_width"] = FUSION_WIDTH
    architecture = payload["architecture"]
    if not isinstance(architecture, dict):
        message = "fusion contract architecture payload is malformed"
        raise TypeError(message)
    architecture["head"] = "BatchNorm -> Linear(384,256) -> GELU -> RMSNorm -> Linear(256,100)"
    architecture["audit_tradeoff"] = (
        "global nonlinear fusion replaces exact coordinate/rank-wise LRQ attribution"
    )
    sources = payload["source_sha256"]
    if not isinstance(sources, dict):
        message = "fusion contract source payload is malformed"
        raise TypeError(message)
    sources["runner"] = base.harness._digest(Path(__file__))
    return json.loads(json.dumps(payload))


def main() -> None:
    base._variant_config = _variant_config
    base._contract = _contract
    base.main()


if __name__ == "__main__":
    main()
