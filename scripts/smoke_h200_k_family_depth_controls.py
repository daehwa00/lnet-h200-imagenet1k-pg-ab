#!/usr/bin/env python3
"""Bind the representative H200 smoke to XL-K96-P128x4-D2282."""

from __future__ import annotations

# pyright: reportImplicitRelativeImport=false, reportPrivateLocalImportUsage=false
# pyright: reportPrivateUsage=false
import run_a2d_r2k3_k_family_depth_controls_imagenet100 as experiment
import smoke_h200_k_family_xl as shared


def _configure_shared_smoke() -> None:
    shared.VARIANT = experiment.XL_D2282
    shared.EXPECTED_PARAMETERS = 3_249_124
    shared.SMOKE_SCHEMA = "lnet.h200.imagenet100.k_family_depth_controls.smoke.v1"
    shared.experiment = experiment
    shared.family = experiment


def main() -> None:
    _configure_shared_smoke()
    shared.main()


if __name__ == "__main__":
    main()
