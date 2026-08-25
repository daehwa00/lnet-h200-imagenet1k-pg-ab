#!/usr/bin/env python3
"""Bind the shared managed H200 worker to the depth-control campaign."""

from __future__ import annotations

# pyright: reportImplicitRelativeImport=false, reportPrivateLocalImportUsage=false
# pyright: reportPrivateUsage=false
import run_a2d_r2k3_k_family_depth_controls_imagenet100 as experiment
import run_h200_imagenet100_k_family_xl as shared

RUNTIME_SCHEMA = "lnet.h200.imagenet100.k_family_depth_controls.runtime.v1"
RUNTIME_ENV_VAR = "H200_K_FAMILY_DEPTH_CONTROLS_WANDB_RUNTIME"
HEARTBEAT_SCHEMA = "lnet.h200.imagenet100.k_family_depth_controls.heartbeat.v1"
PARAMETER_COUNTS = {
    "S-K32-P48x4-D2262": 399_492,
    "S-K32-P48x4-D2242": 337_732,
    "L-K64-P80x4-D2262": 1_273_860,
    "L-K64-P80x4-D2282": 1_479_236,
    "XL-K96-P128x4-D2282": 3_249_124,
}


def _configure_shared_worker() -> None:
    shared.RUNTIME_SCHEMA = RUNTIME_SCHEMA
    shared.RUNTIME_ENV_VAR = RUNTIME_ENV_VAR
    shared.HEARTBEAT_SCHEMA = HEARTBEAT_SCHEMA
    shared.VARIANTS = experiment.VARIANTS
    shared.PARAMETER_COUNTS = PARAMETER_COUNTS
    shared.experiment = experiment
    shared._ORIGINAL_BUILD = experiment._build


def main() -> None:
    _configure_shared_worker()
    shared.main()


if __name__ == "__main__":
    main()
