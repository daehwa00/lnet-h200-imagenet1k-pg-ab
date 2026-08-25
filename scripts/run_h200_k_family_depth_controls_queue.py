#!/usr/bin/env python3
"""Bind the generic persistent queue to the five depth-control cells."""

from __future__ import annotations

# pyright: reportImplicitRelativeImport=false, reportPrivateLocalImportUsage=false
import run_a2d_r2k3_k_family_depth_controls_imagenet100 as experiment
import run_h200_k_family_p_refinement_queue as shared

RUNTIME_SCHEMA = "lnet.h200.imagenet100.k_family_depth_controls.runtime.v1"
RUNTIME_ENV_VAR = "H200_K_FAMILY_DEPTH_CONTROLS_WANDB_RUNTIME"
PARAMETER_COUNTS = {
    "S-K32-P48x4-D2262": 399_492,
    "S-K32-P48x4-D2242": 337_732,
    "L-K64-P80x4-D2262": 1_273_860,
    "L-K64-P80x4-D2282": 1_479_236,
    "XL-K96-P128x4-D2282": 3_249_124,
}


def _configure_shared_queue() -> None:
    shared.RUNTIME_SCHEMA = RUNTIME_SCHEMA
    shared.RUNTIME_ENV_VAR = RUNTIME_ENV_VAR
    shared.VARIANTS = experiment.VARIANTS
    shared.PARAMETER_COUNTS = PARAMETER_COUNTS
    shared.WORKER_SCRIPT = "scripts/run_h200_imagenet100_k_family_depth_controls.py"
    shared.STATUS_FILENAME = "k-family-depth-controls-queue.json"


def main() -> int:
    _configure_shared_queue()
    return shared.main()


if __name__ == "__main__":
    raise SystemExit(main())
