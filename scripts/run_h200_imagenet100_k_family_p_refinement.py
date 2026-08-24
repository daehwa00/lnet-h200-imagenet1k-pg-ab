#!/usr/bin/env python3
"""Bind the shared managed H200 worker to the P-refinement campaign."""

from __future__ import annotations

# pyright: reportImplicitRelativeImport=false, reportPrivateLocalImportUsage=false
# pyright: reportPrivateUsage=false
import run_a2d_r2k3_k_family_p_refinement_imagenet100 as experiment
import run_h200_imagenet100_k_family_xl as shared

RUNTIME_SCHEMA = "lnet.h200.imagenet100.k_family_p_refinement.runtime.v1"
RUNTIME_ENV_VAR = "H200_K_FAMILY_P_REFINEMENT_WANDB_RUNTIME"
HEARTBEAT_SCHEMA = "lnet.h200.imagenet100.k_family_p_refinement.heartbeat.v1"
PARAMETER_COUNTS = {
    "M-K48-P80-80-80-80": 857_124,
    "L-K64-P80-96-80-80": 1_290_308,
    "L-K64-P80-96-96-80": 1_339_652,
    "S-K32-P48-48-64-48": 430_404,
    "XL-K96-P128-144-128-128": 2_814_116,
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
