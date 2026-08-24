#!/usr/bin/env bash
set -euo pipefail

export H200_K_FAMILY_CAMPAIGN_DIR=k_family_p_refinement
export H200_K_FAMILY_RUNTIME_SCHEMA=lnet.h200.imagenet100.k_family_p_refinement.runtime.v1
export H200_K_FAMILY_PROGRAM=h200/run_imagenet100_k_family_p_refinement.sh
export H200_K_FAMILY_VARIANT_COUNT=5
export H200_K_FAMILY_CONTROL_REF=refs/heads/control/imagenet100-k-family-p-refinement
export H200_K_FAMILY_CONTROL_PATH=h200/k_family_p_refinement/control.json
export H200_K_FAMILY_WANDB_RUNTIME_ENV=H200_K_FAMILY_P_REFINEMENT_WANDB_RUNTIME
export H200_K_FAMILY_QUEUE_SCRIPT=scripts/run_h200_k_family_p_refinement_queue.py

exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_imagenet100_k_family_xl.sh"
