#!/usr/bin/env bash
set -euo pipefail

export H200_K_FAMILY_CAMPAIGN_DIR=k_family_depth_controls
export H200_K_FAMILY_RUNTIME_SCHEMA=lnet.h200.imagenet100.k_family_depth_controls.runtime.v1
export H200_K_FAMILY_PROGRAM=h200/run_imagenet100_k_family_depth_controls.sh
export H200_K_FAMILY_VARIANT_COUNT=5
export H200_K_FAMILY_CONTROL_REF=refs/heads/control/imagenet100-k-family-depth-controls
export H200_K_FAMILY_CONTROL_PATH=h200/k_family_depth_controls/control.json
export H200_K_FAMILY_WANDB_RUNTIME_ENV=H200_K_FAMILY_DEPTH_CONTROLS_WANDB_RUNTIME
export H200_K_FAMILY_QUEUE_SCRIPT=scripts/run_h200_k_family_depth_controls_queue.py
export H200_K_FAMILY_SMOKE_SCRIPT=scripts/smoke_h200_k_family_depth_controls.py

exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_imagenet100_k_family_xl.sh"
