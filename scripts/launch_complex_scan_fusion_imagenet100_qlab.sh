#!/usr/bin/env bash
set -euo pipefail

campaign=/home/qlab/experiments/alphabet/complex-scan-fusion256-m32h64-imagenet100-optimized-20260802
export ALPHABET_CAMPAIGN="$campaign"
export ALPHABET_RUNNER="$campaign/runtime/scripts/run_complex_scan_fusion_imagenet100.py"
exec "$campaign/runtime/scripts/launch_complex_scan_zero_init_imagenet100_qlab.sh" "$@"
