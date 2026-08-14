#!/usr/bin/env bash
set -euo pipefail

gpu="${1:-0}"
campaign=/home/daehwa/experiments/alphabet/complex-scan-wide-cifar100-20260802
runtime="$campaign/runtime"
root="$campaign/screen50"
python_bin=/home/daehwa/anaconda3/envs/alphabet/bin/python
runner="$runtime/scripts/run_complex_scan_wide_cifar100.py"
data_root=/home/daehwa/data
log="$campaign/logs/complex-amp-phase-wide-seed401.log"

mkdir -p "$(dirname "$log")"
cd "$runtime"
exec env CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src "$python_bin" -u "$runner" \
  --root "$root" \
  --data-root "$data_root" \
  --variants complex_amp_phase_wide \
  --run-seeds 401 \
  --epochs 50 \
  --batch-size 256 \
  --workers 4 \
  --compile-model \
  --skip-test \
  >>"$log" 2>&1
