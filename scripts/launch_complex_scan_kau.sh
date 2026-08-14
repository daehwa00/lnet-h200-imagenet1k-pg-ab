#!/usr/bin/env bash
set -euo pipefail

gpu="${1:-0}"
campaign=/home/daehwa/experiments/alphabet/complex-scan-cifar100-20260802
runtime="$campaign/runtime"
root="$campaign/screen50"
python_bin=/home/daehwa/anaconda3/envs/alphabet/bin/python
runner="$runtime/scripts/run_complex_scan_cifar100.py"
data_root=/home/daehwa/data
log_root="$campaign/logs"

mkdir -p "$log_root"

run_one() {
  local variant="$1"
  local log="$log_root/${variant}-seed401.log"
  (
    cd "$runtime"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src "$python_bin" "$runner" \
      --root "$root" \
      --data-root "$data_root" \
      --variants "$variant" \
      --run-seeds 401 \
      --epochs 50 \
      --batch-size 256 \
      --workers 4 \
      --compile-model \
      --skip-test
  ) >>"$log" 2>&1
}

run_one complex_linear

date -Is >"$campaign/screen50.done"
