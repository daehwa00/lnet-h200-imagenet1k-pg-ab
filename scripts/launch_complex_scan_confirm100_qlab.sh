#!/usr/bin/env bash
set -euo pipefail

if (( $# != 2 )); then
  echo "usage: $0 <gpu-index> <seed>" >&2
  exit 2
fi

gpu_index=$1
seed=$2
campaign=/home/qlab/experiments/alphabet/complex-scan-cifar100-20260802
runtime="$campaign/runtime"
root="$campaign/confirm100"
python_bin=/home/qlab/.conda/envs/lnet-paper-cu128/bin/python
log="$campaign/logs/complex-linear-confirm100-seed${seed}.log"

mkdir -p "$(dirname "$log")"
cd "$runtime"
exec env CUDA_VISIBLE_DEVICES="$gpu_index" PYTHONPATH=src "$python_bin" -u \
  scripts/run_complex_scan_cifar100.py \
  --root "$root" \
  --data-root /home/qlab/data \
  --variants complex_linear \
  --run-seeds "$seed" \
  --epochs 100 \
  --batch-size 256 \
  --workers 4 \
  --compile-model \
  --skip-test \
  >>"$log" 2>&1
