#!/usr/bin/env bash
set -euo pipefail

if (( $# != 3 )); then
  echo "usage: $0 <gpu-index> <variant> <seed>" >&2
  exit 2
fi

gpu_index=$1
variant=$2
seed=$3
campaign=/home/qlab/experiments/alphabet/complex-scan-augmented-complex-cifar100-20260802
runtime="$campaign/runtime"
root="$campaign/screen50"
python_bin=/home/qlab/.conda/envs/lnet-paper-cu128/bin/python
log="$campaign/logs/${variant}-seed${seed}.log"

mkdir -p "$(dirname "$log")"
cd "$runtime"
exec env CUDA_VISIBLE_DEVICES="$gpu_index" PYTHONPATH=src "$python_bin" -u \
  scripts/run_complex_scan_augmented_cifar100.py \
  --root "$root" \
  --data-root /home/qlab/data \
  --variants "$variant" \
  --run-seeds "$seed" \
  --epochs 50 \
  --batch-size 256 \
  --workers 4 \
  --compile-model \
  --skip-test \
  >>"$log" 2>&1
