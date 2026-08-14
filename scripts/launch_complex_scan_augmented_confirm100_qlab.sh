#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 )); then
  echo "usage: $0 <gpu-index> <seed> [<seed> ...]" >&2
  exit 2
fi

gpu_index=$1
shift
seeds=("$@")
campaign=/home/qlab/experiments/alphabet/complex-scan-augmented-complex-cifar100-20260802
runtime="$campaign/runtime"
root="$campaign/confirm100"
python_bin=/home/qlab/.conda/envs/lnet-paper-cu128/bin/python
seed_label=$(IFS=-; echo "${seeds[*]}")
log="$campaign/logs/augmented-confirm100-gpu${gpu_index}-seeds${seed_label}.log"

mkdir -p "$(dirname "$log")"
cd "$runtime"
exec env CUDA_VISIBLE_DEVICES="$gpu_index" PYTHONPATH=src "$python_bin" -u \
  scripts/run_complex_scan_augmented_cifar100.py \
  --root "$root" \
  --data-root /home/qlab/data \
  --variants augmented_complex_ffn \
  --run-seeds "${seeds[@]}" \
  --epochs 100 \
  --batch-size 256 \
  --workers 4 \
  --compile-model \
  --skip-test \
  >>"$log" 2>&1
