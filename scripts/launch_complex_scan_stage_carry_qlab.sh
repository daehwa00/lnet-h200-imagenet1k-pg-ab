#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 )); then
  echo "usage: $0 <gpu-index> <variant> [<variant> ...]" >&2
  exit 2
fi

gpu_index=$1
shift
variants=("$@")
campaign=/home/qlab/experiments/alphabet/complex-scan-stage-carry-cifar100-20260802
runtime="$campaign/runtime"
root="$campaign/screen50"
python_bin=/home/qlab/.conda/envs/lnet-paper-cu128/bin/python
variant_label=$(IFS=-; echo "${variants[*]}")
log="$campaign/logs/gpu${gpu_index}-${variant_label}-seed401.log"

mkdir -p "$(dirname "$log")"
cd "$runtime"
exec env CUDA_VISIBLE_DEVICES="$gpu_index" PYTHONPATH=src "$python_bin" -u \
  scripts/run_complex_scan_stage_carry_cifar100.py \
  --root "$root" \
  --data-root /home/qlab/data \
  --variants "${variants[@]}" \
  --run-seeds 401 \
  --epochs 50 \
  --batch-size 256 \
  --workers 4 \
  --compile-model \
  --skip-test \
  >>"$log" 2>&1
