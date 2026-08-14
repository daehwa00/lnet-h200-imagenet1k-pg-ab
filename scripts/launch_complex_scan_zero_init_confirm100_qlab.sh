#!/usr/bin/env bash
set -euo pipefail

if (( $# < 3 )); then
  echo "usage: $0 <gpu-index> <lane-label> <seed> [<seed> ...]" >&2
  exit 2
fi

gpu_index=$1
lane_label=$2
shift 2
seeds=("$@")
for seed in "${seeds[@]}"; do
  case "$seed" in
    401|402|403) ;;
    *)
      echo "unsupported seed: $seed" >&2
      exit 2
      ;;
  esac
done

campaign=/home/qlab/experiments/alphabet/complex-scan-zero-init-confirm100-cifar100-20260802
runtime="$campaign/runtime"
root="$campaign/confirm100/$lane_label"
python_bin=/home/qlab/.conda/envs/lnet-paper-cu128/bin/python
log="$campaign/logs/${lane_label}.log"
smoke_log="$campaign/logs/${lane_label}-smoke.log"

mkdir -p "$(dirname "$log")"
cd "$runtime"
env CUDA_VISIBLE_DEVICES="$gpu_index" PYTHONPATH=src "$python_bin" -u \
  scripts/smoke_complex_scan_zero_init_confirm.py \
  >>"$smoke_log" 2>&1
exec env CUDA_VISIBLE_DEVICES="$gpu_index" PYTHONPATH=src "$python_bin" -u \
  scripts/run_complex_scan_stage_carry_cifar100.py \
  --root "$root" \
  --data-root /home/qlab/data \
  --variants s2d_pole_main \
  --run-seeds "${seeds[@]}" \
  --epochs 100 \
  --batch-size 256 \
  --workers 4 \
  --compile-model \
  --skip-test \
  >>"$log" 2>&1
