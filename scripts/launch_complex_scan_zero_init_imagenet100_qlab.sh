#!/usr/bin/env bash
set -euo pipefail

if (( $# < 3 )); then
  echo "usage: $0 <smoke|train> <gpu-index> <lane> [<seed> ...]" >&2
  exit 2
fi

phase=$1
gpu_index=$2
lane=$3
shift 3
seeds=("$@")
campaign=${ALPHABET_CAMPAIGN:-/home/qlab/experiments/alphabet/complex-scan-zero-init-m32h64-lrq64-imagenet100-optimized-20260802}
runtime="$campaign/runtime"
python_bin=/home/qlab/.conda/envs/lnet-paper-cu128/bin/python
runner=${ALPHABET_RUNNER:-$runtime/scripts/run_complex_scan_zero_init_imagenet100.py}
data_root=/home/qlab/data/ImageNet100
log_root="$campaign/logs"
mkdir -p "$log_root"

record_failure() {
  local status="$?"
  printf '%s exit=%s phase=%s lane=%s\n' "$(date -Is)" "$status" "$phase" "$lane" \
    >"$campaign/${phase}-${lane}.failed"
  if [[ "$phase" == "smoke" ]]; then
    touch "$campaign/smoke.failed"
  fi
  exit "$status"
}
trap record_failure ERR

run_with_retry() {
  local root=$1
  local epochs=$2
  local log=$3
  local attempt
  for attempt in 1 2 3; do
    if (
      cd "$runtime"
      CUDA_VISIBLE_DEVICES="$gpu_index" \
        PYTHONPATH=src \
        TORCHINDUCTOR_CACHE_DIR="$campaign/torchinductor-gpu${gpu_index}" \
        TORCHINDUCTOR_COMPILE_THREADS=4 \
        TORCHINDUCTOR_EMULATE_PRECISION_CASTS=1 \
        "$python_bin" "$runner" \
        --root "$root" \
        --data-root "$data_root" \
        --variants s2d_pole_main_zero_init \
        --run-seeds "${seeds[@]}" \
        --epochs "$epochs" \
        --batch-size 256 \
        --workers 8 \
        --precision bfloat16
    ) >>"$log" 2>&1; then
      return 0
    fi
    printf '%s attempt=%s failed; checkpoint resume will retry\n' \
      "$(date -Is)" "$attempt" >>"$log"
    sleep 5
  done
  return 1
}

if [[ "$phase" == "smoke" ]]; then
  if (( ${#seeds[@]} != 1 )); then
    echo "smoke requires exactly one seed" >&2
    exit 2
  fi
  smoke_root="$campaign/smoke1"
  smoke_log="$log_root/smoke-seed${seeds[0]}.log"
  run_with_retry "$smoke_root" 1 "$smoke_log"
  SMOKE_ROOT="$smoke_root" "$python_bin" - <<'PY'
import json
import math
import os
from pathlib import Path

path = Path(os.environ["SMOKE_ROOT"]) / "results" / "s2d_pole_main_zero_init__seed501.json"
row = json.loads(path.read_text())
contract = json.loads((Path(os.environ["SMOKE_ROOT"]) / "contract.json").read_text())
expected_parameters = contract["parameter_counts"]["s2d_pole_main_zero_init"]
if row["parameters"] != expected_parameters:
    raise RuntimeError("ImageNet-100 smoke parameter count changed")
if len(row["history"]) != 1:
    raise RuntimeError("ImageNet-100 smoke did not complete exactly one epoch")
values = (
    row["final_validation"]["accuracy"],
    row["final_validation"]["cross_entropy"],
    row["history"][0]["train_loss"],
)
if not all(math.isfinite(float(value)) for value in values):
    raise RuntimeError("ImageNet-100 smoke produced a non-finite metric")
PY
  touch "$campaign/smoke.passed"
  exit 0
fi

if [[ "$phase" != "train" || ${#seeds[@]} == 0 ]]; then
  echo "train requires at least one seed" >&2
  exit 2
fi
while [[ ! -f "$campaign/smoke.passed" ]]; do
  if [[ -f "$campaign/smoke.failed" ]]; then
    echo "ImageNet-100 smoke failed; refusing to start training" >&2
    exit 1
  fi
  sleep 15
done
run_with_retry "$campaign/run100" 100 "$log_root/train-${lane}.log"
printf '%s seeds=%s\n' "$(date -Is)" "${seeds[*]}" >"$campaign/train-${lane}.done"
