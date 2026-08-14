#!/usr/bin/env bash
set -euo pipefail

if (( $# < 5 )); then
  echo "usage: $0 <smoke|train> <gpu-index> <lane> <variant> <seed> [<seed> ...]" >&2
  exit 2
fi

phase=$1
gpu_index=$2
lane=$3
variant=$4
shift 4
seeds=("$@")
campaign=${ALPHABET_CAMPAIGN:-/home/qlab/experiments/alphabet/complex-scan-followups-imagenet100-20260803}
runtime="$campaign/runtime"
python_bin=/home/qlab/.conda/envs/lnet-paper-cu128/bin/python
runner="$runtime/scripts/run_complex_scan_followups_imagenet100.py"
data_root=/home/qlab/data/ImageNet100
log_root="$campaign/logs"
mkdir -p "$log_root"

record_failure() {
  local status="$?"
  printf '%s exit=%s phase=%s lane=%s variant=%s\n' \
    "$(date -Is)" "$status" "$phase" "$lane" "$variant" \
    >"$campaign/${phase}-${lane}.failed"
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
        --variants "$variant" \
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
  smoke_root="$campaign/smoke/$variant"
  smoke_log="$log_root/smoke-${variant}.log"
  run_with_retry "$smoke_root" 1 "$smoke_log"
  SMOKE_ROOT="$smoke_root" VARIANT="$variant" SEED="${seeds[0]}" "$python_bin" - <<'PY'
import json
import math
import os
from pathlib import Path

root = Path(os.environ["SMOKE_ROOT"])
variant = os.environ["VARIANT"]
seed = int(os.environ["SEED"])
row = json.loads((root / "results" / f"{variant}__seed{seed}.json").read_text())
contract = json.loads((root / "contract.json").read_text())
if row["parameters"] != contract["parameter_counts"][variant]:
    raise RuntimeError("follow-up smoke parameter count changed")
if len(row["history"]) != 1:
    raise RuntimeError("follow-up smoke did not complete exactly one epoch")
values = (
    row["final_validation"]["accuracy"],
    row["final_validation"]["cross_entropy"],
    row["history"][0]["train_loss"],
)
if not all(math.isfinite(float(value)) for value in values):
    raise RuntimeError("follow-up smoke produced a non-finite metric")
PY
  touch "$campaign/smoke-${variant}.passed"
  exit 0
fi

if [[ "$phase" != "train" || ${#seeds[@]} == 0 ]]; then
  echo "train requires at least one seed" >&2
  exit 2
fi
if [[ ! -f "$campaign/smoke-${variant}.passed" ]]; then
  echo "variant smoke has not passed: $variant" >&2
  exit 1
fi
run_with_retry "$campaign/run100" 100 "$log_root/train-${lane}.log"
printf '%s variant=%s seeds=%s\n' "$(date -Is)" "$variant" "${seeds[*]}" \
  >"$campaign/train-${lane}.done"
