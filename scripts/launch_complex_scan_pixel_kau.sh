#!/usr/bin/env bash
set -euo pipefail

if (( $# != 2 )); then
  echo "usage: $0 <smoke|train> <gpu-index>" >&2
  exit 2
fi

phase=$1
gpu_index=$2
variant=complex_pixel_residual_dual_fusion256_lrq64
seed=501
campaign=${ALPHABET_CAMPAIGN:-/home/daehwa/experiments/alphabet/complex-scan-pixel-residual-imagenet100-20260803}
runtime="$campaign/runtime"
python_bin=/home/daehwa/anaconda3/envs/alphabet/bin/python
runner="$runtime/scripts/run_complex_scan_pixel_imagenet100.py"
data_root=/data/ImageNet100
log_root="$campaign/logs"
mkdir -p "$log_root"

record_failure() {
  local status="$?"
  printf '%s exit=%s phase=%s\n' "$(date -Is)" "$status" "$phase" \
    >"$campaign/${phase}.failed"
  exit "$status"
}
trap record_failure ERR

run_job() {
  local root=$1
  local epochs=$2
  local log=$3
  (
    cd "$runtime"
    CUDA_VISIBLE_DEVICES="$gpu_index" \
      PYTHONPATH=src \
      TORCHINDUCTOR_CACHE_DIR="$campaign/torchinductor-gpu${gpu_index}" \
      TORCHINDUCTOR_COMPILE_THREADS=4 \
      TORCHINDUCTOR_EMULATE_PRECISION_CASTS=1 \
      "$python_bin" -u "$runner" \
      --root "$root" \
      --data-root "$data_root" \
      --variants "$variant" \
      --run-seeds "$seed" \
      --epochs "$epochs" \
      --batch-size 256 \
      --workers 8 \
      --precision bfloat16
  ) >>"$log" 2>&1
}

if [[ "$phase" == "smoke" ]]; then
  root="$campaign/smoke1"
  run_job "$root" 1 "$log_root/smoke.log"
  ROOT="$root" VARIANT="$variant" SEED="$seed" "$python_bin" - <<'PY'
import json
import math
import os
from pathlib import Path

root = Path(os.environ["ROOT"])
variant = os.environ["VARIANT"]
seed = int(os.environ["SEED"])
row = json.loads((root / "results" / f"{variant}__seed{seed}.json").read_text())
contract = json.loads((root / "contract.json").read_text())
if row["parameters"] != contract["parameter_counts"][variant]:
    raise RuntimeError("complex pixel smoke parameter count changed")
if len(row["history"]) != 1:
    raise RuntimeError("complex pixel smoke did not complete exactly one epoch")
values = (
    row["final_validation"]["accuracy"],
    row["final_validation"]["cross_entropy"],
    row["history"][0]["train_loss"],
)
if not all(math.isfinite(float(value)) for value in values):
    raise RuntimeError("complex pixel smoke produced a non-finite metric")
PY
  touch "$campaign/smoke.passed"
  exit 0
fi

if [[ "$phase" != "train" ]]; then
  echo "unsupported phase: $phase" >&2
  exit 2
fi
if [[ ! -f "$campaign/smoke.passed" ]]; then
  echo "complex pixel smoke has not passed" >&2
  exit 1
fi
run_job "$campaign/run100" 100 "$log_root/train.log"
date -Is >"$campaign/train.done"
