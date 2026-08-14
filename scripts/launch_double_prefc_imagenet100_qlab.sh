#!/usr/bin/env bash
set -euo pipefail

if (( $# != 2 )); then
  echo "usage: $0 <smoke|train> <gpu-index>" >&2
  exit 2
fi

phase=$1
gpu_index=$2
variant=${DOUBLE_PREFC_VARIANT:-double_prefc}
seed=501
campaign=${ALPHABET_CAMPAIGN:-/home/qlab/experiments/alphabet/polepyramid-double-prefc-imagenet100-20260804}
runtime=${ALPHABET_RUNTIME:-/home/qlab/daehwa/lnet}
python_bin=/home/qlab/.conda/envs/lnet-paper-cu128/bin/python
runner=${DOUBLE_PREFC_RUNNER:-$runtime/scripts/run_double_prefc_imagenet100.py}
data_root=/home/qlab/data/ImageNet100
log_root="$campaign/logs"
mkdir -p "$log_root"

wandb_env_file=${WANDB_ENV_FILE:-$HOME/.config/wandb/qlab.env}
if [[ -f "$wandb_env_file" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$wandb_env_file"
  set +a
fi
export WANDB_PROJECT=${WANDB_PROJECT:-alphabet2d-imagenet100}
export WANDB_GROUP=${WANDB_GROUP:-architecture-search}
export WANDB_NAME=${WANDB_NAME:-DoublePreFC}
export WANDB_CONSOLE=${WANDB_CONSOLE:-off}
export WANDB_SILENT=${WANDB_SILENT:-true}
if [[ "$phase" == "smoke" ]]; then
  export WANDB_MODE=disabled
else
  export WANDB_MODE=${WANDB_MODE:-online}
fi

run() {
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
        "$python_bin" -u "$runner" \
          --root "$root" \
          --data-root "$data_root" \
          --variants "$variant" \
          --run-seeds "$seed" \
          --epochs "$epochs" \
          --batch-size 128 \
          --gradient-accumulation-steps 2 \
          --workers 8 \
          --precision bfloat16
    ) >>"$log" 2>&1; then
      return 0
    fi
    printf '%s attempt=%s failed; retrying from the epoch checkpoint\n' \
      "$(date -Is)" "$attempt" >>"$log"
    sleep 5
  done
  return 1
}

if [[ "$phase" == "smoke" ]]; then
  smoke_root="$campaign/smoke"
  run "$smoke_root" 1 "$log_root/smoke.log"
  ROOT="$smoke_root" VARIANT="$variant" SEED="$seed" "$python_bin" - <<'PY'
import json
import math
import os
from pathlib import Path

root = Path(os.environ["ROOT"])
variant = os.environ["VARIANT"]
seed = int(os.environ["SEED"])
row = json.loads((root / "results" / f"{variant}__seed{seed}.json").read_text())
contract = json.loads((root / "contract.json").read_text())
config = contract["variant_configs"][variant]
assert config["use_precomplex_fc"] is True
assert config["precomplex_fc_layers"] == 2
if variant == "double_prefc_no_norm":
    assert config["stage1_ffn_norm"] is False
values = (
    row["final_validation"]["accuracy"],
    row["final_validation"]["cross_entropy"],
    row["history"][0]["train_loss"],
)
assert all(math.isfinite(float(value)) for value in values)
PY
  touch "$campaign/smoke.passed"
  exit 0
fi

if [[ "$phase" != "train" ]]; then
  echo "unsupported phase: $phase" >&2
  exit 2
fi
if [[ ! -f "$campaign/smoke.passed" ]]; then
  echo "DoublePreFC smoke has not passed" >&2
  exit 1
fi
run "$campaign/run100" 100 "$log_root/train.log"
date -Is >"$campaign/train.done"
