#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 REPO DATA_ROOT OUTPUT_ROOT PYTHON GPU SEED" >&2
  exit 2
fi

repo=$1
data_root=$2
root=$3
python_bin=$4
gpu=$5
seed=$6

mkdir -p "$root/logs"
cd "$repo"
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH="$repo/src:$repo/scripts"
export PYTHONUNBUFFERED=1
export WANDB_PROJECT="${WANDB_PROJECT:-alphabet2d-imagenet100}"
export WANDB_ENTITY="${WANDB_ENTITY:-daehwa}"
export WANDB_GROUP="${WANDB_GROUP:-A2D-P4-W16}"
export LNET_COMPILE_MODE="default"
# The qlab host has 16 physical cores.  Eight PIL augmentation workers leave
# the GPU input-starved; the paired pipeline benchmark selects all 16 cores.
export LNET_DATALOADER_WORKERS="${LNET_DATALOADER_WORKERS:-16}"

for variant in P4-R; do
  run_root="$root/$variant"
  result="$run_root/results/${variant}__seed${seed}.json"
  log="$root/logs/${variant}.log"
  mkdir -p "$run_root" "$root/torchinductor/$variant"
  if [[ -f "$result" ]]; then
    continue
  fi
  export TORCHINDUCTOR_CACHE_DIR="$root/torchinductor/$variant"
  export WANDB_NAME="$variant"
  "$python_bin" -u "$repo/scripts/run_a2d_p4_imagenet100.py" \
    --root "$run_root" \
    --data-root "$data_root" \
    --variants "$variant" \
    --run-seeds "$seed" \
    --epochs 100 \
    --batch-size 128 \
    --gradient-accumulation-steps 2 \
    --workers 8 \
    --precision bfloat16 >>"$log" 2>&1
done

date -Is >"$root/COMPLETE"
