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
variant=P4-Joint128

mkdir -p "$root/logs" "$root/$variant" "$root/torchinductor/$variant"
cd "$repo"
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH="$repo/src:$repo/scripts"
export PYTHONUNBUFFERED=1
export WANDB_PROJECT="${WANDB_PROJECT:-alphabet2d-imagenet100}"
export WANDB_ENTITY="${WANDB_ENTITY:-daehwa}"
export WANDB_GROUP="${WANDB_GROUP:-A2D-P4-Joint128}"
export WANDB_NAME="${WANDB_NAME:-P4-Joint128}"
export LNET_COMPILE_MODE="default"
export LNET_DATALOADER_WORKERS="${LNET_DATALOADER_WORKERS:-16}"
export TORCHINDUCTOR_CACHE_DIR="$root/torchinductor/$variant"

"$python_bin" -u "$repo/scripts/run_a2d_p4_joint_imagenet100.py" \
  --root "$root/$variant" \
  --data-root "$data_root" \
  --variants "$variant" \
  --run-seeds "$seed" \
  --epochs 100 \
  --batch-size 128 \
  --gradient-accumulation-steps 2 \
  --workers 8 \
  --precision bfloat16 >>"$root/logs/$variant.log" 2>&1

date -Is >"$root/COMPLETE"
