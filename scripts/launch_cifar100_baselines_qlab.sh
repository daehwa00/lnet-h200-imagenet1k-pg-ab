#!/usr/bin/env bash
set -euo pipefail

if (( $# < 3 )); then
  echo "usage: $0 CAMPAIGN_ROOT GPU_INDEX VARIANT [VARIANT ...]" >&2
  exit 2
fi

campaign_root=$1
gpu_index=$2
shift 2
variants=("$@")
runtime_root="$campaign_root/runtime"
python_bin=/home/qlab/.conda/envs/lnet-paper-cu128/bin/python
wandb_env=/home/qlab/.config/wandb/qlab.env

if [[ -f "$wandb_env" ]]; then
  # shellcheck disable=SC1090
  source "$wandb_env"
fi

export CUDA_VISIBLE_DEVICES="$gpu_index"
export PYTHONPATH="$runtime_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export WANDB_PROJECT=alphabet2d-cifar100
export WANDB_GROUP=baselines
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

cd "$runtime_root"
exec "$python_bin" -u scripts/run_ultratiny_cifar100_baselines.py \
  --root "$campaign_root/run100" \
  --data-root /home/qlab/data \
  --variants "${variants[@]}" \
  --run-seeds 401 \
  --epochs 100 \
  --batch-size 256 \
  --workers 8
