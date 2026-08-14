#!/usr/bin/env bash
set -u

if [[ $# -ne 5 ]]; then
  echo "usage: $0 REPO DATA_ROOT OUTPUT_ROOT PYTHON GPU" >&2
  exit 2
fi

repo=$1
data_root=$2
output_root=$3
python_bin=$4
gpu=$5
result="$output_root/results/a2d_postcarry__seed501.json"
log="$output_root/train.log"

mkdir -p "$output_root"
cd "$repo"
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH="$repo/src:$repo/scripts"
export PYTHONUNBUFFERED=1
export WANDB_PROJECT="${WANDB_PROJECT:-alphabet2d-imagenet100}"
export WANDB_ENTITY="${WANDB_ENTITY:-daehwa}"
export WANDB_GROUP="A2D-PostCarry"
export WANDB_NAME="A2D-PostCarry"
export TORCHINDUCTOR_CACHE_DIR="$output_root/torchinductor"

attempt=0
while [[ ! -f "$result" ]]; do
  attempt=$((attempt + 1))
  printf '{"event":"launch","attempt":%d,"time":"%s"}\n' \
    "$attempt" "$(date -Is)" >>"$log"
  "$python_bin" -u "$repo/scripts/run_a2d_d4_postcarry_imagenet100.py" \
    --root "$output_root" \
    --data-root "$data_root" \
    --variants a2d_postcarry \
    --run-seeds 501 \
    --epochs 100 \
    --batch-size 64 \
    --gradient-accumulation-steps 4 \
    --workers 8 \
    --precision float32 >>"$log" 2>&1
  status=$?
  printf '{"event":"exit","attempt":%d,"status":%d,"result":%s,"time":"%s"}\n' \
    "$attempt" "$status" "$([[ -f "$result" ]] && echo true || echo false)" \
    "$(date -Is)" >>"$log"
  if [[ ! -f "$result" ]]; then
    sleep 30
  fi
done

date -Is >"$output_root/COMPLETE"
