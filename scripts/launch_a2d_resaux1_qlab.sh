#!/usr/bin/env bash
set -u

if [[ $# -ne 6 ]]; then
  echo "usage: $0 REPO DATA_ROOT OUTPUT_ROOT PYTHON GPU EPOCHS" >&2
  exit 2
fi

repo=$1
data_root=$2
output_root=$3
python_bin=$4
gpu=$5
epochs=$6
variant=A2D-ResAux1
result="$output_root/results/${variant}__seed501.json"
log="$output_root/logs/${variant}.log"

mkdir -p "$output_root/logs"
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH="$repo/src:$repo/scripts"
export PYTHONUNBUFFERED=1
export WANDB_PROJECT="${WANDB_PROJECT:-alphabet2d-imagenet100}"
export WANDB_ENTITY="${WANDB_ENTITY:-daehwa}"
export WANDB_GROUP="A2D-ResAux1-Confirm100"
export WANDB_NAME="$variant"
export TORCHINDUCTOR_CACHE_DIR="$output_root/torchinductor"

while nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits \
  | grep -q '[0-9]'; do
  sleep 30
done

while [[ ! -f "$output_root/SMOKE_OK" ]]; do
  "$python_bin" -u "$repo/scripts/smoke_a2d_resaux1.py" \
    --device cuda --size 64 --batch-size 2 >>"$log" 2>&1
  status=$?
  if [[ $status -eq 0 ]]; then
    date -Is >"$output_root/SMOKE_OK"
    break
  fi
  printf '{"event":"smoke_failed","status":%d,"time":"%s"}\n' \
    "$status" "$(date -Is)" >>"$log"
  sleep 60
done

attempt=0
while [[ ! -f "$result" ]]; do
  attempt=$((attempt + 1))
  printf '{"event":"launch","attempt":%d,"epochs":%d,"time":"%s"}\n' \
    "$attempt" "$epochs" "$(date -Is)" >>"$log"
  "$python_bin" -u "$repo/scripts/run_a2d_resaux1_imagenet100.py" \
    --root "$output_root" \
    --data-root "$data_root" \
    --variants "$variant" \
    --run-seeds 501 \
    --epochs "$epochs" \
    --batch-size 64 \
    --gradient-accumulation-steps 4 \
    --workers 8 \
    --precision float32 >>"$log" 2>&1
  status=$?
  printf '{"event":"exit","attempt":%d,"status":%d,"result":%s,"time":"%s"}\n' \
    "$attempt" "$status" "$([[ -f "$result" ]] && echo true || echo false)" \
    "$(date -Is)" >>"$log"
  if [[ ! -f "$result" ]]; then
    sleep 60
  fi
done

date -Is >"$output_root/LANE_COMPLETE"
