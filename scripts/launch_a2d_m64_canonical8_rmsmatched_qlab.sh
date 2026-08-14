#!/usr/bin/env bash
set -u

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
variant=D4-M64-C8-RMatch
result="$root/results/${variant}__seed${seed}.json"
log="$root/logs/train.log"
calibration="$root/calibration/activation_scales.json"

mkdir -p "$root/logs" "$root/calibration"
cd "$repo"
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH="$repo/src:$repo/scripts"
export PYTHONUNBUFFERED=1
export TORCHINDUCTOR_CACHE_DIR="$root/torchinductor"
export LNET_ACTIVATION_SCALE_JSON="$calibration"
export WANDB_PROJECT="${WANDB_PROJECT:-alphabet2d-imagenet100}"
export WANDB_ENTITY="${WANDB_ENTITY:-daehwa}"
export WANDB_GROUP="M64-Activation"
export WANDB_NAME="M64-RMatch"

if [[ ! -f "$calibration" ]]; then
  if ! "$python_bin" -u "$repo/scripts/calibrate_a2d_m64_radial_scales_imagenet100.py" \
    --data-root "$data_root" \
    --output "$calibration" \
    --seed "$seed" \
    --batch-size 16 \
    --batches 4 >>"$log" 2>&1; then
    printf '{"event":"calibration_failed","time":"%s"}\n' "$(date -Is)" >>"$log"
    exit 1
  fi
fi
if [[ ! -s "$calibration" ]]; then
  echo "activation calibration is missing or empty: $calibration" >&2
  exit 1
fi

attempt=0
while [[ ! -f "$result" ]]; do
  attempt=$((attempt + 1))
  printf '{"event":"launch","attempt":%d,"seed":%d,"time":"%s"}\n' \
    "$attempt" "$seed" "$(date -Is)" >>"$log"
  "$python_bin" -u "$repo/scripts/run_a2d_deep4_m64_canonical8_rmsmatched_imagenet100.py" \
    --root "$root" \
    --data-root "$data_root" \
    --variants "$variant" \
    --run-seeds "$seed" \
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
    sleep 60
  fi
done

date -Is >"$root/COMPLETE"
