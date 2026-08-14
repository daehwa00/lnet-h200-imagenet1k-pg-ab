#!/usr/bin/env bash
set -u

if [[ $# -lt 8 ]]; then
  echo "usage: $0 REPO DATA_ROOT OUTPUT_ROOT PYTHON GPU EPOCHS BATCH VARIANT..." >&2
  exit 2
fi

repo=$1
data_root=$2
output_root=$3
python_bin=$4
gpu=$5
epochs=$6
batch_size=$7
shift 7
variants=("$@")

mkdir -p "$output_root/logs"
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH="$repo/src:$repo/scripts"
export PYTHONUNBUFFERED=1
export WANDB_PROJECT="${WANDB_PROJECT:-alphabet2d-imagenet100}"
export WANDB_ENTITY="${WANDB_ENTITY:-daehwa}"
export WANDB_GROUP="A2D-QHead-Screen30"
export TORCHINDUCTOR_CACHE_DIR="$output_root/torchinductor"

for variant in "${variants[@]}"; do
  result="$output_root/results/${variant}__seed501.json"
  log="$output_root/logs/${variant}.log"
  attempt=0
  while [[ ! -f "$result" ]]; do
    attempt=$((attempt + 1))
    printf '{"event":"launch","variant":"%s","attempt":%d,"time":"%s"}\n' \
      "$variant" "$attempt" "$(date -Is)" >>"$log"
    WANDB_NAME="$variant" "$python_bin" -u \
      "$repo/scripts/run_a2d_affine_qhead_imagenet100.py" \
      --root "$output_root" \
      --data-root "$data_root" \
      --variants "$variant" \
      --run-seeds 501 \
      --epochs "$epochs" \
      --batch-size "$batch_size" \
      --gradient-accumulation-steps $((256 / batch_size)) \
      --workers 8 \
      --precision float32 >>"$log" 2>&1
    status=$?
    printf '{"event":"exit","variant":"%s","attempt":%d,"status":%d,"result":%s,"time":"%s"}\n' \
      "$variant" "$attempt" "$status" "$([[ -f "$result" ]] && echo true || echo false)" \
      "$(date -Is)" >>"$log"
    if [[ ! -f "$result" ]]; then
      sleep 30
    fi
  done
done

date -Is >"$output_root/LANE_COMPLETE"
