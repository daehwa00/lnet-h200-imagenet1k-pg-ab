#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 7 ]]; then
  echo "usage: $0 REPO DATA_ROOT OUTPUT_BASE PYTHON GPU EPOCHS VARIANT..." >&2
  exit 2
fi

repo=$1
data_root=$2
output_base=$3
python_bin=$4
gpu=$5
epochs=$6
shift 6
variants=("$@")

mkdir -p "$output_base"
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH="$repo/src:$repo/scripts"
export PYTHONUNBUFFERED=1
export WANDB_PROJECT="${WANDB_PROJECT:-alphabet2d-imagenet100}"
export WANDB_ENTITY="${WANDB_ENTITY:-daehwa}"
export WANDB_GROUP="A2D-HeadDesign-E2E"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export LNET_PAC_BLOCK_MODES=16
export LNET_PAC_RECURRENCE_FORWARD_NUM_WARPS=8
export LNET_PAC_RECURRENCE_BACKWARD_NUM_WARPS=8

wait_for_gpu() {
  while nvidia-smi -i "$gpu" --query-compute-apps=pid \
    --format=csv,noheader,nounits | grep -q '[0-9]'; do
    sleep 30
  done
}

for variant in "${variants[@]}"; do
  output_root="$output_base/$variant"
  result="$output_root/results/${variant}__seed501.json"
  log="$output_root/runner.log"
  mkdir -p "$output_root"
  export WANDB_NAME="$variant"
  export TORCHINDUCTOR_CACHE_DIR="$output_root/torchinductor"

  if [[ -f "$result" ]]; then
    continue
  fi
  wait_for_gpu
  if [[ ! -f "$output_root/SMOKE_OK" ]]; then
    "$python_bin" -u "$repo/scripts/smoke_a2d_head_design_e2e.py" \
      --size 224 --batch-size 2 --variants "$variant" >>"$log" 2>&1
    date -Is >"$output_root/SMOKE_OK"
  fi

  attempt=0
  while [[ ! -f "$result" ]]; do
    attempt=$((attempt + 1))
    printf '{"event":"launch","attempt":%d,"variant":"%s","time":"%s"}\n' \
      "$attempt" "$variant" "$(date -Is)" >>"$log"
    set +e
    "$python_bin" -u "$repo/scripts/run_a2d_head_design_e2e_imagenet100.py" \
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
    set -e
    printf '{"event":"exit","attempt":%d,"status":%d,"time":"%s"}\n' \
      "$attempt" "$status" "$(date -Is)" >>"$log"
    if [[ ! -f "$result" ]]; then
      sleep 60
    fi
  done
done

date -Is >"$output_base/LANE_COMPLETE_GPU${gpu}"
