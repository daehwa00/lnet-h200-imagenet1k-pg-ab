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
variant=D4-M64-C8-Cal
result="$root/results/${variant}__seed${seed}.json"
log="$root/logs/train.log"

mkdir -p "$root/logs" "$root/torchinductor"
cd "$repo"
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH="$repo/src:$repo/scripts"
export PYTHONUNBUFFERED=1
export TORCHINDUCTOR_CACHE_DIR="$root/torchinductor"
export WANDB_PROJECT="${WANDB_PROJECT:-alphabet2d-imagenet100}"
export WANDB_ENTITY="${WANDB_ENTITY:-daehwa}"
export WANDB_GROUP="M64-Product4"
export WANDB_NAME="M64-Cal-P4"
# Full-model CUDA graph capture is not safe for this model yet: eager pole
# parameterization creates graph breaks whose outputs are reused by later
# compiled stages. Keep TorchInductor enabled without CUDA graphs.
export LNET_COMPILE_MODE="${LNET_COMPILE_MODE:-reduce-overhead}"
export LNET_DATALOADER_WORKERS="${LNET_DATALOADER_WORKERS:-8}"
export LNET_PERSISTENT_WORKERS="${LNET_PERSISTENT_WORKERS:-1}"
printf '%s\n' \
  'architecture=D4-M64-C8-Cal product-only stages' \
  'scan=existing product_scan_coarse4 (four product paths only)' \
  'scan_outputs=coarse product states + exact full-grid raw directional Q' \
  'cffn=shared width-generic compiled packed Cartesian implementation' \
  'optimizer=fused AdamW from the matched recipe' \
  'precision=float32' \
  'batch_size=128' \
  'gradient_accumulation_steps=2' >"$root/optimization-contract.txt"

attempt=0
while [[ ! -f "$result" ]]; do
  attempt=$((attempt + 1))
  printf '{"event":"launch","attempt":%d,"seed":%d,"time":"%s"}\n' \
    "$attempt" "$seed" "$(date -Is)" >>"$log"
  "$python_bin" -u "$repo/scripts/run_a2d_deep4_m64_canonical8_calibrated_imagenet100.py" \
    --root "$root" \
    --data-root "$data_root" \
    --variants "$variant" \
    --run-seeds "$seed" \
    --epochs 100 \
    --batch-size 128 \
    --gradient-accumulation-steps 2 \
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
