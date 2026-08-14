#!/usr/bin/env bash
set -u

if [[ $# -ne 5 ]]; then
  echo "usage: $0 REPO DATA_ROOT OUTPUT_ROOT PYTHON GPU" >&2
  exit 2
fi

repo=$1
data_root=$2
root=$3
python_bin=$4
gpu=$5
runner=run_imagenet100_modern_tiny_baselines.py
variants=(tinynext_t fastvit_t8 repvit_m0_9)
log="$root/logs/train.log"

mkdir -p "$root/logs" "$root/torchinductor"
cd "$repo" || exit 1
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH="$root/vendor:$repo/src:$repo/scripts"
export PYTHONUNBUFFERED=1
export TORCHINDUCTOR_CACHE_DIR="$root/torchinductor"
export WANDB_PROJECT="${WANDB_PROJECT:-alphabet2d-imagenet100}"
export WANDB_ENTITY="${WANDB_ENTITY:-daehwa}"
export WANDB_GROUP="${WANDB_GROUP:-Modern-Tiny-D4-Matched-BF16}"
export LNET_COMPILE_MODE=reduce-overhead
export LNET_DATALOADER_WORKERS="${LNET_DATALOADER_WORKERS:-16}"
export LNET_PERSISTENT_WORKERS="${LNET_PERSISTENT_WORKERS:-1}"

cat >"$root/recipe.txt" <<'EOF'
dataset=ImageNet100
epochs=100
seed=501
batch_size=128
gradient_accumulation_steps=2
effective_batch_size=256
precision=bfloat16
optimizer=fused AdamW
learning_rate=0.003
weight_decay=0.05
warmup_epochs=5
schedule=cosine
label_smoothing=0.1
mixup_alpha=0.8
augmentation=matched D4 ImageNet-100 public recipe
compile=reduce-overhead with CUDA graphs
memory_format=channels_last
EOF

wait_for_gpu() {
  while nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits \
    | grep -q '[0-9]'; do
    active_pids=$(nvidia-smi -i "$gpu" --query-compute-apps=pid \
      --format=csv,noheader,nounits | paste -sd, -)
    printf '{"event":"queue_wait","active_pids":"%s","time":"%s"}\n' \
      "$active_pids" "$(date -Is)" >>"$log"
    sleep 30
  done
}

queue_complete() {
  [[ -s "$root/summary.json" ]] || return 1
  local variant
  for variant in "${variants[@]}"; do
    [[ -s "$root/results/${variant}__seed501.json" ]] || return 1
  done
}

attempt=0
while ! queue_complete; do
  wait_for_gpu
  attempt=$((attempt + 1))
  printf '{"event":"launch","attempt":%d,"time":"%s"}\n' \
    "$attempt" "$(date -Is)" >>"$log"
  "$python_bin" -u "$repo/scripts/$runner" \
    --root "$root" \
    --data-root "$data_root" \
    --variants "${variants[@]}" \
    --run-seeds 501 \
    --epochs 100 \
    --batch-size 128 \
    --gradient-accumulation-steps 2 \
    --workers 8 \
    --precision bfloat16 >>"$log" 2>&1
  status=$?
  printf '{"event":"exit","attempt":%d,"status":%d,"complete":%s,"time":"%s"}\n' \
    "$attempt" "$status" "$(queue_complete && echo true || echo false)" \
    "$(date -Is)" >>"$log"
  if ! queue_complete; then
    sleep 30
  fi
done

date -Is >"$root/QUEUE_COMPLETE"
