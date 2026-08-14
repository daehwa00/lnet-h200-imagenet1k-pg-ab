#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 7 ]]; then
  echo "usage: $0 REPO DATA_ROOT OUTPUT_ROOT PYTHON GPU SEED RUNNER" >&2
  exit 2
fi

repo=$1
data_root=$2
root=$3
python_bin=$4
gpu=$5
seed=$6
runner=$7

if [[ $runner == */* ]]; then
  echo "RUNNER must name one Python file under REPO/scripts: $runner" >&2
  exit 2
fi
runner_path="$repo/scripts/$runner"
if [[ ! -f $runner_path ]]; then
  echo "runner does not exist: $runner_path" >&2
  exit 2
fi

epochs=${LNET_EPOCHS:-100}
batch_size=${LNET_BATCH_SIZE:-128}
gradient_accumulation_steps=${LNET_GRADIENT_ACCUMULATION_STEPS:-2}
workers=${LNET_DATALOADER_WORKERS:-8}
precision=${LNET_PRECISION:-bfloat16}
compile_mode=${LNET_COMPILE_MODE:-}
compile_label=${compile_mode:-recipe-default}
selected_variants=${LNET_VARIANTS:-}
torchinductor_cache=${LNET_TORCHINDUCTOR_CACHE_DIR:-$root/torchinductor}
log="$root/logs/train.log"

variant_args=()
if [[ -n $selected_variants ]]; then
  read -r -a variant_list <<<"$selected_variants"
  variant_args=(--variants "${variant_list[@]}")
fi

mkdir -p "$root/logs" "$torchinductor_cache"
cd "$repo"
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH="$repo/src:$repo/scripts"
export PYTHONUNBUFFERED=1
export TORCHINDUCTOR_CACHE_DIR="$torchinductor_cache"

# This host exposes only the versioned driver library. Triton's first-use
# helper links against -lcuda, so make the CUDA stub visible at link time; the
# resulting extension still resolves libcuda.so.1 from the real driver.
cuda_stub_dir=/usr/local/cuda-12.8/targets/x86_64-linux/lib/stubs
export LIBRARY_PATH="$cuda_stub_dir${LIBRARY_PATH:+:$LIBRARY_PATH}"
# Keep the host's NVML user library aligned with its loaded kernel module so
# CUDA Graph allocator accounting and the queue probe see the same driver.
nvml_dir=/home/qlab/.local/lib/gpupulse-nvml-580.82.07
export LD_LIBRARY_PATH="$nvml_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export WANDB_PROJECT="${WANDB_PROJECT:-alphabet2d-imagenet100}"
export WANDB_ENTITY="${WANDB_ENTITY:-daehwa}"
export LNET_DATALOADER_WORKERS="$workers"
export LNET_PERSISTENT_WORKERS="${LNET_PERSISTENT_WORKERS:-1}"

# Full-model max-autotune can compile hundreds of GEMM candidates. Keep its
# worker count conservative unless the caller has validated the memory budget.
if [[ $compile_mode == max-autotune* ]]; then
  export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
fi

printf '%s\n' \
  "runner=$runner" \
  "variants=${selected_variants:-runner-default}" \
  "compile=$compile_label" \
  "torchinductor_cache=$torchinductor_cache" \
  "precision=$precision" \
  "batch_size=$batch_size" \
  "gradient_accumulation_steps=$gradient_accumulation_steps" \
  "workers=$workers" \
  >"$root/runtime-contract.txt"

while nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits \
  | grep -q '[0-9]'; do
  active_pids=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits \
    | paste -sd, -)
  free_mib=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
  printf '{"event":"queue_wait","active_pids":"%s","free_mib":%s,"time":"%s"}\n' \
    "$active_pids" "$free_mib" "$(date -Is)" >>"$log"
  sleep 60
done

printf '{"event":"launch","runner":"%s","seed":%d,"time":"%s"}\n' \
  "$runner" "$seed" "$(date -Is)" >>"$log"
set +e
"$python_bin" -u "$runner_path" \
  --root "$root" \
  --data-root "$data_root" \
  "${variant_args[@]}" \
  --run-seeds "$seed" \
  --epochs "$epochs" \
  --batch-size "$batch_size" \
  --gradient-accumulation-steps "$gradient_accumulation_steps" \
  --workers "$workers" \
  --precision "$precision" >>"$log" 2>&1
status=$?
set -e
printf '{"event":"exit","runner":"%s","status":%d,"time":"%s"}\n' \
  "$runner" "$status" "$(date -Is)" >>"$log"
if [[ $status -eq 0 ]]; then
  date -Is >"$root/COMPLETE"
fi
exit "$status"
