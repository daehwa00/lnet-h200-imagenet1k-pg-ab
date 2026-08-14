#!/usr/bin/env bash
set -u

if [[ $# -lt 7 ]]; then
  echo "usage: $0 REPO DATA_ROOT OUTPUT_BASE PYTHON GPU EPOCHS RUNNER:VARIANT [...]" >&2
  exit 2
fi

repo=$1
data_root=$2
output_base=$3
python_bin=$4
gpu=$5
epochs=$6
shift 6

wait_for_gpu() {
  while nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits \
    | grep -q '[0-9]'; do
    sleep 30
  done
}

mkdir -p "$output_base"
cd "$repo"
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH="$repo/src:$repo/scripts"
export PYTHONUNBUFFERED=1
export WANDB_PROJECT="${WANDB_PROJECT:-alphabet2d-imagenet100}"
export WANDB_ENTITY="${WANDB_ENTITY:-daehwa}"

for job in "$@"; do
  runner=${job%%:*}
  variant=${job#*:}
  if [[ -z "$runner" || -z "$variant" || "$runner" == "$variant" ]]; then
    echo "invalid queue job: $job" >&2
    exit 2
  fi
  root="$output_base/$variant"
  log="$root/logs/$variant.log"
  result="$root/results/${variant}__seed501.json"
  mkdir -p "$root/logs"
  export WANDB_GROUP="Deep4-Overnight"
  export WANDB_NAME="$variant"
  export TORCHINDUCTOR_CACHE_DIR="$root/torchinductor"

  wait_for_gpu
  while [[ ! -f "$root/SMOKE_OK" ]]; do
    "$python_bin" -u "$repo/scripts/smoke_a2d_runner.py" \
      --runner "$runner" --variant "$variant" \
      --device cuda --size 64 --batch-size 2 >>"$log" 2>&1
    status=$?
    if [[ $status -eq 0 ]]; then
      date -Is >"$root/SMOKE_OK"
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
    "$python_bin" -u "$repo/scripts/${runner}.py" \
      --root "$root" \
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
  date -Is >"$root/LANE_COMPLETE"
done

date -Is >"$output_base/QUEUE_COMPLETE"
