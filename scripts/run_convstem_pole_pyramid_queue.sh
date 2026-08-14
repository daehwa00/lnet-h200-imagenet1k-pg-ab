#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 GPU_ID RUN_ROOT [WAIT_FOR_PID]" >&2
  exit 2
fi

gpu_id=$1
run_root=$2
wait_for_pid=${3:-}
repository=${ALPHABET_REPOSITORY:-LOCAL_HOME_PLACEHOLDER/alphabet-astronomy-20260729}
python=${ALPHABET_PYTHON:-LOCAL_HOME_PLACEHOLDER/miniconda3/envs/brelu/bin/python}
data_root=${ALPHABET_DATA_ROOT:-LOCAL_HOME_PLACEHOLDER/data/ImageNet100}

mkdir -p "$run_root/logs" "$run_root/claims" "$run_root/results"

if [[ -n "$wait_for_pid" ]]; then
  while kill -0 "$wait_for_pid" 2>/dev/null; do
    sleep 30
  done
fi

default_jobs="convstem_pole:501 convstem_pole:509 convstem_pole:521 convstem_average:501 convstem_average:509 convstem_average:521"
read -r -a jobs <<< "${ALPHABET_QUEUE_JOBS:-$default_jobs}"

cd "$repository"
for job in "${jobs[@]}"; do
  variant=${job%%:*}
  seed=${job##*:}
  result="$run_root/results/${variant}__seed${seed}.json"
  claim="$run_root/claims/${variant}__seed${seed}"
  if [[ -f "$result" ]] || ! mkdir "$claim" 2>/dev/null; then
    continue
  fi
  log="$run_root/logs/gpu${gpu_id}-${variant}-seed${seed}.log"
  if CUDA_VISIBLE_DEVICES="$gpu_id" PYTHONPATH=src "$python" \
    scripts/run_convstem_pole_pyramid_imagenet100.py \
    --root "$run_root" \
    --data-root "$data_root" \
    --variants "$variant" \
    --run-seeds "$seed" \
    --epochs 100 \
    --batch-size 256 \
    --workers 8 \
    --precision bfloat16 >"$log" 2>&1; then
    touch "$claim/completed"
  else
    touch "$claim/failed"
  fi
done
