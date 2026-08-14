#!/usr/bin/env bash
set -euo pipefail

root=${1:-.}
campaign=${2:-.omx/results/pac-fair-boundary-baselines-pro6000-20260713}
log_root=${3:-.omx/logs/pac-fair-boundary-baselines-pro6000-20260713}
workers=${PAC_FAIR_BOUNDARY_WORKERS:-6}

cd "$root"
mkdir -p "$campaign" "$log_root"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

PYTHONPATH="$root/src" python -m lnet.pac_fair_boundary_cli \
  --stage enqueue --output-root "$campaign" --shards "$workers"

for index in $(seq 0 $((workers - 1))); do
  shard="$campaign/shards/shard-$(printf '%02d' "$index")"
  log="$log_root/worker-$(printf '%02d' "$index").log"
  pid_file="$log_root/worker-$(printf '%02d' "$index").pid"
  if [[ -f "$pid_file" ]] && kill -0 "$(<"$pid_file")" 2>/dev/null; then
    continue
  fi
  nohup env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 PYTHONPATH="$root/src" \
    python -m lnet.pac_fair_boundary_cli --stage worker \
      --output-root "$campaign" --shard-root "$shard" --device cuda --workers 1 \
      >"$log" 2>&1 </dev/null &
  echo "$!" >"$pid_file"
done

if [[ "${PAC_QUEUE_WAIT:-0}" == 1 ]]; then
  wait
fi
