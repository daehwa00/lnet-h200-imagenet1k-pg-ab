#!/usr/bin/env bash
set -euo pipefail

root=${1:-.}
campaign=${2:-.omx/results/pac-final-alphabet-external16-pro6000-20260713}
log_root=${3:-.omx/logs/pac-final-alphabet-external16-pro6000-20260713}
workers=${PAC_FAIR_EXTERNAL_WORKERS:-4}

cd "$root"
mkdir -p "$campaign" "$log_root"
PYTHONPATH="$root/src" python -m lnet.pac_fair_external_cli \
  --stage enqueue --output-root "$campaign" --workers "$workers"

for index in $(seq 0 $((workers - 1))); do
  manifest="$campaign/manifests/worker-$(printf '%02d' "$index").tsv"
  log="$log_root/worker-$(printf '%02d' "$index").log"
  pid_file="$log_root/worker-$(printf '%02d' "$index").pid"
  if [[ -f "$pid_file" ]] && kill -0 "$(<"$pid_file")" 2>/dev/null; then
    continue
  fi
  env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 bash scripts/run_pro6000_fair_external_worker.sh \
    "$root" "$campaign" "$manifest" >"$log" 2>&1 &
  echo "$!" >"$pid_file"
done

wait
