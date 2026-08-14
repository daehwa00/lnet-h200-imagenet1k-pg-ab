#!/usr/bin/env bash
set -euo pipefail

repo=/home/qlab/daehwa/lnet
cache=/home/qlab/experiments/alphabet/a2d-spectral-prototype-probe-20260805/cache
output=/home/qlab/experiments/alphabet/a2d-head-design-20260806
log="$output/runner.log"

mkdir -p "$output"
exec 9>"$output/runner.lock"
if ! flock -n 9; then
  echo "A2D head-design runner is already active" >&2
  exit 0
fi

cd "$repo"
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$repo/src"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

for supervisor_attempt in $(seq 1 20); do
  echo "$(date --iso-8601=seconds) supervisor attempt $supervisor_attempt" >>"$log"
  set +e
  python -u scripts/run_a2d_frozen_q_head_design.py \
    --cache-root "$cache" \
    --output-root "$output" \
    --device cuda:0 \
    --screen-seed 501 \
    --screen-epochs 12 \
    --final-epochs 30 \
    --finalists 8 \
    --promote-all \
    --seeds 501 509 521 \
    --batch-size 4096 \
    --minimum-batch-size 256 \
    --min-free-gib 3 \
    --memory-poll-seconds 15 \
    --retry-count 2 >>"$log" 2>&1
  status=$?
  set -e
  if [[ $status -eq 0 ]]; then
    set +e
    python -u scripts/analyze_a2d_head_design_complete.py \
      --campaign-root "$output" \
      --cache-root "$cache" \
      --output "$output/complete-analysis.json" \
      --bootstrap-draws 10000 >>"$log" 2>&1
    status=$?
    set -e
    if [[ $status -eq 0 ]]; then
      echo "$(date --iso-8601=seconds) campaign and complete analysis finished" >>"$log"
      exit 0
    fi
  fi
  echo "$(date --iso-8601=seconds) runner exited $status; resumable retry follows" >>"$log"
  sleep 30
done

echo "$(date --iso-8601=seconds) campaign exhausted supervisor retries" >>"$log"
exit 1
