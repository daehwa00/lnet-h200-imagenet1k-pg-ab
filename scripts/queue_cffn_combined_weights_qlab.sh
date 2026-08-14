#!/usr/bin/env bash
set -euo pipefail

experiment_root=${EXPERIMENT_ROOT:-/home/qlab/experiments/alphabet/cffn-combined-weights-20260804}
evaluator="$experiment_root/evaluate.sh"
queue_log="$experiment_root/artifacts/queue.log"

while true; do
  set +e
  "$evaluator" >>"$queue_log" 2>&1
  status=$?
  set -e
  if (( status != 75 )); then
    exit "$status"
  fi
  sleep 60
done
