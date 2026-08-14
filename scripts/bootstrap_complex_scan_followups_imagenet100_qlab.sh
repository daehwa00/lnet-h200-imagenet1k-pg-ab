#!/usr/bin/env bash
set -euo pipefail

campaign=${ALPHABET_CAMPAIGN:-/home/qlab/experiments/alphabet/complex-scan-followups-imagenet100-extreme-kernel-v2-20260803}
session_prefix=${ALPHABET_SESSION_PREFIX:-pole-followups-im100-extreme-v2}
single_seed=${ALPHABET_SINGLE_SEED:-}
runtime="$campaign/runtime"
launcher="$runtime/scripts/launch_complex_scan_followups_imagenet100_qlab.sh"
queue="$runtime/scripts/queue_complex_scan_followups_imagenet100_qlab.sh"

mkdir -p "$campaign/logs"

record_failure() {
  local status=$?
  printf '%s exit=%s\n' "$(date -Is)" "$status" >"$campaign/bootstrap.failed"
  exit "$status"
}
trap record_failure ERR

run_smoke_lane() {
  local gpu=$1
  shift
  local variant
  for variant in "$@"; do
    if [[ ! -f "$campaign/smoke-${variant}.passed" ]]; then
      ALPHABET_CAMPAIGN="$campaign" \
        "$launcher" smoke "$gpu" "smoke-${variant}" "$variant" 501
    fi
  done
}

run_smoke_lane 0 \
  capacity_dual_fusion384_lrq64 \
  >"$campaign/logs/bootstrap-smoke-gpu0.log" 2>&1 &
smoke_gpu0=$!

run_smoke_lane 1 \
  dual_fusion256_lrq64 \
  capacity_fusion384 \
  >"$campaign/logs/bootstrap-smoke-gpu1.log" 2>&1 &
smoke_gpu1=$!

wait "$smoke_gpu0"
wait "$smoke_gpu1"

variants=(
  dual_fusion256_lrq64
  capacity_dual_fusion384_lrq64
  capacity_fusion384
)
for variant in "${variants[@]}"; do
  test -f "$campaign/smoke-${variant}.passed"
done

for gpu in 0 1; do
  session="${session_prefix}-queue-gpu${gpu}"
  tmux kill-session -t "$session" 2>/dev/null || true
  tmux new-session -d -s "$session" \
    "env ALPHABET_CAMPAIGN='$campaign' ALPHABET_PREDECESSOR='' ALPHABET_SINGLE_SEED='$single_seed' bash '$queue' '$gpu'"
done

printf '%s queues=%s,%s\n' \
  "$(date -Is)" "${session_prefix}-queue-gpu0" "${session_prefix}-queue-gpu1" \
  >"$campaign/bootstrap.passed"
