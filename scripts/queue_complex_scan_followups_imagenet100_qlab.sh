#!/usr/bin/env bash
set -euo pipefail

if (( $# != 1 )); then
  echo "usage: $0 <gpu-index>" >&2
  exit 2
fi

gpu_index=$1
campaign=${ALPHABET_CAMPAIGN:-/home/qlab/experiments/alphabet/complex-scan-followups-imagenet100-20260803}
launcher="$campaign/runtime/scripts/launch_complex_scan_followups_imagenet100_qlab.sh"
predecessor=${ALPHABET_PREDECESSOR-"pole-fusion256-im100-queue-gpu${gpu_index}"}
single_seed=${ALPHABET_SINGLE_SEED:-}

if [[ -n "$predecessor" ]]; then
  while tmux has-session -t "$predecessor" 2>/dev/null; do
    sleep 30
  done
fi

variants=(
  dual_fusion256_lrq64
  capacity_dual_fusion384_lrq64
  capacity_fusion384
)

wait_for_smoke() {
  local variant=$1
  while [[ ! -f "$campaign/smoke-${variant}.passed" ]]; do
    if [[ -f "$campaign/smoke-smoke-${variant}.failed" ]]; then
      echo "smoke failed for $variant; refusing downstream training" >&2
      exit 1
    fi
    sleep 15
  done
}

for index in "${!variants[@]}"; do
  variant=${variants[$index]}
  if [[ -n "$single_seed" ]]; then
    if [[ ! -f "$campaign/smoke-${variant}.passed" ]]; then
      echo "single-seed queue requires a passed smoke for $variant" >&2
      exit 1
    fi
    if (( index % 2 != gpu_index )); then
      continue
    fi
    seeds=("$single_seed")
    "$launcher" train "$gpu_index" "gpu${gpu_index}-${variant}" "$variant" "${seeds[@]}"
    continue
  fi
  if (( gpu_index == 0 )); then
    if [[ ! -f "$campaign/smoke-${variant}.passed" ]]; then
      "$launcher" smoke 0 "smoke-${variant}" "$variant" 501
    fi
    if (( index % 2 == 0 )); then
      seeds=(501)
    else
      seeds=(501 521)
    fi
  else
    wait_for_smoke "$variant"
    if (( index % 2 == 0 )); then
      seeds=(509 521)
    else
      seeds=(509)
    fi
  fi
  "$launcher" train "$gpu_index" "gpu${gpu_index}-${variant}" "$variant" "${seeds[@]}"
done
