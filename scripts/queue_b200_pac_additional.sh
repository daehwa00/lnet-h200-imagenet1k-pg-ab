#!/usr/bin/env bash
set -euo pipefail

project_root=${1:?project root is required}
output_root=${2:-.omx/results/pac-additional-canonical-20260710}
datasets=(
  sequential-cifar
  sequential-mnist
  permuted-mnist
  lra-image
  lra-listops
  lra-retrieval
)
batches=(64 64 64 64 32 16)

for index in "${!datasets[@]}"; do
  dataset=${datasets[$index]}
  prepared="$project_root/data/external/$dataset.pt"
  while [[ ! -s "$prepared" ]]; do
    sleep 60
  done
  while (( $(pgrep -fc "$output_root/jobs" || true) > 18 )); do
    sleep 60
  done
  bash "$project_root/scripts/launch_b200_pac_dataset.sh" \
    "$project_root" "$dataset" "${batches[$index]}" "$output_root"
done
