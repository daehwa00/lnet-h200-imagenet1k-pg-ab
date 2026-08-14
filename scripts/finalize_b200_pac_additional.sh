#!/usr/bin/env bash
set -euo pipefail

project_root=${1:?project root is required}
campaign=${2:-.omx/results/pac-additional-canonical-20260710}
base_results=${3:-.omx/results/pac-external-all-final-20260710/results/external_comparisons.csv}
datasets=(
  lra-text
  audioset-balanced
  sequential-cifar
  sequential-mnist
  permuted-mnist
  lra-image
  lra-listops
  lra-retrieval
)
batches=(64 64 64 64 64 64 32 16)
expected=144

while pgrep -f "queue_b200_pac_additional.sh.*$campaign" >/dev/null; do
  sleep 60
done
while pgrep -f "pac_external_benchmarks_cli.*$campaign/jobs" >/dev/null; do
  sleep 60
done

for _ in 1 2 3; do
  done_count=$(find "$project_root/$campaign/jobs" \
    -path '*/results/external_comparisons.csv' -type f \
    -exec grep -l ',done,' {} + | wc -l)
  if (( done_count == expected )); then
    break
  fi
  for index in "${!datasets[@]}"; do
    bash "$project_root/scripts/launch_b200_pac_dataset.sh" \
      "$project_root" "${datasets[$index]}" "${batches[$index]}" "$campaign"
  done
  while pgrep -f "pac_external_benchmarks_cli.*$campaign/jobs" >/dev/null; do
    sleep 60
  done
done

done_count=$(find "$project_root/$campaign/jobs" \
  -path '*/results/external_comparisons.csv' -type f \
  -exec grep -l ',done,' {} + | wc -l)
if (( done_count != expected )); then
  echo "campaign incomplete after retries: $done_count/$expected" >&2
  exit 1
fi

python "$project_root/scripts/aggregate_pac_external_jobs.py" \
  --jobs-root "$project_root/$campaign/jobs" \
  --output-root "$project_root/$campaign/final" \
  --base-results "$project_root/$base_results"
touch "$project_root/$campaign/COMPLETE"
