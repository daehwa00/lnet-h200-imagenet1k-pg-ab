#!/usr/bin/env bash
set -euo pipefail

project_root=${1:?project root is required}
campaign=${2:-.omx/results/pac-extended-baselines-20260711}
base_results=${3:-.omx/results/pac-additional-canonical-20260710/final-combined/results/external_comparisons.csv}
expected=204
slots=${PAC_BASELINE_SLOTS:-16}

wait_for_workers() {
  while (( $(find "$project_root/$campaign" -name 'worker-*.COMPLETE' | wc -l) < slots )); do
    sleep 60
  done
}

wait_for_workers
for _attempt in 1 2; do
  done_count=$(find "$project_root/$campaign/jobs" \
    -path '*/results/external_comparisons.csv' -type f \
    -exec grep -l ',done,' {} + 2>/dev/null | wc -l)
  if (( done_count == expected )); then
    break
  fi
  rm -f "$project_root/$campaign"/worker-*.COMPLETE
  PAC_BASELINE_SLOTS="$slots" bash "$project_root/scripts/launch_b200_extended_baselines.sh" \
    "$project_root" "$campaign" launch
  wait_for_workers
done

done_count=$(find "$project_root/$campaign/jobs" \
  -path '*/results/external_comparisons.csv' -type f \
  -exec grep -l ',done,' {} + 2>/dev/null | wc -l)
if (( done_count != expected )); then
  echo "extended baseline campaign incomplete after retries: $done_count/$expected" >&2
  exit 1
fi

python "$project_root/scripts/aggregate_pac_external_jobs.py" \
  --jobs-root "$project_root/$campaign/jobs" \
  --output-root "$project_root/$campaign/final" \
  --base-results "$project_root/$base_results"
touch "$project_root/$campaign/COMPLETE"
