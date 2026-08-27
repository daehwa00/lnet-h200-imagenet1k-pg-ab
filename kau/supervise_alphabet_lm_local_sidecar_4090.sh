#!/usr/bin/env bash
set -euo pipefail

if (( $# != 1 )); then
  echo "usage: $0 BASELINE_BASE" >&2
  exit 2
fi
readonly BASELINE_BASE="$1"
readonly PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly EXPECTED_COMMIT="$(git -C "${PROJECT_ROOT}" rev-parse HEAD)"
observed=""
while true; do
  processes="$(
    nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader,nounits \
      | sed '/^[[:space:]]*$/d' | sort -u | tr '\n' ' '
  )"
  if [[ -z "${processes// }" ]]; then
    break
  fi
  if [[ "${processes}" != "${observed}" ]]; then
    printf 'LOCAL_SIDECAR_WAITING_FOR_GPU=0 pids=%s time=%s\n' \
      "${processes}" "$(date --iso-8601=seconds)"
    observed="${processes}"
  fi
  sleep 30
done

set +e
KAU_EXPECTED_COMMIT="${EXPECTED_COMMIT}" \
  "${PROJECT_ROOT}/kau/run_alphabet_lm_local_sidecar_4090_2m.sh"
sidecar_status=$?
set -e
printf 'LOCAL_SIDECAR_EXIT=%s time=%s\n' \
  "${sidecar_status}" "$(date --iso-8601=seconds)"
exec "${BASELINE_BASE}/runtime/scripts/wait_and_launch_rtx4090_baseline_lane.sh" \
  "${BASELINE_BASE}" /data/ImageNet/2012 2 0 0-15
