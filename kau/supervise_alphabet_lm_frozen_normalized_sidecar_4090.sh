#!/usr/bin/env bash
set -euo pipefail

if (( $# != 1 )); then
  echo "usage: $0 BASELINE_BASE" >&2
  exit 2
fi
readonly BASELINE_BASE="$1"
readonly PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly EXPECTED_COMMIT="$(git -C "${PROJECT_ROOT}" rev-parse HEAD)"
while nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader,nounits \
  | grep -q '[0-9]'; do
  sleep 15
done
set +e
KAU_EXPECTED_COMMIT="${EXPECTED_COMMIT}" \
  "${PROJECT_ROOT}/kau/run_alphabet_lm_frozen_normalized_sidecar_4090_1m.sh"
sidecar_status=$?
set -e
printf 'FROZEN_NORMALIZED_SIDECAR_EXIT=%s time=%s\n' \
  "${sidecar_status}" "$(date --iso-8601=seconds)"
exec "${BASELINE_BASE}/runtime/scripts/wait_and_launch_rtx4090_baseline_lane.sh" \
  "${BASELINE_BASE}" /data/ImageNet/2012 2 0 0-15
