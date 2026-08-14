#!/usr/bin/env bash
set -euo pipefail

script_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
slug=complex-scan-extreme-kau-v2
remote_root=/home/daehwa/experiments/alphabet/complex-scan-extreme-opt-20260803
local_artifacts="$script_root/.omx/goals/performance/$slug/artifacts"

ssh -o BatchMode=yes kau-daehwa \
  "$remote_root/scripts/evaluate_complex_scan_extreme_kau_remote.sh"
mkdir -p "$local_artifacts"
rsync -a kau-daehwa:"$remote_root/artifacts/" "$local_artifacts/"
