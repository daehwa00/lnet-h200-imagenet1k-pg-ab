#!/usr/bin/env bash
set -euo pipefail

repo=${1:-.}
root=${2:-.omx/results/pac-untouched-ucr12-confirmatory-pro6000-20260713}
workers=${PAC_UNTOUCHED_WORKERS:-4}
min_free_mib=${PAC_UNTOUCHED_MIN_FREE_MIB:-49152}
max_util=${PAC_UNTOUCHED_MAX_GPU_UTIL:-40}

cd "$repo"
mkdir -p "$root" .omx/logs/pac-untouched-ucr12-confirmatory-pro6000-20260713
printf '%s\n' "$$" >"$root/supervisor.pid"

PYTHONPATH="$repo/src" python -m lnet.pac_untouched_ucr_confirmatory \
  --stage prepare --output-root "$root" \
  >.omx/logs/pac-untouched-ucr12-confirmatory-pro6000-20260713/prepare.log 2>&1

while true; do
  read -r free util < <(
    nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits \
      | head -n 1 | tr -d ','
  )
  if (( free >= min_free_mib && util <= max_util )); then
    break
  fi
  printf '%s waiting free_mib=%s util=%s threshold_free=%s threshold_util=%s\n' \
    "$(date --iso-8601=seconds)" "$free" "$util" "$min_free_mib" "$max_util" \
    >>.omx/logs/pac-untouched-ucr12-confirmatory-pro6000-20260713/supervisor.log
  sleep 30
done

PYTHONPATH="$repo/src" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m lnet.pac_untouched_ucr_confirmatory \
  --stage workers --output-root "$root" --device cuda --workers "$workers" \
  >>.omx/logs/pac-untouched-ucr12-confirmatory-pro6000-20260713/worker.log 2>&1
