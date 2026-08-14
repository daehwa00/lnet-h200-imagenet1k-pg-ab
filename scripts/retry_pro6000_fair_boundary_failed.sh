#!/usr/bin/env bash
set -euo pipefail

project_root=${1:-.}
campaign=${2:-.omx/results/pac-fair-boundary-baselines-pro6000-20260713}
log_root=${3:-.omx/logs/pac-fair-boundary-baselines-pro6000-20260713}
max_attempts=${PAC_FAIR_BOUNDARY_RETRY_ATTEMPTS:-3}

cd "$project_root"
mkdir -p "$log_root"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# Do not compete with the six primary workers. Failed rows remain retryable because
# the queue runner only skips keys that already have a successful row.
while :; do
  active=0
  for pid_file in "$log_root"/worker-??.pid; do
    [[ -e "$pid_file" ]] || continue
    if kill -0 "$(<"$pid_file")" 2>/dev/null; then
      active=1
      break
    fi
  done
  [[ "$active" == 0 ]] && break
  sleep 30
done

for attempt in $(seq 1 "$max_attempts"); do
  failed=$(PYTHONPATH="$project_root/src" python -m lnet.pac_fair_boundary_cli \
    --stage status --output-root "$campaign" | \
    python -c 'import json,sys; print(json.load(sys.stdin)["failed"])')
  [[ "$failed" == 0 ]] && exit 0

  # Require sustained headroom so a retry cannot reproduce a transient OOM caused
  # by the large official-TEST and external-task campaigns sharing the PRO6000.
  stable=0
  while (( stable < 2 )); do
    read -r free_mib util_pct < <(
      nvidia-smi --query-gpu=memory.free,utilization.gpu \
        --format=csv,noheader,nounits | awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); print $1, $2}'
    )
    if (( free_mib >= 49152 && util_pct <= 40 )); then
      stable=$((stable + 1))
    else
      stable=0
    fi
    sleep 30
  done

  for shard in "$campaign"/shards/shard-??; do
    PYTHONPATH="$project_root/src" python -m lnet.pac_fair_boundary_cli \
      --stage worker --output-root "$campaign" --shard-root "$shard" \
      --device cuda --workers 1 \
      >>"$log_root/retry-attempt-${attempt}.log" 2>&1
  done
done

PYTHONPATH="$project_root/src" python -m lnet.pac_fair_boundary_cli \
  --stage status --output-root "$campaign" >"$log_root/retry-final-status.json"
