#!/usr/bin/env bash
set -euo pipefail

project_root=${1:-.}
dt_root=${2:-.omx/results/pac-dt-mask-attribution-pro6000-20260713}
eff_root=${3:-.omx/results/pac-matched-efficiency-pro6000-20260713}
log_root=${4:-.omx/logs/pac-submission-followups-resume-pro6000-20260713}

cd "$project_root"
mkdir -p "$log_root" "$dt_root" "$eff_root"
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$project_root/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

completed_dt_jobs() {
  find "$dt_root/jobs" -maxdepth 1 -type f -name 'dt_mask__*.json' 2>/dev/null \
    | wc -l
}

# These measurements are only useful without contention. Waiting for 120 seconds
# with no CUDA compute process also prevents a repeat of the initial OOM.
idle_checks=0
while (( idle_checks < 12 )); do
  active=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
    | sed '/^[[:space:]]*$/d' || true)
  if [[ -z "$active" ]]; then
    idle_checks=$((idle_checks + 1))
  else
    idle_checks=0
  fi
  sleep 10
done

for attempt in 1 2 3; do
  [[ "$(completed_dt_jobs)" -ge 25 ]] && break
  pids=()
  for shard in 0 1 2 3; do
    python -m lnet.pac_dt_mask_attribution worker \
      --root "$dt_root" --shard "$shard" --device cuda \
      >"$log_root/dt-attempt${attempt}-worker${shard}.log" 2>&1 &
    pids+=("$!")
    if (( ${#pids[@]} == 2 )); then
      for pid in "${pids[@]}"; do wait "$pid" || true; done
      pids=()
    fi
  done
  for pid in "${pids[@]}"; do wait "$pid" || true; done
done

if [[ "$(completed_dt_jobs)" -lt 25 ]]; then
  printf '%s\n' "dt/mask retry incomplete at $(date -Is)" >"$log_root/FAILED"
  exit 1
fi

python -m lnet.pac_dt_mask_attribution report --root "$dt_root" \
  >"$log_root/dt-report.log" 2>&1

if [[ ! -f "$eff_root/COMPLETE" ]]; then
  python -m lnet.pac_matched_efficiency enqueue --root "$eff_root" \
    >"$log_root/efficiency-enqueue.log" 2>&1
  flock /tmp/lnet-pro6000-efficiency.lock \
    python -m lnet.pac_matched_efficiency worker --root "$eff_root" --device cuda \
    >"$log_root/efficiency-worker.log" 2>&1
  python -m lnet.pac_matched_efficiency report --root "$eff_root" \
    >"$log_root/efficiency-report.log" 2>&1
fi

printf '%s\n' "all resumed follow-ups complete at $(date -Is)" >"$log_root/COMPLETE"
