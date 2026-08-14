#!/usr/bin/env bash
set -euo pipefail

root=${1:-.}
cd "$root"

dt_root=.omx/results/pac-dt-mask-attribution-pro6000-20260713
eff_root=.omx/results/pac-matched-efficiency-pro6000-20260713
log_root=.omx/logs/pac-submission-followups-pro6000-20260713
mkdir -p "$dt_root" "$eff_root" "$log_root"

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$root/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

fail() {
  printf '%s\n' "failed at $(date -Is)" >"$log_root/FAILED"
}
trap fail ERR

python -m lnet.pac_dt_mask_attribution enqueue --root "$dt_root" --shards 4 \
  >"$log_root/dt-enqueue.log" 2>&1

worker_pids=()
for shard in 0 1 2 3; do
  python -m lnet.pac_dt_mask_attribution worker \
    --root "$dt_root" --shard "$shard" --device cuda \
    >"$log_root/dt-worker${shard}.log" 2>&1 &
  worker_pids+=("$!")
done
for pid in "${worker_pids[@]}"; do
  wait "$pid"
done
python -m lnet.pac_dt_mask_attribution report --root "$dt_root" \
  >"$log_root/dt-report.log" 2>&1

python -m lnet.pac_matched_efficiency enqueue --root "$eff_root" \
  >"$log_root/efficiency-enqueue.log" 2>&1
printf '%s\n' "waiting for 120 seconds of exclusive GPU idleness from $(date -Is)" \
  >"$eff_root/WAITING_FOR_EXCLUSIVE_GPU"

idle_checks=0
while (( idle_checks < 12 )); do
  active=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' || true)
  if [[ -z "$active" ]]; then
    idle_checks=$((idle_checks + 1))
  else
    idle_checks=0
  fi
  sleep 10
done
rm -f "$eff_root/WAITING_FOR_EXCLUSIVE_GPU"
printf '%s\n' "exclusive measurement started at $(date -Is)" >"$eff_root/RUNNING"

flock /tmp/lnet-pro6000-efficiency.lock \
  python -m lnet.pac_matched_efficiency worker --root "$eff_root" --device cuda \
  >"$log_root/efficiency-worker.log" 2>&1
python -m lnet.pac_matched_efficiency report --root "$eff_root" \
  >"$log_root/efficiency-report.log" 2>&1
rm -f "$eff_root/RUNNING"
printf '%s\n' "all follow-ups complete at $(date -Is)" >"$log_root/COMPLETE"
trap - ERR
