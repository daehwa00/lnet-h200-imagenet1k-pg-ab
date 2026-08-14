#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-.omx/results/pac-full-late-dense-20260718/ucr}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PARALLEL_PER_GPU="${PARALLEL_PER_GPU:-6}"
mkdir -p "$ROOT/logs"

run_job() {
  local gpu="$1"
  local dataset="$2"
  local seed="$3"
  local name="${dataset}_full_late_dense_seed${seed}"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=. "$PYTHON_BIN" \
    -m lnet.pac_full_state_terminal_screen \
    --dataset "$dataset" \
    --seed "$seed" \
    --variant full_late_dense \
    --output-root "$ROOT" \
    --device cuda \
    >"$ROOT/logs/${name}.log" 2>&1
}
export -f run_job
export ROOT PYTHON_BIN

datasets=(
  ArrowHead CinCECGTorso CricketX ECG200 ECG5000 ECGFiveDays Earthquakes
  GunPoint ItalyPowerDemand MoteStrain Phoneme Plane StarLightCurves Trace
  TwoLeadECG Wafer
)
seeds=(7 11 19 23 31)
queue0="$ROOT/queue-gpu0.txt"
queue1="$ROOT/queue-gpu1.txt"
: >"$queue0"
: >"$queue1"
index=0
for dataset in "${datasets[@]}"; do
  for seed in "${seeds[@]}"; do
    queue="$queue0"
    if (( index % 2 == 1 )); then
      queue="$queue1"
    fi
    printf '%s %s\n' "$dataset" "$seed" >>"$queue"
    ((index += 1))
  done
done

xargs -n2 -P"$PARALLEL_PER_GPU" bash -c 'run_job 0 "$@"' _ <"$queue0" &
worker0=$!
xargs -n2 -P"$PARALLEL_PER_GPU" bash -c 'run_job 1 "$@"' _ <"$queue1" &
worker1=$!
wait "$worker0"
wait "$worker1"
