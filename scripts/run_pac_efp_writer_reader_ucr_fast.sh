#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-.}"
ROOT="${ROOT:-.omx/results/pac-efp-writer-reader-fast-20260718/ucr}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PYTHONPATH_ROOT="${PYTHONPATH_ROOT:-src}"
GPU_IDS="${GPU_IDS:-0}"
PARALLEL_PER_GPU="${PARALLEL_PER_GPU:-6}"
SHARD_COUNT="${SHARD_COUNT:-1}"
SHARD_IDS="${SHARD_IDS:-0}"

cd "$PROJECT_ROOT"
mkdir -p "$ROOT/logs" "$ROOT/completed" "$ROOT/failed"

IFS=',' read -r -a gpu_ids <<<"$GPU_IDS"
IFS=',' read -r -a shard_ids <<<"$SHARD_IDS"

owns_shard() {
  local candidate="$1"
  local owned
  for owned in "${shard_ids[@]}"; do
    if [[ "$candidate" == "$owned" ]]; then
      return 0
    fi
  done
  return 1
}

run_job() {
  local gpu="$1"
  local dataset="$2"
  local seed="$3"
  local variant="$4"
  local name="${dataset}_${variant}_seed${seed}"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$PYTHONPATH_ROOT" "$PYTHON_BIN" \
    -m lnet.pac_efp_writer_reader_screen \
    --dataset "$dataset" \
    --seed "$seed" \
    --variant "$variant" \
    --output-root "$ROOT" \
    --device cuda \
    >"$ROOT/logs/${name}.log" 2>&1
}
export -f run_job
export ROOT PYTHON_BIN PYTHONPATH_ROOT

datasets=(
  ArrowHead CinCECGTorso CricketX ECG200 ECG5000 ECGFiveDays Earthquakes
  GunPoint ItalyPowerDemand MoteStrain Phoneme Plane StarLightCurves Trace
  TwoLeadECG Wafer
)
seeds=(7 11 19 23 31)
variants=(efp16 zero_state full_state)

queue_files=()
for gpu in "${gpu_ids[@]}"; do
  queue="$ROOT/queue-gpu${gpu}.txt"
  : >"$queue"
  queue_files+=("$queue")
done

index=0
owned_index=0
for dataset in "${datasets[@]}"; do
  for seed in "${seeds[@]}"; do
    for variant in "${variants[@]}"; do
      shard=$((index % SHARD_COUNT))
      if owns_shard "$shard"; then
        queue_index=$((owned_index % ${#gpu_ids[@]}))
        printf '%s %s %s\n' "$dataset" "$seed" "$variant" \
          >>"${queue_files[$queue_index]}"
        ((owned_index += 1))
      fi
      ((index += 1))
    done
  done
done

workers=()
for queue_index in "${!queue_files[@]}"; do
  gpu="${gpu_ids[$queue_index]}"
  queue="${queue_files[$queue_index]}"
  xargs -r -n3 -P"$PARALLEL_PER_GPU" bash -c 'run_job "$0" "$@"' "$gpu" <"$queue" &
  workers+=("$!")
done
for worker in "${workers[@]}"; do
  wait "$worker"
done
