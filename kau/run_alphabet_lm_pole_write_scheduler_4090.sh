#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly PROJECT_ROOT
readonly RUNTIME="${PROJECT_ROOT}/kau/alphabet_lm_4090_pole_write_scheduler/campaign.runtime.json"
readonly PYTHON="${KAU_ALPHABET_PYTHON:-/home/daehwa/lnet-cffn-benchmark-20260808T091010Z/.venv/bin/python3}"
cd "${PROJECT_ROOT}"
if [[ ! "${KAU_EXPECTED_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]] \
  || [[ "$(git rev-parse HEAD)" != "${KAU_EXPECTED_COMMIT}" ]]; then
  echo "ERROR: invalid pole write-scheduler execution commit" >&2
  exit 2
fi
python3 kau/alphabet_lm_4090_pole_write_scheduler/generate_contract.py --check

readonly OUTPUT_ROOT="${KAU_OUTPUT_ROOT:-/home/daehwa/alphabet-lm-4090-pole-write-scheduler-runs/${KAU_EXPECTED_COMMIT:0:16}}"
readonly DATA_ROOT="/home/daehwa/alphabet-lm-data-fineweb-edu-v1"
readonly TRAIN_MANIFEST="${DATA_ROOT}/tokens/train.manifest.json"
readonly VALIDATION_MANIFEST="${DATA_ROOT}/tokens/validation.manifest.json"
readonly SOURCE="/home/daehwa/alphabet-lm-4090-pole-reader-30m-runs/18eda3702b9de858/pole-reader-30m/checkpoint.pt"
readonly EXPECTED_SHA="262d16303b374158822574032ea0d622d0ec305d534ceed92b3ca722e69337a5"
[[ "$(sha256sum "${SOURCE}" | cut -d' ' -f1)" == "${EXPECTED_SHA}" ]] \
  || { echo "ERROR: Dense-K3 30M source changed" >&2; exit 2; }
mkdir -p "${OUTPUT_ROOT}"
exec 9>"${OUTPUT_ROOT}/queue.lock"
flock -n 9 || { echo "ERROR: pole write-scheduler queue already running" >&2; exit 2; }

export H200_EXPECTED_COMMIT="${KAU_EXPECTED_COMMIT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/scripts"
export CUDA_VISIBLE_DEVICES=0
export CUDA_MODULE_LOADING=LAZY
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR="${OUTPUT_ROOT}/cache/torchinductor"
export TRITON_CACHE_DIR="${OUTPUT_ROOT}/cache/triton"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_ENTITY="daehwa"
export WANDB_PROJECT="alphabet-lm-viability"
export WANDB_GROUP="ALPHABET-LM-RTX4090-PoleWriteSchedulerR16-4M-S501-v1"
export WANDB_CONSOLE=off

timeout --signal=TERM --kill-after=5m 1h \
  "${PYTHON}" scripts/train_h200_alphabet_lm_10m.py \
  --model alphabet \
  --run-label pole-write-scheduler-r16-frozen-4m \
  --reader-type dense_k3 \
  --memory-layout local_only \
  --cnn-pole-memory \
  --cnn-pole-interval 2 \
  --cnn-pole-modes 128 \
  --cnn-pole-evidence-width 512 \
  --cnn-pole-kernel-size 4 \
  --cnn-pole-beta-initial 0.01 \
  --no-cnn-pole-use-recurrence \
  --cnn-pole-minimum-half-life 8 \
  --cnn-pole-maximum-half-life 4096 \
  --slow-cnn-pole-memory \
  --slow-cnn-pole-stride 1 \
  --slow-cnn-pole-modes 128 \
  --slow-cnn-pole-evidence-width 512 \
  --slow-cnn-pole-kernel-size 4 \
  --slow-cnn-pole-upper-blocks 4 \
  --slow-cnn-pole-beta-initial 0.01 \
  --slow-cnn-pole-use-recurrence \
  --slow-cnn-pole-minimum-half-life 16 \
  --slow-cnn-pole-maximum-half-life 4096 \
  --slow-cnn-pole-query token \
  --slow-cnn-pole-query-rho 0.5 \
  --slow-cnn-pole-vector-width 16 \
  --slow-cnn-pole-complex-vector-query \
  --slow-cnn-pole-coordinate-read \
  --slow-cnn-pole-specific-reader \
  --slow-cnn-pole-reader-kernel 3 \
  --slow-cnn-pole-write-scheduler \
  --initialize-slow-write-scheduler-checkpoint "${SOURCE}" \
  --target-tokens-override 4000000 \
  --runtime "${RUNTIME}" \
  --train-manifest "${TRAIN_MANIFEST}" \
  --validation-manifest "${VALIDATION_MANIFEST}" \
  --root "${OUTPUT_ROOT}/write-scheduler-4m"

timeout --signal=TERM --kill-after=2m 30m \
  "${PYTHON}" scripts/evaluate_kau_alphabet_lm_context.py \
  --kind alphabet2_pole_reader_write_scheduler_r16 \
  --checkpoint "${OUTPUT_ROOT}/write-scheduler-4m/checkpoint.pt" \
  --validation-manifest "${VALIDATION_MANIFEST}" \
  --sequence-limit 512 \
  --output "${OUTPUT_ROOT}/write-scheduler-4m/context.json"

echo "KAU_ALPHABET_LM_POLE_WRITE_SCHEDULER_COMPLETE=${OUTPUT_ROOT}"
