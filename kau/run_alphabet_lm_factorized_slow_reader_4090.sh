#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly PROJECT_ROOT
readonly RUNTIME="${PROJECT_ROOT}/kau/alphabet_lm_4090_factorized_slow_reader/campaign.runtime.json"
readonly PYTHON="${KAU_ALPHABET_PYTHON:-/home/daehwa/lnet-cffn-benchmark-20260808T091010Z/.venv/bin/python3}"
cd "${PROJECT_ROOT}"
if [[ ! "${KAU_EXPECTED_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]] \
  || [[ "$(git rev-parse HEAD)" != "${KAU_EXPECTED_COMMIT}" ]]; then
  echo "ERROR: invalid factorized slow-reader execution commit" >&2
  exit 2
fi
python3 kau/alphabet_lm_4090_factorized_slow_reader/generate_contract.py --check

readonly OUTPUT_ROOT="/home/daehwa/alphabet-lm-4090-factorized-slow-reader-runs/${KAU_EXPECTED_COMMIT:0:16}"
readonly DATA_ROOT="/home/daehwa/alphabet-lm-data-fineweb-edu-v1"
readonly TRAIN_MANIFEST="${DATA_ROOT}/tokens/train.manifest.json"
readonly VALIDATION_MANIFEST="${DATA_ROOT}/tokens/validation.manifest.json"
readonly TRUNK_CHECKPOINT="/home/daehwa/alphabet-lm-4090-cnn-pole-runs/ebcba2aa1aceca5c/no-recurrence/checkpoint.pt"
[[ -f "${TRAIN_MANIFEST}" && -f "${VALIDATION_MANIFEST}" && -f "${TRUNK_CHECKPOINT}" ]] \
  || { echo "ERROR: factorized slow-reader inputs are missing" >&2; exit 2; }
mkdir -p "${OUTPUT_ROOT}"
exec 9>"${OUTPUT_ROOT}/queue.lock"
flock -n 9 || { echo "ERROR: factorized slow-reader queue is already running" >&2; exit 2; }
echo "$$" >"${OUTPUT_ROOT}/launcher.pid"

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
export WANDB_GROUP="ALPHABET-LM-RTX4090-FactorizedSlowReaderP128-1M-S501-v1"
export WANDB_CONSOLE=off

for rank in 2 4; do
  for recurrence in recurrent no-recurrence; do
    if [[ "${recurrence}" == recurrent ]]; then
      label="slow-r${rank}k4-recurrent-1m"
      kind="slow_r${rank}k4_p128"
      recurrence_flag="--slow-cnn-pole-use-recurrence"
    else
      label="slow-r${rank}k4-no-recurrence-1m"
      kind="slow_r${rank}k4_p128_no_recurrence"
      recurrence_flag="--no-slow-cnn-pole-use-recurrence"
    fi
    run_root="${OUTPUT_ROOT}/r${rank}-${recurrence}"
    timeout --signal=TERM --kill-after=5m 4h \
      "${PYTHON}" scripts/train_h200_alphabet_lm_10m.py \
      --model alphabet \
      --run-label "${label}" \
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
      --slow-cnn-pole-stride 16 \
      --slow-cnn-pole-modes 128 \
      --slow-cnn-pole-evidence-width 512 \
      --slow-cnn-pole-kernel-size 4 \
      --slow-cnn-pole-upper-blocks 4 \
      --slow-cnn-pole-beta-initial 0.01 \
      "${recurrence_flag}" \
      --slow-cnn-pole-minimum-half-life 1 \
      --slow-cnn-pole-maximum-half-life 256 \
      --slow-cnn-pole-reader factorized_complex \
      --slow-cnn-pole-reader-rank "${rank}" \
      --initialize-slow-cnn-pole-trunk-checkpoint "${TRUNK_CHECKPOINT}" \
      --runtime "${RUNTIME}" \
      --train-manifest "${TRAIN_MANIFEST}" \
      --validation-manifest "${VALIDATION_MANIFEST}" \
      --root "${run_root}"
    if [[ ! -f "${run_root}/context.json" ]]; then
      timeout --signal=TERM --kill-after=2m 30m \
        "${PYTHON}" scripts/evaluate_kau_alphabet_lm_context.py \
        --kind "${kind}" \
        --checkpoint "${run_root}/checkpoint.pt" \
        --validation-manifest "${VALIDATION_MANIFEST}" \
        --sequence-limit 512 \
        --output "${run_root}/context.json"
    fi
  done
done
echo "KAU_ALPHABET_LM_FACTORIZED_SLOW_READER_COMPLETE=${OUTPUT_ROOT}"
