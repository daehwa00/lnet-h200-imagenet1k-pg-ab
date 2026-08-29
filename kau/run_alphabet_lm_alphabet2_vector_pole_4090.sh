#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly PROJECT_ROOT
readonly RUNTIME="${PROJECT_ROOT}/kau/alphabet_lm_4090_alphabet2_vector_pole/campaign.runtime.json"
readonly PYTHON="${KAU_ALPHABET_PYTHON:-/home/daehwa/lnet-cffn-benchmark-20260808T091010Z/.venv/bin/python3}"
cd "${PROJECT_ROOT}"
if [[ ! "${KAU_EXPECTED_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]] \
  || [[ "$(git rev-parse HEAD)" != "${KAU_EXPECTED_COMMIT}" ]]; then
  echo "ERROR: invalid vector-pole execution commit" >&2
  exit 2
fi
python3 kau/alphabet_lm_4090_alphabet2_vector_pole/generate_contract.py --check

readonly OUTPUT_ROOT="${KAU_OUTPUT_ROOT:-/home/daehwa/alphabet-lm-4090-vector-pole-r4-runs/${KAU_EXPECTED_COMMIT:0:16}}"
readonly DATA_ROOT="/home/daehwa/alphabet-lm-data-fineweb-edu-v1"
readonly TRAIN_MANIFEST="${DATA_ROOT}/tokens/train.manifest.json"
readonly VALIDATION_MANIFEST="${DATA_ROOT}/tokens/validation.manifest.json"
readonly TOKEN_Q_10M="/home/daehwa/alphabet-lm-4090-token-q-10m-runs/a773e8654408214e/token-q-10m/checkpoint.pt"
[[ -f "${TRAIN_MANIFEST}" && -f "${VALIDATION_MANIFEST}" && -f "${TOKEN_Q_10M}" ]] \
  || { echo "ERROR: vector-pole inputs are missing" >&2; exit 2; }
mkdir -p "${OUTPUT_ROOT}"
exec 9>"${OUTPUT_ROOT}/queue.lock"
flock -n 9 || { echo "ERROR: vector-pole queue is already running" >&2; exit 2; }
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
export WANDB_GROUP="ALPHABET-LM-RTX4090-VectorPoleR4-S501-v1"
export WANDB_CONSOLE=off

run_stage() {
  local label="$1" source="$2" target="$3" root="$4"
  local -a resume_args=()
  if [[ "${source}" != none ]]; then
    resume_args=(--resume-extension-checkpoint "${source}")
  fi
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
    --slow-cnn-pole-use-recurrence \
    --slow-cnn-pole-minimum-half-life 1 \
    --slow-cnn-pole-maximum-half-life 256 \
    --slow-cnn-pole-query token \
    --slow-cnn-pole-query-rho 0.5 \
    --slow-cnn-pole-vector-width 4 \
    --initialize-slow-vector-pole-checkpoint "${TOKEN_Q_10M}" \
    "${resume_args[@]}" \
    --target-tokens-override "${target}" \
    --runtime "${RUNTIME}" \
    --train-manifest "${TRAIN_MANIFEST}" \
    --validation-manifest "${VALIDATION_MANIFEST}" \
    --root "${root}"
  if [[ ! -f "${root}/context.json" ]]; then
    timeout --signal=TERM --kill-after=2m 30m \
      "${PYTHON}" scripts/evaluate_kau_alphabet_lm_context.py \
      --kind alphabet2_vector_pole_r4 \
      --checkpoint "${root}/checkpoint.pt" \
      --validation-manifest "${VALIDATION_MANIFEST}" \
      --sequence-limit 512 \
      --output "${root}/context.json"
  fi
}

run_stage alphabet2-vector-pole-r4-1m none 1000000 "${OUTPUT_ROOT}/vector-pole-1m"
run_stage alphabet2-vector-pole-r4-2m \
  "${OUTPUT_ROOT}/vector-pole-1m/checkpoint.pt" 2000000 \
  "${OUTPUT_ROOT}/vector-pole-2m"
run_stage alphabet2-vector-pole-r4-4m \
  "${OUTPUT_ROOT}/vector-pole-2m/checkpoint.pt" 4000000 \
  "${OUTPUT_ROOT}/vector-pole-4m"
echo "KAU_ALPHABET_LM_VECTOR_POLE_R4_COMPLETE=${OUTPUT_ROOT}"
