#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly PROJECT_ROOT
readonly RUNTIME="${PROJECT_ROOT}/kau/alphabet_lm_4090_retained_factor_state/campaign.runtime.json"
readonly PYTHON="${KAU_ALPHABET_PYTHON:-/home/daehwa/lnet-cffn-benchmark-20260808T091010Z/.venv/bin/python3}"
readonly OUTPUT_ROOT="${KAU_OUTPUT_ROOT:-/home/daehwa/alphabet-lm-4090-retained-factor-state-runs/${KAU_EXPECTED_COMMIT:0:16}}"
readonly DATA_ROOT="/home/daehwa/alphabet-lm-data-fineweb-edu-v1"
readonly TRAIN_MANIFEST="${DATA_ROOT}/tokens/train.manifest.json"
readonly VALIDATION_MANIFEST="${DATA_ROOT}/tokens/validation.manifest.json"
readonly SOURCE="/home/daehwa/alphabet-lm-4090-factorized-gain-decomposition-runs/536ab7eca1559014/factorized-p32r32-js4-4m/checkpoint.pt"
readonly EXPECTED_SHA="3c26e4fec77146511b72b93bf8303de8316486e263cceb9cc299f508bb300f25"

cd "${PROJECT_ROOT}"
if [[ ! "${KAU_EXPECTED_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]] \
  || [[ "$(git rev-parse HEAD)" != "${KAU_EXPECTED_COMMIT}" ]]; then
  echo "ERROR: invalid retained-factor execution commit" >&2
  exit 2
fi
python3 kau/alphabet_lm_4090_retained_factor_state/generate_contract.py --check
[[ "$(sha256sum "${SOURCE}" | cut -d' ' -f1)" == "${EXPECTED_SHA}" ]] \
  || { echo "ERROR: retained-factor source changed" >&2; exit 2; }
mkdir -p "${OUTPUT_ROOT}"
exec 9>"${OUTPUT_ROOT}/queue.lock"
flock -n 9 || { echo "ERROR: retained-factor campaign already running" >&2; exit 2; }

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
export WANDB_GROUP="ALPHABET-LM-RTX4090-RetainedFactorState-S501-v1"
export WANDB_CONSOLE=off

run_variant() {
  local label="$1"
  local kind="$2"
  local learned="$3"
  local learned_args=()
  if [[ "${learned}" == true ]]; then
    learned_args+=(--repeated-vector-pole-learned-factor-read)
  fi
  timeout --signal=TERM --kill-after=5m 2h \
    "${PYTHON}" scripts/train_h200_alphabet_lm_10m.py \
    --model alphabet \
    --run-label "${label}" \
    --reader-type dense_k3 \
    --memory-layout local_only \
    --repeated-vector-pole-memory \
    --repeated-vector-pole-interval 1 \
    --repeated-vector-pole-modes 32 \
    --repeated-vector-pole-width 32 \
    --repeated-vector-pole-reader-kernel 3 \
    --repeated-vector-pole-beta-initial 0.01 \
    --repeated-vector-pole-minimum-half-life 16 \
    --repeated-vector-pole-maximum-half-life 4096 \
    --repeated-vector-pole-factorized \
    --repeated-vector-pole-write-rank 4 \
    --repeated-vector-pole-query-rank 4 \
    --repeated-vector-pole-synthesis-rank 4 \
    --repeated-vector-pole-retain-factor-state \
    --repeated-vector-pole-factor-read-rho 0.5 \
    --repeated-vector-pole-activation-checkpoint \
    "${learned_args[@]}" \
    --initialize-repeated-retained-checkpoint "${SOURCE}" \
    --target-tokens-override 4000000 \
    --runtime "${RUNTIME}" \
    --train-manifest "${TRAIN_MANIFEST}" \
    --validation-manifest "${VALIDATION_MANIFEST}" \
    --root "${OUTPUT_ROOT}/${label}"

  timeout --signal=TERM --kill-after=3m 30m \
    "${PYTHON}" scripts/evaluate_kau_alphabet_lm_context.py \
    --kind "${kind}" \
    --checkpoint "${OUTPUT_ROOT}/${label}/checkpoint.pt" \
    --validation-manifest "${VALIDATION_MANIFEST}" \
    --sequence-limit 128 \
    --output "${OUTPUT_ROOT}/${label}/context.json"
}

run_variant \
  retained-factor-fixed-c-p32j4r32-4m \
  alphabet2_retained_factor_fixed_p32r32_js4 false
run_variant \
  retained-factor-learned-c-p32j4r32-4m \
  alphabet2_retained_factor_learned_p32r32_js4 true

echo "KAU_ALPHABET_LM_RETAINED_FACTOR_STATE_COMPLETE=${OUTPUT_ROOT}"
