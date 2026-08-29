#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly PROJECT_ROOT
readonly RUNTIME="${PROJECT_ROOT}/kau/alphabet_lm_4090_factorized_p32r32/campaign.runtime.json"
readonly PYTHON="${KAU_ALPHABET_PYTHON:-/home/daehwa/lnet-cffn-benchmark-20260808T091010Z/.venv/bin/python3}"
cd "${PROJECT_ROOT}"
if [[ ! "${KAU_EXPECTED_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]] \
  || [[ "$(git rev-parse HEAD)" != "${KAU_EXPECTED_COMMIT}" ]]; then
  echo "ERROR: invalid factorized P32R32 execution commit" >&2
  exit 2
fi
python3 kau/alphabet_lm_4090_factorized_p32r32/generate_contract.py --check

readonly OUTPUT_ROOT="${KAU_OUTPUT_ROOT:-/home/daehwa/alphabet-lm-4090-factorized-p32r32-runs/${KAU_EXPECTED_COMMIT:0:16}}"
readonly DATA_ROOT="/home/daehwa/alphabet-lm-data-fineweb-edu-v1"
readonly TRAIN_MANIFEST="${DATA_ROOT}/tokens/train.manifest.json"
readonly VALIDATION_MANIFEST="${DATA_ROOT}/tokens/validation.manifest.json"
readonly SOURCE="/home/daehwa/alphabet-lm-4090-repeated-vector-pole-30m-runs/29d0c393a1f22e2e/repeated-vector-pole-30m/checkpoint.pt"
readonly EXPECTED_SHA="b0802a7e3036ad49c036d35b4aab79771e28fd3624381457d44096f982b8ea6e"
[[ "$(sha256sum "${SOURCE}" | cut -d' ' -f1)" == "${EXPECTED_SHA}" ]] \
  || { echo "ERROR: repeated P32R4 source changed" >&2; exit 2; }
mkdir -p "${OUTPUT_ROOT}"
exec 9>"${OUTPUT_ROOT}/queue.lock"
flock -n 9 || { echo "ERROR: factorized P32R32 queue already running" >&2; exit 2; }

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
export WANDB_GROUP="ALPHABET-LM-RTX4090-FactorizedP32R32J4-4M-S501-v1"
export WANDB_CONSOLE=off

timeout --signal=TERM --kill-after=5m 1h \
  "${PYTHON}" scripts/train_h200_alphabet_lm_10m.py \
  --model alphabet \
  --run-label factorized-p32r32-j4-frozen-4m \
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
  --repeated-vector-pole-synthesis-rank 16 \
  --repeated-vector-pole-activation-checkpoint \
  --initialize-repeated-factorized-checkpoint "${SOURCE}" \
  --target-tokens-override 4000000 \
  --runtime "${RUNTIME}" \
  --train-manifest "${TRAIN_MANIFEST}" \
  --validation-manifest "${VALIDATION_MANIFEST}" \
  --root "${OUTPUT_ROOT}/factorized-p32r32-4m"

timeout --signal=TERM --kill-after=3m 45m \
  "${PYTHON}" scripts/evaluate_kau_alphabet_lm_context.py \
  --kind alphabet2_factorized_vector_pole_p32r32 \
  --checkpoint "${OUTPUT_ROOT}/factorized-p32r32-4m/checkpoint.pt" \
  --validation-manifest "${VALIDATION_MANIFEST}" \
  --sequence-limit 512 \
  --output "${OUTPUT_ROOT}/factorized-p32r32-4m/context.json"

echo "KAU_ALPHABET_LM_FACTORIZED_P32R32_COMPLETE=${OUTPUT_ROOT}"
