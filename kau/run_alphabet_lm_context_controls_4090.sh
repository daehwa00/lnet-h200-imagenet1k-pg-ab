#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly PROJECT_ROOT
readonly RUNTIME="${PROJECT_ROOT}/kau/alphabet_lm_4090_context_controls/campaign.runtime.json"
readonly PYTHON="${KAU_ALPHABET_PYTHON:-/home/daehwa/lnet-cffn-benchmark-20260808T091010Z/.venv/bin/python3}"
cd "${PROJECT_ROOT}"
if [[ ! "${KAU_EXPECTED_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]] \
  || [[ "$(git rev-parse HEAD)" != "${KAU_EXPECTED_COMMIT}" ]]; then
  echo "ERROR: invalid context-control execution commit" >&2
  exit 2
fi
python3 kau/alphabet_lm_4090_context_controls/generate_contract.py --check

readonly OUTPUT_ROOT="/home/daehwa/alphabet-lm-4090-context-control-runs/${KAU_EXPECTED_COMMIT:0:16}"
readonly DATA_ROOT="/home/daehwa/alphabet-lm-data-fineweb-edu-v1"
readonly TRAIN_MANIFEST="${DATA_ROOT}/tokens/train.manifest.json"
readonly VALIDATION_MANIFEST="${DATA_ROOT}/tokens/validation.manifest.json"
readonly MAMBA_CHECKPOINT="/home/daehwa/alphabet-lm-4090-runs/fc7fd201f2a29398/runs/mamba/checkpoint.pt"
[[ -f "${TRAIN_MANIFEST}" && -f "${VALIDATION_MANIFEST}" && -f "${MAMBA_CHECKPOINT}" ]] \
  || { echo "ERROR: context-control inputs are missing" >&2; exit 2; }
mkdir -p "${OUTPUT_ROOT}"
exec 9>"${OUTPUT_ROOT}/queue.lock"
flock -n 9 || { echo "ERROR: context-control queue is already running" >&2; exit 2; }
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
export WANDB_GROUP="ALPHABET-LM-RTX4090-ContextControls-2M-S501-v1"
export WANDB_CONSOLE=off

if [[ ! -f "${OUTPUT_ROOT}/mamba-context.json" ]]; then
  timeout --signal=TERM --kill-after=2m 20m \
    "${PYTHON}" scripts/evaluate_kau_alphabet_lm_context.py \
    --kind mamba \
    --checkpoint "${MAMBA_CHECKPOINT}" \
    --validation-manifest "${VALIDATION_MANIFEST}" \
    --sequence-limit 512 \
    --output "${OUTPUT_ROOT}/mamba-context.json"
fi

timeout --signal=TERM --kill-after=2m 30m \
  "${PYTHON}" scripts/smoke_kau_alphabet_lm_4090.py --only local_only

readonly RUN_ROOT="${OUTPUT_ROOT}/local-only"
timeout --signal=TERM --kill-after=5m 4h \
  "${PYTHON}" scripts/train_h200_alphabet_lm_10m.py \
  --model alphabet \
  --run-label alphabet-dense-k3-local-only \
  --reader-type dense_k3 \
  --memory-layout local_only \
  --paired-dense-initialization \
  --runtime "${RUNTIME}" \
  --train-manifest "${TRAIN_MANIFEST}" \
  --validation-manifest "${VALIDATION_MANIFEST}" \
  --root "${RUN_ROOT}"

if [[ ! -f "${RUN_ROOT}/context.json" ]]; then
  timeout --signal=TERM --kill-after=2m 20m \
    "${PYTHON}" scripts/evaluate_kau_alphabet_lm_context.py \
    --kind dense_local_only \
    --checkpoint "${RUN_ROOT}/checkpoint.pt" \
    --validation-manifest "${VALIDATION_MANIFEST}" \
    --sequence-limit 512 \
    --output "${RUN_ROOT}/context.json"
fi
echo "KAU_ALPHABET_LM_CONTEXT_CONTROLS_COMPLETE=${OUTPUT_ROOT}"
