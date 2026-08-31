#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly PROJECT_ROOT
readonly RUNTIME="${PROJECT_ROOT}/kau/alphabet_lm_4090_temporal_whitening_30m/campaign.runtime.json"
readonly PYTHON="${KAU_ALPHABET_PYTHON:-/home/daehwa/lnet-cffn-benchmark-20260808T091010Z/.venv/bin/python3}"
readonly OUTPUT_ROOT="${KAU_OUTPUT_ROOT:-/home/daehwa/alphabet-lm-4090-temporal-whitening-30m-runs/${KAU_EXPECTED_COMMIT:0:16}}"
readonly DATA_ROOT="/home/daehwa/alphabet-lm-data-fineweb-edu-v1"
readonly TRAIN_MANIFEST="${DATA_ROOT}/tokens/train.manifest.json"
readonly VALIDATION_MANIFEST="${DATA_ROOT}/tokens/validation.manifest.json"
readonly LABEL="j2-pole-axis-whitening-p32r16-l19-fromscratch-30m"

cd "${PROJECT_ROOT}"
if [[ ! "${KAU_EXPECTED_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]] \
  || [[ "$(git rev-parse HEAD)" != "${KAU_EXPECTED_COMMIT}" ]]; then
  echo "ERROR: invalid temporal-whitening execution commit" >&2
  exit 2
fi
python3 kau/alphabet_lm_4090_temporal_whitening_30m/generate_contract.py --check
mkdir -p "${OUTPUT_ROOT}"
exec 9>"${OUTPUT_ROOT}/queue.lock"
flock -n 9 || { echo "ERROR: temporal-whitening run already active" >&2; exit 2; }

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
export WANDB_GROUP="ALPHABET-LM-RTX4090-TemporalWhitening-30M-S501-v1"
export WANDB_CONSOLE=off

timeout --signal=TERM --kill-after=5m 4h \
  "${PYTHON}" scripts/train_h200_alphabet_lm_10m.py \
  --model alphabet2_temporal_whitening_image_postfusion \
  --run-label "${LABEL}" \
  --laplace-mamba-layers 19 \
  --laplace-mamba-poles 32 \
  --laplace-mamba-state-size 4 \
  --laplace-mamba-head-width 16 \
  --laplace-mamba-content-rank 2 \
  --laplace-mamba-conv-width 3 \
  --target-tokens-override 30000000 \
  --runtime "${RUNTIME}" \
  --train-manifest "${TRAIN_MANIFEST}" \
  --validation-manifest "${VALIDATION_MANIFEST}" \
  --root "${OUTPUT_ROOT}/${LABEL}"

timeout --signal=TERM --kill-after=3m 30m \
  "${PYTHON}" scripts/evaluate_kau_alphabet_lm_context.py \
  --kind alphabet2_temporal_whitening_image_postfusion \
  --checkpoint "${OUTPUT_ROOT}/${LABEL}/checkpoint.pt" \
  --validation-manifest "${VALIDATION_MANIFEST}" \
  --sequence-limit 128 \
  --output "${OUTPUT_ROOT}/${LABEL}/context.json"

echo "KAU_TEMPORAL_WHITENING_30M_COMPLETE=${OUTPUT_ROOT}"
