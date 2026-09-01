#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly PROJECT_ROOT
readonly RUNTIME="${PROJECT_ROOT}/kau/alphabet_lm_4090_content_preserving_image_postfusion/campaign.runtime.json"
readonly PYTHON="${KAU_ALPHABET_PYTHON:-/home/daehwa/lnet-cffn-benchmark-20260808T091010Z/.venv/bin/python3}"
readonly OUTPUT_ROOT="${KAU_OUTPUT_ROOT:-/home/daehwa/alphabet-lm-4090-content-preserving-runs/${KAU_EXPECTED_COMMIT:0:16}}"
readonly DATA_ROOT="/home/daehwa/alphabet-lm-data-fineweb-edu-v1"
readonly TRAIN_MANIFEST="${DATA_ROOT}/tokens/train.manifest.json"
readonly VALIDATION_MANIFEST="${DATA_ROOT}/tokens/validation.manifest.json"
readonly LABEL="content-preserving-fullroute-h4p8k16-l19-fromscratch-100m"

cd "${PROJECT_ROOT}"
if [[ ! "${KAU_EXPECTED_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]] \
  || [[ "$(git rev-parse HEAD)" != "${KAU_EXPECTED_COMMIT}" ]]; then
  echo "ERROR: invalid content-preserving execution commit" >&2
  exit 2
fi
python3 kau/alphabet_lm_4090_content_preserving_image_postfusion/generate_contract.py --check
mkdir -p "${OUTPUT_ROOT}"
exec 9>"${OUTPUT_ROOT}/queue.lock"
flock -n 9 || { echo "ERROR: content-preserving campaign already running" >&2; exit 2; }

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
export WANDB_GROUP="ALPHABET-LM-RTX4090-ContentPreserving-FullRoute-H4P8K16-100M-S501-v2"
export WANDB_CONSOLE=off

timeout --signal=TERM --kill-after=5m 8h \
  "${PYTHON}" scripts/train_h200_alphabet_lm_10m.py \
  --model alphabet2_content_preserving_image_postfusion \
  --run-label "${LABEL}" \
  --laplace-mamba-layers 19 \
  --laplace-mamba-poles 32 \
  --laplace-mamba-head-width 16 \
  --laplace-mamba-conv-width 3 \
  --laplace-mamba-content-preserving-heads 4 \
  --laplace-mamba-content-preserving-poles 8 \
  --laplace-mamba-content-preserving-width 16 \
  --target-tokens-override 100000000 \
  --runtime "${RUNTIME}" \
  --train-manifest "${TRAIN_MANIFEST}" \
  --validation-manifest "${VALIDATION_MANIFEST}" \
  --root "${OUTPUT_ROOT}/${LABEL}"

timeout --signal=TERM --kill-after=3m 30m \
  "${PYTHON}" scripts/evaluate_kau_alphabet_lm_context.py \
  --kind alphabet2_content_preserving_image_postfusion \
  --checkpoint "${OUTPUT_ROOT}/${LABEL}/checkpoint.pt" \
  --validation-manifest "${VALIDATION_MANIFEST}" \
  --sequence-limit 128 \
  --output "${OUTPUT_ROOT}/${LABEL}/context.json"

echo "KAU_CONTENT_PRESERVING_ALPHABET2_COMPLETE=${OUTPUT_ROOT}"
