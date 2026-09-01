#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly PROJECT_ROOT
readonly RUNTIME="${PROJECT_ROOT}/kau/alphabet_lm_4090_laplace_ssd_mamba2/campaign.runtime.json"
readonly PYTHON="${KAU_ALPHABET_PYTHON:-/home/daehwa/lnet-cffn-benchmark-20260808T091010Z/.venv/bin/python3}"
readonly OUTPUT_ROOT="${KAU_OUTPUT_ROOT:-/home/daehwa/alphabet-lm-4090-laplace-ssd-mamba2-runs/${KAU_EXPECTED_COMMIT:0:16}}"
readonly DATA_ROOT="/home/daehwa/alphabet-lm-data-fineweb-edu-v1"
readonly LABEL="laplace-ssd-mamba2-p8-l18-fromscratch-30m"

cd "${PROJECT_ROOT}"
if [[ ! "${KAU_EXPECTED_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]] \
  || [[ "$(git rev-parse HEAD)" != "${KAU_EXPECTED_COMMIT}" ]]; then
  echo "ERROR: invalid Laplace-SSD Mamba-2 execution commit" >&2
  exit 2
fi
python3 kau/alphabet_lm_4090_laplace_ssd_mamba2/generate_contract.py --check
mkdir -p "${OUTPUT_ROOT}"
exec 9>"${OUTPUT_ROOT}/queue.lock"
flock -n 9 || { echo "ERROR: Laplace-SSD Mamba-2 already running" >&2; exit 2; }

export H200_EXPECTED_COMMIT="${KAU_EXPECTED_COMMIT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/scripts"
export CUDA_VISIBLE_DEVICES=0 CUDA_MODULE_LOADING=LAZY
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR="${OUTPUT_ROOT}/cache/torchinductor"
export TRITON_CACHE_DIR="${OUTPUT_ROOT}/cache/triton"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export WANDB_BASE_URL="https://api.wandb.ai" WANDB_ENTITY="daehwa"
export WANDB_PROJECT="alphabet-lm-viability"
export WANDB_GROUP="ALPHABET-LM-RTX4090-LaplaceSSD-Mamba2-P8-30M-S501-v1"
export WANDB_CONSOLE=off

timeout --signal=TERM --kill-after=5m 8h \
  "${PYTHON}" scripts/train_h200_alphabet_lm_10m.py \
  --model mamba2_laplace_ssd --run-label "${LABEL}" \
  --laplace-mamba-layers 18 --laplace-mamba-poles 8 \
  --laplace-mamba-conv-width 4 --laplace-mamba-parallel-static-scan \
  --target-tokens-override 30000000 --runtime "${RUNTIME}" \
  --train-manifest "${DATA_ROOT}/tokens/train.manifest.json" \
  --validation-manifest "${DATA_ROOT}/tokens/validation.manifest.json" \
  --root "${OUTPUT_ROOT}/${LABEL}"

timeout --signal=TERM --kill-after=3m 30m \
  "${PYTHON}" scripts/evaluate_kau_alphabet_lm_context.py \
  --kind mamba2_laplace_ssd --checkpoint "${OUTPUT_ROOT}/${LABEL}/checkpoint.pt" \
  --validation-manifest "${DATA_ROOT}/tokens/validation.manifest.json" \
  --sequence-limit 128 --output "${OUTPUT_ROOT}/${LABEL}/context.json"
