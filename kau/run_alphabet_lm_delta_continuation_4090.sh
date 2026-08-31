#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly PROJECT_ROOT
readonly RUNTIME="${PROJECT_ROOT}/kau/alphabet_lm_4090_delta_continuation/campaign.runtime.json"
readonly PYTHON="${KAU_ALPHABET_PYTHON:-/home/daehwa/lnet-cffn-benchmark-20260808T091010Z/.venv/bin/python3}"
readonly OUTPUT_ROOT="${KAU_OUTPUT_ROOT:-/home/daehwa/alphabet-lm-4090-delta-continuation-runs/${KAU_EXPECTED_COMMIT:0:16}}"
readonly DATA_ROOT="/home/daehwa/alphabet-lm-data-fineweb-edu-v1"
readonly TRAIN_MANIFEST="${DATA_ROOT}/tokens/train.manifest.json"
readonly VALIDATION_MANIFEST="${DATA_ROOT}/tokens/validation.manifest.json"
readonly SOURCE_CHECKPOINT="/home/daehwa/alphabet-lm-4090-vector-image-postfusion-alphabet2-runs/9269556add319c85/vector-image-postfusion-alphabet2-p32r16-l19-fromscratch-100m/checkpoint.pt"

cd "${PROJECT_ROOT}"
if [[ ! "${KAU_EXPECTED_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]] \
  || [[ "$(git rev-parse HEAD)" != "${KAU_EXPECTED_COMMIT}" ]]; then
  echo "ERROR: invalid delta-continuation execution commit" >&2
  exit 2
fi
python3 kau/alphabet_lm_4090_delta_continuation/generate_contract.py --check
if [[ "$(sha256sum "${SOURCE_CHECKPOINT}" | cut -d " " -f 1)" \
  != "3b56bd8e61c53b652982d1e2413af7eba3ad2fa54d7317591244a396504cecc7" ]]; then
  echo "ERROR: Dense source checkpoint digest changed" >&2
  exit 2
fi
mkdir -p "${OUTPUT_ROOT}"
exec 9>"${OUTPUT_ROOT}/queue.lock"
flock -n 9 || { echo "ERROR: delta-continuation campaign already active" >&2; exit 2; }

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
export WANDB_GROUP="ALPHABET-LM-RTX4090-DeltaContinuation-30M-S501-v1"
export WANDB_CONSOLE=off

run_one() {
  local model="$1"
  local label="$2"
  timeout --signal=TERM --kill-after=5m 2h \
    "${PYTHON}" scripts/train_h200_alphabet_lm_10m.py \
    --model "${model}" \
    --run-label "${label}" \
    --laplace-mamba-layers 19 \
    --laplace-mamba-poles 32 \
    --laplace-mamba-state-size 4 \
    --laplace-mamba-head-width 16 \
    --laplace-mamba-delta-hidden 32 \
    --laplace-mamba-delta-log-bound 0.6931471805599453 \
    --laplace-mamba-conv-width 3 \
    --initialize-laplace-continuation-checkpoint "${SOURCE_CHECKPOINT}" \
    --freeze-laplace-poles \
    --target-tokens-override 30000000 \
    --runtime "${RUNTIME}" \
    --train-manifest "${TRAIN_MANIFEST}" \
    --validation-manifest "${VALIDATION_MANIFEST}" \
    --root "${OUTPUT_ROOT}/${label}"
}

run_one alphabet2_vector_image_postfusion dense-frozen-pole-static-continuation-30m
run_one alphabet2_dynamic_delta_image_postfusion dense-frozen-pole-dynamic-delta-continuation-30m

for variant in static dynamic; do
  label="dense-frozen-pole-static-continuation-30m"
  kind="alphabet2_vector_image_postfusion"
  if [[ "${variant}" == "dynamic" ]]; then
    label="dense-frozen-pole-dynamic-delta-continuation-30m"
    kind="alphabet2_dynamic_delta_image_postfusion"
  fi
  timeout --signal=TERM --kill-after=3m 30m \
    "${PYTHON}" scripts/evaluate_kau_alphabet_lm_context.py \
    --kind "${kind}" \
    --checkpoint "${OUTPUT_ROOT}/${label}/checkpoint.pt" \
    --validation-manifest "${VALIDATION_MANIFEST}" \
    --sequence-limit 128 \
    --output "${OUTPUT_ROOT}/${label}/context.json"
done

echo "KAU_DELTA_CONTINUATION_COMPLETE=${OUTPUT_ROOT}"
