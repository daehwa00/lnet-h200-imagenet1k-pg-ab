#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly PROJECT_ROOT
readonly RUNTIME="${PROJECT_ROOT}/kau/alphabet_lm_4090_semantic_edge_extension/campaign.runtime.json"
readonly PYTHON="${KAU_ALPHABET_PYTHON:-/home/daehwa/lnet-cffn-benchmark-20260808T091010Z/.venv/bin/python3}"
cd "${PROJECT_ROOT}"
if [[ ! "${KAU_EXPECTED_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]] \
  || [[ "$(git rev-parse HEAD)" != "${KAU_EXPECTED_COMMIT}" ]]; then
  echo "ERROR: invalid semantic edge extension commit" >&2
  exit 2
fi
python3 kau/alphabet_lm_4090_semantic_edge_extension/generate_contract.py --check

readonly OUTPUT_ROOT="/home/daehwa/alphabet-lm-4090-semantic-edge-extension-runs/${KAU_EXPECTED_COMMIT:0:16}"
readonly DATA_ROOT="/home/daehwa/alphabet-lm-data-fineweb-edu-v1"
readonly TRAIN_MANIFEST="${DATA_ROOT}/tokens/train.manifest.json"
readonly VALIDATION_MANIFEST="${DATA_ROOT}/tokens/validation.manifest.json"
readonly TRUNK_CHECKPOINT="/home/daehwa/alphabet-lm-4090-local-only-10m-runs/3b958854c3c988cb/run/checkpoint.pt"
readonly RECURRENT_1M="/home/daehwa/alphabet-lm-4090-semantic-edge-runs/e6438c7c6ada2fd9/recurrent/checkpoint.pt"
readonly CONTROL_1M="/home/daehwa/alphabet-lm-4090-semantic-edge-runs/e6438c7c6ada2fd9/no-recurrence/checkpoint.pt"
[[ -f "${TRAIN_MANIFEST}" && -f "${VALIDATION_MANIFEST}" && -f "${TRUNK_CHECKPOINT}" \
   && -f "${RECURRENT_1M}" && -f "${CONTROL_1M}" ]] \
  || { echo "ERROR: semantic edge extension inputs are missing" >&2; exit 2; }
mkdir -p "${OUTPUT_ROOT}"
exec 9>"${OUTPUT_ROOT}/queue.lock"
flock -n 9 || { echo "ERROR: semantic edge extension is already running" >&2; exit 2; }
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
export WANDB_GROUP="ALPHABET-LM-RTX4090-SemanticEdgeP128-Extension-S501-v1"
export WANDB_CONSOLE=off

run_stage() {
  local label="$1" kind="$2" recurrence_flag="$3" source="$4" target="$5" root="$6"
  timeout --signal=TERM --kill-after=5m 4h \
    "${PYTHON}" scripts/train_h200_alphabet_lm_10m.py \
    --model alphabet \
    --run-label "${label}" \
    --reader-type dense_k3 \
    --memory-layout local_only \
    --semantic-edge-memory \
    --semantic-edge-stride 16 \
    --semantic-edge-pole-modes 128 \
    --semantic-edge-upper-blocks 4 \
    --semantic-edge-beta-initial 0.01 \
    "${recurrence_flag}" \
    --semantic-edge-minimum-half-life 1 \
    --semantic-edge-maximum-half-life 256 \
    --initialize-semantic-edge-trunk-checkpoint "${TRUNK_CHECKPOINT}" \
    --resume-extension-checkpoint "${source}" \
    --target-tokens-override "${target}" \
    --runtime "${RUNTIME}" \
    --train-manifest "${TRAIN_MANIFEST}" \
    --validation-manifest "${VALIDATION_MANIFEST}" \
    --root "${root}"
  if [[ ! -f "${root}/context.json" ]]; then
    timeout --signal=TERM --kill-after=2m 20m \
      "${PYTHON}" scripts/evaluate_kau_alphabet_lm_context.py \
      --kind "${kind}" \
      --checkpoint "${root}/checkpoint.pt" \
      --validation-manifest "${VALIDATION_MANIFEST}" \
      --sequence-limit 512 \
      --output "${root}/context.json"
  fi
}

run_stage semantic-edge-recurrent-2m semantic_edge_p128 \
  --semantic-edge-use-recurrence "${RECURRENT_1M}" 2000000 "${OUTPUT_ROOT}/recurrent-2m"
run_stage semantic-edge-no-recurrence-2m semantic_edge_p128_no_recurrence \
  --no-semantic-edge-use-recurrence "${CONTROL_1M}" 2000000 "${OUTPUT_ROOT}/control-2m"
run_stage semantic-edge-recurrent-4m semantic_edge_p128 \
  --semantic-edge-use-recurrence "${OUTPUT_ROOT}/recurrent-2m/checkpoint.pt" 4000000 \
  "${OUTPUT_ROOT}/recurrent-4m"
run_stage semantic-edge-no-recurrence-4m semantic_edge_p128_no_recurrence \
  --no-semantic-edge-use-recurrence "${OUTPUT_ROOT}/control-2m/checkpoint.pt" 4000000 \
  "${OUTPUT_ROOT}/control-4m"
echo "KAU_ALPHABET_LM_SEMANTIC_EDGE_EXTENSION_COMPLETE=${OUTPUT_ROOT}"
