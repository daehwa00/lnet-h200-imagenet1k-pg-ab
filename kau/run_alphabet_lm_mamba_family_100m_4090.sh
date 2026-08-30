#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly PROJECT_ROOT
readonly RUNTIME="${PROJECT_ROOT}/kau/alphabet_lm_4090_mamba_family_100m/campaign.runtime.json"
readonly PYTHON="${KAU_ALPHABET_PYTHON:-/home/daehwa/lnet-cffn-benchmark-20260808T091010Z/.venv/bin/python3}"
readonly OUTPUT_ROOT="${KAU_OUTPUT_ROOT:-/home/daehwa/alphabet-lm-4090-mamba-family-100m-runs/${KAU_EXPECTED_COMMIT:0:16}}"
readonly DATA_ROOT="/home/daehwa/alphabet-lm-data-fineweb-edu-v1"
readonly TRAIN_MANIFEST="${DATA_ROOT}/tokens/train.manifest.json"
readonly VALIDATION_MANIFEST="${DATA_ROOT}/tokens/validation.manifest.json"

cd "${PROJECT_ROOT}"
if [[ ! "${KAU_EXPECTED_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]] \
  || [[ "$(git rev-parse HEAD)" != "${KAU_EXPECTED_COMMIT}" ]]; then
  echo "ERROR: invalid Mamba family execution commit" >&2
  exit 2
fi
python3 kau/alphabet_lm_4090_mamba_family_100m/generate_contract.py --check
mkdir -p "${OUTPUT_ROOT}"
exec 9>"${OUTPUT_ROOT}/queue.lock"
flock -n 9 || { echo "ERROR: Mamba family campaign already running" >&2; exit 2; }

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
export WANDB_GROUP="ALPHABET-LM-RTX4090-MambaFamily-100M-S501-v1"
export WANDB_CONSOLE=off

common=(
  --repeated-vector-pole-memory
  --repeated-vector-pole-interval 1
  --repeated-vector-pole-modes 32
  --repeated-vector-pole-width 32
  --repeated-vector-pole-reader-kernel 3
  --repeated-vector-pole-beta-initial 0.01
  --repeated-vector-pole-minimum-half-life 16
  --repeated-vector-pole-maximum-half-life 4096
  --repeated-vector-pole-factorized
  --repeated-vector-pole-write-rank 4
  --repeated-vector-pole-query-rank 4
  --repeated-vector-pole-synthesis-rank 4
  --repeated-vector-pole-retain-factor-state
  --repeated-vector-pole-learned-factor-read
  --repeated-vector-pole-factor-read-rho 0.5
  --repeated-vector-pole-activation-checkpoint
  --target-tokens-override 100000000
  --runtime "${RUNTIME}"
  --train-manifest "${TRAIN_MANIFEST}"
  --validation-manifest "${VALIDATION_MANIFEST}"
)

run_variant() {
  local model="$1"
  local label="$2"
  local kind="$3"
  timeout --signal=TERM --kill-after=5m 6h \
    "${PYTHON}" scripts/train_h200_alphabet_lm_10m.py \
    --model "${model}" \
    --run-label "${label}" \
    "${common[@]}" \
    --root "${OUTPUT_ROOT}/${label}"
  timeout --signal=TERM --kill-after=3m 30m \
    "${PYTHON}" scripts/evaluate_kau_alphabet_lm_context.py \
    --kind "${kind}" \
    --checkpoint "${OUTPUT_ROOT}/${label}/checkpoint.pt" \
    --validation-manifest "${VALIDATION_MANIFEST}" \
    --sequence-limit 128 \
    --output "${OUTPUT_ROOT}/${label}/context.json"
}

run_variant mamba mamba1-parameter-matched-fromscratch-100m mamba1_49m
run_variant mamba2 mamba2-parameter-matched-fromscratch-100m mamba2_48m

echo "KAU_ALPHABET_LM_MAMBA_FAMILY_100M_COMPLETE=${OUTPUT_ROOT}"
