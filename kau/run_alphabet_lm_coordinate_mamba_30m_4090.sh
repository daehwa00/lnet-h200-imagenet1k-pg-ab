#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly PROJECT_ROOT
readonly RUNTIME="${PROJECT_ROOT}/kau/alphabet_lm_4090_coordinate_mamba_30m/campaign.runtime.json"
readonly PYTHON="${KAU_ALPHABET_PYTHON:-/home/daehwa/lnet-cffn-benchmark-20260808T091010Z/.venv/bin/python3}"
cd "${PROJECT_ROOT}"
if [[ ! "${KAU_EXPECTED_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]] \
  || [[ "$(git rev-parse HEAD)" != "${KAU_EXPECTED_COMMIT}" ]]; then
  echo "ERROR: invalid coordinate/Mamba 30M execution commit" >&2
  exit 2
fi
python3 kau/alphabet_lm_4090_coordinate_mamba_30m/generate_contract.py --check

readonly OUTPUT_ROOT="${KAU_OUTPUT_ROOT:-/home/daehwa/alphabet-lm-4090-coordinate-mamba-30m-runs/${KAU_EXPECTED_COMMIT:0:16}}"
readonly DATA_ROOT="/home/daehwa/alphabet-lm-data-fineweb-edu-v1"
readonly TRAIN_MANIFEST="${DATA_ROOT}/tokens/train.manifest.json"
readonly VALIDATION_MANIFEST="${DATA_ROOT}/tokens/validation.manifest.json"
readonly COORDINATE_4M="/home/daehwa/alphabet-lm-4090-vector-arch-search-runs/aedab1db2a734506/coordinate-read-r16-e2e-4m/checkpoint.pt"
readonly MAMBA_4M="/home/daehwa/alphabet-lm-4090-r16-mamba-4m-runs/90589bb568433949/mamba-e2e-4m/checkpoint.pt"
readonly EXPECTED_COORDINATE_SHA="843d49a013c05dfe42984c4b920c140198d2a7b673a0245aee1bed7af8304050"
readonly EXPECTED_MAMBA_SHA="64f09e24395f0eda95483cfbba8ca8d43a6a08d5fc057b969d1367c4985d979a"
[[ "$(sha256sum "${COORDINATE_4M}" | cut -d' ' -f1)" == "${EXPECTED_COORDINATE_SHA}" ]] \
  || { echo "ERROR: coordinate source checkpoint changed" >&2; exit 2; }
[[ "$(sha256sum "${MAMBA_4M}" | cut -d' ' -f1)" == "${EXPECTED_MAMBA_SHA}" ]] \
  || { echo "ERROR: Mamba source checkpoint changed" >&2; exit 2; }
mkdir -p "${OUTPUT_ROOT}"
exec 9>"${OUTPUT_ROOT}/queue.lock"
flock -n 9 || { echo "ERROR: coordinate/Mamba 30M queue already running" >&2; exit 2; }

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
export WANDB_GROUP="ALPHABET-LM-RTX4090-Coordinate-vs-Mamba-30M-S501-v1"
export WANDB_CONSOLE=off

common=(
  --runtime "${RUNTIME}"
  --train-manifest "${TRAIN_MANIFEST}"
  --validation-manifest "${VALIDATION_MANIFEST}"
  --target-tokens-override 30000000
)

timeout --signal=TERM --kill-after=5m 1h \
  "${PYTHON}" scripts/train_h200_alphabet_lm_10m.py \
  --model alphabet \
  --run-label coordinate-read-r16-e2e-30m \
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
  --slow-cnn-pole-vector-width 16 \
  --slow-cnn-pole-complex-vector-excitation \
  --slow-cnn-pole-complex-vector-query \
  --slow-cnn-pole-coordinate-read \
  --resume-extension-checkpoint "${COORDINATE_4M}" \
  "${common[@]}" \
  --root "${OUTPUT_ROOT}/coordinate-30m"

timeout --signal=TERM --kill-after=5m 1h \
  "${PYTHON}" scripts/train_h200_alphabet_lm_10m.py \
  --model mamba \
  --run-label mamba-standard-e2e-30m \
  --resume-extension-checkpoint "${MAMBA_4M}" \
  "${common[@]}" \
  --root "${OUTPUT_ROOT}/mamba-30m"

echo "KAU_ALPHABET_LM_COORDINATE_MAMBA_30M_COMPLETE=${OUTPUT_ROOT}"
