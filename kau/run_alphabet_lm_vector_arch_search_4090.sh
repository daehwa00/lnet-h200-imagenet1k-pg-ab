#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly PROJECT_ROOT
readonly RUNTIME="${PROJECT_ROOT}/kau/alphabet_lm_4090_vector_arch_search/campaign.runtime.json"
readonly PYTHON="${KAU_ALPHABET_PYTHON:-/home/daehwa/lnet-cffn-benchmark-20260808T091010Z/.venv/bin/python3}"
cd "${PROJECT_ROOT}"
if [[ ! "${KAU_EXPECTED_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]] \
  || [[ "$(git rev-parse HEAD)" != "${KAU_EXPECTED_COMMIT}" ]]; then
  echo "ERROR: invalid vector-search execution commit" >&2
  exit 2
fi
python3 kau/alphabet_lm_4090_vector_arch_search/generate_contract.py --check

readonly OUTPUT_ROOT="${KAU_OUTPUT_ROOT:-/home/daehwa/alphabet-lm-4090-vector-arch-search-runs/${KAU_EXPECTED_COMMIT:0:16}}"
readonly DATA_ROOT="/home/daehwa/alphabet-lm-data-fineweb-edu-v1"
readonly TRAIN_MANIFEST="${DATA_ROOT}/tokens/train.manifest.json"
readonly VALIDATION_MANIFEST="${DATA_ROOT}/tokens/validation.manifest.json"
mkdir -p "${OUTPUT_ROOT}"
exec 9>"${OUTPUT_ROOT}/queue.lock"
flock -n 9 || { echo "ERROR: vector-search queue is already running" >&2; exit 2; }

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
export WANDB_GROUP="ALPHABET-LM-RTX4090-VectorArchitectureSearch-S501-v1"
export WANDB_CONSOLE=off

common=(
  --model alphabet
  --reader-type dense_k3
  --memory-layout local_only
  --cnn-pole-memory
  --cnn-pole-interval 2
  --cnn-pole-modes 128
  --cnn-pole-evidence-width 512
  --cnn-pole-kernel-size 4
  --cnn-pole-beta-initial 0.01
  --no-cnn-pole-use-recurrence
  --cnn-pole-minimum-half-life 8
  --cnn-pole-maximum-half-life 4096
  --slow-cnn-pole-memory
  --slow-cnn-pole-modes 128
  --slow-cnn-pole-evidence-width 512
  --slow-cnn-pole-kernel-size 4
  --slow-cnn-pole-upper-blocks 4
  --slow-cnn-pole-beta-initial 0.01
  --slow-cnn-pole-use-recurrence
  --slow-cnn-pole-query token
  --slow-cnn-pole-query-rho 0.5
  --slow-cnn-pole-vector-width 16
  --slow-cnn-pole-complex-vector-excitation
  --slow-cnn-pole-complex-vector-query
  --target-tokens-override 4000000
  --runtime "${RUNTIME}"
  --train-manifest "${TRAIN_MANIFEST}"
  --validation-manifest "${VALIDATION_MANIFEST}"
)

run_variant() {
  local label="$1" stride="$2" min_half="$3" max_half="$4" coordinate="$5"
  local -a coordinate_arg=()
  if [[ "${coordinate}" == true ]]; then
    coordinate_arg=(--slow-cnn-pole-coordinate-read)
  fi
  timeout --signal=TERM --kill-after=5m 4h \
    "${PYTHON}" scripts/train_h200_alphabet_lm_10m.py \
    --run-label "${label}" \
    --slow-cnn-pole-stride "${stride}" \
    --slow-cnn-pole-minimum-half-life "${min_half}" \
    --slow-cnn-pole-maximum-half-life "${max_half}" \
    "${coordinate_arg[@]}" \
    "${common[@]}" \
    --root "${OUTPUT_ROOT}/${label}"
}

run_variant complexq-r16-e2e-4m 16 1 256 false
run_variant token-rate-r16-e2e-4m 1 16 4096 false
run_variant coordinate-read-r16-e2e-4m 16 1 256 true

best="$(${PYTHON} - "${OUTPUT_ROOT}" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
kinds = {
    "complexq-r16-e2e-4m": "alphabet2_complex_query_r16",
    "token-rate-r16-e2e-4m": "alphabet2_token_rate_r16",
    "coordinate-read-r16-e2e-4m": "alphabet2_coordinate_read_r16",
}
label = min(kinds, key=lambda name: json.loads((root / name / "completed.json").read_text())["validation_loss"])
print(f"{label}|{kinds[label]}")
PY
)"
best_label="${best%%|*}"
best_kind="${best#*|}"
timeout --signal=TERM --kill-after=2m 30m \
  "${PYTHON}" scripts/evaluate_kau_alphabet_lm_context.py \
  --kind "${best_kind}" \
  --checkpoint "${OUTPUT_ROOT}/${best_label}/checkpoint.pt" \
  --validation-manifest "${VALIDATION_MANIFEST}" \
  --sequence-limit 512 \
  --output "${OUTPUT_ROOT}/${best_label}/context.json"

echo "KAU_ALPHABET_LM_VECTOR_ARCH_SEARCH_COMPLETE=${OUTPUT_ROOT};best=${best_label}"
