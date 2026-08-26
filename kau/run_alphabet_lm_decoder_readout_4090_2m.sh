#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly PROJECT_ROOT
readonly RUNTIME="${PROJECT_ROOT}/kau/alphabet_lm_4090_decoder_readout/campaign.runtime.json"
readonly PYTHON="${KAU_ALPHABET_PYTHON:-/home/daehwa/lnet-cffn-benchmark-20260808T091010Z/.venv/bin/python3}"
cd "${PROJECT_ROOT}"
if [[ ! "${KAU_EXPECTED_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]] \
  || [[ "$(git rev-parse HEAD)" != "${KAU_EXPECTED_COMMIT}" ]]; then
  echo "ERROR: invalid decoder/readout execution commit" >&2
  exit 2
fi
python3 kau/alphabet_lm_4090_decoder_readout/generate_contract.py --check

readonly OUTPUT_ROOT="/home/daehwa/alphabet-lm-4090-decoder-readout-runs/${KAU_EXPECTED_COMMIT:0:16}"
readonly DATA_ROOT="/home/daehwa/alphabet-lm-data-fineweb-edu-v1"
readonly TRAIN_MANIFEST="${DATA_ROOT}/tokens/train.manifest.json"
readonly VALIDATION_MANIFEST="${DATA_ROOT}/tokens/validation.manifest.json"
[[ -f "${TRAIN_MANIFEST}" && -f "${VALIDATION_MANIFEST}" ]] \
  || { echo "ERROR: shared KAU token manifests are missing" >&2; exit 2; }
mkdir -p "${OUTPUT_ROOT}"
exec 9>"${OUTPUT_ROOT}/queue.lock"
flock -n 9 || { echo "ERROR: decoder/readout queue is already running" >&2; exit 2; }
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
export WANDB_GROUP="ALPHABET-LM-RTX4090-DecoderReadout-2M-S501-v1"
export WANDB_CONSOLE=off

timeout --signal=TERM --kill-after=2m 30m \
  "${PYTHON}" scripts/smoke_kau_alphabet_lm_4090.py --only qread

failed=0
for label in alphabet-legacy-control alphabet-wide-post512 alphabet-qread-r32; do
  post_hidden=384
  memory_readout=fixed
  query_read_initial_scale=0.05
  kind=legacy
  if [[ "${label}" == "alphabet-wide-post512" ]]; then
    post_hidden=512
    kind=wide
  elif [[ "${label}" == "alphabet-qread-r32" ]]; then
    memory_readout=query_low_rank
    query_read_initial_scale=0.15
    kind=qread
  fi
  run_root="${OUTPUT_ROOT}/runs/${label}"
  set +e
  timeout --signal=TERM --kill-after=5m 4h \
    "${PYTHON}" scripts/train_h200_alphabet_lm_10m.py \
    --model alphabet \
    --run-label "${label}" \
    --post-hidden "${post_hidden}" \
    --memory-readout "${memory_readout}" \
    --query-read-rank 32 \
    --query-read-initial-scale "${query_read_initial_scale}" \
    --paired-legacy-initialization \
    --runtime "${RUNTIME}" \
    --train-manifest "${TRAIN_MANIFEST}" \
    --validation-manifest "${VALIDATION_MANIFEST}" \
    --root "${run_root}"
  status=$?
  set -e
  if (( status != 0 )); then
    echo "KAU_ALPHABET_LM_CELL_FAILED=${label}:${status}" >&2
    failed=1
    continue
  fi
  if [[ ! -f "${run_root}/context.json" ]]; then
    set +e
    timeout --signal=TERM --kill-after=2m 20m \
      "${PYTHON}" scripts/evaluate_kau_alphabet_lm_context.py \
      --kind "${kind}" \
      --checkpoint "${run_root}/checkpoint.pt" \
      --validation-manifest "${VALIDATION_MANIFEST}" \
      --sequence-limit 512 \
      --output "${run_root}/context.json"
    status=$?
    set -e
    if (( status != 0 )); then
      echo "KAU_ALPHABET_LM_CONTEXT_FAILED=${label}:${status}" >&2
      failed=1
    fi
  fi
done
echo "KAU_ALPHABET_LM_DECODER_READOUT_COMPLETE=${OUTPUT_ROOT}"
exit "${failed}"
