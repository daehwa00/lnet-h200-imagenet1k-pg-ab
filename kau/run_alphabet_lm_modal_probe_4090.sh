#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly PROJECT_ROOT
readonly PYTHON="${KAU_ALPHABET_PYTHON:-/home/daehwa/lnet-cffn-benchmark-20260808T091010Z/.venv/bin/python3}"
readonly RUNTIME="${PROJECT_ROOT}/kau/alphabet_lm_4090_modal_probe/campaign.runtime.json"
readonly LEGACY_CHECKPOINT="/home/daehwa/alphabet-lm-4090-runs/fc7fd201f2a29398/runs/alphabet-legacy/checkpoint.pt"
readonly DATA_ROOT="/home/daehwa/alphabet-lm-data-fineweb-edu-v1/tokens"
cd "${PROJECT_ROOT}"
if [[ ! "${KAU_EXPECTED_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]] \
  || [[ "$(git rev-parse HEAD)" != "${KAU_EXPECTED_COMMIT}" ]]; then
  echo "ERROR: invalid frozen modal-probe execution commit" >&2
  exit 2
fi
python3 kau/alphabet_lm_4090_modal_probe/generate_contract.py --check
[[ -f "${LEGACY_CHECKPOINT}" ]] || { echo "ERROR: Legacy checkpoint is missing" >&2; exit 2; }
readonly OUTPUT_ROOT="/home/daehwa/alphabet-lm-4090-modal-probe-runs/${KAU_EXPECTED_COMMIT:0:16}"
mkdir -p "${OUTPUT_ROOT}"
exec 9>"${OUTPUT_ROOT}/queue.lock"
flock -n 9 || { echo "ERROR: frozen modal probe is already running" >&2; exit 2; }
echo "$$" >"${OUTPUT_ROOT}/launcher.pid"

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src:${PROJECT_ROOT}/scripts"
export CUDA_VISIBLE_DEVICES=0
export CUDA_MODULE_LOADING=LAZY
export PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_ENTITY="daehwa"
export WANDB_PROJECT="alphabet-lm-viability"
export WANDB_GROUP="ALPHABET-LM-RTX4090-FrozenModalProbe-S501-v2-LowLR"
export WANDB_CONSOLE=off

while true; do
  used_mib="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
  total_mib="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
  free_mib=$((total_mib - used_mib))
  if (( free_mib >= 5500 )); then
    break
  fi
  echo "KAU_MODAL_PROBE_WAITING_FREE_MIB=${free_mib}" >&2
  sleep 30
done

timeout --signal=TERM --kill-after=2m 2h \
  "${PYTHON}" scripts/probe_kau_alphabet_lm_modal_readout.py \
  --runtime "${RUNTIME}" \
  --checkpoint "${LEGACY_CHECKPOINT}" \
  --train-manifest "${DATA_ROOT}/train.manifest.json" \
  --validation-manifest "${DATA_ROOT}/validation.manifest.json" \
  --root "${OUTPUT_ROOT}/run"
echo "KAU_ALPHABET_LM_MODAL_PROBE_COMPLETE=${OUTPUT_ROOT}"
