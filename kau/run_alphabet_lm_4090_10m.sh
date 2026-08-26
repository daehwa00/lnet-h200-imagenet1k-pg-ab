#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly PROJECT_ROOT
readonly RUNTIME="${PROJECT_ROOT}/kau/alphabet_lm_4090/campaign.runtime.json"
readonly PYTHON="${KAU_ALPHABET_PYTHON:-/home/daehwa/.venvs/alphabet-lm-4090/bin/python}"
cd "${PROJECT_ROOT}"

if [[ ! "${KAU_EXPECTED_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "ERROR: KAU_EXPECTED_COMMIT must be the immutable execution commit" >&2
  exit 2
fi
if [[ "$(git rev-parse HEAD)" != "${KAU_EXPECTED_COMMIT}" ]]; then
  echo "ERROR: KAU execution commit mismatch" >&2
  exit 2
fi
python3 kau/alphabet_lm_4090/generate_contract.py --check
[[ -x "${PYTHON}" ]] || { echo "ERROR: missing KAU Python environment" >&2; exit 2; }

readonly OUTPUT_ROOT="/home/daehwa/alphabet-lm-4090-runs/${KAU_EXPECTED_COMMIT:0:16}"
readonly DATA_ROOT="/home/daehwa/alphabet-lm-data-fineweb-edu-v1"
mkdir -p "${OUTPUT_ROOT}" "${DATA_ROOT}"
exec 9>"${OUTPUT_ROOT}/queue.lock"
flock -n 9 || { echo "ERROR: KAU ALPHABET-LM queue is already running" >&2; exit 2; }
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
export HF_HOME="${DATA_ROOT}/huggingface-cache"
export HF_XET_HIGH_PERFORMANCE=1
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_ENTITY="daehwa"
export WANDB_PROJECT="alphabet-lm-viability"
export WANDB_GROUP="ALPHABET-LM-RTX4090-PoleInit-10M-S501-v1"
export WANDB_CONSOLE=off

"${PYTHON}" - <<'PY'
import json
from importlib.metadata import version

import torch
import wandb

expected = {
    "torch": "2.9.1+cu128",
    "triton": "3.5.1",
    "mamba-ssm": "2.3.2.post1",
    "pyarrow": "23.0.1",
}
actual = {name: version(name) for name in expected}
if actual != expected or not torch.cuda.is_available() or "4090" not in torch.cuda.get_device_name():
    raise RuntimeError(f"invalid KAU ALPHABET-LM environment: {actual}")
viewer = wandb.Api(timeout=15).viewer
viewer_name = getattr(viewer, "username", str(viewer))
print("KAU_ALPHABET_LM_ENV=" + json.dumps({"packages": actual, "gpu": torch.cuda.get_device_name(), "wandb_user": viewer_name}, sort_keys=True))
PY

timeout --signal=TERM --kill-after=2m 30m \
  "${PYTHON}" scripts/smoke_kau_alphabet_lm_4090.py --only palette

timeout --signal=TERM --kill-after=5m 90m \
  "${PYTHON}" scripts/prepare_h200_alphabet_lm_data.py \
  --runtime "${RUNTIME}" --root "${DATA_ROOT}"

readonly TRAIN_MANIFEST="${DATA_ROOT}/tokens/train.manifest.json"
readonly VALIDATION_MANIFEST="${DATA_ROOT}/tokens/validation.manifest.json"
for label in alphabet-legacy alphabet-palette mamba; do
  if [[ "${label}" == "mamba" ]]; then
    model=mamba
    initialization=legacy
  elif [[ "${label}" == "alphabet-palette" ]]; then
    model=alphabet
    initialization=lifetime_palette
  else
    model=alphabet
    initialization=legacy
  fi
  timeout --signal=TERM --kill-after=5m 8h \
    "${PYTHON}" scripts/train_h200_alphabet_lm_10m.py \
    --model "${model}" \
    --run-label "${label}" \
    --pole-initialization "${initialization}" \
    --runtime "${RUNTIME}" \
    --train-manifest "${TRAIN_MANIFEST}" \
    --validation-manifest "${VALIDATION_MANIFEST}" \
    --root "${OUTPUT_ROOT}/runs/${label}"
done
echo "KAU_ALPHABET_LM_10M_COMPLETE=${OUTPUT_ROOT}"
