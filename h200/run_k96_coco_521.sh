#!/usr/bin/env bash
set -Eeuo pipefail
TASK_ROOT=/app/output/k96-coco521-v1
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1

if [[ "${1:-}" != "--inside-guard" ]]; then
  mkdir -p "${TASK_ROOT}"
  exec 9>"${TASK_ROOT}/campaign.lock"
  flock -n 9 || { echo 'K96 COCO521 already owns this output'; exit 1; }
  COMMIT="$(git -C "${PROJECT_ROOT}" rev-parse HEAD)"
  [[ -z "$(git -C "${PROJECT_ROOT}" status --porcelain)" ]] || { echo 'Refusing dirty source'; exit 1; }
  CODE_ROOT="${TASK_ROOT}/code/${COMMIT}"
  if [[ ! -d "${CODE_ROOT}" ]]; then
    mkdir -p "${TASK_ROOT}/code"
    STAGE="$(mktemp -d "${TASK_ROOT}/code/.stage.XXXXXX")"
    git -C "${PROJECT_ROOT}" archive HEAD | tar -xf - -C "${STAGE}"
    mv "${STAGE}" "${CODE_ROOT}"
  fi
  exec python3 -B -u "${CODE_ROOT}/scripts/in1k10_guard.py" \
    --root "${TASK_ROOT}" --code-sha "${COMMIT}" \
    --relay-url https://lnet-h200-baseline-relay-v1.gpupulse-monitor.workers.dev/k96coco \
    --stall-seconds 7200 -- /bin/bash "${CODE_ROOT}/h200/run_k96_coco_521.sh" --inside-guard
fi

cd "${PROJECT_ROOT}"
GPU_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
[[ "${GPU_MIB}" =~ ^[0-9]+$ && "${GPU_MIB}" -ge 81920 ]] || { echo 'Request 7 MIG units (one full H200 GPU)'; exit 1; }
export TMPDIR="${TASK_ROOT}/tmp" TMP="${TASK_ROOT}/tmp" TEMP="${TASK_ROOT}/tmp"
mkdir -p "${TMPDIR}"
chmod 700 "${TMPDIR}"
export UV_PYTHON_INSTALL_DIR="${TASK_ROOT}/uv-python" UV_CACHE_DIR="${TASK_ROOT}/uv-cache"
BOOTSTRAP="${TASK_ROOT}/uv-bootstrap"
if [[ ! -d "${BOOTSTRAP}/uv" ]]; then
  python3 -m pip install --target "${BOOTSTRAP}" --require-hashes -r h200/uv-bootstrap.requirements.txt
fi
uv_run() { PYTHONPATH="${BOOTSTRAP}" python3 -m uv "$@"; }
uv_run python install 3.13.11
ENV_ROOT="${TASK_ROOT}/environment"
if [[ ! -x "${ENV_ROOT}/bin/python" ]]; then uv_run venv --python 3.13.11 "${ENV_ROOT}"; fi
uv_run pip install --python "${ENV_ROOT}/bin/python" --index-strategy unsafe-best-match -r h200/dense_transfer/requirements.txt
export PYTHONPATH="${PROJECT_ROOT}/scripts:${PROJECT_ROOT}/src"
export OMP_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4
export TRITON_CACHE_DIR="${TASK_ROOT}/cache/triton" TORCHINDUCTOR_CACHE_DIR="${TASK_ROOT}/cache/inductor"
export LNET_DISABLE_LAUNCH_AUTOTUNE=1 LNET_FUSED_ODD_VERTICAL=1 LNET_BATCH_FINITE_CHECKS=1
export LNET_DENSE_MODE_BUDGET=1024 LNET_DENSE_ADAPTIVE_TILES=1 WANDB_MODE=disabled
CHECKPOINT="${TASK_ROOT}/checkpoints/va_k96_seed501_ep100.pt"
CHECKPOINT_URL=https://github.com/daehwa00/lnet-h200-imagenet1k-pg-ab/releases/download/dense-va-k96-seed501-ep100-v1/va_k96_seed501_ep100.pt
CHECKPOINT_SHA256=a72b70a422dc96d06c0cd88bf29a822793af40c65c67858d48f1467f6503e752
mkdir -p "${TASK_ROOT}/checkpoints"
if [[ ! -f "${CHECKPOINT}" ]]; then
  curl --fail --silent --show-error --location --retry 3 --connect-timeout 30 --continue-at - \
    --output "${CHECKPOINT}.part" "${CHECKPOINT_URL}"
  printf '%s  %s\n' "${CHECKPOINT_SHA256}" "${CHECKPOINT}.part" | sha256sum -c -
  mv "${CHECKPOINT}.part" "${CHECKPOINT}"
fi
printf '%s  %s\n' "${CHECKPOINT_SHA256}" "${CHECKPOINT}" | sha256sum -c -
"${ENV_ROOT}/bin/python" -B -u scripts/prepare_h200_dense_data.py \
  --only-coco --root "${TASK_ROOT}/dense-data"
exec "${ENV_ROOT}/bin/python" -B -u scripts/h200_k96_coco521.py \
  --root "${TASK_ROOT}" --checkpoint "${CHECKPOINT}" --checkpoint-sha256 "${CHECKPOINT_SHA256}" \
  --data-root "${TASK_ROOT}/dense-data/datasets/coco"
