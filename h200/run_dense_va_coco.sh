#!/usr/bin/env bash
# Reuse verified COCO, or prepare it in this same container before training.
set -Eeuo pipefail
trap 'rc=$?; echo "ERROR: line ${LINENO}, exit ${rc}: ${BASH_COMMAND}" >&2; exit "$rc"' ERR
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_ROOT=/app/output/daehwa00/dense-transfer
CHECKPOINT="${TASK_ROOT}/checkpoints/va_k128_seed501_ep100.pt"
CHECKPOINT_SHA=4852a5c6fa04996ae0c59e360c284fd5a378d2813f00ad5a9242212164bd7417
mkdir -p "${TASK_ROOT}"
exec 9>"${TASK_ROOT}/va-coco.lock"
flock -n 9 || { echo 'Another VA-COCO launcher owns the lock'; exit 1; }
if [[ ! -f "${CHECKPOINT}" ]]; then
  mkdir -p "${TASK_ROOT}/checkpoints"
  curl --fail --location --retry 3 --connect-timeout 30 --continue-at - \
    --output "${CHECKPOINT}.part" \
    https://github.com/daehwa00/lnet-h200-imagenet1k-pg-ab/releases/download/dense-va-k128-seed501-ep100-v1/va_k128_seed501_ep100.pt
  printf '%s  %s\n' "${CHECKPOINT_SHA}" "${CHECKPOINT}.part" | sha256sum -c -
  ln "${CHECKPOINT}.part" "${CHECKPOINT}"
  unlink "${CHECKPOINT}.part"
fi
printf '%s  %s\n' "${CHECKPOINT_SHA}" "${CHECKPOINT}" | sha256sum -c -
cd "${PROJECT_ROOT}"
echo "Storage diagnostics: host=$(hostname), task_root=${TASK_ROOT}"
df -h "${TASK_ROOT}"
if [[ ! -d "${TASK_ROOT}/datasets/coco/train2017" ]]; then
  echo 'COCO train2017 is absent in this container. Prior output may be on another volume/node; checking reusable data before download.'
fi
python3 -u scripts/prepare_h200_dense_data.py --only-coco
DATA_ROOT="$(python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["ready"]; print(r["datasets"]["coco"]["path"])' "${TASK_ROOT}/datasets-ready.json")"
echo "Verified COCO root: ${DATA_ROOT}"
export UV_PYTHON_INSTALL_DIR="${TASK_ROOT}/uv-python"
export UV_CACHE_DIR="${TASK_ROOT}/cache/uv"
BOOTSTRAP="${TASK_ROOT}/uv-bootstrap-0.9.26"
if [[ ! -d "${BOOTSTRAP}/uv" ]]; then
  python3 -m pip install --target "${BOOTSTRAP}" --require-hashes -r h200/uv-bootstrap.requirements.txt
fi
uv_run() { PYTHONPATH="${BOOTSTRAP}" python3 -m uv "$@"; }
uv_run python install 3.13.11
ENV_ROOT="${TASK_ROOT}/environment-va-coco-v1"
if [[ ! -x "${ENV_ROOT}/bin/python" ]]; then
  uv_run venv --python 3.13.11 "${ENV_ROOT}"
fi
uv_run pip sync --python "${ENV_ROOT}/bin/python" --index-strategy unsafe-best-match h200/dense_transfer/requirements.txt
export PYTHONPATH="${PROJECT_ROOT}/scripts:${PROJECT_ROOT}/src"
export LNET_DISABLE_LAUNCH_AUTOTUNE=1
export TRITON_CACHE_DIR="${TASK_ROOT}/cache/triton-va-coco-v1"
export OMP_NUM_THREADS=4
export WANDB_MODE=disabled
OUT="${TASK_ROOT}/runs/coco-va_k128-seed501"
mkdir -p "${OUT}"
# Keep physical batch 2 initially: identical padding/accumulation to the 4090
# recipe. Increasing it is a separate measured optimization, not assumed exact.
ARGS=(--task coco --model va_k128 --checkpoint "${CHECKPOINT}"
      --data-root "${DATA_ROOT}" --output-root "${OUT}"
      --physical-batch-size 2 --effective-batch-size 16 --workers 4)
"${ENV_ROOT}/bin/python" -u scripts/validate_dense_transfer_runtime.py "${ARGS[@]}" --mode smoke --max-probe-updates 2 2>&1 | tee "${OUT}/validation.log"
"${ENV_ROOT}/bin/python" -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["status"]=="ready", r' "${OUT}/queue/readiness.json"
RESUME=()
if [[ -f "${OUT}/checkpoints/last.pt" ]]; then
  RESUME=(--resume "${OUT}/checkpoints/last.pt")
fi
echo "Validation passed. Training COCO 12 epochs; results: ${OUT}"
exec "${ENV_ROOT}/bin/python" -u scripts/run_dense_transfer.py "${ARGS[@]}" --mode train --confirm-training DENSE_TRANSFER_TRAIN "${RESUME[@]}"
