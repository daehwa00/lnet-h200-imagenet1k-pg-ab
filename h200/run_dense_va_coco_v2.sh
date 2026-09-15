#!/usr/bin/env bash
# User submits this entrypoint. No real API key belongs in this script/form.
set -Eeuo pipefail
trap 'rc=$?; echo "H200_DENSE_SHELL_FAILURE phase=${PHASE:-bootstrap} line=${LINENO} exit=${rc}" >&2; exit "$rc"' ERR
TASK_ROOT=/app/output/daehwa00/dense-transfer
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${TASK_ROOT}"

if [[ "${H200_PERSISTENT_CODE:-0}" != "1" ]]; then
  exec 9>"${TASK_ROOT}/va-coco.lock"
  flock -n 9 || { echo 'Another VA-COCO launcher owns this output'; exit 1; }
  cd "${PROJECT_ROOT}"
  COMMIT="$(git rev-parse HEAD)"
  [[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
  [[ -z "${H200_EXPECTED_COMMIT:-}" || "${H200_EXPECTED_COMMIT}" == "${COMMIT}" ]]
  [[ -z "$(git status --porcelain --untracked-files=normal)" ]]
  mkdir -p "${TASK_ROOT}/code"
  CODE_ROOT="${TASK_ROOT}/code/${COMMIT}"
  if [[ ! -d "${CODE_ROOT}" ]]; then
    CODE_STAGE="$(mktemp -d "${TASK_ROOT}/code/.stage.XXXXXX")"
    git archive HEAD | tar -xf - -C "${CODE_STAGE}"
    mv "${CODE_STAGE}" "${CODE_ROOT}"
  fi
  [[ -f "${CODE_ROOT}/h200/run_dense_va_coco_v2.sh" ]]
  exec env H200_PERSISTENT_CODE=1 H200_RUNTIME_COMMIT="${COMMIT}" \
    bash "${CODE_ROOT}/h200/run_dense_va_coco_v2.sh"
fi
[[ "${H200_RUNTIME_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]]
[[ "${PROJECT_ROOT}" == "${TASK_ROOT}/code/${H200_RUNTIME_COMMIT}" ]]
cd "${PROJECT_ROOT}"
export H200_DENSE_TASK_ROOT="${TASK_ROOT}"
# Stable private tmp path, short enough for AF_UNIX listener paths. It is not
# /tmp or /app/scratch/input, which may be externally cleaned during long jobs.
TMP_TAG="$(printf '%s' "${HOSTNAME:-dense-job}" | sha256sum | cut -c1-8)"
export TMPDIR="${TASK_ROOT}/t/${TMP_TAG}"
mkdir -p "${TMPDIR}"
chmod 700 "${TMPDIR}"
export TMP="${TMPDIR}" TEMP="${TMPDIR}" PYTHONUNBUFFERED=1
echo "H200_DENSE_BOOT code=${PROJECT_ROOT} tmp=${TMPDIR}"
df -h "${TASK_ROOT}" /dev/shm

PHASE=dependencies
export UV_PYTHON_INSTALL_DIR="${TASK_ROOT}/uv-python"
export UV_CACHE_DIR="${TASK_ROOT}/cache/uv"
BOOTSTRAP="${TASK_ROOT}/uv-bootstrap-0.9.26"
if [[ ! -d "${BOOTSTRAP}/uv" ]]; then
  python3 -m pip install --target "${BOOTSTRAP}" --require-hashes -r h200/uv-bootstrap.requirements.txt
fi
uv_run() { PYTHONPATH="${BOOTSTRAP}" python3 -m uv "$@"; }
uv_run python install 3.13.11
ENV_ROOT="${TASK_ROOT}/environment-va-coco-v2"
if [[ ! -x "${ENV_ROOT}/bin/python" ]]; then uv_run venv --python 3.13.11 "${ENV_ROOT}"; fi
uv_run pip install --python "${ENV_ROOT}/bin/python" --index-strategy unsafe-best-match -r h200/dense_transfer/requirements.txt
uv_run pip check --python "${ENV_ROOT}/bin/python"
export PYTHONPATH="${PROJECT_ROOT}/scripts:${PROJECT_ROOT}/src"
export OMP_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4
export TRITON_CACHE_DIR="${TASK_ROOT}/cache/triton-va-coco-v2"
export LNET_DISABLE_LAUNCH_AUTOTUNE=1 LNET_FUSED_ODD_VERTICAL=1 LNET_BATCH_FINITE_CHECKS=1
export LNET_DENSE_MODE_BUDGET=1024 LNET_DENSE_ADAPTIVE_TILES=1 LNET_MP_SHARING_STRATEGY=file_system
export WANDB_DIR="${TASK_ROOT}/wandb-v2"
mkdir -p "${WANDB_DIR}"
PHASE=logging-canary
"${ENV_ROOT}/bin/python" -u scripts/run_h200_dense_v2.py --root "${TASK_ROOT}" --canary-only

PHASE=checkpoint
CHECKPOINT="${TASK_ROOT}/checkpoints/va_k128_seed501_ep100.pt"
CHECKPOINT_SHA=4852a5c6fa04996ae0c59e360c284fd5a378d2813f00ad5a9242212164bd7417
mkdir -p "${TASK_ROOT}/checkpoints"
if [[ ! -f "${CHECKPOINT}" ]]; then
  curl --fail --silent --show-error --location --retry 3 --connect-timeout 30 --continue-at - \
    --output "${CHECKPOINT}.part" \
    https://github.com/daehwa00/lnet-h200-imagenet1k-pg-ab/releases/download/dense-va-k128-seed501-ep100-v1/va_k128_seed501_ep100.pt
  printf '%s  %s\n' "${CHECKPOINT_SHA}" "${CHECKPOINT}.part" | sha256sum -c -
  ln "${CHECKPOINT}.part" "${CHECKPOINT}"
  unlink "${CHECKPOINT}.part"
fi
printf '%s  %s\n' "${CHECKPOINT_SHA}" "${CHECKPOINT}" | sha256sum -c -
PHASE=dataset
"${ENV_ROOT}/bin/python" -u scripts/prepare_h200_dense_data.py --only-coco
DATA_ROOT="$("${ENV_ROOT}/bin/python" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["ready"]; print(d["datasets"]["coco"]["path"])' "${TASK_ROOT}/datasets-ready.json")"
PHASE=training
exec "${ENV_ROOT}/bin/python" -u scripts/run_h200_dense_v2.py --root "${TASK_ROOT}" --data-root "${DATA_ROOT}"
