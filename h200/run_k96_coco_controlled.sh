#!/usr/bin/env bash
set -Eeuo pipefail
TASK_ROOT=/app/output/daehwa00/dense-k96-coco-v1
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONUNBUFFERED=1
if [[ "${1:-}" != "--inside-guard" ]]; then
  : "${H200_AGENT_TOKEN:?Provide through a private environment, NEVER a public GitHub issue}"
  : "${H200_K96_CHECKPOINT:?Provide the fixed seed501 ImageNet checkpoint path}"
  : "${H200_K96_CHECKPOINT_SHA256:?Provide its verified SHA256}"
  mkdir -p "${TASK_ROOT}"
  exec 9>"${TASK_ROOT}/campaign.lock"
  flock -n 9 || { echo 'Campaign already owned'; exit 1; }
  COMMIT="$(git -C "${PROJECT_ROOT}" rev-parse HEAD)"
  [[ -z "$(git -C "${PROJECT_ROOT}" status --porcelain)" ]] || { echo 'Refusing dirty source'; exit 1; }
  CODE_ROOT="${TASK_ROOT}/code/${COMMIT}"
  if [[ ! -d "${CODE_ROOT}" ]]; then
    mkdir -p "${TASK_ROOT}/code"
    CODE_STAGE="$(mktemp -d "${TASK_ROOT}/code/.stage.XXXXXX")"
    git -C "${PROJECT_ROOT}" archive HEAD | tar -xf - -C "${CODE_STAGE}"
    mv "${CODE_STAGE}" "${CODE_ROOT}"
  fi
  exec python3 -u "${CODE_ROOT}/scripts/h200_http_guard.py" \
    --url https://lnet-h200-k96-control-v1.gpupulse-monitor.workers.dev \
    --root "${TASK_ROOT}" -- /bin/bash "${CODE_ROOT}/h200/run_k96_coco_controlled.sh" --inside-guard
fi
trap 'rc=$?; echo "K96_BOOT_FAILURE line=${LINENO} exit=${rc}" >&2; exit "$rc"' ERR
cd "${PROJECT_ROOT}"
export TMPDIR="${TASK_ROOT}/tmp" TMP="${TASK_ROOT}/tmp" TEMP="${TASK_ROOT}/tmp"
mkdir -p "${TMPDIR}"
chmod 700 "${TMPDIR}"
export UV_PYTHON_INSTALL_DIR="${TASK_ROOT}/uv-python" UV_CACHE_DIR="${TASK_ROOT}/uv-cache"
BOOTSTRAP="${TASK_ROOT}/uv-bootstrap"
echo 'K96_STAGE dependencies'
if [[ ! -d "${BOOTSTRAP}/uv" ]]; then
  python3 -m pip install --target "${BOOTSTRAP}" --require-hashes -r h200/uv-bootstrap.requirements.txt
fi
uv_run() { PYTHONPATH="${BOOTSTRAP}" python3 -m uv "$@"; }
uv_run python install 3.13.11
ENV_ROOT="${TASK_ROOT}/environment"
if [[ ! -x "${ENV_ROOT}/bin/python" ]]; then uv_run venv --python 3.13.11 "${ENV_ROOT}"; fi
uv_run pip install --python "${ENV_ROOT}/bin/python" --index-strategy unsafe-best-match -r h200/dense_transfer/requirements.txt
export PYTHONPATH="${PROJECT_ROOT}/scripts:${PROJECT_ROOT}/src"
export OMP_NUM_THREADS=4 LNET_DISABLE_LAUNCH_AUTOTUNE=1 LNET_FUSED_ODD_VERTICAL=1 LNET_BATCH_FINITE_CHECKS=1
export LNET_DENSE_MODE_BUDGET=1024 LNET_DENSE_ADAPTIVE_TILES=1
echo 'K96_STAGE dataset verification'
"${ENV_ROOT}/bin/python" -u scripts/prepare_h200_dense_data.py --only-coco
DATA_ROOT="$("${ENV_ROOT}/bin/python" -c 'import json; d=json.load(open("/app/output/daehwa00/dense-transfer/datasets-ready.json")); assert d["ready"]; print(d["datasets"]["coco"]["path"])')"
exec "${ENV_ROOT}/bin/python" -u scripts/run_h200_k96_coco_campaign.py \
  --checkpoint "${H200_K96_CHECKPOINT}" --checkpoint-sha256 "${H200_K96_CHECKPOINT_SHA256}" \
  --root "${TASK_ROOT}" --data-root "${DATA_ROOT}"
