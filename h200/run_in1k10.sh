#!/usr/bin/env bash
set -Eeuo pipefail
TASK_ROOT=/app/output/in1k10-local-v2
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
if [[ "${1:-}" != "--inside-guard" ]]; then
  mkdir -p "${TASK_ROOT}"
  exec 9>"${TASK_ROOT}/campaign.lock"
  flock -n 9 || { echo 'IN10 campaign already owns this output'; exit 1; }
  COMMIT="$(git -C "${PROJECT_ROOT}" rev-parse HEAD)"
  [[ -z "$(git -C "${PROJECT_ROOT}" status --porcelain)" ]] || { echo 'Refusing dirty source'; exit 1; }
  CODE_ROOT="${TASK_ROOT}/code/${COMMIT}"
  if [[ ! -d "${CODE_ROOT}" ]]; then
    mkdir -p "${TASK_ROOT}/code"
    STAGE="$(mktemp -d "${TASK_ROOT}/code/.stage.XXXXXX")"
    git -C "${PROJECT_ROOT}" archive HEAD | tar -xf - -C "${STAGE}"
    mv "${STAGE}" "${CODE_ROOT}"
  fi
  exec python3 -B -u "${CODE_ROOT}/scripts/in1k10_guard.py" --root "${TASK_ROOT}" --code-sha "${COMMIT}" \
    -- /bin/bash "${CODE_ROOT}/h200/run_in1k10.sh" --inside-guard
fi
trap 'rc=$?; echo "IN10_BOOT_FAILURE line=${LINENO} exit=${rc}" >&2; exit "$rc"' ERR
cd "${PROJECT_ROOT}"
export TMPDIR="${TASK_ROOT}/tmp" TMP="${TASK_ROOT}/tmp" TEMP="${TASK_ROOT}/tmp"
mkdir -p "${TMPDIR}"; chmod 700 "${TMPDIR}"
export UV_PYTHON_INSTALL_DIR="${TASK_ROOT}/uv-python" UV_CACHE_DIR="${TASK_ROOT}/uv-cache"
BOOTSTRAP="${TASK_ROOT}/uv-bootstrap"
echo 'IN10_BOOT dependencies'
if [[ ! -d "${BOOTSTRAP}/uv" ]]; then
  python3 -m pip install --target "${BOOTSTRAP}" --require-hashes -r h200/uv-bootstrap.requirements.txt
fi
uv_run() { PYTHONPATH="${BOOTSTRAP}" python3 -m uv "$@"; }
uv_run python install 3.13.11
ENV_ROOT="${TASK_ROOT}/environment"
if [[ ! -x "${ENV_ROOT}/bin/python" ]]; then uv_run venv --python 3.13.11 "${ENV_ROOT}"; fi
uv_run pip install --python "${ENV_ROOT}/bin/python" --index-strategy unsafe-best-match -r h200/dense_transfer/requirements.txt
echo 'IN10_BOOT pinned native extension'
WHEEL="${TASK_ROOT}/mamba_ssm-2.3.2.post1-cp313-cp313-linux_x86_64.whl"
if [[ ! -f "${WHEEL}" ]]; then
  curl --fail --silent --show-error --location --retry 3 --connect-timeout 30 --continue-at - \
    --output "${WHEEL}.part" https://github.com/daehwa00/lnet-h200-imagenet1k-pg-ab/releases/download/h200-selective-scan-torch291-cu128-v1/mamba_ssm-2.3.2.post1-cp313-cp313-linux_x86_64.whl
  printf '%s  %s\n' 7201849146fb3b517e1a89741c4042596652dea24f44a94e4a83e6246353f49e "${WHEEL}.part" | sha256sum -c -
  mv "${WHEEL}.part" "${WHEEL}"
fi
printf '%s  %s\n' 7201849146fb3b517e1a89741c4042596652dea24f44a94e4a83e6246353f49e "${WHEEL}" | sha256sum -c -
uv_run pip install --python "${ENV_ROOT}/bin/python" --no-deps "${WHEEL}"
export PYTHONPATH="${PROJECT_ROOT}/scripts:${PROJECT_ROOT}/src"
export OMP_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4
export TRITON_CACHE_DIR="${TASK_ROOT}/cache/triton" TORCHINDUCTOR_CACHE_DIR="${TASK_ROOT}/cache/inductor"
export WANDB_MODE=disabled
echo 'IN10_BOOT subset and sources'
"${ENV_ROOT}/bin/python" -B -u scripts/in1k10_subset.py --root "${TASK_ROOT}/dataset" --data-root /app/data/ImageNet-2012
"${ENV_ROOT}/bin/python" -B -u scripts/in1k10_prepare_sources.py --root "${TASK_ROOT}/sources"
"${ENV_ROOT}/bin/python" -c 'import torch; assert torch.cuda.is_available(); p=torch.cuda.get_device_properties(0); print({"gpu":p.name,"memory":p.total_memory,"torch":torch.__version__}); assert p.total_memory >= 80*1024**3, "Request full GPU (7), not one MIG slice"'
echo 'IN10_BOOT campaign'
exec "${ENV_ROOT}/bin/python" -B -u scripts/in1k10_campaign.py --root "${TASK_ROOT}" \
  --data-root /app/data/ImageNet-2012 --sources "${TASK_ROOT}/sources"
