#!/usr/bin/env bash
set -Eeuo pipefail
TASK_ROOT=/app/output/Lee-Wonwoo1/cub200-scratch-100ep-v1
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "${TASK_ROOT}"
exec 8>"${TASK_ROOT}/launcher.lock"
flock -n 8 || { echo 'CUB launcher already running'; exit 1; }
exec > >(tee -a "${TASK_ROOT}/bootstrap.log") 2>&1
trap 'rc=$?; echo "CUB_BOOT_FAILURE line=${LINENO} exit=${rc}" >&2; exit "$rc"' ERR
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
python3 -c 'import shutil; free=shutil.disk_usage("/app/output/Lee-Wonwoo1/cub200-scratch-100ep-v1").free; assert free>=15*1024**3, "Need at least 15GiB free for environment and CUB data"'
# Credentials must be supplied privately by the runtime, never in a public issue.
if [[ -n "${WANDB_API_KEY_FILE:-}" ]]; then
  [[ -r "${WANDB_API_KEY_FILE}" ]] || { echo 'W&B secret file unreadable'; exit 1; }
  export WANDB_API_KEY="$(<"${WANDB_API_KEY_FILE}")"
fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  export WANDB_BASE_URL=https://lnet-h200-baseline-relay-v1.gpupulse-monitor.workers.dev/cub-v1
  # Public SDK placeholder; the relay authenticates H200 egress and exact scope.
  export WANDB_API_KEY=0000000000000000000000000000000000000000
  [[ "${CUB_TELEMETRY_ATTEMPT:-0}" == 0 ]] || { echo 'Relay only authorizes telemetry attempt 0'; exit 1; }
fi
COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
[[ -z "$(git -C "${REPO_ROOT}" status --porcelain --untracked-files=normal)" ]] || { echo 'Refusing dirty source'; exit 1; }
CODE="${TASK_ROOT}/code/${COMMIT}"
if [[ ! -d "${CODE}" ]]; then
  mkdir -p "${TASK_ROOT}/code"
  STAGE="$(mktemp -d "${TASK_ROOT}/code/.stage.XXXXXX")"
  git -C "${REPO_ROOT}" archive HEAD | tar -xf - -C "${STAGE}"
  mv "${STAGE}" "${CODE}"
fi
cd "${CODE}"
export TMPDIR="${TASK_ROOT}/tmp" TMP="${TASK_ROOT}/tmp" TEMP="${TASK_ROOT}/tmp"
export UV_PYTHON_INSTALL_DIR="${TASK_ROOT}/python" UV_CACHE_DIR="${TASK_ROOT}/uv-cache"
export TRITON_CACHE_DIR="${TASK_ROOT}/triton" TORCHINDUCTOR_CACHE_DIR="${TASK_ROOT}/inductor"
export WANDB_DIR="${TASK_ROOT}/wandb"
mkdir -p "${WANDB_DIR}"
mkdir -p "${TMPDIR}"
chmod 700 "${TMPDIR}"
BOOTSTRAP="${TASK_ROOT}/uv-bootstrap"
if [[ ! -d "${BOOTSTRAP}/uv" ]]; then
  timeout 600 python3 -m pip install --target "${BOOTSTRAP}" --require-hashes -r h200/uv-bootstrap.requirements.txt
fi
uv_run() { timeout 1800 env PYTHONPATH="${BOOTSTRAP}" python3 -m uv "$@"; }
uv_run python install 3.13.11
ENV_ROOT="${TASK_ROOT}/environment"
if [[ ! -x "${ENV_ROOT}/bin/python" ]]; then uv_run venv --python 3.13.11 "${ENV_ROOT}"; fi
uv_run pip install --python "${ENV_ROOT}/bin/python" --index-strategy unsafe-best-match -r h200/dense_transfer/requirements.txt
export PYTHONPATH="${CODE}/scripts:${CODE}/src"
export OMP_NUM_THREADS=4 LNET_DISABLE_LAUNCH_AUTOTUNE=1 LNET_YIELD_BEFORE_FETCH=1
# Fail before downloading data if the account/relay cannot open this project.
"${ENV_ROOT}/bin/python" -c 'import os,wandb; from cub200_run_ids import run_id; r=wandb.init(entity="daehwa",project="alphabet2d-cub200",group="cub200-scratch-100ep-lee-v1",id=run_id("cub200-scratch-100ep-lee-v1","connectivity",os.environ.get("CUB_TELEMETRY_ATTEMPT","0")),name="connectivity",resume="allow",mode="online",settings=wandb.Settings(init_timeout=45,console="off",disable_code=True,disable_git=True)); assert not r.offline; r.log({"connectivity_ok":1}); r.finish()'
"${ENV_ROOT}/bin/python" scripts/prepare_cub200.py --root "${TASK_ROOT}/datasets"
"${ENV_ROOT}/bin/python" scripts/prepare_cub200_sources.py --root "${TASK_ROOT}/sources"
uv_run pip install --python "${ENV_ROOT}/bin/python" --no-deps "${TASK_ROOT}/sources/mamba_ssm-2.3.2.post1-cp313-cp313-linux_x86_64.whl"
exec "${ENV_ROOT}/bin/python" -u scripts/cub200_supervisor.py \
  --root "${TASK_ROOT}" --data-root "${TASK_ROOT}/datasets/CUB_200_2011" \
  --source-root "${TASK_ROOT}/sources" --config h200/cub200/campaign.json "$@"
