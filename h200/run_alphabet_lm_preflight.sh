#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly PROJECT_ROOT
readonly PYTHON_VERSION="3.13.11"
readonly UV_VERSION="0.9.26"
readonly MAMBA_COMMIT="10b5d6358f27966f6a40e4bf0baa17a460688128"
readonly CAUSAL_CONV_COMMIT="cd81f0413cad2fc1e6f17e785ac39f59aae690cd"
cd "${PROJECT_ROOT}"

if [[ ! "${H200_EXPECTED_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "ERROR: H200_EXPECTED_COMMIT must be the immutable preflight commit" >&2
  exit 2
fi
if [[ "$(git rev-parse HEAD)" != "${H200_EXPECTED_COMMIT}" ]]; then
  echo "ERROR: ALPHABET-LM preflight commit mismatch" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
  echo "ERROR: ALPHABET-LM preflight checkout is not clean" >&2
  exit 2
fi

readonly OUTPUT_BASE="/app/output/daehwa00/alphabet-lm-preflight-v1-${H200_EXPECTED_COMMIT:0:16}"
readonly CACHE_ROOT="${OUTPUT_BASE}/cache"
readonly ENV_ROOT="${OUTPUT_BASE}/environment-py${PYTHON_VERSION}"
readonly SOURCE_ROOT="${OUTPUT_BASE}/third-party"
readonly RESULT_ROOT="${OUTPUT_BASE}/result"
readonly UV_BOOTSTRAP="${OUTPUT_BASE}/uv-bootstrap-${UV_VERSION}"
mkdir -p "${CACHE_ROOT}" "${SOURCE_ROOT}" "${RESULT_ROOT}"
export UV_PYTHON_INSTALL_DIR="${OUTPUT_BASE}/uv-python"
export UV_CACHE_DIR="${CACHE_ROOT}/uv"

uv_bootstrap_version() {
  PYTHONPATH="${UV_BOOTSTRAP}" python3 -m uv --version 2>/dev/null || true
}
if [[ "$(uv_bootstrap_version)" != "uv ${UV_VERSION}" ]]; then
  temporary="$(mktemp -d "${OUTPUT_BASE}/.uv-bootstrap.XXXXXX")"
  python3 -m pip install \
    --disable-pip-version-check --no-deps --only-binary=:all: --require-hashes \
    --target "${temporary}" --requirement h200/uv-bootstrap.requirements.txt
  [[ ! -e "${UV_BOOTSTRAP}" ]] || mv "${UV_BOOTSTRAP}" "${UV_BOOTSTRAP}.incomplete-$$"
  mv "${temporary}" "${UV_BOOTSTRAP}"
fi
uv_command() { PYTHONPATH="${UV_BOOTSTRAP}" python3 -m uv "$@"; }
uv_command python install "${PYTHON_VERSION}"
[[ -x "${ENV_ROOT}/bin/python" ]] || uv_command venv --python "${PYTHON_VERSION}" "${ENV_ROOT}"
uv_command pip sync \
  --python "${ENV_ROOT}/bin/python" --index-strategy unsafe-best-match \
  --require-hashes --strict h200/requirements.lock
uv_command pip install --python "${ENV_ROOT}/bin/python" --no-deps einops==0.8.1

checkout_source() {
  local name="$1" url="$2" commit="$3" target="${SOURCE_ROOT}/$1"
  if [[ ! -d "${target}/.git" ]]; then
    git clone --filter=blob:none "${url}" "${target}"
  fi
  git -C "${target}" fetch origin "${commit}"
  git -C "${target}" checkout --detach "${commit}"
  [[ "$(git -C "${target}" rev-parse HEAD)" == "${commit}" ]]
}
checkout_source causal-conv1d https://github.com/Dao-AILab/causal-conv1d.git "${CAUSAL_CONV_COMMIT}"
checkout_source mamba https://github.com/state-spaces/mamba.git "${MAMBA_COMMIT}"

export MAX_JOBS=8
export CAUSAL_CONV1D_FORCE_BUILD=TRUE
export MAMBA_FORCE_BUILD=TRUE
uv_command pip install \
  --python "${ENV_ROOT}/bin/python" --no-build-isolation --no-deps \
  "${SOURCE_ROOT}/causal-conv1d"
uv_command pip install \
  --python "${ENV_ROOT}/bin/python" --no-build-isolation --no-deps \
  "${SOURCE_ROOT}/mamba"

export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/scripts"
export CUDA_VISIBLE_DEVICES=0
export CUDA_MODULE_LOADING=LAZY
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/torchinductor"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export LNET_LAUNCH_CACHE="${CACHE_ROOT}/lnet-launch"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader
"${ENV_ROOT}/bin/python" - <<'PY'
import json
import platform
import torch
import causal_conv1d
import mamba_ssm

if platform.python_version() != "3.13.11":
    raise RuntimeError("unexpected Python in ALPHABET-LM preflight")
if not torch.cuda.is_available() or "H200" not in torch.cuda.get_device_name().upper():
    raise RuntimeError("ALPHABET-LM preflight requires H200")
print(
    "ALPHABET_LM_ENV="
    + json.dumps(
        {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "mamba": getattr(mamba_ssm, "__version__", "unknown"),
            "causal_conv1d": getattr(causal_conv1d, "__version__", "unknown"),
        },
        sort_keys=True,
    )
)
PY

timeout --signal=TERM --kill-after=5m 2h \
  "${ENV_ROOT}/bin/python" scripts/smoke_h200_alphabet_lm.py \
  --root "${RESULT_ROOT}" --microbatch 2 --context-length 2048 --repeats 2
echo "ALPHABET_LM_PREFLIGHT_COMPLETE=${RESULT_ROOT}/preflight.json"
