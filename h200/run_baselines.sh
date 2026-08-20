#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly PROJECT_ROOT
readonly CAMPAIGN_MANIFEST="${PROJECT_ROOT}/h200/baselines/campaign.json"
readonly SOURCE_MANIFEST="${PROJECT_ROOT}/h200/baselines/sources.json"
readonly UNICONV_PATCH="${PROJECT_ROOT}/h200/baselines/patches/uniconvnet-dcnv3-torch29.patch"
readonly WANDB_RUNTIME="${PROJECT_ROOT}/h200/baselines/wandb.runtime.json"
readonly REQUIREMENTS_LOCK="${PROJECT_ROOT}/h200/baselines/requirements.lock"
readonly PYTHON_VERSION="3.13.11"
readonly UV_VERSION="0.9.26"
readonly DUMMY_WANDB_API_KEY="0000000000000000000000000000000000000000"

cd "${PROJECT_ROOT}"
if [[ ! "${H200_EXPECTED_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "ERROR: H200_EXPECTED_COMMIT must be the exact 40-character deployment commit" >&2
  exit 2
fi
if [[ "${H200_ALLOW_NOASSERTION_SOURCES:-}" != "research-only" ]]; then
  echo "ERROR: ParC-Net/EMOv2 require explicit research-only NOASSERTION opt-in" >&2
  exit 2
fi
ACTUAL_COMMIT="$(git rev-parse --verify HEAD)"
readonly ACTUAL_COMMIT
if [[ "${ACTUAL_COMMIT}" != "${H200_EXPECTED_COMMIT}" ]]; then
  echo "ERROR: deployment commit mismatch" >&2
  echo "expected=${H200_EXPECTED_COMMIT}" >&2
  echo "actual=${ACTUAL_COMMIT}" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
  echo "ERROR: deployment checkout is not clean" >&2
  exit 2
fi
for required in \
  "${CAMPAIGN_MANIFEST}" \
  "${SOURCE_MANIFEST}" \
  "${UNICONV_PATCH}" \
  "${WANDB_RUNTIME}" \
  "${REQUIREMENTS_LOCK}"; do
  if [[ ! -f "${required}" ]]; then
    echo "ERROR: required baseline contract is missing: ${required}" >&2
    exit 2
  fi
done
python3 h200/baselines/generate_wandb_contract.py --check

mapfile -t CAMPAIGN_VALUES < <(
  python3 - "${CAMPAIGN_MANIFEST}" "${SOURCE_MANIFEST}" "${WANDB_RUNTIME}" <<'PY'
import hashlib
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit

campaign_path, sources_path, runtime_path = map(Path, sys.argv[1:])
campaign_bytes = campaign_path.read_bytes()
campaign = json.loads(campaign_bytes)
sources = json.loads(sources_path.read_text(encoding="utf-8"))
runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
if campaign.get("schema") != "lnet.h200.imagenet1k.baselines.v1":
    raise SystemExit("invalid baseline campaign schema")
if sources.get("schema") != "lnet.h200.imagenet1k.external_sources.v1":
    raise SystemExit("invalid external-source schema")
if len(campaign.get("models", [])) != 20 or campaign.get("seeds") != [501, 509, 521]:
    raise SystemExit("baseline campaign must contain 20 models and three fixed seeds")
campaign_sha = hashlib.sha256(campaign_bytes).hexdigest()
if runtime.get("campaign_manifest_sha256") != campaign_sha:
    raise SystemExit("baseline W&B runtime is stale")
required = (
    "campaign_id", "wandb_sdk_version", "wandb_base_url", "wandb_app_url",
    "entity", "project", "group", "console", "relay_protocol_version",
)
for key in required:
    value = runtime.get(key)
    if not isinstance(value, str) or not value or "\n" in value or "\0" in value:
        raise SystemExit(f"invalid baseline runtime field: {key}")
for key in ("wandb_base_url", "wandb_app_url"):
    parsed = urlsplit(runtime[key])
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
        raise SystemExit(f"baseline runtime URL is not a safe HTTPS origin: {key}")
if runtime["console"] != "off" or runtime["wandb_sdk_version"] != "0.22.3":
    raise SystemExit("baseline W&B must pin SDK 0.22.3 with console off")
combined = hashlib.sha256(
    campaign_bytes + b"\0" + sources_path.read_bytes() + b"\0" + runtime_path.read_bytes()
).hexdigest()
if re.fullmatch(r"[0-9a-f]{64}", combined) is None:
    raise SystemExit("invalid combined campaign digest")
print(combined)
for key in required:
    print(runtime[key])
PY
)
if (( ${#CAMPAIGN_VALUES[@]} != 10 )); then
  echo "ERROR: failed to load the frozen baseline campaign" >&2
  exit 2
fi
readonly CAMPAIGN_SHA256="${CAMPAIGN_VALUES[0]}"
readonly CAMPAIGN_ID="${CAMPAIGN_VALUES[1]}"
readonly WANDB_SDK_VERSION="${CAMPAIGN_VALUES[2]}"
readonly CAMPAIGN_WANDB_BASE_URL="${CAMPAIGN_VALUES[3]}"
readonly CAMPAIGN_WANDB_APP_URL="${CAMPAIGN_VALUES[4]}"
readonly CAMPAIGN_ENTITY="${CAMPAIGN_VALUES[5]}"
readonly CAMPAIGN_PROJECT="${CAMPAIGN_VALUES[6]}"
readonly CAMPAIGN_GROUP="${CAMPAIGN_VALUES[7]}"
readonly CAMPAIGN_CONSOLE="${CAMPAIGN_VALUES[8]}"
readonly RELAY_PROTOCOL_VERSION="${CAMPAIGN_VALUES[9]}"

readonly OUTPUT_BASE="/app/output/daehwa00/lnet-h200-imagenet1k-baselines-v1-${CAMPAIGN_SHA256:0:12}-${ACTUAL_COMMIT:0:12}"
readonly CACHE_ROOT="${OUTPUT_BASE}/cache"
readonly SOURCE_ROOT="/app/scratch/input/lnet-h200-baseline-sources-${CAMPAIGN_SHA256:0:12}-${ACTUAL_COMMIT:0:12}"
readonly DATASET_MANIFEST="${OUTPUT_BASE}/dataset_manifest.json"
REQUIREMENTS_SHA256="$(sha256sum "${REQUIREMENTS_LOCK}" | cut -d' ' -f1)"
readonly REQUIREMENTS_SHA256
readonly ENV_ROOT="${OUTPUT_BASE}/environment-py${PYTHON_VERSION}-${REQUIREMENTS_SHA256:0:16}"

if [[ -n "${IMAGENET_ROOT:-}" ]]; then
  DATA_ROOT="${IMAGENET_ROOT}"
else
  DATA_ROOT=""
  for candidate in /app/data/ImageNet-2012 /app/data/imagenet /app/data/ImageNet2012; do
    if [[ -d "${candidate}/train" && -d "${candidate}/val" ]]; then
      DATA_ROOT="${candidate}"
      break
    fi
  done
fi
if [[ -z "${DATA_ROOT}" ]]; then
  echo "ERROR: ImageNet-1K train/val directories were not found under /app/data" >&2
  exit 2
fi
readonly DATA_ROOT

mkdir -p "${OUTPUT_BASE}" "${CACHE_ROOT}" "${SOURCE_ROOT}"
cleanup_external_sources() {
  case "${SOURCE_ROOT}" in
    /app/scratch/input/lnet-h200-baseline-sources-*)
      python3 - "${SOURCE_ROOT}" <<'PY'
import shutil, sys
from pathlib import Path
target = Path(sys.argv[1])
if target.is_dir():
    shutil.rmtree(target)
PY
      ;;
    *)
      echo "ERROR: refusing to clean unexpected external source path" >&2
      ;;
  esac
}
trap cleanup_external_sources EXIT
export UV_PYTHON_INSTALL_DIR="${OUTPUT_BASE}/uv-python"
export UV_CACHE_DIR="${CACHE_ROOT}/uv"
readonly UV_BOOTSTRAP="${OUTPUT_BASE}/uv-bootstrap-${UV_VERSION}"
uv_bootstrap_version() {
  PYTHONPATH="${UV_BOOTSTRAP}" python3 -m uv --version 2>/dev/null || true
}
if [[ "$(uv_bootstrap_version)" != "uv ${UV_VERSION}" ]]; then
  UV_BOOTSTRAP_TEMP="$(mktemp -d "${OUTPUT_BASE}/.uv-bootstrap-${UV_VERSION}.XXXXXX")"
  python3 -m pip install \
    --disable-pip-version-check \
    --no-deps \
    --only-binary=:all: \
    --require-hashes \
    --target "${UV_BOOTSTRAP_TEMP}" \
    --requirement "${PROJECT_ROOT}/h200/uv-bootstrap.requirements.txt"
  if [[ -e "${UV_BOOTSTRAP}" ]]; then
    mv "${UV_BOOTSTRAP}" "${UV_BOOTSTRAP}.incomplete-$$"
  fi
  mv "${UV_BOOTSTRAP_TEMP}" "${UV_BOOTSTRAP}"
fi
if [[ "$(uv_bootstrap_version)" != "uv ${UV_VERSION}" ]]; then
  echo "ERROR: uv bootstrap validation failed" >&2
  exit 2
fi
uv_command() {
  PYTHONPATH="${UV_BOOTSTRAP}" python3 -m uv "$@"
}
uv_command python install "${PYTHON_VERSION}"
if [[ ! -x "${ENV_ROOT}/bin/python" ]]; then
  uv_command venv --python "${PYTHON_VERSION}" "${ENV_ROOT}"
fi
uv_command pip sync \
  --python "${ENV_ROOT}/bin/python" \
  --index-strategy unsafe-best-match \
  --require-hashes \
  --strict \
  "${REQUIREMENTS_LOCK}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/scripts"
export CUDA_VISIBLE_DEVICES=0
export CUDA_MODULE_LOADING=LAZY
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/torchinductor"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export H200_BASELINE_SOURCE_ROOT="${SOURCE_ROOT}"
export H200_BASELINE_WANDB_RUNTIME="${WANDB_RUNTIME}"
export LNET_PERSISTENT_WORKERS=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
unset \
  WANDB_CONFIG_PATHS \
  WANDB_IDENTITY_TOKEN_FILE \
  WANDB_JOB_TYPE \
  WANDB_NAME \
  WANDB_RESUME \
  WANDB_RUN_ID \
  WANDB_TAGS \
  WANDB_USER_EMAIL \
  WANDB_USERNAME
export WANDB_MODE=online
export WANDB_API_KEY="${DUMMY_WANDB_API_KEY}"
export WANDB_APP_URL="${CAMPAIGN_WANDB_APP_URL}"
export WANDB_BASE_URL="${CAMPAIGN_WANDB_BASE_URL}"
export WANDB_ENTITY="${CAMPAIGN_ENTITY}"
export WANDB_PROJECT="${CAMPAIGN_PROJECT}"
export WANDB_GROUP="${CAMPAIGN_GROUP}"
export WANDB_CONSOLE="${CAMPAIGN_CONSOLE}"
export WANDB_INIT_TIMEOUT=30

echo "[baseline] commit=${ACTUAL_COMMIT}"
echo "[baseline] campaign=${CAMPAIGN_ID} digest=${CAMPAIGN_SHA256}"
echo "[baseline] relay=${CAMPAIGN_WANDB_BASE_URL} protocol=${RELAY_PROTOCOL_VERSION}"
echo "[baseline] data=${DATA_ROOT}"
echo "[baseline] output=${OUTPUT_BASE}"
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader
H200_GPU_DRIVER_VERSION="$(
  nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1 | tr -d '[:space:]'
)"
readonly H200_GPU_DRIVER_VERSION
export H200_GPU_DRIVER_VERSION

"${ENV_ROOT}/bin/python" - "${WANDB_SDK_VERSION}" <<'PY'
import importlib.metadata
import json
import platform
import sys
import torch

expected = {
    "einops": "0.8.1",
    "numpy": "2.4.6",
    "Pillow": "12.1.0",
    "PyYAML": "6.0.3",
    "timm": "1.0.26",
    "torch": "2.9.1+cu128",
    "torchvision": "0.24.1+cu128",
    "triton": "3.5.1",
    "wandb": sys.argv[1],
}
if platform.python_version() != "3.13.11":
    raise RuntimeError(f"expected Python 3.13.11, got {platform.python_version()}")
actual = {name: importlib.metadata.version(name) for name in expected}
if actual != expected:
    raise RuntimeError(f"locked package mismatch: expected={expected}, actual={actual}")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError("exactly one visible CUDA GPU is required")
major, minor = torch.cuda.get_device_capability()
name = torch.cuda.get_device_name()
if major != 9 or "H200" not in name.upper():
    raise RuntimeError(f"expected NVIDIA H200, got {name} capability {major}.{minor}")
print("H200_BASELINE_ENV=" + json.dumps({"packages": actual, "gpu": name}, sort_keys=True))
PY

"${ENV_ROOT}/bin/python" h200/validate_imagenet1k.py \
  --root "${DATA_ROOT}" \
  --output "${DATASET_MANIFEST}"
LNET_DATASET_IDENTITY_SHA256="$(
  "${ENV_ROOT}/bin/python" - "${DATASET_MANIFEST}" "${DATA_ROOT}" <<'PY'
import json
import sys
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if Path(manifest["dataset_root"]).resolve() != Path(sys.argv[2]).resolve():
    raise SystemExit("persisted dataset manifest belongs to another root")
if manifest["splits"]["train"]["count"] != 1281167 or manifest["splits"]["val"]["count"] != 50000:
    raise SystemExit("persisted ImageNet counts are invalid")
identity = manifest.get("identity_sha256")
if not isinstance(identity, str) or len(identity) != 64:
    raise SystemExit("persisted dataset identity is invalid")
print(identity)
PY
)"
readonly LNET_DATASET_IDENTITY_SHA256
export LNET_DATASET_IDENTITY_SHA256
export LNET_DATASET_MANIFEST_PATH="${DATASET_MANIFEST}"
readonly RUN_ROOT="${OUTPUT_BASE}/run-data${LNET_DATASET_IDENTITY_SHA256:0:12}"
mkdir -p "${RUN_ROOT}"
export WANDB_DIR="${RUN_ROOT}/wandb"

BASELINE_SOURCES_READY=0
for bootstrap_attempt in 1 2 3; do
  if "${ENV_ROOT}/bin/python" scripts/bootstrap_h200_baseline_sources.py \
    --source-root "${SOURCE_ROOT}"; then
    BASELINE_SOURCES_READY=1
    break
  fi
  echo "[baseline] external source bootstrap attempt ${bootstrap_attempt} failed" >&2
done
if (( BASELINE_SOURCES_READY == 0 )); then
  echo "H200_BASELINE_EXTERNAL_SOURCES_DEGRADED=bootstrap_failed" >&2
fi

# UniConvNet-A is the only native-extension lane. Build from a disposable MIT
# source copy so the pinned official checkout remains clean and verifiable.
if [[ -d "${SOURCE_ROOT}/uniconvnet/ops_dcnv3" ]]; then
UNICONV_COMMIT="$(
  "${ENV_ROOT}/bin/python" - "${SOURCE_MANIFEST}" <<'PY'
import json, sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text())["sources"]["uniconvnet"]["commit"])
PY
)"
readonly UNICONV_COMMIT
EXPECTED_PATCH_SHA="$(
  "${ENV_ROOT}/bin/python" - "${SOURCE_MANIFEST}" <<'PY'
import json, sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text())["sources"]["uniconvnet"]["compatibility_patch_sha256"])
PY
)"
readonly EXPECTED_PATCH_SHA
if [[ "$(sha256sum "${UNICONV_PATCH}" | cut -d' ' -f1)" != "${EXPECTED_PATCH_SHA}" ]]; then
  echo "ERROR: UniConvNet compatibility patch digest mismatch" >&2
  exit 2
fi
readonly DCNV3_WHEEL_ROOT="${OUTPUT_BASE}/external-wheels/dcnv3-${UNICONV_COMMIT:0:12}-patch${EXPECTED_PATCH_SHA:0:12}-torch2.9.1-cu128"
readonly DCNV3_WHEEL_SHA_FILE="${DCNV3_WHEEL_ROOT}/wheel.sha256"
mkdir -p "${DCNV3_WHEEL_ROOT}"
shopt -s nullglob
DCNV3_WHEELS=("${DCNV3_WHEEL_ROOT}"/*.whl)
shopt -u nullglob
if (( ${#DCNV3_WHEELS[@]} == 1 )); then
  if [[ ! -f "${DCNV3_WHEEL_SHA_FILE}" ]]; then
    mv "${DCNV3_WHEELS[0]}" "${DCNV3_WHEELS[0]}.unverified-$$"
    DCNV3_WHEELS=()
  else
    EXPECTED_WHEEL_SHA="$(tr -d '[:space:]' < "${DCNV3_WHEEL_SHA_FILE}")"
    ACTUAL_WHEEL_SHA="$(sha256sum "${DCNV3_WHEELS[0]}" | cut -d' ' -f1)"
    if [[ ! "${EXPECTED_WHEEL_SHA}" =~ ^[0-9a-f]{64}$ || "${ACTUAL_WHEEL_SHA}" != "${EXPECTED_WHEEL_SHA}" ]]; then
      mv "${DCNV3_WHEELS[0]}" "${DCNV3_WHEELS[0]}.corrupt-$$"
      DCNV3_WHEELS=()
    fi
  fi
elif (( ${#DCNV3_WHEELS[@]} > 1 )); then
  echo "H200_BASELINE_UNICONV_DISABLED=multiple_DCNv3_wheels" >&2
  wheel_index=0
  for ambiguous_wheel in "${DCNV3_WHEELS[@]}"; do
    mv "${ambiguous_wheel}" "${ambiguous_wheel}.ambiguous-$$-${wheel_index}"
    wheel_index=$((wheel_index + 1))
  done
  DCNV3_WHEELS=()
fi
if (( ${#DCNV3_WHEELS[@]} == 0 )); then
  DCNV3_BUILD_ROOT="$(mktemp -d "${SOURCE_ROOT}/.dcnv3-build.XXXXXX")"
  DCNV3_BUILD_JOBS="$(nproc)"
  if (( DCNV3_BUILD_JOBS > 8 )); then DCNV3_BUILD_JOBS=8; fi
  cp -a "${SOURCE_ROOT}/uniconvnet/ops_dcnv3/." "${DCNV3_BUILD_ROOT}/"
  (
    cd "${DCNV3_BUILD_ROOT}"
    git apply "${UNICONV_PATCH}"
  )
  if ! (
    cd "${DCNV3_BUILD_ROOT}"
    export TORCH_CUDA_ARCH_LIST=9.0
    export MAX_JOBS="${DCNV3_BUILD_JOBS}"
    "${ENV_ROOT}/bin/python" setup.py bdist_wheel --dist-dir "${DCNV3_WHEEL_ROOT}"
  ); then
    echo "H200_BASELINE_UNICONV_DISABLED=DCNv3_build_failed" >&2
  fi
  shopt -s nullglob
  DCNV3_WHEELS=("${DCNV3_WHEEL_ROOT}"/*.whl)
  shopt -u nullglob
  if (( ${#DCNV3_WHEELS[@]} == 1 )); then
    BUILT_WHEEL_SHA="$(sha256sum "${DCNV3_WHEELS[0]}" | cut -d' ' -f1)"
    printf '%s\n' "${BUILT_WHEEL_SHA}" > "${DCNV3_WHEEL_SHA_FILE}.tmp-$$"
    mv "${DCNV3_WHEEL_SHA_FILE}.tmp-$$" "${DCNV3_WHEEL_SHA_FILE}"
  fi
fi
if (( ${#DCNV3_WHEELS[@]} == 1 )); then
  EXPECTED_WHEEL_SHA="$(tr -d '[:space:]' < "${DCNV3_WHEEL_SHA_FILE}")"
  ACTUAL_WHEEL_SHA="$(sha256sum "${DCNV3_WHEELS[0]}" | cut -d' ' -f1)"
  if [[ "${EXPECTED_WHEEL_SHA}" != "${ACTUAL_WHEEL_SHA}" ]]; then
    echo "H200_BASELINE_UNICONV_DISABLED=DCNv3_wheel_hash_mismatch" >&2
    DCNV3_WHEELS=()
  fi
fi
if (( ${#DCNV3_WHEELS[@]} == 1 )); then
  uv_command pip install \
    --python "${ENV_ROOT}/bin/python" \
    --no-deps \
    --reinstall \
    "${DCNV3_WHEELS[0]}"
  export H200_DCNV3_WHEEL_SHA256="${ACTUAL_WHEEL_SHA}"
  export H200_DCNV3_PATCH_SHA256="${EXPECTED_PATCH_SHA}"
  echo "H200_DCNV3_WHEEL_SHA256=${H200_DCNV3_WHEEL_SHA256}"
else
  echo "H200_BASELINE_UNICONV_DISABLED=DCNv3_wheel_unavailable" >&2
fi
else
  echo "H200_BASELINE_UNICONV_DISABLED=source_checkout_unavailable" >&2
fi

QUEUE=(
  "${ENV_ROOT}/bin/python" scripts/run_h200_baseline_queue.py
  --manifest "${CAMPAIGN_MANIFEST}"
  --root "${RUN_ROOT}"
  --repo "${PROJECT_ROOT}"
  --worker "${PROJECT_ROOT}/scripts/run_h200_baseline_worker.py"
  --python "${ENV_ROOT}/bin/python"
  --data-root "${DATA_ROOT}"
  --mps auto
)

QUEUE_EXIT_CODE=0
"${QUEUE[@]}" --mode auto-run || QUEUE_EXIT_CODE=$?
"${ENV_ROOT}/bin/python" scripts/summarize_h200_baselines.py \
  --campaign "${CAMPAIGN_MANIFEST}" \
  --root "${RUN_ROOT}" \
  --output "${RUN_ROOT}/summary.json"
echo "H200_BASELINE_CAMPAIGN_COMPLETE=${RUN_ROOT}/queue-status.json"
if (( QUEUE_EXIT_CODE != 0 )); then
  echo "H200_BASELINE_CAMPAIGN_INCOMPLETE=${RUN_ROOT}/summary.json" >&2
  exit "${QUEUE_EXIT_CODE}"
fi
