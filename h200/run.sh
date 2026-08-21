#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly PROJECT_ROOT
readonly CAMPAIGN_MANIFEST="${PROJECT_ROOT}/h200/campaign.json"
readonly CAMPAIGN_RUNTIME="${PROJECT_ROOT}/h200/campaign.runtime.json"
readonly PYTHON_VERSION="3.13.11"
readonly UV_VERSION="0.9.26"
readonly DUMMY_WANDB_API_KEY="0000000000000000000000000000000000000000"

cd "${PROJECT_ROOT}"

if [[ ! "${H200_EXPECTED_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "ERROR: H200_EXPECTED_COMMIT must be the exact 40-character deployment commit" >&2
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

if [[ ! -f "${CAMPAIGN_MANIFEST}" || ! -f "${CAMPAIGN_RUNTIME}" ]]; then
  echo "ERROR: generated H200 campaign manifests are missing" >&2
  exit 2
fi

mapfile -t CAMPAIGN_VALUES < <(
  python3 - "${CAMPAIGN_RUNTIME}" "${CAMPAIGN_MANIFEST}" <<'PY'
import hashlib
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit

runtime_path = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
required = (
    "campaign_id",
    "output_namespace",
    "manifest_sha256",
    "wandb_sdk_version",
    "wandb_base_url",
    "wandb_app_url",
    "entity",
    "project",
    "group",
    "console",
    "relay_url",
    "relay_protocol_version",
)
if runtime.get("schema_version") != 3:
    raise SystemExit("campaign runtime schema_version must be 3")
missing = [key for key in required if not isinstance(runtime.get(key), str) or not runtime[key]]
if missing:
    raise SystemExit(f"campaign runtime is missing required strings: {missing}")
for key in required:
    if "\n" in runtime[key] or "\0" in runtime[key]:
        raise SystemExit(f"campaign runtime field contains a control byte: {key}")
if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,127}", runtime["campaign_id"]):
    raise SystemExit("invalid campaign_id")
if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,127}", runtime["output_namespace"]):
    raise SystemExit("invalid output_namespace")
if not re.fullmatch(r"[0-9a-f]{64}", runtime["manifest_sha256"]):
    raise SystemExit("invalid campaign manifest digest")
actual_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
if actual_digest != runtime["manifest_sha256"]:
    raise SystemExit("campaign runtime digest does not match campaign.json")
if runtime["wandb_base_url"] != runtime["relay_url"]:
    raise SystemExit("W&B base URL and relay URL must be identical")
for key in ("wandb_base_url", "wandb_app_url", "relay_url"):
    parsed = urlsplit(runtime[key])
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
        raise SystemExit(f"campaign URL must be an HTTPS origin without credentials: {key}")
if runtime["console"] != "off":
    raise SystemExit("campaign must disable W&B console capture")
for key in required:
    print(runtime[key])
PY
)
if (( ${#CAMPAIGN_VALUES[@]} != 12 )); then
  echo "ERROR: failed to load the frozen H200 campaign runtime" >&2
  exit 2
fi
readonly CAMPAIGN_ID="${CAMPAIGN_VALUES[0]}"
readonly OUTPUT_NAMESPACE="${CAMPAIGN_VALUES[1]}"
readonly MANIFEST_SHA256="${CAMPAIGN_VALUES[2]}"
readonly WANDB_SDK_VERSION="${CAMPAIGN_VALUES[3]}"
readonly CAMPAIGN_WANDB_BASE_URL="${CAMPAIGN_VALUES[4]}"
readonly CAMPAIGN_WANDB_APP_URL="${CAMPAIGN_VALUES[5]}"
readonly CAMPAIGN_ENTITY="${CAMPAIGN_VALUES[6]}"
readonly CAMPAIGN_PROJECT="${CAMPAIGN_VALUES[7]}"
readonly CAMPAIGN_GROUP="${CAMPAIGN_VALUES[8]}"
readonly CAMPAIGN_CONSOLE="${CAMPAIGN_VALUES[9]}"
readonly CAMPAIGN_RELAY_URL="${CAMPAIGN_VALUES[10]}"
readonly RELAY_PROTOCOL_VERSION="${CAMPAIGN_VALUES[11]}"

readonly OUTPUT_BASE="/app/output/daehwa00/${OUTPUT_NAMESPACE}-${MANIFEST_SHA256:0:16}"
readonly REQUIREMENTS_LOCK="${PROJECT_ROOT}/h200/requirements.lock"
REQUIREMENTS_SHA256="$(sha256sum "${REQUIREMENTS_LOCK}" | cut -d' ' -f1)"
readonly REQUIREMENTS_SHA256
readonly ENV_ROOT="${OUTPUT_BASE}/environment-py${PYTHON_VERSION}-${REQUIREMENTS_SHA256:0:16}"
readonly RUN_ROOT="${OUTPUT_BASE}/run"
readonly CACHE_ROOT="${OUTPUT_BASE}/cache"
readonly DATASET_MANIFEST="${RUN_ROOT}/dataset_manifest.json"

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
  find /app/data -maxdepth 2 -type d 2>/dev/null | head -n 80 >&2 || true
  exit 2
fi
readonly DATA_ROOT

mkdir -p "${OUTPUT_BASE}" "${RUN_ROOT}" "${CACHE_ROOT}"
export UV_PYTHON_INSTALL_DIR="${OUTPUT_BASE}/uv-python"
export UV_CACHE_DIR="${CACHE_ROOT}/uv"

echo "[h200] commit=${ACTUAL_COMMIT}"
echo "[h200] campaign=${CAMPAIGN_ID} manifest=${MANIFEST_SHA256}"
echo "[h200] relay=${CAMPAIGN_RELAY_URL} protocol=${RELAY_PROTOCOL_VERSION}"
echo "[h200] data=${DATA_ROOT}"
echo "[h200] output=${OUTPUT_BASE}"
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader

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
export LNET_LAUNCH_CACHE="${CACHE_ROOT}/lnet-launch"
export LNET_COMPILE_MODE=default
export LNET_PERSISTENT_WORKERS=0
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
export WANDB_DIR="${RUN_ROOT}/wandb"
CPU_COUNT="$(nproc)"
WORKERS=$((CPU_COUNT / 4))
if (( CPU_COUNT >= 2 && WORKERS < 1 )); then WORKERS=1; fi
if (( WORKERS > 8 )); then WORKERS=8; fi
readonly WORKERS
export LNET_DATALOADER_WORKERS="${WORKERS}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

"${ENV_ROOT}/bin/python" - "${WANDB_SDK_VERSION}" <<'PY'
import importlib.metadata
import json
import platform
import sys

import torch

expected = {
    "numpy": "2.4.6",
    "Pillow": "12.1.0",
    "scipy": "1.16.3",
    "ninja": "1.13.0",
    "torch": "2.9.1+cu128",
    "torchvision": "0.24.1+cu128",
    "triton": "3.5.1",
    "wandb": sys.argv[1],
}
if platform.python_version() != "3.13.11":
    raise RuntimeError(f"expected Python 3.13.11, got {platform.python_version()}")
actual = {name: importlib.metadata.version(name) for name in expected}
mismatches = {
    name: {"expected": expected[name], "actual": actual[name]}
    for name in expected
    if actual[name] != expected[name]
}
if mismatches:
    raise RuntimeError(f"locked package version mismatch: {mismatches}")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError("exactly one visible CUDA GPU is required")
major, minor = torch.cuda.get_device_capability()
gpu_name = torch.cuda.get_device_name()
if major != 9 or "H200" not in gpu_name.upper():
    raise RuntimeError(
        f"expected one NVIDIA H200 (compute capability 9.x), got {gpu_name} {major}.{minor}"
    )
payload = {
    "python": platform.python_version(),
    "packages": actual,
    "cuda_runtime": torch.version.cuda,
    "gpu": gpu_name,
    "compute_capability": [major, minor],
    "memory_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
}
print("H200_ENV=" + json.dumps(payload, sort_keys=True), flush=True)
PY

"${ENV_ROOT}/bin/python" h200/validate_imagenet1k.py \
  --root "${DATA_ROOT}" \
  --output "${DATASET_MANIFEST}"
export LNET_DATASET_MANIFEST_PATH="${DATASET_MANIFEST}"
LNET_DATASET_IDENTITY_SHA256="$(
  "${ENV_ROOT}/bin/python" - "${DATASET_MANIFEST}" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
identity = manifest.get("identity_sha256", "")
if len(identity) != 64:
    raise SystemExit("invalid persisted dataset identity")
print(identity)
PY
)"
readonly LNET_DATASET_IDENTITY_SHA256
export LNET_DATASET_IDENTITY_SHA256

for variant in \
  PGv2-H96-K3-RMSMatch-PGNoWD \
  PGv2-H96-K3-RMSMatch-NoPG-All; do
  "${ENV_ROOT}/bin/python" scripts/smoke_h200_imagenet1k_pg_ab.py \
    --variant "${variant}" \
    --root "${RUN_ROOT}/smoke" \
    --data-root "${DATA_ROOT}" \
    --batch-size 256 \
    --workers "${WORKERS}" \
    --compile-mode default
done

"${ENV_ROOT}/bin/python" scripts/run_h200_imagenet1k_pg_ab.py \
  --root "${RUN_ROOT}" \
  --data-root "${DATA_ROOT}" \
  --variants \
    PGv2-H96-K3-RMSMatch-PGNoWD \
    PGv2-H96-K3-RMSMatch-NoPG-All \
  --run-seeds 501 \
  --epochs 100 \
  --batch-size 256 \
  --gradient-accumulation-steps 1 \
  --workers "${WORKERS}" \
  --precision bfloat16

echo "H200_EXPERIMENT_COMPLETE=${RUN_ROOT}/summary.json"
