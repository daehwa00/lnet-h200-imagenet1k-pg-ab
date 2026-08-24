#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly PROJECT_ROOT
readonly CAMPAIGN_DIR="${H200_K_FAMILY_CAMPAIGN_DIR:-k_family_xl}"
readonly CAMPAIGN_RUNTIME_SCHEMA="${H200_K_FAMILY_RUNTIME_SCHEMA:-lnet.h200.imagenet100.k_family_xl.runtime.v1}"
readonly CAMPAIGN_PROGRAM_EXPECTED="${H200_K_FAMILY_PROGRAM:-h200/run_imagenet100_k_family_xl.sh}"
readonly CAMPAIGN_VARIANT_COUNT="${H200_K_FAMILY_VARIANT_COUNT:-4}"
readonly CAMPAIGN_GENERATOR="h200/${CAMPAIGN_DIR}/generate_contract.py"
readonly CAMPAIGN_MANIFEST="${PROJECT_ROOT}/h200/${CAMPAIGN_DIR}/campaign.json"
readonly CAMPAIGN_RUNTIME="${PROJECT_ROOT}/h200/${CAMPAIGN_DIR}/campaign.runtime.json"
readonly PROTOCOL_MANIFEST="${PROJECT_ROOT}/h200/campaign.json"
readonly PYTHON_VERSION="3.13.11"
readonly UV_VERSION="0.9.26"
readonly DUMMY_WANDB_API_KEY="0000000000000000000000000000000000000000"
readonly IMAGENET100_CANONICAL_MANIFEST_SHA256="6871da811224d961422ae8fe68339c81180e40d06983ce950189f5470add5db9"
readonly CONTROL_REPO_URL="https://github.com/daehwa00/lnet-h200-imagenet1k-pg-ab.git"
readonly CONTROL_REF="${H200_K_FAMILY_CONTROL_REF:-refs/heads/control/imagenet100-k-family-xl}"
readonly CONTROL_PATH="${H200_K_FAMILY_CONTROL_PATH:-h200/k_family_xl/control.json}"
readonly WANDB_RUNTIME_ENV_NAME="${H200_K_FAMILY_WANDB_RUNTIME_ENV:-H200_K_FAMILY_XL_WANDB_RUNTIME}"
readonly QUEUE_SCRIPT="${H200_K_FAMILY_QUEUE_SCRIPT:-scripts/run_h200_k_family_xl_queue.py}"

cd "${PROJECT_ROOT}"

if [[ ! "${H200_EXPECTED_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "ERROR: H200_EXPECTED_COMMIT must be the exact deployment commit" >&2
  exit 2
fi

if [[ "${H200_OWNER_CONTROL_INNER:-0}" != "1" ]]; then
  CONTROL_CAMPAIGN_ID="$(
    python3 - "${CAMPAIGN_RUNTIME}" <<'PY'
import json
import sys
from pathlib import Path

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")).get("campaign_id")
if not isinstance(value, str) or not value:
    raise SystemExit("invalid H200 XL control campaign identity")
print(value)
PY
  )"
  readonly CONTROL_CAMPAIGN_ID
  readonly CONTROL_STATE_ROOT="/app/output/daehwa00/run-control/${CONTROL_CAMPAIGN_ID}/${H200_EXPECTED_COMMIT}"
  readonly CONTROL_STOP_MARKER="${CONTROL_STATE_ROOT}/stopped.json"
  readonly CONTROL_FAST_STOP_MARKER="/dev/shm/lnet-owner-stop-${CONTROL_CAMPAIGN_ID}-${H200_EXPECTED_COMMIT}.json"
  export H200_RUN_CONTROL_REPO_URL="${CONTROL_REPO_URL}"
  export H200_RUN_CONTROL_REF="${CONTROL_REF}"
  export H200_RUN_CONTROL_PATH="${CONTROL_PATH}"
  export H200_RUN_CONTROL_POLL_SECONDS=15
  exec python3 scripts/run_h200_owner_controlled.py \
    --repo-root "${PROJECT_ROOT}" \
    --repo-url "${CONTROL_REPO_URL}" \
    --ref "${CONTROL_REF}" \
    --control-path "${CONTROL_PATH}" \
    --campaign-id "${CONTROL_CAMPAIGN_ID}" \
    --target-commit "${H200_EXPECTED_COMMIT}" \
    --stop-marker "${CONTROL_STOP_MARKER}" \
    --fast-stop-marker "${CONTROL_FAST_STOP_MARKER}" \
    --poll-seconds 15 \
    --grace-seconds 120 \
    --term-seconds 30 \
    -- env H200_OWNER_CONTROL_INNER=1 bash "$0"
fi

ACTUAL_COMMIT="$(git rev-parse --verify HEAD)"
readonly ACTUAL_COMMIT
if [[ "${ACTUAL_COMMIT}" != "${H200_EXPECTED_COMMIT}" ]]; then
  echo "ERROR: deployment commit mismatch: ${ACTUAL_COMMIT}" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
  echo "ERROR: deployment checkout is not clean" >&2
  exit 2
fi
python3 "${CAMPAIGN_GENERATOR}" --check

mapfile -t CAMPAIGN_VALUES < <(
  python3 - "${CAMPAIGN_RUNTIME}" "${CAMPAIGN_MANIFEST}" "${PROTOCOL_MANIFEST}" \
    "${CAMPAIGN_RUNTIME_SCHEMA}" "${CAMPAIGN_PROGRAM_EXPECTED}" \
    "${CAMPAIGN_VARIANT_COUNT}" <<'PY'
import hashlib
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit

runtime_path, manifest_path, protocol_path = map(Path, sys.argv[1:4])
expected_schema, expected_program, expected_variant_count = sys.argv[4:7]
runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
required = (
    "campaign_id", "output_namespace", "campaign_manifest_sha256",
    "wandb_sdk_version", "wandb_base_url", "wandb_app_url", "entity",
    "project", "group", "program", "console", "relay_protocol_version",
)
if runtime.get("schema") != expected_schema:
    raise SystemExit("invalid K-family runtime schema")
missing = [key for key in required if not isinstance(runtime.get(key), str) or not runtime[key]]
if missing:
    raise SystemExit(f"XL runtime is missing required strings: {missing}")
if not re.fullmatch(r"[0-9a-f]{64}", runtime["campaign_manifest_sha256"]):
    raise SystemExit("invalid XL manifest digest")
protocol_source = json.loads(protocol_path.read_text(encoding="utf-8"))
protocol = {
    "graphql_operations": protocol_source["graphql_operations"],
    "protocol": protocol_source["protocol"],
}
actual = hashlib.sha256(
    manifest_path.read_bytes()
    + b"\0"
    + json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
if actual != runtime["campaign_manifest_sha256"]:
    raise SystemExit("XL campaign digest mismatch")
training = runtime.get("training", {})
if (
    training.get("seed") != 501
    or training.get("epochs") != 100
    or training.get("batch_size") != 128
    or training.get("precision") != "bfloat16"
    or training.get("execution") != "one_model_to_epoch_100_then_next"
    or len(training.get("variants", ())) != int(expected_variant_count)
):
    raise SystemExit("K-family training matrix changed")
if runtime["program"] != expected_program:
    raise SystemExit("K-family program identity changed")
for key in ("wandb_base_url", "wandb_app_url"):
    parsed = urlsplit(runtime[key])
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
        raise SystemExit(f"XL campaign URL must be a credential-free HTTPS origin: {key}")
if runtime["console"] != "off":
    raise SystemExit("XL campaign must disable W&B console capture")
for key in required:
    print(runtime[key])
PY
)
if (( ${#CAMPAIGN_VALUES[@]} != 12 )); then
  echo "ERROR: failed to load frozen XL campaign runtime" >&2
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
readonly CAMPAIGN_PROGRAM="${CAMPAIGN_VALUES[9]}"
readonly CAMPAIGN_CONSOLE="${CAMPAIGN_VALUES[10]}"
readonly RELAY_PROTOCOL_VERSION="${CAMPAIGN_VALUES[11]}"

readonly OUTPUT_BASE="/app/output/daehwa00/${OUTPUT_NAMESPACE}-${MANIFEST_SHA256:0:16}"
readonly REQUIREMENTS_LOCK="${PROJECT_ROOT}/h200/requirements.lock"
REQUIREMENTS_SHA256="$(sha256sum "${REQUIREMENTS_LOCK}" | cut -d' ' -f1)"
readonly REQUIREMENTS_SHA256
readonly ENV_ROOT="${OUTPUT_BASE}/environment-py${PYTHON_VERSION}-${REQUIREMENTS_SHA256:0:16}"
readonly RUN_ROOT="${OUTPUT_BASE}/run"
readonly CACHE_ROOT="${OUTPUT_BASE}/cache"
readonly DATASET_MANIFEST="${RUN_ROOT}/dataset_manifest.json"

DATA_ROOT="${IMAGENET_ROOT:-}"
if [[ -z "${DATA_ROOT}" ]]; then
  for candidate in /app/data/ImageNet-2012 /app/data/imagenet /app/data/ImageNet2012; do
    if [[ -d "${candidate}/train" && -d "${candidate}/val" ]]; then
      DATA_ROOT="${candidate}"
      break
    fi
  done
fi
if [[ -z "${DATA_ROOT}" ]]; then
  echo "ERROR: ImageNet-1K train/val directories were not found" >&2
  exit 2
fi
readonly DATA_ROOT

mkdir -p "${OUTPUT_BASE}" "${RUN_ROOT}" "${CACHE_ROOT}"
export UV_PYTHON_INSTALL_DIR="${OUTPUT_BASE}/uv-python"
export UV_CACHE_DIR="${CACHE_ROOT}/uv"
readonly UV_BOOTSTRAP="${OUTPUT_BASE}/uv-bootstrap-${UV_VERSION}"
uv_bootstrap_version() {
  PYTHONPATH="${UV_BOOTSTRAP}" python3 -m uv --version 2>/dev/null || true
}
if [[ "$(uv_bootstrap_version)" != "uv ${UV_VERSION}" ]]; then
  UV_BOOTSTRAP_TEMP="$(mktemp -d "${OUTPUT_BASE}/.uv-bootstrap-${UV_VERSION}.XXXXXX")"
  python3 -m pip install \
    --disable-pip-version-check --no-deps --only-binary=:all: --require-hashes \
    --target "${UV_BOOTSTRAP_TEMP}" \
    --requirement "${PROJECT_ROOT}/h200/uv-bootstrap.requirements.txt"
  [[ ! -e "${UV_BOOTSTRAP}" ]] || mv "${UV_BOOTSTRAP}" "${UV_BOOTSTRAP}.incomplete-$$"
  mv "${UV_BOOTSTRAP_TEMP}" "${UV_BOOTSTRAP}"
fi
uv_command() { PYTHONPATH="${UV_BOOTSTRAP}" python3 -m uv "$@"; }
uv_command python install "${PYTHON_VERSION}"
[[ -x "${ENV_ROOT}/bin/python" ]] || uv_command venv --python "${PYTHON_VERSION}" "${ENV_ROOT}"
uv_command pip sync \
  --python "${ENV_ROOT}/bin/python" \
  --index-strategy unsafe-best-match --require-hashes --strict "${REQUIREMENTS_LOCK}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/scripts"
export CUDA_VISIBLE_DEVICES=0
export CUDA_MODULE_LOADING=LAZY
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/torchinductor"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export LNET_LAUNCH_CACHE="${CACHE_ROOT}/lnet-launch"
export LNET_COMPILE_MODE=default
export LNET_PERSISTENT_WORKERS=1
export LNET_DATALOADER_PREFETCH_FACTOR=2
export LNET_DATALOADER_WORKERS=8
CPU_COUNT="$(nproc)"
readonly CPU_COUNT
if (( CPU_COUNT < 8 )); then
  echo "ERROR: XL campaign requires at least 8 CPU workers, found ${CPU_COUNT}" >&2
  exit 2
fi
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
echo "H200_XL_INPUT_PIPELINE=cpus:${CPU_COUNT},workers:8,prefetch:2"
unset WANDB_CONFIG_PATHS WANDB_IDENTITY_TOKEN_FILE WANDB_JOB_TYPE WANDB_NAME \
  WANDB_RESUME WANDB_RUN_ID WANDB_TAGS WANDB_USER_EMAIL WANDB_USERNAME
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
export "${WANDB_RUNTIME_ENV_NAME}=${CAMPAIGN_RUNTIME}"
export H200_RUN_CONTROL_REPO_URL="${CONTROL_REPO_URL}"
export H200_RUN_CONTROL_REF="${CONTROL_REF}"
export H200_RUN_CONTROL_PATH="${CONTROL_PATH}"
export H200_RUN_CONTROL_POLL_SECONDS=15

"${ENV_ROOT}/bin/python" - "${WANDB_SDK_VERSION}" <<'PY'
import importlib.metadata
import json
import platform
import sys
import torch

expected = {
    "numpy": "2.4.6", "Pillow": "12.1.0", "scipy": "1.16.3",
    "ninja": "1.13.0", "torch": "2.9.1+cu128", "torchvision": "0.24.1+cu128",
    "triton": "3.5.1", "wandb": sys.argv[1],
}
actual = {name: importlib.metadata.version(name) for name in expected}
if platform.python_version() != "3.13.11" or actual != expected:
    raise RuntimeError(f"locked H200 XL environment mismatch: {actual}")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError("exactly one visible CUDA GPU is required")
major, minor = torch.cuda.get_device_capability()
name = torch.cuda.get_device_name()
if major != 9 or "H200" not in name.upper():
    raise RuntimeError(f"expected H200, got {name} {major}.{minor}")
print("H200_XL_ENV=" + json.dumps({"gpu": name, "packages": actual}, sort_keys=True))
PY

if [[ "${H200_XL_SMOKE_ONLY:-0}" == "1" ]]; then
  readonly SMOKE_ROOT="${OUTPUT_BASE}/representative-smoke-${ACTUAL_COMMIT:0:16}"
  "${ENV_ROOT}/bin/python" scripts/smoke_h200_k_family_xl.py \
    --root "${SMOKE_ROOT}" --batch-size 128 --repeat-steps 3
  echo "H200_XL_REPRESENTATIVE_SMOKE_COMPLETE=${SMOKE_ROOT}/smoke.json"
  exit 0
fi

"${ENV_ROOT}/bin/python" h200/validate_imagenet1k.py \
  --root "${DATA_ROOT}" --output "${DATASET_MANIFEST}" --reuse-existing \
  --managed-canonical-receipt h200/imagenet1k_canonical_receipt.json
export LNET_DATASET_MANIFEST_PATH="${DATASET_MANIFEST}"
LNET_DATASET_IDENTITY_SHA256="$(
  "${ENV_ROOT}/bin/python" - "${DATASET_MANIFEST}" <<'PY'
import json
import sys
from pathlib import Path
identity = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")).get("identity_sha256", "")
if not isinstance(identity, str) or len(identity) != 64:
    raise SystemExit("invalid persisted ImageNet-1K identity")
print(identity)
PY
)"
readonly LNET_DATASET_IDENTITY_SHA256
export LNET_DATASET_IDENTITY_SHA256

readonly IMAGENET100_ROOT="${OUTPUT_BASE}/imagenet100-first100-view"
"${ENV_ROOT}/bin/python" h200/stage_allocation/prepare_imagenet100.py \
  --source "${DATA_ROOT}" --output "${IMAGENET100_ROOT}"
LNET_IMAGENET100_EXPECTED_MANIFEST_SHA256="$(
  "${ENV_ROOT}/bin/python" - "${IMAGENET100_ROOT}" <<'PY'
import sys
from pathlib import Path
from imagenet100_data_runtime import dataset_digest
digest, train_images, validation_images = dataset_digest(Path(sys.argv[1]))
if (train_images, validation_images) != (130000, 5000):
    raise SystemExit("invalid ImageNet-100 view counts")
print(digest)
PY
)"
readonly LNET_IMAGENET100_EXPECTED_MANIFEST_SHA256
if [[ "${LNET_IMAGENET100_EXPECTED_MANIFEST_SHA256}" != "${IMAGENET100_CANONICAL_MANIFEST_SHA256}" ]]; then
  echo "ERROR: managed ImageNet-100 manifest changed" >&2
  exit 2
fi
export LNET_IMAGENET100_EXPECTED_MANIFEST_SHA256

readonly STAGED_DATA_ROOT="/app/scratch/datasets/${OUTPUT_NAMESPACE}-${LNET_IMAGENET100_EXPECTED_MANIFEST_SHA256:0:16}"
readonly STAGED_DATA_READY="${STAGED_DATA_ROOT}/.lnet-ready-${LNET_IMAGENET100_EXPECTED_MANIFEST_SHA256}"
mkdir -p /app/scratch/datasets
if [[ ! -f "${STAGED_DATA_READY}" ]]; then
  STAGED_DATA_TEMP="$(mktemp -d "/app/scratch/datasets/.${OUTPUT_NAMESPACE}.XXXXXX")"
  cp --archive --dereference --reflink=auto \
    "${IMAGENET100_ROOT}/train" "${IMAGENET100_ROOT}/val" "${STAGED_DATA_TEMP}/"
  "${ENV_ROOT}/bin/python" - "${STAGED_DATA_TEMP}" "${LNET_IMAGENET100_EXPECTED_MANIFEST_SHA256}" <<'PY'
import sys
from pathlib import Path
from imagenet100_data_runtime import dataset_digest
root, expected = Path(sys.argv[1]), sys.argv[2]
digest, train_images, validation_images = dataset_digest(root)
if digest != expected or (train_images, validation_images) != (130000, 5000):
    raise SystemExit("staged ImageNet-100 mismatch")
(root / f".lnet-ready-{expected}").write_text("ready\n", encoding="utf-8")
PY
  [[ ! -e "${STAGED_DATA_ROOT}" ]] || mv "${STAGED_DATA_ROOT}" "${STAGED_DATA_ROOT}.incomplete-$$"
  mv "${STAGED_DATA_TEMP}" "${STAGED_DATA_ROOT}"
fi

timeout --signal=TERM --kill-after=5m 48h \
  "${ENV_ROOT}/bin/python" "${QUEUE_SCRIPT}" \
  --root "${RUN_ROOT}" --data-root "${STAGED_DATA_ROOT}" --workers 8

"${ENV_ROOT}/bin/python" - "${RUN_ROOT}" "${CAMPAIGN_RUNTIME}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
runtime = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
variants = runtime["training"]["variants"]
epochs = runtime["training"]["epochs"]
seed = runtime["training"]["seed"]
contracts = set()
for variant in variants:
    result = json.loads((root / "results" / f"{variant}__seed{seed}.json").read_text())
    history = result.get("history")
    if (
        result.get("variant") != variant or result.get("seed") != seed
        or result.get("parameters") != runtime["parameter_counts"][variant]
        or not isinstance(history, list) or len(history) != epochs
        or history[-1].get("epoch") != epochs
    ):
        raise SystemExit(f"incomplete XL result: {variant}")
    contract = result.get("contract_sha256")
    if not isinstance(contract, str) or len(contract) != 64:
        raise SystemExit(f"invalid XL result contract: {variant}")
    contracts.add(contract)
if len(contracts) != 1:
    raise SystemExit("XL results do not share one immutable contract")
print(
    "H200_K_FAMILY_RESULTS_COMPLETE="
    + json.dumps({"runs": len(variants), "epochs": epochs})
)
PY
echo "H200_EXPERIMENT_COMPLETE=${RUN_ROOT}/summary.json"
