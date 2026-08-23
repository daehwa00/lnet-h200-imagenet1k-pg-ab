#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly PROJECT_ROOT
readonly CAMPAIGN_MANIFEST="${PROJECT_ROOT}/h200/k64_p_depth_interaction/campaign.json"
readonly CAMPAIGN_RUNTIME="${PROJECT_ROOT}/h200/k64_p_depth_interaction/campaign.runtime.json"
readonly PROTOCOL_MANIFEST="${PROJECT_ROOT}/h200/campaign.json"
readonly PYTHON_VERSION="3.13.11"
readonly UV_VERSION="0.9.26"
readonly DUMMY_WANDB_API_KEY="0000000000000000000000000000000000000000"
readonly IMAGENET100_CANONICAL_MANIFEST_SHA256="6871da811224d961422ae8fe68339c81180e40d06983ce950189f5470add5db9"
readonly CONTROL_REPO_URL="https://github.com/daehwa00/lnet-h200-imagenet1k-pg-ab.git"
readonly CONTROL_REF="refs/heads/control/imagenet100-k64-p-depth-interaction-v3"
readonly CONTROL_PATH="h200/k64_p_depth_interaction/control.json"

cd "${PROJECT_ROOT}"

if [[ ! "${H200_EXPECTED_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "ERROR: H200_EXPECTED_COMMIT must be the exact 40-character deployment commit" >&2
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
    raise SystemExit("invalid H200 control campaign identity")
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
  echo "ERROR: generated H200 K64-P-depth-interaction manifests are missing" >&2
  exit 2
fi
python3 h200/k64_p_depth_interaction/generate_contract.py --check

mapfile -t CAMPAIGN_VALUES < <(
  python3 - "${CAMPAIGN_RUNTIME}" "${CAMPAIGN_MANIFEST}" "${PROTOCOL_MANIFEST}" <<'PY'
import hashlib
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit

runtime_path = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
protocol_path = Path(sys.argv[3])
runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
required = (
    "campaign_id",
    "output_namespace",
    "campaign_manifest_sha256",
    "wandb_sdk_version",
    "wandb_base_url",
    "wandb_app_url",
    "entity",
    "project",
    "group",
    "program",
    "console",
    "relay_protocol_version",
)
if runtime.get("schema") != "lnet.h200.imagenet100.k64_p_depth_interaction.runtime.v2":
    raise SystemExit("invalid K64-P-depth-interaction runtime schema")
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
if not re.fullmatch(r"[0-9a-f]{64}", runtime["campaign_manifest_sha256"]):
    raise SystemExit("invalid campaign manifest digest")
protocol_source = json.loads(protocol_path.read_text(encoding="utf-8"))
protocol = {
    "graphql_operations": protocol_source["graphql_operations"],
    "protocol": protocol_source["protocol"],
}
actual_digest = hashlib.sha256(
    manifest_path.read_bytes()
    + b"\0"
    + json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
if actual_digest != runtime["campaign_manifest_sha256"]:
    raise SystemExit("K64-P-depth-interaction campaign digest mismatch")
training = runtime.get("training", {})
if (
    training.get("seed") != 501
    or training.get("epochs") != 100
    or training.get("batch_size") != 128
    or training.get("execution") != "one_model_to_epoch_100_then_next"
    or len(training.get("variants", [])) != 2
):
    raise SystemExit("K64-P-depth-interaction training matrix changed")
if runtime["program"] != "h200/run_imagenet100_k64_p_depth_interaction.sh":
    raise SystemExit("K64-P-depth-interaction program identity changed")
for key in ("wandb_base_url", "wandb_app_url"):
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
echo "[h200] relay=${CAMPAIGN_WANDB_BASE_URL} protocol=${RELAY_PROTOCOL_VERSION}"
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
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/torchinductor"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export LNET_LAUNCH_CACHE="${CACHE_ROOT}/lnet-launch"
export LNET_COMPILE_MODE=default
export LNET_PERSISTENT_WORKERS=1
export LNET_DATALOADER_PREFETCH_FACTOR=2
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
export H200_K64_P_DEPTH_INTERACTION_WANDB_RUNTIME="${CAMPAIGN_RUNTIME}"
export H200_RUN_CONTROL_REPO_URL="${CONTROL_REPO_URL}"
export H200_RUN_CONTROL_REF="${CONTROL_REF}"
export H200_RUN_CONTROL_PATH="${CONTROL_PATH}"
export H200_RUN_CONTROL_POLL_SECONDS=15
CPU_COUNT="$(nproc)"
if (( CPU_COUNT < 8 )); then
  echo "ERROR: canonical K64 campaign requires at least 8 CPU workers" >&2
  exit 2
fi
WORKERS=8
readonly WORKERS
export LNET_DATALOADER_WORKERS="${WORKERS}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
echo "H200_INPUT_PIPELINE=cpus:${CPU_COUNT},workers:${WORKERS},prefetch:${LNET_DATALOADER_PREFETCH_FACTOR},compile:${LNET_COMPILE_MODE}"

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

# Model and relay smoke were completed before submission; use allocated H200 time for training.
"${ENV_ROOT}/bin/python" h200/validate_imagenet1k.py \
  --root "${DATA_ROOT}" \
  --output "${DATASET_MANIFEST}" \
  --reuse-existing \
  --managed-canonical-receipt h200/imagenet1k_canonical_receipt.json
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

readonly IMAGENET100_ROOT="${OUTPUT_BASE}/imagenet100-first100-view"
"${ENV_ROOT}/bin/python" h200/stage_allocation/prepare_imagenet100.py \
  --source "${DATA_ROOT}" \
  --output "${IMAGENET100_ROOT}"
LNET_IMAGENET100_EXPECTED_MANIFEST_SHA256="$(
  "${ENV_ROOT}/bin/python" - "${IMAGENET100_ROOT}" <<'PY'
import sys
from pathlib import Path

from imagenet100_data_runtime import dataset_digest

digest, train_images, validation_images = dataset_digest(Path(sys.argv[1]))
if (train_images, validation_images) != (130000, 5000):
    raise SystemExit(
        f"invalid ImageNet-100 view counts: train={train_images}, val={validation_images}"
    )
print(digest)
PY
)"
readonly LNET_IMAGENET100_EXPECTED_MANIFEST_SHA256
if [[ "${LNET_IMAGENET100_EXPECTED_MANIFEST_SHA256}" != "${IMAGENET100_CANONICAL_MANIFEST_SHA256}" ]]; then
  echo "ERROR: managed ImageNet-100 manifest differs from the pinned campaign dataset" >&2
  exit 2
fi
export LNET_IMAGENET100_EXPECTED_MANIFEST_SHA256
echo "H200_IMAGENET100_MANIFEST=${LNET_IMAGENET100_EXPECTED_MANIFEST_SHA256}"

readonly STAGED_DATA_PARENT="/app/scratch/datasets"
readonly STAGED_DATA_ROOT="${STAGED_DATA_PARENT}/${OUTPUT_NAMESPACE}-${LNET_IMAGENET100_EXPECTED_MANIFEST_SHA256:0:16}"
readonly STAGED_DATA_READY="${STAGED_DATA_ROOT}/.lnet-ready-${LNET_IMAGENET100_EXPECTED_MANIFEST_SHA256}"
mkdir -p "${STAGED_DATA_PARENT}"
df -h /app/scratch /dev/shm
free -h || true
if [[ ! -f "${STAGED_DATA_READY}" ]]; then
  STAGED_DATA_REQUIRED_BYTES="$(
    du --bytes --summarize --dereference \
      "${IMAGENET100_ROOT}/train" \
      "${IMAGENET100_ROOT}/val" \
      | awk '{ total += $1 } END { printf "%.0f\n", total }'
  )"
  STAGED_DATA_AVAILABLE_BYTES="$(df --block-size=1 --output=avail /app/scratch | tail -n 1 | tr -d ' ')"
  if (( STAGED_DATA_AVAILABLE_BYTES < STAGED_DATA_REQUIRED_BYTES + 2147483648 )); then
    echo "ERROR: /app/scratch lacks ImageNet-100 staging space" >&2
    echo "required=${STAGED_DATA_REQUIRED_BYTES} available=${STAGED_DATA_AVAILABLE_BYTES}" >&2
    exit 2
  fi
  STAGED_DATA_TEMP="$(mktemp -d "${STAGED_DATA_PARENT}/.${OUTPUT_NAMESPACE}.XXXXXX")"
  stage_started="$(date +%s)"
  cp --archive --dereference --reflink=auto \
    "${IMAGENET100_ROOT}/train" \
    "${IMAGENET100_ROOT}/val" \
    "${STAGED_DATA_TEMP}/"
  "${ENV_ROOT}/bin/python" - \
    "${STAGED_DATA_TEMP}" \
    "${LNET_IMAGENET100_EXPECTED_MANIFEST_SHA256}" <<'PY'
import sys
from pathlib import Path

from imagenet100_data_runtime import dataset_digest

root = Path(sys.argv[1])
expected = sys.argv[2]
digest, train_images, validation_images = dataset_digest(root)
if digest != expected or (train_images, validation_images) != (130000, 5000):
    raise SystemExit(
        f"staged ImageNet-100 mismatch: digest={digest}, "
        f"train={train_images}, val={validation_images}"
    )
(root / f".lnet-ready-{expected}").write_text("ready\n", encoding="utf-8")
PY
  if [[ -e "${STAGED_DATA_ROOT}" ]]; then
    mv "${STAGED_DATA_ROOT}" "${STAGED_DATA_ROOT}.incomplete-$$"
  fi
  mv "${STAGED_DATA_TEMP}" "${STAGED_DATA_ROOT}"
  echo "H200_DATA_STAGED_SECONDS=$(( $(date +%s) - stage_started ))"
fi
readonly TRAIN_DATA_ROOT="${STAGED_DATA_ROOT}"
echo "H200_TRAIN_DATA_ROOT=${TRAIN_DATA_ROOT}"

mapfile -t K64_P_DEPTH_INTERACTION_VARIANTS < <(
  "${ENV_ROOT}/bin/python" - "${CAMPAIGN_RUNTIME}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for variant in payload["training"]["variants"]:
    print(variant)
PY
)
if (( ${#K64_P_DEPTH_INTERACTION_VARIANTS[@]} != 2 )); then
  echo "ERROR: expected exactly 2 K64 P-depth-interaction variants" >&2
  exit 2
fi

timeout --signal=TERM --kill-after=5m 48h \
  "${ENV_ROOT}/bin/python" scripts/run_h200_k64_p_depth_interaction_queue.py \
  --root "${RUN_ROOT}" \
  --data-root "${TRAIN_DATA_ROOT}" \
  --workers "${WORKERS}"

"${ENV_ROOT}/bin/python" - "${RUN_ROOT}" "${CAMPAIGN_RUNTIME}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
runtime = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
variants = runtime["training"]["variants"]
epochs = runtime["training"]["epochs"]
seed = runtime["training"]["seed"]
summary_path = root / "summary.json"
try:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
except (FileNotFoundError, json.JSONDecodeError) as error:
    raise SystemExit("all variants completed but summary.json is invalid") from error
if summary.get("completed_runs") != len(variants) or set(summary.get("variants", {})) != set(
    variants
):
    raise SystemExit("summary does not cover both registered K64 P-depth-interaction runs")
contract_sha256 = None
for variant in variants:
    path = root / "results" / f"{variant}__seed{seed}.json"
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise SystemExit(f"missing or invalid result: {variant}") from error
    history = result.get("history")
    if (
        result.get("variant") != variant
        or result.get("seed") != seed
        or not isinstance(history, list)
        or len(history) != epochs
        or history[-1].get("epoch") != epochs
    ):
        raise SystemExit(f"incomplete epoch-100 result: {variant}")
    active_contract = result.get("contract_sha256")
    if not isinstance(active_contract, str) or len(active_contract) != 64:
        raise SystemExit(f"invalid result contract identity: {variant}")
    if contract_sha256 is None:
        contract_sha256 = active_contract
    elif active_contract != contract_sha256:
        raise SystemExit("completed results do not share one immutable contract")
print(
    "H200_K64_P_DEPTH_INTERACTION_RESULTS_COMPLETE="
    + json.dumps(
        {"contract_sha256": contract_sha256, "epochs": epochs, "runs": len(variants)},
        sort_keys=True,
    ),
    flush=True,
)
PY
echo "H200_EXPERIMENT_COMPLETE=${RUN_ROOT}/summary.json"
