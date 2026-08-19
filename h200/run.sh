#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_BASE="${H200_OUTPUT_ROOT:-/app/output/daehwa00/lnet-h200-imagenet1k-pg-ab-v1}"
ENV_ROOT="${OUTPUT_BASE}/environment"
RUN_ROOT="${OUTPUT_BASE}/run"
CACHE_ROOT="${OUTPUT_BASE}/cache"

mkdir -p "${ENV_ROOT}" "${RUN_ROOT}" "${CACHE_ROOT}"
cd "${PROJECT_ROOT}"
export UV_PYTHON_INSTALL_DIR="${OUTPUT_BASE}/uv-python"
export UV_CACHE_DIR="${CACHE_ROOT}/uv"

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

echo "[h200] project=${PROJECT_ROOT}"
echo "[h200] data=${DATA_ROOT}"
echo "[h200] output=${OUTPUT_BASE}"
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader

UV_VERSION="0.9.26"
python3 -m pip install --user --disable-pip-version-check "uv==${UV_VERSION}"
python3 -m uv python install 3.13
if [[ ! -x "${ENV_ROOT}/bin/python" ]]; then
  python3 -m uv venv --python 3.13 "${ENV_ROOT}"
fi
python3 -m uv pip install \
  --python "${ENV_ROOT}/bin/python" \
  --index-strategy unsafe-best-match \
  --requirement "${PROJECT_ROOT}/h200/requirements.txt"

export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/scripts"
export CUDA_VISIBLE_DEVICES=0
export CUDA_MODULE_LOADING=LAZY
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/torchinductor"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export LNET_LAUNCH_CACHE="${CACHE_ROOT}/lnet-launch"
export LNET_COMPILE_MODE=default
export LNET_PERSISTENT_WORKERS=1
export WANDB_MODE=online
export WANDB_API_KEY="${WANDB_API_KEY:-0000000000000000000000000000000000000000}"
export WANDB_APP_URL="${WANDB_APP_URL:-https://wandb.ai}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://lnet-h200-wandb-relay.gpupulse-monitor.workers.dev}"
export WANDB_ENTITY="${WANDB_ENTITY:-daehwa}"
export WANDB_PROJECT="${WANDB_PROJECT:-alphabet2d-imagenet1k-h200}"
export WANDB_GROUP="${WANDB_GROUP:-h200-imagenet1k-k3-rmsmatch-pg-ab-v1}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"
export WANDB_DIR="${RUN_ROOT}/wandb"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

CPU_COUNT="$(nproc)"
WORKERS=$((CPU_COUNT / 2))
if (( WORKERS < 8 )); then WORKERS=8; fi
if (( WORKERS > 32 )); then WORKERS=32; fi
export LNET_DATALOADER_WORKERS="${WORKERS}"

"${ENV_ROOT}/bin/python" - <<'PY'
import json
import platform

import torch
import torchvision
import triton

if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError("exactly one visible CUDA GPU is required")
major, minor = torch.cuda.get_device_capability()
if major != 9:
    raise RuntimeError(f"expected Hopper compute capability 9.x, got {major}.{minor}")
payload = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "triton": triton.__version__,
    "cuda_runtime": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(),
    "compute_capability": [major, minor],
    "memory_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
}
print("H200_ENV=" + json.dumps(payload, sort_keys=True), flush=True)
PY

"${ENV_ROOT}/bin/python" - <<PY
from pathlib import Path
from torchvision.datasets import ImageFolder

root = Path("${DATA_ROOT}")
train = ImageFolder(root / "train")
validation = ImageFolder(root / "val")
if len(train.classes) != 1000 or train.classes != validation.classes:
    raise RuntimeError(
        f"ImageNet-1K contract failed: train={len(train.classes)}, "
        f"val={len(validation.classes)}, class_sets_equal={train.classes == validation.classes}"
    )
print(
    f"IMAGENET1K_DATASET=train:{len(train)},val:{len(validation)},classes:{len(train.classes)}",
    flush=True,
)
PY

for variant in \
  PGv2-H96-K3-RMSMatch-PGNoWD \
  PGv2-H96-K3-RMSMatch-NoPG-All; do
  "${ENV_ROOT}/bin/python" scripts/smoke_h200_imagenet1k_pg_ab.py \
    --variant "${variant}" \
    --root "${RUN_ROOT}/smoke" \
    --data-root "${DATA_ROOT}" \
    --batch-size 4 \
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
