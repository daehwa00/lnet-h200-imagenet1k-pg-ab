#!/usr/bin/env bash
set -euo pipefail

experiment_root=${EXPERIMENT_ROOT:?EXPERIMENT_ROOT is required}
runtime_root=${RUNTIME_ROOT:-"$experiment_root/runtime"}
run_root=${RUN_ROOT:-"$experiment_root/run"}
data_root=${DATA_ROOT:-/home/qlab/data/ImageNet100}
python_bin=${PYTHON_BIN:-/home/qlab/.conda/envs/lnet-paper-cu128/bin/python}
epochs=${PROBE_EPOCHS:-100}
read -r -a probe_variants <<< "${PROBE_VARIANTS:-RN50L3-GAP RN50L3-Energy96 RN50L3-RealPole96}"

if (( ${#probe_variants[@]} == 0 )); then
  echo "PROBE_VARIANTS must select at least one probe" >&2
  exit 2
fi

mkdir -p "$experiment_root/logs" "$run_root" "$experiment_root/cache/inductor"
cd "$runtime_root"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH="$runtime_root/src:$runtime_root/scripts"
export LNET_COMPILE_MODE=reduce-overhead
export LNET_CPU_AFFINITY=${LNET_CPU_AFFINITY:-8-15,24-31}
export LNET_DATALOADER_WORKERS=${LNET_DATALOADER_WORKERS:-8}
export LNET_PERSISTENT_WORKERS=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
export TORCHINDUCTOR_CACHE_DIR="$experiment_root/cache/inductor"
export WANDB_PROJECT=${WANDB_PROJECT:-alphabet2d-imagenet100}
export WANDB_ENTITY=${WANDB_ENTITY:-daehwa}
export WANDB_GROUP=${WANDB_GROUP:-RN50L3-RealPole-Probe}

"$python_bin" -m pytest -q tests/test_pretrained_resnet50_real_pole_probe.py

"$python_bin" scripts/smoke_pretrained_resnet50_real_pole_probe.py \
  --device cuda \
  --pretrained \
  --batch-size 2 \
  --variants "${probe_variants[@]}" \
  2>&1 | tee "$experiment_root/logs/cuda-smoke.log"

exec "$python_bin" -u scripts/run_pretrained_resnet50_real_pole_probe_imagenet100.py \
  --root "$run_root" \
  --data-root "$data_root" \
  --variants "${probe_variants[@]}" \
  --run-seeds 501 \
  --epochs "$epochs" \
  --batch-size 128 \
  --gradient-accumulation-steps 1 \
  --workers 8 \
  --precision bfloat16
