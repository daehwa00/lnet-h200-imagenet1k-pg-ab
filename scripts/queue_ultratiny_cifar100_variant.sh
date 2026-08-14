#!/usr/bin/env bash
set -euo pipefail

wait_pid="${1:?current training PID is required}"
gpu="${2:?GPU index is required}"
variant="${3:?variant is required}"
runtime_root="/home/qlab/cifar100-ultratiny-20260801"

while kill -0 "${wait_pid}" 2>/dev/null; do
  sleep 10
done

cd "${runtime_root}"
exec env CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=src \
  /home/qlab/miniconda3/envs/brelu/bin/python -u \
  scripts/run_ultratiny_cifar100_baselines.py \
  --root "${runtime_root}/run" \
  --data-root /home/qlab/data \
  --variants "${variant}" \
  --run-seeds 401 \
  --epochs 100 \
  --batch-size 256 \
  --workers 4
