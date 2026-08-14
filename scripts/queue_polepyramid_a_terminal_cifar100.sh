#!/usr/bin/env bash
set -euo pipefail

wait_pid="${1:?current training PID is required}"
runtime_root="/home/qai/polepyramid-a-cifar100-20260801"
while kill -0 "${wait_pid}" 2>/dev/null; do
  sleep 30
done

cd "${runtime_root}"
exec env PYTHONPATH=src /home/qai/miniconda3/envs/alphabet/bin/python -u \
  scripts/run_polepyramid_a_terminal_cifar100.py \
  --root "${runtime_root}/run-cifar100-terminal-seed401" \
  --data-root /home/qai/datasets \
  --epochs 100 \
  --batch-size 256 \
  --workers 4
