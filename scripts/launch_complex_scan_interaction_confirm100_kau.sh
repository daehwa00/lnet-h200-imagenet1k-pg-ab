#!/usr/bin/env bash
set -euo pipefail

if (( $# != 2 )); then
  echo "usage: $0 <gpu-index> <seed>" >&2
  exit 2
fi

gpu_index=$1
seed=$2
campaign=/home/daehwa/experiments/alphabet/complex-scan-interaction-cifar100-20260802
runtime="$campaign/runtime"
root="$campaign/confirm100"
python_bin=/home/daehwa/anaconda3/envs/alphabet/bin/python
log="$campaign/logs/complex-interaction-confirm100-seed${seed}.log"

mkdir -p "$(dirname "$log")"
while pgrep -f '[r]un_complex_scan_interaction_cifar100.py.*screen50' >/dev/null; do
  sleep 10
done
cd "$runtime"
exec env CUDA_VISIBLE_DEVICES="$gpu_index" PYTHONPATH=src "$python_bin" -u \
  scripts/run_complex_scan_interaction_cifar100.py \
  --root "$root" \
  --data-root /home/daehwa/data \
  --variants complex_interaction_rank8 \
  --run-seeds "$seed" \
  --epochs 100 \
  --batch-size 256 \
  --workers 4 \
  --compile-model \
  --skip-test \
  >>"$log" 2>&1
