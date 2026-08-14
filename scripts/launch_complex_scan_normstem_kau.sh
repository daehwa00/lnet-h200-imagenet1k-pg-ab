#!/usr/bin/env bash
set -euo pipefail

gpu="${1:-0}"
campaign=/home/daehwa/experiments/alphabet/complex-scan-normstem-cifar100-20260802
runtime="$campaign/runtime"
root="$campaign/screen50"
python_bin=/home/daehwa/anaconda3/envs/alphabet/bin/python
runner="$runtime/scripts/run_complex_scan_normstem_cifar100.py"
data_root=/home/daehwa/data
log="$campaign/logs/complex-linear-norm-no-gelu-stem-seed401.log"

mkdir -p "$(dirname "$log")"
while pgrep -f '[r]un_complex_scan_convstem_cifar100.py' >/dev/null; do
  sleep 10
done

(
  cd "$runtime"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src "$python_bin" "$runner" \
    --root "$root" \
    --data-root "$data_root" \
    --variants complex_linear_norm_no_gelu_stem \
    --run-seeds 401 \
    --epochs 50 \
    --batch-size 256 \
    --workers 4 \
    --compile-model \
    --skip-test
) >>"$log" 2>&1

date -Is >"$campaign/screen50.done"
