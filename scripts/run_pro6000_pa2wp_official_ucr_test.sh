#!/usr/bin/env bash
set -euo pipefail

root=${1:-.}
cd "$root"

campaign=.omx/results/pac-pa2wp-official-ucr-test-pro6000-20260713
log_root=.omx/logs/pac-pa2wp-official-ucr-test-pro6000-20260713
python_bin=${PAC_OFFICIAL_TEST_PYTHON:-python}
mkdir -p "$log_root"

PYTHONPATH="$root/src" "$python_bin" -m lnet.pac_pa2wp_official_test \
  --stage enqueue --output-root "$campaign" >"$log_root/enqueue.log" 2>&1
PYTHONPATH="$root/src" "$python_bin" -m lnet.pac_recommended_low_data_cli \
  --stage workers --output-root "$campaign" --preset full --device cuda \
  --workers 4 --total-slots 4 >"$log_root/worker.log" 2>&1
