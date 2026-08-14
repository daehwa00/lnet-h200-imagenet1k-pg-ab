#!/usr/bin/env bash
set -euo pipefail

root=${1:-.}
cd "$root"

campaign=.omx/results/pac-revised-ucr-official-test-20260712
log_root=.omx/logs/pac-revised-ucr-official-test-pro6000-20260713
python_bin=${PAC_OFFICIAL_TEST_PYTHON:-python}
mkdir -p "$log_root"

for shard in $(seq 0 6); do
  shard_root="$campaign/shards/shard-$shard"
  PYTHONPATH="$root/src" "$python_bin" -m lnet.pac_recommended_low_data_cli \
    --stage workers --output-root "$shard_root" --preset full --device cuda \
    --workers 4 --total-slots 4 \
    >"$log_root/shard-$shard.log" 2>&1
done
