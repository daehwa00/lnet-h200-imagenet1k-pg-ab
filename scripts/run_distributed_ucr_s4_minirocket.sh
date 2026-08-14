#!/usr/bin/env bash
set -euo pipefail

project_root=${1:?project root is required}
local_tuning_pid=${2:?local tuning pid is required}
shift 2
remote_tuning_pids=("$@")
if ((${#remote_tuning_pids[@]} == 0)); then
  echo "at least one remote tuning pid is required" >&2
  exit 2
fi
remote_root=REMOTE_HOME_PLACEHOLDER/lnet-pac-external-20260710
remote_python=REMOTE_ROOT_PLACEHOLDER/miniconda3/envs/ASA_test/bin/python
local_python=LOCAL_HOME_PLACEHOLDER/miniconda3/bin/python
main=.omx/results/pac-ucr-s4-minirocket-20260712
tuning_prefix=${TUNING_PREFIX:-.omx/results/pac-ucr-s4-minirocket-tuning-shard}
final_prefix=${FINAL_PREFIX:-.omx/results/pac-ucr-s4-minirocket-final-shard}
shard_count=${SHARD_COUNT:-5}
shard_weights=${SHARD_WEIGHTS:-"4 1 1 1 1"}
remote_shard_pairs=${REMOTE_SHARD_PAIRS:-"1:4 2:5 3:6 4:7"}
result_rel=results/ucr_s4_minirocket.csv

ssh_remote=(
  ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
  -i LOCAL_HOME_PLACEHOLDER/.codex/attachments/OPAQUE_ID_PLACEHOLDER/SSH_KEY_PLACEHOLDER
  -p 40701
  -o "ProxyCommand=ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i LOCAL_HOME_PLACEHOLDER/.codex/attachments/OPAQUE_ID_PLACEHOLDER/SSH_KEY_PLACEHOLDER -p 8589 secondary_host@REMOTE_HOST_PLACEHOLDER -W %h:%p"
  remote_user@REMOTE_HOST_PLACEHOLDER
)

remote() {
  "${ssh_remote[@]}" "$@"
}

wait_local() {
  local pid=$1
  while kill -0 "$pid" 2>/dev/null; do sleep 30; done
}

wait_remote_tuning() {
  local expression=""
  for pid in "${remote_tuning_pids[@]}"; do
    expression+="kill -0 $pid 2>/dev/null || "
  done
  expression=${expression% || }
  remote "while $expression; do sleep 30; done"
}

run_remote_shards_twice() {
  local prefix=$1
  remote "bash -s" <<REMOTE
set -e
cd "$remote_root"
PY="$remote_python"
declare -a pids
for pair in $remote_shard_pairs; do
  shard=\$(echo "\$pair" | cut -d: -f1)
  gpu=\$(echo "\$pair" | cut -d: -f2)
  (
    export CUDA_VISIBLE_DEVICES=\$gpu PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    ROOT=${prefix}\$shard-20260712
    "\$PY" -m lnet.pac_ucr_s4_minirocket_cli --stage workers --output-root "\$ROOT" --device cuda --workers 8 --total-slots 16
    "\$PY" -m lnet.pac_ucr_s4_minirocket_cli --stage workers --output-root "\$ROOT" --device cuda --workers 8 --total-slots 16
    touch "\$ROOT/COMPLETE"
  ) &
  pids[\$shard]=\$!
done
wait "\${pids[@]}"
REMOTE
}

transfer_local_result() {
  local local_root=$1
  local remote_target=$2
  tar -cf - -C "$project_root" "$local_root/$result_rel" |
    remote "tar -xf - -C '$remote_root' && mkdir -p '$remote_root/$remote_target'"
}

copy_remote_manifest() {
  local remote_source=$1
  local local_target=$2
  rm -rf "$project_root/$local_target"
  mkdir -p "$project_root/$local_target"
  remote "cat '$remote_root/$remote_source/queue_manifest.jsonl'" \
    >"$project_root/$local_target/queue_manifest.jsonl"
}

merge_remote() {
  local prefix=$1
  local shard_args=""
  local index
  for ((index = 0; index < shard_count; index++)); do
    shard_args+=" --shard-root '${prefix}${index}-20260712'"
  done
  remote "cd '$remote_root' && PYTHONPATH=src '$remote_python' -m lnet.pac_ucr_s4_minirocket_cli --stage merge --output-root '$main' $shard_args"
}

cd "$project_root"
wait_local "$local_tuning_pid"
wait_remote_tuning

# Retry only failed tuning keys once on every device.
env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "$local_python" -m lnet.pac_ucr_s4_minirocket_cli --stage workers \
  --output-root "${tuning_prefix}0-20260712" --device cuda --workers 8 --total-slots 16
run_remote_shards_twice "$tuning_prefix"
transfer_local_result "${tuning_prefix}0-20260712" "${tuning_prefix}0-20260712"
merge_remote "$tuning_prefix"

remote "cd '$remote_root' && PYTHONPATH=src '$remote_python' -m lnet.pac_ucr_s4_minirocket_cli --stage select-final --output-root '$main'"
weight_args=""
for weight in $shard_weights; do
  weight_args+=" --shard-weight '$weight'"
done
for ((index = 0; index < shard_count; index++)); do
  remote "cd '$remote_root' && rm -rf '${final_prefix}${index}-20260712' && PYTHONPATH=src '$remote_python' -m lnet.pac_ucr_s4_minirocket_cli --stage shard --source-root '$main' --output-root '${final_prefix}${index}-20260712' --shard-index '$index' --shard-count '$shard_count' $weight_args"
done
copy_remote_manifest "${final_prefix}0-20260712" "${final_prefix}0-20260712"

(
  export CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  "$local_python" -m lnet.pac_ucr_s4_minirocket_cli --stage workers \
    --output-root "${final_prefix}0-20260712" --device cuda --workers 8 --total-slots 16
  "$local_python" -m lnet.pac_ucr_s4_minirocket_cli --stage workers \
    --output-root "${final_prefix}0-20260712" --device cuda --workers 8 --total-slots 16
  touch "${final_prefix}0-20260712/COMPLETE"
) &
local_final=$!
run_remote_shards_twice "$final_prefix" &
remote_final=$!
wait "$local_final" "$remote_final"

transfer_local_result "${final_prefix}0-20260712" "${final_prefix}0-20260712"
merge_remote "$final_prefix"
remote "cd '$remote_root' && PYTHONPATH=src '$remote_python' - <<'PY'
import csv
from pathlib import Path
root=Path('$main')
rows=list(csv.DictReader((root/'$result_rel').open()))
latest={row['job_key']:row for row in rows}
done=sum(row.get('stage')=='test' and row.get('status')=='done' for row in latest.values())
if done != 180:
    raise SystemExit(f'final TEST incomplete: {done}/180')
(root/'COMPLETE').touch()
print('COMPLETE',done)
PY"
