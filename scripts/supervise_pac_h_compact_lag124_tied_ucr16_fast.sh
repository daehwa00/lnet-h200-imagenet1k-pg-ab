#!/usr/bin/env bash
set -euo pipefail

repo=${1:-.}
snapshot=${2:?frozen source snapshot is required}
root=.omx/results/pac-h-compact-lag124-tied-ucr16-fast-20260721
reference=.omx/results/pac-h-compact-lag124-ucr16-fast-20260721
driver=scripts/pac_h_compact_lag124_tied_ucr16_fast.py
log=.omx/logs/pac-h-compact-lag124-tied-ucr16-fast-20260721-supervisor.log
local_python=LOCAL_HOME_PLACEHOLDER/miniconda3/bin/python
local_gpu_python=LOCAL_HOME_PLACEHOLDER/miniconda3/envs/brelu/bin/python
kau_python=REMOTE_HOME_PLACEHOLDER/anaconda3/envs/torch/bin/python
local_gpu_key=LOCAL_HOME_PLACEHOLDER/.ssh/SSH_KEY_PLACEHOLDER
kau_key=LOCAL_HOME_PLACEHOLDER/.codex/attachments/OPAQUE_ID_PLACEHOLDER/SSH_KEY_PLACEHOLDER
local_gpu=local_gpu@REMOTE_HOST_PLACEHOLDER
kau=secondary_host@REMOTE_HOST_PLACEHOLDER
local_gpu_repo=LOCAL_HOME_PLACEHOLDER/lnet-external-20260718
kau_repo=REMOTE_HOME_PLACEHOLDER/lnet-terminal-20260718
snapshot_name=$(basename "$snapshot")

cd "$repo"
mkdir -p "$(dirname "$log")" "$root"/{completed,failed,logs,reports}

sync_remote() {
  local host=$1 port=$2 key=$3 remote_repo=$4 remote_snapshot
  remote_snapshot=$remote_repo/.omx/source-snapshots/$snapshot_name
  ssh -i "$key" -p "$port" -o BatchMode=yes "$host" \
    "chmod -R u+w '$remote_snapshot' 2>/dev/null || true; rm -rf '$remote_snapshot'; mkdir -p '$remote_snapshot' '$remote_repo/$root' '$remote_repo/$reference/reports'"
  rsync -az -e "ssh -i $key -p $port -o BatchMode=yes" \
    "$snapshot/" "$host:$remote_snapshot/"
  rsync -az -e "ssh -i $key -p $port -o BatchMode=yes" \
    "$root/" "$host:$remote_repo/$root/"
  rsync -az -e "ssh -i $key -p $port -o BatchMode=yes" \
    "$reference/contract.json" "$host:$remote_repo/$reference/contract.json"
  rsync -az -e "ssh -i $key -p $port -o BatchMode=yes" \
    "$reference/reports/summary.json" "$host:$remote_repo/$reference/reports/summary.json"
}

start_remote() {
  local host=$1 port=$2 key=$3 remote_repo=$4 python=$5 lane=$6 gpu=$7 label=$8
  local remote_snapshot manifest session lane_log
  remote_snapshot=$remote_repo/.omx/source-snapshots/$snapshot_name
  manifest=$root/manifests/worker-$lane.jsonl
  session=pac-h-lag124-tied-fast-$label
  lane_log=$root/logs/$label.log
  ssh -i "$key" -p "$port" -o BatchMode=yes "$host" \
    "tmux kill-session -t '$session' 2>/dev/null || true; tmux new-session -d -s '$session' \"cd '$remote_repo' && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONSAFEPATH=1 PYTHONPATH='$remote_snapshot/src:$remote_snapshot' CUDA_VISIBLE_DEVICES='$gpu' '$python' '$remote_snapshot/$driver' worker --root '$root' --manifest '$manifest' --device cuda >>'$lane_log' 2>&1\""
}

collect_remote() {
  local host=$1 port=$2 key=$3 remote_repo=$4 bucket
  for bucket in completed failed; do
    rsync -az -e "ssh -i $key -p $port -o BatchMode=yes" \
      "$host:$remote_repo/$root/$bucket/" "$root/$bucket/" 2>/dev/null || true
  done
}

if [[ ! -f $root/contract.json ]]; then
  PYTHONSAFEPATH=1 PYTHONPATH="$snapshot/src:$snapshot" "$local_python" "$snapshot/$driver" \
    enqueue --root "$root" --workers 3 | tee -a "$log"
fi

sync_remote "$local_gpu" 5003 "$local_gpu_key" "$local_gpu_repo"
sync_remote "$kau" 8589 "$kau_key" "$kau_repo"
start_remote "$local_gpu" 5003 "$local_gpu_key" "$local_gpu_repo" "$local_gpu_python" 00 0 local_gpu-gpu0
start_remote "$local_gpu" 5003 "$local_gpu_key" "$local_gpu_repo" "$local_gpu_python" 01 1 local_gpu-gpu1
start_remote "$kau" 8589 "$kau_key" "$kau_repo" "$kau_python" 02 0 kau-gpu0

while true; do
  collect_remote "$local_gpu" 5003 "$local_gpu_key" "$local_gpu_repo"
  collect_remote "$kau" 8589 "$kau_key" "$kau_repo"
  status=$(PYTHONSAFEPATH=1 PYTHONPATH="$snapshot/src:$snapshot" "$local_python" \
    "$snapshot/$driver" status --root "$root")
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$(tr '\n' ' ' <<<"$status")" | tee -a "$log"
  done_flag=$("$local_python" -c 'import json,sys; print(str(json.load(sys.stdin)["done"]).lower())' <<<"$status")
  [[ $done_flag == true ]] && break
  sleep 15
done

PYTHONSAFEPATH=1 PYTHONPATH="$snapshot/src:$snapshot" "$local_python" "$snapshot/$driver" \
  report --root "$root" >"$root/reports-final.json"
