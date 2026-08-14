#!/usr/bin/env bash
set -euo pipefail

repo=${1:-.}
snapshot=${2:?frozen source snapshot is required}
root=${PAC_FAST_ROOT:-.omx/results/pac-logsig-v51-ucr16-fast-20260721}
driver=${PAC_FAST_DRIVER:-scripts/pac_logsig_v51_ucr16_fast.py}
label=${PAC_FAST_LABEL:-V5/V5.1}
session_prefix=${PAC_FAST_SESSION_PREFIX:-pac-logsig-v51-fast}
reference_root=${PAC_FAST_REFERENCE_ROOT:-}
python=LOCAL_HOME_PLACEHOLDER/miniconda3/bin/python
local_gpu_python=LOCAL_HOME_PLACEHOLDER/miniconda3/envs/brelu/bin/python
kau_python=REMOTE_HOME_PLACEHOLDER/anaconda3/envs/torch/bin/python
local_gpu_key=LOCAL_HOME_PLACEHOLDER/.ssh/SSH_KEY_PLACEHOLDER
kau_key=LOCAL_HOME_PLACEHOLDER/.codex/attachments/OPAQUE_ID_PLACEHOLDER/SSH_KEY_PLACEHOLDER
local_gpu=local_gpu@REMOTE_HOST_PLACEHOLDER
kau=secondary_host@REMOTE_HOST_PLACEHOLDER
local_gpu_repo=LOCAL_HOME_PLACEHOLDER/lnet-external-20260718
kau_repo=REMOTE_HOME_PLACEHOLDER/lnet-terminal-20260718
snapshot_name=$(basename "$snapshot")
log=${PAC_FAST_LOG:-.omx/logs/pac-logsig-v51-ucr16-fast-20260721-supervisor.log}

cd "$repo"
mkdir -p "$(dirname "$log")" "$root" "$root/completed" "$root/failed" "$root/logs"

log_line() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$log"
}

sync_remote() {
  local host=$1 port=$2 key=$3 remote_repo=$4 remote_snapshot
  remote_snapshot=$remote_repo/.omx/source-snapshots/$snapshot_name
  ssh -i "$key" -p "$port" -o BatchMode=yes "$host" \
    "chmod -R u+w '$remote_snapshot' 2>/dev/null || true; rm -rf '$remote_snapshot'; mkdir -p '$remote_snapshot' '$remote_repo/$root'"
  rsync -az -e "ssh -i $key -p $port -o BatchMode=yes" \
    "$snapshot/" "$host:$remote_snapshot/"
  rsync -az -e "ssh -i $key -p $port -o BatchMode=yes" \
    "$root/" "$host:$remote_repo/$root/"
  if [[ -n $reference_root ]]; then
    ssh -i "$key" -p "$port" -o BatchMode=yes "$host" \
      "mkdir -p '$remote_repo/$reference_root/reports'"
    rsync -az -e "ssh -i $key -p $port -o BatchMode=yes" \
      "$reference_root/contract.json" "$host:$remote_repo/$reference_root/contract.json"
    rsync -az -e "ssh -i $key -p $port -o BatchMode=yes" \
      "$reference_root/reports/summary.json" \
      "$host:$remote_repo/$reference_root/reports/summary.json"
  fi
  ssh -i "$key" -p "$port" -o BatchMode=yes "$host" "chmod -R a-w '$remote_snapshot'"
}

start_local() {
  local lane=$1 manifest session lane_log
  manifest=$root/manifests/worker-$lane.jsonl
  session=$session_prefix-pro-$lane
  lane_log=$root/logs/pro-$lane.log
  tmux new-session -d -s "$session" \
    "cd '$repo' && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONSAFEPATH=1 PYTHONPATH='$snapshot/src:$snapshot' CUDA_VISIBLE_DEVICES=0 '$python' '$snapshot/$driver' worker --root '$root' --manifest '$manifest' --device cuda >>'$lane_log' 2>&1"
}

start_remote() {
  local host=$1 port=$2 key=$3 remote_repo=$4 remote_python=$5 lane=$6 gpu=$7 label=$8
  local remote_snapshot manifest session lane_log
  remote_snapshot=$remote_repo/.omx/source-snapshots/$snapshot_name
  manifest=$root/manifests/worker-$lane.jsonl
  session=$session_prefix-$label
  lane_log=$root/logs/$label.log
  ssh -i "$key" -p "$port" -o BatchMode=yes "$host" \
    "tmux new-session -d -s '$session' \"cd '$remote_repo' && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONSAFEPATH=1 PYTHONPATH='$remote_snapshot/src:$remote_snapshot' CUDA_VISIBLE_DEVICES='$gpu' '$remote_python' '$remote_snapshot/$driver' worker --root '$root' --manifest '$manifest' --device cuda >>'$lane_log' 2>&1\""
}

collect_remote() {
  local host=$1 port=$2 key=$3 remote_repo=$4 bucket
  for bucket in completed failed; do
    rsync -az -e "ssh -i $key -p $port -o BatchMode=yes" \
      "$host:$remote_repo/$root/$bucket/" "$root/$bucket/" 2>/dev/null || true
  done
}

if [[ ! -f $root/contract.json ]]; then
  PYTHONSAFEPATH=1 PYTHONPATH="$snapshot/src:$snapshot" "$python" "$snapshot/$driver" \
    enqueue --root "$root" --workers 14 | tee -a "$log"
fi

sync_remote "$local_gpu" 5003 "$local_gpu_key" "$local_gpu_repo"
sync_remote "$kau" 8589 "$kau_key" "$kau_repo"

for lane in $(seq -w 0 10); do
  start_local "$lane"
done
start_remote "$local_gpu" 5003 "$local_gpu_key" "$local_gpu_repo" "$local_gpu_python" 11 0 local_gpu-gpu0
start_remote "$local_gpu" 5003 "$local_gpu_key" "$local_gpu_repo" "$local_gpu_python" 12 1 local_gpu-gpu1
start_remote "$kau" 8589 "$kau_key" "$kau_repo" "$kau_python" 13 0 kau-gpu0
job_count=$(PYTHONSAFEPATH=1 PYTHONPATH="$snapshot/src:$snapshot" "$python" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["jobs"])' "$root/contract.json")
log_line "started $job_count-job $label Fast UCR-16 screen on 14 physical lanes"

while true; do
  collect_remote "$local_gpu" 5003 "$local_gpu_key" "$local_gpu_repo"
  collect_remote "$kau" 8589 "$kau_key" "$kau_repo"
  status=$(PYTHONSAFEPATH=1 PYTHONPATH="$snapshot/src:$snapshot" "$python" \
    "$snapshot/$driver" status --root "$root")
  log_line "$(tr '\n' ' ' <<<"$status")"
  done_flag=$(PYTHONSAFEPATH=1 PYTHONPATH="$snapshot/src:$snapshot" "$python" -c \
    'import json,sys; print(str(json.load(sys.stdin)["done"]).lower())' <<<"$status")
  [[ $done_flag == true ]] && break
  sleep 20
done

PYTHONSAFEPATH=1 PYTHONPATH="$snapshot/src:$snapshot" "$python" "$snapshot/$driver" \
  report --root "$root" >"$root/reports-final.json"
log_line "$label Fast UCR-16 screen complete; no Q1 or follow-on campaign was started"
