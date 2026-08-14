#!/usr/bin/env bash
set -uo pipefail

project_root=${PROJECT_ROOT:-.}
campaign=${1:-pac-pointwise-identity-capacity-q1-final-20260722}
local_root="$project_root/.omx/results/$campaign"
local_workspace=${LOCAL_WORKSPACE:-LOCAL_HOME_PLACEHOLDER/lnet-pointwise-q1-final-20260722}
kau_workspace=${REMOTE_WORKSPACE:-REMOTE_HOME_PLACEHOLDER/lnet-pointwise-q1-final-20260722}
local_gpu_workspace=${LOCAL_GPU_WORKSPACE:-LOCAL_HOME_PLACEHOLDER/lnet-pointwise-q1-final-20260722}
local_python=${LOCAL_PYTHON:-LOCAL_HOME_PLACEHOLDER/miniconda3/bin/python}
local_gpu_python=${LOCAL_GPU_PYTHON:-LOCAL_HOME_PLACEHOLDER/miniconda3/envs/flow/bin/python}
kau_python=${REMOTE_PYTHON:-REMOTE_HOME_PLACEHOLDER/anaconda3/envs/vertex/bin/python}
kau_root="$kau_workspace/.omx/results/$campaign"
local_gpu_root="$local_gpu_workspace/.omx/results/$campaign"
kau_ssh="ssh -i LOCAL_HOME_PLACEHOLDER/.codex/attachments/OPAQUE_ID_PLACEHOLDER/SSH_KEY_PLACEHOLDER -p 8589 -o BatchMode=yes"
local_gpu_ssh="ssh -i LOCAL_HOME_PLACEHOLDER/.ssh/SSH_KEY_PLACEHOLDER -p 5003 -o BatchMode=yes"

mkdir -p "$local_root/logs"

sync_results() {
  rsync -az -e "$kau_ssh" \
    "secondary_host@REMOTE_HOST_PLACEHOLDER:$kau_root/final/" "$local_root/final/" || true
  rsync -az -e "$local_gpu_ssh" \
    "local_gpu@REMOTE_HOST_PLACEHOLDER:$local_gpu_root/final/" "$local_root/final/" || true
}

campaign_done() {
  PYTHONPATH="$project_root/src:$project_root" \
    "$local_python" - "$local_root" <<'PY'
import sys
from pathlib import Path
from lnet.pac_pointwise_identity_capacity_campaign import status

raise SystemExit(0 if status(Path(sys.argv[1]))["final"]["done"] else 1)
PY
}

ensure_local_lane() {
  local lane=$1 session="identity-q1-$1"
  tmux has-session -t "$session" 2>/dev/null && return 0
  tmux new-session -d -s "$session" \
    "cd '$local_workspace' && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MAX_STAGNANT_PASSES=3 scripts/run_pointwise_identity_resume_lane.sh '.omx/results/$campaign/final/aggressive-queues/$lane.txt' 0 '$local_python' '.omx/results/$campaign' '.omx/data/ucr' 'data/external'"
}

ensure_remote_lane() {
  local ssh_command=$1 host=$2 workspace=$3 python_path=$4 lane=$5 gpu=$6
  $ssh_command "$host" bash -s -- \
    "$workspace" "$python_path" "$campaign" "$lane" "$gpu" <<'REMOTE'
set -euo pipefail
workspace=$1; python_path=$2; campaign=$3; lane=$4; gpu=$5
session="identity-q1-$lane"
tmux has-session -t "$session" 2>/dev/null && exit 0
tmux new-session -d -s "$session" \
  "cd '$workspace' && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MAX_STAGNANT_PASSES=3 scripts/run_pointwise_identity_resume_lane.sh '.omx/results/$campaign/final/aggressive-queues/$lane.txt' '$gpu' '$python_path' '.omx/results/$campaign' '.omx/data/ucr' 'data/external'"
REMOTE
}

while true; do
  sync_results
  PYTHONPATH="$project_root/src:$project_root" \
    "$local_python" -m lnet.pac_pointwise_identity_capacity_cli \
      --stage status --output-root "$local_root" \
      > "$local_root/logs/latest-status.json" || true
  campaign_done && break
  for index in $(seq 0 7); do
    ensure_local_lane "$(printf 'pro-%02d' "$index")"
  done
  ensure_remote_lane "$local_gpu_ssh" local_gpu@REMOTE_HOST_PLACEHOLDER \
    "$local_gpu_workspace" "$local_gpu_python" local_gpu-g0-0 0 || true
  ensure_remote_lane "$local_gpu_ssh" local_gpu@REMOTE_HOST_PLACEHOLDER \
    "$local_gpu_workspace" "$local_gpu_python" local_gpu-g0-1 0 || true
  ensure_remote_lane "$local_gpu_ssh" local_gpu@REMOTE_HOST_PLACEHOLDER \
    "$local_gpu_workspace" "$local_gpu_python" local_gpu-g1-0 1 || true
  ensure_remote_lane "$local_gpu_ssh" local_gpu@REMOTE_HOST_PLACEHOLDER \
    "$local_gpu_workspace" "$local_gpu_python" local_gpu-g1-1 1 || true
  ensure_remote_lane "$kau_ssh" secondary_host@REMOTE_HOST_PLACEHOLDER \
    "$kau_workspace" "$kau_python" kau-0 0 || true
  ensure_remote_lane "$kau_ssh" secondary_host@REMOTE_HOST_PLACEHOLDER \
    "$kau_workspace" "$kau_python" kau-1 0 || true
  sleep 60
done
sync_results
PYTHONPATH="$project_root/src:$project_root" \
  "$local_python" -m lnet.pac_pointwise_identity_capacity_cli \
    --stage status --output-root "$local_root" \
    > "$local_root/logs/latest-status.json"
