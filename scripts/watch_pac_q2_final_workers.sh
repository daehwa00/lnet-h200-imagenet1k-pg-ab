#!/usr/bin/env bash
set -euo pipefail

project_root=${1:-.}
poll_seconds=${PAC_Q2_WORKER_WATCHDOG_POLL_SECONDS:-60}
once=${PAC_Q2_WORKER_WATCHDOG_ONCE:-0}

root=.omx/results/pac-alphabet-q1q2-final-20260719
local_gpu_key=LOCAL_HOME_PLACEHOLDER/.codex/attachments/OPAQUE_ID_PLACEHOLDER/SSH_KEY_PLACEHOLDER
kau_key=LOCAL_HOME_PLACEHOLDER/.codex/attachments/OPAQUE_ID_PLACEHOLDER/SSH_KEY_PLACEHOLDER
local_gpu_host=local_gpu@REMOTE_HOST_PLACEHOLDER
kau_host=secondary_host@REMOTE_HOST_PLACEHOLDER
local_gpu_repo=LOCAL_HOME_PLACEHOLDER/lnet-external-20260718
kau_repo=REMOTE_HOME_PLACEHOLDER/lnet-terminal-20260718

cd "$project_root"

manifest_has_pending() {
  local manifest=$1
  local completed_dir=$2
  python - "$manifest" "$completed_dir" <<'PY'
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
completed_dir = Path(sys.argv[2])
scheduled = {
    json.loads(line)["key"]
    for line in manifest.read_text().splitlines()
    if line.strip()
}
completed = set()
for path in completed_dir.glob("*.json"):
    try:
        row = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        continue
    if row.get("status") == "done":
        completed.add(row.get("job_key") or row.get("key"))
raise SystemExit(0 if scheduled - completed else 1)
PY
}

ensure_local() {
  local session=$1
  local manifest=$2
  local logfile=$3
  if tmux has-session -t "$session" 2>/dev/null; then
    return
  fi
  if ! manifest_has_pending "$manifest" "$root/q2_final/completed"; then
    return
  fi
  tmux new-session -d -s "$session" \
    "cd '$project_root' && for retry in 1 2 3; do OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 LOCAL_HOME_PLACEHOLDER/miniconda3/bin/python -m lnet.pac_alphabet_q1_q2_final_cli --stage worker --output-root '$root' --manifest '$manifest' --device cuda >>'$logfile' 2>&1; done"
  printf '%s restarted local worker %s\n' "$(date --iso-8601=seconds)" "$session"
}

ensure_remote() {
  local host=$1
  local port=$2
  local key=$3
  local repo=$4
  local session=$5
  local manifest=$6
  local cuda_device=$7
  local python_path=$8
  local source_path=$9
  local logfile=${10}
  ssh -i "$key" -p "$port" -o BatchMode=yes -o ConnectTimeout=10 "$host" \
    bash -s -- "$repo" "$session" "$manifest" "$cuda_device" "$python_path" \
    "$source_path" "$logfile" "$root" <<'REMOTE'
set -euo pipefail
repo=$1
session=$2
manifest=$3
cuda_device=$4
python_path=$5
source_path=$6
logfile=$7
root=$8
cd "$repo"
if tmux has-session -t "$session" 2>/dev/null; then
  exit 0
fi
if ! "$python_path" - "$manifest" "$root/q2_final/completed" <<'PY'
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
completed_dir = Path(sys.argv[2])
scheduled = {
    json.loads(line)["key"]
    for line in manifest.read_text().splitlines()
    if line.strip()
}
completed = set()
for path in completed_dir.glob("*.json"):
    try:
        row = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        continue
    if row.get("status") == "done":
        completed.add(row.get("job_key") or row.get("key"))
raise SystemExit(0 if scheduled - completed else 1)
PY
then
  exit 0
fi
tmux new-session -d -s "$session" \
  "cd '$repo' && for retry in 1 2 3; do OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH='$source_path' CUDA_VISIBLE_DEVICES='$cuda_device' '$python_path' -m lnet.pac_alphabet_q1_q2_final_cli --stage worker --output-root '$root' --manifest '$manifest' --device cuda >>'$logfile' 2>&1; done"
printf '%s restarted remote worker %s\n' "$(date --iso-8601=seconds)" "$session"
REMOTE
}

guard_ensure() {
  local target=$1
  shift
  if ! "$@"; then
    printf '%s watchdog check failed for %s; retrying next poll\n' \
      "$(date --iso-8601=seconds)" "$target" >&2
  fi
}

while [[ ! -f $root/pipeline_complete.json ]]; do
  guard_ensure local0 ensure_local \
    pac-alphabet-q2-final-local0-split-20260720 \
    .omx/tmp/q2-local-split-20260720/q2-final-local0.jsonl \
    .omx/logs/q2-final-local0-split-20260720.log
  guard_ensure local1 ensure_local \
    pac-alphabet-q2-final-local1-split-20260720 \
    .omx/tmp/q2-local-split-20260720/q2-final-local1.jsonl \
    .omx/logs/q2-final-local1-split-20260720.log

  guard_ensure local_gpu0-speech ensure_remote "$local_gpu_host" 5003 "$local_gpu_key" "$local_gpu_repo" \
    pac-alphabet-q2-final-local_gpu0-speech-capable-20260720 \
    .omx/tmp/q2-local_gpu0-split-20260720/worker-local_gpu0-speech-capable.jsonl 0 \
    LOCAL_HOME_PLACEHOLDER/miniconda3/envs/brelu/bin/python . \
    .omx/logs/q2-final-local_gpu0-speech-capable-20260720.log
  guard_ensure local_gpu0-nonspeech ensure_remote "$local_gpu_host" 5003 "$local_gpu_key" "$local_gpu_repo" \
    pac-alphabet-q2-final-local_gpu0-nonspeech-20260720 \
    .omx/tmp/q2-local_gpu0-split-20260720/worker-local_gpu0-nonspeech.jsonl 0 \
    LOCAL_HOME_PLACEHOLDER/miniconda3/envs/brelu/bin/python . \
    .omx/logs/q2-final-local_gpu0-nonspeech-20260720.log
  guard_ensure local_gpu1-0 ensure_remote "$local_gpu_host" 5003 "$local_gpu_key" "$local_gpu_repo" \
    pac-alphabet-q2-final-local_gpu1-0-split-20260720 \
    .omx/tmp/q2-local_gpu1-split-20260720/q2-final-local_gpu1-0.jsonl 1 \
    LOCAL_HOME_PLACEHOLDER/miniconda3/envs/brelu/bin/python . \
    .omx/logs/q2-final-local_gpu1-0-split-20260720.log
  guard_ensure local_gpu1-1 ensure_remote "$local_gpu_host" 5003 "$local_gpu_key" "$local_gpu_repo" \
    pac-alphabet-q2-final-local_gpu1-1-split-20260720 \
    .omx/tmp/q2-local_gpu1-split-20260720/q2-final-local_gpu1-1.jsonl 1 \
    LOCAL_HOME_PLACEHOLDER/miniconda3/envs/brelu/bin/python . \
    .omx/logs/q2-final-local_gpu1-1-split-20260720.log

  guard_ensure kau0 ensure_remote "$kau_host" 8589 "$kau_key" "$kau_repo" \
    pac-alphabet-q2-final-kau0-split-20260720 .omx/tmp/q2-final-kau0-split.jsonl 0 \
    REMOTE_HOME_PLACEHOLDER/anaconda3/envs/torch/bin/python src \
    .omx/logs/q2-final-kau0-split-20260720.log
  guard_ensure kau1 ensure_remote "$kau_host" 8589 "$kau_key" "$kau_repo" \
    pac-alphabet-q2-final-kau1-split-20260720 .omx/tmp/q2-final-kau1-split.jsonl 0 \
    REMOTE_HOME_PLACEHOLDER/anaconda3/envs/torch/bin/python src \
    .omx/logs/q2-final-kau1-split-20260720.log

  printf '%s watchdog sweep complete for 8 immutable worker manifests\n' \
    "$(date --iso-8601=seconds)"
  if [[ $once == 1 ]]; then
    break
  fi
  sleep "$poll_seconds"
done
