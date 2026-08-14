#!/usr/bin/env bash
set -euo pipefail

project_root=${1:-.}
: "${PAC_LOGSIG_Q1_ROOT:?set PAC_LOGSIG_Q1_ROOT}"
: "${PAC_LOGSIG_Q1_SMOKE_ROOT:?set PAC_LOGSIG_Q1_SMOKE_ROOT}"
: "${PAC_LOGSIG_Q1_CLI_MODULE:?set PAC_LOGSIG_Q1_CLI_MODULE}"
: "${PAC_LOGSIG_Q1_CAMPAIGN_MODULE:?set PAC_LOGSIG_Q1_CAMPAIGN_MODULE}"
: "${PAC_LOGSIG_Q1_PREFLIGHT_MODULE:?set PAC_LOGSIG_Q1_PREFLIGHT_MODULE}"
: "${PAC_LOGSIG_Q1_SLUG:?set PAC_LOGSIG_Q1_SLUG}"
: "${PAC_LOGSIG_Q1_LABEL:?set PAC_LOGSIG_Q1_LABEL}"
: "${PAC_LOGSIG_Q1_SMOKE_KEY:?set PAC_LOGSIG_Q1_SMOKE_KEY}"
: "${PAC_LOGSIG_Q1_SOURCE_SNAPSHOT:?set PAC_LOGSIG_Q1_SOURCE_SNAPSHOT}"
: "${PAC_LOGSIG_Q1_EXPECTED_HASH:?set PAC_LOGSIG_Q1_EXPECTED_HASH}"
root=$PAC_LOGSIG_Q1_ROOT
smoke_root=$PAC_LOGSIG_Q1_SMOKE_ROOT
baseline_root=.omx/results/pac-baseline-fairness-maximal-20260714
snapshot_root=$PAC_LOGSIG_Q1_SOURCE_SNAPSHOT
expected_hash=$PAC_LOGSIG_Q1_EXPECTED_HASH
snapshot_name=$(basename "$snapshot_root")
module=$PAC_LOGSIG_Q1_CLI_MODULE
campaign_module=$PAC_LOGSIG_Q1_CAMPAIGN_MODULE
preflight_module=$PAC_LOGSIG_Q1_PREFLIGHT_MODULE
campaign_slug=$PAC_LOGSIG_Q1_SLUG
campaign_label=$PAC_LOGSIG_Q1_LABEL
smoke_key=$PAC_LOGSIG_Q1_SMOKE_KEY
poll_seconds=${PAC_LOGSIG_Q1_POLL_SECONDS:-30}
local_python=LOCAL_HOME_PLACEHOLDER/miniconda3/bin/python
local_gpu_python=LOCAL_HOME_PLACEHOLDER/miniconda3/envs/brelu/bin/python
kau_python=REMOTE_HOME_PLACEHOLDER/anaconda3/envs/torch/bin/python
local_gpu_key=LOCAL_HOME_PLACEHOLDER/.ssh/SSH_KEY_PLACEHOLDER
kau_key=LOCAL_HOME_PLACEHOLDER/.codex/attachments/OPAQUE_ID_PLACEHOLDER/SSH_KEY_PLACEHOLDER
local_gpu_host=local_gpu@REMOTE_HOST_PLACEHOLDER
kau_host=secondary_host@REMOTE_HOST_PLACEHOLDER
local_gpu_repo=LOCAL_HOME_PLACEHOLDER/lnet-external-20260718
kau_repo=REMOTE_HOME_PLACEHOLDER/lnet-terminal-20260718
: "${PAC_LOGSIG_Q1_LOG:?set PAC_LOGSIG_Q1_LOG}"
: "${PAC_LOGSIG_Q1_LOCK:?set PAC_LOGSIG_Q1_LOCK}"
log=$PAC_LOGSIG_Q1_LOG
lock=$PAC_LOGSIG_Q1_LOCK

cd "$project_root"
mkdir -p "$(dirname "$log")" "$(dirname "$lock")"
exec 9>"$lock"
if ! flock -n 9; then
  printf '%s %s supervisor is already active\n' "$(date --iso-8601=seconds)" "$campaign_label" >&2
  exit 0
fi

log_line() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$log"
}

sync_snapshot() {
  local host=$1 port=$2 key=$3 repo=$4 remote_snapshot
  remote_snapshot=$repo/.omx/source-snapshots/$snapshot_name
  ssh -i "$key" -p "$port" -o BatchMode=yes "$host" \
    "chmod -R u+w '$remote_snapshot' 2>/dev/null || true; rm -rf '$remote_snapshot'; mkdir -p '$remote_snapshot/src' '$remote_snapshot/optimization'"
  rsync -az -e "ssh -i $key -p $port -o BatchMode=yes" \
    "$snapshot_root/src/" "$host:$remote_snapshot/src/"
  rsync -az -e "ssh -i $key -p $port -o BatchMode=yes" \
    "$snapshot_root/optimization/" "$host:$remote_snapshot/optimization/"
  ssh -i "$key" -p "$port" -o BatchMode=yes "$host" "chmod -R a-w '$remote_snapshot'"
}

remote_hash() {
  local host=$1 port=$2 key=$3 repo=$4 python_path=$5 remote_snapshot
  remote_snapshot=$repo/.omx/source-snapshots/$snapshot_name
  ssh -i "$key" -p "$port" -o BatchMode=yes "$host" \
    "cd '$repo' && PYTHONDONTWRITEBYTECODE=1 PYTHONSAFEPATH=1 PYTHONPATH='$remote_snapshot/src:$remote_snapshot' '$python_path' -c 'from $campaign_module import code_sha256; print(code_sha256())'"
}

verify_and_sync_sources() {
  local actual
  actual=$(PYTHONDONTWRITEBYTECODE=1 PYTHONSAFEPATH=1 \
    PYTHONPATH="$snapshot_root/src:$snapshot_root" "$local_python" -c \
    "from $campaign_module import code_sha256; print(code_sha256())")
  [[ $actual == "$expected_hash" ]] || {
    log_line "local source hash mismatch: $actual"
    return 1
  }
  sync_snapshot "$local_gpu_host" 5003 "$local_gpu_key" "$local_gpu_repo"
  sync_snapshot "$kau_host" 8589 "$kau_key" "$kau_repo"
  [[ $(remote_hash "$local_gpu_host" 5003 "$local_gpu_key" "$local_gpu_repo" "$local_gpu_python") == "$expected_hash" ]]
  [[ $(remote_hash "$kau_host" 8589 "$kau_key" "$kau_repo" "$kau_python") == "$expected_hash" ]]
  log_line "frozen $campaign_label source verified locally and on both remote hosts: $actual"
}

run_preflight() {
  mkdir -p "$root/preflight"
  PYTHONSAFEPATH=1 PYTHONPATH="$snapshot_root/src:$snapshot_root" CUDA_VISIBLE_DEVICES=0 \
    "$local_python" -m "$preflight_module" --device cuda \
    >"$root/preflight/pro6000.json"
  local remote_snapshot=$local_gpu_repo/.omx/source-snapshots/$snapshot_name
  for gpu in 0 1; do
    ssh -i "$local_gpu_key" -p 5003 -o BatchMode=yes "$local_gpu_host" \
      "cd '$local_gpu_repo' && PYTHONSAFEPATH=1 PYTHONPATH='$remote_snapshot/src:$remote_snapshot' CUDA_VISIBLE_DEVICES='$gpu' '$local_gpu_python' -m '$preflight_module' --device cuda" \
      >"$root/preflight/local_gpu-gpu${gpu}.json"
  done
  remote_snapshot=$kau_repo/.omx/source-snapshots/$snapshot_name
  ssh -i "$kau_key" -p 8589 -o BatchMode=yes "$kau_host" \
    "cd '$kau_repo' && PYTHONSAFEPATH=1 PYTHONPATH='$remote_snapshot/src:$remote_snapshot' CUDA_VISIBLE_DEVICES=0 '$kau_python' -m '$preflight_module' --device cuda" \
    >"$root/preflight/kau-gpu0.json"
  log_line "CUDA preflight passed on PRO6000 and all three RTX4090 GPUs"
}

manifest_has_pending() {
  local python_path=$1 manifest=$2 completed_dir=$3
  "$python_path" - "$manifest" "$completed_dir" <<'PY'
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
completed_dir = Path(sys.argv[2])
scheduled = {json.loads(line)["key"] for line in manifest.read_text().splitlines() if line.strip()}
completed = set()
for path in completed_dir.glob("*.json"):
    try:
        row = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        continue
    if row.get("status") == "done":
        completed.add(row.get("job_key"))
raise SystemExit(0 if scheduled - completed else 1)
PY
}

collect_stage() {
  local stage=$1 bucket completed
  mkdir -p "$root/$stage/completed" "$root/$stage/failed" "$root/$stage/attempts"
  for bucket in completed failed attempts; do
    rsync -az -e "ssh -i $local_gpu_key -p 5003 -o BatchMode=yes" \
      "$local_gpu_host:$local_gpu_repo/$root/$stage/$bucket/" "$root/$stage/$bucket/" 2>/dev/null || true
    rsync -az -e "ssh -i $kau_key -p 8589 -o BatchMode=yes" \
      "$kau_host:$kau_repo/$root/$stage/$bucket/" "$root/$stage/$bucket/" 2>/dev/null || true
  done
  mkdir -p "$root/provenance"
  rsync -az -e "ssh -i $local_gpu_key -p 5003 -o BatchMode=yes" \
    "$local_gpu_host:$local_gpu_repo/$root/provenance/" "$root/provenance/" 2>/dev/null || true
  rsync -az -e "ssh -i $kau_key -p 8589 -o BatchMode=yes" \
    "$kau_host:$kau_repo/$root/provenance/" "$root/provenance/" 2>/dev/null || true
  for completed in "$root/$stage/completed/"*.json; do
    [[ -f $completed ]] || continue
    rm -f "$root/$stage/failed/$(basename "$completed")"
  done
}

sync_stage() {
  local stage=$1 host port key repo
  for remote in local_gpu kau; do
    if [[ $remote == local_gpu ]]; then
      host=$local_gpu_host; port=5003; key=$local_gpu_key; repo=$local_gpu_repo
    else
      host=$kau_host; port=8589; key=$kau_key; repo=$kau_repo
    fi
    ssh -i "$key" -p "$port" -o BatchMode=yes "$host" \
      "mkdir -p '$repo/$root/$stage' '$repo/$root/reports'"
    rsync -az -e "ssh -i $key -p $port -o BatchMode=yes" \
      "$root/$stage/" "$host:$repo/$root/$stage/"
    rsync -az -e "ssh -i $key -p $port -o BatchMode=yes" \
      "$root/reports/" "$host:$repo/$root/reports/"
    [[ ! -f $root/stage1/selection.json ]] || rsync -az \
      -e "ssh -i $key -p $port -o BatchMode=yes" "$root/stage1/selection.json" \
      "$host:$repo/$root/stage1/selection.json"
    [[ ! -f $root/stage2/selection.json ]] || rsync -az \
      -e "ssh -i $key -p $port -o BatchMode=yes" "$root/stage2/selection.json" \
      "$host:$repo/$root/stage2/selection.json"
  done
}

start_local_worker() {
  local stage=$1 lane=$2 manifest session logfile
  manifest=$root/$stage/manifests/$lane.jsonl
  session=$campaign_slug-$stage-$lane
  logfile=$root/$stage/logs/$lane.log
  [[ -f $manifest ]] || return 0
  tmux has-session -t "$session" 2>/dev/null && return 0
  manifest_has_pending "$local_python" "$manifest" "$root/$stage/completed" || return 0
  mkdir -p "$(dirname "$logfile")"
  tmux new-session -d -s "$session" \
    "cd '$project_root' && for retry in 1 2 3; do OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONSAFEPATH=1 PYTHONPATH='$snapshot_root/src:$snapshot_root' CUDA_VISIBLE_DEVICES=0 '$local_python' -m '$module' --stage worker --output-root '$root' --manifest '$manifest' --device cuda >>'$logfile' 2>&1; done"
}

start_remote_worker() {
  local host=$1 port=$2 key=$3 repo=$4 python_path=$5 stage=$6 lane=$7 gpu=$8
  local remote_snapshot manifest session logfile
  remote_snapshot=$repo/.omx/source-snapshots/$snapshot_name
  manifest=$root/$stage/manifests/$lane.jsonl
  session=$campaign_slug-$stage-$lane
  logfile=$root/$stage/logs/$lane.log
  ssh -i "$key" -p "$port" -o BatchMode=yes "$host" bash -s -- \
    "$repo" "$python_path" "$remote_snapshot" "$manifest" "$session" "$logfile" "$gpu" "$root" "$stage" "$module" <<'REMOTE'
set -euo pipefail
repo=$1; python_path=$2; source_root=$3; manifest=$4; session=$5; logfile=$6
gpu=$7; root=$8; stage=$9; module=${10}
cd "$repo"
[[ -f $manifest ]] || exit 0
tmux has-session -t "$session" 2>/dev/null && exit 0
if ! "$python_path" - "$manifest" "$root/$stage/completed" <<'PY'
import json
import sys
from pathlib import Path
manifest = Path(sys.argv[1])
completed = {json.loads(p.read_text()).get("job_key") for p in Path(sys.argv[2]).glob("*.json")}
scheduled = {json.loads(line)["key"] for line in manifest.read_text().splitlines() if line.strip()}
raise SystemExit(0 if scheduled - completed else 1)
PY
then
  exit 0
fi
mkdir -p "$(dirname "$logfile")"
tmux new-session -d -s "$session" \
  "cd '$repo' && for retry in 1 2 3; do OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONSAFEPATH=1 PYTHONPATH='$source_root/src:$source_root' CUDA_VISIBLE_DEVICES='$gpu' '$python_path' -m '$module' --stage worker --output-root '$root' --manifest '$manifest' --device cuda >>'$logfile' 2>&1; done"
REMOTE
}

ensure_workers() {
  local stage=$1 index lane
  for index in $(seq 0 10); do
    lane=$(printf 'pro6000-%02d' "$index")
    start_local_worker "$stage" "$lane"
  done
  start_remote_worker "$local_gpu_host" 5003 "$local_gpu_key" "$local_gpu_repo" "$local_gpu_python" \
    "$stage" local_gpu-gpu0 0 || true
  start_remote_worker "$local_gpu_host" 5003 "$local_gpu_key" "$local_gpu_repo" "$local_gpu_python" \
    "$stage" local_gpu-gpu1 1 || true
  start_remote_worker "$kau_host" 8589 "$kau_key" "$kau_repo" "$kau_python" \
    "$stage" kau-gpu0 0 || true
}

stage_done() {
  local stage=$1
  PYTHONSAFEPATH=1 PYTHONPATH="$snapshot_root/src:$snapshot_root" "$local_python" - \
    "$root" "$stage" "$campaign_module" <<'PY'
import importlib
import sys
from pathlib import Path
status = importlib.import_module(sys.argv[3]).status
payload = status(Path(sys.argv[1]))[sys.argv[2]]
raise SystemExit(0 if payload["done"] else 1)
PY
}

run_stage() {
  local stage=$1 snapshot
  sync_stage "$stage"
  while true; do
    collect_stage "$stage"
    snapshot=$(PYTHONSAFEPATH=1 PYTHONPATH="$snapshot_root/src:$snapshot_root" \
      "$local_python" -m "$module" --stage status --output-root "$root")
    log_line "stage=$stage $(tr '\n' ' ' <<<"$snapshot")"
    stage_done "$stage" && break
    ensure_workers "$stage"
    sleep "$poll_seconds"
  done
}

run_smoke() {
  rm -rf "$smoke_root"
  PYTHONSAFEPATH=1 PYTHONPATH="$snapshot_root/src:$snapshot_root" "$local_python" -m "$module" \
    --stage enqueue-stage1 --output-root "$smoke_root" --lanes 14 >/dev/null
  mkdir -p "$smoke_root/smoke"
  rg --no-filename "$smoke_key" \
    "$smoke_root/stage1/manifests/"*.jsonl >"$smoke_root/smoke/one.jsonl"
  CUDA_VISIBLE_DEVICES=0 PYTHONSAFEPATH=1 PYTHONPATH="$snapshot_root/src:$snapshot_root" \
    "$local_python" -m "$module" --stage worker --output-root "$smoke_root" \
    --manifest "$smoke_root/smoke/one.jsonl" --device cuda >/dev/null
  log_line "ECGFiveDays validation smoke passed"
}

verify_and_sync_sources
run_preflight
run_smoke

if [[ ! -f $root/stage1/contract.json ]]; then
  PYTHONSAFEPATH=1 PYTHONPATH="$snapshot_root/src:$snapshot_root" "$local_python" -m "$module" \
    --stage enqueue-stage1 --output-root "$root" --lanes 14 | tee -a "$log"
fi
run_stage stage1
if [[ ! -f $root/stage1/selection.json ]]; then
  PYTHONSAFEPATH=1 PYTHONPATH="$snapshot_root/src:$snapshot_root" "$local_python" -m "$module" \
    --stage select-stage1 --output-root "$root" --lanes 14 | tee -a "$log"
fi
run_stage stage2
if [[ ! -f $root/stage2/selection.json ]]; then
  PYTHONSAFEPATH=1 PYTHONPATH="$snapshot_root/src:$snapshot_root" "$local_python" -m "$module" \
    --stage select-stage2 --output-root "$root" --baseline-root "$baseline_root" | tee -a "$log"
fi
if [[ ! -f $root/final/contract.json ]]; then
  PYTHONSAFEPATH=1 PYTHONPATH="$snapshot_root/src:$snapshot_root" "$local_python" -m "$module" \
    --stage enqueue-final --output-root "$root" --baseline-root "$baseline_root" --lanes 14 \
    | tee -a "$log"
fi
mkdir -p "$root/provenance"
rsync -a "$baseline_root/provenance/" "$root/provenance/"
run_stage final
log_line "LogSig $campaign_label Q1 execution complete"
