#!/usr/bin/env bash
set -euo pipefail

project_root=${1:-.}
source_root=${2:-$project_root/.omx/source-snapshots/pac-h-compact-lag124-tied-q1-20260721}
snapshot_name=$(basename "$source_root")
root=.omx/results/pac-h-compact-lag124-tied-q1-final-20260721
baseline_root=.omx/results/pac-baseline-fairness-maximal-20260714
module=lnet.pac_h_compact_lag124_tied_q1_cli
campaign=lnet.pac_h_compact_lag124_tied_q1_campaign
lanes=14
poll_seconds=${PAC_H_LAG124_TIED_Q1_POLL_SECONDS:-30}
local_python=LOCAL_HOME_PLACEHOLDER/miniconda3/bin/python
local_gpu_python=LOCAL_HOME_PLACEHOLDER/miniconda3/envs/brelu/bin/python
kau_python=REMOTE_HOME_PLACEHOLDER/anaconda3/envs/torch/bin/python
local_gpu_key=LOCAL_HOME_PLACEHOLDER/.ssh/SSH_KEY_PLACEHOLDER
kau_key=LOCAL_HOME_PLACEHOLDER/.codex/attachments/OPAQUE_ID_PLACEHOLDER/SSH_KEY_PLACEHOLDER
local_gpu_host=local_gpu@REMOTE_HOST_PLACEHOLDER
kau_host=secondary_host@REMOTE_HOST_PLACEHOLDER
local_gpu_repo=LOCAL_HOME_PLACEHOLDER/lnet-external-20260718
kau_repo=REMOTE_HOME_PLACEHOLDER/lnet-terminal-20260718
log=.omx/logs/pac-h-compact-lag124-tied-q1-final-20260721-supervisor.log

cd "$project_root"

sync_sources() {
  local local_gpu_snapshot=$local_gpu_repo/.omx/source-snapshots/$snapshot_name
  local kau_snapshot=$kau_repo/.omx/source-snapshots/$snapshot_name
  ssh -i "$local_gpu_key" -p 5003 -o BatchMode=yes "$local_gpu_host" \
    "chmod -R u+w '$local_gpu_snapshot' 2>/dev/null || true; rm -rf '$local_gpu_snapshot'; mkdir -p '$local_gpu_snapshot'"
  ssh -i "$kau_key" -p 8589 -o BatchMode=yes "$kau_host" \
    "chmod -R u+w '$kau_snapshot' 2>/dev/null || true; rm -rf '$kau_snapshot'; mkdir -p '$kau_snapshot'"
  rsync -az -e "ssh -i $local_gpu_key -p 5003 -o BatchMode=yes" \
    "$source_root/" "$local_gpu_host:$local_gpu_snapshot/"
  rsync -az -e "ssh -i $kau_key -p 8589 -o BatchMode=yes" \
    "$source_root/" "$kau_host:$kau_snapshot/"
  ssh -i "$local_gpu_key" -p 5003 -o BatchMode=yes "$local_gpu_host" "chmod -R a-w '$local_gpu_snapshot'"
  ssh -i "$kau_key" -p 8589 -o BatchMode=yes "$kau_host" "chmod -R a-w '$kau_snapshot'"
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
  mkdir -p "$root/$stage/completed" "$root/$stage/failed" "$root/$stage/attempts" "$root/provenance"
  for bucket in completed failed attempts; do
    rsync -az -e "ssh -i $local_gpu_key -p 5003 -o BatchMode=yes" \
      "$local_gpu_host:$local_gpu_repo/$root/$stage/$bucket/" "$root/$stage/$bucket/" 2>/dev/null || true
    rsync -az -e "ssh -i $kau_key -p 8589 -o BatchMode=yes" \
      "$kau_host:$kau_repo/$root/$stage/$bucket/" "$root/$stage/$bucket/" 2>/dev/null || true
  done
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
  local stage=$1
  ssh -i "$local_gpu_key" -p 5003 -o BatchMode=yes "$local_gpu_host" \
    "mkdir -p '$local_gpu_repo/$root/$stage' '$local_gpu_repo/$root/reports'"
  ssh -i "$kau_key" -p 8589 -o BatchMode=yes "$kau_host" \
    "mkdir -p '$kau_repo/$root/$stage' '$kau_repo/$root/reports'"
  if [[ $stage == stage2 ]]; then
    rsync -az -e "ssh -i $local_gpu_key -p 5003 -o BatchMode=yes" "$root/stage1/selection.json" \
      "$local_gpu_host:$local_gpu_repo/$root/stage1/selection.json"
    rsync -az -e "ssh -i $kau_key -p 8589 -o BatchMode=yes" "$root/stage1/selection.json" \
      "$kau_host:$kau_repo/$root/stage1/selection.json"
  elif [[ $stage == final ]]; then
    rsync -az -e "ssh -i $local_gpu_key -p 5003 -o BatchMode=yes" "$root/stage2/selection.json" \
      "$local_gpu_host:$local_gpu_repo/$root/stage2/selection.json"
    rsync -az -e "ssh -i $kau_key -p 8589 -o BatchMode=yes" "$root/stage2/selection.json" \
      "$kau_host:$kau_repo/$root/stage2/selection.json"
  fi
  rsync -az -e "ssh -i $local_gpu_key -p 5003 -o BatchMode=yes" "$root/$stage/" \
    "$local_gpu_host:$local_gpu_repo/$root/$stage/"
  rsync -az -e "ssh -i $kau_key -p 8589 -o BatchMode=yes" "$root/$stage/" \
    "$kau_host:$kau_repo/$root/$stage/"
  rsync -az -e "ssh -i $local_gpu_key -p 5003 -o BatchMode=yes" "$root/reports/" \
    "$local_gpu_host:$local_gpu_repo/$root/reports/"
  rsync -az -e "ssh -i $kau_key -p 8589 -o BatchMode=yes" "$root/reports/" \
    "$kau_host:$kau_repo/$root/reports/"
}

ensure_local_workers() {
  local stage=$1 index manifest session logfile
  for index in $(seq 0 7); do
    manifest=$(printf '%s/%s/manifests/worker-%02d.jsonl' "$root" "$stage" "$index")
    session=$(printf 'pac-h-lag124-tied-%s-local-%02d' "$stage" "$index")
    logfile=$(printf '%s/%s/logs/local-%02d.log' "$root" "$stage" "$index")
    [[ -f $manifest ]] || continue
    tmux has-session -t "$session" 2>/dev/null && continue
    manifest_has_pending "$local_python" "$manifest" "$root/$stage/completed" || continue
    mkdir -p "$(dirname "$logfile")"
    tmux new-session -d -s "$session" \
      "cd '$project_root' && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONSAFEPATH=1 PYTHONPATH='$source_root/src' CUDA_VISIBLE_DEVICES=0 '$local_python' -m '$module' --stage worker --output-root '$root' --manifest '$manifest' --device cuda >>'$logfile' 2>&1"
  done
}

ensure_remote_worker() {
  local host=$1 port=$2 key=$3 repo=$4 python_path=$5 stage=$6 index=$7 gpu=$8 label=$9
  local manifest session logfile remote_source
  remote_source=$repo/.omx/source-snapshots/$snapshot_name
  manifest=$(printf '%s/%s/manifests/worker-%02d.jsonl' "$root" "$stage" "$index")
  session=$(printf 'pac-h-lag124-tied-%s-%s-%02d' "$stage" "$label" "$index")
  logfile=$(printf '%s/%s/logs/%s-%02d.log' "$root" "$stage" "$label" "$index")
  ssh -i "$key" -p "$port" -o BatchMode=yes -o ConnectTimeout=10 "$host" bash -s -- \
    "$repo" "$python_path" "$manifest" "$stage" "$session" "$logfile" "$gpu" "$root" "$module" "$remote_source" <<'REMOTE'
set -euo pipefail
repo=$1; python_path=$2; manifest=$3; stage=$4; session=$5; logfile=$6; gpu=$7; root=$8; module=$9; source_root=${10}
cd "$repo"
[[ -f $manifest ]] || exit 0
tmux has-session -t "$session" 2>/dev/null && exit 0
if ! "$python_path" - "$manifest" "$root/$stage/completed" <<'PY'
import json
import sys
from pathlib import Path
manifest = Path(sys.argv[1])
completed = Path(sys.argv[2])
scheduled = {json.loads(line)["key"] for line in manifest.read_text().splitlines() if line.strip()}
done = {json.loads(path.read_text()).get("job_key") for path in completed.glob("*.json")}
raise SystemExit(0 if scheduled - done else 1)
PY
then
  exit 0
fi
mkdir -p "$(dirname "$logfile")"
tmux new-session -d -s "$session" \
  "cd '$repo' && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONSAFEPATH=1 PYTHONPATH='$source_root/src' CUDA_VISIBLE_DEVICES='$gpu' '$python_path' -m '$module' --stage worker --output-root '$root' --manifest '$manifest' --device cuda >>'$logfile' 2>&1"
REMOTE
}

stage_done() {
  local stage=$1
  PYTHONPATH="$source_root/src" "$local_python" - "$root" "$stage" "$campaign" <<'PY'
import importlib
import sys
from pathlib import Path
campaign = importlib.import_module(sys.argv[3])
raise SystemExit(0 if campaign.status(Path(sys.argv[1]))[sys.argv[2]]["done"] else 1)
PY
}

run_stage() {
  local stage=$1 snapshot
  sync_stage "$stage"
  while true; do
    collect_stage "$stage"
    snapshot=$(PYTHONPATH="$source_root/src" "$local_python" -m "$module" --stage status --output-root "$root")
    printf '%s stage=%s %s\n' "$(date --iso-8601=seconds)" "$stage" "$(tr '\n' ' ' <<<"$snapshot")" | tee -a "$log"
    stage_done "$stage" && break
    ensure_local_workers "$stage"
    ensure_remote_worker "$local_gpu_host" 5003 "$local_gpu_key" "$local_gpu_repo" "$local_gpu_python" "$stage" 8 0 local_gpu || true
    ensure_remote_worker "$local_gpu_host" 5003 "$local_gpu_key" "$local_gpu_repo" "$local_gpu_python" "$stage" 9 0 local_gpu || true
    ensure_remote_worker "$local_gpu_host" 5003 "$local_gpu_key" "$local_gpu_repo" "$local_gpu_python" "$stage" 10 1 local_gpu || true
    ensure_remote_worker "$local_gpu_host" 5003 "$local_gpu_key" "$local_gpu_repo" "$local_gpu_python" "$stage" 11 1 local_gpu || true
    ensure_remote_worker "$kau_host" 8589 "$kau_key" "$kau_repo" "$kau_python" "$stage" 12 0 kau || true
    ensure_remote_worker "$kau_host" 8589 "$kau_key" "$kau_repo" "$kau_python" "$stage" 13 0 kau || true
    sleep "$poll_seconds"
  done
}

sync_sources
if [[ ! -f $root/stage1/contract.json ]]; then
  PYTHONPATH="$source_root/src" "$local_python" -m "$module" --stage enqueue-stage1 --output-root "$root" --lanes "$lanes" | tee -a "$log"
fi
run_stage stage1
PYTHONPATH="$source_root/src" "$local_python" -m "$module" --stage select-stage1 --output-root "$root" --lanes "$lanes" | tee -a "$log"
run_stage stage2
PYTHONPATH="$source_root/src" "$local_python" -m "$module" --stage select-stage2 --output-root "$root" --baseline-root "$baseline_root" | tee -a "$log"
PYTHONPATH="$source_root/src" "$local_python" -m "$module" --stage enqueue-final --output-root "$root" --baseline-root "$baseline_root" --lanes "$lanes" | tee -a "$log"
mkdir -p "$root/provenance"
rsync -a "$baseline_root/provenance/" "$root/provenance/"
run_stage final
printf '%s tied H-compact lag-(1,2,4) Q1 execution complete\n' "$(date --iso-8601=seconds)" | tee -a "$log"
