#!/usr/bin/env bash
set -euo pipefail

project_root=${1:-.}
default_snapshot=$project_root/.omx/source-snapshots/pac-h-compact-overnight-30task-20260722
source_root=${2:-${PAC_HCO_SOURCE_ROOT:-$default_snapshot}}
snapshot_name=$(basename "$source_root")
root=${PAC_HCO_ROOT:-.omx/results/pac-h-compact-overnight-30task-20260722}
module=${PAC_HCO_MODULE:-lnet.pac_h_compact_overnight_cli}
stage=${PAC_HCO_STAGE:-stage1}
poll_seconds=${PAC_HCO_POLL_SECONDS:-20}
worker_prefix=${PAC_HCO_WORKER_PREFIX:-pac-hco30}
campaign_label=${PAC_HCO_CAMPAIGN_LABEL:-H-compact overnight 30-task screen}
wait_for_session=${PAC_HCO_WAIT_FOR_SESSION:-}
local_python=LOCAL_HOME_PLACEHOLDER/miniconda3/bin/python
local_gpu_python=LOCAL_HOME_PLACEHOLDER/miniconda3/envs/brelu/bin/python
kau_python=REMOTE_HOME_PLACEHOLDER/anaconda3/envs/torch/bin/python
local_gpu_key=LOCAL_HOME_PLACEHOLDER/.ssh/SSH_KEY_PLACEHOLDER
kau_key=LOCAL_HOME_PLACEHOLDER/.codex/attachments/OPAQUE_ID_PLACEHOLDER/SSH_KEY_PLACEHOLDER
local_gpu_host=local_gpu@REMOTE_HOST_PLACEHOLDER
kau_host=secondary_host@REMOTE_HOST_PLACEHOLDER
local_gpu_repo=LOCAL_HOME_PLACEHOLDER/lnet-external-20260718
kau_repo=REMOTE_HOME_PLACEHOLDER/lnet-terminal-20260718
log=${PAC_HCO_LOG:-.omx/logs/pac-h-compact-overnight-30task-20260722-supervisor.log}

cd "$project_root"
mkdir -p "$(dirname "$log")" "$root/$stage/logs"

while [[ -n $wait_for_session ]] && tmux has-session -t "$wait_for_session" 2>/dev/null; do
  printf '%s waiting for predecessor %s\n' "$(date --iso-8601=seconds)" "$wait_for_session" | tee -a "$log"
  sleep 60
done

pending_manifest() {
  local python_path=$1 manifest=$2 completed_dir=$3
  "$python_path" - "$manifest" "$completed_dir" <<'PY'
import json
import sys
from pathlib import Path
manifest = Path(sys.argv[1])
completed = Path(sys.argv[2])
scheduled = {json.loads(line)["key"] for line in manifest.read_text().splitlines() if line.strip()}
done = set()
for path in completed.glob("*.json"):
    try:
        row = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        continue
    if row.get("status") == "done":
        done.add(row.get("job_key"))
raise SystemExit(0 if scheduled - done else 1)
PY
}

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

run_preflight() {
  local local_gpu_snapshot=$local_gpu_repo/.omx/source-snapshots/$snapshot_name
  local kau_snapshot=$kau_repo/.omx/source-snapshots/$snapshot_name
  printf '%s local CUDA preflight start\n' "$(date --iso-8601=seconds)" | tee -a "$log"
  PYTHONPATH="$source_root/src" "$local_python" -m "$module" --stage preflight --device cuda \
    >"$root/preflight-local.json"
  ssh -i "$local_gpu_key" -p 5003 -o BatchMode=yes "$local_gpu_host" \
    "cd '$local_gpu_repo' && PYTHONSAFEPATH=1 PYTHONPATH='$local_gpu_snapshot/src' CUDA_VISIBLE_DEVICES=0 '$local_gpu_python' -m '$module' --stage preflight --device cuda" \
    >"$root/preflight-local_gpu.json"
  ssh -i "$kau_key" -p 8589 -o BatchMode=yes "$kau_host" \
    "cd '$kau_repo' && PYTHONSAFEPATH=1 PYTHONPATH='$kau_snapshot/src' CUDA_VISIBLE_DEVICES=0 '$kau_python' -m '$module' --stage preflight --device cuda" \
    >"$root/preflight-kau.json"
  printf '%s all-host CUDA preflight passed\n' "$(date --iso-8601=seconds)" | tee -a "$log"
}

sync_campaign() {
  for host_spec in local_gpu kau; do
    if [[ $host_spec == local_gpu ]]; then
      host=$local_gpu_host; port=5003; key=$local_gpu_key; repo=$local_gpu_repo
    else
      host=$kau_host; port=8589; key=$kau_key; repo=$kau_repo
    fi
    ssh -i "$key" -p "$port" -o BatchMode=yes "$host" \
      "mkdir -p '$repo/$root/$stage' '$repo/$root/reports'"
    rsync -az -e "ssh -i $key -p $port -o BatchMode=yes" \
      "$root/$stage/" "$host:$repo/$root/$stage/"
    rsync -az -e "ssh -i $key -p $port -o BatchMode=yes" \
      "$root/contract.json" "$host:$repo/$root/contract.json"
    rsync -az -e "ssh -i $key -p $port -o BatchMode=yes" \
      "$root/reports/" "$host:$repo/$root/reports/"
  done
}

collect_remote() {
  mkdir -p "$root/$stage/completed" "$root/$stage/failed" "$root/$stage/attempts" "$root/provenance"
  for spec in "local_gpu:$local_gpu_host:5003:$local_gpu_key:$local_gpu_repo" "kau:$kau_host:8589:$kau_key:$kau_repo"; do
    IFS=: read -r label host port key repo <<<"$spec"
    for bucket in completed failed attempts; do
      rsync -az -e "ssh -i $key -p $port -o BatchMode=yes -o ConnectTimeout=8" \
        "$host:$repo/$root/$stage/$bucket/" "$root/$stage/$bucket/" 2>/dev/null || true
    done
    rsync -az -e "ssh -i $key -p $port -o BatchMode=yes -o ConnectTimeout=8" \
      "$host:$repo/$root/provenance/" "$root/provenance/" 2>/dev/null || true
  done
  local completed
  for completed in "$root/$stage/completed/"*.json; do
    [[ -f $completed ]] || continue
    rm -f "$root/$stage/failed/$(basename "$completed")"
  done
}

ensure_local_workers() {
  local index manifest session logfile
  for index in $(seq 0 7); do
    manifest=$(printf '%s/%s/manifests/worker-%02d.jsonl' "$root" "$stage" "$index")
    session=$(printf '%s-local-%02d' "$worker_prefix" "$index")
    logfile=$(printf '%s/%s/logs/local-%02d.log' "$root" "$stage" "$index")
    [[ -f $manifest ]] || continue
    tmux has-session -t "$session" 2>/dev/null && continue
    pending_manifest "$local_python" "$manifest" "$root/$stage/completed" || continue
    tmux new-session -d -s "$session" \
      "cd '$project_root' && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONSAFEPATH=1 PYTHONPATH='$source_root/src' CUDA_VISIBLE_DEVICES=0 '$local_python' -m '$module' --stage worker --output-root '$root' --manifest '$manifest' --device cuda >>'$logfile' 2>&1"
  done
}

ensure_remote_worker() {
  local label=$1 host=$2 port=$3 key=$4 repo=$5 python_path=$6 index=$7 gpu=$8
  local manifest session logfile remote_source
  manifest=$(printf '%s/%s/manifests/worker-%02d.jsonl' "$root" "$stage" "$index")
  session=$(printf '%s-%s-%02d' "$worker_prefix" "$label" "$index")
  logfile=$(printf '%s/%s/logs/%s-%02d.log' "$root" "$stage" "$label" "$index")
  remote_source=$repo/.omx/source-snapshots/$snapshot_name
  ssh -i "$key" -p "$port" -o BatchMode=yes -o ConnectTimeout=8 "$host" bash -s -- \
    "$repo" "$python_path" "$manifest" "$session" "$logfile" "$gpu" "$root" "$module" "$remote_source" <<'REMOTE'
set -euo pipefail
repo=$1; python_path=$2; manifest=$3; session=$4; logfile=$5; gpu=$6; root=$7; module=$8; source_root=$9
cd "$repo"
[[ -f $manifest ]] || exit 0
tmux has-session -t "$session" 2>/dev/null && exit 0
stage=$(basename "$(dirname "$(dirname "$manifest")")")
if ! "$python_path" - "$manifest" "$root/$stage/completed" <<'PY'
import json
import sys
from pathlib import Path
manifest = Path(sys.argv[1]); completed = Path(sys.argv[2])
scheduled = {json.loads(line)["key"] for line in manifest.read_text().splitlines() if line.strip()}
done = set()
for path in completed.glob("*.json"):
    try: row = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError): continue
    if row.get("status") == "done": done.add(row.get("job_key"))
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

campaign_done() {
  PYTHONPATH="$source_root/src" "$local_python" -m "$module" \
    --stage status --output-root "$root" | "$local_python" -c \
    'import json,sys; stage=sys.argv[1]; raise SystemExit(0 if json.load(sys.stdin)[stage]["done"] else 1)' "$stage"
}

[[ -d $source_root ]] || { echo "missing source snapshot: $source_root" >&2; exit 2; }
sync_sources
run_preflight
if [[ ! -f $root/contract.json ]]; then
  PYTHONPATH="$source_root/src" "$local_python" -m "$module" --stage enqueue --output-root "$root" | tee -a "$log"
fi
sync_campaign

while true; do
  collect_remote
  snapshot=$(PYTHONPATH="$source_root/src" "$local_python" -m "$module" --stage status --output-root "$root")
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$(tr '\n' ' ' <<<"$snapshot")" | tee -a "$log"
  campaign_done && break
  ensure_local_workers
  ensure_remote_worker local_gpu "$local_gpu_host" 5003 "$local_gpu_key" "$local_gpu_repo" "$local_gpu_python" 8 0 || true
  ensure_remote_worker local_gpu "$local_gpu_host" 5003 "$local_gpu_key" "$local_gpu_repo" "$local_gpu_python" 9 0 || true
  ensure_remote_worker local_gpu "$local_gpu_host" 5003 "$local_gpu_key" "$local_gpu_repo" "$local_gpu_python" 10 1 || true
  ensure_remote_worker local_gpu "$local_gpu_host" 5003 "$local_gpu_key" "$local_gpu_repo" "$local_gpu_python" 11 1 || true
  ensure_remote_worker kau "$kau_host" 8589 "$kau_key" "$kau_repo" "$kau_python" 12 0 || true
  ensure_remote_worker kau "$kau_host" 8589 "$kau_key" "$kau_repo" "$kau_python" 13 0 || true
  sleep "$poll_seconds"
done

PYTHONPATH="$source_root/src" "$local_python" -m "$module" --stage report --output-root "$root" | tee -a "$log"
printf '%s %s complete\n' "$(date --iso-8601=seconds)" "$campaign_label" | tee -a "$log"
