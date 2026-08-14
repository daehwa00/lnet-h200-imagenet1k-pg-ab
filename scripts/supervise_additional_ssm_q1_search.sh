#!/usr/bin/env bash
set -euo pipefail

project_root=${PROJECT_ROOT:-.}
root=${ADDITIONAL_SSM_ROOT:-.omx/results/pac-additional-ssm-q1-search-20260722}
module=lnet.pac_additional_ssm_q1_cli
campaign=lnet.pac_additional_ssm_q1_campaign
lanes=14
poll_seconds=${ADDITIONAL_SSM_POLL_SECONDS:-30}
local_python=${LOCAL_PYTHON:-LOCAL_HOME_PLACEHOLDER/miniconda3/bin/python}
local_gpu_python=${LOCAL_GPU_PYTHON:-LOCAL_HOME_PLACEHOLDER/miniconda3/envs/flow/bin/python}
kau_python=${REMOTE_PYTHON:-REMOTE_HOME_PLACEHOLDER/anaconda3/envs/vertex/bin/python}
local_gpu_key=${LOCAL_GPU_SSH_KEY:-LOCAL_HOME_PLACEHOLDER/.ssh/SSH_KEY_PLACEHOLDER}
kau_key=LOCAL_HOME_PLACEHOLDER/.codex/attachments/OPAQUE_ID_PLACEHOLDER/SSH_KEY_PLACEHOLDER
local_gpu_host=local_gpu@REMOTE_HOST_PLACEHOLDER
kau_host=secondary_host@REMOTE_HOST_PLACEHOLDER
local_gpu_repo=${LOCAL_GPU_REPO:-LOCAL_HOME_PLACEHOLDER/lnet-pointwise-q1-final-20260722}
kau_repo=${REMOTE_REPO:-REMOTE_HOME_PLACEHOLDER/lnet-pointwise-q1-final-20260722}
log=.omx/logs/pac-additional-ssm-q1-search-20260722-supervisor.log

cd "$project_root"
mkdir -p "$(dirname "$log")"

sync_sources() {
  local files=(
    src/lnet/pac_additional_ssm_baselines.py
    src/lnet/pac_additional_ssm_q1_campaign.py
    src/lnet/pac_additional_ssm_q1_cli.py
    src/lnet/pac_baseline_fairness_maximal.py
    src/lnet/pac_confirmatory_baselines.py
  )
  local file
  for file in "${files[@]}"; do
    rsync -az -e "ssh -i $local_gpu_key -p 5003 -o BatchMode=yes" \
      "$file" "$local_gpu_host:$local_gpu_repo/$file"
    rsync -az -e "ssh -i $kau_key -p 8589 -o BatchMode=yes" \
      "$file" "$kau_host:$kau_repo/$file"
  done
}

manifest_has_pending() {
  local python_path=$1 manifest=$2 completed_dir=$3
  "$python_path" - "$manifest" "$completed_dir" <<'PY'
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
completed_dir = Path(sys.argv[2])
expected = {json.loads(line)["key"] for line in manifest.read_text().splitlines() if line}
completed = {
    json.loads(path.read_text()).get("job_key")
    for path in completed_dir.glob("*.json")
    if path.is_file()
}
raise SystemExit(0 if expected - completed else 1)
PY
}

collect_stage() {
  local stage=$1 bucket
  mkdir -p "$root/$stage/completed" "$root/$stage/failed" "$root/$stage/attempts" "$root/provenance"
  for bucket in completed failed attempts; do
    rsync -az -e "ssh -i $local_gpu_key -p 5003 -o BatchMode=yes" \
      "$local_gpu_host:$local_gpu_repo/$root/$stage/$bucket/" "$root/$stage/$bucket/" 2>/dev/null || true
    rsync -az -e "ssh -i $kau_key -p 8589 -o BatchMode=yes" \
      "$kau_host:$kau_repo/$root/$stage/$bucket/" "$root/$stage/$bucket/" 2>/dev/null || true
  done
}

sync_stage() {
  local stage=$1
  for remote in local_gpu kau; do
    if [[ $remote == local_gpu ]]; then
      ssh -i "$local_gpu_key" -p 5003 -o BatchMode=yes "$local_gpu_host" \
        "mkdir -p '$local_gpu_repo/$root/$stage' '$local_gpu_repo/$root/reports'"
      [[ $stage != stage2 ]] || rsync -az -e "ssh -i $local_gpu_key -p 5003 -o BatchMode=yes" \
        "$root/stage1/selection.json" "$local_gpu_host:$local_gpu_repo/$root/stage1/selection.json"
      rsync -az -e "ssh -i $local_gpu_key -p 5003 -o BatchMode=yes" \
        "$root/$stage/" "$local_gpu_host:$local_gpu_repo/$root/$stage/"
      rsync -az -e "ssh -i $local_gpu_key -p 5003 -o BatchMode=yes" \
        "$root/reports/" "$local_gpu_host:$local_gpu_repo/$root/reports/"
    else
      ssh -i "$kau_key" -p 8589 -o BatchMode=yes "$kau_host" \
        "mkdir -p '$kau_repo/$root/$stage' '$kau_repo/$root/reports'"
      [[ $stage != stage2 ]] || rsync -az -e "ssh -i $kau_key -p 8589 -o BatchMode=yes" \
        "$root/stage1/selection.json" "$kau_host:$kau_repo/$root/stage1/selection.json"
      rsync -az -e "ssh -i $kau_key -p 8589 -o BatchMode=yes" \
        "$root/$stage/" "$kau_host:$kau_repo/$root/$stage/"
      rsync -az -e "ssh -i $kau_key -p 8589 -o BatchMode=yes" \
        "$root/reports/" "$kau_host:$kau_repo/$root/reports/"
    fi
  done
}

ensure_local_worker() {
  local stage=$1 index=$2 manifest session logfile
  manifest=$(printf '%s/%s/manifests/worker-%02d.jsonl' "$root" "$stage" "$index")
  session=$(printf 'additional-ssm-%s-pro-%02d' "$stage" "$index")
  logfile=$(printf '%s/%s/logs/pro-%02d.log' "$root" "$stage" "$index")
  [[ -f $manifest ]] || return 0
  tmux has-session -t "$session" 2>/dev/null && return 0
  manifest_has_pending "$local_python" "$manifest" "$root/$stage/completed" || return 0
  mkdir -p "$(dirname "$logfile")"
  tmux new-session -d -s "$session" \
    "cd '$project_root' && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONSAFEPATH=1 PYTHONPATH='$project_root/src' CUDA_VISIBLE_DEVICES=0 '$local_python' -m '$module' --stage worker --output-root '$root' --manifest '$manifest' --device cuda >>'$logfile' 2>&1"
}

ensure_remote_worker() {
  local host=$1 port=$2 key=$3 repo=$4 python_path=$5 stage=$6 index=$7 gpu=$8 label=$9
  local manifest session logfile
  manifest=$(printf '%s/%s/manifests/worker-%02d.jsonl' "$root" "$stage" "$index")
  session=$(printf 'additional-ssm-%s-%s-%02d' "$stage" "$label" "$index")
  logfile=$(printf '%s/%s/logs/%s-%02d.log' "$root" "$stage" "$label" "$index")
  ssh -i "$key" -p "$port" -o BatchMode=yes -o ConnectTimeout=10 "$host" bash -s -- \
    "$repo" "$python_path" "$manifest" "$session" "$logfile" "$gpu" "$root" "$module" <<'REMOTE'
set -euo pipefail
repo=$1; python_path=$2; manifest=$3; session=$4; logfile=$5; gpu=$6; root=$7; module=$8
cd "$repo"
[[ -f $manifest ]] || exit 0
tmux has-session -t "$session" 2>/dev/null && exit 0
if ! "$python_path" - "$manifest" "$root" <<'PY'
import json
import sys
from pathlib import Path
manifest = Path(sys.argv[1]); root = Path(sys.argv[2]); stage = manifest.parts[-3]
expected = {json.loads(line)["key"] for line in manifest.read_text().splitlines() if line}
done = {json.loads(path.read_text()).get("job_key") for path in (root / stage / "completed").glob("*.json")}
raise SystemExit(0 if expected - done else 1)
PY
then
  exit 0
fi
mkdir -p "$(dirname "$logfile")"
tmux new-session -d -s "$session" \
  "cd '$repo' && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONSAFEPATH=1 PYTHONPATH='$repo/src' CUDA_VISIBLE_DEVICES='$gpu' '$python_path' -m '$module' --stage worker --output-root '$root' --manifest '$manifest' --device cuda >>'$logfile' 2>&1"
REMOTE
}

stage_done() {
  local stage=$1
  PYTHONPATH="$project_root/src" "$local_python" - "$root" "$stage" "$campaign" <<'PY'
import importlib
import sys
from pathlib import Path
campaign = importlib.import_module(sys.argv[3])
raise SystemExit(0 if campaign.status(Path(sys.argv[1]))[sys.argv[2]]["done"] else 1)
PY
}

run_stage() {
  local stage=$1 index snapshot
  sync_stage "$stage"
  while true; do
    collect_stage "$stage"
    snapshot=$(PYTHONPATH="$project_root/src" "$local_python" -m "$module" --stage status --output-root "$root")
    printf '%s stage=%s %s\n' "$(date --iso-8601=seconds)" "$stage" "$(tr '\n' ' ' <<<"$snapshot")" | tee -a "$log"
    stage_done "$stage" && break
    for index in $(seq 0 7); do ensure_local_worker "$stage" "$index"; done
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
  PYTHONPATH="$project_root/src" "$local_python" -m "$module" --stage enqueue-stage1 --output-root "$root" --lanes "$lanes" | tee -a "$log"
fi
run_stage stage1
PYTHONPATH="$project_root/src" "$local_python" -m "$module" --stage select-stage1 --output-root "$root" --lanes "$lanes" | tee -a "$log"
run_stage stage2
PYTHONPATH="$project_root/src" "$local_python" -m "$module" --stage select-stage2 --output-root "$root" | tee -a "$log"
printf '%s S5/LRU/DSS Q1 Stage-1/2 search complete\n' "$(date --iso-8601=seconds)" | tee -a "$log"
