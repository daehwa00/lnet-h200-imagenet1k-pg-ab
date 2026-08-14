#!/usr/bin/env bash
set -euo pipefail

project_root=${1:-.}
root=.omx/results/pac-two-tap-q1-final-20260720
baseline_root=.omx/results/pac-baseline-fairness-maximal-20260714
poll_seconds=${PAC_TWO_TAP_Q1_POLL_SECONDS:-30}
disable_kau=${PAC_TWO_TAP_DISABLE_KAU:-0}
local_python=LOCAL_HOME_PLACEHOLDER/miniconda3/bin/python
local_gpu_python=LOCAL_HOME_PLACEHOLDER/miniconda3/envs/brelu/bin/python
kau_python=REMOTE_HOME_PLACEHOLDER/anaconda3/envs/torch/bin/python
local_gpu_key=LOCAL_HOME_PLACEHOLDER/.codex/attachments/OPAQUE_ID_PLACEHOLDER/SSH_KEY_PLACEHOLDER
kau_key=LOCAL_HOME_PLACEHOLDER/.codex/attachments/OPAQUE_ID_PLACEHOLDER/SSH_KEY_PLACEHOLDER
local_gpu_host=local_gpu@REMOTE_HOST_PLACEHOLDER
kau_host=secondary_host@REMOTE_HOST_PLACEHOLDER
local_gpu_repo=LOCAL_HOME_PLACEHOLDER/lnet-external-20260718
kau_repo=REMOTE_HOME_PLACEHOLDER/lnet-terminal-20260718
log=.omx/logs/pac-two-tap-q1-final-20260720-supervisor.log

cd "$project_root"

manifest_has_pending() {
  local python_path=$1
  local manifest=$2
  local completed_dir=$3
  "$python_path" - "$manifest" "$completed_dir" <<'PY'
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
        completed.add(row.get("job_key"))
raise SystemExit(0 if scheduled - completed else 1)
PY
}

collect_stage() {
  local stage=$1
  mkdir -p "$root/$stage/completed" "$root/$stage/failed" "$root/$stage/attempts"
  for bucket in completed failed attempts; do
    rsync -az -e "ssh -i $local_gpu_key -p 5003 -o BatchMode=yes" \
      "$local_gpu_host:$local_gpu_repo/$root/$stage/$bucket/" "$root/$stage/$bucket/" \
      2>/dev/null || true
    rsync -az -e "ssh -i $kau_key -p 8589 -o BatchMode=yes" \
      "$kau_host:$kau_repo/$root/$stage/$bucket/" "$root/$stage/$bucket/" \
      2>/dev/null || true
  done
  # Provenance is campaign-scoped rather than stage-scoped.  Collect it from
  # every execution host so each row's provenance_sha256 remains resolvable in
  # the authoritative local campaign bundle.
  mkdir -p "$root/provenance"
  rsync -az -e "ssh -i $local_gpu_key -p 5003 -o BatchMode=yes" \
    "$local_gpu_host:$local_gpu_repo/$root/provenance/" "$root/provenance/" \
    2>/dev/null || true
  rsync -az -e "ssh -i $kau_key -p 8589 -o BatchMode=yes" \
    "$kau_host:$kau_repo/$root/provenance/" "$root/provenance/" \
    2>/dev/null || true
  local completed
  for completed in "$root/$stage/completed/"*.json; do
    [[ -f $completed ]] || continue
    rm -f "$root/$stage/failed/$(basename "$completed")"
  done
}

sync_stage_to_remotes() {
  local stage=$1
  # A stage worker records the preceding selection artifact hash.  Send that
  # artifact explicitly because syncing only the active stage directory leaves
  # distributed rows with a null provenance link.
  if [[ $stage == stage2 ]]; then
    rsync -az -e "ssh -i $local_gpu_key -p 5003 -o BatchMode=yes" \
      "$root/stage1/selection.json" "$local_gpu_host:$local_gpu_repo/$root/stage1/selection.json"
    rsync -az -e "ssh -i $kau_key -p 8589 -o BatchMode=yes" \
      "$root/stage1/selection.json" "$kau_host:$kau_repo/$root/stage1/selection.json"
  elif [[ $stage == final ]]; then
    rsync -az -e "ssh -i $local_gpu_key -p 5003 -o BatchMode=yes" \
      "$root/stage2/selection.json" "$local_gpu_host:$local_gpu_repo/$root/stage2/selection.json"
    rsync -az -e "ssh -i $kau_key -p 8589 -o BatchMode=yes" \
      "$root/stage2/selection.json" "$kau_host:$kau_repo/$root/stage2/selection.json"
  fi
  rsync -az -e "ssh -i $local_gpu_key -p 5003 -o BatchMode=yes" \
    "$root/$stage/" "$local_gpu_host:$local_gpu_repo/$root/$stage/"
  rsync -az -e "ssh -i $kau_key -p 8589 -o BatchMode=yes" \
    "$root/$stage/" "$kau_host:$kau_repo/$root/$stage/"
  rsync -az -e "ssh -i $local_gpu_key -p 5003 -o BatchMode=yes" \
    "$root/reports/" "$local_gpu_host:$local_gpu_repo/$root/reports/"
  rsync -az -e "ssh -i $kau_key -p 8589 -o BatchMode=yes" \
    "$root/reports/" "$kau_host:$kau_repo/$root/reports/"
}

ensure_local_workers() {
  local stage=$1
  local index manifest session logfile
  # Manifests 0--2 carry the three longest permuted-MNIST jobs.  Keep 0 local
  # and place 1/2 on local_gpu's two GPUs; compensate with 8/9 locally.
  for index in 0 3 4 5 6 7 8 9; do
    manifest=$(printf '%s/%s/manifests/worker-%02d.jsonl' "$root" "$stage" "$index")
    session=$(printf 'pac-two-tap-q1-%s-local-%02d' "$stage" "$index")
    logfile=$(printf '%s/%s/logs/local-%02d.log' "$root" "$stage" "$index")
    [[ -f $manifest ]] || continue
    if tmux has-session -t "$session" 2>/dev/null; then
      continue
    fi
    if ! manifest_has_pending "$local_python" "$manifest" "$root/$stage/completed"; then
      continue
    fi
    mkdir -p "$(dirname "$logfile")"
    tmux new-session -d -s "$session" \
      "cd '$project_root' && for retry in 1 2 3; do OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 '$local_python' -m lnet.pac_two_tap_q1_cli --stage worker --output-root '$root' --manifest '$manifest' --device cuda >>'$logfile' 2>&1; done"
  done
}

ensure_remote_worker() {
  local host=$1 port=$2 key=$3 repo=$4 python_path=$5 source_path=$6
  local stage=$7 index=$8 gpu=$9 label=${10}
  local manifest session logfile
  manifest=$(printf '%s/%s/manifests/worker-%02d.jsonl' "$root" "$stage" "$index")
  session=$(printf 'pac-two-tap-q1-%s-%s-%02d' "$stage" "$label" "$index")
  logfile=$(printf '%s/%s/logs/%s-%02d.log' "$root" "$stage" "$label" "$index")
  ssh -i "$key" -p "$port" -o BatchMode=yes -o ConnectTimeout=10 "$host" \
    bash -s -- "$repo" "$python_path" "$source_path" "$manifest" "$stage" \
    "$session" "$logfile" "$gpu" "$root" <<'REMOTE'
set -euo pipefail
repo=$1
python_path=$2
source_path=$3
manifest=$4
stage=$5
session=$6
logfile=$7
gpu=$8
root=$9
cd "$repo"
[[ -f $manifest ]] || exit 0
if tmux has-session -t "$session" 2>/dev/null; then
  exit 0
fi
if ! "$python_path" - "$manifest" "$root/$stage/completed" <<'PY'
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
completed = {
    json.loads(path.read_text()).get("job_key")
    for path in completed_dir.glob("*.json")
}
raise SystemExit(0 if scheduled - completed else 1)
PY
then
  exit 0
fi
mkdir -p "$(dirname "$logfile")"
tmux new-session -d -s "$session" \
  "cd '$repo' && for retry in 1 2 3; do OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH='$source_path' CUDA_VISIBLE_DEVICES='$gpu' '$python_path' -m lnet.pac_two_tap_q1_cli --stage worker --output-root '$root' --manifest '$manifest' --device cuda >>'$logfile' 2>&1; done"
REMOTE
}

ensure_all_workers() {
  local stage=$1 index gpu
  ensure_local_workers "$stage"
  for index in 1 2 $(seq 10 19); do
    gpu=$((index % 2))
    ensure_remote_worker "$local_gpu_host" 5003 "$local_gpu_key" "$local_gpu_repo" \
      "$local_gpu_python" . "$stage" "$index" "$gpu" local_gpu || true
  done
  if [[ $disable_kau != 1 ]]; then
    for index in $(seq 20 23); do
      ensure_remote_worker "$kau_host" 8589 "$kau_key" "$kau_repo" \
        "$kau_python" src "$stage" "$index" 0 kau || true
    done
  fi
}

stage_done() {
  local stage=$1
  PYTHONPATH=src "$local_python" - "$root" "$stage" <<'PY'
import sys
from pathlib import Path
from lnet.pac_two_tap_q1_campaign import status

payload = status(Path(sys.argv[1]))[sys.argv[2]]
raise SystemExit(0 if payload["done"] else 1)
PY
}

run_stage() {
  local stage=$1
  sync_stage_to_remotes "$stage"
  while true; do
    collect_stage "$stage"
    local snapshot
    snapshot=$(PYTHONPATH=src "$local_python" -m lnet.pac_two_tap_q1_cli \
      --stage status --output-root "$root")
    printf '%s stage=%s %s\n' "$(date --iso-8601=seconds)" "$stage" \
      "$(tr '\n' ' ' <<<"$snapshot")" | tee -a "$log"
    if stage_done "$stage"; then
      break
    fi
    ensure_all_workers "$stage"
    sleep "$poll_seconds"
  done
}

run_stage stage1
PYTHONPATH=src "$local_python" -m lnet.pac_two_tap_q1_cli \
  --stage select-stage1 --output-root "$root" --lanes 24 | tee -a "$log"
run_stage stage2
PYTHONPATH=src "$local_python" -m lnet.pac_two_tap_q1_cli \
  --stage select-stage2 --output-root "$root" | tee -a "$log"
PYTHONPATH=src "$local_python" -m lnet.pac_two_tap_q1_cli \
  --stage enqueue-final --output-root "$root" --lanes 24 | tee -a "$log"
# The final ledger reuses 900 sealed baseline rows.  Preserve the provenance
# records referenced by those rows inside the new self-contained Q1 bundle.
mkdir -p "$root/provenance"
rsync -a "$baseline_root/provenance/" "$root/provenance/"
run_stage final
printf '%s learned two-tap Q1/Q1-final execution complete\n' \
  "$(date --iso-8601=seconds)" | tee -a "$log"
