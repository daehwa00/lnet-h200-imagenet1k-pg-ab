#!/usr/bin/env bash
set -euo pipefail

project_root=${1:-.}
root=.omx/results/pac-resolvent-reader-ucr16-fast-20260720
key=LOCAL_HOME_PLACEHOLDER/.codex/attachments/OPAQUE_ID_PLACEHOLDER/SSH_KEY_PLACEHOLDER
host=secondary_host@REMOTE_HOST_PLACEHOLDER
port=8589
remote_repo=REMOTE_HOME_PLACEHOLDER/lnet-terminal-20260718
python_bin=REMOTE_HOME_PLACEHOLDER/anaconda3/envs/torch/bin/python
local_python=LOCAL_HOME_PLACEHOLDER/miniconda3/bin/python
workers=6
skip_source_sync=${PAC_RESOLVENT_SKIP_SOURCE_SYNC:-0}
log=.omx/logs/pac-resolvent-reader-ucr16-fast-kau-20260720.log

cd "$project_root"

sync_sources() {
  ssh -i "$key" -p "$port" -o BatchMode=yes "$host" \
    "mkdir -p '$remote_repo/optimization' '$remote_repo/scripts' '$remote_repo/$root/manifests'"
  rsync -az -e "ssh -i $key -p $port -o BatchMode=yes" \
    optimization/learned_two_tap_alphabet.py \
    optimization/learned_two_tap_resolvent_reader.py \
    optimization/masked_modal_moments.py \
    "$host:$remote_repo/optimization/"
  rsync -az -e "ssh -i $key -p $port -o BatchMode=yes" \
    scripts/pac_pole_attention_ucr16_fast.py \
    scripts/pac_resolvent_reader_ucr16_fast.py \
    "$host:$remote_repo/scripts/"
  rsync -az -e "ssh -i $key -p $port -o BatchMode=yes" \
    "$root/manifests/" "$host:$remote_repo/$root/manifests/"
  rsync -az -e "ssh -i $key -p $port -o BatchMode=yes" \
    "$root/contract.json" "$host:$remote_repo/$root/contract.json"
}

sync_results() {
  mkdir -p "$root/completed" "$root/failed" "$root/reports"
  for bucket in completed failed reports; do
    rsync -az -e "ssh -i $key -p $port -o BatchMode=yes" \
      "$host:$remote_repo/$root/$bucket/" "$root/$bucket/" 2>/dev/null || true
  done
  local completed
  for completed in "$root/completed/"*.json; do
    [[ -f $completed ]] || continue
    rm -f "$root/failed/$(basename "$completed")"
  done
}

ensure_workers() {
  ssh -i "$key" -p "$port" -o BatchMode=yes "$host" bash -s -- \
    "$remote_repo" "$root" "$python_bin" "$workers" <<'REMOTE'
set -euo pipefail
repo=$1
root=$2
python_bin=$3
workers=$4
cd "$repo"
for index in $(seq 0 $((workers - 1))); do
  manifest=$(printf '%s/manifests/worker-%02d.jsonl' "$root" "$index")
  session=$(printf 'pac-resolvent-reader-ucr16-fast-%02d-20260720' "$index")
  if tmux has-session -t "$session" 2>/dev/null; then
    continue
  fi
  if ! "$python_bin" - "$manifest" "$root/completed" <<'PY'
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
    continue
  fi
  logfile=$(printf '%s/logs/worker-%02d.log' "$root" "$index")
  mkdir -p "$(dirname "$logfile")"
  tmux new-session -d -s "$session" \
    "cd '$repo' && for retry in 1 2 3; do OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src:. CUDA_VISIBLE_DEVICES=0 '$python_bin' scripts/pac_resolvent_reader_ucr16_fast.py worker --root '$root' --manifest '$manifest' --device cuda >>'$logfile' 2>&1; done"
done
REMOTE
}

if [[ $skip_source_sync != 1 ]]; then
  sync_sources
fi
while true; do
  sync_results
  status=$(PYTHONPATH=src:. "$local_python" scripts/pac_resolvent_reader_ucr16_fast.py \
    status --root "$root")
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$(tr '\n' ' ' <<<"$status")" | tee -a "$log"
  if "$local_python" -c \
    'import json,sys; raise SystemExit(0 if json.load(sys.stdin)["done"] else 1)' \
    <<<"$status"
  then
    break
  fi
  ensure_workers
  sleep 20
done

PYTHONPATH=src:. "$local_python" scripts/pac_resolvent_reader_ucr16_fast.py \
  report --root "$root" | tee -a "$log"
printf '%s resolvent-reader UCR-16 screen complete\n' \
  "$(date --iso-8601=seconds)" | tee -a "$log"
