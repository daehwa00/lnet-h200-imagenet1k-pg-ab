#!/usr/bin/env bash
set -euo pipefail

project_root=${1:-.}
root=.omx/results/pac-two-tap-ucr16-fast-20260720
key=LOCAL_HOME_PLACEHOLDER/.codex/attachments/OPAQUE_ID_PLACEHOLDER/SSH_KEY_PLACEHOLDER
host=local_gpu@REMOTE_HOST_PLACEHOLDER
port=5003
remote_repo=LOCAL_HOME_PLACEHOLDER/lnet-external-20260718
python_bin=LOCAL_HOME_PLACEHOLDER/miniconda3/envs/brelu/bin/python
log=.omx/logs/pac-two-tap-ucr16-fast-20260720-supervisor.log

cd "$project_root"

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

ensure_remote_workers() {
  ssh -i "$key" -p "$port" -o BatchMode=yes "$host" bash -s -- \
    "$remote_repo" "$root" "$python_bin" <<'REMOTE'
set -euo pipefail
repo=$1
root=$2
python_bin=$3
cd "$repo"
for index in $(seq 0 11); do
  manifest=$(printf '%s/manifests/worker-%02d.jsonl' "$root" "$index")
  session=$(printf 'pac-two-tap-ucr16-fast-%02d-20260720' "$index")
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
  gpu=$((index % 2))
  logfile=$(printf '%s/logs/worker-%02d.log' "$root" "$index")
  mkdir -p "$(dirname "$logfile")"
  tmux new-session -d -s "$session" \
    "cd '$repo' && for retry in 1 2 3; do OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES='$gpu' '$python_bin' scripts/pac_two_tap_ucr16_fast.py worker --root '$root' --manifest '$manifest' --device cuda >>'$logfile' 2>&1; done"
done
REMOTE
}

while true; do
  sync_results
  status=$(PYTHONPATH=src python scripts/pac_two_tap_ucr16_fast.py status --root "$root")
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$(tr '\n' ' ' <<<"$status")" | tee -a "$log"
  if python -c 'import json,sys; raise SystemExit(0 if json.load(sys.stdin)["done"] else 1)' \
    <<<"$status"
  then
    break
  fi
  ensure_remote_workers
  sleep 30
done

PYTHONPATH=src python scripts/pac_two_tap_ucr16_fast.py report --root "$root" \
  | tee -a "$log"

# Resume the immutable Q2 workers through their existing restart-safe watchdog.
if ! tmux has-session -t pac-alphabet-q2-final-worker-watchdog-20260720 2>/dev/null; then
  tmux new-session -d -s pac-alphabet-q2-final-worker-watchdog-20260720 \
    "cd '$project_root' && bash scripts/watch_pac_q2_final_workers.sh >>.omx/logs/q2-final-worker-watchdog-20260720.log 2>&1"
fi
printf '%s two-tap screen complete; Q2 watchdog resumed\n' "$(date --iso-8601=seconds)" \
  | tee -a "$log"
