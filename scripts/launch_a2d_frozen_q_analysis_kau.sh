#!/usr/bin/env bash
set -u

if [[ $# -ne 5 ]]; then
  echo "usage: $0 REPO CACHE_ROOT LOWRANK_ROOT OUTPUT_ROOT PYTHON" >&2
  exit 2
fi

repo=$1
cache_root=$2
lowrank_root=$3
output_root=$4
python_bin=$5
mkdir -p "$output_root/logs"

export PYTHONPATH="$repo/src:$repo/scripts"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export WANDB_PROJECT="${WANDB_PROJECT:-alphabet2d-imagenet100}"
export WANDB_ENTITY="${WANDB_ENTITY:-daehwa}"
export WANDB_GROUP="A2D-QAnalysis"

while true; do
  "$python_bin" -u "$repo/scripts/run_a2d_frozen_q_analysis.py" \
    --cache-root "$cache_root" \
    --lowrank-root "$lowrank_root" \
    --output-root "$output_root" \
    --head-epochs 30 \
    --head-batch-size 4096 \
    --seeds 501 509 521 \
    --device cuda:0 \
    >>"$output_root/logs/worker0.log" 2>&1
  status=$?
  if "$python_bin" - "$output_root/summary.json" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
if not path.exists():
    raise SystemExit(1)
payload = json.loads(path.read_text())
complete = payload.get("completed_seeds") == payload.get("expected_seeds") == 3
raise SystemExit(0 if complete and payload.get("failures") == 0 else 1)
PY
  then
    date -Is >"$output_root/COMPLETE"
    exit 0
  fi
  printf '{"event":"supervisor_restart","status":%d,"time":"%s"}\n' \
    "$status" "$(date -Is)" >>"$output_root/logs/supervisor.log"
  sleep 30
done
