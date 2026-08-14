#!/usr/bin/env bash
set -uo pipefail

queue=$1
gpu=$2
python_bin=$3
output_root=$4
ucr_data_root=${5:-.omx/data/ucr}
external_data_root=${6:-data/external}
max_stagnant_passes=${MAX_STAGNANT_PASSES:-3}

export CUDA_VISIBLE_DEVICES=$gpu
export PYTHONPATH=src:.
export PYTHONSAFEPATH=1

mkdir -p "$output_root/logs"
lane=$(basename "$queue" .txt)
events="$output_root/logs/${lane}-supervisor.log"

completed_count() {
  find "$output_root/final/completed" -maxdepth 1 -type f -name '*.json' 2>/dev/null | wc -l
}

manifest_complete() {
  "$python_bin" - "$1" "$output_root" <<'PY'
import json
import pathlib
import sys

manifest = pathlib.Path(sys.argv[1])
root = pathlib.Path(sys.argv[2])
completed = {
    str(json.loads(path.read_text(encoding="utf-8"))["job_key"])
    for path in (root / "final" / "completed").glob("*.json")
}
expected = {
    str(json.loads(line)["job"]["key"])
    for line in manifest.read_text(encoding="utf-8").splitlines()
    if line
}
raise SystemExit(0 if expected <= completed else 1)
PY
}

trap 'printf "%s interrupted\n" "$(date --iso-8601=seconds)" >> "$events"; exit 143' INT TERM

stagnant=0
while true; do
  before=$(completed_count)
  remaining=0
  while IFS= read -r manifest; do
    [[ -n "$manifest" ]] || continue
    if manifest_complete "$manifest"; then
      continue
    fi
    printf '%s start %s\n' "$(date --iso-8601=seconds)" "$manifest" >> "$events"
    "$python_bin" -m lnet.pac_pointwise_identity_capacity_cli \
      --stage worker \
      --output-root "$output_root" \
      --manifest "$manifest" \
      --device cuda \
      --ucr-data-root "$ucr_data_root" \
      --external-data-root "$external_data_root" \
      >> "$output_root/logs/${lane}-worker.log" 2>&1
    exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
      printf '%s worker-exit=%s %s\n' \
        "$(date --iso-8601=seconds)" "$exit_code" "$manifest" >> "$events"
    fi
    if ! manifest_complete "$manifest"; then
      remaining=$((remaining + 1))
    fi
  done < "$queue"

  if [[ $remaining -eq 0 ]]; then
    printf '%s complete\n' "$(date --iso-8601=seconds)" >> "$events"
    exit 0
  fi
  after=$(completed_count)
  if [[ $after -gt $before ]]; then
    stagnant=0
  else
    stagnant=$((stagnant + 1))
  fi
  printf '%s pass remaining=%s progress=%s stagnant=%s\n' \
    "$(date --iso-8601=seconds)" "$remaining" "$((after - before))" "$stagnant" >> "$events"
  if [[ $stagnant -ge $max_stagnant_passes ]]; then
    printf '%s blocked after %s stagnant passes\n' \
      "$(date --iso-8601=seconds)" "$stagnant" >> "$events"
    exit 2
  fi
  sleep 20
done
