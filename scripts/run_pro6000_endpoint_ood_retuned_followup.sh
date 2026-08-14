#!/usr/bin/env bash
set -euo pipefail

root=${1:-.}
cd "$root"

campaign=.omx/results/pac-endpoint-ood-retuned-pro6000-20260713
log_root=.omx/logs/pac-endpoint-ood-retuned-pro6000-20260713
python_bin=${PAC_ENDPOINT_RETUNE_PYTHON:-python}
mkdir -p "$log_root"

official_complete() {
  PYTHONPATH="$root/src" "$python_bin" - <<'PY'
import json
from pathlib import Path

def done(root: Path, manifests: str, states: str, expected: int) -> bool:
    jobs = {
        json.loads(line)["key"]
        for path in root.glob(manifests)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    latest = {}
    for path in root.glob(states):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                latest[row["key"]] = row["status"]
    return len(jobs) == expected and all(latest.get(key) == "done" for key in jobs)

baseline = done(
    Path(".omx/results/pac-revised-ucr-official-test-20260712"),
    "shards/*/queue_manifest.jsonl",
    "shards/*/queue_state.jsonl",
    720,
)
alphabet = done(
    Path(".omx/results/pac-pa2wp-official-ucr-test-pro6000-20260713"),
    "queue_manifest.jsonl",
    "queue_state.jsonl",
    90,
)
raise SystemExit(0 if baseline and alphabet else 1)
PY
}

while ! official_complete; do
  sleep 60
done

run_stage() {
  PYTHONPATH="$root/src" "$python_bin" -m lnet.pac_endpoint_ood_retuned \
    --output-root "$campaign" "$@"
}

run_stage --stage enqueue-tuning >"$log_root/enqueue-tuning.log" 2>&1
run_stage --stage tune-workers --device cuda --workers 4 >"$log_root/tuning-pass1.log" 2>&1
run_stage --stage tune-workers --device cuda --workers 4 >"$log_root/tuning-pass2.log" 2>&1
run_stage --stage select >"$log_root/select.log" 2>&1
run_stage --stage enqueue-final >"$log_root/enqueue-final.log" 2>&1
run_stage --stage final-workers --device cuda --workers 4 >"$log_root/final-pass1.log" 2>&1
run_stage --stage final-workers --device cuda --workers 4 >"$log_root/final-pass2.log" 2>&1
run_stage --stage report >"$log_root/report.log" 2>&1
PYTHONPATH="$root/src" "$python_bin" -m lnet.pac_boundary_bootstrap \
  >"$log_root/bootstrap.log" 2>&1
