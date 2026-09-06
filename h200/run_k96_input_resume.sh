#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
test "$(git rev-parse HEAD)" = "${H200_EXPECTED_COMMIT:?exact immutable commit required}"
test -z "$(git status --porcelain --untracked-files=normal)"
CAMPAIGN=h200-k96-s521-input-resume-v1
MARKER="/dev/shm/lnet-owner-stop-${CAMPAIGN}-${H200_EXPECTED_COMMIT}.json"
if [[ "${H200_OWNER_CONTROL_INNER:-0}" != 1 ]]; then
  exec python3 scripts/run_h200_owner_controlled.py \
    --repo-root "$PWD" --repo-url https://github.com/daehwa00/lnet-h200-imagenet1k-pg-ab.git \
    --ref refs/heads/control/imagenet1k-k96-input-resume \
    --control-path h200/k96_input_resume/control.json \
    --campaign-id "$CAMPAIGN" --target-commit "$H200_EXPECTED_COMMIT" \
    --stop-marker "/app/output/daehwa00/run-control/${CAMPAIGN}/${H200_EXPECTED_COMMIT}/stopped.json" \
    --fast-stop-marker "$MARKER" --poll-seconds 15 --grace-seconds 2400 --term-seconds 30 \
    -- env H200_OWNER_CONTROL_INNER=1 LNET_EPOCH_STOP_MARKER="$MARKER" bash "$0"
fi
export CUDA_VISIBLE_DEVICES=0
export PYTHONDONTWRITEBYTECODE=1
# Standard-library discovery fails before installing anything or starting fresh.
RESUME_PYTHON="$(python3 scripts/resume_h200_k96_input.py --locate-python)"
exec "$RESUME_PYTHON" -u scripts/resume_h200_k96_input.py
