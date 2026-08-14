#!/usr/bin/env bash
set -euo pipefail

local_root=${1:-.}
remote_root=${PAC_B200_SUBMISSION_ROOT:-REMOTE_HOME_PLACEHOLDER/lnet-pac-submission-20260713}
remote_host=${PAC_B200_HOST:-remote_user@REMOTE_HOST_PLACEHOLDER}
remote_python=${PAC_B200_PYTHON:-REMOTE_HOME_PLACEHOLDER/.venvs/pac-mamba-torch291/bin/python}
ssh_key=${PAC_B200_KEY:-LOCAL_HOME_PLACEHOLDER/.codex/attachments/OPAQUE_ID_PLACEHOLDER/SSH_KEY_PLACEHOLDER}
jump_key=${PAC_B200_JUMP_KEY:-LOCAL_HOME_PLACEHOLDER/.codex/attachments/OPAQUE_ID_PLACEHOLDER/SSH_KEY_PLACEHOLDER}
result=.omx/results/pac-untouched-ucr12-confirmatory-pro6000-20260713
session=pac-untouched-ucr12-b200-gpu3-20260713

remote_ssh=(
  ssh -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null
  -i "$ssh_key" -p 40701
  -o "ProxyCommand=ssh -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i $jump_key -p 8589 secondary_host@REMOTE_HOST_PLACEHOLDER -W %h:%p"
  "$remote_host"
)

status_command=$(cat <<'PY'
import json
from pathlib import Path

root = Path(".omx/results/pac-untouched-ucr12-confirmatory-pro6000-20260713")
keys = {
    json.loads(line)["key"]
    for line in (root / "queue_manifest.jsonl").read_text(encoding="utf-8").splitlines()
    if line
}
latest = {}
state = root / "queue_state.jsonl"
if state.exists():
    for line in state.read_text(encoding="utf-8").splitlines():
        if line:
            row = json.loads(line)
            latest[row["key"]] = row["status"]
print(sum(latest.get(key) == "done" for key in keys), sum(latest.get(key) == "failed" for key in keys))
PY
)

while true; do
  read -r done failed < <("${remote_ssh[@]}" "cd '$remote_root' && '$remote_python' - <<'PY'
$status_command
PY")
  printf '%s done=%s/540 failed=%s\n' "$(date -Is)" "$done" "$failed"
  if [[ "$done" == 540 && "$failed" == 0 ]]; then
    break
  fi
  if ! "${remote_ssh[@]}" "tmux has-session -t '$session'" 2>/dev/null; then
    "${remote_ssh[@]}" "cd '$remote_root' && tmux new-session -d -s '$session' \
      'export CUDA_VISIBLE_DEVICES=0 CUDA_MPS_PIPE_DIRECTORY=/tmp/lnet-mps-gpu3-pipe \
      CUDA_MPS_LOG_DIRECTORY=/tmp/lnet-mps-gpu3-log PYTHONPATH=vendor:src \
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; \
      $remote_python -m lnet.pac_untouched_ucr_confirmatory --stage workers \
      --output-root $result --device cuda --workers 1 \
      >>.omx/logs/pac-untouched-ucr12-confirmatory-b200-gpu3-20260713/worker.log 2>&1'"
  fi
  sleep 60
done

cd "$local_root"
"${remote_ssh[@]}" "cd '$remote_root' && tar -czf - '$result'" | tar -xzf -
touch "$result/REMOTE_COLLECTED_B200_GPU3"
"${remote_ssh[@]}" \
  "CUDA_MPS_PIPE_DIRECTORY=/tmp/lnet-mps-gpu3-pipe nvidia-cuda-mps-control <<<quit" \
  >/dev/null 2>&1 || true
