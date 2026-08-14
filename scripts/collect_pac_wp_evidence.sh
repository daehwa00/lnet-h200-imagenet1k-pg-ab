#!/usr/bin/env bash
set -euo pipefail

root=${1:-.}
remote_root=${PAC_B200_ROOT:-REMOTE_HOME_PLACEHOLDER/lnet-pac-external-20260710}
remote_host=${PAC_B200_HOST:-remote_user@REMOTE_HOST_PLACEHOLDER}
remote_python=${PAC_B200_PYTHON:-REMOTE_ROOT_PLACEHOLDER/miniconda3/envs/ASA_test/bin/python}
ssh_key=${PAC_B200_KEY:-LOCAL_HOME_PLACEHOLDER/.codex/attachments/OPAQUE_ID_PLACEHOLDER/SSH_KEY_PLACEHOLDER}
jump_key=${PAC_B200_JUMP_KEY:-LOCAL_HOME_PLACEHOLDER/.codex/attachments/OPAQUE_ID_PLACEHOLDER/SSH_KEY_PLACEHOLDER}
result=.omx/results/pac-wp-evidence-20260712

remote_ssh=(
  ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
  -i "$ssh_key" -p 40701
  -o "ProxyCommand=ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i $jump_key -p 8589 secondary_host@REMOTE_HOST_PLACEHOLDER -W %h:%p"
  "$remote_host"
)

status_command=$(cat <<'PY'
import csv
from pathlib import Path

root = Path(".omx/results/pac-wp-evidence-20260712")

def count(pattern: str, key: str) -> tuple[int, int]:
    latest = {}
    for path in root.glob(pattern):
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                latest[row.get(key)] = row
    done = sum(row.get("status") == "done" for row in latest.values())
    failed = sum(row.get("status") == "failed" for row in latest.values())
    return done, failed

training = count("evidence-shards/*/results/training.csv", "queue_key")
interpretability = count("evidence-shards/*/results/interpretability.csv", "queue_key")
sensitivity = count("evidence-shards/*/results/sensitivity.csv", "queue_key")
p1p2 = count("p1p2-training-shards/*/results/*.csv", "job_key")
efficiency = count("p1p2-efficiency-shards/*/results/*.csv", "job_key")
print(
    training[0] + interpretability[0] + sensitivity[0],
    p1p2[0],
    efficiency[0],
    training[1] + interpretability[1] + sensitivity[1] + p1p2[1] + efficiency[1],
)
PY
)

while true; do
  status=$("${remote_ssh[@]}" "cd '$remote_root' && '$remote_python' - <<'PY'
$status_command
PY")
  read -r evidence p1p2 efficiency failed <<<"$status"
  if [[ "$evidence" == 895 && "$p1p2" == 160 && "$efficiency" == 45 && "$failed" == 0 ]]; then
    break
  fi
  sleep 20
done

cd "$root"
"${remote_ssh[@]}" "cd '$remote_root' && tar -czf - '$result'" | tar -xzf -
touch "$result/REMOTE_COLLECTED"
