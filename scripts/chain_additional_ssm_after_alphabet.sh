#!/usr/bin/env bash
set -euo pipefail

project_root=${PROJECT_ROOT:-.}
alphabet_root=${ALPHABET_ROOT:-.omx/results/pac-pointwise-identity-capacity-q1-final-20260722}
poll_seconds=${ALPHABET_CHAIN_POLL_SECONDS:-60}
python_bin=${LOCAL_PYTHON:-LOCAL_HOME_PLACEHOLDER/miniconda3/bin/python}
additional_ssm_supervisor=${ADDITIONAL_SSM_SUPERVISOR:-$project_root/scripts/supervise_additional_ssm_q1_search.sh}
log=$project_root/.omx/logs/additional-ssm-after-alphabet.log

cd "$project_root"
mkdir -p "$(dirname "$log")"
while true; do
  if PYTHONPATH="$project_root/src" "$python_bin" - "$alphabet_root" <<'PY'
import sys
from pathlib import Path
from lnet.pac_pointwise_identity_capacity_campaign import status
raise SystemExit(0 if status(Path(sys.argv[1]))["final"]["done"] else 1)
PY
  then
    break
  fi
  printf '%s waiting for ALPHABET Q1-final completion\n' "$(date --iso-8601=seconds)" >> "$log"
  sleep "$poll_seconds"
done

printf '%s ALPHABET complete; starting S5/LRU/DSS Stage-1/2\n' "$(date --iso-8601=seconds)" | tee -a "$log"
exec "$additional_ssm_supervisor" "$project_root"
