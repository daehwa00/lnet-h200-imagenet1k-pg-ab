#!/usr/bin/env bash
set -euo pipefail

project_root=${1:?project root is required}
child_a=${2:?first reserved child pid is required}
child_b=${3:?second reserved child pid is required}
[[ $# -ge 4 ]] || {
  echo "at least one stopped parent pid is required" >&2
  exit 2
}
reserved_parents=("${@:4}")

python_bin=${PAC_UNIFIED_PYTHON:-LOCAL_HOME_PLACEHOLDER/miniconda3/bin/python}
campaign=.omx/results/pac-unified-global-20260712
log_root="$project_root/.omx/logs/pac-unified-global-20260712"
mkdir -p "$log_root"
cd "$project_root"

resume_reserved_workers() {
  kill -CONT "${reserved_parents[@]}" 2>/dev/null || true
}
trap resume_reserved_workers EXIT

pid_is_running() {
  local state
  state=$(ps -o stat= -p "$1" 2>/dev/null | tr -d ' ')
  [[ -n "$state" && "$state" != Z* ]]
}

if [[ ${PAC_UNIFIED_WAIT_FOR_RESERVED_CHILDREN:-1} == 1 ]]; then
  while pid_is_running "$child_a" || pid_is_running "$child_b"; do
    printf '%s waiting child_a=%s child_b=%s\n' \
      "$(date -Is)" \
      "$(pid_is_running "$child_a" && echo running || echo done)" \
      "$(pid_is_running "$child_b" && echo running || echo done)" \
      >"$log_root/reservation.status"
    sleep 30
  done
else
  printf '%s starting immediately alongside child_a=%s child_b=%s\n' \
    "$(date -Is)" "$child_a" "$child_b" >"$log_root/reservation.status"
fi

env CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$project_root/src" \
  "$python_bin" -m lnet.pac_unified_campaign_cli \
  --stage smoke --device cuda --output-root "$campaign" \
  >"$log_root/smoke.log" 2>&1

PYTHONPATH="$project_root/src" "$python_bin" -m lnet.pac_unified_campaign_cli \
  --stage enqueue --workers 2 --output-root "$campaign" \
  >"$log_root/enqueue.log" 2>&1

run_phase() {
  local phase=$1
  local pass=$2
  local pids=()
  for worker in 0 1; do
    env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      PYTHONPATH="$project_root/src" \
      "$python_bin" -m lnet.pac_unified_campaign_cli \
      --stage worker --device cuda --output-root "$campaign" \
      --manifest "$campaign/manifests/$phase-worker-$worker.jsonl" \
      >"$log_root/$phase-worker-$worker-pass$pass.log" 2>&1 &
    pids+=("$!")
  done
  wait "${pids[@]}"
}

run_phase phase1 1
run_phase phase1 2

PYTHONPATH="$project_root/src" "$python_bin" -m lnet.pac_unified_campaign_cli \
  --stage enqueue-test --workers 2 --output-root "$campaign" \
  >"$log_root/enqueue-test.log" 2>&1

run_phase phase2 1
run_phase phase2 2

PYTHONPATH="$project_root/src" "$python_bin" -m lnet.pac_unified_campaign_cli \
  --stage status --output-root "$campaign" \
  >"$log_root/final-status.log" 2>&1
touch "$campaign/COMPLETE"
