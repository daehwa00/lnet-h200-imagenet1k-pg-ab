#!/usr/bin/env bash
set -euo pipefail

project_root=${1:-.}
campaign=${2:-.omx/results/pac-retrained-core-ablation-pro6000-20260713}
log_root=${3:-.omx/logs/pac-retrained-core-ablation-retry-pro6000-20260713}

cd "$project_root"
mkdir -p "$log_root"
export CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$project_root/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

while pgrep -f 'lnet.pac_retrained_ablation_cli --stage worker' >/dev/null; do
  sleep 30
done

for attempt in 1 2 3; do
  status=$(python -m lnet.pac_retrained_ablation_cli --stage status --output-root "$campaign")
  remaining=$(python -c 'import ast,sys; print(ast.literal_eval(sys.stdin.read())["remaining"])' <<<"$status")
  [[ "$remaining" == 0 ]] && exit 0

  stable=0
  while (( stable < 2 )); do
    read -r free_mib util_pct < <(
      nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits \
        | awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); print $1, $2}'
    )
    if (( free_mib >= 49152 && util_pct <= 40 )); then
      stable=$((stable + 1))
    else
      stable=0
    fi
    sleep 30
  done

  pids=()
  for manifest in "$campaign"/manifests/pro6000-gpu0-worker*.jsonl; do
    python -m lnet.pac_retrained_ablation_cli --stage worker \
      --output-root "$campaign" --manifest "$manifest" --device cuda \
      >>"$log_root/attempt-${attempt}.log" 2>&1 &
    pids+=("$!")
    if (( ${#pids[@]} == 2 )); then
      for pid in "${pids[@]}"; do wait "$pid" || true; done
      pids=()
    fi
  done
  for pid in "${pids[@]}"; do wait "$pid" || true; done
done

python -m lnet.pac_retrained_ablation_cli --stage status --output-root "$campaign" \
  >"$log_root/final-status.txt"
