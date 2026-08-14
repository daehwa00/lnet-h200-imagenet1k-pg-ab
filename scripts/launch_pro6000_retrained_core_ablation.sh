#!/usr/bin/env bash
set -euo pipefail

root=${1:-.}
cd "$root"

campaign=.omx/results/pac-retrained-core-ablation-pro6000-20260713
log_root=.omx/logs/pac-retrained-core-ablation-pro6000-20260713
session=pac-retrained-core-ablation-pro6000-20260713
workers=6

mkdir -p "$campaign" "$log_root"
PYTHONPATH="$root/src" python -m lnet.pac_retrained_ablation_cli \
  --stage enqueue --output-root "$campaign" --workers "$workers"

if tmux has-session -t "$session" 2>/dev/null; then
  tmux kill-session -t "$session"
fi

for worker in $(seq 0 $((workers - 1))); do
  manifest="$campaign/manifests/pro6000-gpu0-worker${worker}.jsonl"
  log="$log_root/worker${worker}.log"
  command="cd '$root' && export CUDA_VISIBLE_DEVICES=0 PYTHONPATH='$root/src' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 && python -m lnet.pac_retrained_ablation_cli --stage worker --output-root '$campaign' --manifest '$manifest' --device cuda 2>&1 | tee -a '$log'; exec bash"
  if (( worker == 0 )); then
    tmux new-session -d -s "$session" -n "worker${worker}" "$command"
  else
    tmux new-window -t "$session" -n "worker${worker}" "$command"
  fi
done

tmux select-window -t "$session:worker0"
printf '%s\n' "$session"
