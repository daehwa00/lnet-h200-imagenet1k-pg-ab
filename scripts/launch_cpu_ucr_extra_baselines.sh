#!/usr/bin/env bash
set -euo pipefail

root=${1:-.}
cd "$root"

campaign=.omx/results/pac-ucr-extra-baselines-submission-20260713
worker_root="$campaign/inception-official-test-missing"
log_root=.omx/logs/pac-ucr-extra-baselines-submission-20260713
session=pac-ucr-extra-baselines-submission-20260713

mkdir -p "$campaign" "$worker_root" "$log_root"
PYTHONPATH="$root/src" python -m lnet.pac_ucr_extra_baseline_submission_cli \
  --stage prepare --output-root "$campaign"

if tmux has-session -t "$session" 2>/dev/null; then
  tmux kill-session -t "$session"
fi

command="cd '$root' && export CUDA_VISIBLE_DEVICES='' PYTHONPATH='$root/src' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 && python -m lnet.pac_recommended_low_data_cli --stage workers --output-root '$worker_root' --preset full --device cpu --workers 4 --total-slots 4 2>&1 | tee -a '$log_root/worker.log'; exec bash"
tmux new-session -d -s "$session" -n worker "$command"
printf '%s\n' "$session"
