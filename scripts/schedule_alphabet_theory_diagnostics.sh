#!/usr/bin/env bash
set -euo pipefail

root=${1:-.}
cd "$root"

campaign=.omx/results/pac-alphabet-theory-diagnostics-20260713
checkpoint_root=.omx/results/pac-retrained-core-ablation-pro6000-20260713/checkpoints
log_root=.omx/logs/pac-alphabet-theory-diagnostics-20260713
session=pac-alphabet-theory-diagnostics-20260713

mkdir -p "$campaign" "$log_root"
PYTHONPATH="$root/src" python -m lnet.pac_theory_diagnostics_cli \
  --stage prepare --output-root "$campaign" --checkpoint-root "$checkpoint_root"

if tmux has-session -t "$session" 2>/dev/null; then
  tmux kill-session -t "$session"
fi

command="cd '$root' && export CUDA_VISIBLE_DEVICES=0 PYTHONPATH='$root/src' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; while ! python -c \"from pathlib import Path; import sys; sys.exit(0 if len(list(Path('$checkpoint_root').glob('*/*.pt'))) == 90 else 1)\"; do sleep 30; done; idle=0; while (( idle < 2 )); do util=\$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0 | tr -d ' '); if (( util <= 10 )); then idle=\$((idle + 1)); else idle=0; fi; sleep 30; done; python -m lnet.pac_theory_diagnostics_cli --stage worker --output-root '$campaign' --checkpoint-root '$checkpoint_root' --device cuda 2>&1 | tee -a '$log_root/diagnostics.log'; exec bash"
tmux new-session -d -s "$session" -n supervisor "$command"
printf '%s\n' "$session"
