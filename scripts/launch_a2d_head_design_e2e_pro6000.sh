#!/usr/bin/env bash
set -euo pipefail

repo=/home/qlab/daehwa/lnet
frozen=/home/qlab/experiments/alphabet/a2d-head-design-20260806
root=/home/qlab/experiments/alphabet/a2d-head-design-e2e-20260806
data=/home/qlab/data/ImageNet100
log="$root/runner.log"

mkdir -p "$root"
exec 9>"$root/runner.lock"
if ! flock -n 9; then
  echo "A2D end-to-end head runner is already active" >&2
  exit 0
fi

cd "$repo"
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$repo/src:$repo/scripts"
export PYTHONUNBUFFERED=1
export WANDB_PROJECT="${WANDB_PROJECT:-alphabet2d-imagenet100}"
export WANDB_ENTITY="${WANDB_ENTITY:-daehwa}"
export WANDB_GROUP=A2D-HeadDesign-E2E
export TORCHINDUCTOR_CACHE_DIR="$root/torchinductor"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export LNET_PAC_BLOCK_MODES=16
export LNET_PAC_RECURRENCE_FORWARD_NUM_WARPS=8
export LNET_PAC_RECURRENCE_BACKWARD_NUM_WARPS=8

while [[ ! -f "$frozen/complete-analysis.json" ]]; do
  echo "$(date --iso-8601=seconds) waiting for frozen-head analysis" >>"$log"
  sleep 30
done

wait_for_memory() {
  while ! python - <<'PY'
import sys, torch
free, _ = torch.cuda.mem_get_info(0)
sys.exit(0 if free >= 6 * 1024**3 else 1)
PY
  do
    echo "$(date --iso-8601=seconds) waiting for 6 GiB free GPU memory" >>"$log"
    sleep 30
  done
}

wait_for_memory
if [[ ! -f "$root/SMOKE_OK" ]]; then
  env \
    python -u scripts/smoke_a2d_head_design_e2e.py \
    --size 224 --batch-size 32 >>"$log" 2>&1
  date --iso-8601=seconds >"$root/SMOKE_OK"
fi

for attempt in $(seq 1 20); do
  complete=1
  for variant in A2D-W768 A2D-StageResidual A2D-Drop020; do
    if [[ ! -f "$root/results/${variant}__seed501.json" ]]; then
      complete=0
    fi
  done
  if [[ $complete -eq 1 ]]; then
    echo "$(date --iso-8601=seconds) end-to-end campaign complete" >>"$log"
    exit 0
  fi
  wait_for_memory
  echo "$(date --iso-8601=seconds) end-to-end attempt $attempt" >>"$log"
  set +e
  python -u scripts/run_a2d_head_design_e2e_imagenet100.py \
    --root "$root" \
    --data-root "$data" \
    --variants A2D-W768 A2D-StageResidual A2D-Drop020 \
    --run-seeds 501 \
    --epochs 100 \
    --batch-size 32 \
    --gradient-accumulation-steps 8 \
    --workers 8 \
    --precision float32 >>"$log" 2>&1
  status=$?
  set -e
  echo "$(date --iso-8601=seconds) end-to-end exit $status" >>"$log"
  sleep 30
done

echo "$(date --iso-8601=seconds) end-to-end retries exhausted" >>"$log"
exit 1
