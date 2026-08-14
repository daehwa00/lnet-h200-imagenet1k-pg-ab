#!/usr/bin/env bash
set -euo pipefail

project_root=${1:-"$PWD"}
cd "$project_root"

python_bin=REMOTE_ROOT_PLACEHOLDER/miniconda3/envs/ds_cu128_py310/bin/python
source_root=data/external/pathfinder
prepared_path=data/external/pathfinder.pt
prepared_tmp=data/external/pathfinder.pt.tmp

until [[ -f "$source_root/.complete" ]]; do
  sleep 15
done

rm -f "$prepared_tmp"
PYTHONNOUSERSITE=1 PYTHONPATH=.:src "$python_bin" -c \
  'from pathlib import Path; from lnet.pac_external_tasks import load_pathfinder, save_prepared_task; save_prepared_task(load_pathfinder(Path("data/external/pathfinder")), Path("data/external/pathfinder.pt.tmp"))'
mv "$prepared_tmp" "$prepared_path"

models=(pac tcn transformer mamba gru lstm)
seeds=(7 11 19)
job_index=0
for model in "${models[@]}"; do
  for seed in "${seeds[@]}"; do
    gpu=$((job_index % 3))
    run_name="b200-pathfinder-${model}-seed${seed}"
    log_path=".omx/logs/${run_name}.log"
    pid_path=".omx/pids/${run_name}.pid"
    CUDA_VISIBLE_DEVICES="$gpu" \
      PYTHONNOUSERSITE=1 \
      PYTHONPATH=.:src \
      OMP_NUM_THREADS=2 \
      MKL_NUM_THREADS=2 \
      OPENBLAS_NUM_THREADS=2 \
      nohup conda run -n ds_cu128_py310 python -m lnet.pac_external_benchmarks_cli \
      --data-root data/external \
      --output-root ".omx/results/${run_name}" \
      --dataset pathfinder \
      --model "$model" \
      --pac-model pac_stiefel_depth2_norm_autocorr \
      --device cuda \
      --model-dim 16 \
      --modes 4 \
      --epochs 30 \
      --batch-size 64 \
      --patience 8 \
      --seed "$seed" \
      >"$log_path" 2>&1 < /dev/null &
    echo "$!" >"$pid_path"
    job_index=$((job_index + 1))
  done
done

echo "PATHFINDER32_JOBS_LAUNCHED jobs=$job_index"
