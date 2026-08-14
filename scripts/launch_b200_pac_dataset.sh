#!/usr/bin/env bash
set -euo pipefail

project_root=${1:?project root is required}
dataset=${2:?dataset is required}
batch_size=${3:-64}
output_root=${4:-.omx/results/pac-additional-20260710}
log_root="$project_root/.omx/logs/pac-additional-20260710"
python_bin=REMOTE_ROOT_PLACEHOLDER/miniconda3/envs/ds_cu128_py310/bin/python
mamba_root=REMOTE_ROOT_PLACEHOLDER/ds/Adversarial-Scenario/third_party/Vim/mamba-1p1p1
models=(pac tcn transformer mamba gru lstm)
seeds=(7 11 19)
gpus=(0 1 2)

mkdir -p "$project_root/$output_root/jobs" "$log_root"
index=0
for model in "${models[@]}"; do
  for seed in "${seeds[@]}"; do
    gpu=${gpus[$((index % ${#gpus[@]}))]}
    run_name="${dataset}-${model}-seed${seed}"
    run_root="$project_root/$output_root/jobs/$run_name"
    result="$run_root/results/external_comparisons.csv"
    log="$log_root/$run_name.log"
    if [[ -s "$result" ]] && grep -q ',done,' "$result"; then
      index=$((index + 1))
      continue
    fi
    command=(
      "$python_bin" -c
      'import runpy,typing; from typing_extensions import assert_never; typing.assert_never=assert_never; runpy.run_module("lnet.pac_external_benchmarks_cli",run_name="__main__")'
      --data-root "$project_root/data/external"
      --output-root "$run_root"
      --dataset "$dataset"
      --model "$model"
      --pac-model pac_stiefel_depth2_norm_autocorr
      --device cuda
      --model-dim 16
      --modes 4
      --epochs 30
      --batch-size "$batch_size"
      --patience 8
      --seed "$seed"
    )
    nohup env \
      CUDA_VISIBLE_DEVICES="$gpu" \
      OMP_NUM_THREADS=1 \
      MKL_NUM_THREADS=1 \
      OPENBLAS_NUM_THREADS=1 \
      PYTHONPATH="$project_root/src:$mamba_root" \
      "${command[@]}" >"$log" 2>&1 </dev/null &
    echo "$! $gpu $run_name" >>"$project_root/$output_root/launches.tsv"
    index=$((index + 1))
  done
done
