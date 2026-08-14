#!/usr/bin/env bash
set -euo pipefail

project_root=${1:?project root is required}
output_root=${2:-.omx/results/pac-extended-baselines-20260711}
mode=${3:-launch}
worker_id=${4:-0}
slots=${PAC_BASELINE_SLOTS:-16}
python_bin=REMOTE_ROOT_PLACEHOLDER/miniconda3/envs/ds_cu128_py310/bin/python
mamba_root=REMOTE_ROOT_PLACEHOLDER/ds/Adversarial-Scenario/third_party/Vim/mamba-1p1p1
log_root="$project_root/.omx/logs/pac-extended-baselines-20260711"
models=(s4 s4d minirocket inception_time)
seeds=(7 11 19)
datasets=(
  ptb-xl mit-bih cwru speech-commands pathfinder
  ettm1 ettm2 electricity weather
  lra-listops lra-text lra-retrieval lra-image
  sequential-mnist permuted-mnist sequential-cifar audioset-balanced
)
batches=(64 64 64 64 32 64 64 64 64 32 64 16 64 64 64 64 64)

mkdir -p "$project_root/$output_root/jobs" "$log_root"

if [[ "$mode" == launch ]]; then
  : >"$project_root/$output_root/launches.tsv"
  for ((slot = 0; slot < slots; slot++)); do
    gpu=$((slot % 3))
    log="$log_root/worker-$slot.log"
    nohup env CUDA_VISIBLE_DEVICES="$gpu" \
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      PAC_BASELINE_SLOTS="$slots" \
      bash "$0" "$project_root" "$output_root" worker "$slot" \
      >"$log" 2>&1 </dev/null &
    echo "$! $gpu worker-$slot" >>"$project_root/$output_root/launches.tsv"
  done
  exit 0
fi

if [[ "$mode" != worker ]]; then
  echo "mode must be launch or worker" >&2
  exit 2
fi

global_index=0
for dataset_index in "${!datasets[@]}"; do
  dataset=${datasets[$dataset_index]}
  batch_size=${batches[$dataset_index]}
  for model in "${models[@]}"; do
    for seed in "${seeds[@]}"; do
      if (( global_index % slots != worker_id )); then
        global_index=$((global_index + 1))
        continue
      fi
      run_name="${dataset}-${model}-seed${seed}"
      run_root="$project_root/$output_root/jobs/$run_name"
      result="$run_root/results/external_comparisons.csv"
      if [[ -s "$result" ]] && grep -q ',done,' "$result"; then
        global_index=$((global_index + 1))
        continue
      fi
      echo "START $(date -Is) $run_name"
      rm -rf "$run_root"
      set +e
      PYTHONPATH="$project_root/src:$mamba_root" "$python_bin" -c \
        'import runpy,typing; from typing_extensions import assert_never; typing.assert_never=assert_never; runpy.run_module("lnet.pac_external_benchmarks_cli",run_name="__main__")' \
        --data-root "$project_root/data/external" \
        --output-root "$run_root" \
        --dataset "$dataset" \
        --model "$model" \
        --device cuda \
        --model-dim 16 \
        --modes 4 \
        --epochs 30 \
        --batch-size "$batch_size" \
        --patience 8 \
        --seed "$seed"
      status=$?
      set -e
      echo "END $(date -Is) $run_name status=$status"
      global_index=$((global_index + 1))
    done
  done
done

touch "$project_root/$output_root/worker-$worker_id.COMPLETE"
