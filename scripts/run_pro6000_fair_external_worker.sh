#!/usr/bin/env bash
set -euo pipefail

project_root=${1:?project root is required}
output_root=${2:?output root is required}
manifest=${3:?manifest is required}
python_bin=${PAC_EXTERNAL_PYTHON:-python}
python_path="$project_root/src${PAC_MAMBA_ROOT:+:$PAC_MAMBA_ROOT}"
pac_model=pac_headroom_phase_augmented_ensemble_wp_d64_m16

cd "$project_root"
mkdir -p "$output_root/jobs"
while IFS=$'\t' read -r dataset model seed batch_size; do
  [[ -n "$dataset" ]] || continue
  run_name="${dataset}-${model}-seed${seed}"
  run_root="$output_root/jobs/$run_name"
  result="$run_root/results/external_comparisons.csv"
  if [[ -s "$result" ]] && grep -q ',done,' "$result"; then
    continue
  fi
  rm -rf "$run_root"
  PYTHONPATH="$python_path" "$python_bin" -c \
    'import runpy,typing; from typing_extensions import assert_never; typing.assert_never=assert_never; runpy.run_module("lnet.pac_external_benchmarks_cli",run_name="__main__")' \
    --data-root "$project_root/data/external" \
    --output-root "$run_root" \
    --dataset "$dataset" --model "$model" --seed "$seed" \
    --pac-model "$pac_model" --device cuda --model-dim 64 --modes 16 \
    --max-baseline-width 8192 --parameter-match-tolerance 0.05 \
    --epochs 60 --batch-size "$batch_size" --patience 12 || true
done <"$manifest"
