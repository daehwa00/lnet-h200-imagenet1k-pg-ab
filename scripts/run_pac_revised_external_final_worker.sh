#!/usr/bin/env bash
set -euo pipefail

project_root=${1:?project root is required}
output_root=${2:?output root is required}
manifest=${3:?manifest is required}
completion_marker=${4:?completion marker is required}

python_bin=${PAC_EXTERNAL_PYTHON:-python}
mamba_root=${PAC_MAMBA_ROOT:-}
python_path="$project_root/src"
if [[ -n "$mamba_root" ]]; then
  python_path+=":$mamba_root"
fi

mkdir -p "$project_root/$output_root/jobs" "$(dirname "$completion_marker")"
cd "$project_root"
failures=0
while IFS=$'\t' read -r dataset model seed batch_size; do
  [[ -n "$dataset" ]] || continue
  run_name="${dataset}-${model}-seed${seed}"
  run_root="$project_root/$output_root/jobs/$run_name"
  result="$run_root/results/external_comparisons.csv"
  if [[ -s "$result" ]] && grep -q ',done,' "$result"; then
    continue
  fi
  rm -rf "$run_root"
  set +e
  PYTHONPATH="$python_path" "$python_bin" -c \
    'import runpy,typing; from typing_extensions import assert_never; typing.assert_never=assert_never; runpy.run_module("lnet.pac_external_benchmarks_cli",run_name="__main__")' \
    --data-root "$project_root/data/external" \
    --output-root "$run_root" \
    --dataset "$dataset" \
    --model "$model" \
    --pac-model pac_stiefel_revised_fixed_mean_nogate_untied_d64_m16 \
    --device cuda \
    --model-dim 64 \
    --modes 16 \
    --max-baseline-width 8192 \
    --parameter-match-tolerance 0.05 \
    --epochs 60 \
    --batch-size "$batch_size" \
    --patience 12 \
    --seed "$seed"
  status=$?
  set -e
  if (( status != 0 )) || [[ ! -s "$result" ]] || ! grep -q ',done,' "$result"; then
    failures=$((failures + 1))
    echo -e "${dataset}\t${model}\t${seed}\t${status}" >>"${completion_marker}.failures.tsv"
  fi
done <"$manifest"

echo "$failures" >"${completion_marker}.failure-count"
touch "$completion_marker"
