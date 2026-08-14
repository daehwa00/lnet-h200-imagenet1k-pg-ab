#!/usr/bin/env bash
set -euo pipefail

project_root=${1:?project root is required}
output_root=${2:?output root is required}
manifest=${3:?manifest is required}
data_root=${4:?data root is required}
completion_marker=${5:?completion marker is required}

python_bin=${PAC_EXTERNAL_PYTHON:-python}
mamba_root=${PAC_MAMBA_ROOT:-}
python_path="$project_root/src"
if [[ -n "$mamba_root" ]]; then
  python_path+=":$mamba_root"
fi

cd "$project_root"
while IFS=$'\t' read -r dataset model seed batch_size; do
  [[ -n "$dataset" ]] || continue
  run_name="${dataset}-${model}-seed${seed}"
  run_root="$project_root/$output_root/jobs/$run_name"
  result="$run_root/results/external_comparisons.csv"
  if [[ -s "$result" ]] && grep -q ',done,' "$result"; then
    echo "SKIP $(date -Is) $run_name"
    continue
  fi
  echo "START $(date -Is) $run_name"
  rm -rf "$run_root"
  set +e
  PYTHONPATH="$python_path" "$python_bin" -c \
    'import runpy,typing; from typing_extensions import assert_never; typing.assert_never=assert_never; runpy.run_module("lnet.pac_external_benchmarks_cli",run_name="__main__")' \
    --data-root "$data_root" \
    --output-root "$run_root" \
    --dataset "$dataset" \
    --model "$model" \
    --pac-model pac_stiefel_depth2_norm_autocorr \
    --device cuda \
    --model-dim 64 \
    --modes 16 \
    --epochs 30 \
    --batch-size "$batch_size" \
    --patience 8 \
    --seed "$seed"
  status=$?
  set -e
  echo "END $(date -Is) $run_name status=$status"
done <"$manifest"

touch "$completion_marker"
