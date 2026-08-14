#!/usr/bin/env bash
set -uo pipefail

project_root=${1:?project root is required}
output_root=${2:?output root is required}
manifest=${3:?manifest is required}
python_bin=${4:?python binary is required}
python_path=${5:?python path is required}

cd "$project_root" || exit 1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$python_path"

status=1
for attempt in 1 2 3; do
  "$python_bin" -m lnet.pac_efp_compact_equal_search_cli \
    --stage worker \
    --output-root "$output_root" \
    --manifest "$manifest" \
    --device cuda
  status=$?
  if [[ $status -eq 0 && $attempt -ge 2 ]]; then
    break
  fi
  sleep 20
done
exit "$status"
