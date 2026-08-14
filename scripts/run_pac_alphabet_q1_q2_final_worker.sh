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

stage=$(basename "$(dirname "$(dirname "$manifest")")")
worker=$(basename "$manifest" .jsonl)
marker="$project_root/$output_root/$stage/worker_markers/$worker.done"
mkdir -p "$(dirname "$marker")"
max_passes=3
if [[ $stage == q2_calibration ]]; then
  max_passes=2
fi

status=1
for ((attempt = 1; attempt <= max_passes; attempt += 1)); do
  "$python_bin" -m lnet.pac_alphabet_q1_q2_final_cli \
    --stage worker \
    --output-root "$output_root" \
    --manifest "$manifest" \
    --device cuda
  status=$?
  if [[ $attempt -lt $max_passes ]]; then
    sleep 5
  fi
done
allow_failed=()
if [[ $stage == q2_calibration ]]; then
  allow_failed=(--allow-failed)
fi
if "$python_bin" "$project_root/scripts/check_pac_manifest_complete.py" \
  --output-root "$project_root/$output_root" --manifest "$manifest" "${allow_failed[@]}"; then
  : >"$marker"
  exit 0
fi
exit 1
