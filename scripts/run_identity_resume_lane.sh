#!/usr/bin/env bash
set -euo pipefail

queue=$1
gpu=$2
python_bin=$3
output_root=$4

export CUDA_VISIBLE_DEVICES=$gpu
export PYTHONPATH=src

while IFS= read -r manifest; do
    [[ -n "$manifest" ]] || continue
    "$python_bin" -m lnet.pac_h_compact_identity_capacity_cli \
        --stage worker \
        --output-root "$output_root" \
        --manifest "$manifest" \
        --device cuda
done < "$queue"
