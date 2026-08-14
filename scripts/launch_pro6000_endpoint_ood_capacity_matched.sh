#!/usr/bin/env bash
set -euo pipefail

root=${1:-.}
cd "$root"

campaign=.omx/results/pac-endpoint-ood-capacity-matched-pro6000-20260713
log_root=.omx/logs/pac-endpoint-ood-capacity-matched-pro6000-20260713
protocol=.omx/protocols/pac_tf_confirmatory_20260711.json
selection=.omx/results/pac-tf-confirmatory-unseen-20260711/reports/confirmatory_baseline_selection.json
unseen=.omx/results/pac-tf-confirmatory-unseen-20260711
python_bin=${PAC_ENDPOINT_OOD_PYTHON:-python}

mkdir -p "$campaign" "$log_root"
rm -f "$campaign/FAILED" "$campaign/COMPLETE"

models=(tcn cnn1d gru lstm transformer mamba s4d inception_time)
model_args=()
for model in "${models[@]}"; do
  model_args+=(--model "$model")
done

run_queue() {
  PYTHONPATH="$root/src" "$python_bin" -m lnet.pac_tf_p1p2_cli \
    --output-root "$campaign" \
    --protocol-path "$protocol" \
    --selection-path "$selection" \
    --unseen-root "$unseen" \
    --manifest-package synthetic_ood \
    --synthetic-estimand endpoint \
    --synthetic-target-params 11140 \
    "${model_args[@]}" \
    "$@"
}

fail() {
  printf '%s\n' "failed at $(date -Is)" >"$campaign/FAILED"
}
trap fail ERR

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

run_queue --stage enqueue
run_queue --stage workers --device cuda --workers 16 --total-slots 32
run_queue --stage workers --device cuda --workers 16 --total-slots 32
run_queue --stage report

printf '%s\n' "complete at $(date -Is)" >"$campaign/COMPLETE"
trap - ERR
