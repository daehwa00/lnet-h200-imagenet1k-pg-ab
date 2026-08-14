#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 7 ]]; then
    echo "usage: $0 GPU WAIT_PID CPU_LIST VARIANT EXPERIMENT_ROOT DATA_ROOT RUNTIME_ROOT" >&2
    exit 2
fi

gpu="$1"
wait_pid="$2"
cpu_list="$3"
variant="$4"
experiment_root="$5"
data_root="$6"
runtime_root="$7"
python_bin="/home/qlab/.conda/envs/lnet-paper-cu128/bin/python"

mkdir -p "${experiment_root}/logs"

while kill -0 "${wait_pid}" 2>/dev/null; do
    sleep 30
done

while [[ -n "$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits)" ]]; do
    sleep 30
done

export CUDA_VISIBLE_DEVICES="${gpu}"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTHONPATH="${runtime_root}/src:${runtime_root}/scripts"
export LNET_COMPILE_MODE=default
export TORCHINDUCTOR_CACHE_DIR="${experiment_root}/torchinductor"
export WANDB_PROJECT="${WANDB_PROJECT:-alphabet2d-imagenet100}"
export WANDB_ENTITY="${WANDB_ENTITY:-daehwa}"
export WANDB_GROUP="${WANDB_GROUP:-${variant}}"

cd "${runtime_root}"
taskset -c "${cpu_list}" "${python_bin}" -u \
    scripts/smoke_a2d_pgv2_h96_vector_input.py \
    --variant followup \
    --model-variant "${variant}" \
    --root "${experiment_root}/smoke" \
    --data-root "${data_root}" \
    --batch-size 128 \
    --compile-mode default \
    >"${experiment_root}/logs/smoke.log" 2>&1

exec taskset -c "${cpu_list}" "${python_bin}" -u \
    scripts/run_a2d_deep4_pgv2_h96_scan_reader_followup_imagenet100.py \
    --root "${experiment_root}/run" \
    --data-root "${data_root}" \
    --variants "${variant}" \
    --run-seeds 501 \
    --epochs 100 \
    --batch-size 128 \
    --gradient-accumulation-steps 1 \
    --workers 8 \
    --precision bfloat16 \
    >"${experiment_root}/logs/train.log" 2>&1
