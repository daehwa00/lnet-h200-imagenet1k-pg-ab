#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 8 ]]; then
    echo "usage: $0 GPU PREDECESSOR_UNIT PREDECESSOR_RESULT CPU_LIST VARIANT_CSV CAMPAIGN_ROOT DATA_ROOT RUNTIME_ROOT" >&2
    exit 2
fi

gpu="$1"
predecessor_unit="$2"
predecessor_result="$3"
cpu_list="$4"
variant_csv="$5"
campaign_root="$6"
data_root="$7"
runtime_root="$8"
python_bin="/home/qlab/.conda/envs/lnet-paper-cu128/bin/python"
minimum_free_kib=$((12 * 1024 * 1024))
allowed_variants=",R0_JIT_COMPLEX_K3,R1_REAL_U,R2_DUAL_FULL_K3,R3_CONTENT_DWQ,R4_FIXED_CONTRAST_Q,R5_CONTENT_PWQ,"

if [[ ! -x "${python_bin}" ]]; then
    echo "missing Python interpreter: ${python_bin}" >&2
    exit 1
fi
if [[ ! -d "${runtime_root}" || ! -d "${data_root}/train" ]]; then
    echo "runtime or ImageNet training directory is missing" >&2
    exit 1
fi
if ! nvidia-smi -i "${gpu}" --query-gpu=index --format=csv,noheader,nounits >/dev/null; then
    echo "unknown GPU index: ${gpu}" >&2
    exit 2
fi
if ! taskset -c "${cpu_list}" true; then
    echo "invalid or unavailable CPU affinity: ${cpu_list}" >&2
    exit 2
fi

IFS=',' read -r -a variants <<<"${variant_csv}"
if [[ ${#variants[@]} -eq 0 ]]; then
    echo "at least one reader variant is required" >&2
    exit 2
fi
for variant in "${variants[@]}"; do
    if [[ -z "${variant}" || "${allowed_variants}" != *",${variant},"* ]]; then
        echo "unsupported reader variant: ${variant}" >&2
        exit 2
    fi
done

mkdir -p "${campaign_root}/logs"

while :; do
    unit_state="$(systemctl --user show "${predecessor_unit}" --property=ActiveState --value 2>/dev/null || true)"
    case "${unit_state}" in
        active|activating|deactivating|reloading)
            sleep 30
            ;;
        *)
            break
            ;;
    esac
done

if [[ ! -s "${predecessor_result}" ]]; then
    echo "predecessor ended without its required result: ${predecessor_result}" >&2
    exit 1
fi
"${python_bin}" - "${predecessor_result}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text())
if not isinstance(payload, dict) or not payload.get("final_validation"):
    raise SystemExit(f"invalid predecessor result: {path}")
PY

while [[ -n "$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits)" ]]; do
    sleep 30
done
if [[ -n "$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits)" ]]; then
    echo "GPU ${gpu} did not become compute-idle" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${gpu}"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export LNET_COMPILE_MODE=default
export PYTHONPATH="${runtime_root}/src:${runtime_root}/scripts"
export WANDB_PROJECT="${WANDB_PROJECT:-alphabet2d-imagenet100}"
export WANDB_ENTITY="${WANDB_ENTITY:-daehwa}"
export WANDB_GROUP="${WANDB_GROUP:-pgv2-real-excitation-readers-20260813}"

cd "${runtime_root}"
# Phase 1: compile and exercise every assigned architecture before allowing any
# long training job to start on this lane.
for variant in "${variants[@]}"; do
    while [[ -n "$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits)" ]]; do
        sleep 30
    done
    available_kib="$(df -Pk "${campaign_root}" | awk 'NR == 2 {print $4}')"
    if [[ ! "${available_kib}" =~ ^[0-9]+$ ]] || (( available_kib < minimum_free_kib )); then
        echo "less than 12 GiB is free before ${variant}: ${available_kib:-unknown} KiB" >&2
        exit 1
    fi

    variant_root="${campaign_root}/${variant}"
    mkdir -p "${variant_root}/logs" "${variant_root}/torchinductor"
    export TORCHINDUCTOR_CACHE_DIR="${variant_root}/torchinductor"

    taskset -c "${cpu_list}" "${python_bin}" -u \
        scripts/smoke_a2d_pgv2_real_excitation_readers.py \
        --variant "${variant}" \
        --root "${variant_root}/smoke" \
        --data-root "${data_root}" \
        --batch-size 128 \
        --compile-mode default \
        >"${variant_root}/logs/smoke.log" 2>&1

    "${python_bin}" - "${variant_root}/smoke/smoke.json" "${variant}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
variant = sys.argv[2]
payload = json.loads(path.read_text())
if payload.get("status") != "PASS" or payload.get("variant") != variant:
    raise SystemExit(f"smoke did not pass for {variant}: {path}")
PY
done

# Phase 2: only a fully preflighted lane may consume 100-epoch training time.
for variant in "${variants[@]}"; do
    while [[ -n "$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits)" ]]; do
        sleep 30
    done
    available_kib="$(df -Pk "${campaign_root}" | awk 'NR == 2 {print $4}')"
    if [[ ! "${available_kib}" =~ ^[0-9]+$ ]] || (( available_kib < minimum_free_kib )); then
        echo "less than 12 GiB is free before ${variant}: ${available_kib:-unknown} KiB" >&2
        exit 1
    fi
    variant_root="${campaign_root}/${variant}"
    export TORCHINDUCTOR_CACHE_DIR="${variant_root}/torchinductor"

    taskset -c "${cpu_list}" "${python_bin}" -u \
        scripts/run_a2d_deep4_pgv2_real_excitation_readers_imagenet100.py \
        --root "${variant_root}/run" \
        --data-root "${data_root}" \
        --variants "${variant}" \
        --run-seeds 501 \
        --epochs 100 \
        --batch-size 128 \
        --gradient-accumulation-steps 1 \
        --workers 8 \
        --precision bfloat16 \
        >"${variant_root}/logs/train.log" 2>&1

    result="${variant_root}/run/results/${variant}__seed501.json"
    if [[ ! -s "${result}" ]]; then
        echo "training ended without its required result: ${result}" >&2
        exit 1
    fi
    "${python_bin}" - "${result}" "${variant}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
variant = sys.argv[2]
payload = json.loads(path.read_text())
history = payload.get("history")
if (
    payload.get("variant") != variant
    or payload.get("seed") != 501
    or not isinstance(payload.get("final_validation"), dict)
    or not isinstance(history, list)
    or not history
    or history[-1].get("epoch") != 100
):
    raise SystemExit(f"incomplete training result for {variant}: {path}")
PY
done
