#!/usr/bin/env bash
set -euo pipefail

project_root=${1:?project root is required}
platform=${2:?platform must be b200 or pro6000}

python_bin=${PAC_QUEUE_PYTHON:-python}
mamba_root=${PAC_MAMBA_ROOT:-}
python_path="$project_root/src"
if [[ -n "$mamba_root" ]]; then
  python_path+=":$mamba_root"
fi
external_root=.omx/results/pac-revised-external-final-20260712
ucr_root=.omx/results/pac-revised-ucr-official-test-20260712
log_root="$project_root/.omx/logs/pac-revised-final-20260712"
mkdir -p "$log_root"

launch_external() {
  local index=$1
  local gpu=$2
  local manifest="$project_root/$external_root/manifests/worker-$(printf '%02d' "$index").tsv"
  local marker="$project_root/$external_root/completion/worker-$(printf '%02d' "$index").COMPLETE"
  local log="$log_root/external-worker-$(printf '%02d' "$index").log"
  local pid_file="$log_root/external-worker-$(printf '%02d' "$index").pid"
  if [[ -f "$marker" ]] && [[ "$(<"${marker}.failure-count")" == 0 ]]; then
    return
  fi
  if [[ -f "$pid_file" ]] && kill -0 "$(<"$pid_file")" 2>/dev/null; then
    return
  fi
  nohup env CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 PAC_EXTERNAL_PYTHON="$python_bin" PAC_MAMBA_ROOT="$mamba_root" \
    bash "$project_root/scripts/run_pac_revised_external_final_worker.sh" \
      "$project_root" "$external_root" "$manifest" "$marker" \
      >"$log" 2>&1 </dev/null &
  echo "$!" >"$pid_file"
}

launch_ucr() {
  local shard=$1
  local gpu=$2
  local shard_root="$project_root/$ucr_root/shards/shard-$shard"
  local log="$log_root/ucr-shard-$shard.log"
  local pid_file="$log_root/ucr-shard-$shard.pid"
  if [[ -f "$pid_file" ]] && kill -0 "$(<"$pid_file")" 2>/dev/null; then
    return
  fi
  local workers=6
  [[ "$platform" == pro6000 ]] && workers=4
  nohup env CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 PYTHONPATH="$python_path" \
    "$python_bin" -c \
      'import runpy,typing; from typing_extensions import assert_never; typing.assert_never=assert_never; runpy.run_module("lnet.pac_recommended_low_data_cli",run_name="__main__")' \
      --stage workers --output-root "$shard_root" --preset full --device cuda \
      --workers "$workers" --total-slots "$workers" \
      >"$log" 2>&1 </dev/null &
  echo "$!" >"$pid_file"
}

case "$platform" in
  b200)
    gpus=(0 1 2 4 5 6 7)
    for index in $(seq 0 27); do
      launch_external "$index" "${gpus[$((index % 7))]}"
    done
    for shard in $(seq 0 6); do
      launch_ucr "$shard" "${gpus[$shard]}"
    done
    ;;
  pro6000)
    for index in $(seq 28 31); do
      launch_external "$index" 0
    done
    launch_ucr 7 0
    ;;
  *)
    echo "platform must be b200 or pro6000" >&2
    exit 2
    ;;
esac

if [[ "${PAC_QUEUE_WAIT:-0}" == 1 ]]; then
  wait
fi
