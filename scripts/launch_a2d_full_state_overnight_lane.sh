#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "usage: $0 PHYSICAL_GPU LOCK_FILE PYTHON REPO QUEUE_ARGS..." >&2
  exit 64
fi

physical_gpu=$1
lock_file=$2
python_bin=$3
repo=$4
shift 4

if [[ ! $physical_gpu =~ ^[0-9]+$ ]]; then
  echo "physical GPU must be a nonnegative integer" >&2
  exit 64
fi
if [[ ! -x $python_bin ]]; then
  echo "Python executable is unavailable: $python_bin" >&2
  exit 66
fi
if [[ ! -f $repo/scripts/run_a2d_full_state_overnight_queue.py ]]; then
  echo "overnight queue runner is unavailable under: $repo" >&2
  exit 66
fi

mkdir -p "$(dirname "$lock_file")"
exec 9>"$lock_file"
flock 9

expected_uuid=${LNET_DEVICE_IDENTITY:?LNET_DEVICE_IDENTITY is required}
actual_uuid=$(nvidia-smi -i "$physical_gpu" --query-gpu=uuid --format=csv,noheader,nounits)
if [[ $actual_uuid != "$expected_uuid" ]]; then
  echo "GPU identity mismatch: expected $expected_uuid, got $actual_uuid" >&2
  exit 78
fi

while nvidia-smi -i "$physical_gpu" --query-compute-apps=pid --format=csv,noheader,nounits \
  | grep -Eq '[0-9]+'; do
  sleep 30
done
sleep 5
if nvidia-smi -i "$physical_gpu" --query-compute-apps=pid --format=csv,noheader,nounits \
  | grep -Eq '[0-9]+'; then
  echo "GPU became busy during the launch recheck" >&2
  exit 75
fi

export CUDA_VISIBLE_DEVICES=$physical_gpu
export PYTHONPATH="$repo/src:$repo/scripts${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo"
exec "$python_bin" -u scripts/run_a2d_full_state_overnight_queue.py "$@"
