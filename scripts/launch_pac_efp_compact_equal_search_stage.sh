#!/usr/bin/env bash
set -euo pipefail

project_root=${1:?project root is required}
output_root=${2:?output root is required}
stage=${3:?stage is required}
worker_start=${4:?first worker index is required}
worker_end=${5:?last worker index is required}
workers_per_gpu=${6:?workers per GPU is required}
python_bin=${7:?python binary is required}
python_path=${8:?python path is required}
host_tag=${9:?host tag is required}

session="pac-efp-compact-${stage}-${host_tag}-20260719"
tmux has-session -t "$session" 2>/dev/null && exit 0

worker_script="$project_root/scripts/run_pac_efp_compact_equal_search_worker.sh"
if [[ ! -x $worker_script ]]; then
  worker_script="$project_root/run_pac_efp_compact_equal_search_worker.sh"
fi
log_root="$project_root/.omx/logs/pac-efp-compact-equal-search-20260719"
mkdir -p "$log_root"

window=0
for ((index = worker_start; index <= worker_end; index += 1)); do
  worker=$(printf 'worker-%02d' "$index")
  gpu=$(( (index - worker_start) / workers_per_gpu ))
  manifest="$project_root/$output_root/$stage/manifests/$worker.jsonl"
  [[ -f $manifest ]]
  log="$log_root/${stage}-${host_tag}-${worker}.log"
  command="cd '$project_root' && CUDA_VISIBLE_DEVICES=$gpu bash '$worker_script' '$project_root' '$output_root' '$manifest' '$python_bin' '$python_path' >'$log' 2>&1"
  if [[ $window -eq 0 ]]; then
    tmux new-session -d -s "$session" -n "$worker" "$command"
  else
    tmux new-window -t "$session" -n "$worker" "$command"
  fi
  window=$((window + 1))
done
