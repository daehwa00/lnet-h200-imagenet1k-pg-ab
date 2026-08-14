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
max_active_per_gpu=${PAC_MAX_ACTIVE_PER_GPU:-$workers_per_gpu}

session="pac-efp-compact-ext-${stage}-${host_tag}-20260719"

worker_script="$project_root/scripts/run_pac_efp_compact_external_equal_search_worker.sh"
if [[ ! -x $worker_script ]]; then
  worker_script="$project_root/run_pac_efp_compact_external_equal_search_worker.sh"
fi
log_root="$project_root/.omx/logs/pac-efp-compact-external-equal-search-20260719"
mkdir -p "$log_root"

for ((index = worker_start; index <= worker_end; index += 1)); do
  worker=$(printf 'worker-%02d' "$index")
  marker="$project_root/$output_root/$stage/worker_markers/$worker.done"
  if [[ -f $marker ]]; then
    continue
  fi
  if tmux has-session -t "$session" 2>/dev/null && tmux list-windows -t "$session" \
    -F '#{window_name}' | grep -Fxq "$worker"; then
    continue
  fi
  gpu=$(( (index - worker_start) / workers_per_gpu ))
  active_on_gpu=0
  group_start=$((worker_start + gpu * workers_per_gpu))
  group_end=$((group_start + workers_per_gpu - 1))
  if ((group_end > worker_end)); then
    group_end=$worker_end
  fi
  for ((candidate = group_start; candidate <= group_end; candidate += 1)); do
    candidate_worker=$(printf 'worker-%02d' "$candidate")
    if tmux has-session -t "$session" 2>/dev/null && tmux list-windows -t "$session" \
      -F '#{window_name}' | grep -Fxq "$candidate_worker"; then
      active_on_gpu=$((active_on_gpu + 1))
    fi
  done
  if ((active_on_gpu >= max_active_per_gpu)); then
    continue
  fi
  manifest="$project_root/$output_root/$stage/manifests/$worker.jsonl"
  [[ -f $manifest ]]
  log="$log_root/${stage}-${host_tag}-${worker}.log"
  command="cd '$project_root' && CUDA_VISIBLE_DEVICES=$gpu bash '$worker_script' '$project_root' '$output_root' '$manifest' '$python_bin' '$python_path' >'$log' 2>&1"
  if ! tmux has-session -t "$session" 2>/dev/null; then
    tmux new-session -d -s "$session" -n "$worker" "$command"
  else
    tmux new-window -t "$session" -n "$worker" "$command"
  fi
done
