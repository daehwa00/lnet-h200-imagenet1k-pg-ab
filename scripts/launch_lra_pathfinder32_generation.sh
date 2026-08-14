#!/usr/bin/env bash
set -euo pipefail

project_root=${1:-"$PWD"}
workers=${2:-32}
total_images=${3:-200000}
seed=${4:-1729}

if (( total_images % workers != 0 )); then
  echo "total_images must be divisible by workers" >&2
  exit 2
fi

python_bin=REMOTE_ROOT_PLACEHOLDER/miniconda3/envs/ds_cu128_py310/bin/python
generator_root="$project_root/data/external/pathfinder-generator"
output_root="$project_root/data/external/pathfinder"
log_root="$project_root/.omx/logs/pathfinder32-generation"
images_per_worker=$((total_images / workers))

rm -rf "$output_root/source" "$output_root/manifest.csv"
mkdir -p "$output_root" "$log_root"

pids=()
for ((worker = 0; worker < workers; worker++)); do
  PYTHONNOUSERSITE=1 MPLBACKEND=Agg "$python_bin" \
    "$project_root/scripts/generate_lra_pathfinder32.py" generate-shard \
    --generator-root "$generator_root" \
    --output-root "$output_root" \
    --batch-id "$worker" \
    --images "$images_per_worker" \
    --seed "$seed" \
    >"$log_root/shard-$worker.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=$((failed + 1))
  fi
done
if (( failed > 0 )); then
  echo "$failed Pathfinder generation shard(s) failed" >&2
  exit 1
fi

PYTHONNOUSERSITE=1 "$python_bin" \
  "$project_root/scripts/generate_lra_pathfinder32.py" write-manifest \
  --output-root "$output_root" \
  --expected-images "$total_images" \
  --seed "$seed"

touch "$output_root/.complete"
echo "PATHFINDER32_GENERATION_DONE images=$total_images workers=$workers"
