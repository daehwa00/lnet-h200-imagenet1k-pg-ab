#!/usr/bin/env bash
set -euo pipefail

root=${1:-.}
cd "$root"

upstream=.omx/results/pac-tf-confirmatory-pro6000-20260711
candidate=.omx/results/pac-tf-revised-untied-candidate-20260711
log_root=.omx/logs/pac-tf-revised-untied-candidate-20260711
python_bin=${PAC_REVISED_PYTHON:-python}

mkdir -p "$candidate" "$log_root"
rm -f "$candidate/COMPLETE" "$candidate/FAILED"

fail() {
  printf '%s\n' "failed at $(date -Is)" >"$candidate/FAILED"
}
trap fail ERR

while [[ ! -f "$upstream/COMPLETE" ]]; do
  if [[ -f "$upstream/FAILED" ]]; then
    printf '%s\n' "upstream confirmatory queue failed" >&2
    exit 1
  fi
  sleep 30
done

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

run_candidate() {
  PYTHONPATH="$root/src" "$python_bin" -m lnet.pac_revised_candidate_cli \
    --stage workers --output-root "$candidate" --device cuda \
    --workers 16 --total-slots 32
}

run_candidate
run_candidate
PYTHONPATH="$root/src" "$python_bin" -m lnet.pac_revised_candidate_cli \
  --stage report --output-root "$candidate"

printf '%s\n' "complete at $(date -Is)" >"$candidate/COMPLETE"
trap - ERR
