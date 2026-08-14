#!/usr/bin/env bash
set -euo pipefail

project_root=${1:?project root is required}
external_root=${2:-.omx/results/pac-selected-d64m16-external-20260711}
python_bin=REMOTE_ROOT_PLACEHOLDER/miniconda3/envs/ds/bin/python3.12
mamba_root=REMOTE_ROOT_PLACEHOLDER/ds/Adversarial-Scenario/third_party/Vim/mamba-1p1p1
protocol=.omx/protocols/pac_tf_confirmatory_20260711.json
capacity=.omx/results/pac-tf-confirmatory-clean-selection-20260711
evidence=.omx/results/pac-tf-confirmatory-selected-evidence-20260711
unseen=.omx/results/pac-tf-confirmatory-unseen-20260711
p1p2=.omx/results/pac-tf-p1p2-confirmatory-20260711
state=.omx/results/pac-tf-confirmatory-parallel-20260711
log_root="$project_root/.omx/logs/pac-tf-confirmatory-parallel-20260711"
seeds=(--seeds 7 --seeds 11 --seeds 19 --seeds 23 --seeds 31)

mkdir -p "$project_root/$state" "$log_root"
cd "$project_root"

run_module() {
  local module=$1
  shift
  PYTHONPATH="$project_root/src:$mamba_root" "$python_bin" -c \
    "import runpy,typing; from typing_extensions import assert_never; typing.assert_never=assert_never; runpy.run_module('$module',run_name='__main__')" \
    "$@"
}

resume_evidence() {
  local kind=$1
  run_module lnet.pac_tf_evidence_cli \
    --stage workers --kind "$kind" --device cuda --workers 8 --total-slots 16 \
    --output-root "$evidence"
  # A second pass retries only non-terminal keys and is a no-op for completed work.
  run_module lnet.pac_tf_evidence_cli \
    --stage workers --kind "$kind" --device cuda --workers 8 --total-slots 16 \
    --output-root "$evidence"
}

run_unseen_workers() {
  run_module lnet.pac_recommended_low_data_cli \
    --stage workers --output-root "$unseen" --device cuda --optimizer-mode fused \
    --workers 8 --total-slots 16 "${seeds[@]}"
  run_module lnet.pac_recommended_low_data_cli \
    --stage workers --output-root "$unseen" --device cuda --optimizer-mode fused \
    --workers 8 --total-slots 16 "${seeds[@]}"
}

fail() {
  printf '%s\n' "failed at $(date -Is)" >"$project_root/$state/FAILED"
}
trap fail ERR

# Do not contend with the selected 17-task, D64/M16 external campaign.
while [[ ! -f "$project_root/$external_root/COMPLETE" ]]; do
  sleep 60
done

(
  export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  resume_evidence core_ablation
  resume_evidence sensitivity
  touch "$project_root/$state/lane-evidence-a.COMPLETE"
) >"$log_root/lane-evidence-a.log" 2>&1 &
lane_a=$!

(
  export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  resume_evidence mechanism_checkpoint
  resume_evidence interpretability
  touch "$project_root/$state/lane-evidence-b.COMPLETE"
) >"$log_root/lane-evidence-b.log" 2>&1 &
lane_b=$!

(
  export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  run_module lnet.pac_recommended_low_data_cli \
    --stage enqueue-unseen-validation --output-root "$unseen" \
    --selection-root "$capacity" --protocol-path "$protocol" "${seeds[@]}"
  run_unseen_workers
  run_module lnet.pac_recommended_low_data_cli --stage report --output-root "$unseen"
  run_module lnet.pac_recommended_low_data_cli \
    --stage enqueue-unseen-final --output-root "$unseen" \
    --selection-root "$capacity" --protocol-path "$protocol" "${seeds[@]}"
  run_unseen_workers
  run_module lnet.pac_recommended_low_data_cli --stage report --output-root "$unseen"
  touch "$project_root/$state/lane-baselines.COMPLETE"
) >"$log_root/lane-baselines.log" 2>&1 &
lane_c=$!

wait "$lane_a" "$lane_b" "$lane_c"

selection="$unseen/reports/confirmatory_baseline_selection.json"
run_module lnet.pac_tf_p1p2_cli \
  --stage enqueue --output-root "$p1p2" --protocol-path "$protocol" \
  --selection-path "$selection" --unseen-root "$unseen" --device cuda

run_p1p2_package() {
  local package=$1
  run_module lnet.pac_tf_p1p2_cli \
    --stage workers --package "$package" --output-root "$p1p2" \
    --protocol-path "$protocol" --selection-path "$selection" --unseen-root "$unseen" \
    --device cuda --workers 8 --total-slots 16
  run_module lnet.pac_tf_p1p2_cli \
    --stage workers --package "$package" --output-root "$p1p2" \
    --protocol-path "$protocol" --selection-path "$selection" --unseen-root "$unseen" \
    --device cuda --workers 8 --total-slots 16
}

(
  export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  run_p1p2_package low_data
  run_p1p2_package real_diagnostics
  touch "$project_root/$state/lane-p1p2-a.COMPLETE"
) >"$log_root/lane-p1p2-a.log" 2>&1 &
p1=$!

(
  export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  run_p1p2_package synthetic_ood
  run_p1p2_package real_domain_ood
  touch "$project_root/$state/lane-p1p2-b.COMPLETE"
) >"$log_root/lane-p1p2-b.log" 2>&1 &
p2=$!

(
  export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  run_p1p2_package efficiency
  touch "$project_root/$state/lane-p1p2-c.COMPLETE"
) >"$log_root/lane-p1p2-c.log" 2>&1 &
p3=$!

wait "$p1" "$p2" "$p3"

run_module lnet.pac_tf_evidence_cli \
  --stage report --output-root "$evidence" --protocol "$protocol"
run_module lnet.pac_tf_p1p2_cli --stage report --output-root "$p1p2"
run_module lnet.pac_tf_confirmatory_report_cli \
  --protocol "$protocol" --unseen-root "$unseen" --p1p2-root "$p1p2" \
  --evidence-root "$evidence" --output-root "$state/reports"

rm -f "$project_root/$state/FAILED"
printf '%s\n' "complete at $(date -Is)" >"$project_root/$state/COMPLETE"
trap - ERR
