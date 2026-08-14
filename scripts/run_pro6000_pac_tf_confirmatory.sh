#!/usr/bin/env bash
set -euo pipefail

root=${1:-.}
cd "$root"

protocol=.omx/protocols/pac_tf_confirmatory_20260711.json
capacity=.omx/results/pac-tf-confirmatory-clean-selection-20260711
evidence=.omx/results/pac-tf-confirmatory-selected-evidence-20260711
unseen=.omx/results/pac-tf-confirmatory-unseen-20260711
p1p2=.omx/results/pac-tf-p1p2-confirmatory-20260711
state=.omx/results/pac-tf-confirmatory-pro6000-20260711
log_root=.omx/logs/pac-tf-confirmatory-pro6000-20260711
python_bin=${PAC_CONFIRMATORY_PYTHON:-python}
seeds=(--seeds 7 --seeds 11 --seeds 19 --seeds 23 --seeds 31)

mkdir -p "$state" "$log_root"
rm -f "$state/FAILED" "$state/COMPLETE"

run_module() {
  local module=$1
  shift
  PYTHONPATH="$root/src" "$python_bin" -m "$module" "$@"
}

resume_evidence() {
  local kind=$1
  run_module lnet.pac_tf_evidence_cli \
    --stage workers --kind "$kind" --device cuda --workers 16 --total-slots 32 \
    --output-root "$evidence"
  run_module lnet.pac_tf_evidence_cli \
    --stage workers --kind "$kind" --device cuda --workers 16 --total-slots 32 \
    --output-root "$evidence"
}

run_unseen_workers() {
  run_module lnet.pac_recommended_low_data_cli \
    --stage workers --output-root "$unseen" --device cuda --optimizer-mode fused \
    --workers 16 --total-slots 32 "${seeds[@]}"
  run_module lnet.pac_recommended_low_data_cli \
    --stage workers --output-root "$unseen" --device cuda --optimizer-mode fused \
    --workers 16 --total-slots 32 "${seeds[@]}"
}

fail() {
  printf '%s\n' "failed at $(date -Is)" >"$state/FAILED"
}
trap fail ERR

export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
{
  for kind in core_ablation mechanism_checkpoint interpretability sensitivity; do
    if [[ ! -f "$state/$kind.COMPLETE" ]]; then
      resume_evidence "$kind"
      touch "$state/$kind.COMPLETE"
    fi
  done
} >"$log_root/evidence.log" 2>&1

if [[ ! -f "$state/baselines.COMPLETE" ]]; then
  {
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
    touch "$state/baselines.COMPLETE"
  } >"$log_root/baselines.log" 2>&1
fi

selection="$unseen/reports/confirmatory_baseline_selection.json"
if [[ ! -f "$state/p1p2.COMPLETE" ]]; then
  run_module lnet.pac_tf_p1p2_cli \
    --stage enqueue --output-root "$p1p2" --protocol-path "$protocol" \
    --selection-path "$selection" --unseen-root "$unseen" --device cuda
  run_module lnet.pac_tf_p1p2_cli \
    --stage workers --output-root "$p1p2" --protocol-path "$protocol" \
    --selection-path "$selection" --unseen-root "$unseen" \
    --device cuda --workers 16 --total-slots 32
  run_module lnet.pac_tf_p1p2_cli \
    --stage workers --output-root "$p1p2" --protocol-path "$protocol" \
    --selection-path "$selection" --unseen-root "$unseen" \
    --device cuda --workers 16 --total-slots 32
  touch "$state/p1p2.COMPLETE"
fi

run_module lnet.pac_tf_evidence_cli \
  --stage report --output-root "$evidence" --protocol "$protocol"
run_module lnet.pac_tf_p1p2_cli --stage report --output-root "$p1p2"

remote_root=${PAC_CONFIRMATORY_REMOTE_ROOT:-REMOTE_HOME_PLACEHOLDER/lnet-pac-external-20260710}
remote_host=${PAC_CONFIRMATORY_REMOTE_HOST:-}
if [[ -n "$remote_host" ]]; then
  ssh_key=${PAC_CONFIRMATORY_REMOTE_KEY:?PAC_CONFIRMATORY_REMOTE_KEY is required with remote sync}
  jump_key=${PAC_CONFIRMATORY_JUMP_KEY:?PAC_CONFIRMATORY_JUMP_KEY is required with remote sync}
  remote_ssh=(
    ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
    -o "ProxyCommand=ssh -o BatchMode=yes -o StrictHostKeyChecking=no
      -o UserKnownHostsFile=/dev/null -i $jump_key -p 8589
      secondary_host@REMOTE_HOST_PLACEHOLDER -W %h:%p"
    -i "$ssh_key" -p 40701 "$remote_host"
  )
  tar -czf - "$evidence" "$unseen" "$p1p2" |
    "${remote_ssh[@]}" "cd '$remote_root' && tar -xzf -"
  "${remote_ssh[@]}" "cd '$remote_root' && nohup bash scripts/run_b200_pac64_confirmatory_parallel.sh \"\$PWD\" .omx/results/pac-selected-d64m16-external-20260711 >.omx/logs/pac-tf-confirmatory-parallel-20260711/supervisor.log 2>&1 </dev/null &"
fi

printf '%s\n' "complete at $(date -Is)" >"$state/COMPLETE"
trap - ERR
