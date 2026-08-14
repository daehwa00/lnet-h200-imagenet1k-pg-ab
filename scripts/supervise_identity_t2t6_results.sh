#!/usr/bin/env bash
set -euo pipefail

project_root="."
campaign="${1:-pac-h-compact-identity-capacity-q1-final-t2t6-optimized-v3-20260722}"
local_root="${project_root}/.omx/results/${campaign}"
kau_root="REMOTE_HOME_PLACEHOLDER/lnet-identity-t2t6-20260722/.omx/results/${campaign}"
local_gpu_root="LOCAL_HOME_PLACEHOLDER/lnet-identity-t2t6-20260722/.omx/results/${campaign}"
kau_ssh="ssh -i LOCAL_HOME_PLACEHOLDER/.codex/attachments/OPAQUE_ID_PLACEHOLDER/SSH_KEY_PLACEHOLDER -p 8589"
local_gpu_ssh="ssh -i LOCAL_HOME_PLACEHOLDER/.codex/attachments/OPAQUE_ID_PLACEHOLDER/SSH_KEY_PLACEHOLDER -p 5003"

sync_results() {
  rsync -az -e "${kau_ssh}" \
    "secondary_host@REMOTE_HOST_PLACEHOLDER:${kau_root}/final/" \
    "${local_root}/final/"
  rsync -az -e "${local_gpu_ssh}" \
    "local_gpu@REMOTE_HOST_PLACEHOLDER:${local_gpu_root}/final/" \
    "${local_root}/final/"
}

remote_workers_active() {
  ${kau_ssh} secondary_host@REMOTE_HOST_PLACEHOLDER \
    "tmux list-sessions -F '#{session_name}' 2>/dev/null | grep -q '^idt2t6-kau-'" \
    || ${local_gpu_ssh} local_gpu@REMOTE_HOST_PLACEHOLDER \
      "tmux list-sessions -F '#{session_name}' 2>/dev/null | grep -q '^idt2t6-local_gpu-'"
}

mkdir -p "${local_root}/logs"
while remote_workers_active; do
  sync_results
  PYTHONPATH="${project_root}/src" \
    python -m lnet.pac_h_compact_identity_capacity_cli \
      --stage status \
      --output-root "${local_root}" \
      >"${local_root}/logs/latest-status.json"
  sleep 60
done
sync_results
PYTHONPATH="${project_root}/src" \
  python -m lnet.pac_h_compact_identity_capacity_cli \
    --stage status \
    --output-root "${local_root}" \
    >"${local_root}/logs/latest-status.json"
