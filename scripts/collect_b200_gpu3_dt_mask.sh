#!/usr/bin/env bash
set -euo pipefail

local_root=${1:-.}
remote_root=${PAC_B200_SUBMISSION_ROOT:-REMOTE_HOME_PLACEHOLDER/lnet-pac-submission-20260713}
remote_host=${PAC_B200_HOST:-remote_user@REMOTE_HOST_PLACEHOLDER}
remote_python=${PAC_B200_PYTHON:-REMOTE_ROOT_PLACEHOLDER/miniconda3/envs/ASA_test/bin/python}
ssh_key=${PAC_B200_KEY:-LOCAL_HOME_PLACEHOLDER/.codex/attachments/OPAQUE_ID_PLACEHOLDER/SSH_KEY_PLACEHOLDER}
jump_key=${PAC_B200_JUMP_KEY:-LOCAL_HOME_PLACEHOLDER/.codex/attachments/OPAQUE_ID_PLACEHOLDER/SSH_KEY_PLACEHOLDER}
result=.omx/results/pac-dt-mask-attribution-pro6000-20260713
session=pac-dt-mask-b200-gpu3-20260713

remote_ssh=(
  ssh -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null -i "$ssh_key" -p 40701
  -o "ProxyCommand=ssh -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i $jump_key -p 8589 secondary_host@REMOTE_HOST_PLACEHOLDER -W %h:%p"
  "$remote_host"
)

while true; do
  completed=$("${remote_ssh[@]}" "find '$remote_root/$result/jobs' -maxdepth 1 -type f -name 'dt_mask__*.json' 2>/dev/null | wc -l")
  printf '%s completed=%s/25\n' "$(date -Is)" "$completed"
  if [[ "$completed" == 25 ]]; then
    break
  fi
  if ! "${remote_ssh[@]}" "tmux has-session -t '$session'" 2>/dev/null; then
    "${remote_ssh[@]}" "cd '$remote_root' && tmux new-session -d -s '$session' \
      'export CUDA_VISIBLE_DEVICES=0 CUDA_MPS_PIPE_DIRECTORY=/tmp/lnet-mps-gpu3-pipe \
      CUDA_MPS_LOG_DIRECTORY=/tmp/lnet-mps-gpu3-log PYTHONPATH=src OMP_NUM_THREADS=1 \
      MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; for shard in 0 1 2 3; do \
      $remote_python -m lnet.pac_dt_mask_attribution worker --root $result \
      --shard \$shard --device cuda || true; done; $remote_python -m \
      lnet.pac_dt_mask_attribution report --root $result'"
  fi
  sleep 60
done

"${remote_ssh[@]}" "cd '$remote_root' && '$remote_python' -m lnet.pac_dt_mask_attribution report --root '$result' >/dev/null && tar -czf - '$result'" \
  | (cd "$local_root" && tar -xzf -)
touch "$local_root/$result/REMOTE_COLLECTED_B200_GPU3"
"${remote_ssh[@]}" \
  "CUDA_MPS_PIPE_DIRECTORY=/tmp/lnet-mps-gpu3-pipe nvidia-cuda-mps-control <<<quit" \
  >/dev/null 2>&1 || true

if ! tmux has-session -t pac-submission-followups-resume-20260713 2>/dev/null; then
  tmux new-session -d -s pac-submission-followups-resume-20260713 \
    "cd '$local_root' && bash scripts/resume_pro6000_submission_followups.sh '$local_root'"
fi
