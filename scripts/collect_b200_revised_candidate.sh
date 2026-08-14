#!/usr/bin/env bash
set -euo pipefail

root=${1:-.}
remote_root=${PAC_B200_ROOT:-REMOTE_HOME_PLACEHOLDER/lnet-pac-external-20260710}
remote_host=${PAC_B200_HOST:-remote_user@REMOTE_HOST_PLACEHOLDER}
ssh_key=${PAC_B200_KEY:?PAC_B200_KEY is required}
jump_key=${PAC_B200_JUMP_KEY:?PAC_B200_JUMP_KEY is required}
proxy_command="ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i $jump_key -p 8589 secondary_host@REMOTE_HOST_PLACEHOLDER -W %h:%p"

cd "$root"
candidate=${PAC_REVISED_CANDIDATE_ROOT:-.omx/results/pac-tf-revised-untied-candidate-20260711}
shard_prefix=${PAC_REVISED_SHARD_PREFIX:-.omx/results/pac-tf-revised-b200-shard}
shard_suffix=${PAC_REVISED_SHARD_SUFFIX:--20260711}

remote_ssh=(
  ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
  -o "ProxyCommand=$proxy_command"
  -i "$ssh_key" -p 40701 "$remote_host"
)

while true; do
  completed=$("${remote_ssh[@]}" "cd '$remote_root' && for i in 0 1 2 3; do test -f ${shard_prefix}\${i}${shard_suffix}/COMPLETE && printf 1 || printf 0; done")
  [[ "$completed" == "1111" ]] && break
  sleep 30
done

remote_shards="${shard_prefix}0${shard_suffix} ${shard_prefix}1${shard_suffix} ${shard_prefix}2${shard_suffix} ${shard_prefix}3${shard_suffix}"
"${remote_ssh[@]}" "cd '$remote_root' && tar -czf - $remote_shards" | tar -xzf -

mkdir -p "$candidate/results"
result="$candidate/results/revised_candidate.csv"
first=true
: >"$result"
for i in 0 1 2 3; do
  shard="${shard_prefix}${i}${shard_suffix}"
  csv="$shard/results/revised_candidate.csv"
  if $first; then
    cat "$csv" >>"$result"
    first=false
  else
    tail -n +2 "$csv" >>"$result"
  fi
  cat "$shard/queue_state.jsonl" >>"$candidate/queue_state.jsonl"
done

PYTHONPATH="$root/src" python -m lnet.pac_revised_candidate_cli \
  --stage report --output-root "$candidate"

if grep -q '"status": "complete"' "$candidate/reports/revised_candidate.json"; then
  printf '%s\n' "complete at $(date -Is)" >"$candidate/COMPLETE"
else
  printf '%s\n' "incomplete at $(date -Is)" >"$candidate/FAILED"
  exit 1
fi
