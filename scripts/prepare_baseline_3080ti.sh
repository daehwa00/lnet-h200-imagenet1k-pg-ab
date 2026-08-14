#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
config_path=${2:-}
action=${1:-help}

if [[ -n $config_path ]]; then
  if [[ ! -f $config_path ]]; then
    printf 'configuration file does not exist: %s\n' "$config_path" >&2
    exit 2
  fi
  # This is an operator-owned shell environment file, not an untrusted data file.
  # shellcheck disable=SC1090
  source "$config_path"
fi

host=${PAC_3080_HOST:-}
port=${PAC_3080_PORT:-22}
key=${PAC_3080_KEY:-}
remote_root=${PAC_3080_REMOTE_ROOT:-}
python_path=${PAC_3080_PYTHON:-python}
gpu=${PAC_3080_GPU:-0}
ucr_root=${PAC_3080_UCR_ROOT:-}
external_root=${PAC_3080_EXTERNAL_ROOT:-}
local_ucr_root=${PAC_3080_LOCAL_UCR_ROOT:-$project_root/.omx/data/ucr}
local_external_root=${PAC_3080_LOCAL_EXTERNAL_ROOT:-$project_root/data/external}
snapshot_parent=$project_root/.omx/source-snapshots
state_root=$project_root/.omx/results/baseline-3080ti-setup-20260725

usage() {
  cat <<'EOF'
Usage:
  scripts/prepare_baseline_3080ti.sh ACTION /path/to/3080ti.env

Actions:
  snapshot       Create a content-addressed local source snapshot. No SSH/GPU use.
  probe          Read host, CPU, RAM, disk, driver, GPU, and Python metadata.
  sync           Upload the immutable source snapshot. Does not copy datasets.
  verify         Verify snapshot hashes, imports, CPU model builds, and data paths.
  stage-data     Explicitly copy UCR and external data without deleting remote files.
  gpu-preflight  Require an idle RTX 3080 Ti, then run a tiny CUDA allocation check.
  calibrate      Require an idle RTX 3080 Ti, then run the S4D/S5/LRU backend screen.
  setup          Run snapshot, probe, sync, and verify. It does not use CUDA.

The script never stops processes, never deletes a remote tree, and never starts a
campaign worker. gpu-preflight and calibrate fail closed when the GPU is occupied.
EOF
}

require_connection_config() {
  local missing=()
  [[ -n $host ]] || missing+=(PAC_3080_HOST)
  [[ -n $key ]] || missing+=(PAC_3080_KEY)
  [[ -n $remote_root ]] || missing+=(PAC_3080_REMOTE_ROOT)
  if ((${#missing[@]})); then
    printf 'missing required configuration: %s\n' "${missing[*]}" >&2
    exit 2
  fi
  if [[ ! -f $key ]]; then
    printf 'SSH key does not exist: %s\n' "$key" >&2
    exit 2
  fi
  if [[ ! $port =~ ^[0-9]+$ || ! $gpu =~ ^[0-9]+$ ]]; then
    printf 'PAC_3080_PORT and PAC_3080_GPU must be non-negative integers\n' >&2
    exit 2
  fi
}

require_data_config() {
  require_connection_config
  local missing=()
  [[ -n $ucr_root ]] || missing+=(PAC_3080_UCR_ROOT)
  [[ -n $external_root ]] || missing+=(PAC_3080_EXTERNAL_ROOT)
  if ((${#missing[@]})); then
    printf 'missing required data configuration: %s\n' "${missing[*]}" >&2
    exit 2
  fi
}

ssh_command() {
  printf 'bash -s --'
  local argument quoted
  for argument in "$@"; do
    printf -v quoted '%q' "$argument"
    printf ' %s' "$quoted"
  done
}

run_remote() {
  local script=$1
  shift
  local command
  command=$(ssh_command "$@")
  ssh -i "$key" -p "$port" -o BatchMode=yes -o ConnectTimeout=10 \
    -o ServerAliveInterval=15 -o ServerAliveCountMax=2 \
    "$host" "$command" <<<"$script"
}

snapshot_manifest() {
  local root=$1
  (
    cd "$root"
    find . -type f \
      ! -name SOURCE_MANIFEST.sha256 \
      ! -path '*/__pycache__/*' \
      ! -name '*.pyc' \
      -print0 |
      LC_ALL=C sort -z |
      xargs -0 sha256sum
  )
}

make_snapshot() {
  mkdir -p "$snapshot_parent" "$state_root"
  local temporary manifest_hash snapshot_name snapshot_path
  temporary=$(mktemp -d "$snapshot_parent/.baseline-3080ti.XXXXXX")
  mkdir -p "$temporary/src" "$temporary/tests" "$temporary/scripts"
  rsync -a --exclude '__pycache__' --exclude '*.pyc' \
    "$project_root/src/" "$temporary/src/"
  rsync -a --exclude '__pycache__' --exclude '*.pyc' \
    "$project_root/tests/" "$temporary/tests/"
  cp "$project_root/pyproject.toml" "$project_root/uv.lock" \
    "$project_root/README.md" "$temporary/"
  cp "$project_root/scripts/prepare_baseline_3080ti.sh" "$temporary/scripts/"
  snapshot_manifest "$temporary" >"$temporary/SOURCE_MANIFEST.sha256"
  manifest_hash=$(sha256sum "$temporary/SOURCE_MANIFEST.sha256" | cut -d' ' -f1)
  snapshot_name=baseline-runtime-3080ti-${manifest_hash:0:12}
  snapshot_path=$snapshot_parent/$snapshot_name

  if [[ -e $snapshot_path ]]; then
    (
      cd "$snapshot_path"
      sha256sum -c SOURCE_MANIFEST.sha256 >/dev/null
    )
    chmod -R u+w "$temporary"
    rm -rf -- "$temporary"
  else
    mv "$temporary" "$snapshot_path"
    chmod -R a-w "$snapshot_path"
  fi
  printf '%s\n' "$snapshot_path" | tee "$state_root/current-snapshot.txt"
}

current_snapshot() {
  if [[ ! -s $state_root/current-snapshot.txt ]]; then
    make_snapshot >/dev/null
  fi
  local snapshot_path
  snapshot_path=$(<"$state_root/current-snapshot.txt")
  if [[ ! -d $snapshot_path ]]; then
    printf 'recorded snapshot does not exist: %s\n' "$snapshot_path" >&2
    exit 1
  fi
  (
    cd "$snapshot_path"
    sha256sum -c SOURCE_MANIFEST.sha256 >/dev/null
  )
  printf '%s\n' "$snapshot_path"
}

probe_remote() {
  require_connection_config
  run_remote "$(cat <<'REMOTE'
set -eu
python_path=$1
remote_root=$2
printf 'host=%s\n' "$(hostname -f 2>/dev/null || hostname)"
printf 'kernel=%s\n' "$(uname -srmo)"
printf 'cpu_threads=%s\n' "$(getconf _NPROCESSORS_ONLN)"
awk '/MemTotal:/ {printf "ram_kib=%s\n", $2}' /proc/meminfo
df -Pk "$HOME" | awk 'NR == 2 {printf "home_free_kib=%s\n", $4}'
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu,pstate,driver_version \
    --format=csv,noheader,nounits
  printf 'compute_processes:\n'
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv,noheader,nounits 2>/dev/null || true
else
  printf 'nvidia_smi=missing\n'
fi
if command -v "$python_path" >/dev/null 2>&1 || [[ -x $python_path ]]; then
  "$python_path" - <<'PY'
import platform
import torch
print(f"python={platform.python_version()}")
print(f"torch={torch.__version__}")
print(f"torch_cuda_build={torch.version.cuda}")
PY
else
  printf 'python=missing:%s\n' "$python_path"
fi
if [[ -e $remote_root ]]; then
  du -sk "$remote_root" 2>/dev/null | awk '{printf "remote_root_kib=%s\n", $1}'
else
  printf 'remote_root=absent\n'
fi
REMOTE
)" "$python_path" "$remote_root" | tee "$state_root/probe.txt"
}

sync_snapshot() {
  require_connection_config
  local snapshot_path snapshot_name remote_snapshot temporary
  snapshot_path=$(current_snapshot)
  snapshot_name=$(basename "$snapshot_path")
  remote_snapshot=$remote_root/.omx/source-snapshots/$snapshot_name
  temporary=$remote_root/.omx/source-snapshots/.upload-$snapshot_name-$$

  run_remote "$(cat <<'REMOTE'
set -eu
remote_root=$1
remote_snapshot=$2
temporary=$3
mkdir -p "$remote_root/.omx/source-snapshots"
if [[ -e $remote_snapshot ]]; then
  cd "$remote_snapshot"
  sha256sum -c SOURCE_MANIFEST.sha256 >/dev/null
  printf 'remote_snapshot=%s status=already_verified\n' "$remote_snapshot"
  exit 0
fi
if [[ -e $temporary ]]; then
  printf 'temporary upload path already exists: %s\n' "$temporary" >&2
  exit 1
fi
mkdir -p "$temporary"
printf 'remote_snapshot=%s status=ready_for_upload\n' "$remote_snapshot"
REMOTE
)" "$remote_root" "$remote_snapshot" "$temporary"

  if run_remote 'test -d "$1"' "$remote_snapshot"; then
    printf '%s\n' "$remote_snapshot" | tee "$state_root/remote-snapshot.txt"
    return
  fi

  rsync -az -e "ssh -i $key -p $port -o BatchMode=yes -o ConnectTimeout=10" \
    "$snapshot_path/" "$host:$temporary/"
  run_remote "$(cat <<'REMOTE'
set -eu
remote_snapshot=$1
temporary=$2
cd "$temporary"
sha256sum -c SOURCE_MANIFEST.sha256 >/dev/null
chmod -R a-w "$temporary"
if [[ -e $remote_snapshot ]]; then
  printf 'remote snapshot appeared concurrently: %s\n' "$remote_snapshot" >&2
  exit 1
fi
mv "$temporary" "$remote_snapshot"
printf 'remote_snapshot=%s status=uploaded_verified_immutable\n' "$remote_snapshot"
REMOTE
)" "$remote_snapshot" "$temporary" | tee "$state_root/sync.txt"
  printf '%s\n' "$remote_snapshot" | tee "$state_root/remote-snapshot.txt"
}

remote_snapshot_path() {
  local snapshot_path
  snapshot_path=$(current_snapshot)
  printf '%s/.omx/source-snapshots/%s\n' "$remote_root" "$(basename "$snapshot_path")"
}

verify_remote() {
  require_data_config
  local remote_snapshot
  remote_snapshot=$(remote_snapshot_path)
  run_remote "$(cat <<'REMOTE'
set -eu
snapshot=$1
python_path=$2
ucr_root=$3
external_root=$4
test -d "$snapshot"
cd "$snapshot"
sha256sum -c SOURCE_MANIFEST.sha256 >/dev/null
test -d "$ucr_root/Wafer"
test -f "$external_root/selection-only/sequential-mnist.pt"
test -f "$external_root/selection-only/speech-commands.pt"
printf 'ucr_kib=%s\n' "$(du -sk "$ucr_root" | awk '{print $1}')"
printf 'external_kib=%s\n' "$(du -sk "$external_root" | awk '{print $1}')"
PYTHONSAFEPATH=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$snapshot/src" \
  "$python_path" - <<'PY'
from lnet.pac_baseline_backend_screen import FAMILIES, ScreenConfig, build_screen_model
import torch

for family in FAMILIES:
    model = build_screen_model(
        family,
        length=12,
        batch_size=3,
        config=ScreenConfig(width=16, class_count=5, input_dim=2),
        static_transformer_positions=family == "transformer",
    )
    logits = model(torch.randn(3, 12, 2))
    assert logits.shape == (3, 5)
    assert torch.isfinite(logits).all()
print(f"cpu_model_builds={len(FAMILIES)}")
print("snapshot_hashes=verified")
print("data_contract=verified")
PY
REMOTE
)" "$remote_snapshot" "$python_path" "$ucr_root" "$external_root" |
    tee "$state_root/verify.txt"
}

stage_data() {
  require_data_config
  if [[ ! -d $local_ucr_root || ! -d $local_external_root ]]; then
    printf 'local data roots are missing: %s %s\n' \
      "$local_ucr_root" "$local_external_root" >&2
    exit 1
  fi
  run_remote 'set -eu; mkdir -p "$1" "$2"' "$ucr_root" "$external_root"
  rsync -a --partial --info=progress2 \
    -e "ssh -i $key -p $port -o BatchMode=yes -o ConnectTimeout=10" \
    "$local_ucr_root/" "$host:$ucr_root/"
  rsync -a --partial --info=progress2 \
    -e "ssh -i $key -p $port -o BatchMode=yes -o ConnectTimeout=10" \
    "$local_external_root/" "$host:$external_root/"
  printf 'data staged without remote deletion\n' | tee "$state_root/data-sync.txt"
}

remote_idle_gate() {
  local samples=${PAC_3080_IDLE_SAMPLES:-5}
  local interval=${PAC_3080_IDLE_INTERVAL_SECONDS:-10}
  local max_util=${PAC_3080_MAX_IDLE_UTILIZATION:-5}
  local max_memory=${PAC_3080_MAX_IDLE_MEMORY_MIB:-1024}
  run_remote "$(cat <<'REMOTE'
set -eu
gpu=$1
samples=$2
interval=$3
max_util=$4
max_memory=$5
case $gpu:$samples:$interval:$max_util:$max_memory in
  *[!0-9:]*|'') printf 'idle-gate values must be integers\n' >&2; exit 2 ;;
esac
i=1
while [[ $i -le $samples ]]; do
  processes=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | sed '/^[[:space:]]*$/d' || true)
  if [[ -n $processes ]]; then
    printf 'idle_gate=failed compute_processes=%s\n' "$processes" >&2
    exit 1
  fi
  values=$(nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader,nounits -i "$gpu")
  name=${values%%,*}
  rest=${values#*,}
  memory=${rest%%,*}
  util=${rest##*,}
  memory=$(printf '%s' "$memory" | tr -d ' ')
  util=$(printf '%s' "$util" | tr -d ' ')
  case $name in
    *"RTX 3080 Ti"*) ;;
    *) printf 'idle_gate=failed unexpected_gpu=%s\n' "$name" >&2; exit 1 ;;
  esac
  if [[ $memory -gt $max_memory || $util -gt $max_util ]]; then
    printf 'idle_gate=failed sample=%s memory_mib=%s util=%s\n' "$i" "$memory" "$util" >&2
    exit 1
  fi
  printf 'idle_gate_sample=%s/%s memory_mib=%s util=%s\n' "$i" "$samples" "$memory" "$util"
  if [[ $i -lt $samples ]]; then sleep "$interval"; fi
  i=$((i + 1))
done
printf 'idle_gate=passed\n'
REMOTE
)" "$gpu" "$samples" "$interval" "$max_util" "$max_memory"
}

gpu_preflight() {
  require_connection_config
  local remote_snapshot
  remote_snapshot=$(remote_snapshot_path)
  remote_idle_gate | tee "$state_root/gpu-idle-gate.txt"
  run_remote "$(cat <<'REMOTE'
set -eu
snapshot=$1
python_path=$2
gpu=$3
cd "$snapshot"
CUDA_VISIBLE_DEVICES="$gpu" PYTHONSAFEPATH=1 PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$snapshot/src" "$python_path" - <<'PY'
import json
import torch

assert torch.cuda.is_available()
properties = torch.cuda.get_device_properties(0)
assert "RTX 3080 Ti" in properties.name, properties.name
x = torch.arange(4096, device="cuda", dtype=torch.float32).reshape(64, 64)
y = x @ x.T
torch.cuda.synchronize()
assert torch.isfinite(y).all()
print(json.dumps({
    "device": properties.name,
    "compute_capability": [properties.major, properties.minor],
    "total_memory_bytes": properties.total_memory,
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "allocation_check": "passed",
}, sort_keys=True))
PY
REMOTE
)" "$remote_snapshot" "$python_path" "$gpu" | tee "$state_root/gpu-preflight.json"
}

calibrate_remote() {
  require_connection_config
  local remote_snapshot remote_output
  remote_snapshot=$(remote_snapshot_path)
  remote_output=$remote_root/.omx/results/baseline-runtime-screen-3080ti-20260725
  remote_idle_gate | tee "$state_root/calibrate-idle-gate.txt"
  run_remote "$(cat <<'REMOTE'
set -eu
snapshot=$1
python_path=$2
gpu=$3
output_root=$4
mkdir -p "$output_root"
cd "$snapshot"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES="$gpu" PYTHONSAFEPATH=1 PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$snapshot/src" "$python_path" -m lnet.pac_baseline_backend_screen \
  --families s4d,s5,lru \
  --backends eager_fused,cuda_graph_full_step \
  --lengths 128,1024 \
  --batch-size 64 \
  --width 64 \
  --parity-steps 75 \
  --warmups 5 \
  --groups 5 \
  --iterations-per-group 20 \
  --graph-warmups 5 \
  --output "$output_root/backend-screen.json" \
  >"$output_root/backend-screen.stdout.json"
printf 'calibration_output=%s\n' "$output_root/backend-screen.json"
REMOTE
)" "$remote_snapshot" "$python_path" "$gpu" "$remote_output" |
    tee "$state_root/calibrate.txt"
}

case $action in
  help|-h|--help)
    usage
    ;;
  snapshot)
    make_snapshot
    ;;
  probe)
    mkdir -p "$state_root"
    probe_remote
    ;;
  sync)
    mkdir -p "$state_root"
    sync_snapshot
    ;;
  verify)
    mkdir -p "$state_root"
    verify_remote
    ;;
  stage-data)
    mkdir -p "$state_root"
    stage_data
    ;;
  gpu-preflight)
    mkdir -p "$state_root"
    gpu_preflight
    ;;
  calibrate)
    mkdir -p "$state_root"
    calibrate_remote
    ;;
  setup)
    mkdir -p "$state_root"
    make_snapshot
    probe_remote
    sync_snapshot
    verify_remote
    ;;
  *)
    printf 'unknown action: %s\n' "$action" >&2
    usage >&2
    exit 2
    ;;
esac
