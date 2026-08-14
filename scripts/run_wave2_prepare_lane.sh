#!/usr/bin/env bash
set -uo pipefail

dataset=${1:?dataset is required}
base=${WAVE2_BASE:-DATA_ROOT_PLACEHOLDER/alphabet-wave2}
repo=${WAVE2_REPO:-REMOTE_HOME_PLACEHOLDER/lnet-wave2-runtime-20260727}
python=${WAVE2_PYTHON:-REMOTE_HOME_PLACEHOLDER/anaconda3/envs/alphabet/bin/python}
raw=$base/raw
prepared=$base/prepared
logs=$base/logs

retry() {
  until "$@"; do
    sleep 10
  done
}

download_physionet_records() {
  local source_url=$1
  local records_url=$2
  local destination=$3
  local filter=${4:-}
  local records
  local workers=${WAVE2_DOWNLOAD_WORKERS:-24}
  records=$(mktemp)
  curl -fsL "$records_url" >"$records"
  [[ -s $records ]] || return 1
  if [[ -n $filter ]]; then
    grep "^$filter" "$records" >"$records.filtered"
    mv "$records.filtered" "$records"
    [[ -s $records ]] || return 1
  fi
  export source_url destination
  xargs -r -P "$workers" -n 1 bash -c '
    relative=$1
    output=$destination/$relative
    mkdir -p "$(dirname "$output")"
    wget -q -c -O "$output" "$source_url/$relative"
  ' _ <"$records"
  rm -f "$records"
}

download_cpsc() {
  local workers=${WAVE2_DOWNLOAD_WORKERS:-24}
  export destination=$raw/cpsc2018
  seq 1 6877 | xargs -P "$workers" -n 1 bash -c '
    number=$1
    group=$(( (number - 1) / 1000 + 1 ))
    stem=$(printf "A%04d" "$number")
    mkdir -p "$destination/g$group"
    for suffix in hea mat; do
      [[ -s "$destination/g$group/$stem.$suffix" ]] && continue
      wget -q -c -O "$destination/g$group/$stem.$suffix" \
        "https://physionet.org/files/challenge-2020/1.0.2/training/cpsc_2018/g$group/$stem.$suffix"
    done
  ' _
}

case "$dataset" in
  sleepedf-78)
    retry download_physionet_records \
      https://physionet.org/files/sleep-edfx/1.0.0 \
      https://physionet.org/files/sleep-edfx/1.0.0/RECORDS \
      "$raw/sleep-edfx" sleep-cassette/
    ;;
  chb-mit)
    retry download_physionet_records \
      https://physionet.org/files/chbmit/1.0.0 \
      https://physionet.org/files/chbmit/1.0.0/RECORDS \
      "$raw/chbmit"
    ;;
  cpsc-2018)
    retry download_cpsc
    ;;
  isruc-sleep)
    retry "$python" -c \
      'from huggingface_hub import snapshot_download; snapshot_download("braindecode/isruc-sleep", repo_type="dataset", local_dir="DATA_ROOT_PLACEHOLDER/alphabet-wave2/raw/isruc-hf", max_workers=4)'
    ;;
  *)
    printf 'unsupported Wave-2 preparation lane: %s\n' "$dataset" >&2
    exit 2
    ;;
esac

cd "$repo"
PYTHONPATH=src "$python" scripts/extract_wave2_manifests.py \
  --dataset "$dataset" \
  --raw-root "$raw" \
  --manifest "$prepared/$dataset.csv"

(
  flock 9
  PYTHONPATH=src "$python" scripts/prepare_wave2_tasks.py \
    --dataset "$dataset" \
    --manifest "$prepared/$dataset.csv" \
    --output-root "$prepared"
) 9>"$base/preparation.lock"

printf '%s\n' "$dataset" >"$logs/$dataset-ready"
