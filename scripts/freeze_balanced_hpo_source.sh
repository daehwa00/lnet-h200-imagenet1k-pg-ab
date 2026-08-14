#!/usr/bin/env bash
set -euo pipefail

project_root=${1:-.}
snapshot_parent=${2:-$project_root/.omx/source-snapshots}
python_path=${PAC_BALANCED_HPO_FREEZE_PYTHON:-LOCAL_HOME_PLACEHOLDER/miniconda3/bin/python}

cd "$project_root"
mkdir -p "$snapshot_parent"
staging=$(mktemp -d "$snapshot_parent/.balanced-hpo-staging.XXXXXX")
cleanup() {
  if [[ -d $staging ]]; then
    rm -rf -- "$staging"
  fi
}
trap cleanup EXIT

rsync -a \
  --exclude=__pycache__ \
  --exclude='*.pyc' \
  "$project_root/src" \
  "$project_root/scripts" \
  "$project_root/optimization" \
  "$project_root/pyproject.toml" \
  "$staging/"

if [[ -f $project_root/uv.lock ]]; then
  rsync -a "$project_root/uv.lock" "$staging/"
fi

digest=$(
  cd "$staging"
  PYTHONDONTWRITEBYTECODE=1 PYTHONSAFEPATH=1 PYTHONPATH="$staging/src:$staging" \
    "$python_path" -c \
    'from lnet.pac_balanced_hpo_campaign import code_sha256; print(code_sha256())'
)
destination=$snapshot_parent/balanced-hpo-${digest:0:16}

if [[ -e $destination ]]; then
  existing=$(
    cd "$destination"
    PYTHONDONTWRITEBYTECODE=1 PYTHONSAFEPATH=1 \
      PYTHONPATH="$destination/src:$destination" "$python_path" -c \
      'from lnet.pac_balanced_hpo_campaign import code_sha256; print(code_sha256())'
  )
  [[ $existing == "$digest" ]] || {
    printf 'existing snapshot hash mismatch: %s != %s\n' "$existing" "$digest" >&2
    exit 1
  }
else
  mv "$staging" "$destination"
  chmod -R a-w "$destination"
fi

printf '{"snapshot":"%s","code_sha256":"%s"}\n' "$destination" "$digest"
