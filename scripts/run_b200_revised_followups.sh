#!/usr/bin/env bash
set -euo pipefail

project_root=${1:?project root is required}
python_bin=REMOTE_ROOT_PLACEHOLDER/miniconda3/envs/ASA_test/bin/python
protocol=.omx/protocols/pac_tf_confirmatory_20260711.json
official=.omx/results/pac-tf-revised-confirmatory-20260712
evidence=.omx/results/pac-tf-revised-evidence-20260712
p1p2=.omx/results/pac-tf-revised-p1p2-20260712
selection="$official/reports/confirmatory_baseline_selection.json"
state=.omx/results/pac-tf-revised-followups-20260712
mkdir -p "$project_root/$state" "$project_root/.omx/logs/pac-tf-revised-followups-20260712"
cd "$project_root"

run_module() {
  local module=$1
  shift
  PYTHONPATH="$project_root/src" "$python_bin" -m "$module" "$@"
}

run_official_pass() {
  run_module lnet.pac_revised_confirmatory_cli \
    --stage workers --output-root "$official" --device cuda --workers 8 --total-slots 16
}

official_done() {
  "$python_bin" - "$official" <<'PY'
import csv, sys
from pathlib import Path
root = Path(sys.argv[1])
path = root / "results" / "low_data_recommended_real.csv"
rows = {}
if path.exists():
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows[row.get("job_key", "")] = row.get("status", "")
print(sum(status == "done" for status in rows.values()))
PY
}

run_p1p2_package() {
  local package=$1
  for _attempt in 1 2; do
    run_module lnet.pac_tf_p1p2_cli \
      --stage workers --package "$package" --model pac_tf \
      --output-root "$p1p2" --protocol-path "$protocol" \
      --selection-path "$selection" --unseen-root "$official" \
      --device cuda --workers 8 --total-slots 16
  done
}

(
  export CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  while pgrep -f 'lnet.pac_revised_confirmatory_cli --stage workers.*pac-tf-revised-confirmatory-20260712' >/dev/null; do
    sleep 30
  done
  for _attempt in 1 2 3; do
    [[ $(official_done) == 90 ]] && break
    run_official_pass
  done
  [[ $(official_done) == 90 ]]
  touch "$official/COMPLETE"
  run_module lnet.pac_tf_p1p2_cli \
    --stage enqueue --model pac_tf --output-root "$p1p2" \
    --protocol-path "$protocol" --selection-path "$selection" \
    --unseen-root "$official" --device cuda
  touch "$p1p2/ENQUEUED"
  run_p1p2_package low_data
  run_p1p2_package real_diagnostics
  touch "$state/gpu4.COMPLETE"
) >"$project_root/.omx/logs/pac-tf-revised-followups-20260712/gpu4.log" 2>&1 &
gpu4=$!

(
  export CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  while [[ ! -f "$evidence/interpretability.COMPLETE" || ! -f "$p1p2/ENQUEUED" ]]; do
    sleep 30
  done
  run_p1p2_package synthetic_ood
  run_p1p2_package real_domain_ood
  run_p1p2_package efficiency
  touch "$state/gpu7.COMPLETE"
) >"$project_root/.omx/logs/pac-tf-revised-followups-20260712/gpu7.log" 2>&1 &
gpu7=$!

echo "$gpu4 gpu4" >"$state/launches.tsv"
echo "$gpu7 gpu7" >>"$state/launches.tsv"
