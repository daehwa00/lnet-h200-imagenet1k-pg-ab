#!/usr/bin/env bash
set -euo pipefail

experiment_root=${EXPERIMENT_ROOT:-/home/daehwa/experiments/alphabet/complex-scan-extreme-opt-20260803}
baseline_root="$experiment_root/baseline_runtime"
candidate_root="$experiment_root/candidate_runtime"
artifact_root=${ARTIFACT_ROOT:-$experiment_root/artifacts}
harness="$experiment_root/scripts/benchmark_complex_scan_common_kau.py"
python_bin=${PYTHON_BIN:-/home/daehwa/anaconda3/envs/alphabet/bin/python}

for runtime_root in "$baseline_root" "$candidate_root"; do
  if [[ ! -f "$runtime_root/src/lnet/complex_scan.py" ]]; then
    echo "missing frozen runtime source: $runtime_root" >&2
    exit 2
  fi
done

mkdir -p "$artifact_root" "$artifact_root/inductor-baseline" "$artifact_root/inductor-candidate"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TORCHINDUCTOR_EMULATE_PRECISION_CASTS=1

fixture="$artifact_root/fixture.pt"
baseline_evidence="$artifact_root/baseline-parity-evidence.pt"
baseline_result="$artifact_root/baseline.json"
candidate_result="$artifact_root/candidate.json"
evaluation_result="$artifact_root/evaluation.json"

PYTHONPATH="$baseline_root/src" "$python_bin" "$harness" fixture \
  --output "$fixture" --batch-size 256 >"$artifact_root/fixture.log" 2>&1

PYTHONPATH="$baseline_root/src" \
TORCHINDUCTOR_CACHE_DIR="$artifact_root/inductor-baseline" \
  "$python_bin" "$harness" benchmark \
    --role baseline --runtime-root "$baseline_root" --fixture "$fixture" \
    --evidence-output "$baseline_evidence" --output "$baseline_result" \
    >"$artifact_root/baseline.log" 2>&1

PYTHONPATH="$candidate_root/src" \
TORCHINDUCTOR_CACHE_DIR="$artifact_root/inductor-candidate" \
  "$python_bin" "$harness" benchmark \
    --role candidate --runtime-root "$candidate_root" --fixture "$fixture" \
    --reference-evidence "$baseline_evidence" --output "$candidate_result" \
    >"$artifact_root/candidate.log" 2>&1

focused_tests=(
  tests/test_benchmark_complex_scan_common_kau.py
  tests/test_complex_scan.py
  tests/test_pac_product_scan_pipeline.py
  tests/test_pac_triton_bidirectional_product_scan.py
  tests/test_pac_triton_product_scan_coarse4.py
)
set +e
(
  cd "$candidate_root"
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  PYTHONPATH="$candidate_root/src:$candidate_root" \
    "$python_bin" -c '
import site
import sys
site.addsitedir(sys.argv[1])
import pytest
raise SystemExit(pytest.main(sys.argv[2:]))
' /home/daehwa/anaconda3/envs/aiprogramming/lib/python3.10/site-packages \
      -q "${focused_tests[@]}"
) >"$artifact_root/focused-tests.log" 2>&1
test_exit=$?
set -e
if (( test_exit == 0 )); then
  tests_status=pass
else
  tests_status=fail
fi

PYTHONPATH="$candidate_root/src" "$python_bin" "$harness" evaluate \
  --baseline "$baseline_result" --candidate "$candidate_result" \
  --output "$evaluation_result" --tests-status "$tests_status" \
  --minimum-speedup 1.10 --maximum-memory-ratio 1.0 \
  --maximum-logits-error 2e-2 --maximum-loss-error 5e-3 \
  --maximum-gradient-relative-rmse 2e-2 \
  --maximum-update-relative-rmse 2e-2
