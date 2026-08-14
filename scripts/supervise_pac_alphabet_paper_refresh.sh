#!/usr/bin/env bash
set -euo pipefail

project_root=${1:-.}
poll_seconds=${PAC_PAPER_REFRESH_POLL_SECONDS:-60}
partial_refresh_seconds=${PAC_PARTIAL_PAPER_REFRESH_SECONDS:-300}
last_partial_refresh=0

cd "$project_root"

while true; do
  if PYTHONPATH=src python - <<'PY'
from paper.active_final_campaign import replacement_completion_is_verified

raise SystemExit(0 if replacement_completion_is_verified() else 1)
PY
  then
    break
  fi
  now=$(date +%s)
  if (( now - last_partial_refresh >= partial_refresh_seconds )); then
    printf '%s refreshing verified H-compact Q1 plus partial Q2 paper snapshot\n' \
      "$(date --iso-8601=seconds)"
    if ALPHABET_PAPER_CAMPAIGN="$project_root/.omx/results/pac-alphabet-q1q2-final-20260719" \
      ALPHABET_PARTIAL_Q2=1 PYTHONPATH=src python paper/generate_pac_final_figures.py \
      && python scripts/package_alphabet_submission_evidence.py --pending-ok \
      && python paper/build_submission.py \
        --source paper/main.tex \
        --output-dir paper/build/alphabet-partial-q1q2-20260720 \
      && python paper/build_submission.py \
        --source paper/supplement.tex \
        --output-dir paper/build/alphabet-partial-q1q2-supplement-20260720
    then
      printf '%s partial paper snapshot refreshed successfully\n' \
        "$(date --iso-8601=seconds)"
    else
      printf '%s partial paper snapshot refresh failed; retaining the previous verified snapshot\n' \
        "$(date --iso-8601=seconds)" >&2
    fi
    last_partial_refresh=$now
  fi
  printf '%s replacement completion marker is not ready yet\n' "$(date --iso-8601=seconds)"
  sleep "$poll_seconds"
done

backup_dir=$(mktemp -d)
cp -a paper/tables "$backup_dir/tables"
cp -a paper/generated "$backup_dir/generated"
mkdir -p "$backup_dir/Figures"
cp -a paper/Figures/alphabet_architecture_exact_pole_v2.pdf \
  paper/Figures/alphabet_architecture_exact_pole_v2.png \
  paper/Figures/pac_breadth.pdf paper/Figures/pac_breadth.png \
  "$backup_dir/Figures/"
rollback_pending=1
restore_on_failure() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -n ${extracted_text:-} ]]; then
    rm -f "$extracted_text"
  fi
  if [[ $status -ne 0 && $rollback_pending -eq 1 ]]; then
    cp -a "$backup_dir/tables/." paper/tables/
    cp -a "$backup_dir/generated/." paper/generated/
    cp -a "$backup_dir/Figures/." paper/Figures/
    printf '%s paper-facing assets restored after failed final refresh\n' \
      "$(date --iso-8601=seconds)" >&2
  fi
  rm -rf "$backup_dir"
  exit "$status"
}
trap restore_on_failure EXIT INT TERM

printf '%s regenerating paper-facing Q1/Q2 assets\n' "$(date --iso-8601=seconds)"
ALPHABET_PAPER_CAMPAIGN="$project_root/.omx/results/pac-alphabet-q1q2-final-20260719" \
  PYTHONPATH=src python paper/generate_pac_final_figures.py
PYTHONPATH=src python scripts/audit_pac_alphabet_final_bundle.py
PYTHONPATH=src python scripts/audit_pac_q2_execution_partition.py
PYTHONPATH=src python scripts/audit_alphabet_secondary_evidence.py
PYTHONPATH=src python - <<'PY'
from paper.active_final_campaign import REPLACEMENT_CAMPAIGN, active_campaign

if active_campaign() != REPLACEMENT_CAMPAIGN:
    raise RuntimeError("independently audited replacement campaign did not activate")
PY
uv run pytest -q \
  tests/test_alphabet_public_release.py \
  tests/test_paper_submission.py \
  tests/test_paper_active_final_campaign.py \
  tests/test_audit_pac_alphabet_final_bundle.py \
  tests/test_audit_pac_q2_execution_partition.py \
  tests/test_audit_alphabet_secondary_evidence.py \
  tests/test_package_alphabet_submission_evidence.py
node paper/verify_evidence.mjs || {
  printf '%s assets regenerated; manuscript reconciliation still required\n' \
    "$(date --iso-8601=seconds)"
  exit 2
}
python paper/build_submission.py \
  --source paper/main.tex \
  --output-dir paper/build/alphabet-q1q2-final-20260719 \
  --strict-overfull \
  --engine auto
python paper/build_submission.py \
  --source paper/supplement.tex \
  --output-dir paper/build/alphabet-q1q2-final-supplement-20260719 \
  --strict-overfull \
  --engine auto

for pdf in \
  paper/build/alphabet-q1q2-final-20260719/submission.pdf \
  paper/build/alphabet-q1q2-final-supplement-20260719/submission.pdf
do
  extracted_text=$(mktemp)
  pdftotext "$pdf" "$extracted_text"
  if rg -qi 'H-compact|tuned-EFP|bidirectional|forward then backward|InceptionTime|Mini[ _-]?Rocket|PA2WP|EFP16|DiagnosticModel|DiagnosticArchitecture|energy-complete|information-complete' "$extracted_text"; then
    printf '%s retired or private model label found in rendered PDF: %s\n' \
      "$(date --iso-8601=seconds)" "$pdf" >&2
    exit 3
  fi
  rm -f "$extracted_text"
  extracted_text=
done
rm -rf paper/build/alphabet-q1q2-final-page-audit-20260719
python scripts/render_alphabet_pdf_audit.py \
  --main paper/build/alphabet-q1q2-final-20260719/submission.pdf \
  --supplement paper/build/alphabet-q1q2-final-supplement-20260719/submission.pdf \
  --output-dir paper/build/alphabet-q1q2-final-page-audit-20260719 \
  --report paper/generated/alphabet_final_pdf_audit.json
python scripts/package_alphabet_submission_evidence.py
rollback_pending=0
printf '%s replacement paper surface mechanically audited and strictly built; visual review pending\n' \
  "$(date --iso-8601=seconds)"
