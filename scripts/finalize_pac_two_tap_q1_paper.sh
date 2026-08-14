#!/usr/bin/env bash
set -euo pipefail

project_root=${1:-.}
campaign=.omx/results/pac-two-tap-q1-final-20260720
python=LOCAL_HOME_PLACEHOLDER/miniconda3/bin/python
poll_seconds=${PAC_TWO_TAP_Q1_FINALIZE_POLL_SECONDS:-30}
log=$campaign/paper_finalization.log

cd "$project_root"
mkdir -p "$campaign"

final_is_complete() {
  PYTHONPATH=src "$python" - "$campaign" <<'PY'
import sys
from pathlib import Path

from lnet.pac_two_tap_q1_campaign import status

payload = status(Path(sys.argv[1]))["final"]
raise SystemExit(0 if payload["done"] else 1)
PY
}

while ! final_is_complete; do
  printf '%s waiting for complete Q1-final ledger\n' "$(date --iso-8601=seconds)" >>"$log"
  sleep "$poll_seconds"
done

printf '%s auditing Q1-final ledger\n' "$(date --iso-8601=seconds)" | tee -a "$log"
PYTHONPATH=src "$python" scripts/audit_pac_two_tap_q1_final.py >>"$log" 2>&1

printf '%s regenerating active Q1 paper assets\n' "$(date --iso-8601=seconds)" | tee -a "$log"
PYTHONPATH=src "$python" paper/generate_pac_final_figures.py >>"$log" 2>&1

printf '%s verifying evidence and implementation parity\n' "$(date --iso-8601=seconds)" | tee -a "$log"
node paper/verify_evidence.mjs >>"$log" 2>&1
PYTHONPATH=src "$python" -m pytest -q \
  tests/test_audit_pac_two_tap_q1_final.py \
  tests/test_pac_two_tap_q1_campaign.py \
  tests/test_paper_active_final_campaign.py \
  tests/test_paper_submission.py \
  tests/test_alphabet_public_release.py >>"$log" 2>&1

printf '%s building final main paper and supplement\n' "$(date --iso-8601=seconds)" | tee -a "$log"
"$python" paper/build_submission.py \
  --source paper/main.tex \
  --output-dir paper/build/final-q1-learned-main \
  --engine auto \
  --strict-overfull >>"$log" 2>&1
"$python" paper/build_submission.py \
  --source paper/supplement.tex \
  --output-dir paper/build/final-q1-learned-supplement \
  --engine auto \
  --strict-overfull >>"$log" 2>&1

"$python" - "$campaign" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

campaign = Path(sys.argv[1])
repo = Path.cwd()


def read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


audit_path = campaign / "audit/q1_final_audit.json"
summary_path = repo / "paper/generated/q1_final_summary.json"
main_report_path = repo / "paper/build/final-q1-learned-main/submission-report.json"
supplement_report_path = (
    repo / "paper/build/final-q1-learned-supplement/submission-report.json"
)
audit = read(audit_path)
summary = read(summary_path)
main_report = read(main_report_path)
supplement_report = read(supplement_report_path)
if (
    audit.get("status") != "complete"
    or audit.get("rows") != 1050
    or summary.get("protocol", {}).get("rows") != 1050
    or main_report.get("submission_ready") is not True
    or supplement_report.get("submission_ready") is not True
):
    raise RuntimeError("Q1 paper finalization evidence is incomplete")
payload = {
    "schema": "pac_two_tap_q1_paper_pipeline_complete.v1",
    "status": "complete",
    "chosen_internal_model": "two_tap_h_only",
    "q1_rows": 1050,
    "q2_in_scope": False,
    "audit_sha256": digest(audit_path),
    "q1_summary_sha256": digest(summary_path),
    "main_pdf_sha256": digest(repo / "paper/build/final-q1-learned-main/submission.pdf"),
    "supplement_pdf_sha256": digest(
        repo / "paper/build/final-q1-learned-supplement/submission.pdf"
    ),
    "visual_audit_pending": True,
}
output = campaign / "paper_pipeline_complete.json"
temporary = output.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(output)
PY

printf '%s automated Q1 paper finalization complete; visual audit pending\n' \
  "$(date --iso-8601=seconds)" | tee -a "$log"
