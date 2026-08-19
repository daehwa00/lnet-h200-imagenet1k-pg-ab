from __future__ import annotations

# ruff: noqa: S607
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_generated_campaign_contract_has_no_drift() -> None:
    subprocess.run(
        ["python", "h200/generate_campaign.py", "--check"],
        cwd=ROOT,
        check=True,
    )


def test_runtime_is_bound_to_exact_manifest_bytes() -> None:
    manifest_path = ROOT / "h200" / "campaign.json"
    runtime = json.loads((ROOT / "h200" / "campaign.runtime.json").read_text())
    assert runtime["manifest_sha256"] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert runtime["schema_version"] == 3
    assert runtime["console"] == "off"


def test_canary_and_production_runs_are_distinct() -> None:
    runtime = json.loads((ROOT / "h200" / "campaign.runtime.json").read_text())
    run_ids = {runtime["pg_run_id"], runtime["no_pg_run_id"], runtime["canary_run_id"]}
    assert len(run_ids) == 3


def test_relay_contract_excludes_console_and_secrets() -> None:
    manifest = json.loads((ROOT / "h200" / "campaign.json").read_text())
    assert "output.log" not in manifest["protocol"]["run_files"]
    assert "output.log" not in manifest["protocol"]["stream_files"]
    wrangler = (ROOT / "cloudflare" / "relay" / "wrangler.jsonc").read_text()
    assert "ALLOWED_EGRESS_IPS" not in wrangler
    assert "WANDB_API_KEY" not in wrangler
    assert "203.253." not in wrangler
