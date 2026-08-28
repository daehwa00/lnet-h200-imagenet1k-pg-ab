#!/usr/bin/env python3
"""Generate the Token-Q 10M extension runtime."""

from __future__ import annotations

# ruff: noqa: T201
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "kau/alphabet_lm_4090_alphabet2_query_10m/campaign.json"
RUNTIME = ROOT / "kau/alphabet_lm_4090_alphabet2_query_10m/campaign.runtime.json"
LABEL = "alphabet2-token-q-10m"


def _render() -> str:
    raw = MANIFEST.read_bytes()
    manifest = json.loads(raw)
    variant = manifest["architecture"]["variants"][LABEL]
    if (
        manifest.get("schema") != "lnet.kau.alphabet_lm.alphabet2_query_10m.v1"
        or manifest["training"]["execution"] != [LABEL]
        or manifest["training"]["target_tokens"] != 10_000_000
        or variant["slow_cnn_pole_query"] != "token"
        or variant["slow_cnn_pole_use_recurrence"] is not True
    ):
        raise RuntimeError("invalid Token-Q 10M campaign")
    campaign_id = manifest["campaign_id"]
    runtime = {
        "schema": "lnet.kau.alphabet_lm.alphabet2_query_10m.runtime.v1",
        "campaign_id": campaign_id,
        "campaign_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "output_namespace": manifest["output_namespace"],
        "architecture": manifest["architecture"],
        "sources": manifest["sources"],
        "training": manifest["training"],
        "parameter_counts": manifest["parameter_counts"],
        "total_parameter_counts": manifest["total_parameter_counts"],
        "diagnostics": manifest["diagnostics"],
        "entity": manifest["wandb"]["entity"],
        "project": manifest["wandb"]["project"],
        "group": manifest["wandb"]["group"],
        "runs": {
            LABEL: {
                "id": hashlib.sha256(f"{campaign_id}\0{LABEL}:seed501".encode()).hexdigest()[:16],
                "display_name": "RTX4090-S501-ALPHABET2-TokenQ-10M",
                "tags": ["RTX4090", "ALPHABET2", "TokenQ", "10M", "seed501"],
            }
        },
    }
    return json.dumps(runtime, indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = _render()
    if args.check:
        if not RUNTIME.is_file() or RUNTIME.read_text(encoding="utf-8") != rendered:
            print("stale Token-Q 10M runtime", file=sys.stderr)
            return 1
        return 0
    RUNTIME.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
