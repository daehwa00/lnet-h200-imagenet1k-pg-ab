#!/usr/bin/env python3
"""Generate the RTX 4090 DenseK3 DynamicWrite-R4 runtime."""

from __future__ import annotations

# ruff: noqa: T201
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "kau/alphabet_lm_4090_dynamic_write_r4/campaign.json"
RUNTIME = ROOT / "kau/alphabet_lm_4090_dynamic_write_r4/campaign.runtime.json"
LABEL = "alphabet-dense-k3-dynamic-write-r4"


def _render() -> str:
    raw = MANIFEST.read_bytes()
    manifest = json.loads(raw)
    variant = manifest["architecture"]["variants"][LABEL]
    if (
        manifest.get("schema") != "lnet.kau.alphabet_lm.dynamic_write_r4_2m.v1"
        or manifest["training"]["execution"] != [LABEL]
        or manifest["training"]["target_tokens"] != 2_000_000
        or manifest["training"]["global_sequences"] != 32
        or variant["write_map"] != "dynamic_low_rank"
        or variant["dynamic_write_rank"] != 4
        or set(manifest["parameter_counts"]) != {LABEL}
    ):
        raise RuntimeError("invalid DynamicWrite-R4 campaign")
    campaign_id = manifest["campaign_id"]
    runtime = {
        "schema": "lnet.kau.alphabet_lm.dynamic_write_r4_2m.runtime.v1",
        "campaign_id": campaign_id,
        "campaign_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "output_namespace": manifest["output_namespace"],
        "paper": manifest["paper"],
        "architecture": manifest["architecture"],
        "reused_control": manifest["reused_control"],
        "training": manifest["training"],
        "parameter_counts": manifest["parameter_counts"],
        "decision": manifest["decision"],
        "diagnostics": manifest["diagnostics"],
        "entity": manifest["wandb"]["entity"],
        "project": manifest["wandb"]["project"],
        "group": manifest["wandb"]["group"],
        "runs": {
            LABEL: {
                "id": hashlib.sha256(f"{campaign_id}\0{LABEL}:seed501".encode()).hexdigest()[:16],
                "display_name": "RTX4090-S501-ALPHABET-DenseK3-DynamicWriteR4-2M",
                "tags": [
                    "RTX4090",
                    "FineWeb-Edu",
                    "2M",
                    "DenseK3",
                    "DynamicWrite-R4",
                    "fixed-retention",
                    "seed501",
                    "arXiv-2608.24051",
                ],
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
            print("stale DynamicWrite-R4 runtime", file=sys.stderr)
            return 1
        return 0
    RUNTIME.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
