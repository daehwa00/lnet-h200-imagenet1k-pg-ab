#!/usr/bin/env python3
"""Generate the RTX 4090 fixed-pole dynamic-routing screen runtime."""

from __future__ import annotations

# ruff: noqa: T201
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "kau/alphabet_lm_4090_routing/campaign.json"
RUNTIME = ROOT / "kau/alphabet_lm_4090_routing/campaign.runtime.json"


def _render() -> str:
    raw = MANIFEST.read_bytes()
    manifest = json.loads(raw)
    labels = ["alphabet-dynamic-write", "alphabet-dynamic-write-read"]
    if (
        manifest.get("schema") != "lnet.kau.alphabet_lm.dynamic_routing_2m.v1"
        or manifest["training"]["execution"] != labels
        or manifest["training"]["target_tokens"] != 2_000_000
        or manifest["architecture"]["fixed_lifetimes"] is not True
        or manifest["architecture"]["gate_initialization"] != "neutral_exact_one"
    ):
        raise RuntimeError("invalid dynamic-routing campaign")
    campaign_id = manifest["campaign_id"]
    names = {
        "alphabet-dynamic-write": "RTX4090-S501-ALPHABET-DynamicWrite-2M",
        "alphabet-dynamic-write-read": "RTX4090-S501-ALPHABET-DynamicWriteRead-2M",
    }
    runtime = {
        "schema": "lnet.kau.alphabet_lm.dynamic_routing_2m.runtime.v1",
        "campaign_id": campaign_id,
        "campaign_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "output_namespace": manifest["output_namespace"],
        "paper": manifest["paper"],
        "architecture": manifest["architecture"],
        "training": manifest["training"],
        "parameter_counts": manifest["parameter_counts"],
        "entity": manifest["wandb"]["entity"],
        "project": manifest["wandb"]["project"],
        "group": manifest["wandb"]["group"],
        "runs": {
            label: {
                "id": hashlib.sha256(
                    f"{campaign_id}\0{label}:seed501".encode()
                ).hexdigest()[:16],
                "display_name": names[label],
                "tags": [
                    "RTX4090", "FineWeb-Edu", "2M", label,
                    "fixed-lifetime", "seed501", "arXiv-2608.24051",
                ],
            }
            for label in labels
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
            print("stale dynamic-routing runtime", file=sys.stderr)
            return 1
        return 0
    RUNTIME.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
