#!/usr/bin/env python3
"""Generate the immutable frozen modal-probe runtime."""

from __future__ import annotations

# ruff: noqa: T201
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "kau/alphabet_lm_4090_modal_probe/campaign.json"
RUNTIME = ROOT / "kau/alphabet_lm_4090_modal_probe/campaign.runtime.json"


def _render() -> str:
    raw = MANIFEST.read_bytes()
    manifest = json.loads(raw)
    if (
        manifest.get("schema") != "lnet.kau.alphabet_lm.frozen_modal_probe.v1"
        or manifest["training"]["target_tokens"] != 1_000_000
        or manifest["training"]["frozen_backbone"] is not True
        or set(manifest["probes"]) != {"energy", "complex", "shuffled"}
        or manifest["probe_parameters"] != 819_200
    ):
        raise RuntimeError("invalid frozen modal-probe campaign")
    campaign_id = manifest["campaign_id"]
    runtime = {
        "schema": "lnet.kau.alphabet_lm.frozen_modal_probe.runtime.v1",
        "campaign_id": campaign_id,
        "campaign_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "checkpoint": manifest["checkpoint"],
        "probes": manifest["probes"],
        "probe_parameters": manifest["probe_parameters"],
        "training": manifest["training"],
        "entity": manifest["wandb"]["entity"],
        "project": manifest["wandb"]["project"],
        "group": manifest["wandb"]["group"],
        "run": {
            "id": hashlib.sha256(f"{campaign_id}\0seed501".encode()).hexdigest()[:16],
            "display_name": "RTX4090-S501-ALPHABET-FrozenModalProbe-1M",
            "tags": [
                "RTX4090",
                "FineWeb-Edu",
                "frozen-backbone",
                "terminal-pole-state",
                "linear-probe",
                "seed501",
            ],
        },
    }
    return json.dumps(runtime, indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = _render()
    if args.check:
        if not RUNTIME.is_file() or RUNTIME.read_text() != rendered:
            print("stale frozen modal-probe runtime", file=sys.stderr)
            return 1
        return 0
    RUNTIME.write_text(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
