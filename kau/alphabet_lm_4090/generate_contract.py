#!/usr/bin/env python3
"""Generate the immutable RTX 4090 pole-initialization campaign runtime."""

from __future__ import annotations

# ruff: noqa: T201
# pyright: reportExplicitAny=false
import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "kau/alphabet_lm_4090/campaign.json"
RUNTIME = ROOT / "kau/alphabet_lm_4090/campaign.runtime.json"


def _run_id(campaign_id: str, label: str) -> str:
    return hashlib.sha256(f"{campaign_id}\0{label}".encode()).hexdigest()[:16]


def _render() -> str:
    raw = MANIFEST.read_bytes()
    manifest: dict[str, Any] = json.loads(raw)
    if (
        manifest.get("schema") != "lnet.kau.alphabet_lm.pole_init_10m.v1"
        or manifest.get("campaign_id") != "kau-alphabet-lm-pole-init-10m-s501-v1"
        or manifest["training"]["execution"]
        != ["alphabet-legacy", "alphabet-palette", "mamba"]
        or manifest["training"]["scan_fp32"] is not True
        or manifest["pole_palette"]["half_life_anchors"]
        != [2,4,8,16,32,64,128,256,512,1024,2048,4096,8192]
        or manifest["pole_palette"]["decay_dominant_fraction"] != 0.5
        or manifest["paper"]["arxiv_id"] != "2608.24051"
        or manifest["paper"]["url"] != "https://arxiv.org/abs/2608.24051"
    ):
        raise RuntimeError("invalid KAU ALPHABET-LM campaign")
    campaign_id = manifest["campaign_id"]
    names = {
        "alphabet-legacy": "RTX4090-S501-ALPHABET-LegacyInit-10M",
        "alphabet-palette": "RTX4090-S501-ALPHABET-LifetimePalette-10M",
        "mamba": "RTX4090-S501-OfficialMamba-10M",
    }
    runtime = {
        "schema": "lnet.kau.alphabet_lm.pole_init_10m.runtime.v1",
        "campaign_id": campaign_id,
        "campaign_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "output_namespace": manifest["output_namespace"],
        "paper": manifest["paper"],
        "dataset": manifest["dataset"],
        "training": manifest["training"],
        "pole_palette": manifest["pole_palette"],
        "environment": manifest["environment"],
        "entity": manifest["wandb"]["entity"],
        "project": manifest["wandb"]["project"],
        "group": manifest["wandb"]["group"],
        "runs": {
            label: {
                "id": _run_id(campaign_id, f"{label}:seed501"),
                "display_name": names[label],
                "tags": [
                    "RTX4090", "FineWeb-Edu", "10M", label,
                    "seed501", "arXiv-2608.24051",
                ],
            }
            for label in manifest["training"]["execution"]
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
            print("stale KAU ALPHABET-LM runtime", file=sys.stderr)
            return 1
        return 0
    RUNTIME.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
