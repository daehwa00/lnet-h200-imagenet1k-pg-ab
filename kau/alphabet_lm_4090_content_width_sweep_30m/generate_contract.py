#!/usr/bin/env python3
"""Generate the J2 content-width R32/R64 30M sweep runtime."""

from __future__ import annotations

# ruff: noqa: T201
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "kau/alphabet_lm_4090_content_width_sweep_30m/campaign.json"
RUNTIME = ROOT / "kau/alphabet_lm_4090_content_width_sweep_30m/campaign.runtime.json"


def _render() -> str:
    raw = MANIFEST.read_bytes()
    manifest = json.loads(raw)
    labels = [
        "j2-p32r32-l19-fromscratch-30m",
        "j2-p32r64-l19-fromscratch-30m",
    ]
    if (
        manifest["training"]["execution"] != labels
        or manifest["training"]["target_tokens"] != 30_000_000
        or manifest["parameter_counts"]
        != {
            labels[0]: 69_942_227,
            labels[1]: 110_644_179,
        }
        or not manifest["diagnostics"]["final_validation_only"]
    ):
        raise RuntimeError("invalid content-width sweep campaign")
    campaign_id = manifest["campaign_id"]
    runs = {}
    for label in labels:
        variant = manifest["architecture"]["variants"][label]
        width = variant["head_width"]
        runs[label] = {
            "id": hashlib.sha256(f"{campaign_id}\0{label}:seed501".encode()).hexdigest()[:16],
            "display_name": f"RTX4090-S501-J2-P32R{width}-L19-30M",
            "tags": [
                "RTX4090",
                "ALPHABET2",
                "ContentRank2",
                f"VectorWidth{width}",
                "FromScratch",
                "30M",
            ],
        }
    runtime = {
        "schema": "lnet.kau.alphabet_lm.alphabet2_complex_vector.runtime.v1",
        "campaign_id": campaign_id,
        "campaign_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "output_namespace": manifest["output_namespace"],
        "architecture": manifest["architecture"],
        "source": manifest["source"],
        "training": manifest["training"],
        "parameter_counts": manifest["parameter_counts"],
        "total_parameter_counts": manifest["parameter_counts"],
        "diagnostics": manifest["diagnostics"],
        "entity": manifest["wandb"]["entity"],
        "project": manifest["wandb"]["project"],
        "group": manifest["wandb"]["group"],
        "runs": runs,
    }
    return json.dumps(runtime, indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = _render()
    if args.check:
        if not RUNTIME.is_file() or RUNTIME.read_text(encoding="utf-8") != rendered:
            print("stale content-width sweep runtime", file=sys.stderr)
            return 1
        return 0
    RUNTIME.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
