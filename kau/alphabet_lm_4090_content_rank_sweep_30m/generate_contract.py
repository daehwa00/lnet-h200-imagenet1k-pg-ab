#!/usr/bin/env python3
"""Generate the J2/J4 content-aligned 30M sweep runtime."""

from __future__ import annotations

# ruff: noqa: T201
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "kau/alphabet_lm_4090_content_rank_sweep_30m/campaign.json"
RUNTIME = ROOT / "kau/alphabet_lm_4090_content_rank_sweep_30m/campaign.runtime.json"


def _render() -> str:
    raw = MANIFEST.read_bytes()
    manifest = json.loads(raw)
    labels = [
        "content-aligned-j2-p32r16-l19-fromscratch-30m",
        "content-aligned-j4-p32r16-l19-fromscratch-30m",
    ]
    if (
        manifest["training"]["execution"] != labels
        or manifest["training"]["target_tokens"] != 30_000_000
        or manifest["parameter_counts"]
        != {
            labels[0]: 49_591_251,
            labels[1]: 50_019_283,
        }
        or not manifest["diagnostics"]["final_validation_only"]
    ):
        raise RuntimeError("invalid content-rank sweep campaign")
    campaign_id = manifest["campaign_id"]
    runs = {}
    for label in labels:
        rank = manifest["architecture"]["variants"][label]["content_rank"]
        runs[label] = {
            "id": hashlib.sha256(f"{campaign_id}\0{label}:seed501".encode()).hexdigest()[:16],
            "display_name": f"RTX4090-S501-Content-Aligned-J{rank}-P32R16-L19-30M",
            "tags": [
                "RTX4090",
                "ALPHABET2",
                f"ContentRank{rank}",
                "VectorPole",
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
            print("stale content-rank sweep runtime", file=sys.stderr)
            return 1
        return 0
    RUNTIME.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
