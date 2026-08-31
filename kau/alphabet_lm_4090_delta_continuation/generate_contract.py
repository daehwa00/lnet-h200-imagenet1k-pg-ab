#!/usr/bin/env python3
"""Generate the matched static/dynamic frozen-pole continuation runtime."""

from __future__ import annotations

# ruff: noqa: T201
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "kau/alphabet_lm_4090_delta_continuation/campaign.json"
RUNTIME = ROOT / "kau/alphabet_lm_4090_delta_continuation/campaign.runtime.json"


def _render() -> str:
    raw = MANIFEST.read_bytes()
    manifest = json.loads(raw)
    labels = [
        "dense-frozen-pole-static-continuation-4m",
        "dense-frozen-pole-dynamic-delta-continuation-4m",
    ]
    if (
        manifest["training"]["execution"] != labels
        or manifest["training"]["target_tokens"] != 4_000_000
        or manifest["parameter_counts"]
        != {labels[0]: 64_104_211, labels[1]: 64_439_827}
        or manifest["total_parameter_counts"]
        != {labels[0]: 64_105_427, labels[1]: 64_441_043}
        or not manifest["diagnostics"]["matched_source_batcher_and_rng"]
    ):
        raise RuntimeError("invalid delta-continuation campaign")
    campaign_id = manifest["campaign_id"]
    runs = {}
    for label in labels:
        dynamic = "dynamic" in label
        runs[label] = {
            "id": hashlib.sha256(f"{campaign_id}\0{label}".encode()).hexdigest()[:16],
            "display_name": (
                "RTX4090-S501-Dense100M-FrozenPole-"
                f"{'DynamicDelta' if dynamic else 'Static'}-Continuation-4M"
            ),
            "tags": [
                "RTX4090",
                "ALPHABET2",
                "Dense100MContinuation",
                "FrozenPoles",
                "DynamicDelta" if dynamic else "StaticControl",
                "4M",
            ],
        }
    runtime = {
        "schema": "lnet.kau.alphabet_lm.alphabet2_complex_vector.runtime.v1",
        "campaign_id": campaign_id,
        "campaign_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "output_namespace": manifest["output_namespace"],
        "architecture": manifest["architecture"],
        "training": manifest["training"],
        "parameter_counts": manifest["parameter_counts"],
        "total_parameter_counts": manifest["total_parameter_counts"],
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
            print("stale delta-continuation runtime", file=sys.stderr)
            return 1
        return 0
    RUNTIME.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
