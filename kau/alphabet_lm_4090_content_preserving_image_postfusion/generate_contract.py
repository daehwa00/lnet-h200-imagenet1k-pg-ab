#!/usr/bin/env python3
"""Generate the content-preserving ALPHABET-2 100M runtime contract."""

from __future__ import annotations

# ruff: noqa: T201
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "kau/alphabet_lm_4090_content_preserving_image_postfusion/campaign.json"
RUNTIME = (
    ROOT
    / "kau/alphabet_lm_4090_content_preserving_image_postfusion/campaign.runtime.json"
)
LABEL = "hybrid-content-h4p4k64-dense-p32r8-l19-fromscratch-30m"


def _render() -> str:
    raw = MANIFEST.read_bytes()
    manifest = json.loads(raw)
    variant = manifest["architecture"]["variants"][LABEL]
    if (
        manifest["training"]["execution"] != [LABEL]
        or manifest["training"]["target_tokens"] != 30_000_000
        or variant["content_preserving_heads"] != 4
        or variant["content_preserving_poles_per_head"] != 4
        or variant["content_preserving_width_per_head"] != 64
        or variant["content_feature_width"] != 256
        or variant["hybrid_dense_poles"] != 32
        or variant["hybrid_dense_width"] != 8
        or variant["total_complex_state_per_layer"] != 1_280
        or manifest["parameter_counts"][LABEL] != 56_989_718
    ):
        raise RuntimeError("invalid content-preserving ALPHABET campaign")
    campaign_id = manifest["campaign_id"]
    runtime = {
        "schema": "lnet.kau.alphabet_lm.content_preserving_image_postfusion.runtime.v1",
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
        "runs": {
            LABEL: {
                "id": hashlib.sha256(
                    f"{campaign_id}\0{LABEL}:seed501".encode()
                ).hexdigest()[:16],
                "display_name": "RTX4090-S501-Hybrid-ContentH4P4K64-DenseP32R8-L19-30M",
                "tags": [
                    "RTX4090",
                    "ALPHABET2",
                    "Hybrid",
                    "ContentH4P4K64",
                    "DenseP32R8",
                    "FixedLaplace",
                    "ImagePostFusion",
                    "FromScratch",
                    "30M",
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
            print("stale content-preserving ALPHABET runtime", file=sys.stderr)
            return 1
        return 0
    RUNTIME.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
