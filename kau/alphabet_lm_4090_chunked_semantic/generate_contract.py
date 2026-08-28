#!/usr/bin/env python3
"""Generate the chunk-rate semantic P128 ALPHABET runtime."""

from __future__ import annotations

# ruff: noqa: T201
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "kau/alphabet_lm_4090_chunked_semantic/campaign.json"
RUNTIME = ROOT / "kau/alphabet_lm_4090_chunked_semantic/campaign.runtime.json"
LABEL = "alphabet-chunk32-semantic-p128-upper4"


def _render() -> str:
    raw = MANIFEST.read_bytes()
    manifest = json.loads(raw)
    variant = manifest["architecture"]["variants"][LABEL]
    if (
        manifest.get("schema") != "lnet.kau.alphabet_lm.chunked_semantic_1m.v1"
        or manifest["training"]["execution"] != [LABEL]
        or manifest["training"]["target_tokens"] != 1_000_000
        or variant["chunk_memory"] is not True
        or variant["chunk_size"] != 32
        or variant["chunk_pole_modes"] != 128
        or variant["train_upper_blocks"] != 4
        or set(manifest["parameter_counts"]) != {LABEL}
    ):
        raise RuntimeError("invalid chunked semantic campaign")
    campaign_id = manifest["campaign_id"]
    runtime = {
        "schema": "lnet.kau.alphabet_lm.chunked_semantic_1m.runtime.v1",
        "campaign_id": campaign_id,
        "campaign_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "output_namespace": manifest["output_namespace"],
        "architecture": manifest["architecture"],
        "trunk": manifest["trunk"],
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
                "display_name": "RTX4090-S501-ALPHABET-Chunk32-SemanticP128-1M",
                "tags": [
                    "RTX4090",
                    "FineWeb-Edu",
                    "Chunk32",
                    "SemanticMemory",
                    "P128",
                    "Upper4",
                    "1M",
                    "seed501",
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
            print("stale chunked semantic runtime", file=sys.stderr)
            return 1
        return 0
    RUNTIME.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
