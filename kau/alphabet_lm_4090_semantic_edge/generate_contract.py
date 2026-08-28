#!/usr/bin/env python3
"""Generate recurrent and local Semantic Edge P128 runtimes."""

from __future__ import annotations

# ruff: noqa: T201
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "kau/alphabet_lm_4090_semantic_edge/campaign.json"
RUNTIME = ROOT / "kau/alphabet_lm_4090_semantic_edge/campaign.runtime.json"
RECURRENT = "alphabet-semantic-edge-p128-recurrent"
CONTROL = "alphabet-semantic-edge-p128-no-recurrence"


def _render() -> str:
    raw = MANIFEST.read_bytes()
    manifest = json.loads(raw)
    variants = manifest["architecture"]["variants"]
    if (
        manifest.get("schema") != "lnet.kau.alphabet_lm.semantic_edge_1m.v1"
        or manifest["training"]["execution"] != [RECURRENT, CONTROL]
        or manifest["training"]["target_tokens"] != 1_000_000
        or variants[RECURRENT]["semantic_edge_use_recurrence"] is not True
        or variants[CONTROL]["semantic_edge_use_recurrence"] is not False
        or any(variants[label]["semantic_edge_stride"] != 16 for label in variants)
        or set(manifest["parameter_counts"]) != {RECURRENT, CONTROL}
    ):
        raise RuntimeError("invalid semantic edge campaign")
    campaign_id = manifest["campaign_id"]

    def run(label: str, suffix: str) -> dict[str, object]:
        return {
            "id": hashlib.sha256(f"{campaign_id}\0{label}:seed501".encode()).hexdigest()[:16],
            "display_name": f"RTX4090-S501-ALPHABET-SemanticEdgeP128-{suffix}-1M",
            "tags": [
                "RTX4090",
                "FineWeb-Edu",
                "SemanticEdge",
                "Stride16",
                "P128",
                suffix,
                "1M",
                "seed501",
            ],
        }

    runtime = {
        "schema": "lnet.kau.alphabet_lm.semantic_edge_1m.runtime.v1",
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
            RECURRENT: run(RECURRENT, "Recurrent"),
            CONTROL: run(CONTROL, "NoRecurrence"),
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
            print("stale semantic edge runtime", file=sys.stderr)
            return 1
        return 0
    RUNTIME.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
