#!/usr/bin/env python3
"""Generate the RTX 4090 DenseK3 fixed-vs-selective-decay runtime."""

from __future__ import annotations

# ruff: noqa: T201
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "kau/alphabet_lm_4090_dense_delta/campaign.json"
RUNTIME = ROOT / "kau/alphabet_lm_4090_dense_delta/campaign.runtime.json"
LABELS = ["alphabet-dense-k3-fixed", "alphabet-dense-k3-delta-r16"]


def _render() -> str:
    raw = MANIFEST.read_bytes()
    manifest = json.loads(raw)
    variants = manifest["architecture"]["variants"]
    if (
        manifest.get("schema") != "lnet.kau.alphabet_lm.dense_delta_screen_2m.v1"
        or manifest["training"]["execution"] != LABELS
        or manifest["training"]["target_tokens"] != 2_000_000
        or manifest["training"]["global_sequences"] != 32
        or manifest["architecture"]["reader"] != "DenseComplexK3"
        or set(variants) != set(LABELS)
        or variants[LABELS[0]]["pole_dynamics"] != "fixed"
        or variants[LABELS[1]]["pole_dynamics"] != "delta_select"
        or set(manifest["parameter_counts"]) != set(LABELS)
    ):
        raise RuntimeError("invalid DenseK3 DeltaSelect campaign")
    names = {
        "alphabet-dense-k3-fixed": "RTX4090-S501-ALPHABET-DenseK3-Fixed-2M",
        "alphabet-dense-k3-delta-r16": "RTX4090-S501-ALPHABET-DenseK3-DeltaSelectR16-2M",
    }
    campaign_id = manifest["campaign_id"]
    runtime = {
        "schema": "lnet.kau.alphabet_lm.dense_delta_screen_2m.runtime.v1",
        "campaign_id": campaign_id,
        "campaign_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "output_namespace": manifest["output_namespace"],
        "paper": manifest["paper"],
        "architecture": manifest["architecture"],
        "training": manifest["training"],
        "parameter_counts": manifest["parameter_counts"],
        "diagnostics": manifest["diagnostics"],
        "entity": manifest["wandb"]["entity"],
        "project": manifest["wandb"]["project"],
        "group": manifest["wandb"]["group"],
        "runs": {
            label: {
                "id": hashlib.sha256(f"{campaign_id}\0{label}:seed501".encode()).hexdigest()[:16],
                "display_name": names[label],
                "tags": [
                    "RTX4090",
                    "FineWeb-Edu",
                    "2M",
                    label,
                    "dense-k3",
                    "delta-select-screen",
                    "seed501",
                    "arXiv-2608.24051",
                ],
            }
            for label in LABELS
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
            print("stale DenseK3 DeltaSelect runtime", file=sys.stderr)
            return 1
        return 0
    RUNTIME.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
