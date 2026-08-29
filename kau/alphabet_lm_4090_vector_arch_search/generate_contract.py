#!/usr/bin/env python3
"""Generate the three-way VectorPole architecture-search runtime."""

from __future__ import annotations

# ruff: noqa: T201
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "kau/alphabet_lm_4090_vector_arch_search/campaign.json"
RUNTIME = ROOT / "kau/alphabet_lm_4090_vector_arch_search/campaign.runtime.json"


def _render() -> str:
    raw = MANIFEST.read_bytes()
    manifest = json.loads(raw)
    architecture = manifest["architecture"]
    execution = manifest["training"]["execution"]
    variants = {
        label: {**architecture["common"], **architecture["variants"][label]}
        for label in execution
    }
    expected = [
        "complexq-r16-e2e-4m",
        "token-rate-r16-e2e-4m",
        "coordinate-read-r16-e2e-4m",
    ]
    if (
        execution != expected
        or variants[expected[0]]["slow_cnn_pole_stride"] != 16
        or variants[expected[1]]["slow_cnn_pole_stride"] != 1
        or variants[expected[1]]["slow_cnn_pole_minimum_half_life"] != 16.0
        or variants[expected[1]]["slow_cnn_pole_maximum_half_life"] != 4_096.0
        or not variants[expected[2]]["slow_cnn_pole_coordinate_read"]
        or any(not variant["slow_cnn_pole_complex_vector_query"] for variant in variants.values())
        or not manifest["diagnostics"]["final_best_only"]
    ):
        raise RuntimeError("invalid vector architecture-search campaign")
    campaign_id = manifest["campaign_id"]
    runs = {
        label: {
            "id": hashlib.sha256(f"{campaign_id}\0{label}:seed501".encode()).hexdigest()[:16],
            "display_name": f"RTX4090-S501-{label}",
            "tags": ["RTX4090", "ALPHABET2", "VectorSearch", "4M", "seed501"],
        }
        for label in execution
    }
    runtime = {
        "schema": "lnet.kau.alphabet_lm.alphabet2_complex_vector.runtime.v1",
        "campaign_id": campaign_id,
        "campaign_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "output_namespace": manifest["output_namespace"],
        "architecture": {
            "K": architecture["K"],
            "layers": architecture["layers"],
            "variants": variants,
        },
        "source": manifest["source"],
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
            print("stale vector architecture-search runtime", file=sys.stderr)
            return 1
        return 0
    RUNTIME.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
