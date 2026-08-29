#!/usr/bin/env python3
"""Generate the factorized gain-decomposition runtime contract."""

from __future__ import annotations

# ruff: noqa: T201
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "kau/alphabet_lm_4090_factorized_gain_decomposition/campaign.json"
RUNTIME = ROOT / "kau/alphabet_lm_4090_factorized_gain_decomposition/campaign.runtime.json"


def _render() -> str:
    raw = MANIFEST.read_bytes()
    manifest = json.loads(raw)
    architecture = manifest["architecture"]
    execution = manifest["training"]["execution"]
    variants = {
        label: {**architecture["common"], **architecture["variants"][label]}
        for label in execution
    }
    expected = {
        "factorized-p32r4-interface-js16-4m": (4, 16),
        "factorized-p32r32-js4-4m": (32, 4),
    }
    if execution != list(expected) or manifest["training"]["target_tokens"] != 4_000_000:
        raise RuntimeError("invalid factorized gain-decomposition execution")
    for label, (width, synthesis_rank) in expected.items():
        variant = variants[label]
        if (
            variant["repeated_vector_pole_width"] != width
            or variant["repeated_vector_pole_write_rank"] != 4
            or variant["repeated_vector_pole_query_rank"] != 4
            or variant["repeated_vector_pole_synthesis_rank"] != synthesis_rank
            or not variant["repeated_vector_pole_activation_checkpoint"]
        ):
            raise RuntimeError(f"invalid factorized gain variant: {label}")
    campaign_id = manifest["campaign_id"]
    runtime = {
        "schema": "lnet.kau.alphabet_lm.factorized_gain_decomposition.runtime.v1",
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
        "runs": {
            label: {
                "id": hashlib.sha256(f"{campaign_id}\0{label}:seed501".encode()).hexdigest()[:16],
                "display_name": f"RTX4090-S501-ALPHABET2-{label}",
                "tags": [
                    "RTX4090",
                    "ALPHABET2",
                    "FactorizedVectorPole",
                    "GainDecomposition",
                ],
            }
            for label in execution
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
            print("stale factorized gain-decomposition runtime", file=sys.stderr)
            return 1
        return 0
    RUNTIME.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
