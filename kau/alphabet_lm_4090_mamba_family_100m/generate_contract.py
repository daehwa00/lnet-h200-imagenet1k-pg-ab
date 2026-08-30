#!/usr/bin/env python3
"""Generate the parameter-matched Mamba-1/Mamba-2 100M runtime."""

from __future__ import annotations

# ruff: noqa: T201
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "kau/alphabet_lm_4090_mamba_family_100m/campaign.json"
RUNTIME = ROOT / "kau/alphabet_lm_4090_mamba_family_100m/campaign.runtime.json"


def _render() -> str:
    raw = MANIFEST.read_bytes()
    manifest = json.loads(raw)
    architecture = manifest["architecture"]
    execution = manifest["training"]["execution"]
    expected = [
        "mamba1-parameter-matched-fromscratch-100m",
        "mamba2-parameter-matched-fromscratch-100m",
    ]
    if (
        execution != expected
        or manifest["training"]["target_tokens"] != 100_000_000
        or manifest["training"]["validation_milestone_tokens"]
        != [10_000_000, 30_000_000]
        or architecture["target_parameters"] != 48_587_020
    ):
        raise RuntimeError("invalid Mamba family 100M campaign")
    campaign_id = manifest["campaign_id"]
    runtime = {
        "schema": "lnet.kau.alphabet_lm.alphabet2_complex_vector.runtime.v1",
        "campaign_id": campaign_id,
        "campaign_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "output_namespace": manifest["output_namespace"],
        "architecture": {
            "K": architecture["K"],
            "layers": None,
            "variants": {label: {} for label in execution},
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
                "display_name": f"RTX4090-S501-{label}",
                "tags": ["RTX4090", "Mamba", "FromScratch", "100M"],
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
            print("stale Mamba family runtime", file=sys.stderr)
            return 1
        return 0
    RUNTIME.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
