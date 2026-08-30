#!/usr/bin/env python3
"""Generate the from-scratch direct/gate outer-scaffold factorial runtime."""

from __future__ import annotations

# ruff: noqa: T201
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "kau/alphabet_lm_4090_mamba_outer_factorial/campaign.json"
RUNTIME = ROOT / "kau/alphabet_lm_4090_mamba_outer_factorial/campaign.runtime.json"


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
        "mamba-outer-post-fromscratch-30m": (False, False),
        "mamba-outer-direct-fromscratch-30m": (True, False),
        "mamba-outer-gate-fromscratch-30m": (False, True),
        "mamba-outer-both-fromscratch-30m": (True, True),
    }
    if (
        execution != list(expected)
        or manifest["training"]["target_tokens"] != 30_000_000
        or manifest["training"]["validation_milestone_tokens"] != [10_000_000]
    ):
        raise RuntimeError("invalid Mamba outer factorial execution")
    for label, (direct, gate) in expected.items():
        variant = variants[label]
        if (
            not variant["repeated_vector_pole_mamba_outer"]
            or variant["repeated_vector_pole_outer_direct"] is not direct
            or variant["repeated_vector_pole_outer_gate"] is not gate
        ):
            raise RuntimeError(f"invalid Mamba outer variant: {label}")
    campaign_id = manifest["campaign_id"]
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
        "runs": {
            label: {
                "id": hashlib.sha256(f"{campaign_id}\0{label}:seed501".encode()).hexdigest()[:16],
                "display_name": f"RTX4090-S501-ALPHABET2-{label}",
                "tags": ["RTX4090", "ALPHABET2", "MambaOuter", "FromScratch"],
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
            print("stale Mamba outer factorial runtime", file=sys.stderr)
            return 1
        return 0
    RUNTIME.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
