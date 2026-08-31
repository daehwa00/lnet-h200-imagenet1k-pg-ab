#!/usr/bin/env python3
"""Generate the integrated Laplace-Mamba B/C norm-and-bias runtime."""

from __future__ import annotations

# ruff: noqa: T201
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "kau/alphabet_lm_4090_integrated_laplace_mamba_bc_norm_bias/campaign.json"
RUNTIME = ROOT / "kau/alphabet_lm_4090_integrated_laplace_mamba_bc_norm_bias/campaign.runtime.json"


def _render() -> str:
    raw = MANIFEST.read_bytes()
    manifest = json.loads(raw)
    architecture = manifest["architecture"]
    execution = manifest["training"]["execution"]
    label = "integrated-laplace-mamba-bc-norm-bias-p32n4r16-l19-fromscratch-100m"
    variant = architecture["variants"][label]
    if (
        execution != [label]
        or manifest["training"]["target_tokens"] != 100_000_000
        or variant
        != {
            "laplace_mamba_layers": 19,
            "laplace_mamba_poles": 32,
            "laplace_mamba_state_size": 4,
            "laplace_mamba_head_width": 16,
            "laplace_mamba_conv_width": 4,
            "laplace_mamba_address_norm_bias": True,
        }
        or not manifest["diagnostics"]["final_validation_only"]
    ):
        raise RuntimeError("invalid integrated Laplace-Mamba B/C norm-bias campaign")
    campaign_id = manifest["campaign_id"]
    runtime = {
        "schema": "lnet.kau.alphabet_lm.alphabet2_complex_vector.runtime.v1",
        "campaign_id": campaign_id,
        "campaign_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "output_namespace": manifest["output_namespace"],
        "architecture": {
            "K": architecture["K_real"],
            "layers": architecture["layers"],
            "variants": {label: variant},
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
                "display_name": (
                    "RTX4090-S501-Integrated-Laplace-Mamba-BCNormBias-P32N4R16-L19-100M"
                ),
                "tags": [
                    "RTX4090",
                    "LaplaceMamba",
                    "BCNormBias",
                    "FromScratch",
                    "100M",
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
            print("stale integrated Laplace-Mamba B/C norm-bias runtime", file=sys.stderr)
            return 1
        return 0
    RUNTIME.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
