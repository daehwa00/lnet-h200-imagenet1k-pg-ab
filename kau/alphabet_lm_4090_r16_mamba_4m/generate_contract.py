#!/usr/bin/env python3
"""Generate paired R16 and standard-Mamba 4M from-scratch runtimes."""

from __future__ import annotations

# ruff: noqa: T201
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "kau/alphabet_lm_4090_r16_mamba_4m/campaign.json"
RUNTIME = ROOT / "kau/alphabet_lm_4090_r16_mamba_4m/campaign.runtime.json"


def _render() -> str:
    raw = MANIFEST.read_bytes()
    manifest = json.loads(raw)
    execution = manifest["training"]["execution"]
    architecture = manifest["architecture"]
    variants = {
        label: {**architecture["common"], **architecture["variants"][label]}
        for label in execution
    }
    expected = ["alphabet2-complex-vector-r16-e2e-4m", "mamba-standard-e2e-4m"]
    r16 = variants[expected[0]]
    if (
        manifest.get("schema") != "lnet.kau.alphabet_lm.alphabet2_complex_vector.v1"
        or execution != expected
        or r16["slow_cnn_pole_vector_width"] != 16
        or not r16["slow_cnn_pole_complex_vector_excitation"]
        or manifest["training"]["target_tokens"] != 4_000_000
        or manifest["diagnostics"]["intermediate_evaluations"]
    ):
        raise RuntimeError("invalid R16/Mamba 4M campaign")
    campaign_id = manifest["campaign_id"]
    runs = {
        label: {
            "id": hashlib.sha256(f"{campaign_id}\0{label}:seed501".encode()).hexdigest()[:16],
            "display_name": (
                "RTX4090-S501-ALPHABET2-ComplexVectorR16-E2E-4M"
                if label.startswith("alphabet2")
                else "RTX4090-S501-Mamba-Standard-E2E-4M"
            ),
            "tags": ["RTX4090", "E2E", "4M", "seed501"],
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
        "comparison": manifest["comparison"],
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
            print("stale R16/Mamba 4M runtime", file=sys.stderr)
            return 1
        return 0
    RUNTIME.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
