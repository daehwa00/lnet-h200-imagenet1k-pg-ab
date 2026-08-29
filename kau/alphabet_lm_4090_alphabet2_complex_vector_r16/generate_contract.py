#!/usr/bin/env python3
"""Generate state-capacity-matched FullComplex VectorPole-R16 runtimes."""

from __future__ import annotations

# ruff: noqa: T201
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "kau/alphabet_lm_4090_alphabet2_complex_vector_r16/campaign.json"
RUNTIME = ROOT / "kau/alphabet_lm_4090_alphabet2_complex_vector_r16/campaign.runtime.json"


def _render() -> str:
    raw = MANIFEST.read_bytes()
    manifest = json.loads(raw)
    execution = manifest["training"]["execution"]
    architecture = manifest["architecture"]
    variants = {
        label: {**architecture["common"], **architecture["variants"][label]}
        for label in execution
    }
    expected = [
        "alphabet2-complex-vector-r16-4m",
    ]
    if (
        manifest.get("schema") != "lnet.kau.alphabet_lm.alphabet2_complex_vector.v1"
        or execution != expected
        or any(variant["slow_cnn_pole_vector_width"] != 16 for variant in variants.values())
        or any(
            not variant["slow_cnn_pole_complex_vector_excitation"]
            for variant in variants.values()
        )
        or any(variant["slow_cnn_pole_query"] != "token" for variant in variants.values())
        or any(not variant["slow_cnn_pole_use_recurrence"] for variant in variants.values())
        or not all(
            manifest["diagnostics"][name]
            for name in (
                "complex_excitation_off",
                "complex_excitation_shift",
                "complex_excitation_time_mean",
            )
        )
    ):
        raise RuntimeError("invalid complex-vector R16 campaign")
    campaign_id = manifest["campaign_id"]
    runs = {}
    for label in execution:
        milestone = label.rsplit("-", 1)[-1].upper()
        runs[label] = {
            "id": hashlib.sha256(f"{campaign_id}\0{label}:seed501".encode()).hexdigest()[:16],
            "display_name": f"RTX4090-S501-ALPHABET2-ComplexVectorR16-{milestone}",
            "tags": ["RTX4090", "ALPHABET2", "ComplexVectorR16", milestone, "seed501"],
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
            print("stale complex-vector R16 runtime", file=sys.stderr)
            return 1
        return 0
    RUNTIME.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
