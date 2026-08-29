#!/usr/bin/env python3
"""Generate the end-to-end selective-forcing 30M runtime."""

from __future__ import annotations

# ruff: noqa: T201
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "kau/alphabet_lm_4090_selective_forcing_30m/campaign.json"
RUNTIME = ROOT / "kau/alphabet_lm_4090_selective_forcing_30m/campaign.runtime.json"


def _render() -> str:
    raw = MANIFEST.read_bytes()
    manifest = json.loads(raw)
    architecture = manifest["architecture"]
    training = manifest["training"]
    execution = training["execution"]
    variants = {
        label: {**architecture["common"], **architecture["variants"][label]}
        for label in execution
    }
    label = "selective-forcing-r16-e2e-30m"
    variant = variants[label]
    if (
        execution != [label]
        or training["target_tokens"] != 30_000_000
        or training["validation_milestone_tokens"] != [4_000_000, 10_000_000, 20_000_000]
        or variant["slow_cnn_pole_stride"] != 1
        or variant["slow_cnn_pole_reader_kernel"] != 3
        or not variant["slow_cnn_pole_specific_reader"]
        or not variant["slow_cnn_pole_write_scheduler"]
    ):
        raise RuntimeError("invalid selective-forcing 30M campaign")
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
        "reference": manifest["reference"],
        "training": training,
        "parameter_counts": manifest["parameter_counts"],
        "total_parameter_counts": manifest["total_parameter_counts"],
        "diagnostics": manifest["diagnostics"],
        "entity": manifest["wandb"]["entity"],
        "project": manifest["wandb"]["project"],
        "group": manifest["wandb"]["group"],
        "runs": {
            label: {
                "id": hashlib.sha256(f"{campaign_id}\0{label}:seed501".encode()).hexdigest()[:16],
                "display_name": "RTX4090-S501-ALPHABET2-SelectiveForcingR16-E2E-30M",
                "tags": [
                    "RTX4090",
                    "ALPHABET2",
                    "PoleReader",
                    "SelectiveForcing",
                    "30M",
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
            print("stale selective-forcing 30M runtime", file=sys.stderr)
            return 1
        return 0
    RUNTIME.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
