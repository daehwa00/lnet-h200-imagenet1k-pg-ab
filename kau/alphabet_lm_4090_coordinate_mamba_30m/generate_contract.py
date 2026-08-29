#!/usr/bin/env python3
"""Generate exact coordinate-read and Mamba 4M-to-30M extension runtimes."""

from __future__ import annotations

# ruff: noqa: T201
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "kau/alphabet_lm_4090_coordinate_mamba_30m/campaign.json"
RUNTIME = ROOT / "kau/alphabet_lm_4090_coordinate_mamba_30m/campaign.runtime.json"


def _render() -> str:
    raw = MANIFEST.read_bytes()
    manifest = json.loads(raw)
    architecture = manifest["architecture"]
    execution = manifest["training"]["execution"]
    variants = {
        label: {**architecture["common"], **architecture["variants"][label]}
        for label in execution
    }
    expected = ["coordinate-read-r16-e2e-30m", "mamba-standard-e2e-30m"]
    coordinate = variants[expected[0]]
    if (
        execution != expected
        or manifest["training"]["target_tokens"] != 30_000_000
        or not manifest["comparison"]["resume_is_exact"]
        or not coordinate["slow_cnn_pole_coordinate_read"]
        or not coordinate["slow_cnn_pole_complex_vector_query"]
        or coordinate["slow_cnn_pole_stride"] != 16
        or not manifest["diagnostics"]["final_validation_only"]
    ):
        raise RuntimeError("invalid coordinate/Mamba 30M campaign")
    campaign_id = manifest["campaign_id"]
    runs = {
        label: {
            "id": hashlib.sha256(f"{campaign_id}\0{label}:seed501".encode()).hexdigest()[:16],
            "display_name": (
                "RTX4090-S501-ALPHABET2-CoordinateR16-E2E-30M"
                if label.startswith("coordinate")
                else "RTX4090-S501-Mamba-Standard-E2E-30M"
            ),
            "tags": ["RTX4090", "E2E", "30M", "seed501"],
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
            print("stale coordinate/Mamba 30M runtime", file=sys.stderr)
            return 1
        return 0
    RUNTIME.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
