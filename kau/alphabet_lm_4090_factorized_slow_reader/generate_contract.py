#!/usr/bin/env python3
"""Generate the pole-specific R2K4/R4K4 slow-reader runtime."""

from __future__ import annotations

# ruff: noqa: T201
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "kau/alphabet_lm_4090_factorized_slow_reader/campaign.json"
RUNTIME = ROOT / "kau/alphabet_lm_4090_factorized_slow_reader/campaign.runtime.json"


def _render() -> str:
    raw = MANIFEST.read_bytes()
    manifest = json.loads(raw)
    execution = manifest["training"]["execution"]
    architecture = manifest["architecture"]
    overrides = architecture["variants"]
    variants = {
        label: {**architecture["common"], **overrides[label]} for label in execution
    }
    if (
        manifest.get("schema") != "lnet.kau.alphabet_lm.factorized_slow_reader.v1"
        or len(execution) != 4
        or set(execution) != set(overrides)
        or any(
            variant["slow_cnn_pole_reader"] != "factorized_complex"
            for variant in variants.values()
        )
        or {variant["slow_cnn_pole_reader_rank"] for variant in variants.values()} != {2, 4}
        or any(variant["slow_cnn_pole_stride"] != 16 for variant in variants.values())
    ):
        raise RuntimeError("invalid factorized slow-reader campaign")
    campaign_id = manifest["campaign_id"]
    runs = {}
    for label in execution:
        rank = 2 if "r2k4" in label else 4
        recurrent = "no-recurrence" not in label
        kind = "Recurrent" if recurrent else "NoRecurrence"
        runs[label] = {
            "id": hashlib.sha256(f"{campaign_id}\0{label}:seed501".encode()).hexdigest()[:16],
            "display_name": f"RTX4090-S501-ALPHABET-SlowR{rank}K4-{kind}-1M",
            "tags": [
                "RTX4090",
                f"R{rank}K4",
                "PoleSpecific",
                "Stride16",
                "P128",
                kind,
                "1M",
                "seed501",
            ],
        }
    runtime = {
        "schema": "lnet.kau.alphabet_lm.factorized_slow_reader.runtime.v1",
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
            print("stale factorized slow-reader runtime", file=sys.stderr)
            return 1
        return 0
    RUNTIME.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
