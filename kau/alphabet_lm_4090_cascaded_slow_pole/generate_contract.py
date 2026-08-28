#!/usr/bin/env python3
"""Generate the block-4 plus block-8 cascaded memory runtime."""

from __future__ import annotations

# ruff: noqa: T201
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "kau/alphabet_lm_4090_cascaded_slow_pole/campaign.json"
RUNTIME = ROOT / "kau/alphabet_lm_4090_cascaded_slow_pole/campaign.runtime.json"


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
        manifest.get("schema") != "lnet.kau.alphabet_lm.cascaded_slow_cnn_pole.v1"
        or len(execution) != 6
        or set(execution) != set(overrides)
        or manifest["diagnostics"]["milestones"] != [1_000_000, 2_000_000, 4_000_000]
        or any(variant["additional_slow_cnn_pole_depths"] != [4] for variant in variants.values())
        or any(variant["slow_cnn_pole_use_recurrence"] is not True for variant in variants.values())
    ):
        raise RuntimeError("invalid cascaded slow-bank campaign")
    campaign_id = manifest["campaign_id"]
    runs = {}
    for label in execution:
        recurrent = "no-recurrence" not in label
        milestone = label.rsplit("-", 1)[-1].upper()
        kind = "Recurrent" if recurrent else "NoRecurrence"
        runs[label] = {
            "id": hashlib.sha256(f"{campaign_id}\0{label}:seed501".encode()).hexdigest()[:16],
            "display_name": f"RTX4090-S501-ALPHABET-CascadeD4D8-A{kind}-{milestone}",
            "tags": [
                "RTX4090",
                "CascadeD4D8",
                "Stride16",
                "P128x2",
                kind,
                milestone,
                "seed501",
            ],
        }
    parameter_counts = dict.fromkeys(execution, manifest["parameter_counts"])
    total_parameter_counts = dict.fromkeys(execution, manifest["total_parameter_counts"])
    runtime = {
        "schema": "lnet.kau.alphabet_lm.cascaded_slow_cnn_pole.runtime.v1",
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
        "parameter_counts": parameter_counts,
        "total_parameter_counts": total_parameter_counts,
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
            print("stale cascaded slow-bank runtime", file=sys.stderr)
            return 1
        return 0
    RUNTIME.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
