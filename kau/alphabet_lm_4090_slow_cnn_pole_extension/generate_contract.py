#!/usr/bin/env python3
"""Generate paired slow CNN-pole 2M/4M extension runtimes."""

from __future__ import annotations

# ruff: noqa: T201
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "kau/alphabet_lm_4090_slow_cnn_pole_extension/campaign.json"
RUNTIME = ROOT / "kau/alphabet_lm_4090_slow_cnn_pole_extension/campaign.runtime.json"


def _render() -> str:
    raw = MANIFEST.read_bytes()
    manifest = json.loads(raw)
    execution = manifest["training"]["execution"]
    variants = manifest["architecture"]["variants"]
    if (
        manifest.get("schema") != "lnet.kau.alphabet_lm.slow_cnn_pole_extension.v1"
        or len(execution) != 4
        or set(execution) != set(variants)
        or manifest["diagnostics"]["milestones"] != [2_000_000, 4_000_000]
        or any(variants[label]["slow_cnn_pole_stride"] != 16 for label in variants)
        or any(variants[label]["cnn_pole_use_recurrence"] is not False for label in variants)
        or set(manifest["parameter_counts"]) != set(variants)
    ):
        raise RuntimeError("invalid slow CNN-pole extension campaign")
    campaign_id = manifest["campaign_id"]
    runs = {}
    for label in execution:
        recurrent = "no-recurrence" not in label
        milestone = "2M" if label.endswith("2m") else "4M"
        kind = "Recurrent" if recurrent else "NoRecurrence"
        runs[label] = {
            "id": hashlib.sha256(f"{campaign_id}\0{label}:seed501".encode()).hexdigest()[:16],
            "display_name": f"RTX4090-S501-ALPHABET-CNN6-SlowP128-{kind}-{milestone}",
            "tags": [
                "RTX4090",
                "CNN6",
                "SlowSemantic",
                "Stride16",
                "P128",
                kind,
                milestone,
                "seed501",
            ],
        }
    runtime = {
        "schema": "lnet.kau.alphabet_lm.slow_cnn_pole_extension.runtime.v1",
        "campaign_id": campaign_id,
        "campaign_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "output_namespace": manifest["output_namespace"],
        "architecture": manifest["architecture"],
        "sources": manifest["sources"],
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
            print("stale slow CNN-pole extension runtime", file=sys.stderr)
            return 1
        return 0
    RUNTIME.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
