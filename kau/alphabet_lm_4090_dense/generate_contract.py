#!/usr/bin/env python3
"""Generate the RTX 4090 dense-reader follow-up runtime."""

from __future__ import annotations

# ruff: noqa: T201
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "kau/alphabet_lm_4090_dense/campaign.json"
RUNTIME = ROOT / "kau/alphabet_lm_4090_dense/campaign.runtime.json"


def _render() -> str:
    raw = MANIFEST.read_bytes()
    manifest = json.loads(raw)
    architecture = manifest["architecture"]
    if (
        manifest.get("schema") != "lnet.kau.alphabet_lm.dense_k3_p320_10m.v1"
        or architecture.get("reader") != "dense_complex_causal_k3"
        or architecture.get("P") != 320
        or architecture.get("parameters") != 36_714_496
        or manifest["training"]["scan_fp32"] is not True
    ):
        raise RuntimeError("invalid dense K3 P320 campaign")
    campaign_id = manifest["campaign_id"]
    label = "alphabet-dense-k3-p320"
    run_id = hashlib.sha256(f"{campaign_id}\0{label}:seed501".encode()).hexdigest()[:16]
    runtime = {
        "schema": "lnet.kau.alphabet_lm.dense_k3_p320_10m.runtime.v1",
        "campaign_id": campaign_id,
        "campaign_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "output_namespace": manifest["output_namespace"],
        "paper": manifest["paper"],
        "architecture": architecture,
        "training": manifest["training"],
        "entity": manifest["wandb"]["entity"],
        "project": manifest["wandb"]["project"],
        "group": manifest["wandb"]["group"],
        "parameter_counts": {label: architecture["parameters"]},
        "runs": {
            label: {
                "id": run_id,
                "display_name": "RTX4090-S501-ALPHABET-DenseK3P320-10M",
                "tags": [
                    "RTX4090", "FineWeb-Edu", "10M", "dense-complex-K3",
                    "P320", "seed501", "arXiv-2608.24051",
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
            print("stale dense K3 runtime", file=sys.stderr)
            return 1
        return 0
    RUNTIME.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
