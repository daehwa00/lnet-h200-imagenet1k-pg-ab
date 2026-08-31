#!/usr/bin/env python3
"""Generate the image-PostFusion ALPHABET-2 100M runtime."""

from __future__ import annotations

# ruff: noqa: T201
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "kau/alphabet_lm_4090_image_postfusion_alphabet2/campaign.json"
RUNTIME = ROOT / "kau/alphabet_lm_4090_image_postfusion_alphabet2/campaign.runtime.json"


def _render() -> str:
    raw = MANIFEST.read_bytes()
    manifest = json.loads(raw)
    architecture = manifest["architecture"]
    training = manifest["training"]
    label = "image-postfusion-alphabet2-p32n4r16-l19-fromscratch-100m"
    if (
        training["execution"] != [label]
        or training["target_tokens"] != 100_000_000
        or architecture["K_complex"] != 256
        or architecture["K_real_dof"] != 512
        or architecture["layers"] != 19
        or manifest["parameter_counts"][label] != 41_808_851
        or not manifest["diagnostics"]["final_validation_only"]
    ):
        raise RuntimeError("invalid image-PostFusion ALPHABET-2 campaign")
    campaign_id = manifest["campaign_id"]
    runtime = {
        "schema": "lnet.kau.alphabet_lm.alphabet2_complex_vector.runtime.v1",
        "campaign_id": campaign_id,
        "campaign_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "output_namespace": manifest["output_namespace"],
        "architecture": architecture,
        "source": manifest["source"],
        "training": training,
        "parameter_counts": manifest["parameter_counts"],
        "total_parameter_counts": manifest["parameter_counts"],
        "diagnostics": manifest["diagnostics"],
        "entity": manifest["wandb"]["entity"],
        "project": manifest["wandb"]["project"],
        "group": manifest["wandb"]["group"],
        "runs": {
            label: {
                "id": hashlib.sha256(f"{campaign_id}\0{label}:seed501".encode()).hexdigest()[:16],
                "display_name": "RTX4090-S501-Image-PostFusion-ALPHABET2-P32N4R16-L19-100M",
                "tags": [
                    "RTX4090",
                    "ALPHABET2",
                    "ImagePostFusion",
                    "ComplexHighway",
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
            print("stale image-PostFusion ALPHABET-2 runtime", file=sys.stderr)
            return 1
        return 0
    RUNTIME.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
