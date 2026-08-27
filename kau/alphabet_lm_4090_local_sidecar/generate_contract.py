#!/usr/bin/env python3
"""Generate the RTX 4090 Mamba-2M and LocalSidecar runtime."""

from __future__ import annotations

# ruff: noqa: T201
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "kau/alphabet_lm_4090_local_sidecar/campaign.json"
RUNTIME = ROOT / "kau/alphabet_lm_4090_local_sidecar/campaign.runtime.json"
MAMBA = "mamba-2m"
SIDECAR = "alphabet-dense-k3-local-sidecar"


def _run(campaign_id: str, label: str, display_name: str, tags: list[str]) -> dict[str, object]:
    return {
        "id": hashlib.sha256(f"{campaign_id}\0{label}:seed501".encode()).hexdigest()[:16],
        "display_name": display_name,
        "tags": tags,
    }


def _render() -> str:
    raw = MANIFEST.read_bytes()
    manifest = json.loads(raw)
    variant = manifest["architecture"]["variants"][SIDECAR]
    if (
        manifest.get("schema") != "lnet.kau.alphabet_lm.local_sidecar_2m.v1"
        or manifest["training"]["execution"] != [MAMBA, SIDECAR]
        or manifest["training"]["target_tokens"] != 2_000_000
        or variant["memory_layout"] != "local_sidecar"
        or variant["sidecar_initial_scale"] != 0.01
        or set(manifest["parameter_counts"]) != {MAMBA, SIDECAR}
    ):
        raise RuntimeError("invalid LocalSidecar campaign")
    campaign_id = manifest["campaign_id"]
    runtime = {
        "schema": "lnet.kau.alphabet_lm.local_sidecar_2m.runtime.v1",
        "campaign_id": campaign_id,
        "campaign_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "output_namespace": manifest["output_namespace"],
        "paper": manifest["paper"],
        "architecture": manifest["architecture"],
        "controls": manifest["controls"],
        "training": manifest["training"],
        "parameter_counts": manifest["parameter_counts"],
        "diagnostics": manifest["diagnostics"],
        "entity": manifest["wandb"]["entity"],
        "project": manifest["wandb"]["project"],
        "group": manifest["wandb"]["group"],
        "runs": {
            MAMBA: _run(
                campaign_id,
                MAMBA,
                "RTX4090-S501-OfficialMamba-2M",
                ["RTX4090", "FineWeb-Edu", "2M", "Mamba", "seed501"],
            ),
            SIDECAR: _run(
                campaign_id,
                SIDECAR,
                "RTX4090-S501-ALPHABET-DenseK3-LocalSidecar-2M",
                [
                    "RTX4090",
                    "FineWeb-Edu",
                    "2M",
                    "DenseK3",
                    "LocalSidecar",
                    "P320",
                    "beta0.01",
                    "seed501",
                ],
            ),
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
            print("stale LocalSidecar runtime", file=sys.stderr)
            return 1
        return 0
    RUNTIME.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
