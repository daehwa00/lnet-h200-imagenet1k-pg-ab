#!/usr/bin/env python3
"""Generate the H200 ImageNet-100 K-family depth-control runtime."""

from __future__ import annotations

# ruff: noqa: ANN401, T201
# pyright: reportExplicitAny=false
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "h200/k_family_depth_controls/campaign.json"
PROTOCOL_PATH = ROOT / "h200/campaign.json"
RUNTIME_PATH = ROOT / "h200/k_family_depth_controls/campaign.runtime.json"
HEX_64 = re.compile(r"^[0-9a-f]{64}$")
VARIANTS = (
    "S-K32-P48x4-D2262", "S-K32-P48x4-D2242",
    "L-K64-P80x4-D2262", "L-K64-P80x4-D2282", "XL-K96-P128x4-D2282",
)
PARAMETER_COUNTS = {
    "S-K32-P48x4-D2262": 399_492,
    "S-K32-P48x4-D2242": 337_732,
    "L-K64-P80x4-D2262": 1_273_860,
    "L-K64-P80x4-D2282": 1_479_236,
    "XL-K96-P128x4-D2282": 3_249_124,
}


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _run_id(campaign_id: str, label: str) -> str:
    return hashlib.sha256(f"{campaign_id}\0{label}".encode()).hexdigest()[:16]


def _load() -> tuple[dict[str, Any], dict[str, Any], str]:
    raw = MANIFEST_PATH.read_bytes()
    manifest = json.loads(raw)
    protocol_source = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    protocol = {
        "graphql_operations": protocol_source["graphql_operations"],
        "protocol": protocol_source["protocol"],
    }
    digest = hashlib.sha256(
        raw + b"\0" + json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return manifest, protocol, digest


def _records(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        variant: {
            "id": _run_id(manifest["campaign_id"], f"{variant}:seed501"),
            "display_name": f"H200-I100-S501-{index:02d}-{variant}",
            "tags": [
                "H200", "ImageNet-100", "K-family", "depth-control",
                "uniform-P", "terminal-depth2", "seed501", "authenticated",
            ],
        }
        for index, variant in enumerate(VARIANTS, start=1)
    }


def _validate(manifest: dict[str, Any], protocol: dict[str, Any], digest: str) -> None:
    training = manifest.get("training", {})
    reused = manifest.get("reused_controls", {})
    if (
        manifest.get("schema") != "lnet.h200.imagenet100.k_family_depth_controls.v1"
        or manifest.get("campaign_id")
        != "h200-imagenet100-k-family-depth-controls-s501-v1"
        or manifest.get("output_namespace")
        != "lnet-h200-imagenet100-k-family-depth-controls-v1"
        or not HEX_64.fullmatch(digest)
        or tuple(training.get("variants", ())) != VARIANTS
        or training.get("seed") != 501
        or training.get("epochs") != 100
        or training.get("batch_size") != 128
        or training.get("precision") != "bfloat16"
        or manifest.get("parameter_counts") != PARAMETER_COUNTS
    ):
        raise ValueError("invalid K-family depth-control campaign")
    if (
        reused.get("XL-K96-P128x4-D2262", {}).get("wandb_run_id")
        != "ea0a8a0e72d323c3"
        or "XL-K96-P128x4-D2262" in training.get("variants", ())
    ):
        raise ValueError("completed XL D2262 control reuse changed")
    operations = protocol["graphql_operations"]
    if len(operations) != 7 or any(not HEX_64.fullmatch(value) for value in operations.values()):
        raise ValueError("invalid traced GraphQL operations")


def _runtime(manifest: dict[str, Any], digest: str) -> dict[str, Any]:
    wandb, relay = manifest["wandb"], manifest["relay"]
    return {
        "schema": "lnet.h200.imagenet100.k_family_depth_controls.runtime.v1",
        "campaign_id": manifest["campaign_id"],
        "output_namespace": manifest["output_namespace"],
        "campaign_manifest_sha256": digest,
        "dataset": manifest["dataset"],
        "training": manifest["training"],
        "specs": manifest["specs"],
        "parameter_counts": manifest["parameter_counts"],
        "reused_controls": manifest["reused_controls"],
        "wandb_sdk_version": wandb["sdk_version"],
        "wandb_base_url": wandb["base_url"],
        "wandb_app_url": wandb["app_url"],
        "entity": wandb["entity"], "project": wandb["project"],
        "group": wandb["group"], "console": wandb["console"],
        "program": "h200/run_imagenet100_k_family_depth_controls.sh",
        "relay_protocol_version": relay["protocol_version"],
        "runs": {variant: {"501": record} for variant, record in _records(manifest).items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    manifest, protocol, digest = _load()
    _validate(manifest, protocol, digest)
    rendered = _json(_runtime(manifest, digest))
    if args.check:
        if not RUNTIME_PATH.is_file() or RUNTIME_PATH.read_text(encoding="utf-8") != rendered:
            print("stale generated depth-control runtime", file=sys.stderr)
            return 1
        return 0
    RUNTIME_PATH.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
