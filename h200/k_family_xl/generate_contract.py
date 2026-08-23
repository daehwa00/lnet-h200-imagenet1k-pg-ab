#!/usr/bin/env python3
"""Validate and generate the H200 ImageNet-100 K96/XL campaign runtime."""

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
MANIFEST_PATH = ROOT / "h200/k_family_xl/campaign.json"
PROTOCOL_PATH = ROOT / "h200/campaign.json"
RUNTIME_PATH = ROOT / "h200/k_family_xl/campaign.runtime.json"
HEX_64 = re.compile(r"^[0-9a-f]{64}$")
VARIANTS = (
    "XL-K96-U1",
    "XL-K96-U125",
    "XL-K96-Shaped",
    "XL-K96-Rich",
)
POLE_SCHEDULES = {
    "XL-K96-U1": [96, 96, 96, 96],
    "XL-K96-U125": [128, 128, 128, 128],
    "XL-K96-Shaped": [144, 192, 144, 144],
    "XL-K96-Rich": [144, 192, 192, 144],
}
PARAMETER_COUNTS = {
    "XL-K96-U1": 2_313_892,
    "XL-K96-U125": 2_791_524,
    "XL-K96-Shaped": 2_996_740,
    "XL-K96-Rich": 3_200_068,
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
    campaign_id = manifest["campaign_id"]
    return {
        variant: {
            "id": _run_id(campaign_id, f"{variant}:seed501"),
            "display_name": f"H200-I100-S501-{index:02d}-{variant}",
            "tags": [
                "H200",
                "ImageNet-100",
                "K96",
                "XL-family",
                "P-schedule",
                "seed501",
                "authenticated",
            ],
        }
        for index, variant in enumerate(manifest["training"]["variants"], start=1)
    }


def _validate(manifest: dict[str, Any], protocol: dict[str, Any], digest: str) -> None:
    training = manifest.get("training", {})
    wandb = manifest.get("wandb", {})
    relay = manifest.get("relay", {})
    dataset = manifest.get("dataset", {})
    if (
        manifest.get("schema") != "lnet.h200.imagenet100.k_family_xl.v1"
        or manifest.get("campaign_id") != "h200-imagenet100-k-family-xl-s501-v1"
        or manifest.get("output_namespace") != "lnet-h200-imagenet100-k-family-xl-v1"
        or not HEX_64.fullmatch(digest)
    ):
        raise ValueError("invalid XL campaign identity")
    if (
        dataset.get("train_images") != 130000
        or dataset.get("validation_images") != 5000
        or dataset.get("classes") != 100
        or dataset.get("first_synset") != "n01440764"
        or dataset.get("last_synset") != "n02077923"
    ):
        raise ValueError("ImageNet-100 selection contract changed")
    if (
        tuple(training.get("variants", ())) != VARIANTS
        or training.get("seed") != 501
        or training.get("epochs") != 100
        or training.get("batch_size") != 128
        or training.get("precision") != "bfloat16"
        or training.get("execution") != "one_model_to_epoch_100_then_next"
        or manifest.get("pole_schedules") != POLE_SCHEDULES
        or manifest.get("parameter_counts") != PARAMETER_COUNTS
    ):
        raise ValueError("XL training matrix changed")
    if (
        wandb.get("sdk_version") != "0.22.3"
        or wandb.get("entity") != "daehwa"
        or wandb.get("project") != "alphabet2d-imagenet100"
        or wandb.get("group") != "R2K3-KFamily-XL-H200-S501"
        or wandb.get("console") != "off"
        or relay.get("url") != wandb.get("base_url")
        or relay.get("upstream_origin") != "https://api.wandb.ai"
    ):
        raise ValueError("invalid XL W&B relay contract")
    operations = protocol["graphql_operations"]
    if len(operations) != 7 or any(not HEX_64.fullmatch(value) for value in operations.values()):
        raise ValueError("invalid traced GraphQL operations")
    records = list(_records(manifest).values())
    if len({record["id"] for record in records}) != len(records):
        raise ValueError("derived XL W&B run IDs are not unique")


def _runtime(manifest: dict[str, Any], digest: str) -> dict[str, Any]:
    return {
        "schema": "lnet.h200.imagenet100.k_family_xl.runtime.v1",
        "campaign_id": manifest["campaign_id"],
        "output_namespace": manifest["output_namespace"],
        "campaign_manifest_sha256": digest,
        "dataset": manifest["dataset"],
        "training": manifest["training"],
        "pole_schedules": manifest["pole_schedules"],
        "parameter_counts": manifest["parameter_counts"],
        "wandb_sdk_version": manifest["wandb"]["sdk_version"],
        "wandb_base_url": manifest["wandb"]["base_url"],
        "wandb_app_url": manifest["wandb"]["app_url"],
        "entity": manifest["wandb"]["entity"],
        "project": manifest["wandb"]["project"],
        "group": manifest["wandb"]["group"],
        "program": "h200/run_imagenet100_k_family_xl.sh",
        "console": manifest["wandb"]["console"],
        "relay_protocol_version": manifest["relay"]["protocol_version"],
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
            print("stale generated XL campaign runtime", file=sys.stderr)
            return 1
        return 0
    RUNTIME_PATH.parent.mkdir(parents=True, exist_ok=True)
    RUNTIME_PATH.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
