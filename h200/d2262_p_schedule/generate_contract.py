#!/usr/bin/env python3
# ruff: noqa: ANN401, EM101, T201, TRY003
# pyright: reportExplicitAny=false
"""Validate and generate the H200 ImageNet-100 D2262 P-schedule contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "h200/d2262_p_schedule/campaign.json"
PROTOCOL_PATH = ROOT / "h200/campaign.json"
RUNTIME_PATH = ROOT / "h200/d2262_p_schedule/campaign.runtime.json"
HEX_64 = re.compile(r"^[0-9a-f]{64}$")


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
    records = {}
    for index, variant in enumerate(manifest["training"]["variants"], start=1):
        records[variant] = {
            "id": _run_id(campaign_id, f"{variant}:seed501"),
            "display_name": f"H200-I100-S501-{index:02d}-{variant}",
            "tags": [
                "H200",
                "ImageNet-100",
                "D2262",
                "P-schedule",
                "seed501",
                "authenticated",
            ],
        }
    return records


def _canary(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": _run_id(manifest["campaign_id"], "permanent-canary"),
        "display_name": "H200-I100-D2262-P-schedule-relay-canary-v1",
        "tags": ["H200", "ImageNet-100", "D2262", "P-schedule", "relay-canary"],
    }


def _validate(manifest: dict[str, Any], protocol: dict[str, Any], digest: str) -> None:
    if manifest.get("schema") != "lnet.h200.imagenet100.d2262_p_schedule.v1":
        raise ValueError("invalid D2262 P-schedule campaign schema")
    if not HEX_64.fullmatch(digest):
        raise ValueError("invalid combined campaign digest")
    dataset = manifest["dataset"]
    if (
        dataset["train_images"],
        dataset["validation_images"],
        dataset["classes"],
        dataset["first_synset"],
        dataset["last_synset"],
    ) != (130000, 5000, 100, "n01440764", "n02077923"):
        raise ValueError("ImageNet-100 selection contract changed")
    training = manifest["training"]
    variants = training["variants"]
    if (
        training["seed"] != 501
        or training["epochs"] != 100
        or training["batch_size"] != 128
        or training.get("execution") != "one_model_to_epoch_100_then_next"
        or len(variants) != 6
        or len(set(variants)) != 6
    ):
        raise ValueError("D2262 P-schedule training matrix changed")
    wandb = manifest["wandb"]
    relay = manifest["relay"]
    if (
        wandb["sdk_version"] != "0.22.3"
        or wandb["console"] != "off"
        or relay["url"] != wandb["base_url"]
        or relay["upstream_origin"] != "https://api.wandb.ai"
    ):
        raise ValueError("invalid W&B relay contract")
    operations = protocol["graphql_operations"]
    if len(operations) != 7 or any(not HEX_64.fullmatch(value) for value in operations.values()):
        raise ValueError("invalid traced GraphQL operations")
    records = [*_records(manifest).values(), _canary(manifest)]
    if len({record["id"] for record in records}) != len(records):
        raise ValueError("derived W&B run IDs are not unique")
    if len({record["display_name"] for record in records}) != len(records):
        raise ValueError("derived W&B display names are not unique")


def _runtime(manifest: dict[str, Any], digest: str) -> dict[str, Any]:
    return {
        "schema": "lnet.h200.imagenet100.d2262_p_schedule.runtime.v1",
        "campaign_id": manifest["campaign_id"],
        "output_namespace": manifest["output_namespace"],
        "campaign_manifest_sha256": digest,
        "dataset": manifest["dataset"],
        "training": manifest["training"],
        "wandb_sdk_version": manifest["wandb"]["sdk_version"],
        "wandb_base_url": manifest["wandb"]["base_url"],
        "wandb_app_url": manifest["wandb"]["app_url"],
        "entity": manifest["wandb"]["entity"],
        "project": manifest["wandb"]["project"],
        "group": manifest["wandb"]["group"],
        "program": "h200/run_imagenet100_d2262_p_schedule.sh",
        "console": manifest["wandb"]["console"],
        "relay_protocol_version": manifest["relay"]["protocol_version"],
        "canary": _canary(manifest),
        "runs": {variant: {"501": record} for variant, record in _records(manifest).items()},
    }


def _rendered_outputs(
    manifest: dict[str, Any],
    digest: str,
) -> dict[Path, str]:
    return {RUNTIME_PATH: _json(_runtime(manifest, digest))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    manifest, protocol, digest = _load()
    _validate(manifest, protocol, digest)
    outputs = _rendered_outputs(manifest, digest)
    if args.check:
        stale = [
            str(path.relative_to(ROOT))
            for path, text in outputs.items()
            if not path.is_file() or path.read_text(encoding="utf-8") != text
        ]
        if stale:
            print("stale generated D2262 P-schedule files: " + ", ".join(stale), file=sys.stderr)
            return 1
        return 0
    for path, text in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
