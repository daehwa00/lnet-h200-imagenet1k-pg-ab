#!/usr/bin/env python3
"""Generate the authenticated H200 ALPHABET-LM preflight runtime."""

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
MANIFEST_PATH = ROOT / "h200/alphabet_lm_preflight/campaign.json"
PROTOCOL_PATH = ROOT / "h200/campaign.json"
RUNTIME_PATH = ROOT / "h200/alphabet_lm_preflight/campaign.runtime.json"
HEX_64 = re.compile(r"^[0-9a-f]{64}$")
LABEL = "ALPHABET-Mamba-Compile-Gate"


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _run_id(campaign_id: str) -> str:
    return hashlib.sha256(f"{campaign_id}\0{LABEL}:seed501".encode()).hexdigest()[:16]


def _load() -> tuple[dict[str, Any], str]:
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
    return manifest, digest


def _validate(manifest: dict[str, Any], digest: str) -> None:
    preflight = manifest.get("preflight", {})
    wandb = manifest.get("wandb", {})
    relay = manifest.get("relay", {})
    if (
        manifest.get("schema") != "lnet.h200.alphabet_lm.preflight.v1"
        or manifest.get("campaign_id") != "h200-alphabet-lm-preflight-s501-v1"
        or manifest.get("output_namespace") != "alphabet-lm-preflight-v1"
        or preflight.get("seed") != 501
        or preflight.get("context_length") != 2048
        or preflight.get("microbatch") != 2
        or preflight.get("repeats") != 2
        or preflight.get("precision") != "bfloat16"
        or preflight.get("alphabet_parameters") != 34_794_496
        or preflight.get("mamba_parameter_tolerance_fraction") != 0.03
        or wandb.get("sdk_version") != "0.22.3"
        or wandb.get("entity") != "daehwa"
        or wandb.get("project") != "alphabet-lm-viability"
        or wandb.get("group") != "ALPHABET-LM-H200-Preflight-S501"
        or wandb.get("console") != "off"
        or relay.get("url") != wandb.get("base_url")
        or relay.get("upstream_origin") != "https://api.wandb.ai"
        or not HEX_64.fullmatch(digest)
    ):
        raise ValueError("invalid ALPHABET-LM H200 preflight campaign")


def _runtime(manifest: dict[str, Any], digest: str) -> dict[str, Any]:
    wandb, relay = manifest["wandb"], manifest["relay"]
    return {
        "schema": "lnet.h200.alphabet_lm.preflight.runtime.v1",
        "campaign_id": manifest["campaign_id"],
        "output_namespace": manifest["output_namespace"],
        "campaign_manifest_sha256": digest,
        "preflight": manifest["preflight"],
        "wandb_sdk_version": wandb["sdk_version"],
        "wandb_base_url": wandb["base_url"],
        "wandb_app_url": wandb["app_url"],
        "entity": wandb["entity"],
        "project": wandb["project"],
        "group": wandb["group"],
        "console": wandb["console"],
        "program": "h200/run_alphabet_lm_preflight.sh",
        "relay_protocol_version": relay["protocol_version"],
        "run": {
            "id": _run_id(manifest["campaign_id"]),
            "display_name": "H200-S501-ALPHABET-LM-Mamba-Preflight",
            "tags": [
                "H200", "ALPHABET-LM", "Mamba", "preflight", "seed501", "authenticated"
            ],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    manifest, digest = _load()
    _validate(manifest, digest)
    rendered = _json(_runtime(manifest, digest))
    if args.check:
        if not RUNTIME_PATH.is_file() or RUNTIME_PATH.read_text(encoding="utf-8") != rendered:
            print("stale generated ALPHABET-LM preflight runtime", file=sys.stderr)
            return 1
        return 0
    RUNTIME_PATH.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
