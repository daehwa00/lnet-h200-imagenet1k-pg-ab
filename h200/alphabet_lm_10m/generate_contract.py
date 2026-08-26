#!/usr/bin/env python3
"""Generate the authenticated H200 ALPHABET-LM 10M viability runtime."""

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
MANIFEST_PATH = ROOT / "h200/alphabet_lm_10m/campaign.json"
PROTOCOL_PATH = ROOT / "h200/campaign.json"
RUNTIME_PATH = ROOT / "h200/alphabet_lm_10m/campaign.runtime.json"
HEX_64 = re.compile(r"^[0-9a-f]{64}$")
RUN_LABELS = ("preflight", "alphabet-10m", "mamba-10m")


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _run_id(campaign_id: str, label: str) -> str:
    return hashlib.sha256(f"{campaign_id}\0{label}:seed501".encode()).hexdigest()[:16]


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


def _records(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    names = {
        "preflight": "H200-S501-ALPHABET-LM-Mamba-Preflight-v2",
        "alphabet-10m": "H200-S501-ALPHABET-LM-10M",
        "mamba-10m": "H200-S501-Mamba-10M",
    }
    tags = {
        "preflight": ["H200", "ALPHABET-LM", "Mamba", "preflight-v2"],
        "alphabet-10m": ["H200", "ALPHABET-LM", "FineWeb-Edu", "10M"],
        "mamba-10m": ["H200", "Mamba", "FineWeb-Edu", "10M"],
    }
    return {
        label: {
            "id": _run_id(manifest["campaign_id"], label),
            "display_name": names[label],
            "tags": [*tags[label], "seed501", "authenticated"],
        }
        for label in RUN_LABELS
    }


def _validate(manifest: dict[str, Any], digest: str) -> None:
    preflight = manifest.get("preflight", {})
    dataset = manifest.get("dataset", {})
    training = manifest.get("training", {})
    wandb = manifest.get("wandb", {})
    relay = manifest.get("relay", {})
    if (
        manifest.get("schema") != "lnet.h200.alphabet_lm.viability_10m.v1"
        or manifest.get("campaign_id") != "h200-alphabet-lm-viability-10m-s501-v1"
        or manifest.get("output_namespace") != "alphabet-lm-viability-10m-v1"
        or preflight.get("seed") != 501
        or preflight.get("context_length") != 2048
        or preflight.get("scan_fp32") is not True
        or preflight.get("official_mamba_lm") is not True
        or preflight.get("alphabet_parameters") != 34_794_496
        or preflight.get("mamba_parameters") != 35_425_280
        or dataset.get("revision") != "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
        or dataset.get("filename") != "sample/10BT/000_00000.parquet"
        or dataset.get("size_bytes") != 2_152_819_114
        or dataset.get("sha256")
        != "b1ba7b2ce4cb5ea6ef42dca40263eabb85f37700d01693a68e9b30a31d78e871"
        or dataset.get("vocab_size") != 32_768
        or dataset.get("context_length") != 2_048
        or dataset.get("train_token_limit") != 300_000_000
        or dataset.get("validation_token_limit") != 10_000_000
        or dataset.get("cross_document_packing") is not False
        or training.get("target_tokens") != 10_000_000
        or training.get("horizon_tokens") != 300_000_000
        or training.get("global_sequences") != 32
        or training.get("microbatch") != 8
        or training.get("scan_fp32") is not True
        or training.get("learning_rate") != 3.0e-4
        or training.get("execution") != ["alphabet", "mamba"]
        or wandb.get("sdk_version") != "0.22.3"
        or wandb.get("entity") != "daehwa"
        or wandb.get("project") != "alphabet-lm-viability"
        or wandb.get("group") != "ALPHABET-LM-H200-Viability-10M-S501-v1"
        or wandb.get("console") != "off"
        or relay.get("url") != wandb.get("base_url")
        or relay.get("upstream_origin") != "https://api.wandb.ai"
        or not HEX_64.fullmatch(digest)
    ):
        raise ValueError("invalid ALPHABET-LM H200 10M viability campaign")


def _runtime(manifest: dict[str, Any], digest: str) -> dict[str, Any]:
    wandb, relay = manifest["wandb"], manifest["relay"]
    return {
        "schema": "lnet.h200.alphabet_lm.viability_10m.runtime.v1",
        "campaign_id": manifest["campaign_id"],
        "output_namespace": manifest["output_namespace"],
        "campaign_manifest_sha256": digest,
        "preflight": manifest["preflight"],
        "dataset": manifest["dataset"],
        "training": manifest["training"],
        "wandb_sdk_version": wandb["sdk_version"],
        "wandb_base_url": wandb["base_url"],
        "wandb_app_url": wandb["app_url"],
        "entity": wandb["entity"],
        "project": wandb["project"],
        "group": wandb["group"],
        "console": wandb["console"],
        "program": "h200/run_alphabet_lm_preflight.sh",
        "relay_protocol_version": relay["protocol_version"],
        "runs": _records(manifest),
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
            print("stale generated ALPHABET-LM 10M runtime", file=sys.stderr)
            return 1
        return 0
    RUNTIME_PATH.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
