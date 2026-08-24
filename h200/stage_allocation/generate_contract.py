#!/usr/bin/env python3
# ruff: noqa: ANN401, T201
# pyright: reportExplicitAny=false
"""Validate and generate the H200 ImageNet-100 stage-allocation contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "h200/stage_allocation/campaign.json"
SUPPLEMENTAL_MANIFEST_PATH = ROOT / "h200/d2262_p_schedule/campaign.json"
READER_WL_MANIFEST_PATH = ROOT / "h200/reader_wl/campaign.json"
K64_P_ALLOCATION_MANIFEST_PATH = ROOT / "h200/k64_p_allocation/campaign.json"
K64_P_SMALL_FACTORIAL_MANIFEST_PATH = ROOT / "h200/k64_p_small_factorial/campaign.json"
K64_P_DEPTH_INTERACTION_LEGACY_MANIFEST_PATH = (
    ROOT / "h200/k64_p_depth_interaction/campaign.v1.json"
)
K64_P_DEPTH_INTERACTION_MANIFEST_PATH = ROOT / "h200/k64_p_depth_interaction/campaign.json"
K_FAMILY_XL_MANIFEST_PATH = ROOT / "h200/k_family_xl/campaign.json"
K_FAMILY_P_REFINEMENT_MANIFEST_PATH = ROOT / "h200/k_family_p_refinement/campaign.json"
PROTOCOL_PATH = ROOT / "h200/campaign.json"
RUNTIME_PATH = ROOT / "h200/stage_allocation/campaign.runtime.json"
RELAY_CONSTANTS_PATH = ROOT / "cloudflare/stage-allocation-relay/src/campaign.generated.ts"
WRANGLER_PATH = ROOT / "cloudflare/stage-allocation-relay/wrangler.jsonc"
HEX_64 = re.compile(r"^[0-9a-f]{64}$")
D2262_P_SCHEDULE_VARIANTS = (
    "A-K128-P160-160-160-128-D2262",
    "B-K128-P160-160-192-128-D2262",
    "C-K128-P160-192-160-128-D2262",
    "D-K128-P160-192-192-128-D2262",
    "E-K128-P128-192-192-128-D2262",
    "F-K128-P128-160-192-128-D2262",
)
READER_WL_VARIANTS = (
    "StrictReader-K96-128-128-128-P128-192-192-128-D2262",
    "WLReader-K96-128-128-128-P128-192-192-128-D2262",
)
K64_P_ALLOCATION_VARIANTS = (
    "K64-P96-160-160-128-D2262",
    "K64-P96-160-192-96-D2262",
)
K64_P_SMALL_FACTORIAL_VARIANTS = (
    "K64-P96-128-96-96-D2262",
    "K64-P96-128-128-64-D2262",
)
K64_P_DEPTH_INTERACTION_VARIANTS = (
    "K64-P96-128-96-96-D2263",
    "K64-P96-128-128-96-D2283",
)
K_FAMILY_XL_VARIANTS = (
    "XL-K96-U1",
    "XL-K96-U125",
    "XL-K96-Shaped",
    "XL-K96-Rich",
)
K_FAMILY_P_REFINEMENT_VARIANTS = (
    "M-K48-P80-80-80-80",
    "L-K64-P80-96-80-80",
    "L-K64-P80-96-96-80",
    "S-K32-P48-48-64-48",
    "XL-K96-P128-144-128-128",
)


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
            "group": manifest["wandb"]["group"],
            "project": manifest["wandb"]["project"],
            "program": "h200/run_imagenet100_stage_allocation.sh",
            "tags": [
                "H200",
                "ImageNet-100",
                "stage-allocation",
                "seed501",
                "authenticated",
            ],
        }
    return records


def _canary(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": _run_id(manifest["campaign_id"], "permanent-canary"),
        "display_name": "H200-I100-stage-relay-permanent-canary-v3",
        "group": manifest["wandb"]["group"],
        "project": manifest["wandb"]["project"],
        "program": "h200/run_imagenet100_stage_allocation.sh",
        "tags": ["H200", "ImageNet-100", "relay-canary", "authenticated"],
    }


def _supplemental_manifest() -> dict[str, Any]:
    return json.loads(SUPPLEMENTAL_MANIFEST_PATH.read_text(encoding="utf-8"))


def _supplemental_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    campaign_id = manifest["campaign_id"]
    records = []
    for index, variant in enumerate(manifest["training"]["variants"], start=1):
        records.append(
            {
                "id": _run_id(campaign_id, f"{variant}:seed501"),
                "display_name": f"H200-I100-S501-{index:02d}-{variant}",
                "group": manifest["wandb"]["group"],
                "project": manifest["wandb"]["project"],
                "program": "h200/run_imagenet100_d2262_p_schedule.sh",
                "tags": [
                    "H200",
                    "ImageNet-100",
                    "D2262",
                    "P-schedule",
                    "seed501",
                    "authenticated",
                ],
            }
        )
    records.append(
        {
            "id": _run_id(campaign_id, "permanent-canary"),
            "display_name": "H200-I100-D2262-P-schedule-relay-canary-v2",
            "group": manifest["wandb"]["group"],
            "project": manifest["wandb"]["project"],
            "program": "h200/run_imagenet100_d2262_p_schedule.sh",
            "tags": ["H200", "ImageNet-100", "D2262", "P-schedule", "relay-canary"],
        }
    )
    return records


def _validate_supplemental(
    primary: dict[str, Any],
    supplemental: dict[str, Any],
) -> None:
    training = supplemental.get("training", {})
    wandb = supplemental.get("wandb", {})
    relay = supplemental.get("relay", {})
    if (
        supplemental.get("schema") != "lnet.h200.imagenet100.d2262_p_schedule.v2"
        or supplemental.get("campaign_id") != "h200-imagenet100-d2262-p-schedule-s501-v2"
        or supplemental.get("output_namespace")
        != "lnet-h200-imagenet100-d2262-p-schedule-v2"
        or tuple(training.get("variants", ())) != D2262_P_SCHEDULE_VARIANTS
        or training.get("seed") != 501
        or training.get("epochs") != 100
        or training.get("batch_size") != 128
        or training.get("precision") != "bfloat16"
        or training.get("execution") != "one_model_to_epoch_100_then_next"
        or wandb.get("group") != "h200-imagenet100-d2262-p-schedule-s501-v2"
        or wandb.get("group") == primary["wandb"]["group"]
        or wandb.get("base_url") != primary["wandb"]["base_url"]
        or wandb.get("entity") != primary["wandb"]["entity"]
        or wandb.get("project") != primary["wandb"]["project"]
        or relay.get("url") != primary["relay"]["url"]
        or relay.get("worker_name") != primary["relay"]["worker_name"]
        or relay.get("protocol_version") != primary["relay"]["protocol_version"]
    ):
        raise ValueError("invalid supplemental D2262 P-schedule relay contract")


def _reader_wl_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    campaign_id = manifest["campaign_id"]
    records = [
        {
            "id": _run_id(campaign_id, f"{variant}:seed501"),
            "display_name": f"H200-I100-S501-{index:02d}-{variant}",
            "group": manifest["wandb"]["group"],
            "project": manifest["wandb"]["project"],
            "program": "h200/run_imagenet100_reader_wl.sh",
            "tags": [
                "H200",
                "ImageNet-100",
                "Reader-study",
                "strict-vs-WL",
                "seed501",
                "authenticated",
            ],
        }
        for index, variant in enumerate(manifest["training"]["variants"], start=1)
    ]
    records.append(
        {
            "id": _run_id(campaign_id, "permanent-canary"),
            "display_name": "H200-I100-Reader-WL-relay-canary-v1",
            "group": manifest["wandb"]["group"],
            "project": manifest["wandb"]["project"],
            "program": "h200/run_imagenet100_reader_wl.sh",
            "tags": [
                "H200",
                "ImageNet-100",
                "Reader-study",
                "strict-vs-WL",
                "relay-canary",
            ],
        }
    )
    return records


def _validate_reader_wl(primary: dict[str, Any], manifest: dict[str, Any]) -> None:
    training = manifest.get("training", {})
    wandb = manifest.get("wandb", {})
    relay = manifest.get("relay", {})
    occupied_groups = {
        primary["wandb"]["group"],
        _supplemental_manifest()["wandb"]["group"],
    }
    if (
        manifest.get("schema") != "lnet.h200.imagenet100.reader_wl.v1"
        or manifest.get("campaign_id") != "h200-imagenet100-reader-wl-s501-v1"
        or manifest.get("output_namespace") != "lnet-h200-imagenet100-reader-wl-v1"
        or tuple(training.get("variants", ())) != READER_WL_VARIANTS
        or training.get("seed") != 501
        or training.get("epochs") != 100
        or training.get("batch_size") != 128
        or training.get("precision") != "bfloat16"
        or training.get("execution") != "one_model_to_epoch_100_then_next"
        or wandb.get("group") != "h200-imagenet100-reader-wl-s501-v1"
        or wandb.get("group") in occupied_groups
        or wandb.get("base_url") != primary["wandb"]["base_url"]
        or wandb.get("entity") != primary["wandb"]["entity"]
        or wandb.get("project") != primary["wandb"]["project"]
        or relay.get("url") != primary["relay"]["url"]
        or relay.get("worker_name") != primary["relay"]["worker_name"]
        or relay.get("protocol_version") != primary["relay"]["protocol_version"]
    ):
        raise ValueError("invalid supplemental strict-versus-WL Reader relay contract")


def _k64_p_allocation_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    campaign_id = manifest["campaign_id"]
    records = [
        {
            "id": _run_id(campaign_id, f"{variant}:seed501"),
            "display_name": f"H200-I100-S501-{index:02d}-{variant}",
            "group": manifest["wandb"]["group"],
            "project": manifest["wandb"]["project"],
            "program": "h200/run_imagenet100_k64_p_allocation.sh",
            "tags": [
                "H200",
                "ImageNet-100",
                "K64",
                "P-allocation",
                "D2262",
                "seed501",
                "authenticated",
            ],
        }
        for index, variant in enumerate(manifest["training"]["variants"], start=1)
    ]
    records.append(
        {
            "id": _run_id(campaign_id, "permanent-canary"),
            "display_name": "H200-I100-K64-P-allocation-D2262-relay-canary-v1",
            "group": manifest["wandb"]["group"],
            "project": manifest["wandb"]["project"],
            "program": "h200/run_imagenet100_k64_p_allocation.sh",
            "tags": [
                "H200",
                "ImageNet-100",
                "K64",
                "P-allocation",
                "D2262",
                "relay-canary",
            ],
        }
    )
    return records


def _validate_k64_p_allocation(primary: dict[str, Any], manifest: dict[str, Any]) -> None:
    training = manifest.get("training", {})
    wandb = manifest.get("wandb", {})
    relay = manifest.get("relay", {})
    if (
        manifest.get("schema") != "lnet.h200.imagenet100.k64_p_allocation.v1"
        or manifest.get("campaign_id")
        != "h200-imagenet100-k64-p-allocation-d2262-s501-v1"
        or manifest.get("output_namespace")
        != "lnet-h200-imagenet100-k64-p-allocation-d2262-v1"
        or tuple(training.get("variants", ())) != K64_P_ALLOCATION_VARIANTS
        or training.get("seed") != 501
        or training.get("epochs") != 100
        or training.get("batch_size") != 128
        or training.get("precision") != "bfloat16"
        or training.get("execution") != "one_model_to_epoch_100_then_next"
        or wandb.get("entity") != primary["wandb"]["entity"]
        or wandb.get("project") != "alphabet2d-imagenet100"
        or wandb.get("group") != "R2K3-K64-PAllocation-D2262-H200-S501"
        or wandb.get("base_url") != primary["wandb"]["base_url"]
        or relay.get("url") != primary["relay"]["url"]
        or relay.get("worker_name") != primary["relay"]["worker_name"]
        or relay.get("protocol_version") != primary["relay"]["protocol_version"]
    ):
        raise ValueError("invalid supplemental K64 P-allocation relay contract")


def _k64_p_small_factorial_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    campaign_id = manifest["campaign_id"]
    records = [
        {
            "id": _run_id(campaign_id, f"{variant}:seed501"),
            "display_name": f"H200-I100-S501-{index:02d}-{variant}",
            "group": manifest["wandb"]["group"],
            "project": manifest["wandb"]["project"],
            "program": "h200/run_imagenet100_k64_p_small_factorial.sh",
            "tags": [
                "H200",
                "ImageNet-100",
                "K64",
                "P-small-factorial",
                "D2262",
                "seed501",
                "authenticated",
            ],
        }
        for index, variant in enumerate(manifest["training"]["variants"], start=1)
    ]
    records.append(
        {
            "id": _run_id(campaign_id, "permanent-canary"),
            "display_name": "H200-I100-K64-P-small-factorial-D2262-relay-canary-v1",
            "group": manifest["wandb"]["group"],
            "project": manifest["wandb"]["project"],
            "program": "h200/run_imagenet100_k64_p_small_factorial.sh",
            "tags": [
                "H200",
                "ImageNet-100",
                "K64",
                "P-small-factorial",
                "D2262",
                "relay-canary",
            ],
        }
    )
    return records


def _validate_k64_p_small_factorial(
    primary: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    training = manifest.get("training", {})
    wandb = manifest.get("wandb", {})
    relay = manifest.get("relay", {})
    if (
        manifest.get("schema") != "lnet.h200.imagenet100.k64_p_small_factorial.v1"
        or manifest.get("campaign_id")
        != "h200-imagenet100-k64-p-small-factorial-d2262-s501-v1"
        or manifest.get("output_namespace")
        != "lnet-h200-imagenet100-k64-p-small-factorial-d2262-v1"
        or tuple(training.get("variants", ())) != K64_P_SMALL_FACTORIAL_VARIANTS
        or training.get("seed") != 501
        or training.get("epochs") != 100
        or training.get("batch_size") != 128
        or training.get("precision") != "bfloat16"
        or training.get("execution") != "one_model_to_epoch_100_then_next"
        or wandb.get("entity") != primary["wandb"]["entity"]
        or wandb.get("project") != "alphabet2d-imagenet100"
        or wandb.get("group") != "R2K3-K64-PSmallFactorial-D2262-H200-S501"
        or wandb.get("base_url") != primary["wandb"]["base_url"]
        or relay.get("url") != primary["relay"]["url"]
        or relay.get("worker_name") != primary["relay"]["worker_name"]
        or relay.get("protocol_version") != primary["relay"]["protocol_version"]
    ):
        raise ValueError("invalid supplemental K64 P-small-factorial relay contract")


def _k64_p_depth_interaction_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    campaign_id = manifest["campaign_id"]
    return [
        {
            "id": _run_id(campaign_id, f"{variant}:seed501"),
            "display_name": f"H200-I100-S501-{index:02d}-{variant}",
            "group": manifest["wandb"]["group"],
            "project": manifest["wandb"]["project"],
            "program": "h200/run_imagenet100_k64_p_depth_interaction.sh",
            "tags": [
                "H200",
                "ImageNet-100",
                "K64",
                "P-depth-interaction",
                "DepthInteraction",
                "seed501",
                "authenticated",
            ],
        }
        for index, variant in enumerate(manifest["training"]["variants"], start=1)
    ]


def _validate_k64_p_depth_interaction(
    primary: dict[str, Any],
    manifest: dict[str, Any],
    *,
    legacy: bool,
) -> None:
    training = manifest.get("training", {})
    wandb = manifest.get("wandb", {})
    relay = manifest.get("relay", {})
    suffix = "v1" if legacy else "v2"
    expected_group = (
        "R2K3-K64-PDepthInteraction-H200-S501"
        if legacy
        else "R2K3-K64-PDepthInteraction-H200-S501-v2"
    )
    if (
        manifest.get("schema")
        != f"lnet.h200.imagenet100.k64_p_depth_interaction.{suffix}"
        or manifest.get("campaign_id")
        != f"h200-imagenet100-k64-p-depth-interaction-s501-{suffix}"
        or manifest.get("output_namespace")
        != f"lnet-h200-imagenet100-k64-p-depth-interaction-{suffix}"
        or tuple(training.get("variants", ())) != K64_P_DEPTH_INTERACTION_VARIANTS
        or training.get("seed") != 501
        or training.get("epochs") != 100
        or training.get("batch_size") != 128
        or training.get("precision") != "bfloat16"
        or training.get("execution") != "one_model_to_epoch_100_then_next"
        or wandb.get("entity") != primary["wandb"]["entity"]
        or wandb.get("project") != "alphabet2d-imagenet100"
        or wandb.get("group") != expected_group
        or wandb.get("base_url") != primary["wandb"]["base_url"]
        or relay.get("url") != primary["relay"]["url"]
        or relay.get("worker_name") != primary["relay"]["worker_name"]
        or relay.get("protocol_version") != primary["relay"]["protocol_version"]
    ):
        raise ValueError("invalid supplemental K64 P-depth-interaction relay contract")


def _k_family_xl_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    campaign_id = manifest["campaign_id"]
    return [
        {
            "id": _run_id(campaign_id, f"{variant}:seed501"),
            "display_name": f"H200-I100-S501-{index:02d}-{variant}",
            "group": manifest["wandb"]["group"],
            "project": manifest["wandb"]["project"],
            "program": "h200/run_imagenet100_k_family_xl.sh",
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
    ]


def _validate_k_family_xl(primary: dict[str, Any], manifest: dict[str, Any]) -> None:
    training = manifest.get("training", {})
    wandb = manifest.get("wandb", {})
    relay = manifest.get("relay", {})
    if (
        manifest.get("schema") != "lnet.h200.imagenet100.k_family_xl.v1"
        or manifest.get("campaign_id") != "h200-imagenet100-k-family-xl-s501-v1"
        or manifest.get("output_namespace") != "lnet-h200-imagenet100-k-family-xl-v1"
        or tuple(training.get("variants", ())) != K_FAMILY_XL_VARIANTS
        or training.get("seed") != 501
        or training.get("epochs") != 100
        or training.get("batch_size") != 128
        or training.get("precision") != "bfloat16"
        or training.get("execution") != "one_model_to_epoch_100_then_next"
        or wandb.get("entity") != primary["wandb"]["entity"]
        or wandb.get("project") != "alphabet2d-imagenet100"
        or wandb.get("group") != "R2K3-KFamily-XL-H200-S501"
        or wandb.get("base_url") != primary["wandb"]["base_url"]
        or relay.get("url") != primary["relay"]["url"]
        or relay.get("worker_name") != primary["relay"]["worker_name"]
        or relay.get("protocol_version") != primary["relay"]["protocol_version"]
    ):
        raise ValueError("invalid supplemental K96/XL family relay contract")


def _k_family_p_refinement_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": _run_id(manifest["campaign_id"], f"{variant}:seed501"),
            "display_name": f"H200-I100-S501-{index:02d}-{variant}",
            "group": manifest["wandb"]["group"],
            "project": manifest["wandb"]["project"],
            "program": "h200/run_imagenet100_k_family_p_refinement.sh",
            "tags": [
                "H200", "ImageNet-100", "K-family", "P-refinement",
                "D2262", "seed501", "authenticated",
            ],
        }
        for index, variant in enumerate(manifest["training"]["variants"], start=1)
    ]


def _validate_k_family_p_refinement(
    primary: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    training = manifest.get("training", {})
    wandb = manifest.get("wandb", {})
    relay = manifest.get("relay", {})
    if (
        manifest.get("schema") != "lnet.h200.imagenet100.k_family_p_refinement.v1"
        or manifest.get("campaign_id")
        != "h200-imagenet100-k-family-p-refinement-s501-v1"
        or manifest.get("output_namespace")
        != "lnet-h200-imagenet100-k-family-p-refinement-v1"
        or tuple(training.get("variants", ())) != K_FAMILY_P_REFINEMENT_VARIANTS
        or training.get("seed") != 501
        or training.get("epochs") != 100
        or training.get("batch_size") != 128
        or training.get("precision") != "bfloat16"
        or wandb.get("entity") != primary["wandb"]["entity"]
        or wandb.get("project") != "alphabet2d-imagenet100"
        or wandb.get("group") != "R2K3-KFamily-PRefinement-H200-S501"
        or wandb.get("base_url") != primary["wandb"]["base_url"]
        or relay.get("url") != primary["relay"]["url"]
        or relay.get("worker_name") != primary["relay"]["worker_name"]
        or relay.get("protocol_version") != primary["relay"]["protocol_version"]
    ):
        raise ValueError("invalid supplemental K-family P-refinement relay contract")


def _validate(manifest: dict[str, Any], protocol: dict[str, Any], digest: str) -> None:
    if manifest.get("schema") != "lnet.h200.imagenet100.stage_allocation.v1":
        raise ValueError("invalid stage-allocation campaign schema")
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
        or len(variants) != 13
        or len(set(variants)) != 13
    ):
        raise ValueError("stage-allocation training matrix changed")
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
    supplemental = _supplemental_manifest()
    _validate_supplemental(manifest, supplemental)
    reader_wl = json.loads(READER_WL_MANIFEST_PATH.read_text(encoding="utf-8"))
    _validate_reader_wl(manifest, reader_wl)
    k64_p_allocation = json.loads(
        K64_P_ALLOCATION_MANIFEST_PATH.read_text(encoding="utf-8")
    )
    _validate_k64_p_allocation(manifest, k64_p_allocation)
    k64_p_small_factorial = json.loads(
        K64_P_SMALL_FACTORIAL_MANIFEST_PATH.read_text(encoding="utf-8")
    )
    _validate_k64_p_small_factorial(manifest, k64_p_small_factorial)
    legacy_k64_p_depth_interaction = json.loads(
        K64_P_DEPTH_INTERACTION_LEGACY_MANIFEST_PATH.read_text(encoding="utf-8")
    )
    _validate_k64_p_depth_interaction(
        manifest,
        legacy_k64_p_depth_interaction,
        legacy=True,
    )
    k64_p_depth_interaction = json.loads(
        K64_P_DEPTH_INTERACTION_MANIFEST_PATH.read_text(encoding="utf-8")
    )
    _validate_k64_p_depth_interaction(manifest, k64_p_depth_interaction, legacy=False)
    k_family_xl = json.loads(K_FAMILY_XL_MANIFEST_PATH.read_text(encoding="utf-8"))
    _validate_k_family_xl(manifest, k_family_xl)
    k_family_p_refinement = json.loads(
        K_FAMILY_P_REFINEMENT_MANIFEST_PATH.read_text(encoding="utf-8")
    )
    _validate_k_family_p_refinement(manifest, k_family_p_refinement)
    all_records = [*records, *_supplemental_records(supplemental)]
    all_records.extend(_reader_wl_records(reader_wl))
    all_records.extend(_k64_p_allocation_records(k64_p_allocation))
    all_records.extend(_k64_p_small_factorial_records(k64_p_small_factorial))
    all_records.extend(_k64_p_depth_interaction_records(legacy_k64_p_depth_interaction))
    all_records.extend(_k64_p_depth_interaction_records(k64_p_depth_interaction))
    all_records.extend(_k_family_xl_records(k_family_xl))
    all_records.extend(_k_family_p_refinement_records(k_family_p_refinement))
    if len({record["id"] for record in all_records}) != len(all_records):
        raise ValueError("combined relay run IDs are not unique")


def _runtime(manifest: dict[str, Any], digest: str) -> dict[str, Any]:
    return {
        "schema": "lnet.h200.imagenet100.stage_allocation.runtime.v1",
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
        "console": manifest["wandb"]["console"],
        "relay_protocol_version": manifest["relay"]["protocol_version"],
        "canary": _canary(manifest),
        "runs": {variant: {"501": record} for variant, record in _records(manifest).items()},
    }


def _relay(
    manifest: dict[str, Any],
    protocol: dict[str, Any],
    digest: str,
) -> dict[str, Any]:
    records = [
        *_records(manifest).values(),
        _canary(manifest),
        *_supplemental_records(_supplemental_manifest()),
        *_reader_wl_records(json.loads(READER_WL_MANIFEST_PATH.read_text(encoding="utf-8"))),
        *_k64_p_allocation_records(
            json.loads(K64_P_ALLOCATION_MANIFEST_PATH.read_text(encoding="utf-8"))
        ),
        *_k64_p_small_factorial_records(
            json.loads(K64_P_SMALL_FACTORIAL_MANIFEST_PATH.read_text(encoding="utf-8"))
        ),
        *_k64_p_depth_interaction_records(
            json.loads(
                K64_P_DEPTH_INTERACTION_LEGACY_MANIFEST_PATH.read_text(encoding="utf-8")
            )
        ),
        *_k64_p_depth_interaction_records(
            json.loads(K64_P_DEPTH_INTERACTION_MANIFEST_PATH.read_text(encoding="utf-8"))
        ),
        *_k_family_xl_records(
            json.loads(K_FAMILY_XL_MANIFEST_PATH.read_text(encoding="utf-8"))
        ),
        *_k_family_p_refinement_records(
            json.loads(K_FAMILY_P_REFINEMENT_MANIFEST_PATH.read_text(encoding="utf-8"))
        ),
    ]
    source = protocol["protocol"]
    authorization_digest = hashlib.sha256(
        MANIFEST_PATH.read_bytes()
        + b"\0"
        + SUPPLEMENTAL_MANIFEST_PATH.read_bytes()
        + b"\0"
        + READER_WL_MANIFEST_PATH.read_bytes()
        + b"\0"
        + K64_P_ALLOCATION_MANIFEST_PATH.read_bytes()
        + b"\0"
        + K64_P_SMALL_FACTORIAL_MANIFEST_PATH.read_bytes()
        + b"\0"
        + K64_P_DEPTH_INTERACTION_LEGACY_MANIFEST_PATH.read_bytes()
        + b"\0"
        + K64_P_DEPTH_INTERACTION_MANIFEST_PATH.read_bytes()
        + b"\0"
        + K_FAMILY_XL_MANIFEST_PATH.read_bytes()
        + b"\0"
        + K_FAMILY_P_REFINEMENT_MANIFEST_PATH.read_bytes()
        + b"\0"
        + json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schemaVersion": 1,
        "campaignId": manifest["campaign_id"],
        "manifestSha256": digest,
        "authorizationManifestSha256": authorization_digest,
        "protocolVersion": manifest["relay"]["protocol_version"],
        "sdkVersion": manifest["wandb"]["sdk_version"],
        "upstreamOrigin": manifest["relay"]["upstream_origin"],
        "entity": manifest["wandb"]["entity"],
        "project": manifest["wandb"]["project"],
        "group": manifest["wandb"]["group"],
        "program": "h200/run_imagenet100_stage_allocation.sh",
        "maxGraphqlBodyBytes": manifest["relay"]["max_graphql_body_bytes"],
        "maxFileStreamBodyBytes": manifest["relay"]["max_file_stream_body_bytes"],
        "graphqlOperations": protocol["graphql_operations"],
        "graphqlEnvelopeKeys": source["graphql_envelope_keys"],
        "noVariableOperations": source["no_variable_operations"],
        "runFiles": source["run_files"],
        "streamFiles": source["stream_files"],
        "upsertBucketVariableKeys": source["upsert_bucket_variable_keys"],
        "allowedStates": source["allowed_states"],
        "runsById": {
            record["id"]: {
                "displayName": record["display_name"],
                "group": record["group"],
                "project": record["project"],
                "program": record["program"],
                "tags": record["tags"],
            }
            for record in records
        },
    }


def _rendered_outputs(
    manifest: dict[str, Any],
    protocol: dict[str, Any],
    digest: str,
) -> dict[Path, str]:
    relay = json.dumps(_relay(manifest, protocol, digest), indent=2, sort_keys=True)
    wrangler = {
        "$schema": "./node_modules/wrangler/config-schema.json",
        "name": manifest["relay"]["worker_name"],
        "main": "src/index.ts",
        "compatibility_date": "2026-08-21",
        "compatibility_flags": ["nodejs_compat"],
        "workers_dev": True,
        "ratelimits": [
            {
                "name": "RELAY_RATE_LIMITER",
                "namespace_id": manifest["relay"]["rate_limit_namespace_id"],
                "simple": {"limit": 600, "period": 60},
            }
        ],
        "observability": {
            "enabled": True,
            "logs": {"enabled": True, "head_sampling_rate": 1},
            "traces": {"enabled": True, "head_sampling_rate": 1},
        },
    }
    return {
        RUNTIME_PATH: _json(_runtime(manifest, digest)),
        RELAY_CONSTANTS_PATH: (
            "// Generated by h200/stage_allocation/generate_contract.py; do not edit.\n"
            f"export const CAMPAIGN = {relay} as const;\n"
        ),
        WRANGLER_PATH: _json(wrangler),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    manifest, protocol, digest = _load()
    _validate(manifest, protocol, digest)
    outputs = _rendered_outputs(manifest, protocol, digest)
    if args.check:
        stale = [
            str(path.relative_to(ROOT))
            for path, text in outputs.items()
            if not path.is_file() or path.read_text(encoding="utf-8") != text
        ]
        if stale:
            print("stale generated stage-allocation files: " + ", ".join(stale), file=sys.stderr)
            return 1
        return 0
    for path, text in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
