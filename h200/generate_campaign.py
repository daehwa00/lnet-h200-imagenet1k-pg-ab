"""Validate the H200 campaign manifest and materialize its derived contracts."""

from __future__ import annotations

# ruff: noqa: ANN401, C901, EM101, EM102, I001, PLR0912, T201, TRY003, TRY004
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "h200" / "campaign.json"
RUNTIME_PATH = ROOT / "h200" / "campaign.runtime.json"
RELAY_CONSTANTS_PATH = ROOT / "cloudflare" / "relay" / "src" / "campaign.generated.ts"
WRANGLER_PATH = ROOT / "cloudflare" / "relay" / "wrangler.jsonc"
HEX_16 = re.compile(r"^[0-9a-f]{16}$")
HEX_64 = re.compile(r"^[0-9a-f]{64}$")


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def load_manifest() -> tuple[dict[str, Any], str]:
    raw = MANIFEST_PATH.read_bytes()
    manifest = json.loads(raw)
    if not isinstance(manifest, dict):
        raise ValueError("campaign manifest must be a JSON object")
    return manifest, hashlib.sha256(raw).hexdigest()


def validate_manifest(m: dict[str, Any]) -> None:
    if m.get("schema_version") != 3:
        raise ValueError("schema_version must be 3")
    if not str(m.get("campaign_id", "")).endswith("-v3"):
        raise ValueError("campaign_id must be a v3 identifier")
    if not str(m.get("output_namespace", "")).endswith("-v3"):
        raise ValueError("output_namespace must be a v3 namespace")

    dataset = m["dataset"]
    expected_dataset = (1281167, 50000, 1000)
    actual_dataset = (
        dataset["train_images"],
        dataset["validation_images"],
        dataset["classes"],
    )
    if actual_dataset != expected_dataset:
        raise ValueError(f"ImageNet-1K counts must be {expected_dataset}, got {actual_dataset}")

    wandb = m["wandb"]
    relay = m["relay"]
    if wandb["sdk_version"] != "0.22.3" or wandb["console"] != "off":
        raise ValueError("the traced SDK must be 0.22.3 with console=off")
    if relay["upstream_origin"] != "https://api.wandb.ai":
        raise ValueError("relay upstream is immutable")
    if relay["url"] != wandb["base_url"]:
        raise ValueError("W&B base URL and relay URL must match")
    if not relay["url"].startswith("https://"):
        raise ValueError("relay URL must use HTTPS")

    run_records = [*m["runs"].values(), relay["permanent_canary"]]
    run_ids = [record["id"] for record in run_records]
    if any(not HEX_16.fullmatch(run_id) for run_id in run_ids):
        raise ValueError("all W&B run IDs must be lowercase 16-character hex")
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("production and canary run IDs must be unique")
    if len({record["display_name"] for record in run_records}) != len(run_records):
        raise ValueError("display names must be unique")

    hashes = list(m["graphql_operations"].values())
    if len(hashes) != 7 or any(not HEX_64.fullmatch(value) for value in hashes):
        raise ValueError("exactly seven traced GraphQL SHA-256 hashes are required")
    if len(set(hashes)) != len(hashes):
        raise ValueError("GraphQL hashes must be unique")

    protocol = m["protocol"]
    if "output.log" in protocol["run_files"] or "output.log" in protocol["stream_files"]:
        raise ValueError("console=off protocol must not permit output.log")
    if not set(protocol["no_variable_operations"]).issubset(m["graphql_operations"]):
        raise ValueError("no-variable operation is missing a traced hash")
    if set(protocol["graphql_envelope_keys"]) != set(m["graphql_operations"]):
        raise ValueError("every traced operation needs an exact envelope contract")
    rate = relay["rate_limit"]
    if not str(rate["namespace_id"]).isdigit() or rate["period_seconds"] not in (10, 60):
        raise ValueError("invalid Cloudflare rate-limit contract")
    if not (1 <= rate["limit"] <= 10000):
        raise ValueError("rate limit must be bounded")


def runtime_contract(m: dict[str, Any], manifest_sha256: str) -> dict[str, Any]:
    pg = m["runs"]["pg"]
    no_pg = m["runs"]["no_pg"]
    canary = m["relay"]["permanent_canary"]
    dataset = m["dataset"]
    wandb = m["wandb"]
    return {
        "schema_version": m["schema_version"],
        "campaign_id": m["campaign_id"],
        "output_namespace": m["output_namespace"],
        "manifest_sha256": manifest_sha256,
        "dataset_name": dataset["name"],
        "dataset_train_images": dataset["train_images"],
        "dataset_validation_images": dataset["validation_images"],
        "dataset_classes": dataset["classes"],
        "dataset_identity_algorithm": dataset["identity_algorithm"],
        "wandb_sdk_version": wandb["sdk_version"],
        "wandb_base_url": wandb["base_url"],
        "wandb_app_url": wandb["app_url"],
        "entity": wandb["entity"],
        "project": wandb["project"],
        "group": wandb["group"],
        "console": wandb["console"],
        "relay_url": m["relay"]["url"],
        "relay_protocol_version": m["relay"]["protocol_version"],
        "pg_run_id": pg["id"],
        "pg_display_name": pg["display_name"],
        "pg_tags": pg["tags"],
        "no_pg_run_id": no_pg["id"],
        "no_pg_display_name": no_pg["display_name"],
        "no_pg_tags": no_pg["tags"],
        "canary_run_id": canary["id"],
        "canary_display_name": canary["display_name"],
    }


def relay_contract(m: dict[str, Any], manifest_sha256: str) -> dict[str, Any]:
    runs = [*m["runs"].values(), m["relay"]["permanent_canary"]]
    return {
        "schemaVersion": m["schema_version"],
        "campaignId": m["campaign_id"],
        "manifestSha256": manifest_sha256,
        "protocolVersion": m["relay"]["protocol_version"],
        "sdkVersion": m["wandb"]["sdk_version"],
        "upstreamOrigin": m["relay"]["upstream_origin"],
        "entity": m["wandb"]["entity"],
        "project": m["wandb"]["project"],
        "group": m["wandb"]["group"],
        "maxGraphqlBodyBytes": m["relay"]["max_graphql_body_bytes"],
        "maxFileStreamBodyBytes": m["relay"]["max_file_stream_body_bytes"],
        "graphqlOperations": m["graphql_operations"],
        "graphqlEnvelopeKeys": m["protocol"]["graphql_envelope_keys"],
        "noVariableOperations": m["protocol"]["no_variable_operations"],
        "runFiles": m["protocol"]["run_files"],
        "streamFiles": m["protocol"]["stream_files"],
        "upsertBucketVariableKeys": m["protocol"]["upsert_bucket_variable_keys"],
        "allowedStates": m["protocol"]["allowed_states"],
        "runsById": {
            run["id"]: {"displayName": run["display_name"], "tags": run["tags"]}
            for run in runs
        },
    }


def render_relay_constants(m: dict[str, Any], manifest_sha256: str) -> str:
    payload = json.dumps(relay_contract(m, manifest_sha256), indent=2, sort_keys=True)
    return (
        "// Generated by h200/generate_campaign.py; do not edit.\n"
        f"export const CAMPAIGN = {payload} as const;\n"
    )


def render_wrangler(m: dict[str, Any]) -> str:
    rate = m["relay"]["rate_limit"]
    config = {
        "$schema": "./node_modules/wrangler/config-schema.json",
        "name": m["relay"]["worker_name"],
        "main": "src/index.ts",
        "compatibility_date": "2026-08-19",
        "compatibility_flags": ["nodejs_compat"],
        "workers_dev": True,
        "ratelimits": [
            {
                "name": "RELAY_RATE_LIMITER",
                "namespace_id": rate["namespace_id"],
                "simple": {"limit": rate["limit"], "period": rate["period_seconds"]},
            }
        ],
        "observability": {
            "enabled": True,
            "logs": {"enabled": True, "head_sampling_rate": 1},
            "traces": {"enabled": True, "head_sampling_rate": 1},
        },
    }
    return _json(config)


def rendered_outputs(m: dict[str, Any], manifest_sha256: str) -> dict[Path, str]:
    return {
        RUNTIME_PATH: _json(runtime_contract(m, manifest_sha256)),
        RELAY_CONSTANTS_PATH: render_relay_constants(m, manifest_sha256),
        WRANGLER_PATH: render_wrangler(m),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if generated files drift")
    parser.add_argument("--print", choices=("runtime", "relay", "wrangler"))
    args = parser.parse_args()
    manifest, manifest_sha256 = load_manifest()
    validate_manifest(manifest)
    outputs = rendered_outputs(manifest, manifest_sha256)
    if args.print:
        selected = {
            "runtime": RUNTIME_PATH,
            "relay": RELAY_CONSTANTS_PATH,
            "wrangler": WRANGLER_PATH,
        }[args.print]
        sys.stdout.write(outputs[selected])
        return 0
    if args.check:
        stale = [
            str(path.relative_to(ROOT))
            for path, text in outputs.items()
            if not path.is_file() or path.read_text() != text
        ]
        if stale:
            print("stale generated campaign files: " + ", ".join(stale), file=sys.stderr)
            return 1
        return 0
    for path, text in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
