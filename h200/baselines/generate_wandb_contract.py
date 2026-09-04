"""Generate the baseline and internal-control W&B relay contract."""

from __future__ import annotations

# pyright: reportAny=false, reportExplicitAny=false
import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN_PATH = ROOT / "h200" / "baselines" / "campaign.json"
PROTOCOL_PATH = ROOT / "h200" / "campaign.json"
RUNTIME_PATH = ROOT / "h200" / "baselines" / "wandb.runtime.json"
BASELINE_RELAY = ROOT / "cloudflare" / "baseline-relay"
GENERATED_CAMPAIGN_PATH = BASELINE_RELAY / "src" / "campaign.generated.ts"
GENERATED_INDEX_PATH = BASELINE_RELAY / "src" / "index.ts"
GENERATED_TEST_PATH = BASELINE_RELAY / "test" / "index.test.ts"
WRANGLER_PATH = BASELINE_RELAY / "wrangler.jsonc"
ENTITY = "daehwa"
PROJECT = "alphabet2d-imagenet1k-h200-baselines"
WORKER_NAME = "lnet-h200-baseline-relay-v1"
RELAY_URL = f"https://{WORKER_NAME}.gpupulse-monitor.workers.dev"
PROTOCOL_VERSION = "wandb-0.22.3-h200-baselines-v1"
CANARY_KEY = "relay_canary"
LNET_K96_MODEL_KEY = "lnet_k96_p128x4_d2262_clean_restart_v3"
LNET_K96_SEEDS = (509, 521)
MIG1_MODELS = {
    "tinyvim_s": "TinyViM-S",
    "efficientvim_m1": "EfficientViM-M1",
    "mambaout_femto": "MambaOut-Femto",
}
MIG1_SEEDS = (501, 509, 521)
MIG1_TINYNEXT_KEY = "tinynext_t_mig1_clean"


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON object required: {path}")
    return payload


def _sha256(value: str | bytes) -> str:
    encoded = value.encode() if isinstance(value, str) else value
    return hashlib.sha256(encoded).hexdigest()


def _run_id(campaign_id: str, model_key: str, seed: int) -> str:
    return _sha256(f"{campaign_id}::{model_key}::seed{seed}")[:16]


def _validate(campaign: dict[str, Any]) -> None:
    models = campaign.get("models")
    seeds = campaign.get("seeds")
    if not isinstance(models, list) or len(models) != 20:
        raise ValueError("baseline campaign must contain exactly 20 models")
    if seeds != [501, 509, 521]:
        raise ValueError("baseline seeds must be [501, 509, 521]")
    keys = [model.get("key") for model in models if isinstance(model, dict)]
    if len(keys) != 20 or len(set(keys)) != 20:
        raise ValueError("baseline model keys must be 20 unique strings")


def _records(campaign: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    campaign_id = str(campaign["campaign_id"])
    runtime_runs: dict[str, Any] = {}
    relay_runs: dict[str, Any] = {}
    for model in campaign["models"]:
        model_key = str(model["key"])
        display_name = str(model["display_name"])
        by_seed: dict[str, Any] = {}
        for seed in campaign["seeds"]:
            run_id = _run_id(campaign_id, model_key, int(seed))
            run = {
                "id": run_id,
                "display_name": f"H200-BL-{model_key}-s{seed}",
                "tags": [
                    "H200",
                    "ImageNet-1K",
                    "matched-baseline",
                    "100ep",
                    "three-seed",
                    "ip-scoped-untrusted",
                ],
            }
            by_seed[str(seed)] = run
            relay_runs[run_id] = {
                "displayName": run["display_name"],
                "tags": run["tags"],
            }
        runtime_runs[model_key] = {
            "display_name": display_name,
            "seeds": by_seed,
        }
    lnet_by_seed: dict[str, Any] = {}
    for seed in LNET_K96_SEEDS:
        run_id = _sha256(f"{campaign_id}:{LNET_K96_MODEL_KEY}:seed{seed}")[:16]
        run = {
            "id": run_id,
            "display_name": f"H200-LNet-I1K-K96-P128x4-s{seed}",
            "tags": [
                "H200",
                "ImageNet-1K",
                "LNet",
                "K96",
                "P128x4",
                "D2262",
                "100ep",
                f"seed{seed}",
            ],
        }
        lnet_by_seed[str(seed)] = run
        relay_runs[run_id] = {
            "displayName": run["display_name"],
            "tags": run["tags"],
        }
    runtime_runs[LNET_K96_MODEL_KEY] = {
        "display_name": "LNet-K96-P128x4-D2262",
        "seeds": lnet_by_seed,
    }
    for model_key, display_name in MIG1_MODELS.items():
        by_seed = {}
        for seed in MIG1_SEEDS:
            run_id = _sha256(f"{campaign_id}:mig1:{model_key}:seed{seed}")[:16]
            run = {
                "id": run_id,
                "display_name": f"H200-MIG1-BL-{model_key}-s{seed}",
                "tags": [
                    "H200",
                    "MIG-1g.18gb",
                    "ImageNet-1K",
                    "matched-baseline",
                    "100ep",
                    f"seed{seed}",
                ],
            }
            by_seed[str(seed)] = run
            relay_runs[run_id] = {
                "displayName": run["display_name"],
                "tags": run["tags"],
            }
        runtime_runs[model_key] = {
            "display_name": display_name,
            "seeds": by_seed,
        }
    run_id = _sha256(f"{campaign_id}:mig1:{MIG1_TINYNEXT_KEY}:seed521")[:16]
    tiny_run = {
        "id": run_id,
        "display_name": "H200-MIG1-BL-tinynext_t-clean-s521",
        "tags": [
            "H200",
            "MIG-1g.18gb",
            "ImageNet-1K",
            "matched-baseline",
            "100ep",
            "seed521",
            "clean-restart",
        ],
    }
    runtime_runs[MIG1_TINYNEXT_KEY] = {
        "display_name": "TinyNeXt-T (MIG1 clean seed521)",
        "seeds": {"521": tiny_run},
    }
    relay_runs[run_id] = {
        "displayName": tiny_run["display_name"],
        "tags": tiny_run["tags"],
    }
    canary_id = _sha256(f"{campaign_id}::{CANARY_KEY}")[:16]
    canary = {
        "id": canary_id,
        "display_name": "H200-baseline-relay-permanent-canary-v1",
        "tags": ["H200", "ImageNet-1K", "relay-canary", "ip-scoped-untrusted"],
    }
    relay_runs[canary_id] = {
        "displayName": canary["display_name"],
        "tags": canary["tags"],
    }
    return runtime_runs, {"canary": canary, "runs_by_id": relay_runs}


def _outputs() -> dict[Path, str]:
    campaign = _load(CAMPAIGN_PATH)
    protocol_source = _load(PROTOCOL_PATH)
    _validate(campaign)
    raw_campaign = CAMPAIGN_PATH.read_bytes()
    manifest_sha256 = _sha256(raw_campaign)
    runtime_runs, relay_records = _records(campaign)
    group = str(campaign["campaign_id"])
    runtime = {
        "schema_version": 1,
        "campaign_id": group,
        "campaign_manifest_sha256": manifest_sha256,
        "wandb_sdk_version": "0.22.3",
        "wandb_base_url": RELAY_URL,
        "wandb_app_url": "https://wandb.ai",
        "entity": ENTITY,
        "project": PROJECT,
        "group": group,
        "console": "off",
        "relay_url": RELAY_URL,
        "relay_protocol_version": PROTOCOL_VERSION,
        "runs": runtime_runs,
        "canary": relay_records["canary"],
    }
    relay = protocol_source["relay"]
    protocol = protocol_source["protocol"]
    relay_contract = {
        "schemaVersion": 1,
        "campaignId": group,
        "manifestSha256": manifest_sha256,
        "protocolVersion": PROTOCOL_VERSION,
        "sdkVersion": "0.22.3",
        "upstreamOrigin": "https://api.wandb.ai",
        "entity": ENTITY,
        "project": PROJECT,
        "group": group,
        "maxGraphqlBodyBytes": relay["max_graphql_body_bytes"],
        "maxFileStreamBodyBytes": relay["max_file_stream_body_bytes"],
        "graphqlOperations": protocol_source["graphql_operations"],
        "graphqlEnvelopeKeys": protocol["graphql_envelope_keys"],
        "noVariableOperations": protocol["no_variable_operations"],
        "runFiles": protocol["run_files"],
        "streamFiles": protocol["stream_files"],
        "upsertBucketVariableKeys": protocol["upsert_bucket_variable_keys"],
        "allowedStates": protocol["allowed_states"],
        "runsById": relay_records["runs_by_id"],
    }
    relay_json = json.dumps(relay_contract, indent=2, sort_keys=True)
    generated_ts = (
        "// Generated by h200/baselines/generate_wandb_contract.py; do not edit.\n"
        f"export const CAMPAIGN = {relay_json} as const;\n"
    )
    wrangler = {
        "$schema": "./node_modules/wrangler/config-schema.json",
        "name": WORKER_NAME,
        "main": "src/index.ts",
        "compatibility_date": "2026-08-20",
        "compatibility_flags": ["nodejs_compat"],
        "workers_dev": True,
        "ratelimits": [
            {
                "name": "RELAY_RATE_LIMITER",
                "namespace_id": "31704",
                "simple": {"limit": 600, "period": 60},
            }
        ],
        "observability": {
            "enabled": True,
            "logs": {"enabled": True, "head_sampling_rate": 1},
            "traces": {"enabled": True, "head_sampling_rate": 1},
        },
    }
    source_index = (ROOT / "cloudflare" / "relay" / "src" / "index.ts").read_text()
    source_index = source_index.replace(
        'program: "h200/run.sh"',
        'program: "h200/run_baselines.sh"',
    )
    source_test = (ROOT / "cloudflare" / "relay" / "test" / "index.test.ts").read_text()
    return {
        RUNTIME_PATH: json.dumps(runtime, indent=2, sort_keys=True) + "\n",
        GENERATED_CAMPAIGN_PATH: generated_ts,
        GENERATED_INDEX_PATH: source_index,
        GENERATED_TEST_PATH: source_test,
        WRANGLER_PATH: json.dumps(wrangler, indent=2, sort_keys=True) + "\n",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    outputs = _outputs()
    if args.check:
        stale = [
            str(path.relative_to(ROOT))
            for path, value in outputs.items()
            if not path.is_file() or path.read_text() != value
        ]
        if stale:
            sys.stderr.write("stale generated baseline W&B files: " + ", ".join(stale) + "\n")
            return 1
        return 0
    for path, value in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
