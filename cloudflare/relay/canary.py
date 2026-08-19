"""Exercise the deployed relay with the permanent non-production W&B run."""

from __future__ import annotations

# ruff: noqa: T201
import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    campaign_path = ROOT / "h200" / "campaign.json"
    campaign_bytes = campaign_path.read_bytes()
    campaign = json.loads(campaign_bytes)
    wandb_config = campaign["wandb"]
    relay = campaign["relay"]
    canary = relay["permanent_canary"]
    expected_sdk = wandb_config["sdk_version"]
    actual_sdk = importlib.metadata.version("wandb")
    if actual_sdk != expected_sdk:
        message = f"canary requires wandb {expected_sdk}, got {actual_sdk}"
        raise RuntimeError(message)

    os.environ["WANDB_API_KEY"] = "0" * 40
    os.environ["WANDB_APP_URL"] = wandb_config["app_url"]
    os.environ["WANDB_BASE_URL"] = relay["url"]
    os.environ["WANDB_CONSOLE"] = "off"
    os.environ["WANDB_MODE"] = "online"

    import wandb  # noqa: PLC0415

    run = wandb.init(
        project=wandb_config["project"],
        entity=wandb_config["entity"],
        group=wandb_config["group"],
        name=canary["display_name"],
        id=canary["id"],
        resume="allow",
        mode="online",
        anonymous="never",
        force=True,
        tags=tuple(canary["tags"]),
        config={
            "campaign_manifest_sha256": hashlib.sha256(campaign_bytes).hexdigest(),
            "purpose": "permanent relay protocol canary; never production evidence",
            "relay_protocol_version": relay["protocol_version"],
        },
        settings=wandb.Settings(
            console="off",
            disable_code=True,
            disable_git=True,
            disable_job_creation=True,
            init_timeout=30,
            save_code=False,
            x_disable_meta=True,
            x_disable_stats=True,
            x_disable_viewer=True,
            x_extra_http_headers={
                "User-Agent": "Mozilla/5.0 lnet-h200-wandb-canary/3"
            },
            x_save_requirements=False,
        ),
    )
    if run is None or not run.url:
        message = "relay canary did not create a W&B run"
        raise RuntimeError(message)
    step = int(time.time())
    run.log({"relay_canary/ok": 1, "relay_canary/unix_time": step}, step=step)
    run.summary["relay_canary_status"] = "ok"
    run.finish()
    print(f"H200_RELAY_CANARY_OK={run.url}", flush=True)


if __name__ == "__main__":
    main()
