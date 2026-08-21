"""Exercise the stage-allocation relay with its permanent W&B canary."""

from __future__ import annotations

# pyright: reportMissingImports=false
# ruff: noqa: T201
import importlib.metadata
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    runtime = json.loads(
        (ROOT / "h200" / "stage_allocation" / "campaign.runtime.json").read_text(encoding="utf-8")
    )
    expected_sdk = runtime["wandb_sdk_version"]
    actual_sdk = importlib.metadata.version("wandb")
    if actual_sdk != expected_sdk:
        message = f"canary requires wandb {expected_sdk}, got {actual_sdk}"
        raise RuntimeError(message)
    canary = runtime["canary"]
    os.environ.update(
        {
            "WANDB_API_KEY": "0" * 40,
            "WANDB_APP_URL": runtime["wandb_app_url"],
            "WANDB_BASE_URL": runtime["wandb_base_url"],
            "WANDB_CONSOLE": "off",
            "WANDB_MODE": "online",
        }
    )

    import wandb  # noqa: PLC0415

    run = wandb.init(
        project=runtime["project"],
        entity=runtime["entity"],
        group=runtime["group"],
        name=canary["display_name"],
        id=canary["id"],
        resume="allow",
        mode="online",
        anonymous="never",
        force=True,
        tags=tuple(canary["tags"]),
        config={
            "campaign_manifest_sha256": runtime["campaign_manifest_sha256"],
            "purpose": "permanent stage-allocation relay canary; never experiment evidence",
            "relay_protocol_version": runtime["relay_protocol_version"],
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
            x_extra_http_headers={"User-Agent": "Mozilla/5.0 lnet-h200-stage-wandb-canary/1"},
            x_save_requirements=False,
        ),
    )
    if run is None or not run.url:
        message = "stage-allocation relay canary did not create a W&B run"
        raise RuntimeError(message)
    internal_log = Path(run.settings.log_internal)
    step = int(time.time())
    run.log({"relay_canary/ok": 1, "relay_canary/unix_time": step}, step=step)
    for _ in range(30):
        if run.status().sync_items_pending == 0:
            break
        time.sleep(1)
    else:
        message = "stage-allocation relay canary did not drain its initial uploads"
        raise RuntimeError(message)
    time.sleep(35)
    run.log({"relay_canary/heartbeat_ok": 1}, step=step + 1)
    run.summary["relay_canary_status"] = "ok"
    run.finish()
    internal_text = internal_log.read_text(encoding="utf-8", errors="replace")
    if "filestream: fatal error" in internal_text:
        message = "stage-allocation relay canary observed a fatal filestream error"
        raise RuntimeError(message)
    print(f"H200_STAGE_RELAY_CANARY_OK={run.url}", flush=True)


if __name__ == "__main__":
    main()
