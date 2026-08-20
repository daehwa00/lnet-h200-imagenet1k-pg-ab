#!/usr/bin/env python3
# ruff: noqa: BLE001, EM101, EM102, PLC0415, SLF001, T201, TRY003
"""Run the stage-allocation screen with relay-bound H200 W&B identities."""

from __future__ import annotations

# pyright: reportExplicitAny=false, reportImplicitRelativeImport=false
# pyright: reportPrivateUsage=false
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import run_a2d_r2k3_stage_allocation_screen_imagenet100 as stage
import run_alphabet2d_imagenet100_nano as harness

if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run as WandbRun


def _runtime() -> dict[str, Any]:
    value = os.environ.get("H200_STAGE_ALLOCATION_WANDB_RUNTIME")
    if not value:
        raise RuntimeError("H200_STAGE_ALLOCATION_WANDB_RUNTIME is required")
    payload = json.loads(Path(value).read_text(encoding="utf-8"))
    if payload.get("schema") != "lnet.h200.imagenet100.stage_allocation.runtime.v1":
        raise RuntimeError("invalid H200 stage-allocation W&B runtime")
    return cast("dict[str, Any]", payload)


def _metadata(variant: str, seed: int) -> dict[str, Any]:
    payload = _runtime()
    if variant not in payload["training"]["variants"] or seed != payload["training"]["seed"]:
        raise RuntimeError(f"unregistered H200 stage-allocation run: {variant}/seed{seed}")
    record = payload["runs"][variant][str(seed)]
    expected = {
        "WANDB_API_KEY": "0" * 40,
        "WANDB_APP_URL": payload["wandb_app_url"],
        "WANDB_BASE_URL": payload["wandb_base_url"],
        "WANDB_ENTITY": payload["entity"],
        "WANDB_PROJECT": payload["project"],
        "WANDB_GROUP": payload["group"],
        "WANDB_CONSOLE": payload["console"],
    }
    if any(os.environ.get(name) != value for name, value in expected.items()):
        raise RuntimeError("H200 stage-allocation W&B environment changed")
    return cast("dict[str, Any]", record)


def _initialize_required_wandb_run(
    root: Path,
    contract: dict[str, Any],
    *,
    variant: str,
    seed: int,
    parameters: int,
) -> WandbRun | None:
    import wandb

    metadata = _metadata(variant, seed)
    tracking_root = root / "wandb"
    tracking_root.mkdir(parents=True, exist_ok=True)
    variant_config = contract.get("variant_configs", {}).get(variant)
    try:
        run = wandb.init(
            project=os.environ["WANDB_PROJECT"],
            entity=os.environ["WANDB_ENTITY"],
            group=os.environ["WANDB_GROUP"],
            name=metadata["display_name"],
            id=metadata["id"],
            tags=metadata["tags"],
            resume="allow",
            dir=str(tracking_root),
            mode="online",
            anonymous="never",
            force=True,
            settings=wandb.Settings(
                disable_code=True,
                console="off",
                disable_git=True,
                disable_job_creation=True,
                init_timeout=float(os.environ.get("WANDB_INIT_TIMEOUT", "30")),
                save_code=False,
                x_disable_meta=True,
                x_disable_stats=True,
                x_disable_viewer=True,
                x_extra_http_headers={
                    "User-Agent": "Mozilla/5.0 lnet-h200-wandb-client/1"
                },
                x_save_requirements=False,
            ),
            config={
                "variant": variant,
                "seed": seed,
                "parameters": parameters,
                "model": variant_config or contract["model"],
                "model_template": contract["model"],
                "variant_config": variant_config,
                "recipe": contract["recipe"],
                "schema": contract["schema"],
                "h200_campaign": _runtime()["campaign_id"],
            },
        )
    except Exception as error:  # W&B is a non-authoritative mirror.
        print(f"H200_STAGE_WANDB_DEGRADED={type(error).__name__}", flush=True)
        return None
    if run is None or not run.url:
        print("H200_STAGE_WANDB_DEGRADED=missing_run_url", flush=True)
        return None
    print(f"WANDB_RUN_URL={run.url}", flush=True)
    return run


def main() -> None:
    _runtime()
    harness._initialize_wandb_run = _initialize_required_wandb_run
    stage.main()


if __name__ == "__main__":
    main()
