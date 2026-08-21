#!/usr/bin/env python3
# ruff: noqa: ANN401, BLE001, EM101, EM102, PLC0415, SLF001, T201, TRY003
"""Run the stage-allocation screen with relay-bound H200 W&B identities."""

from __future__ import annotations

# pyright: reportExplicitAny=false, reportImplicitRelativeImport=false
# pyright: reportPrivateUsage=false
import json
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import remote_run_control
import run_a2d_r2k3_stage_allocation_screen_imagenet100 as stage
import run_alphabet2d_imagenet100_nano as harness

if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run as WandbRun


_ACTIVE_VARIANT: str | None = None
_OPTIMIZER_UPDATES_SEEN = 0
_ORIGINAL_TRAIN_EPOCH_WITH_STEP_COUNT = harness._train_epoch_with_step_count


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

    if os.environ.get("WANDB_MODE") == "disabled":
        return None
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
                x_extra_http_headers={"User-Agent": "Mozilla/5.0 lnet-h200-wandb-client/1"},
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


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(descriptor, data)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    temporary.replace(path)


def _write_heartbeat() -> None:
    path = _argument_path("--root") / "control-heartbeat.json"
    _atomic_json(
        path,
        {
            "schema": "lnet.h200.imagenet100.stage_allocation.heartbeat.v1",
            "campaign_id": _runtime()["campaign_id"],
            "target_commit": os.environ["H200_EXPECTED_COMMIT"],
            "variant": _ACTIVE_VARIANT,
            "optimizer_updates_seen": _OPTIMIZER_UPDATES_SEEN,
            "updated_at_unix": int(time.time()),
            "control_generation": int(os.environ["H200_CONTROL_START_GENERATION"]),
        },
    )


def _read_owner_stop_marker(marker: Path) -> dict[str, Any] | None:
    if not marker.is_file():
        return None
    payload = json.loads(marker.read_text(encoding="utf-8"))
    expected = {
        "schema": "lnet.h200.owner_stop.v1",
        "campaign_id": _runtime()["campaign_id"],
        "target_commit": os.environ["H200_EXPECTED_COMMIT"],
    }
    if any(payload.get(name) != value for name, value in expected.items()):
        raise RuntimeError("H200 owner-stop marker identity changed")
    generation = payload.get("generation")
    reason = payload.get("reason")
    updated_at = payload.get("control_updated_at")
    if (
        isinstance(generation, bool)
        or not isinstance(generation, int)
        or generation < 1
        or not isinstance(reason, str)
        or not reason
        or not isinstance(updated_at, str)
    ):
        raise RuntimeError("H200 owner-stop marker is invalid")
    return {
        "schema": remote_run_control.SCHEMA,
        "campaign_id": expected["campaign_id"],
        "target_commit": expected["target_commit"],
        "generation": generation,
        "action": "stop",
        "updated_at": updated_at,
        "reason": reason,
    }


def _raise_if_owner_stopped() -> None:
    marker = Path(os.environ["H200_CONTROL_FAST_STOP_MARKER"])
    record = _read_owner_stop_marker(marker)
    if record is not None:
        raise remote_run_control.StopRequestedError(record)


def _controlled_train_epoch_with_step_count(*args: Any, **kwargs: Any) -> Any:
    """Check the supervisor's atomic stop marker after a successful update."""
    optimizer = args[4]
    original_step = optimizer.step

    def step_then_check(*step_args: Any, **step_kwargs: Any) -> Any:
        global _OPTIMIZER_UPDATES_SEEN  # noqa: PLW0603
        result = original_step(*step_args, **step_kwargs)
        _OPTIMIZER_UPDATES_SEEN += 1
        _raise_if_owner_stopped()
        if _OPTIMIZER_UPDATES_SEEN % 100 == 0:
            _write_heartbeat()
        return result

    optimizer.step = step_then_check
    try:
        return _ORIGINAL_TRAIN_EPOCH_WITH_STEP_COUNT(*args, **kwargs)
    finally:
        optimizer.step = original_step


def _argument_path(name: str) -> Path:
    try:
        return Path(sys.argv[sys.argv.index(name) + 1]).resolve()
    except (IndexError, ValueError) as error:
        raise RuntimeError(f"required argument is missing: {name}") from error


def _write_stop_marker(error: remote_run_control.StopRequestedError) -> Path:
    configured = os.environ.get("H200_CONTROL_STOP_MARKER")
    marker = (
        Path(configured).resolve() if configured else _argument_path("--root") / "control-stop.json"
    )
    payload = {
        "schema": "lnet.h200.owner_stop.v1",
        "campaign_id": _runtime()["campaign_id"],
        "target_commit": os.environ["H200_EXPECTED_COMMIT"],
        "variant": _ACTIVE_VARIANT,
        "optimizer_updates_seen": _OPTIMIZER_UPDATES_SEEN,
        "generation": error.record["generation"],
        "reason": error.record["reason"],
        "control_updated_at": error.record["updated_at"],
        "observed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "phase": "optimizer_step_boundary",
        "forced": False,
        "checkpoint_policy": "last completed epoch remains authoritative",
        "partial_epoch_discarded": True,
    }
    _atomic_json(marker, payload)
    return marker


def _selected_variant() -> str:
    try:
        start = sys.argv.index("--variants") + 1
    except ValueError as error:
        raise RuntimeError("H200 controlled runs require exactly one --variants value") from error
    values: list[str] = []
    for value in sys.argv[start:]:
        if value.startswith("--"):
            break
        values.append(value)
    if len(values) != 1:
        raise RuntimeError("H200 controlled runs require exactly one --variants value")
    return values[0]


def main() -> None:
    global _ACTIVE_VARIANT  # noqa: PLW0603
    _runtime()
    required_environment = (
        "H200_CONTROL_START_GENERATION",
        "H200_CONTROL_FAST_STOP_MARKER",
        "H200_CONTROL_STOP_MARKER",
        "H200_EXPECTED_COMMIT",
    )
    missing = [name for name in required_environment if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"required H200 run-control environment is missing: {missing}")
    _ACTIVE_VARIANT = _selected_variant()
    harness._initialize_wandb_run = _initialize_required_wandb_run
    harness._train_epoch_with_step_count = _controlled_train_epoch_with_step_count
    initial_stop = _read_owner_stop_marker(Path(os.environ["H200_CONTROL_FAST_STOP_MARKER"]))
    if initial_stop is not None:
        error = remote_run_control.StopRequestedError(initial_stop)
        marker = _write_stop_marker(error)
        print(f"H200_KILL_SWITCH_STOPPED={marker}", flush=True)
        return
    try:
        stage.main()
    except remote_run_control.StopRequestedError as error:
        marker = _write_stop_marker(error)
        print(f"H200_KILL_SWITCH_STOPPED={marker}", flush=True)


if __name__ == "__main__":
    main()
