#!/usr/bin/env python3
"""Run one canonical K64 P-small-factorial model with H200 control and relay identities."""

from __future__ import annotations

# ruff: noqa: BLE001
# pyright: reportExplicitAny=false, reportImplicitRelativeImport=false
# pyright: reportPrivateUsage=false
import json
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import remote_run_control
import run_a2d_affine_qhead_imagenet100 as heads
import run_a2d_r2k3_k64_p_allocation_d2262_imagenet100 as experiment
import run_alphabet2d_imagenet100_nano as harness

if TYPE_CHECKING:
    from torch import Tensor, nn
    from wandb.sdk.wandb_run import Run as WandbRun

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig


_ACTIVE_VARIANT: str | None = None
_BATCH_BOUNDARIES_SEEN = 0
_ORIGINAL_BUILD = experiment._build
_ORIGINAL_AFTER_BATCH = heads._after_training_batch


def _runtime() -> dict[str, Any]:
    value = os.environ.get("H200_K64_P_SMALL_FACTORIAL_WANDB_RUNTIME")
    if not value:
        raise RuntimeError("H200_K64_P_SMALL_FACTORIAL_WANDB_RUNTIME is required")
    payload = json.loads(Path(value).read_text(encoding="utf-8"))
    if payload.get("schema") != "lnet.h200.imagenet100.k64_p_small_factorial.runtime.v1":
        raise RuntimeError("invalid H200 K64 P-small-factorial W&B runtime")
    return cast("dict[str, Any]", payload)


def _metadata(variant: str, seed: int) -> dict[str, Any]:
    payload = _runtime()
    if variant not in payload["training"]["variants"] or seed != payload["training"]["seed"]:
        raise RuntimeError(f"unregistered H200 K64-P-small-factorial run: {variant}/seed{seed}")
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
        raise RuntimeError("H200 K64-P-small-factorial W&B environment changed")
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
                "variant_config": contract["variant_configs"][variant],
                "recipe": contract["recipe"],
                "schema": contract["schema"],
                "h200_campaign": _runtime()["campaign_id"],
            },
        )
    except Exception as error:
        print(f"H200_K64_P_SMALL_FACTORIAL_WANDB_DEGRADED={type(error).__name__}", flush=True)
        return None
    if run is None or not run.url:
        print("H200_K64_P_SMALL_FACTORIAL_WANDB_DEGRADED=missing_run_url", flush=True)
        return None
    print(f"WANDB_RUN_URL={run.url}", flush=True)
    return run


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _argument_path(name: str) -> Path:
    try:
        return Path(sys.argv[sys.argv.index(name) + 1]).resolve()
    except (IndexError, ValueError) as error:
        raise RuntimeError(f"required argument is missing: {name}") from error


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
    if (
        isinstance(generation, bool)
        or not isinstance(generation, int)
        or generation < 1
        or not isinstance(reason, str)
        or not reason
        or not isinstance(payload.get("control_updated_at"), str)
    ):
        raise RuntimeError("H200 owner-stop marker is invalid")
    return {
        "schema": remote_run_control.SCHEMA,
        "campaign_id": expected["campaign_id"],
        "target_commit": expected["target_commit"],
        "generation": payload["generation"],
        "action": "stop",
        "updated_at": payload["control_updated_at"],
        "reason": payload["reason"],
    }


def _raise_if_owner_stopped() -> None:
    record = _read_owner_stop_marker(Path(os.environ["H200_CONTROL_FAST_STOP_MARKER"]))
    if record is not None:
        raise remote_run_control.StopRequestedError(record)


def _write_heartbeat() -> None:
    _atomic_json(
        _argument_path("--root") / "control-heartbeat.json",
        {
            "schema": "lnet.h200.imagenet100.k64_p_small_factorial.heartbeat.v1",
            "campaign_id": _runtime()["campaign_id"],
            "target_commit": os.environ["H200_EXPECTED_COMMIT"],
            "variant": _ACTIVE_VARIANT,
            "batch_boundaries_seen": _BATCH_BOUNDARIES_SEEN,
            "updated_at_unix": int(time.time()),
            "control_generation": int(os.environ["H200_CONTROL_START_GENERATION"]),
        },
    )


def _controlled_build(
    variant: str,
    config: ComplexScanConfig,
) -> ComplexScanBackbone:
    global _ACTIVE_VARIANT  # noqa: PLW0603
    _ACTIVE_VARIANT = variant
    _raise_if_owner_stopped()
    return _ORIGINAL_BUILD(variant, config)


def _controlled_after_training_batch(
    model: nn.Module,
    output: Tensor | tuple[Tensor, ...],
    targets: Tensor,
    permuted_targets: Tensor,
    mixing: float,
) -> None:
    global _BATCH_BOUNDARIES_SEEN  # noqa: PLW0603
    _ORIGINAL_AFTER_BATCH(model, output, targets, permuted_targets, mixing)
    _BATCH_BOUNDARIES_SEEN += 1
    _raise_if_owner_stopped()
    if _BATCH_BOUNDARIES_SEEN % 50 == 0:
        _write_heartbeat()


def _selected_variant() -> str:
    try:
        start = sys.argv.index("--variants") + 1
    except ValueError as error:
        raise RuntimeError("H200 K64-P-small-factorial worker requires --variants") from error
    values: list[str] = []
    for value in sys.argv[start:]:
        if value.startswith("--"):
            break
        values.append(value)
    if len(values) != 1 or values[0] not in experiment.H200_VARIANTS:
        raise RuntimeError("H200 worker requires exactly one registered K64 P-small-factorial variant")
    return values[0]


def _write_stop_marker(error: remote_run_control.StopRequestedError) -> Path:
    marker = Path(os.environ["H200_CONTROL_STOP_MARKER"]).resolve()
    _atomic_json(
        marker,
        {
            "schema": "lnet.h200.owner_stop.v1",
            "campaign_id": _runtime()["campaign_id"],
            "target_commit": os.environ["H200_EXPECTED_COMMIT"],
            "variant": _ACTIVE_VARIANT,
            "batch_boundaries_seen": _BATCH_BOUNDARIES_SEEN,
            "generation": error.record["generation"],
            "reason": error.record["reason"],
            "control_updated_at": error.record["updated_at"],
            "observed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "phase": "model_batch_boundary",
            "forced": False,
            "checkpoint_policy": "last completed model epoch remains authoritative",
            "partial_epoch_discarded": True,
        },
    )
    return marker


def main() -> None:
    _runtime()
    required = (
        "H200_CONTROL_START_GENERATION",
        "H200_CONTROL_FAST_STOP_MARKER",
        "H200_CONTROL_STOP_MARKER",
        "H200_EXPECTED_COMMIT",
    )
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"required H200 run-control environment is missing: {missing}")
    selected = _selected_variant()
    harness._initialize_wandb_run = _initialize_required_wandb_run
    experiment._build = _controlled_build
    heads._after_training_batch = _controlled_after_training_batch
    initial_stop = _read_owner_stop_marker(Path(os.environ["H200_CONTROL_FAST_STOP_MARKER"]))
    if initial_stop is not None:
        marker = _write_stop_marker(remote_run_control.StopRequestedError(initial_stop))
        print(f"H200_KILL_SWITCH_STOPPED={marker}", flush=True)
        return
    print(f"H200_K64_P_SMALL_FACTORIAL_MODEL_START={selected}", flush=True)
    try:
        experiment.main()
    except remote_run_control.StopRequestedError as error:
        marker = _write_stop_marker(error)
        print(f"H200_KILL_SWITCH_STOPPED={marker}", flush=True)


if __name__ == "__main__":
    main()
