#!/usr/bin/env python3
"""Run one XL family cell with H200 control and frozen W&B identity."""

from __future__ import annotations

# pyright: reportExplicitAny=false, reportImplicitRelativeImport=false
# pyright: reportMissingImports=false, reportPrivateUsage=false
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import remote_run_control
import run_a2d_affine_qhead_imagenet100 as heads
import run_a2d_r2k3_k_family_wave_a_h200_imagenet100 as experiment
import run_alphabet2d_imagenet100_nano as harness
import torch

if TYPE_CHECKING:
    from torch import Tensor, nn

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig


RUNTIME_SCHEMA = "lnet.h200.imagenet100.k_family_xl.runtime.v1"
RUNTIME_ENV_VAR = "H200_K_FAMILY_XL_WANDB_RUNTIME"
HEARTBEAT_SCHEMA = "lnet.h200.imagenet100.k_family_xl.heartbeat.v1"
VARIANTS = experiment.VARIANTS
PARAMETER_COUNTS = {
    "XL-K96-U1": 2_313_892,
    "XL-K96-U125": 2_791_524,
    "XL-K96-Shaped": 2_996_740,
    "XL-K96-Rich": 3_200_068,
}
_active_variant: str | None = None
_batch_boundaries_seen = 0
_initial_resume_state: dict[str, dict[str, Any]] = {}
_ORIGINAL_BUILD = experiment._build
_ORIGINAL_AFTER_BATCH = heads._after_training_batch


def _runtime() -> dict[str, Any]:
    value = os.environ.get(RUNTIME_ENV_VAR)
    if not value:
        raise RuntimeError(f"{RUNTIME_ENV_VAR} is required")
    payload = json.loads(Path(value).read_text(encoding="utf-8"))
    training = payload.get("training", {})
    if (
        payload.get("schema") != RUNTIME_SCHEMA
        or tuple(training.get("variants", ())) != VARIANTS
        or training.get("seed") != 501
        or training.get("epochs") != 100
        or training.get("batch_size") != 128
        or training.get("precision") != "bfloat16"
        or payload.get("parameter_counts") != PARAMETER_COUNTS
        or payload.get("dataset", {}).get("train_images") != 130000
    ):
        raise RuntimeError("invalid H200 XL family runtime")
    return cast("dict[str, Any]", payload)


def _metadata(variant: str, seed: int) -> dict[str, Any]:
    payload = _runtime()
    if variant not in VARIANTS or seed != 501:
        raise RuntimeError(f"unregistered H200 XL run: {variant}/seed{seed}")
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
    if any(os.environ.get(name) != expected_value for name, expected_value in expected.items()):
        raise RuntimeError("H200 XL W&B environment changed")
    return cast("dict[str, Any]", record)


def _resume_state(root: Path, variant: str, seed: int) -> dict[str, Any]:
    checkpoint = root / "checkpoints" / f"{variant}__seed{seed}.pt"
    if not checkpoint.is_file():
        return {"checkpoint_found": False, "resumed_from_epoch": 0}
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("variant") != variant or payload.get("seed") != seed:
        raise RuntimeError("XL checkpoint identity changed before W&B initialization")
    epoch = payload.get("epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or not 0 <= epoch <= 100:
        raise RuntimeError("XL checkpoint epoch is invalid")
    return {"checkpoint_found": True, "resumed_from_epoch": epoch}


class _WandbTelemetryProxy:
    def __init__(self, run: Any, *, training_examples: int, resume_state: dict[str, Any]) -> None:
        self._run = run
        self._training_examples = training_examples
        self._previous_training_seconds = 0.0
        self._latest_images_per_second = 0.0
        self.summary = run.summary
        self.url = run.url
        self.summary["checkpoint_found_at_start"] = resume_state["checkpoint_found"]
        self.summary["resumed_from_epoch"] = resume_state["resumed_from_epoch"]

    def log(self, metrics: dict[str, Any], *, step: int) -> None:
        enriched = dict(metrics)
        total = float(enriched["time/training_seconds"])
        elapsed = total - self._previous_training_seconds
        if elapsed > 0.0:
            self._latest_images_per_second = self._training_examples / elapsed
            enriched["throughput/images_per_second"] = self._latest_images_per_second
        self._previous_training_seconds = total
        enriched["memory/peak_vram_bytes"] = float(torch.cuda.max_memory_allocated())
        self._run.log(enriched, step=step)

    def finish(self) -> None:
        self.summary["peak_vram_bytes"] = float(torch.cuda.max_memory_allocated())
        self.summary["latest_images_per_second"] = self._latest_images_per_second
        self._run.finish()


def _initialize_required_wandb_run(
    root: Path,
    contract: dict[str, Any],
    *,
    variant: str,
    seed: int,
    parameters: int,
) -> _WandbTelemetryProxy:
    import wandb

    if os.environ.get("WANDB_MODE") == "disabled":
        raise RuntimeError("H200 XL requires online W&B telemetry")
    runtime = _runtime()
    metadata = _metadata(variant, seed)
    if parameters != PARAMETER_COUNTS[variant]:
        raise RuntimeError(f"{variant} parameter count changed: {parameters}")
    resume_state = _initial_resume_state.get(variant)
    if resume_state is None:
        resume_state = _resume_state(root, variant, seed)
    tracking_root = root / "wandb"
    tracking_root.mkdir(parents=True, exist_ok=True)
    contract_sha256 = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
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
            "source_commit": os.environ["H200_EXPECTED_COMMIT"],
            "campaign_id": runtime["campaign_id"],
            "campaign_manifest_sha256": runtime["campaign_manifest_sha256"],
            "runner_contract_sha256": contract_sha256,
            "checkpoint_resume": resume_state,
        },
    )
    if run is None or not run.url:
        raise RuntimeError("H200 XL W&B initialization returned no online run URL")
    print(f"WANDB_RUN_URL={run.url}", flush=True)
    return _WandbTelemetryProxy(
        run,
        training_examples=int(runtime["dataset"]["train_images"]),
        resume_state=resume_state,
    )


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
        raise RuntimeError("H200 XL owner-stop marker identity changed")
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
        raise RuntimeError("H200 XL owner-stop marker is invalid")
    return {
        "schema": remote_run_control.SCHEMA,
        "campaign_id": expected["campaign_id"],
        "target_commit": expected["target_commit"],
        "generation": generation,
        "action": "stop",
        "updated_at": payload["control_updated_at"],
        "reason": reason,
    }


def _raise_if_owner_stopped() -> None:
    record = _read_owner_stop_marker(Path(os.environ["H200_CONTROL_FAST_STOP_MARKER"]))
    if record is not None:
        raise remote_run_control.StopRequestedError(record)


def _write_heartbeat() -> None:
    _atomic_json(
        _argument_path("--root") / "control-heartbeat.json",
        {
            "schema": HEARTBEAT_SCHEMA,
            "campaign_id": _runtime()["campaign_id"],
            "target_commit": os.environ["H200_EXPECTED_COMMIT"],
            "variant": _active_variant,
            "batch_boundaries_seen": _batch_boundaries_seen,
            "updated_at_unix": int(time.time()),
            "control_generation": int(os.environ["H200_CONTROL_START_GENERATION"]),
        },
    )


def _controlled_build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    global _active_variant  # noqa: PLW0603
    _active_variant = variant
    _initial_resume_state.setdefault(variant, _resume_state(_argument_path("--root"), variant, 501))
    _raise_if_owner_stopped()
    return _ORIGINAL_BUILD(variant, config)


def _controlled_after_training_batch(
    model: nn.Module,
    output: Tensor | tuple[Tensor, ...],
    targets: Tensor,
    permuted_targets: Tensor,
    mixing: float,
) -> None:
    global _batch_boundaries_seen  # noqa: PLW0603
    _ORIGINAL_AFTER_BATCH(model, output, targets, permuted_targets, mixing)
    _batch_boundaries_seen += 1
    _raise_if_owner_stopped()
    if _batch_boundaries_seen % 50 == 0:
        _write_heartbeat()


def _selected_variant() -> str:
    try:
        start = sys.argv.index("--variants") + 1
    except ValueError as error:
        raise RuntimeError("H200 XL worker requires --variants") from error
    values: list[str] = []
    for value in sys.argv[start:]:
        if value.startswith("--"):
            break
        values.append(value)
    if len(values) != 1 or values[0] not in VARIANTS:
        raise RuntimeError("H200 XL worker requires exactly one registered variant")
    return values[0]


def _write_stop_marker(error: remote_run_control.StopRequestedError) -> Path:
    marker = Path(os.environ["H200_CONTROL_STOP_MARKER"]).resolve()
    _atomic_json(
        marker,
        {
            "schema": "lnet.h200.owner_stop.v1",
            "campaign_id": _runtime()["campaign_id"],
            "target_commit": os.environ["H200_EXPECTED_COMMIT"],
            "variant": _active_variant,
            "batch_boundaries_seen": _batch_boundaries_seen,
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
        raise RuntimeError(f"required H200 XL run-control environment is missing: {missing}")
    selected = _selected_variant()
    harness._initialize_wandb_run = _initialize_required_wandb_run
    experiment._build = _controlled_build
    heads._after_training_batch = _controlled_after_training_batch
    initial_stop = _read_owner_stop_marker(Path(os.environ["H200_CONTROL_FAST_STOP_MARKER"]))
    if initial_stop is not None:
        marker = _write_stop_marker(remote_run_control.StopRequestedError(initial_stop))
        print(f"H200_KILL_SWITCH_STOPPED={marker}", flush=True)
        return
    print(f"H200_K_FAMILY_XL_MODEL_START={selected}", flush=True)
    try:
        experiment.main()
    except remote_run_control.StopRequestedError as error:
        marker = _write_stop_marker(error)
        print(f"H200_KILL_SWITCH_STOPPED={marker}", flush=True)


if __name__ == "__main__":
    main()
