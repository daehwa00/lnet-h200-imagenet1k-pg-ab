"""Train the stage-allocation screen in one shared-batch lockstep cohort."""

from __future__ import annotations

# ruff: noqa: BLE001, C901, EM101, EM102, PLR0911, PLR0912, PLR0915, SLF001, T201, TRY003
# pyright: reportExplicitAny=false, reportImplicitRelativeImport=false
# pyright: reportPrivateUsage=false
import argparse
import hashlib
import json
import os
import platform
import random
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import a2d_r2k3_runtime as runtime
import numpy as np
import run_a2d_affine_qhead_imagenet100 as heads
import run_a2d_deep4_calibrated_uniform_p96_phase_gated_imagenet100 as model_metrics
import run_a2d_qhead_e2e_imagenet100 as structured
import run_a2d_r2k3_stage_allocation_screen_imagenet100 as stage
import run_a2d_resaux1_imagenet100 as prepare
import run_alphabet2d_imagenet100_nano as harness
import shared_batch_cohort as shared
import torch
from torch import nn

SCHEMA = "lnet.a2d.r2k3.shared_batch_cohort.v1"
PREFLIGHT_SCHEMA = "lnet.h200.shared_batch_preflight.v1"
SEED = 501


@dataclass(slots=True)
class MemberState:
    variant: str
    model: nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    runtime: nn.Module | None = None
    parameters: int = 0
    history: list[dict[str, float]] = field(default_factory=list)
    training_seconds: float = 0.0
    global_step: int = 0
    last_validation: dict[str, float] | None = None

    def active(self) -> shared.CohortMember:
        if self.runtime is None:
            raise RuntimeError(f"compiled runtime is missing for {self.variant}")
        return shared.CohortMember(
            variant=self.variant,
            model=self.model,
            runtime=self.runtime,
            optimizer=self.optimizer,
        )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--variants", choices=stage.VARIANTS, nargs="+", default=list(stage.VARIANTS)
    )
    parser.add_argument("--run-seeds", type=int, nargs="+", default=[SEED])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--precision", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--minimum-cohort-model-images-per-second", type=float, default=800.0)
    parser.add_argument("--initialize-only", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cohort_token(contract_sha256: str, epoch: int, variants: tuple[str, ...]) -> str:
    payload = json.dumps(
        {
            "contract_sha256": contract_sha256,
            "epoch": epoch,
            "schema": SCHEMA,
            "seed": SEED,
            "variants": variants,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _contract(args: argparse.Namespace) -> dict[str, Any]:
    payload = stage._contract(args)
    payload.get("runtime", {}).pop("hostname", None)
    payload.get("recipe", {}).pop("cpu_affinity", None)
    payload["execution"] = {
        "schema": SCHEMA,
        "mode": "single_process_shared_batch_lockstep",
        "ordered_variants": list(stage.VARIANTS),
        "shared": [
            "ImageFolder",
            "DataLoader",
            "augmented_cpu_batch",
            "H2D_tensor",
            "mixup_permutation",
            "mixup_lambda",
            "validation_batch",
        ],
        "per_variant": [
            "model",
            "optimizer",
            "scheduler",
            "compiled_runtime",
            "history",
            "checkpoint",
            "wandb_run",
        ],
        "compile_mode": os.environ.get("LNET_COMPILE_MODE", "default"),
        "loader_workers": harness._active_loader_workers(args.workers),
        "loader_prefetch_factor": harness._active_loader_prefetch_factor(),
        "torch_dynamo_recompile_limit": 64,
        "checkpoint_commit": "all member files then atomic cohort manifest",
        "preflight": {
            "epoch": 2,
            "minimum_cohort_model_images_per_second": (args.minimum_cohort_model_images_per_second),
            "maximum_peak_reserved_ratio": 0.85,
            "minimum_free_memory_bytes": 24 * 2**30,
            "maximum_allocated_drift_floor_bytes": 512 * 2**20,
            "maximum_allocated_drift_ratio": 0.005,
        },
    }
    payload["source_sha256"]["shared_batch_cohort"] = harness._digest(Path(__file__))
    payload["source_sha256"]["shared_batch_helper"] = harness._digest(
        Path("scripts/shared_batch_cohort.py")
    )
    payload["source_sha256"]["shared_batch_evaluation"] = harness._digest(
        Path("scripts/run_a2d_affine_qhead_imagenet100.py")
    )
    return json.loads(json.dumps(payload))


def _write_runtime_evidence(root: Path, args: argparse.Namespace) -> None:
    harness._atomic_json(
        root / "runtime-evidence.json",
        {
            "schema": "lnet.h200.cohort_runtime_evidence.v1",
            "hostname": platform.node(),
            "cpu_affinity": os.environ.get("LNET_CPU_AFFINITY_ACTIVE"),
            "loader_workers": harness._active_loader_workers(args.workers),
            "loader_prefetch_factor": harness._active_loader_prefetch_factor(),
            "observed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )


def _build_members(contract: dict[str, Any], device: torch.device) -> list[MemberState]:
    recipe = contract["recipe"]
    members: list[MemberState] = []
    for variant in stage.VARIANTS:
        random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        model = stage._build(variant, runtime.model_config()).to(device)
        model = prepare._prepare_model(model, recipe)
        optimizer = runtime.build_optimizer(model, recipe)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda epoch: harness._learning_rate_factor(epoch, recipe["epochs"]),
        )
        members.append(
            MemberState(
                variant=variant,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                parameters=sum(parameter.numel() for parameter in model.parameters()),
            )
        )
    return members


def _checkpoint_root(root: Path) -> Path:
    return root / "checkpoints" / "shared-cohort"


def _manifest_is_valid(
    root: Path,
    manifest: object,
    *,
    contract_sha256: str,
) -> bool:
    if not isinstance(manifest, dict):
        return False
    expected_keys = {
        "cohort_token",
        "contract_sha256",
        "end_allocated_bytes",
        "epoch",
        "files",
        "schema",
        "seed",
        "shared",
        "preflight",
        "variants",
    }
    if set(manifest) != expected_keys:
        return False
    if (
        manifest["schema"] != SCHEMA
        or manifest["contract_sha256"] != contract_sha256
        or manifest["seed"] != SEED
        or manifest["variants"] != list(stage.VARIANTS)
        or not isinstance(manifest["epoch"], int)
        or manifest["epoch"] < 1
        or not isinstance(manifest["files"], dict)
        or set(manifest["files"]) != set(stage.VARIANTS)
        or not isinstance(manifest["shared"], dict)
        or not isinstance(manifest["end_allocated_bytes"], int)
        or manifest["end_allocated_bytes"] < 0
    ):
        return False
    preflight = manifest["preflight"]
    if preflight is not None:
        if not isinstance(preflight, dict):
            return False
        if (
            preflight.get("schema") != PREFLIGHT_SCHEMA
            or preflight.get("contract_sha256") != contract_sha256
            or preflight.get("cohort_token") != manifest["cohort_token"]
            or preflight.get("compile_mode") != "default"
            or preflight.get("variants") != list(stage.VARIANTS)
            or preflight.get("status") != "passed"
        ):
            return False
    records = [*manifest["files"].values(), manifest["shared"]]
    for record in records:
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            return False
        path = root / str(record["path"])
        if not path.is_file() or _sha256(path) != record["sha256"]:
            return False
    expected_token = _cohort_token(
        contract_sha256,
        int(manifest["epoch"]),
        stage.VARIANTS,
    )
    return manifest["cohort_token"] == expected_token


def _latest_manifest(root: Path, contract_sha256: str) -> dict[str, Any] | None:
    checkpoint_root = _checkpoint_root(root)
    candidates = sorted(
        (checkpoint_root / "manifests").glob("epoch-*.json"),
        reverse=True,
    )
    candidates.append(checkpoint_root / "latest.json")
    for path in candidates:
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            continue
        if _manifest_is_valid(root, manifest, contract_sha256=contract_sha256):
            return cast("dict[str, Any]", manifest)
    return None


def _restore(
    root: Path,
    members: list[MemberState],
    *,
    contract_sha256: str,
    training_generator: torch.Generator,
    mixup_generator: np.random.Generator,
    permutation_generator: torch.Generator,
) -> tuple[int, dict[str, Any] | None, int | None]:
    manifest = _latest_manifest(root, contract_sha256)
    if manifest is None:
        return 0, None, None
    epoch = int(manifest["epoch"])
    token = str(manifest["cohort_token"])
    by_variant = {member.variant: member for member in members}
    for variant, record in manifest["files"].items():
        payload = torch.load(root / record["path"], map_location="cpu", weights_only=True)
        if (
            payload.get("variant") != variant
            or payload.get("seed") != SEED
            or payload.get("contract_sha256") != contract_sha256
            or payload.get("cohort_token") != token
            or payload.get("epoch") != epoch
        ):
            raise RuntimeError(f"cohort checkpoint identity changed for {variant}")
        member = by_variant[variant]
        member.model.load_state_dict(payload["model"])
        member.optimizer.load_state_dict(payload["optimizer"])
        member.scheduler.load_state_dict(payload["scheduler"])
        member.history = list(payload["history"])
        member.training_seconds = float(payload["training_seconds"])
        member.global_step = int(payload["global_step"])
        harness._restore_optimizer_runtime_options(member.optimizer, payload["recipe"])
    shared_payload = torch.load(
        root / manifest["shared"]["path"],
        map_location="cpu",
        weights_only=True,
    )
    if (
        shared_payload.get("cohort_token") != token
        or shared_payload.get("epoch") != epoch
        or shared_payload.get("variants") != list(stage.VARIANTS)
    ):
        raise RuntimeError("shared cohort RNG checkpoint identity changed")
    training_generator.set_state(shared_payload["training_generator_state"])
    torch.set_rng_state(shared_payload["torch_rng_state"])
    torch.cuda.set_rng_state_all(shared_payload["cuda_rng_state"])
    random.setstate(shared_payload["python_rng_state"])
    mixup_generator.bit_generator.state = shared_payload["mixup_rng_state"]
    permutation_generator.set_state(shared_payload["permutation_generator_state"])
    return (
        epoch,
        cast("dict[str, Any] | None", manifest["preflight"]),
        int(manifest["end_allocated_bytes"]),
    )


def _save_checkpoint(
    root: Path,
    members: list[MemberState],
    *,
    epoch: int,
    contract_sha256: str,
    recipe: dict[str, Any],
    training_generator: torch.Generator,
    mixup_generator: np.random.Generator,
    permutation_generator: torch.Generator,
    preflight: dict[str, Any] | None,
    end_allocated_bytes: int = 0,
) -> dict[str, Any]:
    checkpoint_root = _checkpoint_root(root)
    epoch_root = checkpoint_root / "epochs" / f"epoch-{epoch:04d}"
    token = _cohort_token(contract_sha256, epoch, stage.VARIANTS)
    if preflight is not None:
        preflight = {**preflight, "cohort_token": token}
    files: dict[str, dict[str, str]] = {}
    for member in members:
        path = epoch_root / f"{member.variant}__seed{SEED}.pt"
        harness._atomic_torch(
            path,
            {
                "schema": SCHEMA,
                "variant": member.variant,
                "seed": SEED,
                "contract_sha256": contract_sha256,
                "cohort_token": token,
                "epoch": epoch,
                "global_step": member.global_step,
                "model": member.model.state_dict(),
                "optimizer": member.optimizer.state_dict(),
                "scheduler": member.scheduler.state_dict(),
                "history": member.history,
                "training_seconds": member.training_seconds,
                "recipe": recipe,
            },
        )
        files[member.variant] = {
            "path": str(path.relative_to(root)),
            "sha256": _sha256(path),
        }
    shared_path = epoch_root / "shared-rng.pt"
    harness._atomic_torch(
        shared_path,
        {
            "schema": SCHEMA,
            "cohort_token": token,
            "epoch": epoch,
            "variants": list(stage.VARIANTS),
            "training_generator_state": training_generator.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all(),
            "python_rng_state": random.getstate(),
            "mixup_rng_state": mixup_generator.bit_generator.state,
            "permutation_generator_state": permutation_generator.get_state(),
        },
    )
    manifest = {
        "schema": SCHEMA,
        "contract_sha256": contract_sha256,
        "cohort_token": token,
        "epoch": epoch,
        "end_allocated_bytes": end_allocated_bytes,
        "seed": SEED,
        "variants": list(stage.VARIANTS),
        "files": files,
        "shared": {
            "path": str(shared_path.relative_to(root)),
            "sha256": _sha256(shared_path),
        },
        "preflight": preflight,
    }
    manifest_path = checkpoint_root / "manifests" / f"epoch-{epoch:04d}.json"
    harness._atomic_json(manifest_path, manifest)
    harness._atomic_json(checkpoint_root / "latest.json", manifest)
    retained = sorted((checkpoint_root / "manifests").glob("epoch-*.json"), reverse=True)
    for stale_manifest in retained[2:]:
        stale_epoch = stale_manifest.stem.removeprefix("epoch-")
        if len(stale_epoch) != 4 or not stale_epoch.isdigit():
            continue
        stale_root = checkpoint_root / "epochs" / f"epoch-{stale_epoch}"
        shutil.rmtree(stale_root, ignore_errors=True)
        stale_manifest.unlink(missing_ok=True)
    return manifest


def _spool_path(root: Path, variant: str) -> Path:
    return harness._telemetry_spool_path(root, variant=variant, seed=SEED)


def _receipt_path(root: Path, variant: str) -> Path:
    return root / "telemetry-receipts" / f"{variant}__seed{SEED}.json"


def _delivered_receipt(
    root: Path,
    variant: str,
    contract_sha256: str,
) -> set[tuple[str, int | None]]:
    try:
        payload = json.loads(_receipt_path(root, variant).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return set()
    if (
        payload.get("schema") != "lnet.wandb.telemetry_receipt.v1"
        or payload.get("contract_sha256") != contract_sha256
        or payload.get("variant") != variant
        or not isinstance(payload.get("final"), bool)
    ):
        return set()
    epoch = payload.get("epoch", 0)
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        return set()
    delivered = {("epoch", step) for step in range(1, epoch + 1)}
    if payload.get("final") is True:
        delivered.add(("final", None))
    return delivered


def _publish_variant_spool(
    root: Path,
    contract: dict[str, Any],
    contract_sha256: str,
    *,
    variant: str,
    parameters: int,
) -> bool:
    spool = _spool_path(root, variant)
    delivered = _delivered_receipt(root, variant, contract_sha256)
    run = harness._best_effort_initialize_wandb(
        root,
        contract,
        variant=variant,
        seed=SEED,
        parameters=parameters,
    )
    if run is None:
        return False
    internal_log = Path(run.settings.log_internal)
    run = harness._sync_or_abandon_wandb(run, spool, delivered)
    if run is None:
        return False
    try:
        run.finish()
    except Exception as error:
        harness._report_wandb_degraded("cohort-finish", error)
        return False
    try:
        internal_text = internal_log.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        harness._report_wandb_degraded("cohort-internal-log", error)
        return False
    if "filestream: fatal error" in internal_text:
        harness._report_wandb_degraded(
            "cohort-filestream",
            RuntimeError("W&B filestream reported a fatal upload error"),
        )
        return False
    records = harness._read_telemetry_spool(spool)
    epochs = [int(record["step"]) for record in records if record["kind"] == "epoch"]
    harness._atomic_json(
        _receipt_path(root, variant),
        {
            "contract_sha256": contract_sha256,
            "epoch": max(epochs, default=0),
            "final": any(record["kind"] == "final" for record in records),
            "schema": "lnet.wandb.telemetry_receipt.v1",
            "variant": variant,
        },
    )
    return True


def _publish_epoch_telemetry(
    root: Path,
    contract: dict[str, Any],
    contract_sha256: str,
    members: list[MemberState],
) -> None:
    python_state = random.getstate()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all()
    try:
        for member in members:
            spool = _spool_path(root, member.variant)
            harness._backfill_checkpoint_telemetry(spool, member.history)
            success = _publish_variant_spool(
                root,
                contract,
                contract_sha256,
                variant=member.variant,
                parameters=member.parameters,
            )
            if not success:
                continue
    finally:
        random.setstate(python_state)
        torch.set_rng_state(torch_state)
        torch.cuda.set_rng_state_all(cuda_state)


def _write_results(
    root: Path,
    contract: dict[str, Any],
    contract_sha256: str,
    members: list[MemberState],
) -> None:
    epochs = int(contract["recipe"]["epochs"])
    train_images = int(contract["data"]["train_images"])
    for member in members:
        if member.last_validation is None or len(member.history) != epochs:
            raise RuntimeError(f"cohort result is incomplete for {member.variant}")
        result = {
            "variant": member.variant,
            "seed": SEED,
            "contract_sha256": contract_sha256,
            "parameters": member.parameters,
            "global_step": member.global_step,
            "final_validation": member.last_validation,
            "best_validation_accuracy_diagnostic": max(
                row["validation_accuracy"] for row in member.history
            ),
            "training_seconds": member.training_seconds,
            "complete_training_examples_per_second": (
                epochs * train_images / member.training_seconds
            ),
            "cohort_model_images_per_second": (
                epochs * train_images * len(members) / member.training_seconds
            ),
            "history": member.history,
        }
        path = root / "results" / f"{member.variant}__seed{SEED}.json"
        harness._atomic_json(path, result)
        spool = _spool_path(root, member.variant)
        harness._append_telemetry_record(
            spool,
            {"kind": "final", "summary": harness._final_telemetry_summary(result)},
        )
    expected = {root / "results" / f"{variant}__seed{SEED}.json" for variant in stage.VARIANTS}
    if not all(path.is_file() for path in expected):
        raise RuntimeError("cohort did not publish all variant results")
    heads._summarize(root, contract)


def _results_complete(root: Path, contract_sha256: str, epochs: int) -> bool:
    paths = [root / "results" / f"{variant}__seed{SEED}.json" for variant in stage.VARIANTS]
    existing = [path.is_file() for path in paths]
    if not any(existing):
        return False
    if not all(existing):
        return False
    for path, variant in zip(paths, stage.VARIANTS, strict=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return False
        history = payload.get("history")
        if (
            payload.get("variant") != variant
            or payload.get("seed") != SEED
            or payload.get("contract_sha256") != contract_sha256
            or not isinstance(history, list)
            or len(history) != epochs
            or history[-1].get("epoch") != epochs
        ):
            return False
    return True


def _run(args: argparse.Namespace) -> None:
    if tuple(args.variants) != stage.VARIANTS:
        raise ValueError("shared-batch H200 cohort requires the frozen 13-variant order")
    if args.run_seeds != [SEED] or args.gradient_accumulation_steps != 1:
        raise ValueError("shared-batch cohort requires seed501 and accumulation=1")
    if os.environ.get("LNET_COMPILE_MODE", "default") != "default":
        raise ValueError("shared-batch cohort forbids CUDA Graph compile modes")
    runtime.configure(stage.VARIANTS, (SEED,))
    heads.VARIANTS = stage.VARIANTS
    heads.SEEDS = (SEED,)
    structured._training_objective = heads._training_objective
    structured._after_training_batch = heads._after_training_batch
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    dynamo_config = cast("Any", torch)._dynamo.config
    dynamo_config.recompile_limit = 64
    dynamo_config.cache_size_limit = 64

    contract = _contract(args)
    harness._configure_compile_runtime(args.root, contract["recipe"])
    args.root.mkdir(parents=True, exist_ok=True)
    _write_runtime_evidence(args.root, args)
    harness._initialize(args.root, contract)
    if args.initialize_only:
        return
    contract_sha256 = harness._contract_sha256(contract)
    if _results_complete(args.root, contract_sha256, int(contract["recipe"]["epochs"])):
        for variant in stage.VARIANTS:
            result = json.loads(
                (args.root / "results" / f"{variant}__seed{SEED}.json").read_text(encoding="utf-8")
            )
            harness._ensure_completed_telemetry_spool(
                _spool_path(args.root, variant),
                result,
            )
            _publish_variant_spool(
                args.root,
                contract,
                contract_sha256,
                variant=variant,
                parameters=int(result["parameters"]),
            )
        heads._summarize(args.root, contract)
        print(f"H200_COHORT_ALREADY_COMPLETE={args.root / 'summary.json'}", flush=True)
        return
    device = torch.device("cuda")
    members = _build_members(contract, device)
    recipe = contract["recipe"]
    training_generator = torch.Generator().manual_seed(SEED)
    mixup_generator = np.random.default_rng(SEED)
    permutation_generator = torch.Generator(device=device).manual_seed(SEED)
    train_loader, validation_loader = harness._loaders(
        args.data_root,
        batch_size=recipe["batch_size"],
        workers=args.workers,
        training_generator=training_generator,
    )
    for member in members:
        member.runtime = harness._build_runtime(member.model, recipe)
    start_epoch, preflight_evidence, previous_end_allocated = _restore(
        args.root,
        members,
        contract_sha256=contract_sha256,
        training_generator=training_generator,
        mixup_generator=mixup_generator,
        permutation_generator=permutation_generator,
    )
    if start_epoch == 0:
        random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    active = [member.active() for member in members]
    channels_last = bool(recipe.get("channels_last", False))
    if start_epoch >= 2 and preflight_evidence is None:
        raise RuntimeError("committed cohort epoch lacks matching preflight evidence")
    total_memory = torch.cuda.get_device_properties(device).total_memory

    for epoch_index in range(start_epoch, int(recipe["epochs"])):
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        metrics, optimizer_steps, host_wait = shared.train_epoch(
            active,
            train_loader,
            device=device,
            mixup_generator=mixup_generator,
            permutation_generator=permutation_generator,
            mixup_alpha=float(recipe["mixup_alpha"]),
            precision=str(recipe["precision"]),
            channels_last=channels_last,
            objective=heads._training_objective,
            after_model_batch=heads._after_training_batch,
            after_cohort_batch=_after_cohort_batch,
        )
        torch.cuda.synchronize(device)
        epoch_seconds = time.perf_counter() - started
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        end_allocated = torch.cuda.memory_allocated(device)
        free_memory, _ = torch.cuda.mem_get_info(device)
        allocated_drift = (
            0 if previous_end_allocated is None else max(0, end_allocated - previous_end_allocated)
        )
        maximum_drift = max(512 * 2**20, int(0.005 * total_memory))
        cohort_rate = len(cast("Any", train_loader).dataset) * len(members) / epoch_seconds
        if (
            epoch_index + 1 == 2
            and preflight_evidence is None
            and (
                cohort_rate < args.minimum_cohort_model_images_per_second
                or peak_reserved / total_memory >= 0.85
                or free_memory < 24 * 2**30
                or allocated_drift > maximum_drift
            )
        ):
            raise RuntimeError(
                "shared cohort preflight failed: "
                f"model_images_per_second={cohort_rate:.3f}, "
                f"peak_reserved_ratio={peak_reserved / total_memory:.4f}, "
                f"free_gib={free_memory / 2**30:.3f}, "
                f"allocated_drift_gib={allocated_drift / 2**30:.3f}"
            )
        previous_end_allocated = end_allocated
        next_preflight = preflight_evidence
        if epoch_index + 1 == 2 and preflight_evidence is None:
            next_preflight = {
                "schema": PREFLIGHT_SCHEMA,
                "status": "passed",
                "contract_sha256": contract_sha256,
                "compile_mode": "default",
                "variants": list(stage.VARIANTS),
                "epoch": 2,
                "minimum_cohort_model_images_per_second": (
                    args.minimum_cohort_model_images_per_second
                ),
                "maximum_peak_reserved_ratio": 0.85,
                "cohort_model_images_per_second": cohort_rate,
                "host_input_wait_fraction": host_wait / epoch_seconds,
                "peak_allocated_bytes": peak_allocated,
                "peak_reserved_bytes": peak_reserved,
                "free_memory_bytes": free_memory,
                "allocated_drift_bytes": allocated_drift,
                "maximum_allocated_drift_bytes": maximum_drift,
                "total_memory_bytes": total_memory,
            }
        validation = shared.evaluate(
            active,
            validation_loader,
            device=device,
            precision=str(recipe["precision"]),
            channels_last=channels_last,
            finalize=heads._evaluation_from_collected,
        )
        for member in members:
            member.training_seconds += epoch_seconds
            member.global_step += optimizer_steps
            member.last_validation = validation[member.variant]
            row = {
                "epoch": epoch_index + 1,
                "learning_rate": member.optimizer.param_groups[0]["lr"],
                "train_loss": metrics[member.variant]["loss"],
                "train_mixed_accuracy": metrics[member.variant]["mixed_accuracy"],
                "validation_accuracy": validation[member.variant]["accuracy"],
                "validation_cross_entropy": validation[member.variant]["cross_entropy"],
                "training_seconds": member.training_seconds,
                "global_step": member.global_step,
                "optimizer_steps": optimizer_steps,
                "host_input_wait_seconds": host_wait,
                "cohort_model_images_per_second": cohort_rate,
            }
            member.history.append(row)
            member.scheduler.step()
        manifest = _save_checkpoint(
            args.root,
            members,
            epoch=epoch_index + 1,
            contract_sha256=contract_sha256,
            recipe=recipe,
            training_generator=training_generator,
            mixup_generator=mixup_generator,
            permutation_generator=permutation_generator,
            preflight=next_preflight,
            end_allocated_bytes=end_allocated,
        )
        preflight_evidence = cast("dict[str, Any] | None", manifest["preflight"])
        for member in members:
            telemetry = harness._epoch_telemetry_metrics(member.history[-1])
            try:
                telemetry.update(model_metrics._wandb_model_metrics(member.model))
            except Exception as error:
                harness._report_wandb_degraded("cohort-model-metrics", error)
            harness._append_telemetry_record(
                _spool_path(args.root, member.variant),
                {"kind": "epoch", "metrics": telemetry, "step": epoch_index + 1},
            )
        print(
            "H200_COHORT_PROGRESS="
            + json.dumps(
                {
                    "epoch": epoch_index + 1,
                    "cohort_model_images_per_second": cohort_rate,
                    "host_input_wait_fraction": host_wait / epoch_seconds,
                    "peak_reserved_gib": peak_reserved / 2**30,
                    "checkpoint": manifest["cohort_token"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        _publish_epoch_telemetry(
            args.root,
            contract,
            contract_sha256,
            members,
        )
    if any(member.last_validation is None for member in members):
        validation = shared.evaluate(
            active,
            validation_loader,
            device=device,
            precision=str(recipe["precision"]),
            channels_last=channels_last,
            finalize=heads._evaluation_from_collected,
        )
        for member in members:
            member.last_validation = validation[member.variant]
    _write_results(args.root, contract, contract_sha256, members)
    _publish_epoch_telemetry(args.root, contract, contract_sha256, members)


def _after_cohort_batch(batch_index: int) -> None:
    """H200 wrapper replaces this with its owner-control boundary."""
    del batch_index


def main() -> None:
    args = _arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("shared-batch cohort requires CUDA")
    harness._configure_cpu_affinity()
    _run(args)


if __name__ == "__main__":
    main()
