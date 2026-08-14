from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from time import sleep
from typing import TYPE_CHECKING

from .hybrid_experiment_types import resolve_device
from .pac_overnight_audit import run_param_audit
from .pac_overnight_diagnostics import run_damping_diagnostics
from .pac_overnight_efficiency import run_efficiency_audit
from .pac_overnight_io import append_queue_event, prepare_overnight_dirs
from .pac_overnight_knockout import run_expanded_knockout
from .pac_overnight_real import run_real_baselines
from .pac_overnight_report import write_overnight_summary
from .pac_overnight_types import OvernightConfig, QueueEvent, QueueStatus
from .pac_types import PACExperimentConfig

if TYPE_CHECKING:
    from collections.abc import Callable

    from .pac_types import PACDevice


def sanity_config(root: Path, device: PACDevice) -> PACExperimentConfig:
    return PACExperimentConfig(
        sample_count=16,
        validation_count=8,
        test_count=8,
        sequence_length=40,
        raw_input_dim=2,
        output_dim=2,
        model_dim=8,
        modes=4,
        tap_kernel_size=8,
        fir_kernel_size=5,
        epochs=1,
        batch_size=8,
        learning_rate=3.0e-3,
        weight_decay=1.0e-4,
        seeds=(7,),
        device=device,
        output_dir=root,
    )


def full_config(root: Path, device: PACDevice) -> PACExperimentConfig:
    return PACExperimentConfig(2048, 512, 512, 64, device=device, output_dir=root)


def run_sanity(config: OvernightConfig) -> None:
    root = config.output_root
    prepare_overnight_dirs(root)
    device = resolve_device(config.device)
    pac_config = sanity_config(root, config.device)
    _event(root, "sanity", "start", "running")
    run_param_audit(root, replace(pac_config, raw_input_dim=1), _audit_models())
    run_efficiency_audit(
        root,
        replace(pac_config, raw_input_dim=1, output_dim=3),
        device,
        ("pac_lite", "gru"),
        (32,),
        warmup_iters=1,
        timed_iters=1,
    )
    run_damping_diagnostics(root, pac_config, device, (7,), (0.0, 0.5))
    write_overnight_summary(root)
    _event(root, "sanity", "complete", "done")


def run_queue(config: OvernightConfig, *, wait_for_current_synthetic: bool) -> None:
    root = config.output_root
    prepare_overnight_dirs(root)
    if wait_for_current_synthetic:
        _wait_for_current_synthetic(root)
    device = resolve_device(config.device)
    pac_config = full_config(root, config.device)

    def param_audit_stage() -> None:
        run_param_audit(root, replace(pac_config, raw_input_dim=1), _audit_models())

    def efficiency_audit_stage() -> None:
        run_efficiency_audit(
            root,
            replace(pac_config, raw_input_dim=1, output_dim=5),
            device,
            _efficiency_models(),
            (128, 256, 512, 1024, 2048, 4096),
        )

    def damping_diagnostics_stage() -> None:
        run_damping_diagnostics(root, pac_config, device, config.seeds, (0.0, 0.5, 1.0))

    def real_baselines_stage() -> None:
        run_real_baselines(
            root, pac_config, device, ("ECG5000", "FordA"), _real_models(), config.seeds
        )

    def expanded_knockout_stage() -> None:
        run_expanded_knockout(root, pac_config, device, config.seeds)

    _run_stage(root, "param_audit", param_audit_stage)
    _run_stage(root, "efficiency_audit", efficiency_audit_stage)
    _run_stage(root, "damping_diagnostics", damping_diagnostics_stage)
    _run_stage(root, "real_baselines", real_baselines_stage)
    _run_stage(root, "expanded_knockout", expanded_knockout_stage)
    write_overnight_summary(root)


def _run_stage(root: Path, stage: str, action: Callable[[], None]) -> None:
    if _stage_done(root, stage):
        _event(root, stage, "skip_already_done", "done")
        return
    _event(root, stage, "start", "running")
    try:
        action()
    except (RuntimeError, ValueError, OSError, KeyError) as error:
        _event(root, stage, "error", "failed", str(error))
    else:
        _event(root, stage, "complete", "done")


def _stage_done(root: Path, stage: str) -> bool:
    path = root / "queue_state.jsonl"
    if not path.exists():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("stage") == stage and event.get("item") == "complete":
            return event.get("status") == "done"
    return False


def _wait_for_current_synthetic(root: Path) -> None:
    _event(root, "wait_for_current_synthetic", "start", "running")
    while _current_synthetic_running():
        sleep(300)
    _event(root, "wait_for_current_synthetic", "complete", "done")


def _current_synthetic_running() -> bool:
    proc_root = Path("/proc")
    for cmdline in proc_root.glob("[0-9]*/cmdline"):
        try:
            text = cmdline.read_text(errors="ignore").replace("\x00", " ")
        except OSError:
            continue
        if "pac_hybrid_cli --mode synthetic" in text or "synthetic_seed_" in text:
            return True
    return False


def _event(root: Path, stage: str, item: str, status: QueueStatus, notes: str = "") -> None:
    append_queue_event(root, QueueEvent(stage, item, status, notes))


def _audit_models() -> tuple[str, ...]:
    return (
        "pac_lite",
        "pac_full",
        "controlled_tapped_prl_only",
        "fixed_prl",
        "gru",
        "lstm",
        "cnn1d",
        "tcn",
        "transformer_tiny",
        "fir_classifier",
    )


def _efficiency_models() -> tuple[str, ...]:
    return (
        "pac_lite",
        "pac_full",
        "gru",
        "lstm",
        "cnn1d",
        "tcn",
        "transformer_tiny",
        "fir_classifier",
    )


def _real_models() -> tuple[str, ...]:
    return (
        "pac_lite",
        "pac_full",
        "pac_no_damping_control",
        "gru",
        "lstm",
        "cnn1d",
        "cnn1d_small",
        "tcn",
        "tcn_small",
        "transformer_tiny",
        "fir_classifier",
    )
