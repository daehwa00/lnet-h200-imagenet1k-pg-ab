# pyright: reportPrivateUsage=false
# ruff: noqa: SLF001
"""Sealed 30-task Q1 campaign for three-stage causal LogSig V5."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Final

from optimization.three_stage_causal_log_signature import (
    FullyPoleNativeThreeStageCausalALPHABET,
)

from . import pac_baseline_fairness_maximal as runner
from . import pac_two_tap_q1_campaign as campaign
from .pac_baseline_fairness_maximal import ResourceLane
from .pac_campaign_utils import canonical_json_sha256, source_file_hashes

if TYPE_CHECKING:
    from torch import nn

    from .pac_baseline_fairness_maximal import FairnessJob
    from .pac_external_tasks import ExternalSelectionTask, ExternalTask
    from .pac_types import PACDevice, PACExperimentConfig

DEFAULT_ROOT: Final = Path(".omx/results/pac-logsig-v5-q1-final-20260721")
DEFAULT_BASELINE_ROOT: Final = campaign.DEFAULT_BASELINE_ROOT
CANDIDATE: Final = "logsig_v5_three_stage_causal"

_ORIGINAL_EXPERIMENT_CONFIG = runner._experiment_config
_ORIGINAL_ENABLE_OPTIMIZED = runner._enable_pac_optimized_training
_ORIGINAL_BUILD_UCR_MODEL = runner._build_ucr_model
_ORIGINAL_BUILD_EXTERNAL_MODEL = runner._build_external_model


def _source_paths() -> tuple[Path, ...]:
    return (
        Path("src/lnet/pac_logsig_v5_q1_campaign.py"),
        Path("src/lnet/pac_logsig_v5_q1_cli.py"),
        Path("src/lnet/pac_logsig_v5_preflight.py"),
        Path("src/lnet/pac_two_tap_q1_campaign.py"),
        Path("src/lnet/pac_baseline_fairness_maximal.py"),
        Path("optimization/three_stage_causal_log_signature.py"),
        Path("optimization/pointwise_causal_log_signature.py"),
        Path("optimization/learned_two_tap_log_signature.py"),
        Path("optimization/learned_two_tap_alphabet.py"),
        Path("optimization/masked_modal_moments.py"),
        Path("optimization/stage2_recurrence.py"),
        Path("optimization/stage2_tail_metadata.py"),
        Path("src/lnet/pac_recurrence.py"),
        Path("src/lnet/pac_triton_log_signature.py"),
        Path("src/lnet/pac_triton_log_signature_training.py"),
    )


def _source_hashes() -> dict[str, str]:
    return source_file_hashes(
        [str(p) for p in _source_paths()],
        project_root=Path(__file__).resolve().parents[2],
    )


def _source_manifest() -> dict[str, object]:
    body: dict[str, object] = {
        "schema": "pac_logsig_v5_q1_source_manifest.v1",
        "source_sha256": _source_hashes(),
        "candidate": CANDIDATE,
        "model_contract": {
            "token_grid": "nodes",
            "input_local_lift": "pointwise-linear--causal-dwconv-k5-d1--silu",
            "writer_local_lift": "causal-dwconv-k5-d2--silu--residual-scale-0.1",
            "reader_local_lift": "pointwise-linear--causal-dwconv-k5-d2--silu--residual-scale-0.1",
            "writer_event_dimension": 7,
            "reader_event_dimension": 7,
            "degree_two_log_signature_dimension": 28,
            "descriptor_width_per_mode": 5,
            "fixed_writer_lags": False,
            "fixed_reader_lags": False,
        },
    }
    return {**body, "sha256": canonical_json_sha256(body)}


def code_sha256() -> str:
    return canonical_json_sha256(_source_hashes())


def _build_candidate(
    config: PACExperimentConfig,
    output_dim: int,
) -> FullyPoleNativeThreeStageCausalALPHABET:
    return FullyPoleNativeThreeStageCausalALPHABET(
        config.raw_input_dim,
        config.model_dim,
        config.modes,
        output_dim,
    )


def _build_ucr_model(
    job: FairnessJob,
    config: PACExperimentConfig,
    class_count: int,
) -> nn.Module:
    if job.model == CANDIDATE:
        return _build_candidate(config, class_count)
    return _ORIGINAL_BUILD_UCR_MODEL(job, config, class_count)


def _build_external_model(
    job: FairnessJob,
    config: PACExperimentConfig,
    task: ExternalTask | ExternalSelectionTask,
) -> nn.Module:
    if job.model == CANDIDATE:
        return _build_candidate(config, task.output_dim)
    return _ORIGINAL_BUILD_EXTERNAL_MODEL(job, config, task)


def _experiment_config(
    *,
    train_count: int,
    validation_count: int,
    test_count: int,
    sequence_length: int,
    input_dim: int,
    output_dim: int,
    job: FairnessJob,
    device: PACDevice,
) -> PACExperimentConfig:
    active_job = replace(job, model="two_tap_h_only") if job.model == CANDIDATE else job
    return _ORIGINAL_EXPERIMENT_CONFIG(
        train_count=train_count,
        validation_count=validation_count,
        test_count=test_count,
        sequence_length=sequence_length,
        input_dim=input_dim,
        output_dim=output_dim,
        job=active_job,
        device=device,
    )


def _enable_optimized(model: nn.Module, model_name: str) -> None:
    alias = "two_tap_h_only" if model_name == CANDIDATE else model_name
    _ORIGINAL_ENABLE_OPTIMIZED(model, alias)


def install_worker_overrides() -> None:
    """Install V5-specific hooks without mutating the baseline runner source."""
    runner._build_ucr_model = _build_ucr_model
    runner._build_external_model = _build_external_model
    runner._experiment_config = _experiment_config
    runner._enable_pac_optimized_training = _enable_optimized
    runner._code_sha256 = code_sha256


def _activate_campaign() -> None:
    campaign.CANDIDATE = CANDIDATE
    campaign._source_manifest = _source_manifest


def default_lanes(count: int = 14) -> tuple[ResourceLane, ...]:
    """Describe the physical V5 workers with measured relative throughput."""
    if count != 14:
        return campaign.default_lanes(count)
    pro = tuple(
        ResourceLane(f"pro6000-{index:02d}", "pro6000", 0, index, 1.0)
        for index in range(11)
    )
    local_gpu = (
        ResourceLane("local_gpu-gpu0", "pro6000", 0, 0, 3.5),
        ResourceLane("local_gpu-gpu1", "pro6000", 1, 0, 3.5),
    )
    kau = (ResourceLane("kau-gpu0", "pro6000", 0, 0, 1.2),)
    return (*pro, *local_gpu, *kau)


def enqueue_stage1(root: Path = DEFAULT_ROOT, *, lanes: tuple[ResourceLane, ...] | None = None):  # noqa: ANN201
    _activate_campaign()
    return campaign.enqueue_stage1(root, lanes=lanes)


def select_stage1(root: Path = DEFAULT_ROOT, *, lanes: tuple[ResourceLane, ...] | None = None):  # noqa: ANN201
    _activate_campaign()
    return campaign.select_stage1(root, lanes=lanes)


def select_stage2(  # noqa: ANN201
    root: Path = DEFAULT_ROOT,
    *,
    baseline_root: Path = DEFAULT_BASELINE_ROOT,
):
    _activate_campaign()
    return campaign.select_stage2(root, baseline_root=baseline_root)


def enqueue_final(  # noqa: ANN201
    root: Path = DEFAULT_ROOT,
    *,
    baseline_root: Path = DEFAULT_BASELINE_ROOT,
    lanes: tuple[ResourceLane, ...] | None = None,
):
    _activate_campaign()
    return campaign.enqueue_final(root, baseline_root=baseline_root, lanes=lanes)


def status(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    _activate_campaign()
    return campaign.status(root)


__all__ = [
    "CANDIDATE",
    "DEFAULT_BASELINE_ROOT",
    "DEFAULT_ROOT",
    "code_sha256",
    "default_lanes",
    "enqueue_final",
    "enqueue_stage1",
    "install_worker_overrides",
    "select_stage1",
    "select_stage2",
    "status",
]
