"""Q-only single-FC follow-up on the frozen UCR18 configurations."""

# pyright: reportPrivateUsage=false
# ruff: noqa: SLF001

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

from . import pac_baseline_fairness_maximal as runner
from .pac_baseline_fairness_maximal import FairnessJob, ResourceLane
from .pac_campaign_utils import canonical_json_sha256, file_sha256, write_once
from .pac_confirmatory_baselines import confirmatory_trial_spec
from .pac_efp16_final_campaign import UCR_DATASETS
from .pac_final_validation import UCR_SECONDS
from .pac_q_only_linear import QOnlyLinearPAC

if TYPE_CHECKING:
    from torch import nn

    from .pac_types import PACDevice, PACExperimentConfig


DEFAULT_ROOT: Final = Path(
    ".omx/results/pac-q-only-linear-ucr18-local_gpu-20260724"
)
FINAL_SEEDS: Final = (61, 67, 71, 73, 79, 83, 89, 97, 101, 103)
MODEL_NAME: Final = "q_only_linear"


def default_lanes() -> tuple[ResourceLane, ...]:
    return tuple(
        ResourceLane(
            f"local_gpu-gpu{index // 2}-lane{index % 2:02d}",
            "local_gpu",
            index // 2,
            index % 2,
            1.0,
        )
        for index in range(4)
    )


def jobs(selection: dict[str, object]) -> list[FairnessJob]:
    selected = cast("dict[str, dict[str, object]]", selection["selected"])
    active: list[FairnessJob] = []
    for dataset in UCR_DATASETS:
        choice = selected[dataset]
        width = int(choice["width"])
        modes = int(choice["modes"])
        trial = int(choice["trial"])
        recipe = confirmatory_trial_spec("pac_tf", trial)
        active.extend(
            FairnessJob(
                stage="final",
                suite="ucr",
                dataset=dataset,
                model=MODEL_NAME,
                width_tier=1,
                width=width,
                modes=modes,
                trial=trial,
                split_seed=seed,
                train_seed=seed,
                epochs=100,
                batch_size=recipe.batch_size,
                learning_rate=recipe.learning_rate,
                weight_decay=recipe.weight_decay,
                grad_clip_norm=recipe.grad_clip_norm,
                evaluation_split="test",
                estimated_seconds=UCR_SECONDS[dataset]
                * max(0.75, width / 32.0),
            )
            for seed in FINAL_SEEDS
        )
    return active


def _source_manifest() -> dict[str, object]:
    project = Path(__file__).resolve().parents[2]
    names = (
        "src/lnet/alphabet_backbone.py",
        "src/lnet/pac_q_only_linear.py",
        "src/lnet/pac_q_only_linear_campaign.py",
        "src/lnet/pac_baseline_fairness_maximal.py",
        "src/lnet/pac_training.py",
    )
    hashes = {
        name: file_sha256(project / name)
        for name in names
    }
    body: dict[str, object] = {
        "schema": "pac_q_only_linear_source_manifest.v1",
        "source_sha256": hashes,
    }
    return {**body, "sha256": canonical_json_sha256(body)}


def enqueue(
    selection_path: Path,
    root: Path = DEFAULT_ROOT,
) -> dict[str, object]:
    selection_text = selection_path.read_text(encoding="utf-8")
    selection = json.loads(selection_text)
    active_jobs = jobs(selection)
    loads = runner._write_manifests(root, "final", active_jobs, default_lanes())
    write_once(root / "stage2" / "selection.json", selection_text)
    source = _source_manifest()
    write_once(
        root / "reports/source_manifest.json",
        json.dumps(source, indent=2, sort_keys=True) + "\n",
    )
    contract: dict[str, object] = {
        "schema": "pac_q_only_linear_contract.v1",
        "purpose": "Q-only, no-pooled, no-lag, single-linear-head comparison",
        "head_input": "writer and reader log-energy only (2M)",
        "head": "Linear(2M, K)",
        "nonlinearity": False,
        "adapter": False,
        "jobs": len(active_jobs),
        "seeds": list(FINAL_SEEDS),
        "selection_source": str(selection_path),
        "official_test_accessed": True,
        "source_manifest_sha256": source["sha256"],
        "estimated_normalized_lane_seconds": loads,
    }
    write_once(
        root / "contract.json",
        json.dumps(contract, indent=2, sort_keys=True) + "\n",
    )
    return contract


_hooks_installed = False


def install_runner_hooks() -> None:
    global _hooks_installed  # noqa: PLW0603
    if _hooks_installed:
        return
    original_ucr_builder = runner._build_ucr_model
    original_experiment_config = runner._experiment_config
    original_enable = runner._enable_pac_optimized_training

    def build_ucr(job: FairnessJob, config: PACExperimentConfig, output_dim: int):  # noqa: ANN202
        if job.model == MODEL_NAME:
            return QOnlyLinearPAC(config, output_dim, objective="classification")
        return original_ucr_builder(job, config, output_dim)

    def experiment_config(
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
        active = (
            replace(job, model="h_compact_lag124_tied")
            if job.model == MODEL_NAME
            else job
        )
        config = original_experiment_config(
            train_count=train_count,
            validation_count=validation_count,
            test_count=test_count,
            sequence_length=sequence_length,
            input_dim=input_dim,
            output_dim=output_dim,
            job=active,
            device=device,
        )
        return replace(config, optimizer_mode="default")

    def enable_optimized(model: nn.Module, model_name: str) -> None:
        if model_name == MODEL_NAME:
            return
        original_enable(model, model_name)

    runner._build_ucr_model = build_ucr
    runner._experiment_config = experiment_config
    runner._enable_pac_optimized_training = enable_optimized
    _hooks_installed = True


def run_manifest(
    root: Path,
    manifest: Path,
    *,
    device: str = "cuda",
    ucr_data_root: Path = Path(".omx/data/ucr"),
) -> None:
    install_runner_hooks()
    runner.run_manifest(
        root,
        manifest,
        device=cast("PACDevice", device),
        ucr_data_root=ucr_data_root,
    )


def status(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    return {
        "schema": "pac_q_only_linear_status.v1",
        "final": runner.campaign_status(root)["final"],
    }


__all__ = [
    "DEFAULT_ROOT",
    "enqueue",
    "jobs",
    "run_manifest",
    "status",
]
