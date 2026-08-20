"""Stable ImageNet-100 runtime boundary for the active R2K3 runners.

Model construction still goes through the frozen PGv2 ancestry because changing
that sequence changes seeded initialization.  This module keeps that
compatibility dependency in one place instead of exposing its nested module
graph to every current runner.
"""

from __future__ import annotations

# pyright: reportExplicitAny=false, reportImplicitRelativeImport=false
# pyright: reportPrivateLocalImportUsage=false, reportPrivateUsage=false
import functools
import hashlib
import importlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import a2d_r2k3_source_manifest as source_manifest
import torch

from lnet.complex_scan import ComplexScanConfig
from lnet.image_layers import ModeScaledTwoConvStem, ResidualPreComplexMixer
from lnet.pac_pole_initialization import (
    TERMINAL_DAMPING_MAX,
    TERMINAL_GEOMETRIC_DAMPING_RANGE,
)
from lnet.pac_pole_initialization import (
    install_short_damping as _install_short_damping,
)

if TYPE_CHECKING:
    from argparse import Namespace
    from collections.abc import Callable, Sequence

    from torch import nn

    from lnet.complex_scan import ComplexScanStage


DEFAULT_SEEDS = (501,)
STAGE_NAMES = ("stage1", "stage2", "stage3", "terminal")
_LEGACY_TRAINING_MODULE = (
    "run_a2d_pgv2_h96_rank2_polewise_k3_q4_nopg_short_pole_termmax20_rawq_imagenet100"
)

# The frozen training recipe is assembled from these ancestry modules.  They
# used to be reached by walking module aliases through the experiment chain --
# an attribute path seven levels deep -- which hid the real dependency from
# grep and from readers.  Naming them keeps
# the same modules and the same import order while making the coupling
# explicit.  tests/test_r2k3_runtime_recipe.py pins these against the walk.
_HARNESS_MODULE = "run_alphabet2d_imagenet100_nano"
_HEADS_MODULE = "run_a2d_affine_qhead_imagenet100"
_STRUCTURED_MODULE = "run_a2d_qhead_e2e_imagenet100"
_PREPARE_MODULE = "run_a2d_resaux1_imagenet100"
_METRICS_MODULE = "run_a2d_deep4_calibrated_uniform_p96_phase_gated_imagenet100"


@functools.cache
def _legacy_module(name: str) -> Any:
    return importlib.import_module(name)


def _legacy_training() -> Any:
    return _legacy_module(_LEGACY_TRAINING_MODULE)


def configure(
    variants: tuple[str, ...],
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> None:
    """Configure the frozen training harness for one current R2K3 campaign."""
    if not variants or not seeds:
        message = "R2K3 runtime requires at least one variant and seed"
        raise ValueError(message)
    legacy = _legacy_training()
    legacy._configure_ramp()
    ramp = legacy._ramp()
    ramp.VARIANT = variants[0]
    ramp.VARIANTS = variants
    ramp.SEEDS = seeds


def model_config(*, output_dim: int = 100) -> ComplexScanConfig:
    return ComplexScanConfig(output_dim=output_dim, stem_strides=(2, 2))


def base_contract(args: Namespace) -> dict[str, Any]:
    return _legacy_training()._contract(args)


def digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_dependency_paths(repo: Path, roots: Sequence[str]) -> tuple[Path, ...]:
    return source_manifest.dependency_paths(repo, roots)


def source_fingerprint(repo: Path, paths: Sequence[Path]) -> str:
    return source_manifest.fingerprint(repo, paths)


def build_optimizer(
    model: nn.Module,
    recipe: dict[str, Any],
) -> torch.optim.Optimizer:
    return _legacy_training()._build_optimizer(model, recipe)


def make_stem(complex_width: int, strides: tuple[int, int]) -> nn.Module:
    return ModeScaledTwoConvStem(
        complex_width,
        strides,
        hidden_width=32,
    )


def wrap_precomplex_mixer(source: nn.Module) -> nn.Module:
    return ResidualPreComplexMixer(source)


def install_short_damping(stage: ComplexScanStage, *, terminal: bool) -> None:
    _install_short_damping(stage, terminal=terminal)


def run(
    *,
    variants: tuple[str, ...],
    build_model: Callable[..., nn.Module],
    contract: Callable[[Namespace], dict[str, Any]],
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> None:
    """Run one R2K3 campaign through the established training recipe."""
    configure(variants, seeds)
    harness = _legacy_module(_HARNESS_MODULE)
    heads = _legacy_module(_HEADS_MODULE)
    structured = _legacy_module(_STRUCTURED_MODULE)
    prepare = _legacy_module(_PREPARE_MODULE)
    metrics = _legacy_module(_METRICS_MODULE)
    heads.VARIANTS = variants
    heads.SEEDS = seeds
    structured._training_objective = heads._training_objective
    structured._after_training_batch = heads._after_training_batch
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    harness.main(
        harness.runner_bindings(
            variants=variants,
            seeds=seeds,
            model_config=ComplexScanConfig,
            build_model=build_model,
            contract=contract,
            build_optimizer=build_optimizer,
            prepare_model=prepare._prepare_model,
            train_epoch=structured._train_epoch,
            evaluate=heads._evaluate,
            wandb_model_metrics=metrics._wandb_model_metrics,
            summarize=heads._summarize,
        )
    )


__all__ = [
    "DEFAULT_SEEDS",
    "STAGE_NAMES",
    "TERMINAL_DAMPING_MAX",
    "TERMINAL_GEOMETRIC_DAMPING_RANGE",
    "base_contract",
    "build_optimizer",
    "configure",
    "digest",
    "install_short_damping",
    "make_stem",
    "model_config",
    "run",
    "source_dependency_paths",
    "source_fingerprint",
    "wrap_precomplex_mixer",
]
