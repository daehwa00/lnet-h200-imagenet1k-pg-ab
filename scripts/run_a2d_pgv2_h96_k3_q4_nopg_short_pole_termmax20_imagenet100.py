"""Train the ShortDamp control with terminal damping headroom raised to 2.0."""

from __future__ import annotations

# pyright: reportArgumentType=false, reportExplicitAny=false
# pyright: reportImplicitRelativeImport=false, reportPrivateUsage=false
import math
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import run_a2d_pgv2_h96_k3_rmsmatch_q4_affine_polelr1_decoupled_nopg_all_imagenet100 as no_pg
import torch

from lnet.pac_pole_initialization import (
    DAMPING_QUANTILE_POWER,
    NONTERMINAL_GEOMETRIC_DAMPING_RANGE,
    TERMINAL_DAMPING_MAX,
    TERMINAL_GEOMETRIC_DAMPING_RANGE,
)
from lnet.pac_pole_initialization import (
    damping as _damping,
)
from lnet.pac_pole_initialization import (
    install_short_damping as _install_short_damping,
)

if TYPE_CHECKING:
    from argparse import Namespace

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig, ComplexScanStage


SHORT_DAMP_VARIANT = "H96-K3-Q4-NoPG-ShortDamp-TermMax20"
TERMINAL_ATLAS_VARIANT = "H96-K3-Q4-NoPG-ShortDamp-TermMax20-TermAtlas"
VARIANTS = (SHORT_DAMP_VARIANT, TERMINAL_ATLAS_VARIANT)
SEEDS = no_pg.SEEDS
STAGE_NAMES = ("stage1", "stage2", "stage3", "terminal")

# These ranges are smooth distribution targets, not per-mode tables.  The map
# works for any complete decoupled atlas and preserves its pole ordering and
# exact x/y-paired anisotropy.
TERMINAL_FREQUENCY_MAX_RATIO = 0.75
TERMINAL_SECOND_RADIUS_RATIO = 0.65


def _configure_ramp() -> None:
    no_pg._configure_ramp()
    ramp = no_pg.control.base.local_reader.control.control.control.stemres.uniform.base
    ramp.VARIANT = SHORT_DAMP_VARIANT
    ramp.VARIANTS = VARIANTS
    ramp.SEEDS = SEEDS


def _install_terminal_frequency_atlas(stage: ComplexScanStage) -> None:
    orientations = no_pg.control.ORIENTATIONS
    if orientations != 8 or stage.modes % orientations:
        message = "terminal frequency candidate requires complete canonical-8 groups"
        raise ValueError(message)
    radial_levels = stage.modes // orientations
    if radial_levels < 3:
        message = "terminal frequency candidate requires at least three radial levels"
        raise ValueError(message)

    phase_x = stage.phase_x.detach().reshape(radial_levels, orientations)
    phase_y = stage.phase_y.detach().reshape(radial_levels, orientations)
    old_radius = torch.sqrt(phase_x.square() + phase_y.square())[:, 0]
    positive_min = old_radius[old_radius > 0].min()
    reduced_max = old_radius.max() * TERMINAL_FREQUENCY_MAX_RATIO
    middle = positive_min * TERMINAL_SECOND_RADIUS_RATIO
    remaining = torch.logspace(
        math.log10(float(positive_min)),
        math.log10(float(reduced_max)),
        radial_levels - 2,
        dtype=old_radius.dtype,
        device=old_radius.device,
    )
    radius = torch.cat((old_radius.new_zeros(1), middle.view(1), remaining))
    orientation = old_radius.new_tensor(
        (
            0.0,
            0.0,
            math.pi / 8.0,
            math.pi / 4.0,
            math.pi / 4.0,
            3.0 * math.pi / 8.0,
            math.pi / 2.0,
            math.pi / 2.0,
        )
    )
    with torch.no_grad():
        stage.phase_x.copy_((radius[:, None] * orientation.cos()).flatten())
        stage.phase_y.copy_((radius[:, None] * orientation.sin()).flatten())


def _assert_candidate(model: ComplexScanBackbone, variant: str) -> None:
    no_pg._assert_model(model)
    for stage_name in STAGE_NAMES:
        stage = getattr(model, stage_name)
        radial_levels = stage.modes // no_pg.control.ORIENTATIONS
        damping_x = _damping(stage, "x").reshape(radial_levels, -1)
        damping_y = _damping(stage, "y").reshape(radial_levels, -1)
        if not torch.allclose(damping_x, damping_y.flip(1), rtol=3.0e-6, atol=1.0e-7):
            message = f"{stage_name} lost exact x/y damping pairing"
            raise RuntimeError(message)
        geometric = torch.sqrt(damping_x * damping_y)
        lower, upper = (
            TERMINAL_GEOMETRIC_DAMPING_RANGE
            if stage_name == "terminal"
            else NONTERMINAL_GEOMETRIC_DAMPING_RANGE
        )
        if not torch.isclose(geometric.min(), geometric.new_tensor(lower), atol=2.0e-6):
            message = f"{stage_name} lost the short-memory lower endpoint"
            raise RuntimeError(message)
        if not torch.isclose(geometric.max(), geometric.new_tensor(upper), atol=2.0e-6):
            message = f"{stage_name} lost the short-memory upper endpoint"
            raise RuntimeError(message)
    if model.terminal.damping_max != TERMINAL_DAMPING_MAX:
        message = "terminal damping bound was not expanded"
        raise RuntimeError(message)
    if variant == TERMINAL_ATLAS_VARIANT:
        phase_x = model.terminal.phase_x.reshape(-1, no_pg.control.ORIENTATIONS)
        phase_y = model.terminal.phase_y.reshape(-1, no_pg.control.ORIENTATIONS)
        radius = torch.sqrt(phase_x.square() + phase_y.square())
        if not torch.allclose(radius[:, :1], radius, rtol=1.0e-6, atol=1.0e-7):
            message = "terminal atlas radial groups are inconsistent"
            raise RuntimeError(message)
        if not torch.equal(radius[0], torch.zeros_like(radius[0])):
            message = "terminal atlas lost its zero-frequency group"
            raise RuntimeError(message)


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    if variant not in VARIANTS:
        message = f"unsupported short-memory pole variant: {variant}"
        raise ValueError(message)
    model = no_pg._build(no_pg.VARIANT, config)
    for stage_name in STAGE_NAMES:
        _install_short_damping(
            getattr(model, stage_name),
            terminal=stage_name == "terminal",
        )
    if variant == TERMINAL_ATLAS_VARIANT:
        _install_terminal_frequency_atlas(model.terminal)
    _configure_ramp()
    _assert_candidate(model, variant)
    return model


def _variant_config(variant: str) -> dict[str, Any]:
    payload = deepcopy(no_pg._variant_config())
    payload["backbone"]["name"] = "A2D-H96-K3-Q4-NoPG-ShortMemoryPoleInit"
    payload["backbone"]["pole_initialization"]["short_memory"] = {
        "mapping": "log-damping quantile power map; no per-mode table",
        "quantile_power": DAMPING_QUANTILE_POWER,
        "nonterminal_geometric_range": list(NONTERMINAL_GEOMETRIC_DAMPING_RANGE),
        "terminal_geometric_range": list(TERMINAL_GEOMETRIC_DAMPING_RANGE),
        "terminal_damping_max": TERMINAL_DAMPING_MAX,
        "preserves": "pole order, damping rank, and exact x/y-paired anisotropy",
    }
    payload["backbone"]["pole_initialization"]["terminal_frequency_atlas"] = (
        {
            "enabled": True,
            "radial_groups": "one zero, one low-frequency, remaining log-spaced",
            "maximum_ratio_vs_control": TERMINAL_FREQUENCY_MAX_RATIO,
            "orientations": "50% axial, 25% diagonal, 25% intermediate",
        }
        if variant == TERMINAL_ATLAS_VARIANT
        else {"enabled": False, "frequency_atlas": "unchanged DecoupledInit control"}
    )
    return payload


def _contract(args: Namespace) -> dict[str, Any]:
    payload = no_pg._contract(args)
    ramp = no_pg.control.base.local_reader.control.control.control.stemres.uniform.base
    config = ramp.PoleModelConfig(output_dim=100, stem_strides=(2, 2))
    models = {variant: _build(variant, config) for variant in VARIANTS}
    payload["schema"] = "lnet.a2d.h96_k3_q4_nopg.short_memory_terminal_max20.imagenet100.v1"
    payload["evidence_status"] = "untrained controlled pole-initialization candidates"
    payload["variants"] = list(VARIANTS)
    payload["seeds"] = list(SEEDS)
    payload["variant_configs"] = {variant: _variant_config(variant) for variant in VARIANTS}
    payload["parameter_counts"] = {
        variant: sum(parameter.numel() for parameter in model.parameters())
        for variant, model in models.items()
    }
    payload["architecture"] = {
        SHORT_DAMP_VARIANT: (
            "Exact DecoupledInit Strict-K3 NoPG-All Q4Affine control with only a "
            "stage-aware short-memory damping prior and terminal damping_max=2.0."
        ),
        TERMINAL_ATLAS_VARIANT: (
            "The ShortDamp candidate with only the terminal frequency atlas additionally "
            "changed to include zero/low-frequency and axis-heavy groups."
        ),
    }
    payload["source_sha256"]["short_pole_runner"] = ramp.heads.harness._digest(Path(__file__))
    return payload


def _build_optimizer(model: torch.nn.Module, recipe: dict[str, Any]) -> torch.optim.Optimizer:
    return no_pg.control.base._build_optimizer(model, recipe)


def main() -> None:
    _configure_ramp()
    ramp = no_pg.control.base.local_reader.control.control.control.stemres.uniform.base
    source = ramp.canonical8.fair_init.backbone.deep4.baseline.baseline
    harness = source.heads.harness
    source.heads.VARIANTS = VARIANTS
    source.heads.SEEDS = SEEDS
    source.structured._training_objective = source.heads._training_objective
    source.structured._after_training_batch = source.heads._after_training_batch
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    harness.main(
        harness.runner_bindings(
            variants=VARIANTS,
            seeds=SEEDS,
            model_config=ramp.PoleModelConfig,
            build_model=_build,
            contract=_contract,
            build_optimizer=_build_optimizer,
            prepare_model=source._prepare_model,
            train_epoch=source.structured._train_epoch,
            evaluate=source.heads._evaluate,
            wandb_model_metrics=no_pg.control.base.local_reader.control.control._wandb_model_metrics,
            summarize=source.heads._summarize,
        )
    )


if __name__ == "__main__":
    main()
