#!/usr/bin/env python3
"""Train PGv2-H96 with an identity-initialized local complex scan reader."""

from __future__ import annotations

# pyright: reportArgumentType=false, reportExplicitAny=false
# pyright: reportImplicitRelativeImport=false, reportPrivateLocalImportUsage=false
# pyright: reportPrivateUsage=false
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import run_a2d_deep4_calibrated_uniform_p96_phase_gated_h96_imagenet100 as control
import torch
from torch import nn

from lnet.pac_complex_scan_reader import PackedComplexConv2dReader
from lnet.pac_phase_gated_transition import (
    AveragePoolMagnitudeGateTransition,
    AveragePoolPhaseGatedRefinementTransition,
    MemoryOnlyMagnitudeGateTransition,
    PathOnlyCollapse,
    PathPhaseGatedCollapse,
    PhaseGatedModeResidualPathCollapse,
    PureMagnitudeGateTransition,
)

if TYPE_CHECKING:
    from argparse import Namespace

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig, ComplexScanStage
    from lnet.pac_mean_one_magnitude_gate import MeanOneMagnitudeGate
    from lnet.pac_phase_gated_cffn import PhaseGatedComplexFFN


KERNEL_SIZE = 3
VARIANT = "PGv2-H96-ScanConvK3-UnitVar"
RMS_MATCH_VARIANT = "PGv2-H96-K3-RMSMatch-PGNoWD"
UNIT_ROW_VARIANT = "PGv2-H96-K3-RMSMatch-UnitRow"
MODE_GAIN_VARIANT = "PGv2-H96-K3-RMSMatch-ModeGain"
NO_PG_STAGE3_VARIANT = "PGv2-H96-K3-RMSMatch-NoPG-S3"
NO_PG_ALL_VARIANT = "PGv2-H96-K3-RMSMatch-NoPG-All"
PATH_PG_VARIANT = "PGv2-H96-K3-RMSMatch-PathPG8"
PATH_PG_NO_SILU_VARIANT = "PGv2-H96-K3-RMSMatch-PathPG8-NoSiLU"
AVG_REEXCITE_VARIANT = "PGv2-H96-K3-RMSMatch-PathPG8-AvgRePG"
AVG_PURE_GATE_VARIANT = "PGv2-H96-K3-RMSMatch-PathPG8-AvgPureGate"
AVG_PURE_GATE_NO_STEM_RMS_VARIANT = "PGv2-H96-K3-RMSMatch-PathPG8-AvgPureGate-NoStemRMS"
MEMORY_ONLY_PURE_GATE_VARIANT = "PGv2-H96-K3-RMSMatch-PathPG8-PureGate-NoCarry"
AVG_REEXCITE_NOTE_PREFIX = (
    "; the collapsed memory is added to a fixed 2x2 average of the pre-reader"
)
AVG_REEXCITE_NOTE_MIDDLE = "excitation, then a shared H96 PG applies an exact unit-scale residual"
AVG_REEXCITE_NOTE_SUFFIX = "refinement. Q1536, head, and recipe are unchanged."
AVG_REEXCITE_ARCHITECTURE_NOTE = (
    f"{AVG_REEXCITE_NOTE_PREFIX} {AVG_REEXCITE_NOTE_MIDDLE} {AVG_REEXCITE_NOTE_SUFFIX}"
)
AVG_PURE_GATE_NOTE_PREFIX = "; fixed 2x2 average carry is merged with memory and pure mean-one"
AVG_PURE_GATE_NOTE_SUFFIX = "magnitude gating computes E_next=X*g without projections or gamma."
AVG_PURE_GATE_ARCHITECTURE_NOTE = f"{AVG_PURE_GATE_NOTE_PREFIX} {AVG_PURE_GATE_NOTE_SUFFIX}"
MEMORY_ONLY_PURE_GATE_NOTE_PREFIX = (
    "; S2D carry is absent and pure mean-one magnitude gating is applied"
)
MEMORY_ONLY_PURE_GATE_NOTE_SUFFIX = (
    "directly to collapsed scan memory without projections or gamma."
)
MEMORY_ONLY_PURE_GATE_ARCHITECTURE_NOTE = (
    f"{MEMORY_ONLY_PURE_GATE_NOTE_PREFIX} {MEMORY_ONLY_PURE_GATE_NOTE_SUFFIX}"
)
NO_STEM_RMS_NOTE_PREFIX = " The RMSNorm between the residual real stem mixer and orthogonal"
NO_STEM_RMS_NOTE_SUFFIX = "complex analysis is removed."
NO_STEM_RMS_ARCHITECTURE_NOTE = f"{NO_STEM_RMS_NOTE_PREFIX} {NO_STEM_RMS_NOTE_SUFFIX}"
RMS_MATCH_VARIANTS = (
    RMS_MATCH_VARIANT,
    UNIT_ROW_VARIANT,
    MODE_GAIN_VARIANT,
    NO_PG_STAGE3_VARIANT,
    NO_PG_ALL_VARIANT,
    PATH_PG_VARIANT,
    PATH_PG_NO_SILU_VARIANT,
    AVG_REEXCITE_VARIANT,
    AVG_PURE_GATE_VARIANT,
    AVG_PURE_GATE_NO_STEM_RMS_VARIANT,
    MEMORY_ONLY_PURE_GATE_VARIANT,
)
PATH_PG_VARIANTS = (
    PATH_PG_VARIANT,
    PATH_PG_NO_SILU_VARIANT,
    AVG_REEXCITE_VARIANT,
    AVG_PURE_GATE_VARIANT,
    AVG_PURE_GATE_NO_STEM_RMS_VARIANT,
    MEMORY_ONLY_PURE_GATE_VARIANT,
)
PURE_GATE_VARIANTS = (
    AVG_PURE_GATE_VARIANT,
    AVG_PURE_GATE_NO_STEM_RMS_VARIANT,
    MEMORY_ONLY_PURE_GATE_VARIANT,
)
VARIANTS = (
    VARIANT,
    RMS_MATCH_VARIANT,
    UNIT_ROW_VARIANT,
    MODE_GAIN_VARIANT,
    NO_PG_STAGE3_VARIANT,
    NO_PG_ALL_VARIANT,
    PATH_PG_VARIANT,
    PATH_PG_NO_SILU_VARIANT,
    AVG_REEXCITE_VARIANT,
    AVG_PURE_GATE_VARIANT,
    AVG_PURE_GATE_NO_STEM_RMS_VARIANT,
    MEMORY_ONLY_PURE_GATE_VARIANT,
)
SEEDS = control.SEEDS
P = control.P
PG_RESIDUAL_SCALE_MAX = 0.5
PG_OUTPUT_GAIN_MAX = 0.5


def _configure_ramp() -> None:
    control._configure_ramp()
    ramp = control.control.control.stemres.uniform.base
    ramp.VARIANT = VARIANT
    ramp.VARIANTS = VARIANTS
    ramp.SEEDS = SEEDS


def _install_local_reader(stage: ComplexScanStage, *, match_input_rms: bool) -> None:
    if stage.modes != P:
        message = "PGv2-H96 local reader requires an unchanged uniform-P stage"
        raise RuntimeError(message)
    # Construction must not perturb the control's data-order or augmentation RNG.
    with torch.random.fork_rng(devices=[]):
        reader = PackedComplexConv2dReader(
            P,
            P,
            kernel_size=KERNEL_SIZE,
            match_input_rms=match_input_rms,
        )
    reader.initialize_identity_()
    stage.pole_input_projection = reader


def _install_unit_row_pg(model: ComplexScanBackbone) -> None:
    for name in ("stage1", "stage2", "stage3"):
        mixer = getattr(model, name).quadrant_path_mode_combiner
        if not isinstance(mixer, PhaseGatedModeResidualPathCollapse):
            message = f"{name} is missing the Phase-Gated mode block"
            raise TypeError(message)
        mixer.mode.enable_unit_row_contract_(
            residual_scale_max=PG_RESIDUAL_SCALE_MAX,
        )


def _install_mode_gain_pg(model: ComplexScanBackbone) -> None:
    for name in ("stage1", "stage2", "stage3"):
        mixer = getattr(model, name).quadrant_path_mode_combiner
        if not isinstance(mixer, PhaseGatedModeResidualPathCollapse):
            message = f"{name} is missing the Phase-Gated mode block"
            raise TypeError(message)
        mixer.mode.enable_direction_output_gain_contract_(
            output_gain_max=PG_OUTPUT_GAIN_MAX,
        )


def _install_path_only_stage(model: ComplexScanBackbone, name: str) -> None:
    stage = getattr(model, name)
    baseline = stage.quadrant_path_mode_combiner
    if not isinstance(baseline, PhaseGatedModeResidualPathCollapse):
        message = f"{name} is missing the Phase-Gated mode block"
        raise TypeError(message)
    # The ablation must not perturb the shared data-order or augmentation RNG.
    with torch.random.fork_rng(devices=[]):
        replacement = PathOnlyCollapse(P, path_hidden=baseline.path_input.output_paths)
    replacement.copy_path_from(baseline)
    stage.quadrant_path_mode_combiner = replacement


def _install_path_pg_stage(
    model: ComplexScanBackbone,
    name: str,
    *,
    apply_cartesian_silu: bool,
) -> None:
    stage = getattr(model, name)
    baseline = stage.quadrant_path_mode_combiner
    if not isinstance(baseline, PhaseGatedModeResidualPathCollapse):
        message = f"{name} is missing the Phase-Gated mode block"
        raise TypeError(message)
    # Copy the exact PG and GWL parameters so this variant changes only operation order.
    with torch.random.fork_rng(devices=[]):
        replacement = PathPhaseGatedCollapse(
            P,
            mode_hidden=baseline.mode.hidden_modes,
            path_hidden=baseline.path_input.output_paths,
            apply_cartesian_silu=apply_cartesian_silu,
        )
    replacement.copy_from(baseline)
    stage.quadrant_path_mode_combiner = replacement


def _install_average_reexcitation(model: ComplexScanBackbone) -> None:
    # Draw distinct parameters for all three stages without changing the shared
    # experiment/data RNG state outside model construction.
    with torch.random.fork_rng(devices=[]):
        for name in ("stage1", "stage2", "stage3"):
            stage = getattr(model, name)
            stage.augmented = AveragePoolPhaseGatedRefinementTransition(
                P,
                refine_hidden=control.MODE_HIDDEN,
            )


def _install_pure_gate(
    model: ComplexScanBackbone,
    *,
    use_average_carry: bool,
) -> None:
    transition_type = (
        AveragePoolMagnitudeGateTransition
        if use_average_carry
        else MemoryOnlyMagnitudeGateTransition
    )
    for name in ("stage1", "stage2", "stage3"):
        getattr(model, name).augmented = transition_type(P)


def _install_stem_interface(
    model: ComplexScanBackbone,
    *,
    no_stem_rmsnorm: bool,
) -> None:
    if no_stem_rmsnorm:
        model.input_norm = nn.Identity()


def _assert_path_pg_mixer(
    name: str,
    mixer: object,
    *,
    apply_cartesian_silu: bool,
) -> None:
    if not isinstance(mixer, PathPhaseGatedCollapse):
        message = f"{name} did not move its PG block between the path projections"
        raise TypeError(message)
    if (
        mixer.path_input.input_paths != 4
        or mixer.path_input.output_paths != control.PATH_HIDDEN
        or mixer.mode.modes != P
        or mixer.mode.hidden_modes != control.MODE_HIDDEN
        or mixer.path_output.input_paths != control.PATH_HIDDEN
        or mixer.path_output.output_paths != 1
        or mixer.apply_cartesian_silu is not apply_cartesian_silu
    ):
        message = f"{name} changed the shared Path-PG contract"
        raise RuntimeError(message)


def _assert_average_reexcitation(model: ComplexScanBackbone) -> None:
    for name in ("stage1", "stage2", "stage3"):
        transition = getattr(model, name).augmented
        if not isinstance(
            transition,
            AveragePoolPhaseGatedRefinementTransition,
        ):
            message = f"{name} is missing the fixed-average re-excitation block"
            raise TypeError(message)
        refine = transition.refine
        if (
            transition.carry_input_modes != 4 * P
            or refine.modes != P
            or refine.hidden_modes != control.MODE_HIDDEN
            or refine.learnable_residual_scale
            or refine.gamma is not None
            or float(refine.effective_residual_scale()) != 1.0
        ):
            message = f"{name} changed the unit-scale re-excitation contract"
            raise RuntimeError(message)


def _assert_pure_gate(
    model: ComplexScanBackbone,
    *,
    use_average_carry: bool,
) -> None:
    expected_type = (
        AveragePoolMagnitudeGateTransition
        if use_average_carry
        else MemoryOnlyMagnitudeGateTransition
    )
    for name in ("stage1", "stage2", "stage3"):
        transition = getattr(model, name).augmented
        if type(transition) is not expected_type:
            message = f"{name} is missing its declared pure magnitude gate"
            raise TypeError(message)
        if (
            transition.norm.modes != P
            or transition.gate.modes != P
            or (use_average_carry and transition.carry_input_modes != 4 * P)
            or (not use_average_carry and hasattr(transition, "carry_input_modes"))
        ):
            message = f"{name} changed the pure magnitude-gate contract"
            raise RuntimeError(message)


def _assert_stem_interface(
    model: ComplexScanBackbone,
    *,
    no_stem_rmsnorm: bool,
) -> None:
    if no_stem_rmsnorm:
        if type(model.input_norm) is not nn.Identity:
            message = "NoStemRMS variant retained its pre-analysis RMSNorm"
            raise TypeError(message)
    elif not isinstance(model.input_norm, nn.RMSNorm):
        message = "local-reader control lost its pre-analysis RMSNorm"
        raise TypeError(message)


def _assert_local_readers(
    model: ComplexScanBackbone,
    *,
    match_input_rms: bool,
) -> None:
    for name in ("stage1", "stage2", "stage3", "terminal"):
        reader = getattr(model, name).pole_input_projection
        if (
            not isinstance(reader, PackedComplexConv2dReader)
            or reader.input_modes != P
            or reader.output_modes != P
            or reader.kernel_size != KERNEL_SIZE
            or reader.match_input_rms is not match_input_rms
        ):
            message = f"{name} changed the local strict-complex reader contract"
            raise RuntimeError(message)


def _assert_model(
    model: ComplexScanBackbone,
    *,
    match_input_rms: bool,
    unit_row_pg: bool,
    mode_gain_pg: bool,
    no_pg_stages: frozenset[str],
    path_pg_cartesian_activation: bool | None,
    average_reexcitation: bool,
    pure_gate_carry: bool | None,
    no_stem_rmsnorm: bool,
) -> None:
    if not no_pg_stages and path_pg_cartesian_activation is None:
        control._assert_model(model)
    _assert_local_readers(model, match_input_rms=match_input_rms)
    if average_reexcitation:
        _assert_average_reexcitation(model)
    if pure_gate_carry is not None:
        _assert_pure_gate(model, use_average_carry=pure_gate_carry)
    _assert_stem_interface(model, no_stem_rmsnorm=no_stem_rmsnorm)
    for name in ("stage1", "stage2", "stage3"):
        mixer = getattr(model, name).quadrant_path_mode_combiner
        if path_pg_cartesian_activation is not None:
            _assert_path_pg_mixer(
                name,
                mixer,
                apply_cartesian_silu=path_pg_cartesian_activation,
            )
            continue
        if name in no_pg_stages:
            if not isinstance(mixer, PathOnlyCollapse):
                message = f"{name} did not remove only its Phase-Gated mode block"
                raise TypeError(message)
            continue
        if not isinstance(mixer, PhaseGatedModeResidualPathCollapse):
            message = f"{name} is missing the Phase-Gated mode block"
            raise TypeError(message)
        mode = mixer.mode
        if (
            mode.unit_row_projections is not unit_row_pg
            or mode.projected_direction_rows is not mode_gain_pg
            or mode.residual_scale_max != (PG_RESIDUAL_SCALE_MAX if unit_row_pg else None)
            or mode.output_gain_max != (PG_OUTPUT_GAIN_MAX if mode_gain_pg else None)
        ):
            message = f"{name} changed the Phase-Gated scale contract"
            raise RuntimeError(message)


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    if variant not in VARIANTS:
        message = f"unsupported PGv2-H96 local-reader variant: {variant}"
        raise ValueError(message)
    match_input_rms = variant in RMS_MATCH_VARIANTS
    unit_row_pg = variant == UNIT_ROW_VARIANT
    mode_gain_pg = variant == MODE_GAIN_VARIANT
    no_pg_stages = (
        frozenset(("stage3",))
        if variant == NO_PG_STAGE3_VARIANT
        else frozenset(("stage1", "stage2", "stage3"))
        if variant == NO_PG_ALL_VARIANT
        else frozenset()
    )
    path_pg_cartesian_activation = (
        variant == PATH_PG_VARIANT if variant in PATH_PG_VARIANTS else None
    )
    average_reexcitation = variant == AVG_REEXCITE_VARIANT
    pure_gate_carry = (
        variant != MEMORY_ONLY_PURE_GATE_VARIANT if variant in PURE_GATE_VARIANTS else None
    )
    no_stem_rmsnorm = variant == AVG_PURE_GATE_NO_STEM_RMS_VARIANT
    model = control._build(control.VARIANT, config)
    if unit_row_pg:
        _install_unit_row_pg(model)
    if mode_gain_pg:
        _install_mode_gain_pg(model)
    for name in ("stage1", "stage2", "stage3"):
        if path_pg_cartesian_activation is not None:
            _install_path_pg_stage(
                model,
                name,
                apply_cartesian_silu=path_pg_cartesian_activation,
            )
        elif name in no_pg_stages:
            _install_path_only_stage(model, name)
    if average_reexcitation:
        _install_average_reexcitation(model)
    if pure_gate_carry is not None:
        _install_pure_gate(model, use_average_carry=pure_gate_carry)
    _install_stem_interface(model, no_stem_rmsnorm=no_stem_rmsnorm)
    for name in ("stage1", "stage2", "stage3", "terminal"):
        _install_local_reader(
            getattr(model, name),
            match_input_rms=match_input_rms,
        )
    _configure_ramp()
    _assert_model(
        model,
        match_input_rms=match_input_rms,
        unit_row_pg=unit_row_pg,
        mode_gain_pg=mode_gain_pg,
        no_pg_stages=no_pg_stages,
        path_pg_cartesian_activation=path_pg_cartesian_activation,
        average_reexcitation=average_reexcitation,
        pure_gate_carry=pure_gate_carry,
        no_stem_rmsnorm=no_stem_rmsnorm,
    )
    return model


def _variant_config(variant: str) -> dict[str, Any]:
    match_input_rms = variant in RMS_MATCH_VARIANTS
    unit_row_pg = variant == UNIT_ROW_VARIANT
    mode_gain_pg = variant == MODE_GAIN_VARIANT
    payload = deepcopy(control._variant_config())
    payload["backbone"]["name"] = f"A2D-{variant}"
    payload["backbone"]["pole_input"] = {
        "operator": "one unit-row-energy full channel-mixing strict-complex K3 convolution",
        "shape": f"{P}-to-{P} before every Stage1-3 and terminal scan",
        "initialization": "exact point identity; learned spatial residual starts at zero",
        "replaces": "the strict-complex 1x1 B_s reader; no separate pointwise reader remains",
        "variance_contract": (
            "sum_{input,dy,dx} abs(W[output,input,dy,dx])^2 == 1 for every output pole; "
            "this removes learned drive-scale drift but does not assume away data spatial "
            "covariance"
        ),
        "scope": "scan branch only; local S2D carry remains unchanged",
        "activation_gain": (
            "per-token shared real RMS(U)=RMS(E) matching"
            if match_input_rms
            else "unconstrained data-dependent activation gain"
        ),
    }
    if unit_row_pg:
        mode_gate = payload["backbone"]["stage_transition"]["mode_gate"]
        mode_gate["projection_scale_contract"] = (
            "strict-complex unit energy independently on every output row"
        )
        mode_gate["residual_scale"] = {
            "parameter": "theta",
            "effective": "gamma_max * tanh(theta)",
            "initial": 0.01,
            "maximum_absolute": PG_RESIDUAL_SCALE_MAX,
        }
    if mode_gain_pg:
        mode_gate = payload["backbone"]["stage_transition"]["mode_gate"]
        mode_gate["projection_scale_contract"] = (
            "optimizer-projected unit-sphere strict-complex input/output rows"
        )
        mode_gate["residual_scale"] = {
            "parameter": "one theta per output mode; no global gamma",
            "effective": "beta_r = beta_max * tanh(theta_r)",
            "initial": 0.01,
            "maximum_absolute": PG_OUTPUT_GAIN_MAX,
            "application": "folded into output projection rows before fused dispatch",
        }
    if variant in (NO_PG_STAGE3_VARIANT, NO_PG_ALL_VARIANT):
        payload["backbone"]["stage_transition"]["mode_residual"] = (
            "identity at Stage 3; unchanged PGv2-H96 at Stages 1-2"
            if variant == NO_PG_STAGE3_VARIANT
            else "identity at Stages 1-3"
        )
    transition = payload["backbone"]["stage_transition"]
    if variant in PATH_PG_VARIANTS:
        transition["mode_residual"] = "removed before path expansion at Stages 1-3"
        transition["path_processing"] = {
            "input_projection": f"GroupedWidelyLinear 4-to-{control.PATH_HIDDEN}",
            "shared_mode_residual": (
                f"one PhaseGatedComplexFFNv2-{P}-{control.MODE_HIDDEN}-{P} shared "
                f"over all {control.PATH_HIDDEN} hidden paths"
            ),
            "activation": (
                "Cartesian SiLU after the shared mode residual"
                if variant == PATH_PG_VARIANT
                else "identity; no activation between shared mode residual and collapse"
            ),
            "output_projection": f"GroupedWidelyLinear {control.PATH_HIDDEN}-to-1",
        }
    if variant == AVG_REEXCITE_VARIANT:
        transition["local_shortcut"] = {
            "operator": "fixed complex AvgPool2d",
            "kernel_size": 2,
            "stride": 2,
            "learned_parameters": 0,
            "source": "the original excitation before the K3 scan reader",
        }
        transition["post_fusion"] = {
            "merge": "collapsed D4 memory plus fixed local average",
            "operator": f"PhaseGatedComplexFFNv2-{P}-{control.MODE_HIDDEN}-{P}",
            "residual_scale": "fixed exactly at one; no gamma parameter",
        }
    if variant in (AVG_PURE_GATE_VARIANT, AVG_PURE_GATE_NO_STEM_RMS_VARIANT):
        transition["local_shortcut"] = {
            "operator": "fixed complex AvgPool2d",
            "kernel_size": 2,
            "stride": 2,
            "learned_parameters": 0,
            "source": "the original excitation before the K3 scan reader",
        }
        transition["post_fusion"] = {
            "merge": "X = collapsed D4 memory plus fixed local average",
            "operator": "CRMSNorm then mean-one magnitude gate",
            "equation": "E_next = X * g = X + X * (g - 1)",
            "projections": "none",
            "residual_scale": "none",
        }
    if variant == MEMORY_ONLY_PURE_GATE_VARIANT:
        transition["local_shortcut"] = {
            "operator": "none",
            "source": "S2D carry removed; the scan-memory branch is the sole transition input",
        }
        transition["post_fusion"] = {
            "merge": "X = collapsed D4 memory; no local residual is added",
            "operator": "CRMSNorm then mean-one magnitude gate",
            "equation": "E_next = M * g = M + M * (g - 1)",
            "projections": "none",
            "residual_scale": "none",
        }
    if variant == AVG_PURE_GATE_NO_STEM_RMS_VARIANT:
        payload["backbone"]["stem"]["interface_norm"] = "identity; RMSNorm192 removed"
    return payload


def _selected_variants(args: Namespace) -> tuple[str, ...]:
    requested = tuple(getattr(args, "variants", ()))
    return requested or VARIANTS


def _contract(args: Namespace) -> dict[str, Any]:
    payload = control._contract(args)
    selected = _selected_variants(args)
    ramp = control.control.control.stemres.uniform.base
    config = ramp.PoleModelConfig(output_dim=100, stem_strides=(2, 2))
    models = {variant: _build(variant, config) for variant in selected}
    payload["schema"] = "lnet.a2d.pgv2_h96.local_reader_k3.imagenet100.v9"
    payload["evidence_status"] = "controlled PGv2-H96 local-reader gain experiment"
    payload["variants"] = list(selected)
    payload["seeds"] = list(SEEDS)
    payload["variant_configs"] = {variant: _variant_config(variant) for variant in selected}
    if selected == (MODE_GAIN_VARIANT,):
        payload["recipe"]["phase_gated_optimizer"] = {
            "direction_learning_rate": 3.0e-3,
            "direction_weight_decay": 0.0,
            "direction_retraction": (
                "complex output rows projected to unit energy after every optimizer step"
            ),
            "alpha_beta_crmsnorm_learning_rate": 3.0e-3,
            "alpha_beta_crmsnorm_weight_decay": 0.0,
            "output_strength": "signed beta_r = 0.5 * tanh(theta_r); no global gamma",
        }
    payload["parameter_counts"] = {
        variant: sum(parameter.numel() for parameter in model.parameters())
        for variant, model in models.items()
    }
    full_reader_parameters = 4 * 2 * P * P * KERNEL_SIZE**2
    point_reader_parameters = 4 * 2 * P * P
    original_parameters = sum(
        parameter.numel() for parameter in control._build(control.VARIANT, config).parameters()
    )
    payload["parameter_comparison"] = {
        "original_pgv2_h96_all3e3": original_parameters,
        "strict_1x1_vector_input_control": original_parameters + point_reader_parameters,
        "full_k3_candidate_with_all_mode_pg": original_parameters + full_reader_parameters,
        "selected_candidates": payload["parameter_counts"],
        "delta_vs_original": full_reader_parameters,
        "delta_vs_strict_1x1_control": full_reader_parameters - point_reader_parameters,
    }
    architecture_prefix = "Exact PGv2-H96-All3e-3 except that each scan branch reads a local K3"
    architecture_prefix += (
        " neighborhood with one unit-row-energy strict-complex full P-to-P convolution,"
    )
    architecture_prefix += (
        " replacing rather than stacking with the conceptual strict 1x1 B_s reader."
    )
    architecture_prefix += " The reader begins as the exact identity"
    unit_row_prefix = "; every PG strict-complex projection has unit output-row energy and "
    unit_row_note = f"{unit_row_prefix}the sole branch-strength gamma is bounded to (-0.5, 0.5)"
    mode_gain_note = (
        "; PG input/output rows are optimizer-projected directions, and signed "
        "bounded per-output-mode gains replace global gamma"
    )
    payload["architecture"] = {
        variant: (
            architecture_prefix
            + (
                " and matches each output token's shared complex RMS to its input"
                if variant in RMS_MATCH_VARIANTS
                else ""
            )
            + (unit_row_note if variant == UNIT_ROW_VARIANT else "")
            + (mode_gain_note if variant == MODE_GAIN_VARIANT else "")
            + (
                "; Stage 3 PG mode residual is removed while its Path GWL is preserved"
                if variant == NO_PG_STAGE3_VARIANT
                else "; Stage 1-3 PG mode residuals are removed while every Path GWL is preserved"
                if variant == NO_PG_ALL_VARIANT
                else (
                    "; at Stages 1-3 the shared H96 PG is moved between the 4-to-8 "
                    "and 8-to-1 Path GWL projections"
                )
                if variant in PATH_PG_VARIANTS
                else ""
            )
            + (
                "; Cartesian SiLU is omitted"
                if variant in (PATH_PG_NO_SILU_VARIANT, AVG_REEXCITE_VARIANT)
                or variant in PURE_GATE_VARIANTS
                else ""
            )
            + (
                AVG_REEXCITE_ARCHITECTURE_NOTE
                if variant == AVG_REEXCITE_VARIANT
                else MEMORY_ONLY_PURE_GATE_ARCHITECTURE_NOTE
                if variant == MEMORY_ONLY_PURE_GATE_VARIANT
                else AVG_PURE_GATE_ARCHITECTURE_NOTE
                if variant in (AVG_PURE_GATE_VARIANT, AVG_PURE_GATE_NO_STEM_RMS_VARIANT)
                else "; path collapse, carry, Q1536, head, and recipe are unchanged."
            )
            + (
                NO_STEM_RMS_ARCHITECTURE_NOTE
                if variant == AVG_PURE_GATE_NO_STEM_RMS_VARIANT
                else ""
            )
        )
        for variant in selected
    }
    digest = ramp.heads.harness._digest
    payload["source_sha256"]["complex_scan_reader"] = digest(
        Path("src/lnet/pac_complex_scan_reader.py")
    )
    payload["source_sha256"]["phase_gated_cffn"] = digest(Path("src/lnet/pac_phase_gated_cffn.py"))
    payload["source_sha256"]["mean_one_magnitude_gate"] = digest(
        Path("src/lnet/pac_mean_one_magnitude_gate.py")
    )
    payload["source_sha256"]["phase_gated_transition"] = digest(
        Path("src/lnet/pac_phase_gated_transition.py")
    )
    payload["source_sha256"]["pgv2_h96_local_reader_runner"] = digest(Path(__file__))
    return payload


def _gate_metrics(
    prefix: str,
    module: PhaseGatedComplexFFN | MeanOneMagnitudeGate,
) -> dict[str, float]:
    values = module.diagnostic_metrics()
    values.update(module.gradient_metrics())
    return {f"{prefix}{name}": value for name, value in values.items()}


def _wandb_model_metrics(model: torch.nn.Module) -> dict[str, float]:
    metrics = control.control._wandb_model_metrics(model)
    for index, name in enumerate(("stage1", "stage2", "stage3"), start=1):
        mixer = getattr(model, name).quadrant_path_mode_combiner
        if isinstance(mixer, PathPhaseGatedCollapse):
            metrics.update(
                _gate_metrics(
                    f"phase_gated/stage{index}/path_shared_",
                    mixer.mode,
                )
            )
        transition = getattr(model, name).augmented
        if isinstance(transition, AveragePoolPhaseGatedRefinementTransition):
            metrics.update(
                _gate_metrics(
                    f"phase_gated/stage{index}/reexcite_",
                    transition.refine,
                )
            )
        if isinstance(transition, PureMagnitudeGateTransition):
            metrics.update(
                _gate_metrics(
                    f"magnitude_gate/stage{index}/",
                    transition.gate,
                )
            )
    return metrics


def main() -> None:
    _configure_ramp()
    ramp = control.control.control.stemres.uniform.base
    source = ramp.canonical8.fair_init.backbone.deep4.baseline.baseline
    residuals = ramp.backbone.a2d_base.residuals
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
            build_optimizer=residuals.optimizer_source._build_optimizer,
            prepare_model=source._prepare_model,
            train_epoch=source.structured._train_epoch,
            evaluate=source.heads._evaluate,
            wandb_model_metrics=_wandb_model_metrics,
            summarize=source.heads._summarize,
        )
    )


if __name__ == "__main__":
    main()
