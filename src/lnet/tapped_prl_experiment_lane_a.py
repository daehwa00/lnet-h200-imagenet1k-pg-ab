from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from lnet.hybrid import BRANCH_ORDER, BranchName, FusionVariant, HybridModalPRLBlock

if TYPE_CHECKING:
    from lnet.tapped_prl import TapParameterization

ComparisonType = Literal[
    "branch_ablation",
    "k_sweep",
    "m_sweep",
    "tap_parameterization",
    "gate_variant",
]
IsolationLabel = Literal["isolated", "joint", "not_applicable"]


@dataclass(frozen=True, slots=True)
class LaneAExperimentConfig:
    raw_input_dim: int = 2
    output_dim: int = 2
    model_dim: int = 8
    modes: int = 4
    fir_kernel_size: int = 13
    prl_tap_kernel_size: int = 13
    prl_tap_kernel_sizes: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    mode_values: tuple[int, ...] = (1, 2, 3, 4, 6, 8)
    branch_sets: tuple[tuple[BranchName, ...], ...] = (
        ("prl",),
        ("fir",),
        ("mlp",),
        ("prl", "fir"),
        ("prl", "mlp"),
        ("fir", "mlp"),
        BRANCH_ORDER,
    )
    tap_parameterizations: tuple[TapParameterization, ...] = (
        "shared_scalar",
        "tap_specific_reader",
        "low_rank_reader",
        "normalized_taps",
    )


@dataclass(frozen=True, slots=True)
class LaneAModelSpec:
    comparison_type: ComparisonType
    comparison_group: str
    gate_selection_scope: str
    active_branches: tuple[BranchName, ...]
    model_label: str
    gate_variant_label: str
    prl_tap_kernel_size: int
    fir_kernel_size: int
    mode_count: int
    tap_parameterization: TapParameterization
    fusion_variant: FusionVariant
    fusion_temperature: float
    isolated_vs_joint: IsolationLabel


def branch_ablation_specs(
    config: LaneAExperimentConfig,
    *,
    teacher_label: str,
) -> tuple[LaneAModelSpec, ...]:
    return tuple(
        _lane_a_spec(
            comparison_type="branch_ablation",
            teacher_label=teacher_label,
            scope="components",
            active_branches=branches,
            prl_tap_kernel_size=config.prl_tap_kernel_size,
            fir_kernel_size=config.fir_kernel_size,
            mode_count=config.modes,
            tap_parameterization="shared_scalar",
            fusion_variant="softmax",
            fusion_temperature=1.0,
            isolated_vs_joint="not_applicable",
        )
        for branches in config.branch_sets
    )


def k_sweep_specs(
    config: LaneAExperimentConfig,
    *,
    teacher_label: str,
) -> tuple[LaneAModelSpec, ...]:
    return tuple(
        _lane_a_spec(
            comparison_type="k_sweep",
            teacher_label=teacher_label,
            scope="k_values",
            active_branches=("prl",),
            prl_tap_kernel_size=tap_kernel_size,
            fir_kernel_size=config.fir_kernel_size,
            mode_count=config.modes,
            tap_parameterization="shared_scalar",
            fusion_variant="softmax",
            fusion_temperature=1.0,
            isolated_vs_joint="isolated",
        )
        for tap_kernel_size in config.prl_tap_kernel_sizes
    )


def fusion_variant_specs(
    config: LaneAExperimentConfig,
    *,
    teacher_label: str,
) -> tuple[LaneAModelSpec, ...]:
    specs = [
        _lane_a_spec(
            comparison_type="gate_variant",
            teacher_label=teacher_label,
            scope="gate_values",
            active_branches=BRANCH_ORDER,
            prl_tap_kernel_size=config.prl_tap_kernel_size,
            fir_kernel_size=config.fir_kernel_size,
            mode_count=config.modes,
            tap_parameterization="shared_scalar",
            fusion_variant="no_gate_sum",
            fusion_temperature=1.0,
            isolated_vs_joint="not_applicable",
        ),
        _lane_a_spec(
            comparison_type="gate_variant",
            teacher_label=teacher_label,
            scope="gate_values",
            active_branches=BRANCH_ORDER,
            prl_tap_kernel_size=config.prl_tap_kernel_size,
            fir_kernel_size=config.fir_kernel_size,
            mode_count=config.modes,
            tap_parameterization="shared_scalar",
            fusion_variant="fixed_equal",
            fusion_temperature=1.0,
            isolated_vs_joint="not_applicable",
        ),
        _lane_a_spec(
            comparison_type="gate_variant",
            teacher_label=teacher_label,
            scope="gate_values",
            active_branches=BRANCH_ORDER,
            prl_tap_kernel_size=config.prl_tap_kernel_size,
            fir_kernel_size=config.fir_kernel_size,
            mode_count=config.modes,
            tap_parameterization="shared_scalar",
            fusion_variant="fixed_learned_scalar",
            fusion_temperature=1.0,
            isolated_vs_joint="not_applicable",
        ),
        _lane_a_spec(
            comparison_type="gate_variant",
            teacher_label=teacher_label,
            scope="gate_values",
            active_branches=BRANCH_ORDER,
            prl_tap_kernel_size=config.prl_tap_kernel_size,
            fir_kernel_size=config.fir_kernel_size,
            mode_count=config.modes,
            tap_parameterization="shared_scalar",
            fusion_variant="softmax",
            fusion_temperature=1.0,
            isolated_vs_joint="not_applicable",
        ),
        _lane_a_spec(
            comparison_type="gate_variant",
            teacher_label=teacher_label,
            scope="gate_values",
            active_branches=BRANCH_ORDER,
            prl_tap_kernel_size=config.prl_tap_kernel_size,
            fir_kernel_size=config.fir_kernel_size,
            mode_count=config.modes,
            tap_parameterization="shared_scalar",
            fusion_variant="temperature_softmax",
            fusion_temperature=0.5,
            isolated_vs_joint="not_applicable",
        ),
        _lane_a_spec(
            comparison_type="gate_variant",
            teacher_label=teacher_label,
            scope="gate_values",
            active_branches=BRANCH_ORDER,
            prl_tap_kernel_size=config.prl_tap_kernel_size,
            fir_kernel_size=config.fir_kernel_size,
            mode_count=config.modes,
            tap_parameterization="shared_scalar",
            fusion_variant="temperature_softmax",
            fusion_temperature=2.0,
            isolated_vs_joint="not_applicable",
        ),
    ]
    return tuple(specs)


def mode_sweep_specs(
    config: LaneAExperimentConfig,
    *,
    teacher_label: str,
) -> tuple[LaneAModelSpec, ...]:
    return tuple(
        _lane_a_spec(
            comparison_type="m_sweep",
            teacher_label=teacher_label,
            scope="mode_values",
            active_branches=("prl",),
            prl_tap_kernel_size=config.prl_tap_kernel_size,
            fir_kernel_size=config.fir_kernel_size,
            mode_count=mode_count,
            tap_parameterization="shared_scalar",
            fusion_variant="softmax",
            fusion_temperature=1.0,
            isolated_vs_joint="not_applicable",
        )
        for mode_count in config.mode_values
    )


def tap_parameterization_specs(
    config: LaneAExperimentConfig,
    *,
    teacher_label: str,
) -> tuple[LaneAModelSpec, ...]:
    return tuple(
        _lane_a_spec(
            comparison_type="tap_parameterization",
            teacher_label=teacher_label,
            scope="tap_parameterizations",
            active_branches=("prl",),
            prl_tap_kernel_size=config.prl_tap_kernel_size,
            fir_kernel_size=config.fir_kernel_size,
            mode_count=config.modes,
            tap_parameterization=tap_parameterization,
            fusion_variant="softmax",
            fusion_temperature=1.0,
            isolated_vs_joint="not_applicable",
        )
        for tap_parameterization in config.tap_parameterizations
    )


def instantiate_lane_a_model(
    spec: LaneAModelSpec,
    config: LaneAExperimentConfig,
) -> HybridModalPRLBlock:
    return HybridModalPRLBlock(
        raw_input_dim=config.raw_input_dim,
        model_dim=config.model_dim,
        output_dim=config.output_dim,
        modes=spec.mode_count,
        fir_kernel_size=spec.fir_kernel_size,
        prl_tap_kernel_size=spec.prl_tap_kernel_size,
        active_branches=spec.active_branches,
        tap_parameterization=spec.tap_parameterization,
        fusion_variant=spec.fusion_variant,
        fusion_temperature=spec.fusion_temperature,
    )


def _lane_a_spec(
    *,
    comparison_type: ComparisonType,
    teacher_label: str,
    scope: str,
    active_branches: tuple[BranchName, ...],
    prl_tap_kernel_size: int,
    fir_kernel_size: int,
    mode_count: int,
    tap_parameterization: TapParameterization,
    fusion_variant: FusionVariant,
    fusion_temperature: float,
    isolated_vs_joint: IsolationLabel,
) -> LaneAModelSpec:
    return LaneAModelSpec(
        comparison_type=comparison_type,
        comparison_group=f"stage1/{comparison_type}/{teacher_label}/{scope}",
        gate_selection_scope=f"gatectx/{teacher_label}/stage1_lane_a",
        active_branches=active_branches,
        model_label=_model_label(
            active_branches,
            prl_tap_kernel_size,
            fusion_variant,
            fusion_temperature,
        ),
        gate_variant_label=_gate_variant_label(fusion_variant, fusion_temperature),
        prl_tap_kernel_size=prl_tap_kernel_size,
        fir_kernel_size=fir_kernel_size,
        mode_count=mode_count,
        tap_parameterization=tap_parameterization,
        fusion_variant=fusion_variant,
        fusion_temperature=fusion_temperature,
        isolated_vs_joint=isolated_vs_joint,
    )


def _model_label(
    active_branches: tuple[BranchName, ...],
    prl_tap_kernel_size: int,
    fusion_variant: FusionVariant,
    fusion_temperature: float,
) -> str:
    if active_branches == ("prl",):
        return "instantaneous_prl" if prl_tap_kernel_size == 1 else "tapped_prl"
    branch_label = "_".join(active_branches)
    return f"hybrid_{branch_label}_{_gate_variant_label(fusion_variant, fusion_temperature)}"


def _gate_variant_label(fusion_variant: FusionVariant, fusion_temperature: float) -> str:
    if fusion_variant != "temperature_softmax":
        return fusion_variant
    return (
        "temperature_softmax_tau_0_5"
        if fusion_temperature == 0.5
        else "temperature_softmax_tau_2_0"
    )
