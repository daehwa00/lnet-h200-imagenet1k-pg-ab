"""Capacity-aware model construction for uniform R2K3 stages."""

from __future__ import annotations

# pyright: reportArgumentType=false, reportExplicitAny=false
# pyright: reportImplicitRelativeImport=false, reportPrivateUsage=false
# pyright: reportAttributeAccessIssue=false, reportPrivateLocalImportUsage=false
# pyright: reportUnusedFunction=false
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import a2d_r2k3_runtime as runtime
import a2d_r2k3_seeded_builder as backbone
import torch
from torch import Tensor, nn
from torch.nn.utils import parametrize
from torch.nn.utils.parametrizations import orthogonal

from lnet.a2d_q_heads import A2DAffineQClassifier
from lnet.complex_scan_stage import ComplexScanStage
from lnet.image_layers import StandardizedAffineModalHead
from lnet.pac_factorized_complex_scan_reader import FactorizedComplexConv2dReader
from lnet.pac_phase_gated_transition import (
    PathOnlyCollapse,
    PhaseGatedModeResidualPathCollapse,
)
from lnet.pac_pole_excitation_transition import (
    PoleExcitationS2DPostFusionTransition,
)
from lnet.pac_pole_initialization import (
    DAMPING_QUANTILE_POWER,
    NONTERMINAL_GEOMETRIC_DAMPING_RANGE,
    R2K3_DAMPING_OFFSETS,
    R2K3_DAMPING_SCALES,
    R2K3_FREQUENCY_SCALES,
    R2K3_LOG_ANISOTROPY,
    R2K3_MAXIMUM_PHASES,
    R2K3_ORIENTATIONS,
    TERMINAL_DAMPING_MAX,
    TERMINAL_GEOMETRIC_DAMPING_RANGE,
    install_calibrated_initialization,
    install_decoupled_initialization,
    install_r2k3_pole_initialization,
    install_short_damping,
)

if TYPE_CHECKING:
    from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig


K128_P128 = "R2K3-K128-P128x4-RawQ-Orth-NoPG"
STAGE_NAMES = runtime.STAGE_NAMES
READER_RANK = 2
KERNEL_SIZE = 3
PATH_HIDDEN = 8
DEFAULT_POST_HIDDEN_RATIO = 1.5


@dataclass(frozen=True, slots=True)
class CapacitySpec:
    excitation_modes: tuple[int, int, int, int]
    pole_modes: tuple[int, int, int, int]
    post_hidden_ratio: float = DEFAULT_POST_HIDDEN_RATIO

    @property
    def descriptor_dim(self) -> int:
        return 4 * sum(self.pole_modes)

    @property
    def q4_dim(self) -> int:
        return 4 * self.pole_modes[-1]

    @property
    def post_hidden_modes(self) -> tuple[int, ...]:
        return tuple(
            round(modes * self.post_hidden_ratio)
            for modes in self.excitation_modes
        )


SPECS = {
    K128_P128: CapacitySpec(
        excitation_modes=(128, 128, 128, 128),
        pole_modes=(128, 128, 128, 128),
        post_hidden_ratio=2.0,
    ),
}


class CapacityQ4OnlyAffineClassifier(A2DAffineQClassifier):
    """Select terminal raw-Q coordinates for a dynamic pole schedule."""

    def __init__(self, descriptor_dim: int, q4_dim: int, output_dim: int) -> None:
        super().__init__(
            q4_dim,
            output_dim,
            main="affine",
            affine=StandardizedAffineModalHead(q4_dim, output_dim),
            fusion=None,
            lrq=None,
            beta_lrq=None,
            affine_auxiliary_weight=0.0,
        )
        self.full_descriptor_dim = descriptor_dim
        self.q4_dim = q4_dim

    def select_q4(self, descriptor: Tensor) -> Tensor:
        if descriptor.shape[-1] != self.full_descriptor_dim:
            message = "capacity Q4 head received an incompatible full descriptor"
            raise ValueError(message)
        return descriptor[..., -self.q4_dim :]

    def forward(self, descriptor: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        return super().forward(self.select_q4(descriptor))


def resize_terminal_poles_(
    model: ComplexScanBackbone,
    *,
    target_poles: int,
) -> None:
    """Rebuild the terminal reader, pole bank, and Q4 head at ``target_poles``.

    Every target pole participates in one joint initialization.  No source reader,
    pole, normalization, or classifier coordinates are copied or privileged.
    """
    source = model.terminal
    source_poles = source.modes
    classifier = model.classifier
    source_q4_dim = getattr(classifier, "q4_dim", None)
    if not isinstance(source_q4_dim, int):
        raise TypeError("terminal resize requires a capacity Q4 affine classifier")
    if source_q4_dim % source_poles:
        raise RuntimeError("terminal Q4 width is not divisible by the source pole count")
    directions = source_q4_dim // source_poles
    prefix_dim = model.descriptor_dim - source_q4_dim
    if target_poles <= 0 or target_poles % R2K3_ORIENTATIONS:
        raise ValueError("terminal pole width must contain complete orientation groups")
    descriptor_dim = prefix_dim + directions * target_poles
    q4_dim = directions * target_poles
    if target_poles == source_poles:
        return

    source_reader = source.pole_input_projection
    if type(source_reader) is not FactorizedComplexConv2dReader:
        raise TypeError("terminal resize requires a factorized complex reader")
    with torch.random.fork_rng(devices=[]):
        terminal = ComplexScanStage(
            target_poles,
            maximum_phase=R2K3_MAXIMUM_PHASES[-1],
            output_modes=None,
            scan_memory_policy=source.scan_memory_policy,
            damping_min=source.damping_min,
            damping_max=source.damping_max,
        )
        terminal.product_gain_normalization = source.product_gain_normalization
        terminal.input_modes = source.input_modes
        reader = FactorizedComplexConv2dReader(
            source_reader.input_modes,
            target_poles,
            rank=source_reader.rank,
            kernel_size=source_reader.kernel_size,
            variance_epsilon=source_reader.variance_epsilon,
            normalize_input=source_reader.normalize_input,
            match_input_rms=source_reader.match_input_rms,
        )
        reader.initialize_semi_orthogonal_()
        terminal.pole_input_projection = reader
        install_calibrated_initialization(
            terminal,
            R2K3_MAXIMUM_PHASES[-1],
            R2K3_FREQUENCY_SCALES[-1],
            R2K3_DAMPING_SCALES[-1],
        )
        install_decoupled_initialization(terminal)
        install_short_damping(terminal, terminal=True)

        source_affine = getattr(classifier, "affine", None)
        if not isinstance(source_affine, StandardizedAffineModalHead):
            raise TypeError("terminal resize requires a standardized affine head")
        expanded = CapacityQ4OnlyAffineClassifier(
            descriptor_dim,
            q4_dim,
            source_affine.linear.out_features,
        )

    terminal.train(source.training)
    expanded.train(classifier.training)
    model.terminal = terminal
    model.descriptor_dim = descriptor_dim
    model.classifier = expanded


def _deep4_spec(spec: CapacitySpec) -> backbone.Deep4BackboneSpec:
    poles = spec.pole_modes
    stem_width = 2 * max(spec.excitation_modes[0], poles[0])
    return backbone.Deep4BackboneSpec(
        modes=poles,
        stem_width=stem_width,
        mode_cffn_widths=tuple(2 * modes for modes in poles[:3]),
        augmented_widths=tuple(2 * modes for modes in poles[:3]),
        post_ffn_widths=tuple(2 * modes for modes in poles[:3]),
    )


def _make_reader(
    excitation_modes: int,
    pole_modes: int,
) -> FactorizedComplexConv2dReader:
    reader = FactorizedComplexConv2dReader(
        excitation_modes,
        pole_modes,
        rank=READER_RANK,
        kernel_size=KERNEL_SIZE,
        match_input_rms=True,
    )
    if excitation_modes == pole_modes:
        reader.initialize_orthogonal_()
    else:
        reader.initialize_semi_orthogonal_()
    return reader


def _replace_analysis(model: ComplexScanBackbone, excitation_modes: int) -> None:
    stem_width = 2 * excitation_modes
    analysis = nn.Linear(stem_width, 2 * excitation_modes, bias=False)
    nn.init.orthogonal_(analysis.weight)
    orthogonal(
        analysis,
        "weight",
        orthogonal_map="matrix_exp",
        use_trivialization=True,
    )
    model.analysis = analysis


def _make_path_only_collapse(modes: int) -> PathOnlyCollapse:
    source = PhaseGatedModeResidualPathCollapse(
        modes,
        mode_hidden=modes,
        path_hidden=PATH_HIDDEN,
    )
    return PathOnlyCollapse(source)


def _install_stage(
    stage: ComplexScanStage,
    *,
    excitation_modes: int,
    pole_modes: int,
    output_modes: int,
    post_hidden: int,
) -> None:
    if stage.modes != pole_modes:
        message = "capacity builder emitted an unexpected pole width"
        raise RuntimeError(message)
    stage.input_modes = excitation_modes
    stage.output_modes = output_modes
    stage.carry_basis = "s2d"
    stage.transition = None
    stage.interaction = None
    stage.widely_bridge = None
    stage.bridge = None
    stage.output_norm = None
    stage.post_transition_ffn = None
    stage.quadrant_path_mode_combiner = _make_path_only_collapse(pole_modes)
    stage.augmented = PoleExcitationS2DPostFusionTransition(
        pole_modes,
        excitation_modes,
        output_modes,
        post_hidden=post_hidden,
    )
    with torch.random.fork_rng(devices=[]):
        stage.pole_input_projection = _make_reader(excitation_modes, pole_modes)


def _install_terminal(
    stage: ComplexScanStage,
    *,
    excitation_modes: int,
    pole_modes: int,
) -> None:
    if stage.modes != pole_modes:
        message = "capacity builder emitted an unexpected terminal pole width"
        raise RuntimeError(message)
    stage.input_modes = excitation_modes
    with torch.random.fork_rng(devices=[]):
        stage.pole_input_projection = _make_reader(excitation_modes, pole_modes)


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    try:
        spec = SPECS[variant]
    except KeyError as error:
        message = f"unsupported R2K3 capacity factory variant: {variant}"
        raise ValueError(message) from error

    return _build_spec(spec, config)


def _build_spec(spec: CapacitySpec, config: ComplexScanConfig) -> ComplexScanBackbone:
    model = backbone.build(_deep4_spec(spec), config)
    excitation_width = 2 * spec.excitation_modes[0]
    model.stem = runtime.make_stem(spec.excitation_modes[0], config.stem_strides)
    if model.precomplex_fc is None:
        message = "capacity model requires the established pre-complex mixer"
        raise TypeError(message)
    if model.config.stem_width == excitation_width:
        model.precomplex_fc = runtime.wrap_precomplex_mixer(model.precomplex_fc)
    else:
        model.input_norm = nn.RMSNorm(excitation_width)
        model.precomplex_fc = runtime.wrap_precomplex_mixer(
            nn.Sequential(
                nn.Linear(excitation_width, excitation_width),
                nn.GELU(),
                nn.Linear(excitation_width, excitation_width),
                nn.Identity(),
            )
        )
    _replace_analysis(model, spec.excitation_modes[0])

    for index, name in enumerate(STAGE_NAMES[:3]):
        _install_stage(
            getattr(model, name),
            excitation_modes=spec.excitation_modes[index],
            pole_modes=spec.pole_modes[index],
            output_modes=spec.excitation_modes[index + 1],
            post_hidden=spec.post_hidden_modes[index + 1],
        )
    _install_terminal(
        model.terminal,
        excitation_modes=spec.excitation_modes[-1],
        pole_modes=spec.pole_modes[-1],
    )
    install_r2k3_pole_initialization(model, STAGE_NAMES)
    model.classifier = CapacityQ4OnlyAffineClassifier(
        spec.descriptor_dim,
        spec.q4_dim,
        config.output_dim,
    )
    _assert_model(model, spec)
    return model


def _assert_model(model: ComplexScanBackbone, spec: CapacitySpec) -> None:
    stages = tuple(getattr(model, name) for name in STAGE_NAMES)
    if model.descriptor_dim != spec.descriptor_dim:
        message = "capacity model raw descriptor width changed"
        raise RuntimeError(message)
    if (
        not hasattr(model.stem, "output_width")
        or model.stem.output_width != 2 * spec.excitation_modes[0]
        or not isinstance(model.input_norm, nn.RMSNorm)
        or model.input_norm.normalized_shape != (2 * spec.excitation_modes[0],)
        or not isinstance(model.analysis, nn.Linear)
        or model.analysis.in_features != 2 * spec.excitation_modes[0]
        or model.analysis.out_features != 2 * spec.excitation_modes[0]
        or model.analysis.bias is not None
        or not parametrize.is_parametrized(model.analysis, "weight")
    ):
        message = "capacity model changed its excitation interface contract"
        raise RuntimeError(message)

    for index, (name, stage) in enumerate(zip(STAGE_NAMES, stages, strict=True)):
        excitation_modes = spec.excitation_modes[index]
        pole_modes = spec.pole_modes[index]
        reader = stage.pole_input_projection
        if (
            stage.input_modes != excitation_modes
            or stage.modes != pole_modes
            or type(reader) is not FactorizedComplexConv2dReader
            or reader.input_modes != excitation_modes
            or reader.output_modes != pole_modes
            or reader.rank != READER_RANK
            or reader.kernel_size != KERNEL_SIZE
            or not reader.match_input_rms
        ):
            message = f"{name} lost its decoupled excitation-to-pole reader contract"
            raise RuntimeError(message)
        if name == "terminal":
            if stage.output_modes is not None:
                message = "capacity terminal unexpectedly emits an excitation"
                raise RuntimeError(message)
            continue

        output_modes = spec.excitation_modes[index + 1]
        post_hidden = spec.post_hidden_modes[index + 1]
        transition = stage.augmented
        if (
            stage.output_modes != output_modes
            or not isinstance(transition, PoleExcitationS2DPostFusionTransition)
            or transition.input_modes != pole_modes
            or transition.excitation_modes != excitation_modes
            or transition.output_modes != output_modes
            or transition.carry_input_modes != 4 * excitation_modes
            or transition.post_hidden != post_hidden
        ):
            message = f"{name} lost its pole-to-excitation transition contract"
            raise RuntimeError(message)

    classifier = model.classifier
    if (
        not isinstance(classifier, CapacityQ4OnlyAffineClassifier)
        or classifier.input_dim != spec.q4_dim
        or classifier.full_descriptor_dim != spec.descriptor_dim
    ):
        message = "capacity model lost its terminal Raw-Q affine head"
        raise RuntimeError(message)


def _variant_config(variant: str) -> dict[str, Any]:
    return _variant_config_for_spec(variant, SPECS[variant])


def _variant_config_for_spec(variant: str, spec: CapacitySpec) -> dict[str, Any]:
    stages = runtime.STAGE_NAMES
    pole_modes = spec.pole_modes
    excitation_modes = spec.excitation_modes
    return {
        "backbone": {
            "name": f"A2D-{variant}",
            "excitation_schedule": list(excitation_modes),
            "pole_schedule": list(pole_modes),
            "spatial_resolutions": [56, 28, 14, 7],
            "product_paths": 4,
            "descriptor_dim": spec.descriptor_dim,
            "stem": {
                "convolutions": f"3-to-32 stride2 then 32-to-{2 * excitation_modes[0]} stride2",
                "precomplex_mixer": (
                    f"residual Linear{2 * excitation_modes[0]}-GELU-Linear{2 * excitation_modes[0]}"
                ),
                "interface_norm": f"RMSNorm{2 * excitation_modes[0]}",
                "complex_projection": (
                    f"semi-orthogonal Linear{2 * excitation_modes[0]}-to-"
                    f"{2 * excitation_modes[0]} then real/imag split"
                ),
            },
            "pole_input": {
                "operator": "rank-2 strict-complex K3 excitation-to-pole reader",
                "shape_by_stage": [
                    f"{excitation}-to-{poles}"
                    for excitation, poles in zip(
                        excitation_modes,
                        pole_modes,
                        strict=True,
                    )
                ],
                "initialization": "orthogonal if square, semi-orthogonal if rectangular",
                "normalization": "unit kernel row energy plus token RMSMatch",
            },
            "transition": {
                "memory": "path collapse at pole width, optional strict CL to next excitation width",
                "carry": "S2D average at excitation width, optional strict CL to next excitation width",
                "merge": "projected memory plus projected carry",
                "post_fusion": {
                    "operator": "target-width residual WL PostFusion",
                    "hidden_ratio": spec.post_hidden_ratio,
                    "hidden_modes_by_excitation_stage": list(spec.post_hidden_modes),
                },
            },
            "descriptor": {
                "operator": "direct raw directional log-energy",
                "shape_by_stage": [4 * modes for modes in pole_modes],
                "basis_transform": "none",
            },
            "pole_initialization": {
                "modes_per_stage": list(pole_modes),
                "radial_levels": [modes // R2K3_ORIENTATIONS for modes in pole_modes],
                "orientations": R2K3_ORIENTATIONS,
                "frequency_scale_by_stage": dict(zip(stages, R2K3_FREQUENCY_SCALES, strict=True)),
                "damping_scale_by_stage": dict(zip(stages, R2K3_DAMPING_SCALES, strict=True)),
                "damping_index_rule": (
                    "coprime radial permutation with orientation offsets "
                    f"{list(R2K3_DAMPING_OFFSETS)}"
                ),
                "log_anisotropy_by_orientation": list(R2K3_LOG_ANISOTROPY),
                "short_memory": {
                    "quantile_power": DAMPING_QUANTILE_POWER,
                    "nonterminal_geometric_range": list(NONTERMINAL_GEOMETRIC_DAMPING_RANGE),
                    "terminal_geometric_range": list(TERMINAL_GEOMETRIC_DAMPING_RANGE),
                    "terminal_damping_max": TERMINAL_DAMPING_MAX,
                },
            },
        },
        "head": {
            "descriptor_source": f"terminal raw Q only; final {spec.q4_dim} coordinates",
            "operator": f"BatchNorm{spec.q4_dim}-affine-false-Linear100",
            "auxiliary": False,
        },
        "optimizer": {"pole_geometry_learning_rate_multiplier": 1.0},
    }
