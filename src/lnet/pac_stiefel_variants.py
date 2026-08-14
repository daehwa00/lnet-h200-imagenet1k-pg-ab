from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Final, Literal

FrameParameterization = Literal["matrix_exp", "qr_retraction", "unconstrained"]


@dataclass(frozen=True, slots=True)
class StiefelVariant:
    synthesis_scale: float
    layer_scale_init: float
    align_moments: bool
    log_energy: bool
    normalize_autocorrelation: bool
    frame_parameterization: FrameParameterization
    split_residual_scales: bool = False
    qr_retraction_interval: int = 1
    use_mode_gate: bool = True
    use_backward_block: bool = True
    use_modal_moments: bool = True
    use_ordered_pool: bool = True
    use_local_convolution: bool = True
    tie_analysis_synthesis: bool = True
    stem_kernel: int = 9
    stem_stride: int = 2
    local_kernel: int = 5
    depth: int = 2
    moment_lags: tuple[int, ...] = (1, 4)
    pooling_scales: tuple[int, ...] = (1, 2, 4)
    damping_min: float = 1.0e-3
    damping_max: float = 2.0
    gate_max: float = 2.0


LEGACY_VARIANT: Final[StiefelVariant] = StiefelVariant(
    synthesis_scale=2.0,
    layer_scale_init=1.0e-2,
    align_moments=False,
    log_energy=False,
    normalize_autocorrelation=False,
    frame_parameterization="matrix_exp",
)
_B_CANONICAL_SYNTHESIS: Final[StiefelVariant] = StiefelVariant(
    synthesis_scale=1.0,
    layer_scale_init=1.0e-2,
    align_moments=False,
    log_energy=False,
    normalize_autocorrelation=False,
    frame_parameterization="matrix_exp",
)
_C_SCALE_MATCHED: Final[StiefelVariant] = StiefelVariant(
    synthesis_scale=1.0,
    layer_scale_init=2.0e-2,
    align_moments=False,
    log_energy=False,
    normalize_autocorrelation=False,
    frame_parameterization="matrix_exp",
)
_D_BACKWARD_ALIGNED: Final[StiefelVariant] = StiefelVariant(
    synthesis_scale=1.0,
    layer_scale_init=1.0e-2,
    align_moments=True,
    log_energy=False,
    normalize_autocorrelation=False,
    frame_parameterization="matrix_exp",
)
_E_LOG_ENERGY: Final[StiefelVariant] = StiefelVariant(
    synthesis_scale=1.0,
    layer_scale_init=1.0e-2,
    align_moments=True,
    log_energy=True,
    normalize_autocorrelation=False,
    frame_parameterization="matrix_exp",
)
CORRECTED_VARIANT: Final[StiefelVariant] = StiefelVariant(
    synthesis_scale=1.0,
    layer_scale_init=1.0e-2,
    align_moments=True,
    log_energy=True,
    normalize_autocorrelation=True,
    frame_parameterization="matrix_exp",
)
REVISED_UNTIED_VARIANT: Final[StiefelVariant] = replace(
    CORRECTED_VARIANT,
    use_mode_gate=False,
    use_ordered_pool=False,
    tie_analysis_synthesis=False,
)
REVISED_UNTIED_MODEL: Final[str] = "pac_stiefel_revised_fixed_mean_nogate_untied_d64_m16"
DEFAULT_PAC_MODEL: Final[str] = REVISED_UNTIED_MODEL
_G_QR_RETRACTION: Final[StiefelVariant] = StiefelVariant(
    synthesis_scale=1.0,
    layer_scale_init=1.0e-2,
    align_moments=True,
    log_energy=True,
    normalize_autocorrelation=True,
    frame_parameterization="qr_retraction",
)
_H_SPLIT_RESIDUAL: Final[StiefelVariant] = StiefelVariant(
    synthesis_scale=1.0,
    layer_scale_init=1.0e-2,
    align_moments=True,
    log_energy=True,
    normalize_autocorrelation=True,
    frame_parameterization="matrix_exp",
    split_residual_scales=True,
)
_I_PERIODIC_QR4: Final[StiefelVariant] = StiefelVariant(
    synthesis_scale=1.0,
    layer_scale_init=1.0e-2,
    align_moments=True,
    log_energy=True,
    normalize_autocorrelation=True,
    frame_parameterization="qr_retraction",
    qr_retraction_interval=4,
)

_ABLATE_UNCONSTRAINED_FRAME: Final[StiefelVariant] = replace(
    CORRECTED_VARIANT,
    frame_parameterization="unconstrained",
)
_ABLATE_NO_MODE_GATE: Final[StiefelVariant] = replace(
    CORRECTED_VARIANT,
    use_mode_gate=False,
)
_ABLATE_FORWARD_ONLY: Final[StiefelVariant] = replace(
    CORRECTED_VARIANT,
    use_backward_block=False,
)
_ABLATE_NO_MODAL_MOMENTS: Final[StiefelVariant] = replace(
    CORRECTED_VARIANT,
    use_modal_moments=False,
)
_ABLATE_MEAN_POOL: Final[StiefelVariant] = replace(
    CORRECTED_VARIANT,
    use_ordered_pool=False,
)
_ABLATE_NO_LOCAL_CONV: Final[StiefelVariant] = replace(
    CORRECTED_VARIANT,
    use_local_convolution=False,
)
_ABLATE_UNNORMALIZED_CORRELATION: Final[StiefelVariant] = replace(
    CORRECTED_VARIANT,
    normalize_autocorrelation=False,
)
_ABLATE_UNTIED_SYNTHESIS: Final[StiefelVariant] = replace(
    CORRECTED_VARIANT,
    tie_analysis_synthesis=False,
)

STIEFEL_MODELS: Final[tuple[str, ...]] = ("pac_stiefel_depth2_norm_autocorr",)
STIEFEL_FACTORIAL_MODELS: Final[tuple[str, ...]] = (
    "pac_stiefel_b_canonical_synthesis",
    "pac_stiefel_c_scale_matched",
    "pac_stiefel_d_backward_aligned",
    "pac_stiefel_e_log_energy",
    "pac_stiefel_g_qr_retraction",
    "pac_stiefel_h_split_residual",
    "pac_stiefel_i_periodic_qr4",
)
STIEFEL_OPTIMIZED_ABLATION_MODELS: Final[tuple[str, ...]] = (
    "pac_tight_frame_depth2_autocorr",
    "pac_stiefel_depth2_norm_autocorr",
    "pac_stiefel_g_qr_retraction",
    "pac_stiefel_h_split_residual",
    "pac_stiefel_i_periodic_qr4",
)
STIEFEL_CAPACITY_MODELS: Final[tuple[str, ...]] = (
    "pac_stiefel_depth2_norm_autocorr",
    "pac_stiefel_depth2_norm_autocorr_d24_m4",
    "pac_stiefel_depth2_norm_autocorr_d24_m6",
    "pac_stiefel_depth2_norm_autocorr_d32_m8",
)
STIEFEL_LARGE_CAPACITY_MODELS: Final[tuple[str, ...]] = (
    "pac_stiefel_depth2_norm_autocorr_d64_m8",
    "pac_stiefel_depth2_norm_autocorr_d64_m16",
    "pac_stiefel_depth2_norm_autocorr_d64_m32",
    "pac_stiefel_depth2_norm_autocorr_d128_m8",
    "pac_stiefel_depth2_norm_autocorr_d128_m16",
    "pac_stiefel_depth2_norm_autocorr_d128_m32",
)
STIEFEL_ALL_CAPACITY_MODELS: Final[tuple[str, ...]] = (
    *STIEFEL_CAPACITY_MODELS,
    *STIEFEL_LARGE_CAPACITY_MODELS,
)
STIEFEL_CONFIRMATORY_CAPACITY_MODELS: Final[tuple[str, ...]] = tuple(
    model
    for model in STIEFEL_ALL_CAPACITY_MODELS
    if model != "pac_stiefel_depth2_norm_autocorr"
)
STIEFEL_CORE_COMPONENT_ABLATION_MODELS: Final[tuple[str, ...]] = (
    "pac_stiefel_depth2_norm_autocorr_d64_m16",
    "pac_stiefel_ablate_unconstrained_frame_d64_m16",
    "pac_stiefel_ablate_no_mode_gate_d64_m16",
    "pac_stiefel_ablate_forward_only_d64_m16",
    "pac_stiefel_ablate_no_modal_moments_d64_m16",
    "pac_stiefel_ablate_mean_pool_d64_m16",
    "pac_stiefel_ablate_no_local_conv_d64_m16",
    "pac_stiefel_ablate_unnormalized_correlation_d64_m16",
    "pac_stiefel_ablate_untied_synthesis_d64_m16",
)
_CAPACITY_BY_MODEL: Final[dict[str, tuple[int, int]]] = {
    "pac_stiefel_depth2_norm_autocorr_d24_m4": (24, 4),
    "pac_stiefel_depth2_norm_autocorr_d24_m6": (24, 6),
    "pac_stiefel_depth2_norm_autocorr_d32_m8": (32, 8),
    "pac_stiefel_depth2_norm_autocorr_d64_m8": (64, 8),
    "pac_stiefel_depth2_norm_autocorr_d64_m16": (64, 16),
    "pac_stiefel_depth2_norm_autocorr_d64_m32": (64, 32),
    "pac_stiefel_depth2_norm_autocorr_d128_m8": (128, 8),
    "pac_stiefel_depth2_norm_autocorr_d128_m16": (128, 16),
    "pac_stiefel_depth2_norm_autocorr_d128_m32": (128, 32),
    **dict.fromkeys(STIEFEL_CORE_COMPONENT_ABLATION_MODELS, (64, 16)),
    REVISED_UNTIED_MODEL: (64, 16),
}
_MODEL_VARIANTS: Final[dict[str, StiefelVariant]] = {
    "pac_tight_frame_depth2_autocorr": LEGACY_VARIANT,
    "pac_stiefel_b_canonical_synthesis": _B_CANONICAL_SYNTHESIS,
    "pac_stiefel_c_scale_matched": _C_SCALE_MATCHED,
    "pac_stiefel_d_backward_aligned": _D_BACKWARD_ALIGNED,
    "pac_stiefel_e_log_energy": _E_LOG_ENERGY,
    "pac_stiefel_depth2_norm_autocorr": CORRECTED_VARIANT,
    "pac_stiefel_g_qr_retraction": _G_QR_RETRACTION,
    "pac_stiefel_h_split_residual": _H_SPLIT_RESIDUAL,
    "pac_stiefel_i_periodic_qr4": _I_PERIODIC_QR4,
    "pac_stiefel_depth2_norm_autocorr_d24_m4": CORRECTED_VARIANT,
    "pac_stiefel_depth2_norm_autocorr_d24_m6": CORRECTED_VARIANT,
    "pac_stiefel_depth2_norm_autocorr_d32_m8": CORRECTED_VARIANT,
    "pac_stiefel_depth2_norm_autocorr_d64_m8": CORRECTED_VARIANT,
    "pac_stiefel_depth2_norm_autocorr_d64_m16": CORRECTED_VARIANT,
    "pac_stiefel_depth2_norm_autocorr_d64_m32": CORRECTED_VARIANT,
    "pac_stiefel_depth2_norm_autocorr_d128_m8": CORRECTED_VARIANT,
    "pac_stiefel_depth2_norm_autocorr_d128_m16": CORRECTED_VARIANT,
    "pac_stiefel_depth2_norm_autocorr_d128_m32": CORRECTED_VARIANT,
    REVISED_UNTIED_MODEL: REVISED_UNTIED_VARIANT,
    "pac_stiefel_ablate_unconstrained_frame_d64_m16": _ABLATE_UNCONSTRAINED_FRAME,
    "pac_stiefel_ablate_no_mode_gate_d64_m16": _ABLATE_NO_MODE_GATE,
    "pac_stiefel_ablate_forward_only_d64_m16": _ABLATE_FORWARD_ONLY,
    "pac_stiefel_ablate_no_modal_moments_d64_m16": _ABLATE_NO_MODAL_MOMENTS,
    "pac_stiefel_ablate_mean_pool_d64_m16": _ABLATE_MEAN_POOL,
    "pac_stiefel_ablate_no_local_conv_d64_m16": _ABLATE_NO_LOCAL_CONV,
    "pac_stiefel_ablate_unnormalized_correlation_d64_m16": (
        _ABLATE_UNNORMALIZED_CORRELATION
    ),
    "pac_stiefel_ablate_untied_synthesis_d64_m16": _ABLATE_UNTIED_SYNTHESIS,
}


def variant_for_model(name: str) -> StiefelVariant | None:
    return _MODEL_VARIANTS.get(name)


def capacity_for_model(name: str) -> tuple[int, int] | None:
    return _CAPACITY_BY_MODEL.get(name)


def uses_full_modal_frame(name: str) -> bool:
    return name == "pac_stiefel_depth2_norm_autocorr_d64_m32"
