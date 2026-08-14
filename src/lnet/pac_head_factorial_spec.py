from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

Branch = Literal["lite", "full"]
Direction = Literal["causal", "bidirectional"]
Source = Literal["last", "all_concat", "all_learned_mix", "cross_block"]
RealPool = Literal["none", "mean_max", "pyramid"]
ModalFeature = Literal[
    "final_state",
    "envelope_phase",
    "complex_stats",
    "hermitian",
    "normalized_hermitian",
    "lagged_hermitian",
    "temporal_hermitian_pyramid",
    "lagged_temporal_hermitian",
    "cross_block_hermitian",
    "cross_block_lagged_hermitian",
    "modal_attention",
    "drive_stats",
    "modal_dynamics",
    "hermitian_dynamics_lite",
    "hermitian_dynamics",
]

DATASETS: Final[tuple[str, ...]] = ("ECG5000", "FordA", "FordB", "Wafer")
LAGS: Final[tuple[int, ...]] = (1, 2, 4, 8)
SEGMENTS: Final[tuple[int, ...]] = (1, 2, 4)
MODAL_FEATURES: Final[tuple[ModalFeature, ...]] = (
    "final_state",
    "envelope_phase",
    "complex_stats",
    "hermitian",
    "normalized_hermitian",
    "lagged_hermitian",
    "temporal_hermitian_pyramid",
    "lagged_temporal_hermitian",
    "cross_block_hermitian",
    "cross_block_lagged_hermitian",
    "modal_attention",
    "drive_stats",
    "modal_dynamics",
    "hermitian_dynamics_lite",
    "hermitian_dynamics",
)


@dataclass(frozen=True, slots=True)
class PACHeadSpec:
    branch: Branch
    depth: int
    direction: Direction
    source: Source
    modal_feature: ModalFeature
    real_pool: RealPool
    damping_aux: bool
    fir_aux: bool
    branch_aux: bool


def variant_id(spec: PACHeadSpec) -> str:
    parts = ["pac", spec.branch, f"depth{spec.depth}", spec.direction, spec.source]
    parts.extend((spec.modal_feature, f"real{spec.real_pool}"))
    if spec.damping_aux:
        parts.append("damping")
    if spec.fir_aux:
        parts.append("firaux")
    if spec.branch_aux:
        parts.append("branchaux")
    return "_".join(parts)


def is_cross_feature(feature: ModalFeature) -> bool:
    return feature in {"cross_block_hermitian", "cross_block_lagged_hermitian"}


def valid_spec(spec: PACHeadSpec) -> bool:
    if spec.depth not in {1, 2}:
        return False
    if spec.source == "cross_block" and spec.depth != 2:
        return False
    return is_cross_feature(spec.modal_feature) == (spec.source == "cross_block")


def normalized_spec(spec: PACHeadSpec) -> PACHeadSpec | None:
    source = "last" if spec.depth == 1 and spec.source != "cross_block" else spec.source
    normalized = PACHeadSpec(
        branch=spec.branch,
        depth=spec.depth,
        direction=spec.direction,
        source=source,
        modal_feature=spec.modal_feature,
        real_pool=spec.real_pool,
        damping_aux=spec.damping_aux,
        fir_aux=spec.fir_aux,
        branch_aux=spec.branch_aux,
    )
    return normalized if valid_spec(normalized) else None


def modal_feature_dim(feature: ModalFeature, modes: int) -> int:
    return {
        "final_state": 3 * modes,
        "envelope_phase": 5 * modes,
        "complex_stats": 8 * modes,
        "hermitian": modes * modes,
        "normalized_hermitian": modes * modes,
        "lagged_hermitian": len(LAGS) * modes * modes,
        "temporal_hermitian_pyramid": sum(SEGMENTS) * modes * modes,
        "lagged_temporal_hermitian": sum(SEGMENTS) * len(LAGS) * modes * modes,
        "cross_block_hermitian": modes * modes,
        "cross_block_lagged_hermitian": len(LAGS) * modes * modes,
        "modal_attention": 3 * modes,
        "drive_stats": 6 * modes,
        "modal_dynamics": 5 * modes * modes,
        "hermitian_dynamics_lite": modes * modes + 3 * modes,
        "hermitian_dynamics": modes * modes + 6 * modes,
    }[feature]


def feature_dim(spec: PACHeadSpec, model_dim: int, modes: int) -> int:
    modal_depth = spec.depth if spec.source == "all_concat" else 1
    dim = modal_depth * modal_feature_dim(spec.modal_feature, modes)
    if spec.real_pool == "mean_max":
        dim += 2 * model_dim
    elif spec.real_pool == "pyramid":
        dim += 14 * model_dim
    if spec.damping_aux:
        dim += 6 * modes
    if spec.fir_aux:
        dim += 2 * model_dim
    if spec.branch_aux:
        dim += (6 if spec.branch == "full" else 4) * model_dim
    return dim * (2 if spec.direction == "bidirectional" else 1)
