"""Configuration schema and validation for the complex scan backbone."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .complex_scan_types import ComplexCarryBasis, ComplexCarryMerge, ComplexStem


def _validate_transition_config(
    transition_widths: tuple[int, int] | None,
    interaction_ranks: tuple[int, int] | None,
    *,
    widely_linear_bridges: bool,
    augmented_widths: tuple[int, int] | None,
) -> None:
    if transition_widths is not None and (
        len(transition_widths) != 2 or any(width <= 0 for width in transition_widths)
    ):
        message = "complex scan requires two positive transition widths"
        raise ValueError(message)
    if interaction_ranks is not None and (
        len(interaction_ranks) != 2 or any(rank <= 0 for rank in interaction_ranks)
    ):
        message = "complex scan requires two positive interaction ranks"
        raise ValueError(message)
    if augmented_widths is not None and (
        len(augmented_widths) != 2 or any(width <= 0 for width in augmented_widths)
    ):
        message = "complex scan requires two positive augmented widths"
        raise ValueError(message)
    enabled = sum(
        (
            transition_widths is not None,
            interaction_ranks is not None,
            widely_linear_bridges,
            augmented_widths is not None,
        )
    )
    if enabled > 1:
        message = "complex scan accepts only one transition family"
        raise ValueError(message)


def _validate_carry_config(
    carry_bases: tuple[ComplexCarryBasis, ComplexCarryBasis],
    carry_merge: ComplexCarryMerge,
    carry_scale_initial: float,
    augmented_widths: tuple[int, int] | None,
) -> None:
    if len(carry_bases) != 2 or any(basis not in {"none", "s2d"} for basis in carry_bases):
        message = "complex scan requires two valid carry bases"
        raise ValueError(message)
    if any(basis != "none" for basis in carry_bases) and augmented_widths is None:
        message = "complex scan stage carry requires augmented transitions"
        raise ValueError(message)
    if carry_merge not in {"pole_main", "carry_main"}:
        message = f"unsupported complex scan carry merge: {carry_merge}"
        raise ValueError(message)
    if carry_scale_initial < 0.0:
        message = "complex scan carry scale cannot be negative"
        raise ValueError(message)


def _validate_stem_config(stem: ComplexStem, strides: tuple[int, int]) -> None:
    if stem not in {
        "normalized",
        "normalized_no_activation",
        "conv_only",
        "local_fourier",
        "complex_pixel",
    }:
        message = f"unsupported complex scan stem: {stem}"
        raise ValueError(message)
    if len(strides) != 2 or any(stride <= 0 for stride in strides):
        message = "complex scan requires two positive stem strides"
        raise ValueError(message)


@dataclass(frozen=True, slots=True)
class ComplexScanConfig:
    output_dim: int = 100
    stem_width: int = 64
    stem: ComplexStem = "normalized"
    stem_strides: tuple[int, int] = (1, 1)
    use_post_stem_rmsnorm: bool = True
    use_precomplex_fc: bool = False
    precomplex_fc_layers: int = 1
    modes: tuple[int, int, int] = (16, 24, 32)
    transition_widths: tuple[int, int] | None = None
    interaction_ranks: tuple[int, int] | None = None
    widely_linear_bridges: bool = False
    augmented_widths: tuple[int, int] | None = None
    carry_bases: tuple[ComplexCarryBasis, ComplexCarryBasis] = ("none", "none")
    carry_merge: ComplexCarryMerge = "pole_main"
    carry_scale_initial: float = 1.0e-2
    coherence_gated_carry: bool = False
    use_pole_aligned_shortcuts: bool = False
    use_cccn_shortcuts: bool = False
    zero_gated_pole_aligned_residuals: bool = False
    quadrant_path_mode_cffn_widths: tuple[int, int] | None = None
    quadrant_path_cffn_widths: tuple[int, int] | None = None
    post_transition_widths: tuple[int, int] | None = None
    stage_residual_scale_initial: float = 0.1
    scan_memory_policy: Literal["retain", "recompute"] = "retain"
    quadratic_rank: int = 16
    fusion_width: int | None = None
    dual_fusion_lrq_head: bool = False
    damping_min: float = 0.01
    damping_max: float = 0.7

    def _validate_core(self) -> None:
        if self.output_dim <= 0 or self.stem_width <= 0:
            message = "complex scan output and stem widths must be positive"
            raise ValueError(message)
        _validate_stem_config(self.stem, self.stem_strides)
        if len(self.modes) != 3 or any(modes <= 0 or modes % 4 for modes in self.modes):
            message = "complex scan requires three positive mode counts divisible by four"
            raise ValueError(message)
        _validate_transition_config(
            self.transition_widths,
            self.interaction_ranks,
            widely_linear_bridges=self.widely_linear_bridges,
            augmented_widths=self.augmented_widths,
        )
        _validate_carry_config(
            self.carry_bases,
            self.carry_merge,
            self.carry_scale_initial,
            self.augmented_widths,
        )

    def _validate_shortcuts(self) -> None:
        if self.use_pole_aligned_shortcuts and len(set(self.modes)) != 1:
            message = "pole-aligned shortcuts require equal mode counts across stages"
            raise ValueError(message)
        if self.use_pole_aligned_shortcuts and any(basis != "none" for basis in self.carry_bases):
            message = "pole-aligned shortcuts replace S2D stage carry"
            raise ValueError(message)
        if self.use_cccn_shortcuts and len(set(self.modes)) != 1:
            message = "complex CCCN shortcuts require equal mode counts across stages"
            raise ValueError(message)
        if self.use_cccn_shortcuts and (
            self.use_pole_aligned_shortcuts or any(basis != "none" for basis in self.carry_bases)
        ):
            message = "complex CCCN shortcuts replace S2D and pole-aligned shortcuts"
            raise ValueError(message)
        if self.coherence_gated_carry and any(basis == "none" for basis in self.carry_bases):
            message = "coherence-gated carry requires carry inputs at every stage"
            raise ValueError(message)
        if self.zero_gated_pole_aligned_residuals and len(set(self.modes)) != 1:
            message = "zero-gated pole-aligned residuals require equal mode counts"
            raise ValueError(message)
        if self.zero_gated_pole_aligned_residuals and self.use_pole_aligned_shortcuts:
            message = "zero-gated and replacement pole-aligned residuals are exclusive"
            raise ValueError(message)
        if self.stage_residual_scale_initial <= 0.0:
            message = "complex stage residual scale must be positive"
            raise ValueError(message)

    def _validate_paths(self) -> None:
        if (self.quadrant_path_mode_cffn_widths is None) != (
            self.quadrant_path_cffn_widths is None
        ):
            message = "quadrant path/mode CFFN requires both stage width tuples"
            raise ValueError(message)
        if self.quadrant_path_mode_cffn_widths is not None and (
            len(self.quadrant_path_mode_cffn_widths) != 2
            or any(width <= 0 for width in self.quadrant_path_mode_cffn_widths)
            or self.quadrant_path_cffn_widths is None
            or len(self.quadrant_path_cffn_widths) != 2
            or any(width <= 0 for width in self.quadrant_path_cffn_widths)
        ):
            message = "quadrant path/mode CFFN requires two positive width tuples"
            raise ValueError(message)
        if self.post_transition_widths is not None and (
            len(self.post_transition_widths) != 2
            or any(width <= 0 for width in self.post_transition_widths)
        ):
            message = "post-transition CFFN requires two positive stage widths"
            raise ValueError(message)

    def _validate_stem_projection(self) -> None:
        if self.stem not in {"local_fourier", "complex_pixel"} and (
            2 * self.modes[0] > self.stem_width
        ):
            message = "first complex projection requires 2M <= stem width"
            raise ValueError(message)
        if self.stem == "complex_pixel" and self.modes[0] != (
            2 * math.prod(self.stem_strides) ** 2
        ):
            message = "complex pixel modes must match its lossless S2D packing width"
            raise ValueError(message)
        if self.use_precomplex_fc and self.stem in {"local_fourier", "complex_pixel"}:
            message = "pre-complex FC requires a real-valued stem"
            raise ValueError(message)
        if self.precomplex_fc_layers <= 0:
            message = "pre-complex FC depth must be positive"
            raise ValueError(message)

    def _validate_head_and_numeric_bounds(self) -> None:
        if self.quadratic_rank <= 0:
            message = "complex scan requires a positive LRQ rank"
            raise ValueError(message)
        if self.fusion_width is not None and self.fusion_width <= 0:
            message = "complex scan fusion width must be positive"
            raise ValueError(message)
        if self.dual_fusion_lrq_head and self.fusion_width is None:
            message = "dual Fusion/LRQ head requires a fusion width"
            raise ValueError(message)
        if self.scan_memory_policy not in {"retain", "recompute"}:
            message = f"unsupported scan memory policy: {self.scan_memory_policy}"
            raise ValueError(message)
        if not 0.0 < self.damping_min < self.damping_max:
            message = "complex scan damping bounds are invalid"
            raise ValueError(message)

    def validate(self) -> None:
        self._validate_core()
        self._validate_shortcuts()
        self._validate_paths()
        self._validate_stem_projection()
        self._validate_head_and_numeric_bounds()
