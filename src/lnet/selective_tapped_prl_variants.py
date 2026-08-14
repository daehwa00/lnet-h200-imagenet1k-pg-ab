from __future__ import annotations

from typing import TYPE_CHECKING, Final, Literal

from .laplace import LaplaceParameterError, LaplaceShapeError

if TYPE_CHECKING:
    from torch import Tensor

SelectiveVariant = Literal[
    "fixed",
    "input_gate",
    "tap_selective",
    "input_tap",
    "full",
    "damping",
    "damping_full",
]
SELECTIVE_VARIANTS: Final[tuple[SelectiveVariant, ...]] = (
    "fixed",
    "input_gate",
    "tap_selective",
    "input_tap",
    "full",
    "damping",
    "damping_full",
)


def uses_input_gate(variant: SelectiveVariant) -> bool:
    return variant in {"input_gate", "input_tap", "full", "damping_full"}


def uses_tap_selectivity(variant: SelectiveVariant) -> bool:
    return variant in {"tap_selective", "input_tap", "full", "damping_full"}


def uses_read_gate(variant: SelectiveVariant) -> bool:
    return variant in {"full", "damping_full"}


def uses_damping_modulation(variant: SelectiveVariant) -> bool:
    return variant in {"damping", "damping_full"}


def check_projected(projected: Tensor, model_dim: int) -> None:
    if projected.ndim != 3 or projected.shape[-1] != model_dim:
        raise LaplaceShapeError(
            actual_shape=tuple(projected.shape),
            expected_rank=3,
            expected_features=model_dim,
        )


def require_positive(value: int, name: str) -> None:
    if value > 0:
        return
    raise LaplaceParameterError(reason=f"{name} must be positive")
