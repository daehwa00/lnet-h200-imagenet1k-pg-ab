"""Leaf contracts shared by the complex scan implementation."""

from __future__ import annotations

from typing import Literal

from torch import Tensor

ComplexField = tuple[Tensor, Tensor]
DirectionalState = tuple[Tensor, Tensor, Tensor]
ComplexStem = Literal[
    "normalized",
    "normalized_no_activation",
    "conv_only",
    "local_fourier",
    "complex_pixel",
]
ComplexCarryBasis = Literal["none", "s2d"]
ComplexCarryMerge = Literal["pole_main", "carry_main"]

DIRECTIONS = ((1, 1), (-1, 1), (1, -1), (-1, -1))
