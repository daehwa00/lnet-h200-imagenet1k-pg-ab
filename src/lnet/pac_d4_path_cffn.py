"""Public D4 path-CFFN family with semantic kernel dispatch."""

from .pac_triton_grouped_path_cffn import (
    d4_grouped_cell_path_collapse,
    d4_grouped_cell_path_collapse_reference,
    d4_grouped_cell_path_swiglu_reference,
    d4_grouped_path_collapse,
    d4_grouped_path_collapse_reference,
    d4_grouped_path_swiglu,
    d4_grouped_path_swiglu_reference,
    supports_d4_grouped_cell_path_collapse,
    supports_d4_grouped_path_collapse,
    supports_d4_grouped_path_swiglu,
)

__all__ = [
    "d4_grouped_cell_path_collapse",
    "d4_grouped_cell_path_collapse_reference",
    "d4_grouped_cell_path_swiglu_reference",
    "d4_grouped_path_collapse",
    "d4_grouped_path_collapse_reference",
    "d4_grouped_path_swiglu",
    "d4_grouped_path_swiglu_reference",
    "supports_d4_grouped_cell_path_collapse",
    "supports_d4_grouped_path_collapse",
    "supports_d4_grouped_path_swiglu",
]
