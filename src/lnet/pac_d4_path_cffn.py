"""Public D4 path-CFFN family with semantic kernel dispatch."""

from .pac_triton_grouped_path_cffn import (
    d4_grouped_path_collapse,
    d4_grouped_path_collapse_reference,
    supports_d4_grouped_path_collapse,
)

__all__ = [
    "d4_grouped_path_collapse",
    "d4_grouped_path_collapse_reference",
    "supports_d4_grouped_path_collapse",
]
