from __future__ import annotations

from typing import Literal, assert_never

import torch
from torch import Tensor

HybridBackend = Literal["generic", "pac_lite_prl_fused", "pac_lite_block_fused", "auto"]


def is_pac_lite_fused_candidate(
    *,
    active_branches: tuple[str, ...],
    fusion: str,
    has_gate: bool,
    inputs: Tensor,
    model_dim: int,
    modes: int,
    tap_kernel_size: int,
    fir_kernel_size: int,
) -> bool:
    return (
        active_branches == ("prl", "fir")
        and fusion == "learned_scalar_sum"
        and not has_gate
        and inputs.is_cuda
        and inputs.dtype == torch.float32
        and model_dim <= 16
        and modes <= 8
        and tap_kernel_size <= 16
        and fir_kernel_size <= 16
    )


def selected_hybrid_backend(
    requested: HybridBackend,
    *,
    is_candidate: bool,
) -> HybridBackend:
    match requested:
        case "generic":
            return "generic"
        case "auto":
            return "generic"
        case "pac_lite_prl_fused" | "pac_lite_block_fused":
            return requested if is_candidate else "generic"
        case unreachable:
            assert_never(unreachable)
