from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from .pac_headroom_efficient_models import PhaseAugmentedWaveletPacketPAC
from .pac_tight_frame_runtime import (
    BorrowedInputCudaGraphInference,
    CompiledTightFrameInference,
)

if TYPE_CHECKING:
    from torch import nn

PA2WPRuntime = Literal[
    "static_poles_fp32",
    "compiled_static_fp32",
    "manual_graph_static_fp32",
    "phase_batched_fp32",
    "phase_batched_manual_graph_fp32",
    "phase_batched_dynamic_poles_fp32",
    "phase_batched_block_scan_fp32",
    "fused_stem_fp32",
    "fused_stem_default_compile_fp32",
    "ceiling_optimized_fp32",
    "packed_states_fp32",
    "persistent_core_fp32",
]


def prepare_pa2wp_inference(
    model: nn.Module,
    *,
    sequence_length: int,
    batch_size: int,
    runtime: str,
) -> nn.Module:
    """Prepare one auditable PA2WP inference ablation or the measured ceiling path."""
    if sequence_length < 1 or batch_size < 1:
        message = "PA2WP inference requires positive sequence and batch sizes"
        raise ValueError(message)
    active = _checked_pa2wp(model)
    if runtime in {
        "phase_batched_fp32",
        "phase_batched_manual_graph_fp32",
        "phase_batched_dynamic_poles_fp32",
        "phase_batched_block_scan_fp32",
        "fused_stem_fp32",
        "fused_stem_default_compile_fp32",
        "ceiling_optimized_fp32",
        "packed_states_fp32",
        "persistent_core_fp32",
    }:
        active.prepare_batched_phase_inference_()
    use_fused_stem = runtime in {
        "fused_stem_fp32",
        "fused_stem_default_compile_fp32",
        "ceiling_optimized_fp32",
        "packed_states_fp32",
        "persistent_core_fp32",
    } and not (
        runtime
        in {
            "ceiling_optimized_fp32",
            "packed_states_fp32",
            "persistent_core_fp32",
        }
        and sequence_length >= 2048
        and batch_size >= 64
    )
    if use_fused_stem:
        _prepare_fused_stem(active)
    use_block_scan = runtime == "phase_batched_block_scan_fp32"
    use_static_poles = runtime not in {
        "phase_batched_dynamic_poles_fp32",
        "phase_batched_block_scan_fp32",
    }
    use_persistent_packed_io = runtime == "persistent_core_fp32" and not (
        sequence_length >= 2048 and batch_size >= 64
    )
    _prepare_blocks(
        active,
        use_block_scan=use_block_scan,
        use_static_poles=use_static_poles,
        use_packed_states=runtime == "packed_states_fp32" or use_persistent_packed_io,
        use_packed_drive=runtime == "persistent_core_fp32",
        use_single_warp=runtime == "persistent_core_fp32",
    )

    if runtime in {"static_poles_fp32", "phase_batched_fp32"}:
        return active
    if runtime == "compiled_static_fp32":
        return CompiledTightFrameInference(
            active,
            "max-autotune-no-cudagraphs",
            copy_output=False,
        )
    if runtime in {
        "manual_graph_static_fp32",
        "phase_batched_manual_graph_fp32",
        "phase_batched_dynamic_poles_fp32",
        "phase_batched_block_scan_fp32",
        "fused_stem_fp32",
        "fused_stem_default_compile_fp32",
        "ceiling_optimized_fp32",
        "packed_states_fp32",
        "persistent_core_fp32",
    }:
        return BorrowedInputCudaGraphInference(
            active,
            compile_mode=(
                "default"
                if runtime == "fused_stem_default_compile_fp32"
                else "max-autotune-no-cudagraphs"
            ),
            copy_output=False,
        )
    message = f"unknown PA2WP runtime: {runtime}"
    raise ValueError(message)


def prepare_pa2wp_ceiling_inference(
    model: nn.Module,
    *,
    sequence_length: int,
    batch_size: int,
) -> nn.Module:
    """Prepare exact-FP32 PA2WP with static poles and caller-owned CUDA Graph I/O."""
    return prepare_pa2wp_inference(
        model,
        sequence_length=sequence_length,
        batch_size=batch_size,
        runtime="ceiling_optimized_fp32",
    )


def prepare_pa2wp_persistent_core_inference(
    model: nn.Module,
    *,
    sequence_length: int,
    batch_size: int,
) -> nn.Module:
    """Prepare the measured exact-FP32 PA2WP low-level RTX 4090 path."""
    return prepare_pa2wp_inference(
        model,
        sequence_length=sequence_length,
        batch_size=batch_size,
        runtime="persistent_core_fp32",
    )


def _checked_pa2wp(model: nn.Module) -> PhaseAugmentedWaveletPacketPAC:
    if not isinstance(model, PhaseAugmentedWaveletPacketPAC) or not model.ensemble_inference:
        message = "PA2WP runtime requires an ensemble-inference PA2WP model"
        raise TypeError(message)
    model.eval()
    return model


def _prepare_blocks(
    model: PhaseAugmentedWaveletPacketPAC,
    *,
    use_block_scan: bool,
    use_static_poles: bool,
    use_packed_states: bool,
    use_packed_drive: bool,
    use_single_warp: bool,
) -> None:
    for block in (model.forward_block, model.backward_block):
        block.prepare_for_inference_(
            use_block_scan=use_block_scan,
            use_fused_recurrence_moments=not use_block_scan,
            use_static_pole_recurrence_moments=use_static_poles,
            use_packed_static_recurrence_moments=use_packed_states,
            use_packed_static_recurrence_drive=use_packed_drive,
            use_single_warp_static_recurrence_moments=use_single_warp,
        )


def _prepare_fused_stem(model: PhaseAugmentedWaveletPacketPAC) -> None:
    prepare = getattr(model, "prepare_fused_pa2wp_stem_inference_", None)
    if not callable(prepare):
        message = "PA2WP model has no fused-stem inference implementation"
        raise TypeError(message)
    prepare()
