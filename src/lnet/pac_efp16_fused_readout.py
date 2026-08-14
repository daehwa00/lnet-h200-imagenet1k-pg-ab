from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

from .pac_tight_frame_runtime import (
    InferenceCompileMode,
    prepare_efp16_manual_graph_inference,
)

if TYPE_CHECKING:
    from .pac_headroom_models import HeadroomPACClassifier


def prepare_efp16_fused_readout_candidate(
    model: HeadroomPACClassifier,
    *,
    sequence_length: int,
    batch_size: int,
    compile_mode: InferenceCompileMode = "max-autotune-no-cudagraphs",
    copy_output: bool = False,
) -> nn.Module:
    """Prepare an opt-in ceiling-equivalent EFP16 runtime with fused readout."""
    if sequence_length < 2 or batch_size < 1:
        message = "EFP16 fused readout requires sequence_length>=2 and batch_size>=1"
        raise ValueError(message)
    _validate_canonical_model(model)
    model.use_fused_efp16_inference_readout = True
    use_n2048_b1_packed_scan = sequence_length == 2048 and batch_size == 1
    if compile_mode != "none":
        return prepare_efp16_manual_graph_inference(
            model,
            sequence_length=sequence_length,
            batch_size=batch_size,
            use_static_poles=True,
            compile_mode=compile_mode,
            copy_output=copy_output,
            use_packed_block_scan=use_n2048_b1_packed_scan,
            use_single_warp_block_scan=use_n2048_b1_packed_scan,
            block_scan_block_size=64 if use_n2048_b1_packed_scan else 256,
        )
    model.eval()
    use_block_scan = sequence_length >= 2048 and batch_size == 1
    if batch_size == 1 or sequence_length <= 128:
        prepare_stem = getattr(model.stem, "prepare_fused_raw_inference_", None)
        if not callable(prepare_stem):
            message = "EFP16 model has no inference-ready edge-frame stem"
            raise TypeError(message)
        prepare_stem()
    for block in (model.forward_block, model.backward_block):
        block.prepare_for_inference_(
            use_block_scan=use_block_scan,
            use_fused_recurrence_moments=not use_block_scan,
            use_static_pole_recurrence_moments=not use_block_scan,
            use_packed_static_recurrence_moments=use_n2048_b1_packed_scan,
            use_packed_static_recurrence_drive=use_n2048_b1_packed_scan,
            use_single_warp_static_recurrence_moments=use_n2048_b1_packed_scan,
            use_static_pole_block_scan=use_n2048_b1_packed_scan,
            static_pole_block_scan_block_size=(
                64 if use_n2048_b1_packed_scan else 256
            ),
        )
    return model


def _validate_canonical_model(model: HeadroomPACClassifier) -> None:
    head = model.head
    classifier = getattr(head, "classifier", None)
    final_norm = model.final_norm
    spec_name = getattr(model.spec, "name", None)
    forward_lags = getattr(model.forward_block, "moment_lags", None)
    backward_lags = getattr(model.backward_block, "moment_lags", None)
    if (
        model.model_dim != 32
        or model.modes != 16
        or spec_name != "B"
        or final_norm is None
        or final_norm.weight is None
        or not isinstance(classifier, nn.Linear)
        or classifier.in_features != 192
        or classifier.out_features != 5
        or classifier.bias is None
        or not bool(getattr(head, "use_modal_moments", False))
        or not bool(getattr(head, "use_backward_moments", False))
        or forward_lags != (1, 4)
        or backward_lags != (1, 4)
    ):
        message = (
            "fused readout candidate requires canonical EFP16 D32/M16, "
            "two 80-wide moment branches, and a biased 192-to-5 invariant head"
        )
        raise ValueError(message)
