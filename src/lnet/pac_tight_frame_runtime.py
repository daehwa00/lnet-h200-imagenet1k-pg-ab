from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from .pac_headroom_models import HeadroomPACClassifier
    from .pac_tight_frame_models import TightFrameClassifier

InferenceCompileMode = Literal[
    "none",
    "default",
    "dynamic-no-cudagraph",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
]

_CapturedInputSignature = tuple[
    torch.Size,
    tuple[int, ...],
    torch.dtype,
    torch.device,
    int,
]


def _input_signature(inputs: Tensor) -> _CapturedInputSignature:
    return (
        inputs.shape,
        inputs.stride(),
        inputs.dtype,
        inputs.device,
        int(inputs.storage_offset()),
    )


class CompiledTightFrameInference(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        mode: InferenceCompileMode,
        *,
        copy_output: bool = True,
    ) -> None:
        super().__init__()
        self.uses_cuda_graphs = mode in {"reduce-overhead", "max-autotune"}
        self.copy_output = copy_output
        self.compiled = _compile_inference_model(model, mode)

    @torch.no_grad()
    def forward(self, inputs: Tensor) -> Tensor:
        if self.uses_cuda_graphs and inputs.is_cuda:
            torch.compiler.cudagraph_mark_step_begin()
        output = self.compiled(inputs)
        # CUDA Graph outputs use static buffers. The safe default clones the small
        # logits tensor; latency-critical callers can explicitly consume it in place.
        return output.clone() if self.copy_output else output


class BorrowedInputCudaGraphInference(nn.Module):
    """Replay a manually captured graph over one caller-owned CUDA input buffer.

    The first call compiles and captures against ``inputs``. Later calls must reuse
    that exact allocation; callers may update its contents in place before replay.
    Both the input allocation and returned output are borrowed graph buffers.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        compile_mode: InferenceCompileMode = "default",
        copy_output: bool = False,
    ) -> None:
        super().__init__()
        self.compiled = _compile_inference_model(model, compile_mode)
        self.copy_output = copy_output
        self.graph: torch.cuda.CUDAGraph | None = None
        self.captured_input: Tensor | None = None
        self.captured_input_data_ptr: int | None = None
        self.captured_input_signature: _CapturedInputSignature | None = None
        self.output: Tensor | None = None

    @torch.no_grad()
    def forward(self, inputs: Tensor) -> Tensor:
        if not inputs.is_cuda:
            message = "manual CUDA Graph inference requires CUDA inputs"
            raise ValueError(message)
        if self.graph is None:
            self.compiled(inputs)
            torch.cuda.synchronize(inputs.device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = self.compiled(inputs)
            self.graph = graph
            self.captured_input = inputs
            self.captured_input_data_ptr = inputs.data_ptr()
            self.captured_input_signature = _input_signature(inputs)
            self.output = output
            graph.replay()
            return output.clone() if self.copy_output else output
        if (
            inputs is not self.captured_input
            or inputs.data_ptr() != self.captured_input_data_ptr
        ):
            message = "manual CUDA Graph inference requires the captured input allocation"
            raise ValueError(message)
        if _input_signature(inputs) != self.captured_input_signature:
            message = "manual CUDA Graph inference requires the captured input layout"
            raise ValueError(message)
        self.graph.replay()
        if self.output is None:
            message = "manual CUDA Graph output buffer was not captured"
            raise RuntimeError(message)
        return self.output.clone() if self.copy_output else self.output


def prepare_tight_frame_inference(
    model: TightFrameClassifier,
    *,
    compile_mode: InferenceCompileMode = "reduce-overhead",
) -> nn.Module:
    """Mutate a dedicated inference model, then optionally compile it for static replay."""
    model.eval()
    if compile_mode == "none":
        return model.materialize_frames_for_inference_()
    model.prepare_compiled_inference_()
    return CompiledTightFrameInference(model, compile_mode)


def prepare_efp16_inference(
    model: HeadroomPACClassifier,
    *,
    compile_mode: InferenceCompileMode = "reduce-overhead",
    use_block_scan: bool = False,
    use_fused_edge_stem: bool = False,
    use_static_pole_recurrence_moments: bool = False,
    copy_output: bool = True,
) -> nn.Module:
    """Prepare a dedicated EFP16 model for static-shape inference."""
    model.eval()
    if use_fused_edge_stem:
        prepare_stem = getattr(model.stem, "prepare_fused_raw_inference_", None)
        if not callable(prepare_stem):
            message = "EFP16 model has no inference-ready edge-frame stem"
            raise TypeError(message)
        prepare_stem()
    use_fused = compile_mode != "none" and not use_block_scan
    for block in (model.forward_block, model.backward_block):
        block.prepare_for_inference_(
            use_block_scan=use_block_scan,
            use_fused_recurrence_moments=use_fused,
            use_static_pole_recurrence_moments=use_static_pole_recurrence_moments,
        )
    if compile_mode == "none":
        return model
    return CompiledTightFrameInference(model, compile_mode, copy_output=copy_output)


def prepare_efp16_max_inference(
    model: HeadroomPACClassifier,
    *,
    sequence_length: int,
    batch_size: int,
    copy_output: bool = False,
) -> nn.Module:
    """Select the measured RTX 4090 EFP16 path for a static input shape.

    The default returns a borrowed CUDA Graph output buffer for minimum latency.
    Consume it before the next replay, or set ``copy_output=True`` when retaining logits.
    """
    if sequence_length < 2 or batch_size < 1:
        message = "EFP16 maximum inference requires sequence_length>=2 and batch_size>=1"
        raise ValueError(message)
    return prepare_efp16_inference(
        model,
        compile_mode="reduce-overhead",
        use_block_scan=sequence_length >= 2048 and batch_size == 1,
        use_fused_edge_stem=batch_size == 1 or sequence_length <= 128,
        copy_output=copy_output,
    )


def prepare_efp16_ceiling_inference(
    model: HeadroomPACClassifier,
    *,
    sequence_length: int,
    batch_size: int,
    copy_output: bool = False,
) -> nn.Module:
    """Select the measured exact-FP32 RTX 4090 latency-ceiling path.

    The first call max-autotunes and captures the supplied CUDA allocation. Update
    that allocation in place for later requests; passing a different tensor raises.
    The default output is also borrowed until the next replay. Set
    ``copy_output=True`` only when logits must outlive that replay.
    """
    if sequence_length < 2 or batch_size < 1:
        message = "EFP16 ceiling inference requires sequence_length>=2 and batch_size>=1"
        raise ValueError(message)
    selected_tail_config = {
        (128, 64): (32, 4),
        (2048, 1): (16, 4),
    }.get((sequence_length, batch_size))
    use_fused_tail = False
    if selected_tail_config is not None and _supports_efp16_fused_tail(model):
        block_time, num_warps = selected_tail_config
        use_fused_tail = True
        model.use_fused_efp16_inference_tail = True
        model.use_fused_efp16_inference_readout = False
        model.efp16_inference_tail_block_time = block_time
        model.efp16_inference_tail_num_warps = num_warps
    else:
        model.use_fused_efp16_inference_tail = False
    use_n2048_b1_packed_scan = sequence_length == 2048 and batch_size == 1
    return prepare_efp16_manual_graph_inference(
        model,
        sequence_length=sequence_length,
        batch_size=batch_size,
        use_static_poles=True,
        compile_mode="max-autotune-no-cudagraphs",
        copy_output=copy_output,
        use_packed_block_scan=use_n2048_b1_packed_scan,
        use_single_warp_block_scan=use_n2048_b1_packed_scan,
        block_scan_block_size=64 if use_n2048_b1_packed_scan else 256,
        use_packed_static_recurrence_output=use_fused_tail,
    )


def _supports_efp16_fused_tail(model: HeadroomPACClassifier) -> bool:
    from .pac_headroom_efficient_models import EdgeFramePAC  # noqa: PLC0415

    if not isinstance(model, EdgeFramePAC):
        return False
    final_norm = model.final_norm
    classifier = getattr(model.head, "classifier", None)
    backward = model.backward_block
    return (
        model.model_dim == 32
        and model.modes == 16
        and getattr(model.spec, "name", None) == "B"
        and final_norm is not None
        and final_norm.weight is not None
        and isinstance(classifier, nn.Linear)
        and classifier.in_features == 192
        and classifier.out_features == 5
        and classifier.bias is not None
        and bool(getattr(model.head, "use_modal_moments", False))
        and bool(getattr(model.head, "use_backward_moments", False))
        and backward.independent_synthesis_frame is not None
        and backward.direct_scale is not None
        and backward.layer_scale is not None
        and backward.synthesis_scale == 1.0
        and not backward.split_residual_scales
        and backward.canonical_identity_elision
    )


def prepare_efp16_manual_graph_inference(
    model: HeadroomPACClassifier,
    *,
    sequence_length: int,
    batch_size: int,
    use_static_poles: bool,
    compile_mode: InferenceCompileMode = "default",
    copy_output: bool = False,
    use_packed_block_scan: bool = False,
    use_single_warp_block_scan: bool = False,
    block_scan_block_size: int = 256,
    use_packed_static_recurrence_output: bool = False,
) -> nn.Module:
    """Prepare same-address input/output manual CUDA Graph inference."""
    if sequence_length < 2 or batch_size < 1:
        message = "EFP16 manual graph inference requires sequence_length>=2 and batch_size>=1"
        raise ValueError(message)
    if block_scan_block_size not in {64, 128, 256}:
        message = "block_scan_block_size must be one of {64, 128, 256}"
        raise ValueError(message)
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
            use_static_pole_recurrence_moments=use_static_poles and not use_block_scan,
            use_packed_static_recurrence_moments=(
                (use_block_scan and use_packed_block_scan)
                or (not use_block_scan and use_packed_static_recurrence_output)
            ),
            use_packed_static_recurrence_drive=(
                use_block_scan and use_packed_block_scan
            ),
            use_single_warp_static_recurrence_moments=(
                use_block_scan and use_single_warp_block_scan
            ),
            use_static_pole_block_scan=(use_block_scan and use_packed_block_scan),
            static_pole_block_scan_block_size=block_scan_block_size,
        )
    return BorrowedInputCudaGraphInference(
        model,
        compile_mode=compile_mode,
        copy_output=copy_output,
    )


def _compile_inference_model(model: nn.Module, mode: InferenceCompileMode) -> nn.Module:
    if mode == "dynamic-no-cudagraph":
        compiled = torch.compile(
            model,
            fullgraph=True,
            dynamic=True,
            options={"triton.cudagraphs": False},
        )
    else:
        compile_mode = None if mode == "default" else mode
        compiled = torch.compile(
            model,
            fullgraph=True,
            dynamic=False,
            mode=compile_mode,
        )
    return cast("nn.Module", compiled)
