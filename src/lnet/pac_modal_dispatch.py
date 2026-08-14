from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .pac_real2d_math import modal_output_real2d
from .pac_recurrence import RecurrenceBackend, recurrence_real2d
from .pac_triton_modal import triton_modal_fused_output
from .pac_triton_modal_reduce import triton_modal_reduce_output

if TYPE_CHECKING:
    from torch import Tensor


@dataclass(frozen=True, slots=True)
class ModalDispatchInputs:
    decay_real: Tensor
    decay_imag: Tensor
    input_real: Tensor
    input_imag: Tensor
    writer_real: Tensor
    writer_imag: Tensor
    backend: RecurrenceBackend


def modal_real2d_output(args: ModalDispatchInputs) -> Tensor:
    match args.backend:
        case "triton_modal_fused" if args.decay_real.is_cuda:
            return triton_modal_fused_output(
                args.decay_real,
                args.decay_imag,
                args.input_real,
                args.input_imag,
                args.writer_real,
                args.writer_imag,
            )
        case "triton_modal_reduce" | "triton_modal_reduce_recompute" if args.decay_real.is_cuda:
            return triton_modal_reduce_output(
                args.decay_real,
                args.decay_imag,
                args.input_real,
                args.input_imag,
                args.writer_real,
                args.writer_imag,
                recompute_backward=args.backend == "triton_modal_reduce_recompute",
            )
        case _:
            states_real, states_imag = recurrence_real2d(
                args.decay_real,
                args.decay_imag,
                args.input_real,
                args.input_imag,
                args.backend,
            )
            return modal_output_real2d(states_real, states_imag, args.writer_real, args.writer_imag)
