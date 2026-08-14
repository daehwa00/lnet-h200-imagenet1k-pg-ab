from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor

from .pac_real2d_math import discrete_pole_real2d, modal_output_real2d
from .pac_recurrence import real2d_loop_recurrence
from .pac_triton_fixed import triton_fixed_recurrence


class FixedReal2DBranch(Protocol):
    @property
    def modes(self) -> int: ...

    @property
    def dt(self) -> float: ...

    @property
    def reader(self) -> Tensor: ...

    @property
    def writer_real(self) -> Tensor: ...

    @property
    def writer_imag(self) -> Tensor: ...

    @property
    def direct_term(self) -> Tensor: ...

    @property
    def bias(self) -> Tensor: ...

    def base_damping_values(self) -> Tensor: ...

    def frequency_values(self) -> Tensor: ...

    def tapped_drive_sequence(self, instant_drive: Tensor) -> Tensor: ...


def fixed_real2d_branch_output(branch: FixedReal2DBranch, inputs: Tensor) -> Tensor:
    inputs_work = inputs.to(dtype=branch.reader.dtype)
    damping = branch.base_damping_values().to(device=inputs_work.device, dtype=inputs_work.dtype)
    frequency = branch.frequency_values().to(device=inputs_work.device, dtype=inputs_work.dtype)
    decay_real, decay_imag, gamma_real, gamma_imag = discrete_pole_real2d(
        damping, frequency, branch.dt
    )
    instant_drive = torch.einsum("bnd,md->bnm", inputs_work, branch.reader)
    tapped_drive = branch.tapped_drive_sequence(instant_drive)
    input_real = gamma_real.view(1, 1, branch.modes) * tapped_drive
    input_imag = gamma_imag.view(1, 1, branch.modes) * tapped_drive
    if inputs_work.is_cuda:
        states_real, states_imag = triton_fixed_recurrence(
            decay_real, decay_imag, input_real, input_imag
        )
    else:
        states_real, states_imag = real2d_loop_recurrence(
            decay_real.view(1, 1, branch.modes).expand_as(input_real),
            decay_imag.view(1, 1, branch.modes).expand_as(input_imag),
            input_real,
            input_imag,
        )
    modal = modal_output_real2d(states_real, states_imag, branch.writer_real, branch.writer_imag)
    direct = torch.matmul(inputs_work, branch.direct_term.transpose(0, 1)) + branch.bias
    return (modal + direct).to(dtype=inputs.dtype)
