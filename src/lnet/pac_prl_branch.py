from __future__ import annotations

from math import atanh, pi
from typing import assert_never

import torch
from torch import Tensor, nn
from torch.nn import functional

from .laplace import LaplaceParameterError, LaplaceShapeError
from .pac_fixed_path import fixed_real2d_branch_output
from .pac_modal_dispatch import ModalDispatchInputs, modal_real2d_output
from .pac_real2d_math import compiled_pole_gamma_from_control_real2d, discrete_pole_real2d
from .pac_recurrence import RecurrenceBackend, recurrence_states


def stable_expm1_over_p(poles: Tensor, dt: float, threshold: float = 1.0e-6) -> Tensor:
    scaled = poles * dt
    small = torch.abs(scaled) < threshold
    safe_poles = torch.where(small, torch.ones_like(poles), poles)
    raw = torch.expm1(scaled) / safe_poles
    return torch.where(small, torch.full_like(raw, complex(dt, 0.0)), raw)


class PACControlledTappedPRLBranch(nn.Module):
    def __init__(
        self,
        *,
        model_dim: int,
        modes: int,
        tap_kernel_size: int,
        dt: float = 1.0,
        min_decay: float = 1.0e-3,
        damping_control_range: float = 1.0,
        frequency_bound: float = pi,
        recurrence_backend: RecurrenceBackend = "auto",
    ) -> None:
        super().__init__()
        _require_positive(model_dim, "model_dim")
        _require_positive(modes, "modes")
        _require_positive(tap_kernel_size, "tap_kernel_size")
        self.model_dim = model_dim
        self.modes = modes
        self.tap_kernel_size = tap_kernel_size
        self.dt = dt
        self.min_decay = min_decay
        self.damping_control_range = damping_control_range
        self.frequency_bound = frequency_bound
        self.recurrence_backend: RecurrenceBackend = recurrence_backend
        initial_decay = torch.logspace(-1.3, 0.3, modes, dtype=torch.float32)
        frequency_grid = torch.linspace(0.0, 0.75 * frequency_bound, modes, dtype=torch.float32)
        self.raw_decay = nn.Parameter(torch.log(torch.expm1(initial_decay)))
        self.raw_frequency = nn.Parameter(_inverse_tanh_grid(frequency_grid / frequency_bound))
        self.reader = nn.Parameter(0.02 * torch.randn(modes, model_dim, dtype=torch.float32))
        self.tap_logits = nn.Parameter(torch.zeros(modes, tap_kernel_size, dtype=torch.float32))
        self.writer_real = nn.Parameter(0.02 * torch.randn(modes, model_dim, dtype=torch.float32))
        self.writer_imag = nn.Parameter(0.02 * torch.randn(modes, model_dim, dtype=torch.float32))
        self.direct_term = nn.Parameter(torch.zeros(model_dim, model_dim, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(model_dim, dtype=torch.float32))
        self.damping_control = nn.Linear(model_dim, modes)
        nn.init.zeros_(self.damping_control.weight)
        nn.init.zeros_(self.damping_control.bias)

    def base_damping_values(self) -> Tensor:
        return self.min_decay + torch.nn.functional.softplus(self.raw_decay)

    def frequency_values(self) -> Tensor:
        return self.frequency_bound * torch.tanh(self.raw_frequency)

    def continuous_poles(self) -> Tensor:
        return torch.complex(-self.base_damping_values(), self.frequency_values())

    def damping_control_values(self, projected: Tensor) -> Tensor:
        _check_projected(projected, self.model_dim)
        if self.damping_control_range == 0.0:
            return torch.zeros(
                projected.shape[0],
                projected.shape[1],
                self.modes,
                device=projected.device,
                dtype=projected.dtype,
            )
        control_input = projected.to(dtype=self.damping_control.weight.dtype)
        return self.damping_control_range * torch.tanh(self.damping_control(control_input))

    def effective_damping_values(self, projected: Tensor) -> Tensor:
        control = self.damping_control_values(projected)
        raw = self.raw_decay.to(device=projected.device, dtype=control.dtype).view(1, 1, self.modes)
        return self.min_decay + torch.nn.functional.softplus(raw + control)

    def effective_discrete_decay(self, projected: Tensor) -> Tensor:
        return torch.exp(self._effective_poles(projected) * self.dt)

    def effective_tap_weights(self) -> Tensor:
        return torch.softmax(self.tap_logits, dim=-1)

    def tapped_drive_reference(self, instant_drive: Tensor, time_index: int) -> Tensor:
        return self._tapped_drive(instant_drive, time_index)

    def tapped_drive_sequence(self, instant_drive: Tensor) -> Tensor:
        return self._tapped_drive_sequence(instant_drive)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.forward_with_backend(inputs, self.recurrence_backend)

    def forward_with_backend(self, inputs: Tensor, backend: RecurrenceBackend) -> Tensor:
        match backend:
            case (
                "real2d_e2e"
                | "triton_scan_blocks"
                | "triton_modal_fused"
                | "triton_modal_reduce"
                | "triton_modal_reduce_recompute"
                | "pac_lite_fast"
                | "fixed_real2d_fast"
                | "fused_pole_gamma"
            ):
                return self._forward_real2d(inputs, backend)
            case (
                "complex_loop"
                | "real2d_loop"
                | "compiled_real2d"
                | "triton_fused"
                | "triton_scan"
                | "auto"
            ):
                return self._forward_complex_compat(inputs, backend)
            case unreachable:
                assert_never(unreachable)

    def _forward_complex_compat(self, inputs: Tensor, backend: RecurrenceBackend) -> Tensor:
        _check_projected(inputs, self.model_dim)
        inputs_work = inputs.to(dtype=self.reader.dtype)
        poles = self._effective_poles(inputs_work)
        decay = torch.exp(poles * self.dt)
        drive_scale = stable_expm1_over_p(poles, self.dt)
        instant_drive = torch.einsum("bnd,md->bnm", inputs_work, self.reader)
        tapped_drive = self._tapped_drive_sequence(instant_drive).to(dtype=drive_scale.dtype)
        states = recurrence_states(decay, drive_scale * tapped_drive, backend)
        writer = torch.complex(self.writer_real, self.writer_imag)
        modal = 2.0 * torch.einsum("bnm,md->bnd", states, writer).real
        direct = torch.matmul(inputs_work, self.direct_term.transpose(0, 1)) + self.bias
        return (modal + direct).to(dtype=inputs.dtype)

    def _forward_real2d(self, inputs: Tensor, backend: RecurrenceBackend) -> Tensor:
        _check_projected(inputs, self.model_dim)
        if backend == "fixed_real2d_fast" and self.damping_control_range == 0.0:
            return fixed_real2d_branch_output(self, inputs)
        inputs_work = inputs.to(dtype=self.reader.dtype)
        if backend == "fused_pole_gamma":
            control = self.damping_control_values(inputs_work)
            frequency = self.frequency_values().to(
                device=inputs_work.device, dtype=inputs_work.dtype
            )
            decay_real, decay_imag, gamma_real, gamma_imag = (
                compiled_pole_gamma_from_control_real2d(
                    self.raw_decay, frequency, control, self.min_decay, self.dt
                )
            )
        else:
            damping = self.effective_damping_values(inputs_work)
            frequency = (
                self.frequency_values()
                .to(device=inputs_work.device, dtype=damping.dtype)
                .view(1, 1, self.modes)
                .expand_as(damping)
            )
            decay_real, decay_imag, gamma_real, gamma_imag = discrete_pole_real2d(
                damping, frequency, self.dt
            )
        instant_drive = torch.einsum("bnd,md->bnm", inputs_work, self.reader)
        tapped_drive = self._tapped_drive_sequence(instant_drive)
        input_real = gamma_real * tapped_drive
        input_imag = gamma_imag * tapped_drive
        modal = self._modal_real2d(
            decay_real,
            decay_imag,
            input_real,
            input_imag,
            backend,
        )
        direct = torch.matmul(inputs_work, self.direct_term.transpose(0, 1)) + self.bias
        return (modal + direct).to(dtype=inputs.dtype)

    def _modal_real2d(
        self,
        decay_real: Tensor,
        decay_imag: Tensor,
        input_real: Tensor,
        input_imag: Tensor,
        backend: RecurrenceBackend,
    ) -> Tensor:
        return modal_real2d_output(
            ModalDispatchInputs(
                decay_real=decay_real,
                decay_imag=decay_imag,
                input_real=input_real,
                input_imag=input_imag,
                writer_real=self.writer_real,
                writer_imag=self.writer_imag,
                backend=backend,
            )
        )

    def forward_reference(self, inputs: Tensor) -> Tensor:
        _check_projected(inputs, self.model_dim)
        inputs_work = inputs.to(dtype=self.reader.dtype)
        poles = self._effective_poles(inputs_work)
        decay = torch.exp(poles * self.dt)
        drive_scale = stable_expm1_over_p(poles, self.dt)
        instant_drive = torch.einsum("bnd,md->bnm", inputs_work, self.reader)
        writer = torch.complex(self.writer_real, self.writer_imag)
        state = torch.zeros(
            inputs.shape[0], self.modes, dtype=drive_scale.dtype, device=inputs.device
        )
        outputs: list[Tensor] = []
        for time_index, current_input in enumerate(inputs_work.unbind(dim=1)):
            tapped_drive = self._tapped_drive(instant_drive, time_index)
            state = decay[:, time_index, :] * state + drive_scale[:, time_index, :] * tapped_drive
            modal = 2.0 * torch.einsum("bm,md->bd", state, writer).real
            direct = torch.matmul(current_input, self.direct_term.transpose(0, 1))
            outputs.append(modal + direct + self.bias)
        return torch.stack(outputs, dim=1).to(dtype=inputs.dtype)

    def _effective_poles(self, projected: Tensor) -> Tensor:
        damping = self.effective_damping_values(projected)
        frequency = (
            self.frequency_values()
            .to(device=projected.device, dtype=damping.dtype)
            .view(1, 1, self.modes)
        )
        return torch.complex(-damping, frequency.expand_as(damping))

    def _tapped_drive(self, instant_drive: Tensor, time_index: int) -> Tensor:
        tap_count = min(time_index + 1, self.tap_kernel_size)
        start = time_index - tap_count + 1
        recent = torch.flip(instant_drive[:, start : time_index + 1, :], dims=(1,))
        weights = self.effective_tap_weights()[:, :tap_count].transpose(0, 1)
        normalized = weights / weights.sum(dim=0, keepdim=True).clamp_min(1.0e-12)
        return torch.sum(recent * normalized.view(1, tap_count, self.modes), dim=1)

    def _tapped_drive_sequence(self, instant_drive: Tensor) -> Tensor:
        _check_drive(instant_drive, self.modes)
        drive_by_mode = instant_drive.transpose(1, 2)
        weights = self.effective_tap_weights().to(
            device=instant_drive.device, dtype=instant_drive.dtype
        )
        kernel = torch.flip(weights, dims=(-1,)).view(self.modes, 1, self.tap_kernel_size)
        padded = functional.pad(drive_by_mode, (self.tap_kernel_size - 1, 0))
        numerator = functional.conv1d(padded, kernel, groups=self.modes)
        mask = torch.ones(
            1,
            self.modes,
            instant_drive.shape[1],
            device=instant_drive.device,
            dtype=instant_drive.dtype,
        )
        denominator = functional.conv1d(
            functional.pad(mask, (self.tap_kernel_size - 1, 0)),
            kernel,
            groups=self.modes,
        )
        return (numerator / denominator.clamp_min(1.0e-12)).transpose(1, 2)


def _inverse_tanh_grid(values: Tensor) -> Tensor:
    return torch.tensor(
        [atanh(float(value.clamp(-0.95, 0.95))) for value in values], dtype=torch.float32
    )


def _check_projected(inputs: Tensor, model_dim: int) -> None:
    if inputs.ndim != 3 or inputs.shape[-1] != model_dim:
        raise LaplaceShapeError(tuple(inputs.shape), 3, model_dim)


def _check_drive(inputs: Tensor, modes: int) -> None:
    if inputs.ndim != 3 or inputs.shape[-1] != modes:
        raise LaplaceShapeError(tuple(inputs.shape), 3, modes)


def _require_positive(value: int, name: str) -> None:
    if value <= 0:
        raise LaplaceParameterError(reason=f"{name} must be positive")
