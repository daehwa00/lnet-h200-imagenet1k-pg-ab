from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

from lnet.laplace import LaplaceShapeError

TapParameterization = Literal[
    "shared_scalar",
    "tap_specific_reader",
    "low_rank_reader",
    "normalized_taps",
]


class TappedPRLBranch(nn.Module):
    def __init__(
        self,
        *,
        model_dim: int,
        modes: int,
        tap_kernel_size: int,
        tap_parameterization: TapParameterization = "shared_scalar",
        low_rank_rank: int = 2,
        dt: float = 1.0,
        min_decay: float = 1.0e-3,
    ) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.modes = modes
        self.tap_kernel_size = tap_kernel_size
        self.tap_parameterization = tap_parameterization
        self.low_rank_rank = low_rank_rank
        self.dt = dt
        self.min_decay = min_decay

        initial_decay = torch.linspace(0.2, 1.0, modes, dtype=torch.float64)
        initial_frequency = torch.linspace(0.0, 1.5, modes, dtype=torch.float64)
        initial_taps = torch.zeros(modes, tap_kernel_size, dtype=torch.float64)
        initial_taps[:, 0] = 1.0
        self.raw_decay = nn.Parameter(torch.log(torch.expm1(initial_decay)))
        self.frequency = nn.Parameter(initial_frequency)
        self.tap_weights: nn.Parameter | None = None
        self.tap_logits: nn.Parameter | None = None
        self.input_residue_real: nn.Parameter | None = None
        self.input_residue_imag: nn.Parameter | None = None
        self.tap_specific_reader_real: nn.Parameter | None = None
        self.tap_specific_reader_imag: nn.Parameter | None = None
        self.low_rank_basis: nn.Parameter | None = None
        self.low_rank_coefficients: nn.Parameter | None = None
        match tap_parameterization:
            case "shared_scalar":
                self.tap_weights = nn.Parameter(initial_taps)
                self.input_residue_real = nn.Parameter(
                    0.05 * torch.randn(modes, model_dim, dtype=torch.float64),
                )
                self.input_residue_imag = nn.Parameter(
                    0.05 * torch.randn(modes, model_dim, dtype=torch.float64),
                )
            case "normalized_taps":
                self.tap_logits = nn.Parameter(initial_taps)
                self.input_residue_real = nn.Parameter(
                    0.05 * torch.randn(modes, model_dim, dtype=torch.float64),
                )
                self.input_residue_imag = nn.Parameter(
                    0.05 * torch.randn(modes, model_dim, dtype=torch.float64),
                )
            case "tap_specific_reader":
                self.tap_specific_reader_real = nn.Parameter(
                    0.05 * torch.randn(modes, tap_kernel_size, model_dim, dtype=torch.float64),
                )
                self.tap_specific_reader_imag = nn.Parameter(
                    0.05 * torch.randn(modes, tap_kernel_size, model_dim, dtype=torch.float64),
                )
            case "low_rank_reader":
                self.low_rank_basis = nn.Parameter(
                    0.05 * torch.randn(low_rank_rank, model_dim, dtype=torch.float64),
                )
                low_rank_coefficients = torch.zeros(
                    modes,
                    tap_kernel_size,
                    low_rank_rank,
                    dtype=torch.float64,
                )
                low_rank_coefficients[:, 0, 0] = 1.0
                self.low_rank_coefficients = nn.Parameter(low_rank_coefficients)
            case _:
                raise _tap_parameterization_error(tap_parameterization)
        self.output_residue_real = nn.Parameter(
            0.05 * torch.randn(modes, model_dim, dtype=torch.float64),
        )
        self.output_residue_imag = nn.Parameter(
            0.05 * torch.randn(modes, model_dim, dtype=torch.float64),
        )
        self.direct_term = nn.Parameter(torch.zeros(model_dim, model_dim, dtype=torch.float64))
        self.bias = nn.Parameter(torch.zeros(model_dim, dtype=torch.float64))

    def continuous_poles(self) -> Tensor:
        real_part = -(torch.nn.functional.softplus(self.raw_decay) + self.min_decay)
        return torch.complex(real_part, self.frequency)

    def effective_tap_weights(self) -> Tensor:
        match self.tap_parameterization:
            case "shared_scalar":
                if self.tap_weights is None:
                    message = "shared_scalar taps are not initialized"
                    raise _state_error(message)
                return self.tap_weights
            case "normalized_taps":
                if self.tap_logits is None:
                    message = "normalized tap logits are not initialized"
                    raise _state_error(message)
                return torch.softmax(self.tap_logits, dim=-1)
            case "tap_specific_reader":
                if self.tap_specific_reader_real is None or self.tap_specific_reader_imag is None:
                    message = "tap-specific readers are not initialized"
                    raise _state_error(message)
                reader_norm = torch.sqrt(
                    self.tap_specific_reader_real.square() + self.tap_specific_reader_imag.square(),
                )
                return torch.linalg.vector_norm(reader_norm, dim=-1)
            case "low_rank_reader":
                if self.low_rank_basis is None or self.low_rank_coefficients is None:
                    message = "low-rank readers are not initialized"
                    raise _state_error(message)
                basis_norm = torch.linalg.vector_norm(self.low_rank_basis, dim=-1)
                return torch.sum(
                    torch.abs(self.low_rank_coefficients) * basis_norm.view(1, 1, -1),
                    dim=-1,
                )
            case _:
                raise _tap_parameterization_error(self.tap_parameterization)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 3:
            raise LaplaceShapeError(
                actual_shape=tuple(inputs.shape),
                expected_rank=3,
                expected_features=self.model_dim,
            )
        if inputs.shape[-1] != self.model_dim:
            raise LaplaceShapeError(
                actual_shape=tuple(inputs.shape),
                expected_rank=3,
                expected_features=self.model_dim,
            )

        inputs_64 = inputs.to(dtype=torch.float64)
        poles = self.continuous_poles()
        discrete_decay = torch.exp(poles * self.dt).view(1, self.modes)
        discrete_drive = (torch.expm1(poles * self.dt) / poles).view(1, self.modes)
        output_residue = torch.complex(self.output_residue_real, self.output_residue_imag)
        instant_drive: Tensor | None = None
        low_rank_drive: Tensor | None = None
        match self.tap_parameterization:
            case "shared_scalar" | "normalized_taps":
                if self.input_residue_real is None or self.input_residue_imag is None:
                    message = "shared tap readers are not initialized"
                    raise _state_error(message)
                input_residue = torch.complex(self.input_residue_real, self.input_residue_imag)
                instant_drive = self._instant_drive(inputs_64, input_residue)
            case "tap_specific_reader":
                pass
            case "low_rank_reader":
                if self.low_rank_basis is None:
                    message = "low-rank basis is not initialized"
                    raise _state_error(message)
                low_rank_drive = torch.einsum("bnd,rd->bnr", inputs_64, self.low_rank_basis)
            case _:
                raise _tap_parameterization_error(self.tap_parameterization)
        state = torch.zeros(
            inputs_64.shape[0],
            self.modes,
            dtype=torch.complex128,
            device=inputs_64.device,
        )
        outputs: list[Tensor] = []
        for time_index, current_input in enumerate(inputs_64.unbind(dim=1)):
            tapped_drive = self._tapped_drive(
                inputs=inputs_64,
                instant_drive=instant_drive,
                low_rank_drive=low_rank_drive,
                time_index=time_index,
            )
            state = (discrete_decay * state) + (discrete_drive * tapped_drive)
            modal_output = 2.0 * torch.einsum("bm,md->bd", state, output_residue).real
            direct_output = torch.matmul(current_input, self.direct_term.transpose(0, 1))
            outputs.append(modal_output + direct_output + self.bias)
        return torch.stack(outputs, dim=1).to(dtype=inputs.dtype)

    def _instant_drive(self, inputs: Tensor, input_residue: Tensor) -> Tensor:
        drive_real = torch.einsum("bnd,md->bnm", inputs, input_residue.real)
        drive_imag = torch.einsum("bnd,md->bnm", inputs, input_residue.imag)
        return torch.complex(drive_real, drive_imag)

    def _tapped_drive(
        self,
        *,
        inputs: Tensor,
        instant_drive: Tensor | None,
        low_rank_drive: Tensor | None,
        time_index: int,
    ) -> Tensor:
        tap_count = min(time_index + 1, self.tap_kernel_size)
        start_index = time_index - tap_count + 1
        match self.tap_parameterization:
            case "shared_scalar" | "normalized_taps":
                if instant_drive is None:
                    message = "shared instant drive is missing"
                    raise _state_error(message)
                recent_drive = torch.flip(
                    instant_drive[:, start_index : time_index + 1, :],
                    dims=(1,),
                )
                weights = self.effective_tap_weights()[:, :tap_count].transpose(0, 1)
                return torch.sum(recent_drive * weights.view(1, tap_count, self.modes), dim=1)
            case "tap_specific_reader":
                if self.tap_specific_reader_real is None or self.tap_specific_reader_imag is None:
                    message = "tap-specific readers are missing"
                    raise _state_error(message)
                tapped_drive = torch.zeros(
                    inputs.shape[0],
                    self.modes,
                    dtype=torch.complex128,
                    device=inputs.device,
                )
                for tap_index in range(tap_count):
                    delayed_input = inputs[:, time_index - tap_index, :]
                    reader_real = self.tap_specific_reader_real[:, tap_index, :]
                    reader_imag = self.tap_specific_reader_imag[:, tap_index, :]
                    drive_real = torch.einsum("bd,md->bm", delayed_input, reader_real)
                    drive_imag = torch.einsum("bd,md->bm", delayed_input, reader_imag)
                    tapped_drive = tapped_drive + torch.complex(drive_real, drive_imag)
                return tapped_drive
            case "low_rank_reader":
                if low_rank_drive is None or self.low_rank_coefficients is None:
                    message = "low-rank readers are missing"
                    raise _state_error(message)
                recent_drive = torch.flip(
                    low_rank_drive[:, start_index : time_index + 1, :],
                    dims=(1,),
                )
                coefficients = self.low_rank_coefficients[:, :tap_count, :]
                drive = torch.einsum("btr,mtr->bm", recent_drive, coefficients)
                return torch.complex(drive, torch.zeros_like(drive))
            case _:
                raise _tap_parameterization_error(self.tap_parameterization)


def _state_error(message: str) -> RuntimeError:
    return RuntimeError(message)


def _tap_parameterization_error(name: str) -> RuntimeError:
    message = f"unsupported tap parameterization: {name}"
    return RuntimeError(message)
