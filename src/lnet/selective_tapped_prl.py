from __future__ import annotations

import torch
from torch import Tensor, nn

from .laplace import LaplaceParameterError, LaplaceShapeError
from .selective_tapped_prl_variants import (
    SELECTIVE_VARIANTS,
    SelectiveVariant,
    check_projected,
    require_positive,
    uses_damping_modulation,
    uses_input_gate,
    uses_read_gate,
    uses_tap_selectivity,
)

__all__ = ["SELECTIVE_VARIANTS", "SelectiveTappedPRLBlock", "SelectiveVariant"]


class SelectiveTappedPRLBlock(nn.Module):
    def __init__(
        self,
        *,
        raw_input_dim: int,
        model_dim: int,
        output_dim: int,
        modes: int,
        tap_kernel_size: int,
        variant: SelectiveVariant = "full",
        dt: float = 1.0,
        min_decay: float = 1.0e-3,
        damping_beta: float = 0.5,
    ) -> None:
        super().__init__()
        require_positive(raw_input_dim, "raw_input_dim")
        require_positive(model_dim, "model_dim")
        require_positive(output_dim, "output_dim")
        require_positive(modes, "modes")
        require_positive(tap_kernel_size, "tap_kernel_size")
        self.raw_input_dim = raw_input_dim
        self.model_dim = model_dim
        self.output_dim = output_dim
        self.modes = modes
        self.tap_kernel_size = tap_kernel_size
        self.variant = variant
        self.dt = dt
        self.min_decay = min_decay
        self.damping_beta = damping_beta
        if damping_beta < 0.0:
            raise LaplaceParameterError(reason="damping_beta must be nonnegative")

        self.input_projection = nn.Linear(raw_input_dim, model_dim)
        initial_decay = torch.linspace(0.2, 1.0, modes, dtype=torch.float64)
        initial_frequency = torch.linspace(0.0, 1.5, modes, dtype=torch.float64)
        self.raw_decay = nn.Parameter(torch.log(torch.expm1(initial_decay)))
        self.frequency = nn.Parameter(initial_frequency)
        self.input_residue_real = nn.Parameter(
            0.05 * torch.randn(modes, model_dim, dtype=torch.float64),
        )
        self.input_residue_imag = nn.Parameter(
            0.05 * torch.randn(modes, model_dim, dtype=torch.float64),
        )
        self.output_residue_real = nn.Parameter(
            0.05 * torch.randn(modes, model_dim, dtype=torch.float64),
        )
        self.output_residue_imag = nn.Parameter(
            0.05 * torch.randn(modes, model_dim, dtype=torch.float64),
        )
        initial_taps = torch.full((modes, tap_kernel_size), -4.0, dtype=torch.float64)
        initial_taps[:, 0] = 4.0
        self.fixed_tap_logits: nn.Parameter | None = None
        if not uses_tap_selectivity(variant):
            self.fixed_tap_logits = nn.Parameter(initial_taps)
        self.direct_term = nn.Parameter(torch.zeros(model_dim, model_dim, dtype=torch.float64))
        self.bias = nn.Parameter(torch.zeros(model_dim, dtype=torch.float64))
        self.input_gate = nn.Linear(model_dim, modes) if uses_input_gate(variant) else None
        self.tap_selector = (
            nn.Linear(model_dim, modes * tap_kernel_size) if uses_tap_selectivity(variant) else None
        )
        self.read_gate = nn.Linear(model_dim, modes) if uses_read_gate(variant) else None
        self.damping_controller = (
            nn.Linear(model_dim, modes) if uses_damping_modulation(variant) else None
        )
        self.activation = nn.GELU()
        self.output_projection = nn.Linear(model_dim, model_dim)
        self.readout_projection = nn.Linear(model_dim, output_dim)

    def continuous_poles(self) -> Tensor:
        real_part = -self.base_damping_values()
        return torch.complex(real_part, self.frequency)

    def base_damping_values(self) -> Tensor:
        return torch.nn.functional.softplus(self.raw_decay) + self.min_decay

    def effective_damping_values(self, projected: Tensor) -> Tensor:
        check_projected(projected, self.model_dim)
        decay_scale = torch.nn.functional.softplus(self.raw_decay).to(
            device=projected.device,
            dtype=torch.float64,
        )
        base = decay_scale.view(1, 1, self.modes)
        if self.damping_controller is None or self.damping_beta == 0.0:
            modulation = torch.ones(
                projected.shape[0],
                projected.shape[1],
                self.modes,
                device=projected.device,
                dtype=torch.float64,
            )
        else:
            control = self.damping_control_values(projected).to(dtype=torch.float64)
            modulation = torch.exp(self.damping_beta * control)
        return self.min_decay + (base * modulation)

    def damping_control_values(self, projected: Tensor) -> Tensor:
        check_projected(projected, self.model_dim)
        if self.damping_controller is None or self.damping_beta == 0.0:
            return torch.zeros(
                projected.shape[0],
                projected.shape[1],
                self.modes,
                device=projected.device,
                dtype=projected.dtype,
            )
        return torch.tanh(self.damping_controller(projected))

    def effective_discrete_decay(self, projected: Tensor) -> Tensor:
        return torch.exp(self._effective_poles(projected) * self.dt)

    def effective_fixed_taps(self) -> Tensor:
        if self.fixed_tap_logits is None:
            message = "fixed taps are not initialized for tap-selective variants"
            raise RuntimeError(message)
        return torch.softmax(self.fixed_tap_logits, dim=-1)

    def input_gate_values(self, projected: Tensor) -> Tensor:
        check_projected(projected, self.model_dim)
        if self.input_gate is None:
            return torch.ones(
                projected.shape[0],
                projected.shape[1],
                self.modes,
                device=projected.device,
                dtype=projected.dtype,
            )
        return torch.sigmoid(self.input_gate(projected))

    def read_gate_values(self, projected: Tensor) -> Tensor:
        check_projected(projected, self.model_dim)
        if self.read_gate is None:
            return torch.ones(
                projected.shape[0],
                projected.shape[1],
                self.modes,
                device=projected.device,
                dtype=projected.dtype,
            )
        return torch.sigmoid(self.read_gate(projected))

    def tap_selection_values(self, projected: Tensor) -> Tensor:
        check_projected(projected, self.model_dim)
        if self.tap_selector is None:
            taps = self.effective_fixed_taps().to(device=projected.device, dtype=projected.dtype)
            return taps.view(1, 1, self.modes, self.tap_kernel_size).expand(
                projected.shape[0],
                projected.shape[1],
                -1,
                -1,
            )
        logits = self.tap_selector(projected).view(
            projected.shape[0],
            projected.shape[1],
            self.modes,
            self.tap_kernel_size,
        )
        return torch.softmax(logits, dim=-1)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 3:
            raise LaplaceShapeError(
                actual_shape=tuple(inputs.shape),
                expected_rank=3,
                expected_features=self.raw_input_dim,
            )
        if inputs.shape[-1] != self.raw_input_dim:
            raise LaplaceShapeError(
                actual_shape=tuple(inputs.shape),
                expected_rank=3,
                expected_features=self.raw_input_dim,
            )
        projected = self.input_projection(inputs)
        temporal = self._temporal_forward(projected)
        residual = projected + self.output_projection(self.activation(temporal))
        return self.readout_projection(residual)

    def _temporal_forward(self, projected: Tensor) -> Tensor:
        projected_64 = projected.to(dtype=torch.float64)
        effective_poles = self._effective_poles(projected)
        discrete_decay = torch.exp(effective_poles * self.dt)
        discrete_drive = torch.expm1(effective_poles * self.dt) / effective_poles
        input_residue = torch.complex(self.input_residue_real, self.input_residue_imag)
        output_residue = torch.complex(self.output_residue_real, self.output_residue_imag)
        instant_drive = self._instant_drive(projected_64, input_residue)
        input_gate = self.input_gate_values(projected).to(dtype=torch.float64)
        read_gate = self.read_gate_values(projected).to(dtype=torch.float64)
        tap_selection = self.tap_selection_values(projected).to(dtype=torch.float64)
        state = torch.zeros(
            projected.shape[0],
            self.modes,
            dtype=torch.complex128,
            device=projected.device,
        )
        outputs: list[Tensor] = []
        for time_index, current_input in enumerate(projected_64.unbind(dim=1)):
            drive = self._tapped_drive(
                instant_drive=instant_drive,
                tap_selection=tap_selection,
                time_index=time_index,
            )
            gated_drive = input_gate[:, time_index, :] * drive
            state = (discrete_decay[:, time_index, :] * state) + (
                discrete_drive[:, time_index, :] * gated_drive
            )
            gated_state = read_gate[:, time_index, :] * state
            modal_output = 2.0 * torch.einsum("bm,md->bd", gated_state, output_residue).real
            direct_output = torch.matmul(current_input, self.direct_term.transpose(0, 1))
            outputs.append(modal_output + direct_output + self.bias)
        return torch.stack(outputs, dim=1).to(dtype=projected.dtype)

    def _instant_drive(self, projected: Tensor, input_residue: Tensor) -> Tensor:
        drive_real = torch.einsum("bnd,md->bnm", projected, input_residue.real)
        drive_imag = torch.einsum("bnd,md->bnm", projected, input_residue.imag)
        return torch.complex(drive_real, drive_imag)

    def _tapped_drive(
        self,
        *,
        instant_drive: Tensor,
        tap_selection: Tensor,
        time_index: int,
    ) -> Tensor:
        tap_count = min(time_index + 1, self.tap_kernel_size)
        start_index = time_index - tap_count + 1
        recent_drive = torch.flip(
            instant_drive[:, start_index : time_index + 1, :],
            dims=(1,),
        )
        weights = tap_selection[:, time_index, :, :tap_count]
        normalized = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
        return torch.sum(recent_drive * normalized.transpose(1, 2), dim=1)

    def _effective_poles(self, projected: Tensor) -> Tensor:
        damping = self.effective_damping_values(projected)
        frequency = self.frequency.to(device=projected.device, dtype=torch.float64).view(
            1,
            1,
            self.modes,
        )
        return torch.complex(-damping, frequency.expand_as(damping))
