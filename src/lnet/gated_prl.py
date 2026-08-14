from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

from lnet.laplace import PoleResidueTemporalMixing, ProjectedPRLBlock

GateVariant = Literal["fixed", "input", "output", "input_output"]


class GatedPRLBlock(nn.Module):
    def __init__(
        self,
        *,
        raw_input_dim: int,
        model_dim: int,
        output_dim: int,
        modes: int,
        gate_variant: GateVariant,
        dt: float = 1.0,
    ) -> None:
        super().__init__()
        self.raw_input_dim = raw_input_dim
        self.model_dim = model_dim
        self.output_dim = output_dim
        self.modes = modes
        self.gate_variant = gate_variant
        self.input_projection = nn.Linear(raw_input_dim, model_dim)
        self.temporal_mixer = PoleResidueTemporalMixing(model_dim, modes, dt=dt)
        self.activation = nn.Tanh()
        self.output_projection = nn.Linear(model_dim, output_dim)
        self.input_gate = (
            nn.Linear(model_dim, modes) if gate_variant in {"input", "input_output"} else None
        )
        self.output_gate = (
            nn.Linear(model_dim, modes) if gate_variant in {"output", "input_output"} else None
        )

    @classmethod
    def from_projected_prl(
        cls,
        block: ProjectedPRLBlock,
        *,
        gate_variant: GateVariant,
    ) -> GatedPRLBlock:
        gated = cls(
            raw_input_dim=block.raw_input_dim,
            model_dim=block.model_dim,
            output_dim=block.output_dim,
            modes=block.modes,
            gate_variant=gate_variant,
            dt=block.temporal_mixer.dt,
        )
        with torch.no_grad():
            gated.input_projection.weight.copy_(block.input_projection.weight)
            gated.input_projection.bias.copy_(block.input_projection.bias)
            gated.temporal_mixer.raw_decay.copy_(block.temporal_mixer.raw_decay)
            gated.temporal_mixer.residues.copy_(block.temporal_mixer.residues)
            gated.temporal_mixer.direct_term.copy_(block.temporal_mixer.direct_term)
            gated.temporal_mixer.bias.copy_(block.temporal_mixer.bias)
            gated.output_projection.weight.copy_(block.output_projection.weight)
            gated.output_projection.bias.copy_(block.output_projection.bias)
            if gated.input_gate is not None:
                gated.input_gate.weight.zero_()
                gated.input_gate.bias.zero_()
            if gated.output_gate is not None:
                gated.output_gate.weight.zero_()
                gated.output_gate.bias.zero_()
        return gated

    def forward(self, inputs: Tensor) -> Tensor:
        projected = self.input_projection(inputs)
        inputs_64 = projected.to(dtype=torch.float64)
        poles = self.temporal_mixer.continuous_poles()
        scaled_poles = poles * self.temporal_mixer.dt
        discrete_decay = torch.exp(scaled_poles).view(1, self.modes, 1)
        discrete_drive = (
            self.temporal_mixer.dt
            * torch.where(
                scaled_poles.abs() < 1.0e-4,
                1.0 + (scaled_poles / 2.0) + (scaled_poles.square() / 6.0),
                torch.expm1(scaled_poles) / scaled_poles,
            )
        ).view(1, self.modes, 1)
        state = torch.zeros(
            inputs_64.shape[0],
            self.modes,
            self.model_dim,
            dtype=torch.float64,
            device=inputs_64.device,
        )
        outputs: list[Tensor] = []
        for step_input, step_input_64 in zip(
            projected.unbind(dim=1),
            inputs_64.unbind(dim=1),
            strict=True,
        ):
            if self.input_gate is None:
                gated_drive = step_input_64.unsqueeze(1)
            else:
                input_gate_layer = self.input_gate
                if input_gate_layer is None:
                    message = "input gate layer missing"
                    raise RuntimeError(message)
                input_gate = torch.sigmoid(input_gate_layer(step_input)).to(dtype=torch.float64)
                gated_drive = input_gate.unsqueeze(-1) * step_input_64.unsqueeze(1)
            state = (discrete_decay * state) + (discrete_drive * gated_drive)
            if self.output_gate is None:
                effective_state = state
            else:
                output_gate_layer = self.output_gate
                if output_gate_layer is None:
                    message = "output gate layer missing"
                    raise RuntimeError(message)
                output_gate = torch.sigmoid(output_gate_layer(step_input)).to(
                    dtype=torch.float64,
                )
                effective_state = output_gate.unsqueeze(-1) * state
            modal_output = torch.einsum(
                "omi,bmi->bo",
                self.temporal_mixer.residues,
                effective_state,
            )
            direct_output = torch.matmul(
                step_input_64,
                self.temporal_mixer.direct_term.transpose(0, 1),
            )
            mixed = modal_output + direct_output + self.temporal_mixer.bias
            outputs.append(self.output_projection(self.activation(mixed.to(dtype=projected.dtype))))
        return torch.stack(outputs, dim=1)
