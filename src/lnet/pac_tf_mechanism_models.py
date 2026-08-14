from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from .pac_builders import build_regression_model
from .pac_paper_queue_models import MambaSSMRegressor, TCNRegressor
from .pac_real2d_math import discrete_pole_real2d
from .pac_recurrence import recurrence_real2d_directional
from .pac_tight_frame_models import build_tight_frame_regressor

if TYPE_CHECKING:
    from .pac_types import PACExperimentConfig

MECHANISM_MODELS: tuple[str, ...] = (
    "pac_tf",
    "pac_tf_fixed_damping",
    "s4d",
    "gru",
    "tcn",
    "mamba",
)
PAC_TF_MODELS: tuple[str, ...] = ("pac_tf", "pac_tf_fixed_damping")


class S4DRegressor(nn.Module):
    """Causal diagonal state-space baseline with stable complex poles."""

    def __init__(self, config: PACExperimentConfig, model_dim: int = 53, modes: int = 26) -> None:
        super().__init__()
        self.input_projection = nn.Linear(config.raw_input_dim, model_dim)
        self.state_projection = nn.Linear(model_dim, 2 * modes)
        self.raw_decay = nn.Parameter(torch.linspace(-3.0, 1.0, modes))
        grid = torch.linspace(0.0, 0.9, modes).clamp(max=0.999)
        self.raw_frequency = nn.Parameter(torch.atanh(grid))
        self.state_output = nn.Linear(2 * modes, model_dim)
        self.norm = nn.RMSNorm(model_dim)
        self.output_projection = nn.Linear(model_dim, config.output_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        features = torch.nn.functional.silu(self.input_projection(inputs))
        excitation_real, excitation_imag = self.state_projection(features).chunk(2, dim=-1)
        damping = 1.0e-3 + (2.0 - 1.0e-3) * torch.sigmoid(self.raw_decay)
        frequency = torch.pi * torch.tanh(self.raw_frequency)
        decay_real, decay_imag, gamma_real, gamma_imag = discrete_pole_real2d(
            damping.view(1, 1, -1), frequency.view(1, 1, -1), 1.0
        )
        decay_real = decay_real.expand(inputs.shape[0], inputs.shape[1], -1)
        decay_imag = decay_imag.expand(inputs.shape[0], inputs.shape[1], -1)
        input_real = gamma_real * excitation_real - gamma_imag * excitation_imag
        input_imag = gamma_real * excitation_imag + gamma_imag * excitation_real
        state_real, state_imag = recurrence_real2d_directional(
            decay_real,
            decay_imag,
            input_real,
            input_imag,
            "auto",
            "forward",
        )
        state_features = self.state_output(torch.cat((state_real, state_imag), dim=-1))
        return self.output_projection(self.norm(features + state_features))


def build_mechanism_model(name: str, config: PACExperimentConfig) -> nn.Module:
    pac_model = build_tight_frame_regressor(name, config)
    if pac_model is not None:
        return pac_model
    match name:
        case "s4d":
            return S4DRegressor(config)
        case "gru":
            return build_regression_model("gru", replace(config, model_dim=30))
        case "tcn":
            return TCNRegressor(
                raw_input_dim=config.raw_input_dim,
                channels=33,
                levels=2,
                output_dim=config.output_dim,
            )
        case "mamba":
            return MambaSSMRegressor(
                raw_input_dim=config.raw_input_dim,
                model_dim=22,
                output_dim=config.output_dim,
            )
        case _:
            message = f"unsupported mechanism-recovery model: {name}"
            raise KeyError(message)
