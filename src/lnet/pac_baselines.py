from __future__ import annotations

import torch
from torch import Tensor, nn

from .pac_recurrence import recurrence_real2d


class LSTMSequenceBaseline(nn.Module):
    def __init__(self, *, raw_input_dim: int, model_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_projection = nn.Linear(raw_input_dim, model_dim)
        self.lstm = nn.LSTM(input_size=model_dim, hidden_size=model_dim, batch_first=True)
        self.output_projection = nn.Linear(model_dim, output_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        encoded, _ = self.lstm(self.input_projection(inputs))
        return self.output_projection(encoded)


class SelectiveDiagonalSSMBaseline(nn.Module):
    def __init__(
        self,
        *,
        raw_input_dim: int,
        model_dim: int,
        modes: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(raw_input_dim, model_dim)
        self.decay_projection = nn.Linear(model_dim, modes)
        self.drive_projection = nn.Linear(model_dim, modes)
        self.readout_projection = nn.Linear(modes, model_dim)
        self.output_projection = nn.Linear(model_dim, output_dim)
        self.activation = nn.GELU()

    def forward(self, inputs: Tensor) -> Tensor:
        projected = self.input_projection(inputs)
        decay = torch.sigmoid(self.decay_projection(projected))
        drive = self.drive_projection(projected)
        zeros = torch.zeros_like(drive)
        states, _ = recurrence_real2d(decay, zeros, drive, zeros, "auto")
        encoded = self.activation(self.readout_projection(states))
        return self.output_projection(encoded)
