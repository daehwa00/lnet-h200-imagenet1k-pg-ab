from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from lnet.gated_prl import GatedPRLBlock, GateVariant
from lnet.laplace import ProjectedPRLBlock


class PerStepMLPBaseline(nn.Module):
    def __init__(
        self,
        *,
        raw_input_dim: int,
        model_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(raw_input_dim, model_dim)
        self.hidden_projection = nn.Linear(model_dim, model_dim)
        self.output_projection = nn.Linear(model_dim, output_dim)
        self.activation = nn.Tanh()

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = self.activation(self.input_projection(inputs))
        hidden = self.activation(self.hidden_projection(hidden))
        return self.output_projection(hidden)


class LinearRecurrentBaseline(nn.Module):
    def __init__(
        self,
        *,
        raw_input_dim: int,
        model_dim: int,
        state_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(raw_input_dim, model_dim)
        self.state_projection = nn.Linear(state_dim, state_dim, bias=False)
        self.drive_projection = nn.Linear(model_dim, state_dim)
        self.readout_projection = nn.Linear(state_dim, model_dim)
        self.skip_projection = nn.Linear(model_dim, model_dim)
        self.output_projection = nn.Linear(model_dim, output_dim)
        self.activation = nn.Tanh()
        with torch.no_grad():
            self.state_projection.weight.mul_(0.2)

    def forward(self, inputs: Tensor) -> Tensor:
        projected = self.input_projection(inputs)
        state = torch.zeros(
            inputs.shape[0],
            self.state_projection.in_features,
            dtype=inputs.dtype,
            device=inputs.device,
        )
        outputs: list[Tensor] = []
        for current_input in projected.unbind(dim=1):
            state = self.activation(
                self.state_projection(state) + self.drive_projection(current_input),
            )
            mixed = self.activation(
                self.readout_projection(state) + self.skip_projection(current_input),
            )
            outputs.append(self.output_projection(mixed))
        return torch.stack(outputs, dim=1)


class FIRSequenceBaseline(nn.Module):
    def __init__(
        self,
        *,
        raw_input_dim: int,
        model_dim: int,
        output_dim: int,
        kernel_size: int,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(raw_input_dim, model_dim)
        self.convolution = nn.Conv1d(
            in_channels=model_dim,
            out_channels=model_dim,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
        )
        self.output_projection = nn.Linear(model_dim, output_dim)
        self.activation = nn.Tanh()

    def forward(self, inputs: Tensor) -> Tensor:
        projected = self.input_projection(inputs).transpose(1, 2)
        convolved = self.convolution(projected)[..., : inputs.shape[1]].transpose(1, 2)
        return self.output_projection(self.activation(convolved))


class GRUSequenceBaseline(nn.Module):
    def __init__(
        self,
        *,
        raw_input_dim: int,
        model_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(raw_input_dim, model_dim)
        self.gru = nn.GRU(input_size=model_dim, hidden_size=model_dim, batch_first=True)
        self.output_projection = nn.Linear(model_dim, output_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        projected = self.input_projection(inputs)
        encoded, _ = self.gru(projected)
        return self.output_projection(encoded)


class TransformerSequenceBaseline(nn.Module):
    def __init__(
        self,
        *,
        raw_input_dim: int,
        model_dim: int,
        output_dim: int,
        attention_heads: int,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(raw_input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=attention_heads,
            dim_feedforward=model_dim * 2,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.output_projection = nn.Linear(model_dim, output_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        projected = self.input_projection(inputs)
        encoded = self.encoder(projected + _sinusoidal_positions(projected))
        return self.output_projection(encoded)


def _sinusoidal_positions(inputs: Tensor) -> Tensor:
    """Return a parameter-free absolute position code for a batch-first sequence."""
    length, width = inputs.shape[1:3]
    positions = torch.arange(length, device=inputs.device, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, width, 2, device=inputs.device, dtype=torch.float32)
        * (-math.log(10_000.0) / max(width, 1))
    )
    angles = positions * frequencies.unsqueeze(0)
    encoding = torch.zeros(length, width, device=inputs.device, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(angles)
    if width > 1:
        encoding[:, 1::2] = torch.cos(angles[:, : encoding[:, 1::2].shape[1]])
    return encoding.to(dtype=inputs.dtype).unsqueeze(0)


class PRLSequenceClassifier(nn.Module):
    def __init__(
        self,
        *,
        raw_input_dim: int,
        model_dim: int,
        modes: int,
        class_count: int,
    ) -> None:
        super().__init__()
        self.encoder = ProjectedPRLBlock(
            raw_input_dim=raw_input_dim,
            model_dim=model_dim,
            output_dim=model_dim,
            modes=modes,
        )
        self.classifier = nn.Linear(model_dim, class_count)

    def forward(self, inputs: Tensor) -> Tensor:
        encoded = self.encoder(inputs)
        pooled = torch.mean(encoded, dim=1)
        return self.classifier(pooled)


class GatedPRLSequenceClassifier(nn.Module):
    def __init__(
        self,
        *,
        raw_input_dim: int,
        model_dim: int,
        modes: int,
        class_count: int,
        gate_variant: GateVariant,
    ) -> None:
        super().__init__()
        self.encoder = GatedPRLBlock(
            raw_input_dim=raw_input_dim,
            model_dim=model_dim,
            output_dim=model_dim,
            modes=modes,
            gate_variant=gate_variant,
        )
        self.classifier = nn.Linear(model_dim, class_count)

    def forward(self, inputs: Tensor) -> Tensor:
        encoded = self.encoder(inputs)
        pooled = torch.mean(encoded, dim=1)
        return self.classifier(pooled)


class MLPSequenceClassifier(nn.Module):
    def __init__(
        self,
        *,
        raw_input_dim: int,
        model_dim: int,
        class_count: int,
    ) -> None:
        super().__init__()
        self.encoder = PerStepMLPBaseline(
            raw_input_dim=raw_input_dim,
            model_dim=model_dim,
            output_dim=model_dim,
        )
        self.classifier = nn.Linear(model_dim, class_count)

    def forward(self, inputs: Tensor) -> Tensor:
        encoded = self.encoder(inputs)
        pooled = torch.mean(encoded, dim=1)
        return self.classifier(pooled)


class FIRSequenceClassifier(nn.Module):
    def __init__(
        self,
        *,
        raw_input_dim: int,
        model_dim: int,
        class_count: int,
        kernel_size: int,
    ) -> None:
        super().__init__()
        self.encoder = FIRSequenceBaseline(
            raw_input_dim=raw_input_dim,
            model_dim=model_dim,
            output_dim=model_dim,
            kernel_size=kernel_size,
        )
        self.classifier = nn.Linear(model_dim, class_count)

    def forward(self, inputs: Tensor) -> Tensor:
        encoded = self.encoder(inputs)
        pooled = torch.mean(encoded, dim=1)
        return self.classifier(pooled)


class GRUSequenceClassifier(nn.Module):
    def __init__(
        self,
        *,
        raw_input_dim: int,
        model_dim: int,
        class_count: int,
    ) -> None:
        super().__init__()
        self.encoder = GRUSequenceBaseline(
            raw_input_dim=raw_input_dim,
            model_dim=model_dim,
            output_dim=model_dim,
        )
        self.classifier = nn.Linear(model_dim, class_count)

    def forward(self, inputs: Tensor) -> Tensor:
        encoded = self.encoder(inputs)
        pooled = torch.mean(encoded, dim=1)
        return self.classifier(pooled)


class LinearRecurrentClassifier(nn.Module):
    def __init__(
        self,
        *,
        raw_input_dim: int,
        model_dim: int,
        state_dim: int,
        class_count: int,
    ) -> None:
        super().__init__()
        self.encoder = LinearRecurrentBaseline(
            raw_input_dim=raw_input_dim,
            model_dim=model_dim,
            state_dim=state_dim,
            output_dim=model_dim,
        )
        self.classifier = nn.Linear(model_dim, class_count)

    def forward(self, inputs: Tensor) -> Tensor:
        encoded = self.encoder(inputs)
        pooled = torch.mean(encoded, dim=1)
        return self.classifier(pooled)


class TransformerSequenceClassifier(nn.Module):
    def __init__(
        self,
        *,
        raw_input_dim: int,
        model_dim: int,
        class_count: int,
        attention_heads: int,
    ) -> None:
        super().__init__()
        self.encoder = TransformerSequenceBaseline(
            raw_input_dim=raw_input_dim,
            model_dim=model_dim,
            output_dim=model_dim,
            attention_heads=attention_heads,
        )
        self.classifier = nn.Linear(model_dim, class_count)

    def forward(self, inputs: Tensor) -> Tensor:
        encoded = self.encoder(inputs)
        pooled = torch.mean(encoded, dim=1)
        return self.classifier(pooled)
