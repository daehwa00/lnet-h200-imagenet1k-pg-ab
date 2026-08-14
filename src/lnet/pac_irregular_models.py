"""Native PyTorch baselines for irregularly sampled multivariate series."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional


class GRUDClassifier(nn.Module):
    """GRU-D with feature and hidden-state exponential decay."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        static_dim: int = 0,
        depth: int = 1,
    ) -> None:
        super().__init__()
        if depth < 1:
            message = "GRU-D depth must be positive"
            raise ValueError(message)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.gamma_x_weight = nn.Parameter(torch.ones(input_dim))
        self.gamma_x_bias = nn.Parameter(torch.zeros(input_dim))
        self.gamma_h = nn.ModuleList(
            nn.Linear(input_dim, hidden_dim) for _ in range(depth)
        )
        self.cells = nn.ModuleList(
            [
                nn.GRUCell(2 * input_dim, hidden_dim),
                *(nn.GRUCell(hidden_dim, hidden_dim) for _ in range(depth - 1)),
            ]
        )
        self.register_buffer("_feature_mean", torch.zeros(input_dim))
        self.head = nn.Linear(hidden_dim + static_dim, output_dim)

    @property
    def feature_mean(self) -> Tensor:
        return self.get_buffer("_feature_mean")

    def set_feature_mean(self, mean: Tensor) -> None:
        if mean.shape != (self.input_dim,):
            message = f"feature mean must have shape ({self.input_dim},)"
            raise ValueError(message)
        self.feature_mean.copy_(mean)

    def forward(
        self,
        values: Tensor,
        observed: Tensor,
        time_delta: Tensor,
        valid: Tensor,
        static: Tensor | None = None,
    ) -> Tensor:
        if values.shape != observed.shape or values.shape != time_delta.shape:
            message = "values, observed, and time_delta must have identical [B,T,D] shapes"
            raise ValueError(message)
        batch, steps, _ = values.shape
        hidden = [
            values.new_zeros((batch, self.hidden_dim)) for _ in range(self.depth)
        ]
        previous = self.feature_mean.expand(batch, -1)
        mean = self.feature_mean.expand(batch, -1)
        for step in range(steps):
            mask = observed[:, step].to(values.dtype)
            delta = time_delta[:, step]
            gamma_x = torch.exp(
                -functional.relu(delta * self.gamma_x_weight + self.gamma_x_bias)
            )
            imputed = mask * values[:, step] + (1.0 - mask) * (
                gamma_x * previous + (1.0 - gamma_x) * mean
            )
            active = valid[:, step].reshape(batch, 1).to(torch.bool)
            layer_input = torch.cat((imputed, mask), dim=-1)
            for layer, (cell, decay) in enumerate(zip(self.cells, self.gamma_h, strict=True)):
                gamma_h = torch.exp(-functional.relu(decay(delta)))
                candidate = cell(layer_input, gamma_h * hidden[layer])
                hidden[layer] = torch.where(active, candidate, hidden[layer])
                layer_input = hidden[layer]
            previous = torch.where(mask.to(torch.bool), values[:, step], previous)
        features = hidden[-1] if static is None else torch.cat((hidden[-1], static), dim=-1)
        return self.head(features)


def observed_feature_mean(values: Tensor, observed: Tensor, indices: Tensor) -> Tensor:
    selected_values = values[indices]
    selected_observed = observed[indices].to(values.dtype)
    numerator = (selected_values * selected_observed).sum(dim=(0, 1))
    denominator = selected_observed.sum(dim=(0, 1)).clamp_min(1.0)
    return numerator / denominator


def _append_static(hidden: Tensor, static: Tensor | None) -> Tensor:
    return hidden if static is None else torch.cat((hidden, static), dim=-1)


class _ODEVectorField(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, hidden: Tensor) -> Tensor:
        return self.net(hidden)


def _ode_step(
    field: _ODEVectorField,
    hidden: Tensor,
    delta: Tensor,
    *,
    solver: str,
) -> Tensor:
    dt = delta.reshape(hidden.shape[0], 1).to(hidden.dtype)
    if solver == "euler":
        return hidden + dt * field(hidden)
    k1 = field(hidden)
    k2 = field(hidden + dt * (1 / 5) * k1)
    k3 = field(hidden + dt * ((3 / 40) * k1 + (9 / 40) * k2))
    k4 = field(
        hidden
        + dt
        * ((44 / 45) * k1 - (56 / 15) * k2 + (32 / 9) * k3)
    )
    k5 = field(
        hidden
        + dt
        * (
            (19372 / 6561) * k1
            - (25360 / 2187) * k2
            + (64448 / 6561) * k3
            - (212 / 729) * k4
        )
    )
    k6 = field(
        hidden
        + dt
        * (
            (9017 / 3168) * k1
            - (355 / 33) * k2
            + (46732 / 5247) * k3
            + (49 / 176) * k4
            - (5103 / 18656) * k5
        )
    )
    return hidden + dt * (
        (35 / 384) * k1
        + (500 / 1113) * k3
        + (125 / 192) * k4
        - (2187 / 6784) * k5
        + (11 / 84) * k6
    )


class ODERNNClassifier(nn.Module):
    """ODE-RNN with neural-ODE evolution between masked GRU observations."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        solver: str,
        static_dim: int = 0,
    ) -> None:
        super().__init__()
        if solver not in {"euler", "dopri5"}:
            message = f"unsupported ODE-RNN solver: {solver}"
            raise ValueError(message)
        self.solver = solver
        self.field = _ODEVectorField(hidden_dim)
        self.update = nn.GRUCell(2 * input_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim + static_dim, output_dim)

    def forward(
        self,
        values: Tensor,
        observed: Tensor,
        interval_delta: Tensor,
        _feature_delta: Tensor,
        valid: Tensor,
        static: Tensor | None = None,
    ) -> Tensor:
        batch, steps, _ = values.shape
        hidden = values.new_zeros((batch, self.update.hidden_size))
        for step in range(steps):
            active = valid[:, step].reshape(batch, 1).bool()
            evolved = _ode_step(
                self.field,
                hidden,
                interval_delta[:, step],
                solver=self.solver,
            )
            inputs = torch.cat(
                (values[:, step], observed[:, step].to(values.dtype)),
                dim=-1,
            )
            updated = self.update(inputs, evolved)
            has_observation = observed[:, step].any(dim=-1, keepdim=True)
            candidate = torch.where(has_observation, updated, evolved)
            hidden = torch.where(active, candidate, hidden)
        return self.head(_append_static(hidden, static))


class LatentODEClassifier(nn.Module):
    """Variational latent-ODE encoder followed by a classification head."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        encoder: str,
        static_dim: int = 0,
        kl_weight: float = 1.0e-3,
    ) -> None:
        super().__init__()
        if encoder not in {"rnn-encoder", "ode-encoder"}:
            message = f"unsupported Latent ODE encoder: {encoder}"
            raise ValueError(message)
        self.encoder = encoder
        self.kl_weight = kl_weight
        self.field = _ODEVectorField(hidden_dim)
        self.update = nn.GRUCell(2 * input_dim + 1, hidden_dim)
        self.posterior = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.head = nn.Linear(hidden_dim + static_dim, output_dim)
        self.auxiliary_loss = torch.tensor(0.0)

    def forward(
        self,
        values: Tensor,
        observed: Tensor,
        interval_delta: Tensor,
        _feature_delta: Tensor,
        valid: Tensor,
        static: Tensor | None = None,
    ) -> Tensor:
        batch, steps, _ = values.shape
        hidden = values.new_zeros((batch, self.update.hidden_size))
        for step in range(steps):
            active = valid[:, step].reshape(batch, 1).bool()
            if self.encoder == "ode-encoder":
                prior = _ode_step(
                    self.field,
                    hidden,
                    interval_delta[:, step],
                    solver="dopri5",
                )
            else:
                prior = hidden
            inputs = torch.cat(
                (
                    values[:, step],
                    observed[:, step].to(values.dtype),
                    interval_delta[:, step],
                ),
                dim=-1,
            )
            updated = self.update(inputs, prior)
            has_observation = observed[:, step].any(dim=-1, keepdim=True)
            candidate = torch.where(has_observation, updated, prior)
            hidden = torch.where(active, candidate, hidden)
        mean, log_variance = self.posterior(hidden).chunk(2, dim=-1)
        log_variance = log_variance.clamp(-12, 12)
        if self.training:
            latent = mean + torch.randn_like(mean) * torch.exp(0.5 * log_variance)
        else:
            latent = mean
        kl = 0.5 * (mean.square() + log_variance.exp() - 1 - log_variance)
        self.auxiliary_loss = self.kl_weight * kl.mean()
        return self.head(_append_static(latent, static))


class _LearnedTimeEmbedding(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        if embed_dim < 2:
            message = "time embedding dimension must be at least two"
            raise ValueError(message)
        self.linear = nn.Linear(1, 1)
        self.periodic = nn.Linear(1, embed_dim - 1)

    def forward(self, times: Tensor) -> Tensor:
        times = times.unsqueeze(-1)
        return torch.cat((self.linear(times), torch.sin(self.periodic(times))), dim=-1)


class MTANClassifier(nn.Module):
    """Multi-Time Attention Network classifier with learned query times."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        heads: int,
        static_dim: int = 0,
        query_count: int = 32,
    ) -> None:
        super().__init__()
        embed_dim = max(8, heads * 4)
        if embed_dim % heads:
            message = "mTAN time embedding must be divisible by the head count"
            raise ValueError(message)
        self.heads = heads
        self.embed_dim = embed_dim
        self.time_embedding = _LearnedTimeEmbedding(embed_dim)
        self.query = nn.Parameter(torch.linspace(0, 1, query_count))
        self.query_projection = nn.Linear(embed_dim, embed_dim)
        self.key_projection = nn.Linear(embed_dim, embed_dim)
        self.value_projection = nn.Linear(2 * input_dim * heads, hidden_dim)
        self.encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim + static_dim, output_dim)

    def forward(
        self,
        values: Tensor,
        observed: Tensor,
        interval_delta: Tensor,
        _feature_delta: Tensor,
        valid: Tensor,
        static: Tensor | None = None,
    ) -> Tensor:
        batch, _, features = values.shape
        times = interval_delta.squeeze(-1).cumsum(dim=1)
        duration = (times * valid.squeeze(-1)).amax(dim=1, keepdim=True).clamp_min(1)
        normalized_time = times / duration
        keys = self.key_projection(self.time_embedding(normalized_time))
        queries = self.query_projection(
            self.time_embedding(self.query).expand(batch, -1, -1)
        )
        head_dim = self.embed_dim // self.heads
        keys = keys.view(batch, -1, self.heads, head_dim).transpose(1, 2)
        queries = queries.view(batch, -1, self.heads, head_dim).transpose(1, 2)
        scores = torch.matmul(queries, keys.transpose(-1, -2)) / math.sqrt(head_dim)
        scores = scores.masked_fill(~valid.squeeze(-1)[:, None, None, :], -1.0e4)
        attention = torch.softmax(scores, dim=-1)
        masked_values = torch.cat((values, observed.to(values.dtype)), dim=-1)
        attended = torch.einsum("bhqt,btd->bhqd", attention, masked_values)
        attended = attended.transpose(1, 2).reshape(
            batch,
            self.query.numel(),
            2 * features * self.heads,
        )
        sequence = self.value_projection(attended)
        _, hidden = self.encoder(sequence)
        return self.head(_append_static(hidden[-1], static))


class NeuralCDEClassifier(nn.Module):
    """Neural CDE classifier using a native piecewise interpolation solver."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        interpolation: str,
        static_dim: int = 0,
    ) -> None:
        super().__init__()
        if interpolation not in {"linear", "cubic"}:
            message = f"unsupported Neural CDE interpolation: {interpolation}"
            raise ValueError(message)
        self.interpolation = interpolation
        self.control_dim = 2 * input_dim + 1
        self.initial = nn.Linear(self.control_dim, hidden_dim)
        self.vector_field = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim * self.control_dim),
        )
        self.hidden_dim = hidden_dim
        self.head = nn.Linear(hidden_dim + static_dim, output_dim)

    def _derivative(self, hidden: Tensor, increment: Tensor) -> Tensor:
        field = self.vector_field(hidden).view(
            hidden.shape[0],
            self.hidden_dim,
            self.control_dim,
        )
        return torch.einsum("bhc,bc->bh", field, increment)

    def forward(
        self,
        values: Tensor,
        observed: Tensor,
        interval_delta: Tensor,
        _feature_delta: Tensor,
        valid: Tensor,
        static: Tensor | None = None,
    ) -> Tensor:
        control = torch.cat(
            (values, observed.to(values.dtype), interval_delta),
            dim=-1,
        )
        hidden = torch.tanh(self.initial(control[:, 0]))
        for step in range(1, control.shape[1]):
            active = valid[:, step].reshape(values.shape[0], 1).bool()
            increment = control[:, step] - control[:, step - 1]
            if self.interpolation == "linear":
                candidate = torch.tanh(
                    hidden + self._derivative(hidden, increment)
                )
            else:
                first = self._derivative(hidden, increment)
                midpoint = hidden + 0.5 * first
                candidate = torch.tanh(
                    hidden + self._derivative(midpoint, increment)
                )
            hidden = torch.where(active, candidate, hidden)
        return self.head(_append_static(hidden, static))


class _DenseObservationPropagation(nn.Module):
    def __init__(self, sensor_dim: int, observation_dim: int) -> None:
        super().__init__()
        self.adjacency_logits = nn.Parameter(torch.zeros(sensor_dim, sensor_dim))
        self.self_projection = nn.Linear(observation_dim, observation_dim, bias=False)
        self.message_projection = nn.Linear(
            observation_dim,
            observation_dim,
            bias=False,
        )

    def forward(self, hidden: Tensor, observed: Tensor) -> Tensor:
        adjacency = torch.softmax(self.adjacency_logits, dim=-1)
        messages = torch.einsum("ij,btjo->btio", adjacency, hidden)
        candidate = functional.relu(
            self.self_projection(hidden) + self.message_projection(messages)
        )
        mask = observed.unsqueeze(-1)
        return torch.where(mask, candidate + hidden, candidate)


class RaindropClassifier(nn.Module):
    """Dense PyTorch implementation of Raindrop observation propagation."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        propagation_layers: int,
        static_dim: int = 0,
    ) -> None:
        super().__init__()
        if propagation_layers not in {1, 2}:
            message = "Raindrop supports one or two propagation layers"
            raise ValueError(message)
        observation_dim = max(2, math.ceil(hidden_dim / input_dim))
        temporal_dim = observation_dim * input_dim
        heads = next(
            head for head in (4, 2, 1) if temporal_dim % head == 0
        )
        self.sensor_embedding = nn.Parameter(
            torch.empty(input_dim, observation_dim)
        )
        nn.init.xavier_uniform_(self.sensor_embedding)
        self.propagation = nn.ModuleList(
            _DenseObservationPropagation(input_dim, observation_dim)
            for _ in range(propagation_layers)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=temporal_dim,
            nhead=heads,
            dim_feedforward=max(2 * temporal_dim, hidden_dim),
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Linear(temporal_dim + static_dim, output_dim)

    def forward(
        self,
        values: Tensor,
        observed: Tensor,
        _interval_delta: Tensor,
        _feature_delta: Tensor,
        valid: Tensor,
        static: Tensor | None = None,
    ) -> Tensor:
        hidden = functional.relu(values.unsqueeze(-1) * self.sensor_embedding)
        for layer in self.propagation:
            hidden = layer(hidden, observed)
        sequence = hidden.flatten(-2)
        padding = ~valid.squeeze(-1).bool()
        encoded = self.temporal(sequence, src_key_padding_mask=padding)
        weights = valid.to(encoded.dtype)
        pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        return self.head(_append_static(pooled, static))


__all__ = [
    "GRUDClassifier",
    "LatentODEClassifier",
    "MTANClassifier",
    "NeuralCDEClassifier",
    "ODERNNClassifier",
    "RaindropClassifier",
    "observed_feature_mean",
]
