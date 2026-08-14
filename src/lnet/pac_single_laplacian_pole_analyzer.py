from __future__ import annotations

import math
from typing import TYPE_CHECKING, Final

import torch
from torch import Tensor, nn
from torch.nn import functional
from torch.nn.utils.parametrizations import orthogonal

from .pac_headroom_efficient_models import (
    _apply_raw_mask,  # pyright: ignore[reportPrivateUsage]
    _degree_normalized_edge_analysis,  # pyright: ignore[reportPrivateUsage]
    _edge_or_singleton_mask,  # pyright: ignore[reportPrivateUsage]
)
from .pac_laplacian_pole_stack import (
    _last_valid_state,  # pyright: ignore[reportPrivateUsage]
    _masked_mean,  # pyright: ignore[reportPrivateUsage]
    _metadata_3d,  # pyright: ignore[reportPrivateUsage]
)
from .pac_raw_efficiency_candidates import (
    _stable_discrete_pole_real2d,  # pyright: ignore[reportPrivateUsage]
)
from .pac_recurrence import recurrence_real2d_directional

if TYPE_CHECKING:
    from .pac_types import PACExperimentConfig


MODEL_NAME: Final = "single_laplacian_pole_rich_readout"
_MODES: Final = 16
_LOCAL_KERNEL: Final = 5
_LOCAL_DILATION: Final = 4
_DAMPING_MIN: Final = 1.0e-3
_DAMPING_MAX: Final = 2.0
_EPSILON: Final = 1.0e-8


def _combined_mask(first: Tensor | None, second: Tensor | None) -> Tensor | None:
    if first is None:
        return second
    if second is None:
        return first
    return torch.minimum(first, second)


def _repeat_bands(metadata: Tensor | None) -> Tensor | None:
    if metadata is None:
        return None
    return _metadata_3d(metadata).repeat_interleave(2, dim=0)


def _normalized_positions(states: Tensor, mask: Tensor | None) -> Tensor:
    steps = states.shape[1]
    indices = torch.arange(steps, device=states.device, dtype=states.dtype).view(1, -1, 1)
    if mask is None:
        denominator = states.new_tensor(max(steps - 1, 1))
        return indices / denominator
    active = _metadata_3d(mask).to(device=states.device, dtype=states.dtype)
    lengths = active.sum(dim=1, keepdim=True).clamp_min(1.0)
    return indices / lengths.sub(1.0).clamp_min(1.0)


def rich_complex_readout(
    common_real: Tensor,
    common_imag: Tensor,
    variation_real: Tensor,
    variation_imag: Tensor,
    mask: Tensor | None,
) -> Tensor:
    """Read terminal memory, two complex moments, energy, and band coherence."""
    terminal = torch.cat(
        (
            _last_valid_state(common_real, mask),
            _last_valid_state(common_imag, mask),
            _last_valid_state(variation_real, mask),
            _last_valid_state(variation_imag, mask),
        ),
        dim=-1,
    )
    common_mu0_real = _masked_mean(common_real, mask)
    common_mu0_imag = _masked_mean(common_imag, mask)
    variation_mu0_real = _masked_mean(variation_real, mask)
    variation_mu0_imag = _masked_mean(variation_imag, mask)
    position = _normalized_positions(common_real, mask)
    common_mu1_real = _masked_mean(position * common_real, mask)
    common_mu1_imag = _masked_mean(position * common_imag, mask)
    variation_mu1_real = _masked_mean(position * variation_real, mask)
    variation_mu1_imag = _masked_mean(position * variation_imag, mask)

    common_power = common_real.square() + common_imag.square()
    variation_power = variation_real.square() + variation_imag.square()
    common_energy = _masked_mean(common_power, mask)
    variation_energy = _masked_mean(variation_power, mask)
    cross_real = _masked_mean(
        common_real * variation_real + common_imag * variation_imag,
        mask,
    )
    cross_imag = _masked_mean(
        common_imag * variation_real - common_real * variation_imag,
        mask,
    )
    denominator = torch.sqrt(
        (common_energy * variation_energy).clamp_min(_EPSILON * _EPSILON)
    )
    return torch.cat(
        (
            terminal,
            common_mu0_real,
            common_mu0_imag,
            variation_mu0_real,
            variation_mu0_imag,
            common_mu1_real,
            common_mu1_imag,
            variation_mu1_real,
            variation_mu1_imag,
            common_energy,
            variation_energy,
            cross_real / denominator,
            cross_imag / denominator,
        ),
        dim=-1,
    )


class SingleLaplacianPoleAnalyzer(nn.Module):
    supports_observation_mask: Final[bool] = True
    supports_time_delta: Final[bool] = True

    def __init__(self, config: PACExperimentConfig, output_dim: int) -> None:
        super().__init__()
        if config.model_dim != 32 or config.modes != _MODES:
            message = "the first single-analyzer screen is locked to D=32 and M=16"
            raise ValueError(message)
        self.model_dim = config.model_dim
        self.modes = config.modes
        self.raw_projection = nn.Linear(config.raw_input_dim, config.model_dim)
        self.input_projection = nn.Linear(config.model_dim, config.model_dim, bias=False)
        nn.init.orthogonal_(self.input_projection.weight)
        self.local = nn.Conv1d(
            config.model_dim,
            config.model_dim,
            kernel_size=_LOCAL_KERNEL,
            dilation=_LOCAL_DILATION,
            padding=0,
            groups=config.model_dim,
        )
        self.analysis_frame = nn.Linear(2 * config.modes, config.model_dim, bias=False)
        nn.init.orthogonal_(self.analysis_frame.weight)
        orthogonal(
            self.analysis_frame,
            "weight",
            orthogonal_map="matrix_exp",
            use_trivialization=True,
        )
        self.raw_decay = nn.Parameter(torch.linspace(-3.0, 1.0, config.modes))
        frequency_grid = torch.linspace(0.0, 0.75, config.modes).clamp(max=0.999)
        self.raw_frequency = nn.Parameter(torch.atanh(frequency_grid))
        readout_dim = config.model_dim + 2 * config.model_dim + 16 * config.modes
        self.head = nn.Linear(readout_dim, output_dim)

    def _local_features(self, coefficients: Tensor) -> Tensor:
        if coefficients.ndim != 4 or coefficients.shape[2] != 2:
            message = "coefficients must have shape [B,T,2,D]"
            raise ValueError(message)
        batch, steps, bands, channels = coefficients.shape
        projected = self.input_projection(coefficients)
        packed = projected.permute(0, 2, 3, 1).reshape(batch * bands, channels, steps)
        left_padding = _LOCAL_DILATION * (_LOCAL_KERNEL - 1)
        local = functional.silu(self.local(functional.pad(packed, (left_padding, 0))))
        return local.reshape(batch, bands, channels, steps).permute(0, 3, 1, 2)

    def _scan(
        self,
        local: Tensor,
        *,
        edge_delta: Tensor | None,
        observation_mask: Tensor | None,
        valid_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        batch, steps, bands, channels = local.shape
        packed = local.permute(0, 2, 1, 3).reshape(batch * bands, steps, channels)
        excitation = torch.matmul(packed, self.analysis_frame.weight)
        excitation_real, excitation_imag = excitation.chunk(2, dim=-1)
        damping = (
            _DAMPING_MIN
            + (_DAMPING_MAX - _DAMPING_MIN) * torch.sigmoid(self.raw_decay)
        ).view(1, 1, -1).expand_as(excitation_real)
        frequency = (math.pi * torch.tanh(self.raw_frequency)).view(1, 1, -1)
        frequency = frequency.expand_as(excitation_real)
        active_delta = _repeat_bands(edge_delta)
        step: float | Tensor = 1.0 if active_delta is None else active_delta
        decay_real, decay_imag, gain_real, gain_imag = _stable_discrete_pole_real2d(
            damping,
            frequency,
            step,
        )
        input_real = gain_real * excitation_real - gain_imag * excitation_imag
        input_imag = gain_real * excitation_imag + gain_imag * excitation_real
        active_mask = _combined_mask(
            _repeat_bands(observation_mask),
            _repeat_bands(valid_mask),
        )
        if active_mask is not None:
            weight = active_mask.to(device=input_real.device, dtype=input_real.dtype)
            input_real = input_real * weight
            input_imag = input_imag * weight
        states_real, states_imag = recurrence_real2d_directional(
            decay_real,
            decay_imag,
            input_real,
            input_imag,
            "auto",
            "forward",
        )
        return (
            states_real.reshape(batch, bands, steps, self.modes),
            states_imag.reshape(batch, bands, steps, self.modes),
        )

    def forward(
        self,
        inputs: Tensor,
        *,
        time_delta: Tensor | None = None,
        observation_mask: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        stem_inputs = _apply_raw_mask(inputs, observation_mask, valid_mask)
        encoded = self.raw_projection(stem_inputs)
        node_mask = valid_mask if valid_mask is not None else observation_mask
        if node_mask is not None:
            node_weight = _metadata_3d(node_mask).to(
                device=encoded.device,
                dtype=encoded.dtype,
            )
            encoded = encoded * node_weight
        common, variation, edge_delta = _degree_normalized_edge_analysis(encoded, time_delta)
        coefficients = torch.stack((common, variation), dim=2)
        active_observation = _edge_or_singleton_mask(observation_mask)
        active_valid = _edge_or_singleton_mask(valid_mask)
        edge_mask = _combined_mask(active_observation, active_valid)
        edge_weight: Tensor | None = None
        if edge_mask is not None:
            edge_weight = _metadata_3d(edge_mask).to(
                device=coefficients.device,
                dtype=coefficients.dtype,
            )
            coefficients = coefficients * edge_weight.unsqueeze(2)
        local = self._local_features(coefficients)
        if edge_weight is not None:
            local = local * edge_weight.unsqueeze(2)
        states_real, states_imag = self._scan(
            local,
            edge_delta=edge_delta,
            observation_mask=active_observation,
            valid_mask=active_valid,
        )
        raw_pool = _masked_mean(encoded, node_mask)
        common_local = _masked_mean(local[:, :, 0], edge_mask)
        variation_local = _masked_mean(local[:, :, 1], edge_mask)
        pole_readout = rich_complex_readout(
            states_real[:, 0],
            states_imag[:, 0],
            states_real[:, 1],
            states_imag[:, 1],
            edge_mask,
        )
        features = torch.cat((raw_pool, common_local, variation_local, pole_readout), dim=-1)
        return self.head(features)

    def post_optimizer_step(self) -> None:
        """Matrix-exponential parametrization remains orthogonal automatically."""

    def finalize_constraints(self) -> None:
        """Matrix-exponential parametrization remains orthogonal automatically."""


def build_single_laplacian_pole_analyzer(
    config: PACExperimentConfig,
    output_dim: int,
) -> SingleLaplacianPoleAnalyzer:
    return SingleLaplacianPoleAnalyzer(config, output_dim)
