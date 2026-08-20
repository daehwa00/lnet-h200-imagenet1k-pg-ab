"""Two-dimensional Laplace product-pole fields and auditable image readout."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, cast

import torch
from torch import Tensor, nn
from torch.nn import functional
from torch.nn.utils.parametrizations import orthogonal

from .pac_real2d_math import discrete_pole_real2d
from .pac_recurrence import RecurrenceBackend, recurrence_real2d_directional

SpatialWindows = Literal["global", "global_2x2"]
SpatialLag = tuple[int, int]
_QUADRANTS = ((1, 1), (-1, 1), (1, -1), (-1, -1))
_DEFAULT_LAGS: tuple[SpatialLag, ...] = ((1, 0), (0, 1), (1, 1), (1, -1))


@dataclass(frozen=True, slots=True)
class Alphabet2DConfig:
    input_channels: int
    output_dim: int
    image_size: int = 224
    patch_size: int = 16
    model_dim: int = 192
    modes: int = 16
    depth: int = 8
    mlp_ratio: float = 2.0
    lags: tuple[SpatialLag, ...] = _DEFAULT_LAGS
    windows: SpatialWindows = "global_2x2"
    fixed_direct_atlas: bool = True
    recurrence_backend: RecurrenceBackend = "auto"
    layer_scale_init: float = 1.0e-2


def _complex_multiply(
    left_real: Tensor,
    left_imag: Tensor,
    right_real: Tensor,
    right_imag: Tensor,
) -> tuple[Tensor, Tensor]:
    return (
        left_real * right_real - left_imag * right_imag,
        left_real * right_imag + left_imag * right_real,
    )


def _axis_scan(
    input_real: Tensor,
    input_imag: Tensor,
    decay_real: Tensor,
    decay_imag: Tensor,
    *,
    direction: int,
    recurrence_backend: RecurrenceBackend,
) -> tuple[Tensor, Tensor]:
    if direction not in {-1, 1}:
        message = "scan direction must be -1 or 1"
        raise ValueError(message)
    expanded_decay_real = decay_real.view(1, 1, -1).expand_as(input_real)
    expanded_decay_imag = decay_imag.view(1, 1, -1).expand_as(input_imag)
    return recurrence_real2d_directional(
        expanded_decay_real,
        expanded_decay_imag,
        input_real,
        input_imag,
        recurrence_backend,
        "forward" if direction == 1 else "backward",
    )


def product_pole_scan_2d(
    excitation_real: Tensor,
    excitation_imag: Tensor,
    *,
    damping_x: Tensor,
    damping_y: Tensor,
    frequency_x: Tensor,
    frequency_y: Tensor,
    spacing_x: float,
    spacing_y: float,
    direction_x: int,
    direction_y: int,
    recurrence_backend: RecurrenceBackend = "auto",
) -> tuple[Tensor, Tensor]:
    """Apply one separable directional 2D exact-ZOH product-pole scan."""
    if excitation_real.shape != excitation_imag.shape or excitation_real.ndim != 4:
        message = "excitations must have identical [B,H,W,M] shapes"
        raise ValueError(message)
    modes = excitation_real.shape[-1]
    pole_values = (damping_x, damping_y, frequency_x, frequency_y)
    if any(value.shape != (modes,) for value in pole_values):
        message = "each pole coordinate must have shape [M]"
        raise ValueError(message)
    if spacing_x <= 0.0 or spacing_y <= 0.0:
        message = "spatial spacing must be positive"
        raise ValueError(message)

    decay_x_real, decay_x_imag, gamma_x_real, gamma_x_imag = discrete_pole_real2d(
        damping_x,
        frequency_x,
        spacing_x,
    )
    drive_x_real, drive_x_imag = _complex_multiply(
        excitation_real,
        excitation_imag,
        gamma_x_real,
        gamma_x_imag,
    )
    batch, height, width, _ = excitation_real.shape
    horizontal_real, horizontal_imag = _axis_scan(
        drive_x_real.reshape(batch * height, width, modes),
        drive_x_imag.reshape(batch * height, width, modes),
        decay_x_real,
        decay_x_imag,
        direction=direction_x,
        recurrence_backend=recurrence_backend,
    )
    horizontal_real = horizontal_real.reshape(batch, height, width, modes)
    horizontal_imag = horizontal_imag.reshape(batch, height, width, modes)

    decay_y_real, decay_y_imag, gamma_y_real, gamma_y_imag = discrete_pole_real2d(
        damping_y,
        frequency_y,
        spacing_y,
    )
    drive_y_real, drive_y_imag = _complex_multiply(
        horizontal_real,
        horizontal_imag,
        gamma_y_real,
        gamma_y_imag,
    )
    vertical_input_real = drive_y_real.permute(0, 2, 1, 3).reshape(
        batch * width,
        height,
        modes,
    )
    vertical_input_imag = drive_y_imag.permute(0, 2, 1, 3).reshape(
        batch * width,
        height,
        modes,
    )
    vertical_real, vertical_imag = _axis_scan(
        vertical_input_real,
        vertical_input_imag,
        decay_y_real,
        decay_y_imag,
        direction=direction_y,
        recurrence_backend=recurrence_backend,
    )
    output_shape = (batch, width, height, modes)
    return (
        vertical_real.reshape(output_shape).permute(0, 2, 1, 3).contiguous(),
        vertical_imag.reshape(output_shape).permute(0, 2, 1, 3).contiguous(),
    )


def _atlas(modes: int) -> tuple[Tensor, Tensor, Tensor]:
    orientations = torch.tensor((0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4))
    orientation = orientations[torch.arange(modes) % orientations.numel()]
    radial_level = torch.arange(modes) // orientations.numel()
    level_count = max(1, math.ceil(modes / orientations.numel()))
    if level_count == 1:
        radial_cycles = torch.ones(modes)
    else:
        levels = torch.logspace(math.log10(0.5), math.log10(6.0), level_count)
        radial_cycles = levels[radial_level]
    omega = 2.0 * math.pi * radial_cycles
    alpha = (omega / 3.0).clamp_min(0.5)
    return alpha, omega * torch.cos(orientation), omega * torch.sin(orientation)


class ProductPoleField2D(nn.Module):
    """Shared-pole four-quadrant product field over a channel-last feature map."""

    def __init__(
        self,
        model_dim: int,
        modes: int,
        *,
        fixed_atlas: bool = False,
        recurrence_backend: RecurrenceBackend = "auto",
    ) -> None:
        super().__init__()
        if model_dim < 2 * modes:
            message = "model_dim must be at least twice the mode count"
            raise ValueError(message)
        if modes < 1:
            message = "modes must be positive"
            raise ValueError(message)
        self.model_dim = model_dim
        self.modes = modes
        self.recurrence_backend: RecurrenceBackend = recurrence_backend
        self.analysis = nn.Linear(model_dim, 2 * modes, bias=False)
        nn.init.orthogonal_(self.analysis.weight)
        orthogonal(
            self.analysis,
            "weight",
            orthogonal_map="matrix_exp",
            use_trivialization=True,
        )
        alpha, omega_x, omega_y = _atlas(modes)
        minimum_damping = 1.0e-3
        inverse_softplus = torch.log(torch.expm1(alpha - minimum_damping))
        self.raw_damping_x = nn.Parameter(inverse_softplus.clone())
        self.raw_damping_y = nn.Parameter(inverse_softplus.clone())
        self.register_buffer("base_frequency_x", omega_x)
        self.register_buffer("base_frequency_y", omega_y)
        self.frequency_offset_x = nn.Parameter(torch.zeros(modes))
        self.frequency_offset_y = nn.Parameter(torch.zeros(modes))
        self.minimum_damping = minimum_damping
        self.frequency_offset_bound = math.pi / 8.0 if fixed_atlas else math.pi
        if fixed_atlas:
            self.raw_damping_x.requires_grad = False
            self.raw_damping_y.requires_grad = False

    def damping_x(self) -> Tensor:
        return self.minimum_damping + functional.softplus(self.raw_damping_x)

    def damping_y(self) -> Tensor:
        return self.minimum_damping + functional.softplus(self.raw_damping_y)

    def frequency_x(self) -> Tensor:
        return self.get_buffer("base_frequency_x") + self.frequency_offset_bound * torch.tanh(
            self.frequency_offset_x
        )

    def frequency_y(self) -> Tensor:
        return self.get_buffer("base_frequency_y") + self.frequency_offset_bound * torch.tanh(
            self.frequency_offset_y
        )

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        if inputs.ndim != 4 or inputs.shape[-1] != self.model_dim:
            message = f"product field requires [B,H,W,{self.model_dim}] inputs"
            raise ValueError(message)
        excitation_real, excitation_imag = self.analysis(inputs).chunk(2, dim=-1)
        spacing_x = 1.0 / inputs.shape[2]
        spacing_y = 1.0 / inputs.shape[1]
        quadrants = [
            product_pole_scan_2d(
                excitation_real,
                excitation_imag,
                damping_x=self.damping_x(),
                damping_y=self.damping_y(),
                frequency_x=self.frequency_x(),
                frequency_y=self.frequency_y(),
                spacing_x=spacing_x,
                spacing_y=spacing_y,
                direction_x=direction_x,
                direction_y=direction_y,
                recurrence_backend=self.recurrence_backend,
            )
            for direction_x, direction_y in _QUADRANTS
        ]
        return (
            torch.stack([state[0] for state in quadrants], dim=1),
            torch.stack([state[1] for state in quadrants], dim=1),
        )

    def synthesize(self, states_real: Tensor, states_imag: Tensor) -> Tensor:
        """Apply the tied real-frame transpose to quadrant-mean modal coordinates."""
        if states_real.shape != states_imag.shape or states_real.ndim != 5:
            message = "synthesis states must have identical [B,Q,H,W,M] shapes"
            raise ValueError(message)
        mean_real = states_real.mean(dim=1)
        mean_imag = states_imag.mean(dim=1)
        frame_real, frame_imag = self.analysis.weight.chunk(2, dim=0)
        return torch.matmul(mean_real, frame_real) + torch.matmul(mean_imag, frame_imag)

    def audit(self) -> dict[str, Tensor]:
        return {
            "damping_x": self.damping_x().detach(),
            "damping_y": self.damping_y().detach(),
            "frequency_x": self.frequency_x().detach(),
            "frequency_y": self.frequency_y().detach(),
        }


class _ProductPoleBlock(nn.Module):
    def __init__(
        self,
        model_dim: int,
        modes: int,
        *,
        mlp_ratio: float,
        fixed_atlas: bool,
        recurrence_backend: RecurrenceBackend,
        layer_scale_init: float,
    ) -> None:
        super().__init__()
        hidden_dim = max(model_dim, round(model_dim * mlp_ratio))
        self.local = nn.Conv2d(model_dim, model_dim, 3, padding=1, groups=model_dim)
        self.norm = nn.RMSNorm(model_dim)
        self.field = ProductPoleField2D(
            model_dim,
            modes,
            fixed_atlas=fixed_atlas,
            recurrence_backend=recurrence_backend,
        )
        self.pole_scale = nn.Parameter(torch.full((model_dim,), layer_scale_init))
        self.direct_scale = nn.Parameter(torch.zeros(model_dim))
        self.mlp_norm = nn.RMSNorm(model_dim)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, model_dim),
        )
        self.mlp_scale = nn.Parameter(torch.full((model_dim,), layer_scale_init))

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        local = self.local(inputs.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        normalized = self.norm(functional.silu(local))
        states_real, states_imag = self.field(normalized)
        pole_update = self.field.synthesize(states_real, states_imag)
        updated = inputs + self.pole_scale * (pole_update + self.direct_scale * normalized)
        output = updated + self.mlp_scale * self.mlp(self.mlp_norm(updated))
        return output, states_real, states_imag


def _window_slices(
    height: int,
    width: int,
    windows: SpatialWindows,
) -> tuple[tuple[slice, slice], ...]:
    global_window = (slice(0, height), slice(0, width))
    if windows == "global":
        return (global_window,)
    if windows != "global_2x2":
        message = f"unsupported spatial windows: {windows}"
        raise ValueError(message)
    middle_y = max(1, height // 2)
    middle_x = max(1, width // 2)
    return (
        global_window,
        (slice(0, middle_y), slice(0, middle_x)),
        (slice(0, middle_y), slice(middle_x, width)),
        (slice(middle_y, height), slice(0, middle_x)),
        (slice(middle_y, height), slice(middle_x, width)),
    )


def _lagged_pair(
    states: Tensor,
    delta_x: int,
    delta_y: int,
) -> tuple[Tensor, Tensor]:
    height, width = states.shape[2:4]
    if abs(delta_x) >= width or abs(delta_y) >= height:
        message = f"lag ({delta_x}, {delta_y}) exceeds a {height}x{width} window"
        raise ValueError(message)
    current_y = slice(max(delta_y, 0), height + min(delta_y, 0))
    previous_y = slice(max(-delta_y, 0), height - max(delta_y, 0))
    current_x = slice(max(delta_x, 0), width + min(delta_x, 0))
    previous_x = slice(max(-delta_x, 0), width - max(delta_x, 0))
    return (
        states[:, :, current_y, current_x, :],
        states[:, :, previous_y, previous_x, :],
    )


def _window_moments(
    states_real: Tensor,
    states_imag: Tensor,
    *,
    lags: tuple[SpatialLag, ...],
    normalize: bool,
) -> Tensor:
    energy = (states_real.square() + states_imag.square()).mean(dim=(2, 3))
    coordinates = [torch.log1p(energy)]
    epsilon = max(1.0e-8, torch.finfo(states_real.dtype).eps)
    for delta_x, delta_y in lags:
        current_real, previous_real = _lagged_pair(states_real, delta_x, delta_y)
        current_imag, previous_imag = _lagged_pair(states_imag, delta_x, delta_y)
        correlation_real = (current_real * previous_real + current_imag * previous_imag).mean(
            dim=(2, 3)
        )
        correlation_imag = (current_imag * previous_real - current_real * previous_imag).mean(
            dim=(2, 3)
        )
        if normalize:
            current_energy = (current_real.square() + current_imag.square()).mean(dim=(2, 3))
            previous_energy = (previous_real.square() + previous_imag.square()).mean(dim=(2, 3))
            denominator = torch.sqrt(
                (current_energy * previous_energy).clamp_min(epsilon * epsilon)
            )
            correlation_real = correlation_real / denominator
            correlation_imag = correlation_imag / denominator
        coordinates.extend((correlation_real, correlation_imag))
    return torch.stack(coordinates, dim=-1).flatten(start_dim=1)


def spatial_modal_moments(
    states_real: Tensor,
    states_imag: Tensor,
    *,
    lags: tuple[SpatialLag, ...] = _DEFAULT_LAGS,
    windows: SpatialWindows = "global_2x2",
    normalize: bool = True,
) -> Tensor:
    """Return fixed energy and complex vector-lag coordinates for each window."""
    if states_real.shape != states_imag.shape or states_real.ndim != 5:
        message = "modal states must have identical [B,Q,H,W,M] shapes"
        raise ValueError(message)
    descriptors = []
    for row_slice, column_slice in _window_slices(
        states_real.shape[2],
        states_real.shape[3],
        windows,
    ):
        descriptors.append(
            _window_moments(
                states_real[:, :, row_slice, column_slice, :],
                states_imag[:, :, row_slice, column_slice, :],
                lags=lags,
                normalize=normalize,
            )
        )
    return torch.cat(descriptors, dim=-1)


class Alphabet2D(nn.Module):
    """Flat ALPHABET-2D-T0 classifier with a terminal cascaded pole reader."""

    reader: ProductPoleField2D

    def __init__(self, config: Alphabet2DConfig) -> None:
        super().__init__()
        if config.input_channels < 1 or config.output_dim < 1:
            message = "input_channels and output_dim must be positive"
            raise ValueError(message)
        if config.image_size < config.patch_size or config.patch_size < 1:
            message = "patch_size must be positive and no larger than image_size"
            raise ValueError(message)
        if config.depth < 1:
            message = "depth must be positive"
            raise ValueError(message)
        if config.model_dim < 2 * config.modes:
            message = "model_dim must be at least twice the mode count"
            raise ValueError(message)
        if config.windows not in {"global", "global_2x2"}:
            message = f"unsupported spatial windows: {config.windows}"
            raise ValueError(message)
        self.config = config
        self.patch_embed = nn.Conv2d(
            config.input_channels,
            config.model_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.blocks = nn.ModuleList(
            [
                _ProductPoleBlock(
                    config.model_dim,
                    config.modes,
                    mlp_ratio=config.mlp_ratio,
                    fixed_atlas=config.fixed_direct_atlas,
                    recurrence_backend=config.recurrence_backend,
                    layer_scale_init=config.layer_scale_init,
                )
                for _ in range(config.depth)
            ]
        )
        self.reader_local = nn.Conv2d(
            config.model_dim,
            config.model_dim,
            3,
            padding=1,
            groups=config.model_dim,
        )
        self.reader_norm = nn.RMSNorm(config.model_dim)
        self.reader = ProductPoleField2D(
            config.model_dim,
            config.modes,
            recurrence_backend=config.recurrence_backend,
        )
        window_count = 1 if config.windows == "global" else 5
        coordinates_per_mode = 1 + 2 * len(config.lags)
        self.descriptor_dim = (
            2 * window_count * len(_QUADRANTS) * config.modes * coordinates_per_mode
        )
        self.classifier = nn.Linear(self.descriptor_dim, config.output_dim)

    def forward_features(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        if inputs.ndim != 4 or inputs.shape[1] != self.config.input_channels:
            message = f"ALPHABET-2D requires [B,{self.config.input_channels},H,W] inputs"
            raise ValueError(message)
        features = self.patch_embed(inputs).permute(0, 2, 3, 1)
        maximum_lag_x = max((abs(lag[0]) for lag in self.config.lags), default=0)
        maximum_lag_y = max((abs(lag[1]) for lag in self.config.lags), default=0)
        minimum_window_width = (
            features.shape[2] if self.config.windows == "global" else (features.shape[2] // 2)
        )
        minimum_window_height = (
            features.shape[1] if self.config.windows == "global" else (features.shape[1] // 2)
        )
        if minimum_window_width <= maximum_lag_x or minimum_window_height <= maximum_lag_y:
            message = "patch grid is too small for the configured spatial windows and lags"
            raise ValueError(message)
        direct_real: Tensor | None = None
        direct_imag: Tensor | None = None
        for block in self.blocks:
            features, direct_real, direct_imag = block(features)
        if direct_real is None or direct_imag is None:
            message = "ALPHABET-2D requires at least one direct block"
            raise RuntimeError(message)
        reader_input = self.reader_local(features.permute(0, 3, 1, 2)).permute(
            0,
            2,
            3,
            1,
        )
        reader_input = self.reader_norm(functional.silu(reader_input))
        reader_real, reader_imag = self.reader(reader_input)
        direct_descriptor = spatial_modal_moments(
            direct_real,
            direct_imag,
            lags=self.config.lags,
            windows=self.config.windows,
        )
        reader_descriptor = spatial_modal_moments(
            reader_real,
            reader_imag,
            lags=self.config.lags,
            windows=self.config.windows,
        )
        return torch.cat((direct_descriptor, reader_descriptor), dim=-1), features

    def forward(self, inputs: Tensor) -> Tensor:
        descriptor, _ = self.forward_features(inputs)
        return self.classifier(descriptor)

    def pole_audit(self) -> dict[str, dict[str, Tensor]]:
        audit = {
            f"direct_{index}": cast("_ProductPoleBlock", block).field.audit()
            for index, block in enumerate(self.blocks)
        }
        audit["reader"] = self.reader.audit()
        return audit
