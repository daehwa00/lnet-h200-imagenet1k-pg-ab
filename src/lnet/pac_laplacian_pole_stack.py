from __future__ import annotations

from typing import TYPE_CHECKING, Final, Literal

import torch
from torch import Tensor, nn

from .pac_headroom_efficient_models import (
    EDGE_FRAME_VARIANT,
    _apply_raw_mask,  # pyright: ignore[reportPrivateUsage]
    _degree_normalized_edge_analysis,  # pyright: ignore[reportPrivateUsage]
    _edge_or_singleton_mask,  # pyright: ignore[reportPrivateUsage]
)
from .pac_tight_frame_models import (
    _BlockVariant,  # pyright: ignore[reportPrivateUsage]
    _TightFrameBlock,  # pyright: ignore[reportPrivateUsage]
)

if TYPE_CHECKING:
    from .pac_types import PACExperimentConfig


LaplacianPoleStackVariant = Literal[
    "laplacian_pole_stack_l1",
    "laplacian_pole_stack_l2",
    "laplacian_pole_stack_l3",
]
LAPLACIAN_POLE_STACK_VARIANTS: Final[tuple[LaplacianPoleStackVariant, ...]] = (
    "laplacian_pole_stack_l1",
    "laplacian_pole_stack_l2",
    "laplacian_pole_stack_l3",
)
MODE_BUDGETS: Final[dict[LaplacianPoleStackVariant, tuple[int, ...]]] = {
    "laplacian_pole_stack_l1": (16,),
    "laplacian_pole_stack_l2": (8, 8),
    "laplacian_pole_stack_l3": (6, 5, 5),
}
_SUMMARY_DIM: Final = 32
_EPSILON: Final = 1.0e-8


def _fix_core_layer_scale(core: _TightFrameBlock) -> None:
    layer_scale = core.layer_scale
    if layer_scale is None:
        message = "Laplacian-pole blocks require the coupled residual core"
        raise RuntimeError(message)
    delattr(core, "layer_scale")
    core.register_buffer("layer_scale", torch.ones_like(layer_scale))


def _metadata_3d(value: Tensor) -> Tensor:
    if value.ndim == 2:
        return value.unsqueeze(-1)
    if value.ndim != 3:
        message = "metadata must have shape [B,T] or [B,T,1]"
        raise ValueError(message)
    return value


def _edge_mask(mask: Tensor | None, *, like: Tensor) -> Tensor | None:
    if mask is None:
        return None
    return _metadata_3d(mask).to(device=like.device, dtype=like.dtype)


def _degree_normalized_edge_adjoint(
    level: Tensor,
    detail: Tensor,
    *,
    node_count: int,
    time_delta: Tensor | None,
) -> Tensor:
    """Apply the exact adjoint of ``_degree_normalized_edge_analysis``."""
    if level.shape != detail.shape:
        message = "level and detail edge messages must have identical shapes"
        raise ValueError(message)
    if node_count < 1:
        message = "node_count must be positive"
        raise ValueError(message)
    if node_count == 1:
        if level.shape[1] != 1:
            message = "singleton analysis must contain one coefficient"
            raise ValueError(message)
        return level
    if level.shape[1] != node_count - 1:
        message = "edge count must equal node_count - 1"
        raise ValueError(message)

    if time_delta is None:
        scale = level.new_tensor(1.0 / (2.0**0.5))
        first_weight: Tensor | float = scale
        second_weight: Tensor | float = scale
    else:
        delta = _metadata_3d(time_delta).clamp_min(0.0)
        first_delta = delta[:, :-1]
        second_delta = delta[:, 1:]
        total = first_delta + second_delta
        safe_total = total.clamp_min(torch.finfo(level.dtype).eps)
        first_weight = torch.sqrt(first_delta / safe_total).to(dtype=level.dtype)
        second_weight = torch.sqrt(second_delta / safe_total).to(dtype=level.dtype)
        empty = total <= 0.0
        equal = level.new_tensor(1.0 / (2.0**0.5))
        first_weight = torch.where(empty, equal, first_weight)
        second_weight = torch.where(empty, equal, second_weight)

    output = level.new_zeros(level.shape[0], node_count, level.shape[-1])
    output[:, :-1] += first_weight * level + second_weight * detail
    output[:, 1:] += second_weight * level - first_weight * detail
    degree = level.new_ones(node_count)
    if node_count > 2:
        degree[1:-1] = 2.0
    return output * degree.rsqrt().view(1, -1, 1)


def _last_valid_state(states: Tensor, mask: Tensor | None) -> Tensor:
    if mask is None:
        return states[:, -1]
    active = _metadata_3d(mask)
    lengths = active.squeeze(-1).sum(dim=1).long().clamp(min=1, max=states.shape[1])
    indices = lengths.sub(1).view(-1, 1, 1).expand(-1, 1, states.shape[-1])
    return states.gather(1, indices).squeeze(1)


def _masked_mean(values: Tensor, mask: Tensor | None) -> Tensor:
    if mask is None:
        return values.mean(dim=1)
    active = _metadata_3d(mask).to(device=values.device, dtype=values.dtype)
    numerator = (values * active).sum(dim=1)
    return numerator / active.sum(dim=1).clamp_min(1.0)


def _joint_modal_summary(
    level_real: Tensor,
    level_imag: Tensor,
    detail_real: Tensor,
    detail_imag: Tensor,
    mask: Tensor | None,
) -> Tensor:
    final = torch.cat(
        (
            _last_valid_state(level_real, mask),
            _last_valid_state(level_imag, mask),
            _last_valid_state(detail_real, mask),
            _last_valid_state(detail_imag, mask),
        ),
        dim=-1,
    )
    level_power = level_real.square() + level_imag.square()
    detail_power = detail_real.square() + detail_imag.square()
    level_energy = _masked_mean(level_power, mask)
    detail_energy = _masked_mean(detail_power, mask)
    cross_real = _masked_mean(
        level_real * detail_real + level_imag * detail_imag,
        mask,
    )
    cross_imag = _masked_mean(
        level_imag * detail_real - level_real * detail_imag,
        mask,
    )
    denominator = torch.sqrt(
        (level_energy * detail_energy).clamp_min(_EPSILON * _EPSILON)
    )
    coherence = torch.cat((cross_real / denominator, cross_imag / denominator), dim=-1)
    return torch.cat((final, level_energy, detail_energy, coherence), dim=-1)


class LaplacianPoleResidualBlock(nn.Module):
    """One symmetric path-edge analysis, exact-pole update, and adjoint synthesis."""

    def __init__(self, model_dim: int, modes: int) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.modes = modes
        self.level_projection = nn.Linear(model_dim, model_dim, bias=False)
        self.detail_projection = nn.Linear(model_dim, model_dim, bias=False)
        nn.init.orthogonal_(self.level_projection.weight)
        nn.init.orthogonal_(self.detail_projection.weight)
        self.level_core = _TightFrameBlock(
            model_dim,
            modes,
            _BlockVariant("forward", EDGE_FRAME_VARIANT),
        )
        self.detail_core = _TightFrameBlock(
            model_dim,
            modes,
            _BlockVariant("forward", EDGE_FRAME_VARIANT),
        )
        for core in (self.level_core, self.detail_core):
            core.use_input_norm = False
            core.norm = nn.Identity()  # pyright: ignore[reportAttributeAccessIssue]
            _fix_core_layer_scale(core)
        # Bands retain independent analysis/synthesis maps but use the same pole bank.
        self.detail_core.raw_decay = self.level_core.raw_decay
        self.detail_core.raw_frequency = self.level_core.raw_frequency
        self.residual_scale = nn.Parameter(torch.full((model_dim,), 1.0e-2))

    def forward(
        self,
        inputs: Tensor,
        *,
        time_delta: Tensor | None = None,
        observation_mask: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        level, detail, edge_delta = _degree_normalized_edge_analysis(inputs, time_delta)
        active_observation = _edge_or_singleton_mask(observation_mask)
        active_valid = _edge_or_singleton_mask(valid_mask)
        combined_mask = active_valid if active_observation is None else active_observation
        if active_observation is not None and active_valid is not None:
            combined_mask = torch.minimum(active_observation, active_valid)
        edge_weight = _edge_mask(combined_mask, like=level)
        if edge_weight is not None:
            level = level * edge_weight
            detail = detail * edge_weight

        level_input = self.level_projection(level)
        detail_input = self.detail_projection(detail)
        level_output, _, level_real, level_imag = self.level_core(
            level_input,
            time_delta=edge_delta,
            observation_mask=active_observation,
            valid_mask=active_valid,
            return_modal_states=True,
        )
        detail_output, _, detail_real, detail_imag = self.detail_core(
            detail_input,
            time_delta=edge_delta,
            observation_mask=active_observation,
            valid_mask=active_valid,
            return_modal_states=True,
        )
        level_message = level_output - level_input
        detail_message = detail_output - detail_input
        if edge_weight is not None:
            level_message = level_message * edge_weight
            detail_message = detail_message * edge_weight
        update = _degree_normalized_edge_adjoint(
            level_message,
            detail_message,
            node_count=inputs.shape[1],
            time_delta=time_delta,
        )
        output = inputs + self.residual_scale.view(1, 1, -1) * update
        node_mask = valid_mask if valid_mask is not None else observation_mask
        if node_mask is not None:
            active = _metadata_3d(node_mask).to(device=output.device, dtype=output.dtype)
            output = output * active
        summary = _joint_modal_summary(
            level_real,
            level_imag,
            detail_real,
            detail_imag,
            combined_mask,
        )
        return output, summary

    def post_optimizer_step(self) -> None:
        self.level_core.retract_frame()
        self.detail_core.retract_frame()

    def finalize_constraints(self) -> None:
        self.level_core.finalize_frame()
        self.detail_core.finalize_frame()


class LaplacianPoleResidualStackPAC(nn.Module):
    supports_observation_mask: Final[bool] = True
    supports_time_delta: Final[bool] = True

    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        *,
        mode_budget: tuple[int, ...],
    ) -> None:
        super().__init__()
        if sum(mode_budget) != 16:
            message = "all stack depths must use the fixed total pole budget of 16"
            raise ValueError(message)
        self.model_dim = config.model_dim
        self.mode_budget = mode_budget
        self.raw_projection = nn.Linear(config.raw_input_dim, config.model_dim)
        self.blocks = nn.ModuleList(
            LaplacianPoleResidualBlock(config.model_dim, modes) for modes in mode_budget
        )
        self.summary_projections = nn.ModuleList(
            nn.Linear(8 * modes, _SUMMARY_DIM) for modes in mode_budget
        )
        self.layer_logits = nn.Parameter(torch.zeros(len(mode_budget)))
        self.head = nn.Linear(config.model_dim + _SUMMARY_DIM, output_dim)

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
            active = _metadata_3d(node_mask).to(device=encoded.device, dtype=encoded.dtype)
            encoded = encoded * active
        summaries: list[Tensor] = []
        for block, projection in zip(self.blocks, self.summary_projections, strict=True):
            encoded, summary = block(
                encoded,
                time_delta=time_delta,
                observation_mask=observation_mask,
                valid_mask=valid_mask,
            )
            summaries.append(projection(summary))
        weights = torch.softmax(self.layer_logits, dim=0).to(dtype=encoded.dtype)
        dynamic = torch.stack(
            [weight * summary for weight, summary in zip(weights, summaries, strict=True)],
            dim=0,
        ).sum(dim=0)
        pooled = _masked_mean(encoded, node_mask)
        return self.head(torch.cat((pooled, dynamic), dim=-1))

    def post_optimizer_step(self) -> None:
        for block in self.blocks:
            block.post_optimizer_step()

    def finalize_constraints(self) -> None:
        for block in self.blocks:
            block.finalize_constraints()


def build_laplacian_pole_stack(
    variant: LaplacianPoleStackVariant,
    config: PACExperimentConfig,
    output_dim: int,
) -> LaplacianPoleResidualStackPAC:
    try:
        mode_budget = MODE_BUDGETS[variant]
    except KeyError as error:
        message = f"unknown Laplacian-pole stack variant: {variant}"
        raise ValueError(message) from error
    return LaplacianPoleResidualStackPAC(config, output_dim, mode_budget=mode_budget)
