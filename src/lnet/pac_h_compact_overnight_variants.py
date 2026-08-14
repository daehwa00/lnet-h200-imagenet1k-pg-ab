"""Controlled overnight variants of tied H-compact lag-(1,2,4)."""

# ruff: noqa: EM101, EM102, TRY003

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

import torch
from torch import Tensor, nn
from torch.nn import functional

from .pac_h_compact_lag124_tied import HCompactLag124TiedPAC
from .pac_headroom_efficient_models import (
    _apply_raw_mask,  # pyright: ignore[reportPrivateUsage]
    _combined_edge_mask,  # pyright: ignore[reportPrivateUsage]
    _degree_normalized_edge_analysis,  # pyright: ignore[reportPrivateUsage]
    _edge_or_singleton_mask,  # pyright: ignore[reportPrivateUsage]
)
from .pac_tight_frame_models import (
    _InvariantMomentHead,  # pyright: ignore[reportPrivateUsage]
)

if TYPE_CHECKING:
    from .pac_headroom_models import HeadroomObjective
    from .pac_types import PACExperimentConfig


ActivationName = Literal["silu", "gelu"]


@dataclass(frozen=True, slots=True)
class OvernightVariantSpec:
    model_dim: int
    modes: int
    lags: tuple[int, ...] = (1, 2, 4)
    stem_dilation: int = 4
    reader_dilation: int = 4
    reader_activation: ActivationName = "silu"
    residual_head_width: int | None = None
    second_projection_rank: int | None = None
    identity_second_projection: bool = False


VARIANT_SPECS: Final[dict[str, OvernightVariantSpec]] = {
    "hco_control_d32m16": OvernightVariantSpec(32, 16),
    "hco_head_mlp32": OvernightVariantSpec(32, 16, residual_head_width=32),
    "hco_head_mlp64": OvernightVariantSpec(32, 16, residual_head_width=64),
    "hco_conv_stem1_reader1": OvernightVariantSpec(32, 16, stem_dilation=1, reader_dilation=1),
    "hco_conv_stem1_reader4": OvernightVariantSpec(32, 16, stem_dilation=1, reader_dilation=4),
    "hco_conv_stem4_reader1": OvernightVariantSpec(32, 16, stem_dilation=4, reader_dilation=1),
    "hco_lag12": OvernightVariantSpec(32, 16, lags=(1, 2)),
    "hco_lag1248": OvernightVariantSpec(32, 16, lags=(1, 2, 4, 8)),
    "hco_d128m16": OvernightVariantSpec(128, 16),
    "hco_d128m32": OvernightVariantSpec(128, 32),
    "hco_d128m64": OvernightVariantSpec(128, 64),
    "hco_reader_gelu": OvernightVariantSpec(32, 16, reader_activation="gelu"),
}
VARIANTS: Final = tuple(VARIANT_SPECS)
CONTROL_VARIANT: Final = "hco_control_d32m16"

LOW_RANK_VARIANT_SPECS: Final[dict[str, OvernightVariantSpec]] = {
    "hco_d128m16_projidentity": OvernightVariantSpec(
        128,
        16,
        identity_second_projection=True,
    ),
    "hco_d128m16_projrank8": OvernightVariantSpec(128, 16, second_projection_rank=8),
    "hco_d128m16_projrank16": OvernightVariantSpec(128, 16, second_projection_rank=16),
    "hco_d128m32_projidentity": OvernightVariantSpec(
        128,
        32,
        identity_second_projection=True,
    ),
    "hco_d128m32_projrank8": OvernightVariantSpec(128, 32, second_projection_rank=8),
    "hco_d128m32_projrank16": OvernightVariantSpec(128, 32, second_projection_rank=16),
}
LOW_RANK_VARIANTS: Final = tuple(LOW_RANK_VARIANT_SPECS)
IDENTITY_CAPACITY_VARIANT_SPECS: Final[dict[str, OvernightVariantSpec]] = {
    f"hco_identity_d{model_dim}m{modes}": OvernightVariantSpec(
        model_dim,
        modes,
        identity_second_projection=True,
    )
    for model_dim, modes in (
        (16, 8),
        (64, 4),
        (64, 16),
        (64, 32),
        (128, 16),
        (128, 32),
    )
}
IDENTITY_CAPACITY_VARIANTS: Final = tuple(IDENTITY_CAPACITY_VARIANT_SPECS)
ALL_VARIANT_SPECS: Final = {**VARIANT_SPECS, **LOW_RANK_VARIANT_SPECS}
ALL_VARIANT_SPECS.update(IDENTITY_CAPACITY_VARIANT_SPECS)


class _IdentityPlusLowRank(nn.Module):
    """Identity reader map plus a zero-initialized rank-constrained update."""

    def __init__(self, model_dim: int, rank: int) -> None:
        super().__init__()
        if not 0 < rank < model_dim:
            raise ValueError("rank must satisfy 0 < rank < model_dim")
        self.rank = rank
        self.down = nn.Linear(model_dim, rank, bias=False)
        self.up = nn.Linear(rank, model_dim, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs + self.up(self.down(inputs))


class _ResidualMomentHead(nn.Module):
    """Add a zero-initialized nonlinear residual to the canonical linear head."""

    def __init__(self, base: _InvariantMomentHead, hidden_dim: int) -> None:
        super().__init__()
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        self.use_modal_moments = base.use_modal_moments
        self.use_backward_moments = base.use_backward_moments
        self.classifier = base.classifier
        input_dim = self.classifier.in_features
        output_dim = self.classifier.out_features
        self.adapter_in = nn.Linear(input_dim, hidden_dim)
        self.adapter_out = nn.Linear(hidden_dim, output_dim, bias=False)
        nn.init.zeros_(self.adapter_out.weight)

    def _features(
        self,
        inputs: Tensor,
        forward_moments: Tensor,
        backward_moments: Tensor,
    ) -> Tensor:
        if not self.use_modal_moments:
            return inputs
        if self.use_backward_moments:
            return torch.cat((inputs, forward_moments, backward_moments), dim=-1)
        return torch.cat((inputs, forward_moments), dim=-1)

    def forward(
        self,
        inputs: Tensor,
        forward_moments: Tensor,
        backward_moments: Tensor,
    ) -> Tensor:
        features = self._features(inputs, forward_moments, backward_moments)
        residual = self.adapter_out(functional.silu(self.adapter_in(features)))
        return self.classifier(features) + residual


def _set_dilation(conv: nn.Conv1d, dilation: int) -> None:
    if dilation < 1:
        raise ValueError("dilation must be positive")
    kernel = conv.kernel_size[0]
    if kernel % 2 == 0:
        raise ValueError("same-length controlled convolution requires an odd kernel")
    conv.dilation = (dilation,)
    conv.padding = (dilation * (kernel - 1) // 2,)


def _replace_lag_head(model: HCompactLag124TiedPAC, lags: tuple[int, ...]) -> None:
    if not lags or any(lag < 1 for lag in lags) or tuple(sorted(set(lags))) != lags:
        raise ValueError(f"lags must be unique positive ascending integers: {lags}")
    if lags == (1, 2, 4):
        return
    old_head = model.head
    if not isinstance(old_head, _InvariantMomentHead):
        raise TypeError("lag replacement requires the canonical invariant-moment head")
    model_dim = model.model_dim
    modes = model.modes
    rng_state = torch.random.get_rng_state()
    new_head = _InvariantMomentHead(
        model_dim,
        modes,
        old_head.classifier.out_features,
        use_modal_moments=True,
        use_backward_moments=True,
        lags=lags,
    )
    torch.random.set_rng_state(rng_state)
    source_lags = (1, 2, 4)
    source_moment_dim = modes * (1 + 2 * len(source_lags))
    target_moment_dim = modes * (1 + 2 * len(lags))
    with torch.no_grad():
        new_head.classifier.weight.zero_()
        new_head.classifier.weight[:, :model_dim].copy_(old_head.classifier.weight[:, :model_dim])
        for bank in range(2):
            source_start = model_dim + bank * source_moment_dim
            target_start = model_dim + bank * target_moment_dim
            new_head.classifier.weight[:, target_start : target_start + modes].copy_(
                old_head.classifier.weight[:, source_start : source_start + modes]
            )
            for target_index, lag in enumerate(lags):
                if lag not in source_lags:
                    continue
                source_index = source_lags.index(lag)
                source = source_start + modes * (1 + 2 * source_index)
                target = target_start + modes * (1 + 2 * target_index)
                new_head.classifier.weight[:, target : target + 2 * modes].copy_(
                    old_head.classifier.weight[:, source : source + 2 * modes]
                )
        new_head.classifier.bias.copy_(old_head.classifier.bias)
    model.forward_block.moment_lags = lags
    model.backward_block.moment_lags = lags
    model.head = new_head


class HCompactOvernightPAC(HCompactLag124TiedPAC):
    """Tied H-compact with exactly one controlled overnight modification."""

    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        variant: str,
        *,
        objective: HeadroomObjective,
    ) -> None:
        try:
            spec = ALL_VARIANT_SPECS[variant]
        except KeyError as error:
            raise ValueError(f"unknown overnight variant: {variant}") from error
        if config.model_dim != spec.model_dim or config.modes != spec.modes:
            message = (
                f"{variant} requires D={spec.model_dim}, M={spec.modes}; got "
                f"D={config.model_dim}, M={config.modes}"
            )
            raise ValueError(message)
        super().__init__(config, output_dim, objective=objective)
        self.overnight_variant = variant
        self.reader_activation: ActivationName = spec.reader_activation
        if not isinstance(self.stem.local, nn.Conv1d):
            raise TypeError("overnight variants require the edge-frame Conv1d stem")
        _set_dilation(self.stem.local, spec.stem_dilation)
        _set_dilation(self.second_local, spec.reader_dilation)
        _replace_lag_head(self, spec.lags)
        if spec.identity_second_projection and spec.second_projection_rank is not None:
            raise ValueError("identity and low-rank second projections are mutually exclusive")
        if spec.identity_second_projection:
            self.second_projection = nn.Identity()
        elif spec.second_projection_rank is not None:
            self.second_projection = _IdentityPlusLowRank(
                self.model_dim,
                spec.second_projection_rank,
            )
        if spec.residual_head_width is not None:
            if not isinstance(self.head, _InvariantMomentHead):
                raise TypeError("residual head requires an invariant-moment base head")
            setattr(  # noqa: B010 - preserves nn.Module registration while widening the subtype.
                self,
                "head",
                _ResidualMomentHead(self.head, spec.residual_head_width),
            )
            self.use_fused_efp16_inference_readout = False

    def _activate_reader(self, inputs: Tensor) -> Tensor:
        if self.reader_activation == "silu":
            return functional.silu(inputs)
        return functional.gelu(inputs)

    def forward(
        self,
        inputs: Tensor,
        *,
        time_delta: Tensor | None = None,
        observation_mask: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        stem_inputs = _apply_raw_mask(inputs, observation_mask, valid_mask)
        level, detail, active_delta = _degree_normalized_edge_analysis(stem_inputs, time_delta)
        active_observation = _edge_or_singleton_mask(observation_mask)
        active_valid = _edge_or_singleton_mask(valid_mask)
        edge_mask = _combined_edge_mask(active_observation, active_valid)
        edge_features = torch.cat((level, detail), dim=-1)
        if edge_mask is not None:
            edge_features = edge_features * edge_mask.to(
                device=edge_features.device,
                dtype=edge_features.dtype,
            )
        first_local = self._mask_features(self.stem(edge_features), active_valid)
        first_stream, first_moments = self.forward_block(
            first_local,
            time_delta=active_delta,
            observation_mask=active_observation,
            valid_mask=active_valid,
        )
        second_projected = self.second_projection(first_stream)
        reader_pre_activation = self.second_local(second_projected.transpose(1, 2)).transpose(1, 2)
        encoded = self._mask_features(
            self._activate_reader(reader_pre_activation),
            active_valid,
        )
        second_moments = self.backward_block(
            encoded,
            time_delta=active_delta,
            observation_mask=active_observation,
            valid_mask=active_valid,
            return_moments_only=True,
        )
        return self._readout(encoded, first_moments, second_moments, active_valid)


def build_overnight_variant(
    config: PACExperimentConfig,
    output_dim: int,
    variant: str,
    *,
    objective: HeadroomObjective,
) -> HCompactOvernightPAC:
    return HCompactOvernightPAC(config, output_dim, variant, objective=objective)


__all__ = [
    "ALL_VARIANT_SPECS",
    "CONTROL_VARIANT",
    "IDENTITY_CAPACITY_VARIANTS",
    "IDENTITY_CAPACITY_VARIANT_SPECS",
    "LOW_RANK_VARIANTS",
    "LOW_RANK_VARIANT_SPECS",
    "VARIANTS",
    "VARIANT_SPECS",
    "HCompactOvernightPAC",
    "OvernightVariantSpec",
    "build_overnight_variant",
]
