from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final, Literal

import torch
from torch import Tensor, nn

from .pac_headroom_efficient_models import (
    _apply_raw_mask,  # pyright: ignore[reportPrivateUsage]
    _pair_mask,  # pyright: ignore[reportPrivateUsage]
    _weighted_haar,  # pyright: ignore[reportPrivateUsage]
    build_efficient_headroom_classifier,
)
from .pac_headroom_models import HEADROOM_SPECS, HeadroomPACClassifier
from .pac_metrics import count_parameters

if TYPE_CHECKING:
    from .pac_types import PACExperimentConfig

AblationVariant = Literal[
    "raw_shared",
    "low_only_fixed",
    "detail_only_fixed",
    "shared_two_band_fixed",
    "unshared_two_band_fixed",
    "alphabet_dual",
]

ABLATION_VARIANTS: Final[tuple[AblationVariant, ...]] = (
    "raw_shared",
    "low_only_fixed",
    "detail_only_fixed",
    "shared_two_band_fixed",
    "unshared_two_band_fixed",
    "alphabet_dual",
)


@dataclass(frozen=True, slots=True)
class AblationModelMetadata:
    variant: AblationVariant
    band_policy: str
    parameter_sharing: str
    training_origin_policy: str
    inference_origin_policy: str
    model_dim: int
    modes: int
    params_trainable: int
    target_params: int
    relative_param_error: float


class SingleBandPAC(nn.Module):
    """One fixed-origin Haar band followed by the unchanged ALPHABET core."""

    supports_observation_mask: Final[bool] = True
    supports_time_delta: Final[bool] = True

    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        *,
        band: Literal["low", "detail"],
    ) -> None:
        super().__init__()
        self.band = band
        pair_config = replace(config, sequence_length=(config.sequence_length + 1) // 2)
        self.core = HeadroomPACClassifier(
            pair_config,
            output_dim,
            HEADROOM_SPECS["B"],
            objective="classification",
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
        low, detail, pair_delta = _weighted_haar(stem_inputs, time_delta)
        selected = low if self.band == "low" else detail
        return self.core(
            selected,
            time_delta=pair_delta,
            observation_mask=_pair_mask(observation_mask),
            valid_mask=_pair_mask(valid_mask),
        )

    def post_optimizer_step(self) -> None:
        self.core.post_optimizer_step()

    def finalize_constraints(self) -> None:
        self.core.finalize_constraints()


class UnsharedTwoBandPAC(nn.Module):
    """Fixed-origin Haar bands with independent, capacity-matched ALPHABET cores."""

    supports_observation_mask: Final[bool] = True
    supports_time_delta: Final[bool] = True

    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        *,
        branch_model_dim: int,
        branch_modes: int,
    ) -> None:
        super().__init__()
        branch_config = replace(
            config,
            sequence_length=(config.sequence_length + 1) // 2,
            model_dim=branch_model_dim,
            modes=branch_modes,
        )
        self.low_core = HeadroomPACClassifier(
            branch_config,
            output_dim,
            HEADROOM_SPECS["B"],
            objective="classification",
        )
        self.detail_core = HeadroomPACClassifier(
            branch_config,
            output_dim,
            HEADROOM_SPECS["B"],
            objective="classification",
        )
        self.band_logits = nn.Parameter(torch.zeros(2))

    def forward(
        self,
        inputs: Tensor,
        *,
        time_delta: Tensor | None = None,
        observation_mask: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        stem_inputs = _apply_raw_mask(inputs, observation_mask, valid_mask)
        low, detail, pair_delta = _weighted_haar(stem_inputs, time_delta)
        pair_observation = _pair_mask(observation_mask)
        pair_valid = _pair_mask(valid_mask)
        low_logits = self.low_core(
            low,
            time_delta=pair_delta,
            observation_mask=pair_observation,
            valid_mask=pair_valid,
        )
        detail_logits = self.detail_core(
            detail,
            time_delta=pair_delta,
            observation_mask=pair_observation,
            valid_mask=pair_valid,
        )
        weights = torch.softmax(self.band_logits, dim=0).to(dtype=low_logits.dtype)
        return weights[0] * low_logits + weights[1] * detail_logits

    def post_optimizer_step(self) -> None:
        self.low_core.post_optimizer_step()
        self.detail_core.post_optimizer_step()

    def finalize_constraints(self) -> None:
        self.low_core.finalize_constraints()
        self.detail_core.finalize_constraints()


def build_retrained_ablation_model(
    variant: AblationVariant,
    config: PACExperimentConfig,
    output_dim: int,
    *,
    parameter_tolerance: float = 0.05,
) -> tuple[nn.Module, AblationModelMetadata]:
    if variant not in ABLATION_VARIANTS:
        message = f"unknown retrained ablation variant: {variant}"
        raise ValueError(message)
    target_model = build_efficient_headroom_classifier(
        "PA2WP", config, output_dim, objective="classification"
    )
    target_params = count_parameters(target_model)
    del target_model

    if variant == "raw_shared":
        model: nn.Module = HeadroomPACClassifier(
            config,
            output_dim,
            HEADROOM_SPECS["B"],
            objective="classification",
        )
        band_policy = "raw input; no pair transform"
        sharing = "single core"
        train_origin = infer_origin = "not applicable"
        active_dim, active_modes = config.model_dim, config.modes
    elif variant in {"low_only_fixed", "detail_only_fixed"}:
        band = "low" if variant == "low_only_fixed" else "detail"
        model = SingleBandPAC(config, output_dim, band=band)
        band_policy = f"{band} band only"
        sharing = "single core"
        train_origin = infer_origin = "fixed origin 0"
        active_dim, active_modes = config.model_dim, config.modes
    elif variant == "shared_two_band_fixed":
        model = build_efficient_headroom_classifier(
            "WP", config, output_dim, objective="classification"
        )
        band_policy = "low and detail"
        sharing = "one parameter-shared core"
        train_origin = infer_origin = "fixed origin 0"
        active_dim, active_modes = config.model_dim, config.modes
    elif variant == "alphabet_dual":
        model = build_efficient_headroom_classifier(
            "PA2WP", config, output_dim, objective="classification"
        )
        band_policy = "low and detail"
        sharing = "one parameter-shared core"
        train_origin = "uniform random origin 0/1 per forward"
        infer_origin = "mean logits from origins 0 and 1"
        active_dim, active_modes = config.model_dim, config.modes
    else:
        active_dim, active_modes = _match_unshared_capacity(config, output_dim, target_params)
        model = UnsharedTwoBandPAC(
            config,
            output_dim,
            branch_model_dim=active_dim,
            branch_modes=active_modes,
        )
        band_policy = "low and detail"
        sharing = "independent low/detail cores"
        train_origin = infer_origin = "fixed origin 0"

    params = count_parameters(model)
    relative_error = abs(params - target_params) / target_params
    if variant == "unshared_two_band_fixed" and relative_error > parameter_tolerance:
        message = (
            f"unshared control parameter error {relative_error:.4f} exceeds "
            f"tolerance {parameter_tolerance:.4f}"
        )
        raise RuntimeError(message)
    metadata = AblationModelMetadata(
        variant=variant,
        band_policy=band_policy,
        parameter_sharing=sharing,
        training_origin_policy=train_origin,
        inference_origin_policy=infer_origin,
        model_dim=active_dim,
        modes=active_modes,
        params_trainable=params,
        target_params=target_params,
        relative_param_error=relative_error,
    )
    return model, metadata


def _match_unshared_capacity(
    config: PACExperimentConfig, output_dim: int, target_params: int
) -> tuple[int, int]:
    best: tuple[float, int, int] | None = None
    minimum_dim = max(8, config.model_dim // 3)
    for model_dim in range(minimum_dim, config.model_dim + 1):
        center = max(1, min(config.modes, model_dim // 4))
        candidate_modes = range(max(1, center - 2), min(config.modes, model_dim // 4) + 1)
        for modes in candidate_modes:
            candidate = UnsharedTwoBandPAC(
                config,
                output_dim,
                branch_model_dim=model_dim,
                branch_modes=modes,
            )
            params = count_parameters(candidate)
            error = abs(params - target_params) / target_params
            score = (error, model_dim, modes)
            if best is None or score < best:
                best = score
            del candidate
    if best is None:
        message = "unable to construct an unshared two-band control"
        raise RuntimeError(message)
    return best[1], best[2]
