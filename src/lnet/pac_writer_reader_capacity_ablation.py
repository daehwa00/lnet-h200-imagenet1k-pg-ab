"""Capacity-matched writer/reader ablations for the identity ALPHABET.

The primary control removes the terminal pole reader and replaces the missing
capacity with an active classifier adapter.  The adapter is part of the
logit-producing path; it is not a bank of dormant parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
from torch import Tensor, nn
from torch.nn import functional

from .alphabet_backbone import AlphabetBackbone

if TYPE_CHECKING:
    from .pac_headroom_models import HeadroomObjective
    from .pac_types import PACExperimentConfig


WriterReaderVariant = Literal[
    "full",
    "one_scan_writer",
    "reader_lift_only",
    "reader_only",
    "pooled_real_only",
    "energy_only",
    "lag_only",
]

VARIANTS: tuple[WriterReaderVariant, ...] = (
    "full",
    "one_scan_writer",
    "reader_lift_only",
    "reader_only",
    "pooled_real_only",
    "energy_only",
    "lag_only",
)


@dataclass(frozen=True, slots=True)
class CapacityMatch:
    target_parameters: int
    actual_parameters: int
    relative_error: float
    adapter_width: int
    selected_feature_dim: int


def trainable_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


class _SelectiveActiveHead(nn.Module):
    """Select moment coordinates and use every added parameter in the logits."""

    def __init__(
        self,
        pooled_dim: int,
        modes: int,
        class_count: int,
        *,
        variant: WriterReaderVariant,
        adapter_width: int,
    ) -> None:
        super().__init__()
        self.use_modal_moments = variant != "pooled_real_only"
        self.use_backward_moments = variant in {
            "reader_only",
            "energy_only",
            "lag_only",
        }
        self.variant = variant
        self.modes = modes
        feature_dim = self.feature_dim(pooled_dim, modes, variant)
        self.classifier = nn.Linear(feature_dim, class_count)
        self.adapter_in = (
            nn.Linear(feature_dim, adapter_width) if adapter_width > 0 else None
        )
        self.adapter_out = (
            nn.Linear(adapter_width, class_count, bias=False)
            if adapter_width > 0
            else None
        )

    @staticmethod
    def feature_dim(
        pooled_dim: int,
        modes: int,
        variant: WriterReaderVariant,
    ) -> int:
        moment_dim = 7 * modes
        return {
            "one_scan_writer": pooled_dim + moment_dim,
            "reader_lift_only": pooled_dim + moment_dim,
            "reader_only": pooled_dim + moment_dim,
            "pooled_real_only": pooled_dim,
            "energy_only": pooled_dim + 2 * modes,
            "lag_only": pooled_dim + 12 * modes,
            "full": pooled_dim + 2 * moment_dim,
        }[variant]

    def _features(
        self,
        pooled: Tensor,
        forward_moments: Tensor,
        backward_moments: Tensor,
    ) -> Tensor:
        modes = self.modes
        if self.variant in {"one_scan_writer", "reader_lift_only"}:
            return torch.cat((pooled, forward_moments), dim=-1)
        if self.variant == "reader_only":
            return torch.cat((pooled, backward_moments), dim=-1)
        if self.variant == "pooled_real_only":
            return pooled
        if self.variant == "energy_only":
            return torch.cat(
                (
                    pooled,
                    forward_moments[..., :modes],
                    backward_moments[..., :modes],
                ),
                dim=-1,
            )
        if self.variant == "lag_only":
            return torch.cat(
                (
                    pooled,
                    forward_moments[..., modes:],
                    backward_moments[..., modes:],
                ),
                dim=-1,
            )
        return torch.cat((pooled, forward_moments, backward_moments), dim=-1)

    def forward(
        self,
        pooled: Tensor,
        forward_moments: Tensor,
        backward_moments: Tensor,
    ) -> Tensor:
        features = self._features(pooled, forward_moments, backward_moments)
        logits = self.classifier(features)
        if self.adapter_in is not None and self.adapter_out is not None:
            logits = logits + self.adapter_out(functional.silu(self.adapter_in(features)))
        return logits


class WriterReaderCapacityAblationPAC(AlphabetBackbone):
    """Identity ALPHABET with a deterministic, active capacity match."""

    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        *,
        variant: WriterReaderVariant,
        objective: HeadroomObjective = "classification",
        tolerance: float = 0.03,
    ) -> None:
        if variant not in VARIANTS:
            message = f"unknown writer/reader ablation variant: {variant}"
            raise ValueError(message)
        super().__init__(config, output_dim, objective=objective)
        self.variant = variant
        target = trainable_parameter_count(self)

        # These ablations deliberately use the common eager training path.  The
        # exact-split kernels assume the canonical two-reader graph.
        self.use_efp16_exact_split_training = False
        self.require_external_exact_split_training = False
        self.use_fused_efp16_inference_readout = False
        self.use_fused_rmsnorm_mean_training = False
        self.use_fused_rmsnorm_mean_backward_training = False
        self.use_d32_rmsnorm_backward_training = False
        self.use_fused_terminal_reader_local_training = False
        self.use_fused_terminal_reader_scan_training = False
        self.use_fused_writer_reader_local_training = False
        self.use_fused_writer_modal_reader_local_training = False

        if variant == "full":
            self.capacity_match = CapacityMatch(
                target,
                target,
                0.0,
                0,
                self.model_dim + 14 * self.modes,
            )
            return

        if variant in {"one_scan_writer", "reader_lift_only", "pooled_real_only"}:
            self.backward_block = nn.Identity()
        if variant == "one_scan_writer":
            self.second_local = nn.Identity()

        # Count the retained encoder before installing the replacement head.
        self.head = nn.Identity()
        retained = trainable_parameter_count(self)
        selected_dim = _SelectiveActiveHead.feature_dim(
            self.model_dim,
            self.modes,
            variant,
        )
        width, actual = self._choose_adapter_width(
            retained,
            selected_dim,
            output_dim,
            target,
        )
        self.head = _SelectiveActiveHead(
            self.model_dim,
            self.modes,
            output_dim,
            variant=variant,
            adapter_width=width,
        )
        actual = trainable_parameter_count(self)
        relative_error = (actual - target) / target
        if abs(relative_error) > tolerance:
            message = (
                f"{variant} capacity mismatch {relative_error:+.2%} exceeds "
                f"{tolerance:.2%}: target={target}, actual={actual}"
            )
            raise ValueError(message)
        self.capacity_match = CapacityMatch(
            target,
            actual,
            relative_error,
            width,
            selected_dim,
        )

    @staticmethod
    def _choose_adapter_width(
        retained_parameters: int,
        feature_dim: int,
        output_dim: int,
        target: int,
    ) -> tuple[int, int]:
        base_head = feature_dim * output_dim + output_dim
        per_hidden = feature_dim + 1 + output_dim
        candidates = range(1, 4097)
        width = min(
            candidates,
            key=lambda value: (
                abs(retained_parameters + base_head + value * per_hidden - target),
                value,
            ),
        )
        actual = retained_parameters + base_head + width * per_hidden
        if abs(actual - target) / target <= 0.03:
            return width, actual
        # A wide selected feature vector can already consume the full capacity.
        # Preserve the original positive-width choice whenever it is admissible;
        # use no adapter only for cells where H=1 itself breaches the tolerance.
        without_adapter = retained_parameters + base_head
        if abs(without_adapter - target) < abs(actual - target):
            return 0, without_adapter
        return width, actual

    def _reader_lift(
        self,
        first_stream: Tensor,
        active_valid: Tensor | None,
    ) -> Tensor:
        projected = self.second_projection(first_stream)
        if isinstance(self.second_local, nn.Identity):
            return projected
        pre_activation = self.second_local(projected.transpose(1, 2)).transpose(1, 2)
        return self._mask_features(functional.silu(pre_activation), active_valid)

    def forward(
        self,
        inputs: Tensor,
        *,
        time_delta: Tensor | None = None,
        observation_mask: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        if self.variant not in {
            "one_scan_writer",
            "reader_lift_only",
            "pooled_real_only",
        }:
            return super().forward(
                inputs,
                time_delta=time_delta,
                observation_mask=observation_mask,
                valid_mask=valid_mask,
            )
        first_local, active_delta, active_observation, active_valid = self._edge_stem(
            inputs,
            time_delta,
            observation_mask,
            valid_mask,
        )
        first_stream, first_moments = self._writer(
            first_local,
            active_delta,
            active_observation,
            active_valid,
        )
        encoded = self._reader_lift(first_stream, active_valid)
        empty_reader_moments = first_moments.new_zeros(first_moments.shape)
        return self._readout(
            encoded,
            first_moments,
            empty_reader_moments,
            active_valid,
        )

    @torch.no_grad()
    def post_optimizer_step(self) -> None:
        self.forward_block.retract_frame()
        if hasattr(self.backward_block, "retract_frame"):
            self.backward_block.retract_frame()
        stem = self.stem
        if hasattr(stem, "project_weight_"):
            stem.project_weight_()

    def finalize_constraints(self) -> None:
        self.forward_block.finalize_frame()
        if hasattr(self.backward_block, "finalize_frame"):
            self.backward_block.finalize_frame()


__all__ = [
    "VARIANTS",
    "CapacityMatch",
    "WriterReaderCapacityAblationPAC",
    "WriterReaderVariant",
    "trainable_parameter_count",
]
