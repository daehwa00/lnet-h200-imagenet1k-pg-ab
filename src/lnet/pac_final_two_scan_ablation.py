"""Focused structural ablations for the final radial-log ALPHABET.

The controls in this module inherit the actual :class:`lnet.alphabet.Alphabet`
model used by the paper.  They therefore preserve the raw-input stem,
physical-time exact-ZOH dynamics, radial-log lag-(1,2,4) descriptor, and
modal-only affine task head.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final, Literal

import torch
from torch import Tensor, nn

from .alphabet import Alphabet
from .pac_laplace_native_input import _RawForcingStem

if TYPE_CHECKING:
    from .pac_headroom_models import HeadroomObjective
    from .pac_types import PACExperimentConfig


FinalTwoScanVariant = Literal[
    "full",
    "wider_one_scan",
    "no_synthesis",
    "shared_poles",
    "fixed_random_poles",
    "energy_only",
]

VARIANTS: Final[tuple[FinalTwoScanVariant, ...]] = (
    "full",
    "wider_one_scan",
    "no_synthesis",
    "shared_poles",
    "fixed_random_poles",
    "energy_only",
)


@dataclass(frozen=True, slots=True)
class CapacityAudit:
    variant: FinalTwoScanVariant
    target_parameters: int
    actual_parameters: int
    relative_error: float
    model_dim: int
    modes: int


def trainable_parameter_count(module: nn.Module) -> int:
    """Count parameters that are updated by the optimizer."""

    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def final_alphabet_parameter_formula(
    raw_input_dim: int,
    output_dim: int,
    model_dim: int,
    modes: int,
) -> int:
    """Exact trainable count of the final two-scan modal-only model."""

    return (
        raw_input_dim * model_dim
        + 16 * model_dim
        + 4 * model_dim * modes
        + 4 * modes
        + output_dim * (14 * modes + 1)
    )


def one_scan_parameter_formula(
    raw_input_dim: int,
    output_dim: int,
    model_dim: int,
    modes: int,
) -> int:
    """Exact count of a moments-only one-scan model with no synthesis path."""

    return (
        raw_input_dim * model_dim
        + 7 * model_dim
        + 2 * model_dim * modes
        + 2 * modes
        + output_dim * (7 * modes + 1)
    )


def choose_wider_one_scan(
    config: PACExperimentConfig,
    output_dim: int,
    target_parameters: int,
) -> tuple[int, int]:
    """Match both descriptor width and total count without a second scan.

    The full model exposes two ``7M`` modal summaries.  The strongest one-scan
    control therefore uses ``M'=2M``, giving the same number of head inputs,
    and adjusts ``D'`` to match the full model's total trainable count.
    """

    modes = 2 * config.modes
    minimum_dim = max(config.model_dim + 1, 2 * modes)
    candidates: list[tuple[float, int, int]] = []
    for model_dim in range(minimum_dim, 4 * config.model_dim + 1):
        actual = one_scan_parameter_formula(
            config.raw_input_dim,
            output_dim,
            model_dim,
            modes,
        )
        candidates.append(
            (
                abs(actual - target_parameters) / target_parameters,
                model_dim,
                actual,
            )
        )
    if not candidates:
        message = (
            "one-scan matching requires M'=2M, D'>D, and 2M'<=D'; "
            f"received D={config.model_dim}, M={config.modes}"
        )
        raise ValueError(message)
    _, model_dim, _ = min(candidates)
    return model_dim, modes


class _OneScanModalHead(nn.Module):
    """Affine head consuming only one bank's radial-log modal descriptor."""

    def __init__(self, modes: int, output_dim: int) -> None:
        super().__init__()
        self.modes = modes
        self.classifier = nn.Linear(7 * modes, output_dim)

    def forward(self, moments: Tensor) -> Tensor:
        return self.classifier(moments)


class _EnergyOnlyClassifier(nn.Module):
    """Nonlinear classifier with the dispatch attributes expected by ALPHABET."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.first = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.GELU()
        self.second = nn.Linear(hidden_dim, output_dim)

    @property
    def weight(self) -> Tensor:
        return self.second.weight

    @property
    def bias(self) -> Tensor | None:
        return self.second.bias

    def forward(self, inputs: Tensor) -> Tensor:
        return self.second(self.activation(self.first(inputs)))


class _EnergyOnlyHead(nn.Module):
    """Capacity-matched MLP over the two banks' lag-zero energies."""

    def __init__(self, modes: int, output_dim: int) -> None:
        super().__init__()
        input_dim = 2 * modes
        target = output_dim * (14 * modes + 1)
        hidden = max(1, round((target - output_dim) / (input_dim + output_dim + 1)))
        self.modes = modes
        self.mode_map = None
        self.classifier = _EnergyOnlyClassifier(input_dim, hidden, output_dim)

    def feature_group_slices(self) -> dict[str, tuple[slice, ...]]:
        return {"raw_modal": (slice(0, 2 * self.modes),), "mode_branch": ()}

    def forward(self, writer_moments: Tensor, reader_moments: Tensor) -> Tensor:
        energy = torch.cat(
            (writer_moments[:, : self.modes], reader_moments[:, : self.modes]),
            dim=-1,
        )
        return self.classifier(energy)


class FinalTwoScanAblation(Alphabet):
    """Final-model structural controls used by the confirmatory ablation."""

    def __init__(
        self,
        config: PACExperimentConfig,
        output_dim: int,
        *,
        variant: FinalTwoScanVariant,
        objective: HeadroomObjective = "classification",
        parameter_tolerance: float = 0.01,
        random_pole_seed: int = 17_071,
    ) -> None:
        if variant not in VARIANTS:
            message = f"unknown final two-scan ablation variant: {variant}"
            raise ValueError(message)

        target_parameters = final_alphabet_parameter_formula(
            config.raw_input_dim,
            output_dim,
            config.model_dim,
            config.modes,
        )
        if variant == "wider_one_scan":
            active_dim, active_modes = choose_wider_one_scan(
                config,
                output_dim,
                target_parameters,
            )
            active_config = replace(
                config,
                model_dim=active_dim,
                modes=active_modes,
            )
        else:
            active_config = config

        super().__init__(active_config, output_dim, objective=objective)
        self.variant = variant
        self.target_model_dim = config.model_dim
        self.target_modes = config.modes
        self._disable_specialized_paths()

        if variant == "wider_one_scan":
            self.backward_block = nn.Identity()
            self.second_local = nn.Identity()
            self._remove_writer_synthesis_parameters()
            self.head = _OneScanModalHead(self.modes, output_dim)  # type: ignore[assignment]
        elif variant == "no_synthesis":
            self._remove_writer_synthesis_parameters()
        elif variant == "shared_poles":
            self.backward_block.raw_decay = self.forward_block.raw_decay
            self.backward_block.raw_frequency = self.forward_block.raw_frequency
        elif variant == "fixed_random_poles":
            self._freeze_random_poles(random_pole_seed)
        elif variant == "energy_only":
            self.head = _EnergyOnlyHead(self.modes, output_dim)  # type: ignore[assignment]

        actual_parameters = trainable_parameter_count(self)
        relative_error = (actual_parameters - target_parameters) / target_parameters
        if variant == "wider_one_scan" and abs(relative_error) > parameter_tolerance:
            message = (
                f"wider one-scan parameter mismatch {relative_error:+.2%} exceeds "
                f"{parameter_tolerance:.2%}: target={target_parameters}, "
                f"actual={actual_parameters}"
            )
            raise ValueError(message)
        self.capacity_audit = CapacityAudit(
            variant=variant,
            target_parameters=target_parameters,
            actual_parameters=actual_parameters,
            relative_error=relative_error,
            model_dim=self.model_dim,
            modes=self.modes,
        )

    def _disable_specialized_paths(self) -> None:
        """Use the same eager mathematical path for every structural control."""

        for name in (
            "use_efp16_exact_split_training",
            "use_external_exact_split_training",
            "require_external_exact_split_training",
            "use_fused_efp16_inference_readout",
            "use_fused_rmsnorm_mean_training",
            "use_fused_rmsnorm_mean_backward_training",
            "use_d32_rmsnorm_backward_training",
            "use_fused_terminal_reader_local_training",
            "use_fused_terminal_reader_scan_training",
            "use_fused_terminal_reader_scan_inference",
            "use_fused_writer_reader_local_training",
            "use_fused_writer_modal_reader_local_training",
            "use_fused_writer_terminal_reader_inference",
            "use_fused_modal_affine_readout_inference",
        ):
            if hasattr(self, name):
                setattr(self, name, False)

    def _remove_writer_synthesis_parameters(self) -> None:
        """Remove trainable scales used only by the writer's real-stream output."""

        self.forward_block.register_parameter("direct_scale", None)
        self.forward_block.register_parameter("layer_scale", None)

    @torch.no_grad()
    def _freeze_random_poles(self, seed: int) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        for block in (self.forward_block, self.backward_block):
            decay = torch.empty_like(block.raw_decay).uniform_(
                -3.0,
                1.0,
                generator=generator,
            )
            bounded_frequency = torch.empty_like(block.raw_frequency).uniform_(
                -0.95,
                0.95,
                generator=generator,
            )
            block.raw_decay.copy_(decay)
            block.raw_frequency.copy_(torch.atanh(bounded_frequency))
            block.raw_decay.requires_grad_(requires_grad=False)
            block.raw_frequency.requires_grad_(requires_grad=False)

    def _writer_moments_only(
        self,
        first_local: Tensor,
        active_delta: Tensor | None,
        active_observation: Tensor | None,
        active_valid: Tensor | None,
    ) -> Tensor:
        return self.forward_block(
            first_local,
            time_delta=active_delta,
            observation_mask=active_observation,
            valid_mask=active_valid,
            metadata_prevalidated=True,
            return_moments_only=True,
        )

    def _represented_moments(
        self,
        moments: Tensor,
        block: object,
        *,
        metadata_free: bool,
    ) -> Tensor:
        return self._represent_moments(
            moments,
            block,
            metadata_free=metadata_free,
        )

    def forward(
        self,
        inputs: Tensor,
        *,
        time_delta: Tensor | None = None,
        observation_mask: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        if self.variant not in {"wider_one_scan", "no_synthesis"}:
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
        writer_moments = self._writer_moments_only(
            first_local,
            active_delta,
            active_observation,
            active_valid,
        )
        metadata_free = active_delta is None and active_observation is None and active_valid is None
        writer_representation = self._represented_moments(
            writer_moments,
            self.forward_block,
            metadata_free=metadata_free,
        )

        if self.variant == "wider_one_scan":
            head = self.head
            if not isinstance(head, _OneScanModalHead):
                message = "wider one-scan lost its modal-only head"
                raise RuntimeError(message)
            return head(writer_representation)

        reader_moments = self._terminal_reader_moments(
            first_local,
            active_delta,
            None,
            active_valid,
        )
        reader_representation = self._represented_moments(
            reader_moments,
            self.backward_block,
            metadata_free=metadata_free,
        )
        return self.head(writer_representation, reader_representation)

    @torch.no_grad()
    def post_optimizer_step(self) -> None:
        if self.variant != "wider_one_scan":
            super().post_optimizer_step()
            return
        self.forward_block.retract_frame()
        stem = self.stem
        if isinstance(stem, _RawForcingStem):
            stem.project_weight_()

    def finalize_constraints(self) -> None:
        if self.variant != "wider_one_scan":
            super().finalize_constraints()
            return
        self.forward_block.finalize_frame()


__all__ = [
    "VARIANTS",
    "CapacityAudit",
    "FinalTwoScanAblation",
    "FinalTwoScanVariant",
    "choose_wider_one_scan",
    "final_alphabet_parameter_formula",
    "one_scan_parameter_formula",
    "trainable_parameter_count",
]
