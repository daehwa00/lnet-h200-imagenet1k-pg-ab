"""Complete, auditable writer/reader controls for the identity ALPHABET.

The historical seven-way ablation is intentionally left untouched.  This
module defines the larger confirmatory family requested after that campaign:
component controls, pole/frame controls, local-only controls, and
parameter-matched non-pole readers.

With the exception of ``log_energy_only``, every control is matched to the
trainable parameter count of the canonical model within three percent.
Capacity added for matching is always on the logit-producing path.  The strict
``log_energy_only`` diagnostic deliberately uses only the two banks' energy
coordinates and one linear classifier; adding a nonlinear capacity adapter
would invalidate that diagnostic.
"""

# pyright: reportArgumentType=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportCallIssue=false
# pyright: reportIncompatibleMethodOverride=false
# pyright: reportIncompatibleUnannotatedOverride=false

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final, Literal, cast

import torch
from torch import Tensor, nn
from torch.nn import functional
from torch.nn.utils import parametrize

from .alphabet_backbone import AlphabetBackbone

if TYPE_CHECKING:
    from .pac_headroom_models import HeadroomObjective
    from .pac_types import PACExperimentConfig


CompleteWriterReaderVariant = Literal[
    "full",
    "learned_poles",
    "one_scan_writer",
    "reader_statistics_removed",
    "energy_only",
    "log_energy_only",
    "lag_only",
    "pooled_real_only",
    "fixed_random_poles",
    "no_semi_orthogonality",
    "local_convolution_only",
    "wider_one_scan",
    "trajectory_mlp_reader",
    "trajectory_convolution_reader",
]

VARIANTS: Final[tuple[CompleteWriterReaderVariant, ...]] = (
    "full",
    "learned_poles",
    "one_scan_writer",
    "reader_statistics_removed",
    "energy_only",
    "log_energy_only",
    "lag_only",
    "pooled_real_only",
    "fixed_random_poles",
    "no_semi_orthogonality",
    "local_convolution_only",
    "wider_one_scan",
    "trajectory_mlp_reader",
    "trajectory_convolution_reader",
)

CAPACITY_MATCHED_VARIANTS: Final[tuple[CompleteWriterReaderVariant, ...]] = tuple(
    variant for variant in VARIANTS if variant != "log_energy_only"
)

_FeatureSelection = Literal[
    "full",
    "writer",
    "pooled",
    "energy",
    "log_energy",
    "lag",
]
_FLOP_ESTIMATOR_VERSION: Final = "writer_reader_major_ops.v1"


@dataclass(frozen=True, slots=True)
class CapacityMatch:
    """Parameter and estimated-FLOP audit attached to every constructed model."""

    target_parameters: int
    actual_parameters: int
    relative_error: float
    parameter_match_required: bool
    parameter_match_status: Literal["matched", "diagnostic_unmatched"]
    adapter_width: int
    sparse_adapter_parameters: int
    selected_feature_dim: int
    model_dim: int
    modes: int
    target_model_dim: int
    target_modes: int
    target_flops: int
    estimated_flops: int
    relative_flop_error: float
    flop_match_status: Literal["matched", "structural_exception"]
    flop_estimator: str
    structural_exception: str | None


def trainable_parameter_count(module: nn.Module) -> int:
    """Return parameters updated by the optimizer, excluding frozen controls."""

    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def _masked_mean(values: Tensor, valid_mask: Tensor | None) -> Tensor:
    if valid_mask is None:
        return values.mean(dim=1)
    active = valid_mask
    if active.ndim == 2:
        active = active.unsqueeze(-1)
    weights = active.to(device=values.device, dtype=values.dtype)
    numerator = (values * weights).sum(dim=1)
    denominator = weights.sum(dim=1).clamp_min(1.0)
    return numerator / denominator


class _ActiveFeatureHead(nn.Module):
    """Feature selector with an optional active residual capacity adapter."""

    def __init__(
        self,
        pooled_dim: int,
        modes: int,
        output_dim: int,
        *,
        selection: _FeatureSelection,
        adapter_width: int,
        sparse_adapter_parameters: int = 0,
    ) -> None:
        super().__init__()
        self.selection = selection
        self.modes = modes
        self.adapter_width = adapter_width
        feature_dim = self.feature_dim(pooled_dim, modes, selection)
        self.classifier = nn.Linear(feature_dim, output_dim)
        self.adapter_in = (
            nn.Linear(feature_dim, adapter_width)
            if adapter_width > 0
            else None
        )
        self.adapter_out = (
            nn.Linear(adapter_width, output_dim, bias=False)
            if adapter_width > 0
            else None
        )
        self.sparse_adapter = (
            nn.Parameter(torch.zeros(sparse_adapter_parameters))
            if sparse_adapter_parameters > 0
            else None
        )
        self.register_buffer(
            "sparse_feature_index",
            torch.arange(sparse_adapter_parameters, dtype=torch.long)
            % feature_dim,
            persistent=False,
        )
        self.register_buffer(
            "sparse_output_index",
            torch.arange(sparse_adapter_parameters, dtype=torch.long)
            % output_dim,
            persistent=False,
        )

    @staticmethod
    def feature_dim(
        pooled_dim: int,
        modes: int,
        selection: _FeatureSelection,
    ) -> int:
        return {
            "full": pooled_dim + 14 * modes,
            "writer": pooled_dim + 7 * modes,
            "pooled": pooled_dim,
            "energy": pooled_dim + 2 * modes,
            "log_energy": 2 * modes,
            "lag": pooled_dim + 12 * modes,
        }[selection]

    def _select(
        self,
        pooled: Tensor,
        writer_moments: Tensor,
        reader_moments: Tensor,
    ) -> Tensor:
        modes = self.modes
        if self.selection == "full":
            return torch.cat((pooled, writer_moments, reader_moments), dim=-1)
        if self.selection == "writer":
            return torch.cat((pooled, writer_moments), dim=-1)
        if self.selection == "pooled":
            return pooled
        if self.selection == "energy":
            return torch.cat(
                (
                    pooled,
                    writer_moments[..., :modes],
                    reader_moments[..., :modes],
                ),
                dim=-1,
            )
        if self.selection == "log_energy":
            return torch.cat(
                (
                    writer_moments[..., :modes],
                    reader_moments[..., :modes],
                ),
                dim=-1,
            )
        return torch.cat(
            (
                pooled,
                writer_moments[..., modes:],
                reader_moments[..., modes:],
            ),
            dim=-1,
        )

    def forward(
        self,
        pooled: Tensor,
        writer_moments: Tensor,
        reader_moments: Tensor,
    ) -> Tensor:
        features = self._select(pooled, writer_moments, reader_moments)
        logits = self.classifier(features)
        if self.adapter_in is not None and self.adapter_out is not None:
            logits = logits + self.adapter_out(
                functional.silu(self.adapter_in(features))
            )
        if self.sparse_adapter is not None:
            selected = features.index_select(-1, self.sparse_feature_index)
            weighted = selected * self.sparse_adapter.unsqueeze(0)
            correction = logits.new_zeros(logits.shape)
            correction.index_add_(1, self.sparse_output_index, weighted)
            logits = logits + correction
        return logits


class _TrajectoryMLPReader(nn.Module):
    """Pointwise trajectory reader with a pooled learned descriptor."""

    def __init__(self, model_dim: int, modes: int, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_projection = nn.Linear(model_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, model_dim)
        self.descriptor = nn.Linear(hidden_dim, 7 * modes)

    def forward(
        self,
        values: Tensor,
        valid_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        hidden = functional.silu(self.input_projection(values))
        encoded = functional.silu(self.output_projection(hidden))
        if valid_mask is not None:
            active = (
                valid_mask
                if valid_mask.ndim == 3
                else valid_mask.unsqueeze(-1)
            )
            encoded = encoded * active.to(
                device=encoded.device,
                dtype=encoded.dtype,
            )
        descriptor = self.descriptor(_masked_mean(hidden, valid_mask))
        return encoded, descriptor


class _TrajectoryConvolutionReader(nn.Module):
    """Local convolutional trajectory reader with a pooled descriptor."""

    def __init__(self, model_dim: int, modes: int, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.depthwise = nn.Conv1d(
            model_dim,
            model_dim,
            kernel_size=5,
            padding=2,
            groups=model_dim,
        )
        self.input_projection = nn.Conv1d(model_dim, hidden_dim, kernel_size=1)
        self.output_projection = nn.Conv1d(hidden_dim, model_dim, kernel_size=1)
        self.descriptor = nn.Linear(hidden_dim, 7 * modes)

    def forward(
        self,
        values: Tensor,
        valid_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        channels = values.transpose(1, 2)
        local = functional.silu(self.depthwise(channels))
        hidden = functional.silu(self.input_projection(local))
        encoded = functional.silu(self.output_projection(hidden)).transpose(1, 2)
        hidden_time = hidden.transpose(1, 2)
        if valid_mask is not None:
            active = (
                valid_mask
                if valid_mask.ndim == 3
                else valid_mask.unsqueeze(-1)
            )
            encoded = encoded * active.to(
                device=encoded.device,
                dtype=encoded.dtype,
            )
        descriptor = self.descriptor(_masked_mean(hidden_time, valid_mask))
        return encoded, descriptor


class _LocalConvolutionStack(nn.Module):
    """Pole-free local temporal stack used by the local-only control."""

    def __init__(self, model_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.depthwise = nn.Conv1d(
            model_dim,
            model_dim,
            kernel_size=5,
            padding=2,
            groups=model_dim,
        )
        self.input_projection = nn.Conv1d(model_dim, hidden_dim, kernel_size=1)
        self.output_projection = nn.Conv1d(hidden_dim, model_dim, kernel_size=1)

    def forward(self, values: Tensor, valid_mask: Tensor | None) -> Tensor:
        channels = values.transpose(1, 2)
        local = functional.silu(self.depthwise(channels))
        hidden = functional.silu(self.input_projection(local))
        encoded = functional.silu(self.output_projection(hidden)).transpose(1, 2)
        if valid_mask is None:
            return encoded
        active = valid_mask if valid_mask.ndim == 3 else valid_mask.unsqueeze(-1)
        return encoded * active.to(device=encoded.device, dtype=encoded.dtype)


def _canonical_parameter_formula(
    model_dim: int,
    modes: int,
    input_dim: int,
    output_dim: int,
) -> int:
    """Exact parameter formula for the canonical identity graph."""

    return (
        input_dim * model_dim
        + 17 * model_dim
        + 4 * model_dim * modes
        + 4 * modes
        + output_dim * (model_dim + 14 * modes + 1)
    )


def _one_scan_parameter_formula(
    model_dim: int,
    modes: int,
    input_dim: int,
    output_dim: int,
) -> int:
    """Exact count for a writer-only graph without a matching adapter."""

    return (
        input_dim * model_dim
        + 10 * model_dim
        + 2 * model_dim * modes
        + 2 * modes
        + output_dim * (model_dim + 7 * modes + 1)
    )


def _choose_wider_one_scan(
    config: PACExperimentConfig,
    output_dim: int,
    target: int,
) -> tuple[int, int]:
    """Choose genuinely wider D and M, never a head-only capacity shim."""

    minimum_dim = config.model_dim + 1
    maximum_dim = max(config.model_dim * 4, minimum_dim + 16)
    candidates: list[tuple[float, int, int, int]] = []
    for model_dim in range(minimum_dim, maximum_dim + 1):
        minimum_modes = config.modes + 1
        for modes in range(minimum_modes, model_dim // 2 + 1):
            actual = _one_scan_parameter_formula(
                model_dim,
                modes,
                config.raw_input_dim,
                output_dim,
            )
            candidates.append(
                (
                    abs(actual - target) / target,
                    model_dim * modes,
                    model_dim,
                    modes,
                )
            )
    if not candidates:
        message = (
            "wider_one_scan needs D'>D, M'>M, and 2M'<=D'; "
            f"received D={config.model_dim}, M={config.modes}"
        )
        raise ValueError(message)
    _, _, model_dim, modes = min(candidates)
    return model_dim, modes


def _choose_adapter_budget(
    retained_parameters: int,
    feature_dim: int,
    output_dim: int,
    target_parameters: int,
) -> tuple[int, int]:
    base = retained_parameters + feature_dim * output_dim + output_dim
    cost = feature_dim + output_dim + 1
    if base >= target_parameters:
        return 0, 0
    width = (target_parameters - base) // max(cost, 1)
    remainder = target_parameters - (base + width * cost)
    return width, remainder


def _reader_hidden_width(
    variant: Literal[
        "trajectory_mlp_reader",
        "trajectory_convolution_reader",
    ],
    model_dim: int,
    modes: int,
    retained_without_reader_and_head: int,
    full_head_parameters: int,
    target_parameters: int,
) -> int:
    fixed = (
        model_dim + 7 * modes
        if variant == "trajectory_mlp_reader"
        else 7 * model_dim + 7 * modes
    )
    slope = 2 * model_dim + 7 * modes + 1
    candidates = range(1, max(2, 2 * model_dim + 1))
    return min(
        candidates,
        key=lambda hidden: abs(
            retained_without_reader_and_head
            + full_head_parameters
            + fixed
            + hidden * slope
            - target_parameters
        ),
    )


def _local_hidden_width(
    model_dim: int,
    retained_without_stack_and_head: int,
    pooled_head_parameters: int,
    target_parameters: int,
) -> int:
    fixed = 7 * model_dim
    slope = 2 * model_dim + 1
    candidates = range(1, max(2, 4 * model_dim + 1))
    return min(
        candidates,
        key=lambda hidden: abs(
            retained_without_stack_and_head
            + pooled_head_parameters
            + fixed
            + hidden * slope
            - target_parameters
        ),
    )


def _head_macs(head: nn.Module) -> int:
    classifier = getattr(head, "classifier", None)
    if not isinstance(classifier, nn.Linear):
        return 0
    macs = classifier.in_features * classifier.out_features
    adapter_in = getattr(head, "adapter_in", None)
    adapter_out = getattr(head, "adapter_out", None)
    if isinstance(adapter_in, nn.Linear) and isinstance(adapter_out, nn.Linear):
        macs += (
            adapter_in.in_features * adapter_in.out_features
            + adapter_out.in_features * adapter_out.out_features
        )
    sparse_adapter = getattr(head, "sparse_adapter", None)
    if isinstance(sparse_adapter, nn.Parameter):
        macs += sparse_adapter.numel()
    return macs


def _canonical_flops(
    sequence_length: int,
    input_dim: int,
    model_dim: int,
    modes: int,
    output_dim: int,
) -> int:
    # Major multiply/add pairs: stem, local lifts, frame projections,
    # synthesis, recurrence updates, and the terminal linear head.
    token_macs = (
        input_dim * model_dim
        + 10 * model_dim
        + 6 * model_dim * modes
        + 24 * modes
    )
    head_macs = output_dim * (model_dim + 14 * modes)
    return 2 * (sequence_length * token_macs + head_macs)


def _estimated_flops(
    model: CompleteWriterReaderAblationPAC,
    sequence_length: int,
    input_dim: int,
) -> int:
    variant = model.variant
    model_dim = model.model_dim
    modes = model.modes
    stem = input_dim * model_dim + 5 * model_dim
    writer = 4 * model_dim * modes + 12 * modes
    reader = 2 * model_dim * modes + 12 * modes
    local = 5 * model_dim
    head = _head_macs(model.head)

    if variant in {
        "full",
        "learned_poles",
        "energy_only",
        "log_energy_only",
        "lag_only",
        "fixed_random_poles",
        "no_semi_orthogonality",
    }:
        token = stem + writer + local + reader
        extra = 0
    elif variant in {"one_scan_writer", "wider_one_scan"}:
        token = stem + writer
        extra = 0
    elif variant in {"reader_statistics_removed", "pooled_real_only"}:
        token = stem + writer + local
        extra = 0
    elif variant == "trajectory_mlp_reader":
        active = cast("_TrajectoryMLPReader", model.trajectory_reader)
        token = stem + writer + 2 * model_dim * active.hidden_dim
        extra = active.hidden_dim * 7 * modes
    elif variant == "trajectory_convolution_reader":
        active = cast("_TrajectoryConvolutionReader", model.trajectory_reader)
        token = (
            stem
            + writer
            + 5 * model_dim
            + 2 * model_dim * active.hidden_dim
        )
        extra = active.hidden_dim * 7 * modes
    else:
        active = cast("_LocalConvolutionStack", model.local_stack)
        token = stem + 5 * model_dim + 2 * model_dim * active.hidden_dim
        extra = 0
    return 2 * (sequence_length * token + extra + head)


class CompleteWriterReaderAblationPAC(AlphabetBackbone):
    """Identity ALPHABET with comprehensive writer/reader controls."""

    def __init__(  # noqa: C901, PLR0912, PLR0915
        self,
        config: PACExperimentConfig,
        output_dim: int,
        *,
        variant: CompleteWriterReaderVariant,
        objective: HeadroomObjective = "classification",
        tolerance: float = 0.03,
        flop_tolerance: float = 0.10,
        random_seed: int = 17_071,
    ) -> None:
        if variant not in VARIANTS:
            message = f"unknown complete writer/reader variant: {variant}"
            raise ValueError(message)

        target_model_dim = config.model_dim
        target_modes = config.modes
        if variant == "wider_one_scan":
            rng_state = torch.random.get_rng_state()
            reference = AlphabetBackbone(
                config,
                output_dim,
                objective=objective,
            )
            target_parameters = trainable_parameter_count(reference)
            del reference
            torch.random.set_rng_state(rng_state)
            wider_dim, wider_modes = _choose_wider_one_scan(
                config,
                output_dim,
                target_parameters,
            )
            active_config = replace(
                config,
                model_dim=wider_dim,
                modes=wider_modes,
            )
        else:
            active_config = config
            target_parameters = -1

        super().__init__(active_config, output_dim, objective=objective)
        self.variant = variant
        self.target_model_dim = target_model_dim
        self.target_modes = target_modes
        self.random_seed = random_seed
        self.trajectory_reader: nn.Module = nn.Identity()
        self.local_stack: nn.Module = nn.Identity()
        self._disable_optimized_paths()

        if target_parameters < 0:
            target_parameters = trainable_parameter_count(self)
        # The exact formula is kept as an independent audit of upstream graph
        # drift.  Do not silently use it if the canonical graph changes.
        formula_target = _canonical_parameter_formula(
            target_model_dim,
            target_modes,
            config.raw_input_dim,
            output_dim,
        )
        if formula_target != target_parameters:
            message = (
                "canonical parameter formula drifted: "
                f"measured={target_parameters}, formula={formula_target}"
            )
            raise RuntimeError(message)

        selection = self._selection_for(variant)
        if variant in {
            "one_scan_writer",
            "wider_one_scan",
        }:
            self.backward_block = nn.Identity()
            self.second_local = nn.Identity()
        elif variant in {
            "reader_statistics_removed",
            "pooled_real_only",
        }:
            self.backward_block = nn.Identity()
        elif variant in {
            "trajectory_mlp_reader",
            "trajectory_convolution_reader",
        }:
            self.backward_block = nn.Identity()
            self.second_local = nn.Identity()
        elif variant == "local_convolution_only":
            self.forward_block = nn.Identity()
            self.backward_block = nn.Identity()
            self.second_local = nn.Identity()

        if variant == "fixed_random_poles":
            self._freeze_random_poles(random_seed)
        elif variant == "no_semi_orthogonality":
            self._remove_semi_orthogonality()

        original_head = self.head
        self.head = nn.Identity()
        if variant == "log_energy_only":
            # No pooled-real normalization is part of this strict diagnostic.
            self.final_norm = nn.Identity()

        if variant in {
            "trajectory_mlp_reader",
            "trajectory_convolution_reader",
        }:
            retained = trainable_parameter_count(self)
            full_head_parameters = output_dim * (
                self.model_dim + 14 * self.modes + 1
            )
            hidden = _reader_hidden_width(
                variant,
                self.model_dim,
                self.modes,
                retained,
                full_head_parameters,
                target_parameters,
            )
            if variant == "trajectory_mlp_reader":
                self.trajectory_reader = _TrajectoryMLPReader(
                    self.model_dim,
                    self.modes,
                    hidden,
                )
            else:
                self.trajectory_reader = _TrajectoryConvolutionReader(
                    self.model_dim,
                    self.modes,
                    hidden,
                )
        elif variant == "local_convolution_only":
            retained = trainable_parameter_count(self)
            pooled_head_parameters = output_dim * (self.model_dim + 1)
            hidden = _local_hidden_width(
                self.model_dim,
                retained,
                pooled_head_parameters,
                target_parameters,
            )
            self.local_stack = _LocalConvolutionStack(self.model_dim, hidden)

        feature_dim = _ActiveFeatureHead.feature_dim(
            self.model_dim,
            self.modes,
            selection,
        )
        adapter_width = 0
        sparse_adapter_parameters = 0
        if variant in {"full", "learned_poles", "no_semi_orthogonality"}:
            # Preserve the canonical head and its initialization exactly.
            self.head = original_head
            adapter_width = 0
        elif variant in {"wider_one_scan", "log_energy_only"}:
            self.head = _ActiveFeatureHead(
                self.model_dim,
                self.modes,
                output_dim,
                selection=selection,
                adapter_width=0,
            )
            adapter_width = 0
            sparse_adapter_parameters = 0
        else:
            retained = trainable_parameter_count(self)
            adapter_width, sparse_adapter_parameters = _choose_adapter_budget(
                retained,
                feature_dim,
                output_dim,
                target_parameters,
            )
            self.head = _ActiveFeatureHead(
                self.model_dim,
                self.modes,
                output_dim,
                selection=selection,
                adapter_width=adapter_width,
                sparse_adapter_parameters=sparse_adapter_parameters,
            )

        if variant in {"full", "learned_poles", "no_semi_orthogonality"}:
            sparse_adapter_parameters = 0

        actual_parameters = trainable_parameter_count(self)
        relative_error = (actual_parameters - target_parameters) / target_parameters
        parameter_match_required = variant != "log_energy_only"
        if parameter_match_required and abs(relative_error) > tolerance:
            message = (
                f"{variant} capacity mismatch {relative_error:+.2%} exceeds "
                f"{tolerance:.2%}: target={target_parameters}, "
                f"actual={actual_parameters}"
            )
            raise ValueError(message)

        target_flops = _canonical_flops(
            config.sequence_length,
            config.raw_input_dim,
            target_model_dim,
            target_modes,
            output_dim,
        )
        estimated_flops = _estimated_flops(
            self,
            config.sequence_length,
            config.raw_input_dim,
        )
        relative_flop_error = (estimated_flops - target_flops) / target_flops
        structural_exception = self._structural_flop_exception(
            variant,
            relative_flop_error,
            flop_tolerance,
        )
        self.capacity_match = CapacityMatch(
            target_parameters=target_parameters,
            actual_parameters=actual_parameters,
            relative_error=relative_error,
            parameter_match_required=parameter_match_required,
            parameter_match_status=(
                "matched" if parameter_match_required else "diagnostic_unmatched"
            ),
            adapter_width=adapter_width,
            sparse_adapter_parameters=sparse_adapter_parameters,
            selected_feature_dim=feature_dim,
            model_dim=self.model_dim,
            modes=self.modes,
            target_model_dim=target_model_dim,
            target_modes=target_modes,
            target_flops=target_flops,
            estimated_flops=estimated_flops,
            relative_flop_error=relative_flop_error,
            flop_match_status=(
                "matched"
                if abs(relative_flop_error) <= flop_tolerance
                else "structural_exception"
            ),
            flop_estimator=_FLOP_ESTIMATOR_VERSION,
            structural_exception=structural_exception,
        )

    @staticmethod
    def _selection_for(
        variant: CompleteWriterReaderVariant,
    ) -> _FeatureSelection:
        if variant in {
            "full",
            "learned_poles",
            "fixed_random_poles",
            "no_semi_orthogonality",
            "trajectory_mlp_reader",
            "trajectory_convolution_reader",
        }:
            return "full"
        if variant in {
            "one_scan_writer",
            "wider_one_scan",
            "reader_statistics_removed",
        }:
            return "writer"
        if variant in {"pooled_real_only", "local_convolution_only"}:
            return "pooled"
        if variant == "energy_only":
            return "energy"
        if variant == "log_energy_only":
            return "log_energy"
        return "lag"

    def _disable_optimized_paths(self) -> None:
        # Exact/fused kernels assume the canonical two-pole graph.
        for name in (
            "use_efp16_exact_split_training",
            "require_external_exact_split_training",
            "use_fused_efp16_inference_readout",
            "use_fused_rmsnorm_mean_training",
            "use_fused_rmsnorm_mean_backward_training",
            "use_d32_rmsnorm_backward_training",
            "use_fused_terminal_reader_local_training",
            "use_fused_terminal_reader_scan_training",
            "use_fused_writer_reader_local_training",
            "use_fused_writer_modal_reader_local_training",
            "use_fused_efp16_stem_training",
            "use_fused_efp16_c2_stem_training",
        ):
            if hasattr(self, name):
                setattr(self, name, False)

    @torch.no_grad()
    def _freeze_random_poles(self, random_seed: int) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(random_seed)
        for block in (self.forward_block, self.backward_block):
            decay = torch.empty_like(block.raw_decay).uniform_(
                -3.0,
                1.0,
                generator=generator,
            )
            frequency = torch.empty_like(block.raw_frequency).uniform_(
                -0.95,
                0.95,
                generator=generator,
            )
            block.raw_decay.copy_(decay)
            block.raw_frequency.copy_(torch.atanh(frequency))
            block.raw_decay.requires_grad = False
            block.raw_frequency.requires_grad = False

    @torch.no_grad()
    def _remove_semi_orthogonality(self) -> None:
        stem = getattr(self.stem, "projection", None)
        if isinstance(stem, nn.Linear):
            nn.init.xavier_uniform_(stem.weight)
        for block in (self.forward_block, self.backward_block):
            if parametrize.is_parametrized(block.frame, "weight"):
                parametrize.remove_parametrizations(
                    block.frame,
                    "weight",
                    leave_parametrized=True,
                )
            block.frame_parameterization = "unconstrained"
            nn.init.xavier_uniform_(block.frame.weight)

    @staticmethod
    def _structural_flop_exception(
        variant: CompleteWriterReaderVariant,
        relative_error: float,
        tolerance: float,
    ) -> str | None:
        if abs(relative_error) <= tolerance:
            return None
        if variant == "log_energy_only":
            return (
                "strict 2M log-energy diagnostic intentionally has neither "
                "pooled-real features nor a nonlinear capacity adapter"
            )
        if variant == "wider_one_scan":
            return (
                "matching two-bank parameters with one bank requires larger "
                "D and M, so per-token frame work cannot simultaneously match"
            )
        if variant in {
            "one_scan_writer",
            "reader_statistics_removed",
            "pooled_real_only",
        }:
            return (
                "removed per-token recurrence capacity is matched in the active "
                "sequence-level head; this cannot match sequence-length FLOPs"
            )
        if variant == "local_convolution_only":
            return (
                "pole recurrence and finite local convolution have different "
                "per-token major-operation structure at equal parameters"
            )
        return (
            "the alternative reader is parameter matched, but its operator "
            "structure does not permit a simultaneous ten-percent FLOP match"
        )

    def _terminal_reader(
        self,
        first_stream: Tensor,
        active_delta: Tensor | None,
        active_observation: Tensor | None,
        active_valid: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        del active_observation
        if self.variant in {"one_scan_writer", "wider_one_scan"}:
            return first_stream, first_stream.new_zeros(
                first_stream.shape[0],
                7 * self.modes,
            )
        if self.variant in {
            "reader_statistics_removed",
            "pooled_real_only",
        }:
            encoded = functional.silu(
                self.second_local(first_stream.transpose(1, 2)).transpose(1, 2)
            )
            encoded = self._mask_features(encoded, active_valid)
            return encoded, encoded.new_zeros(encoded.shape[0], 7 * self.modes)
        if self.variant in {
            "trajectory_mlp_reader",
            "trajectory_convolution_reader",
        }:
            reader = cast(
                "_TrajectoryMLPReader | _TrajectoryConvolutionReader",
                self.trajectory_reader,
            )
            return reader(first_stream, active_valid)
        return super()._terminal_reader(
            first_stream,
            active_delta,
            None,
            active_valid,
        )

    def _can_fuse_writer_reader_local(
        self,
        first_local: Tensor,
        active_delta: Tensor | None,
        active_observation: Tensor | None,
        active_valid: Tensor | None,
    ) -> bool:
        del first_local, active_delta, active_observation, active_valid
        return False

    def _can_fuse_writer_modal_reader_local(
        self,
        first_local: Tensor,
        active_delta: Tensor | None,
        active_observation: Tensor | None,
        active_valid: Tensor | None,
    ) -> bool:
        del first_local, active_delta, active_observation, active_valid
        return False

    def _readout(
        self,
        inputs: Tensor,
        forward_moments: Tensor,
        backward_moments: Tensor,
        valid_mask: Tensor | None,
    ) -> Tensor:
        normalized = self.final_norm(inputs)
        pooled = _masked_mean(normalized, valid_mask)
        return self.head(pooled, forward_moments, backward_moments)

    def forward(
        self,
        inputs: Tensor,
        *,
        time_delta: Tensor | None = None,
        observation_mask: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        if self.variant != "local_convolution_only":
            return super().forward(
                inputs,
                time_delta=time_delta,
                observation_mask=observation_mask,
                valid_mask=valid_mask,
            )
        first_local, _, _, active_valid = self._edge_stem(
            inputs,
            time_delta,
            observation_mask,
            valid_mask,
        )
        active = cast("_LocalConvolutionStack", self.local_stack)
        encoded = active(first_local, active_valid)
        empty = encoded.new_zeros(encoded.shape[0], 7 * self.modes)
        return self._readout(encoded, empty, empty, active_valid)

    @torch.no_grad()
    def post_optimizer_step(self) -> None:
        if self.variant == "no_semi_orthogonality":
            return
        for block in (self.forward_block, self.backward_block):
            retract = getattr(block, "retract_frame", None)
            if callable(retract):
                retract()
        project = getattr(self.stem, "project_weight_", None)
        if callable(project):
            project()

    @torch.no_grad()
    def finalize_constraints(self) -> None:
        if self.variant == "no_semi_orthogonality":
            return
        for block in (self.forward_block, self.backward_block):
            finalize = getattr(block, "finalize_frame", None)
            if callable(finalize):
                finalize()


__all__ = [
    "CAPACITY_MATCHED_VARIANTS",
    "VARIANTS",
    "CapacityMatch",
    "CompleteWriterReaderAblationPAC",
    "CompleteWriterReaderVariant",
    "trainable_parameter_count",
]
