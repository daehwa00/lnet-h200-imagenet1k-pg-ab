from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, log, sin
from typing import TYPE_CHECKING, assert_never

import torch
from torch import Tensor, nn

from lnet.hybrid import BRANCH_ORDER, HybridModalPRLBlock

if TYPE_CHECKING:
    from lnet.hybrid_delay_tasks import TeacherMetadata
    from lnet.tapped_prl import TappedPRLBranch


@dataclass(frozen=True, slots=True)
class HybridGateDiagnostic:
    mean_prl_weight: float
    mean_fir_weight: float
    mean_mlp_weight: float
    mean_gate_entropy: float
    prl_contribution_norm: float
    fir_contribution_norm: float
    mlp_contribution_norm: float
    normalized_prl_contribution_norm: float
    normalized_fir_contribution_norm: float
    normalized_mlp_contribution_norm: float
    gate_contribution_alignment: float
    scale_dominance_ratio: float


@dataclass(frozen=True, slots=True)
class TapDiagnostic:
    dominant_mode: int
    tap_peak_index: int
    tap_mass_near_true_delay: float | None
    tap_peak_error: int | None
    metadata_status: str


@dataclass(frozen=True, slots=True)
class PoleDiagnostic:
    mean_pole_error: float | None
    max_pole_error: float | None
    metadata_status: str
    matched_pole_count: int


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _mean_branch_values(values: Tensor) -> tuple[float, float, float]:
    means = values.detach().cpu().mean(dim=0)
    return (
        float(means[0].item()),
        float(means[1].item()),
        float(means[2].item()),
    )


def hybrid_gate_diagnostic(
    model: HybridModalPRLBlock,
    inputs: Tensor,
) -> HybridGateDiagnostic:
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        projected = model.input_projection(inputs.to(device=device))
        weights = model.branch_weights(projected)
        branch_outputs = model.branch_outputs(projected)
        contributions = weights.unsqueeze(-1) * torch.stack(branch_outputs, dim=2)
        normalized_outputs = tuple(_rms_normalize(output) for output in branch_outputs)
        normalized_contributions = weights.unsqueeze(-1) * torch.stack(normalized_outputs, dim=2)
        flattened_weights = weights.reshape(-1, len(BRANCH_ORDER))
        entropy = -(weights * torch.log(weights.clamp_min(1.0e-12))).sum(dim=-1) / log(
            len(BRANCH_ORDER),
        )
        contribution_norms = torch.linalg.vector_norm(contributions, dim=-1).mean(dim=(0, 1))
        normalized_norms = torch.linalg.vector_norm(normalized_contributions, dim=-1).mean(
            dim=(0, 1),
        )
        normalized_share = normalized_norms / normalized_norms.sum().clamp_min(1.0e-12)
        mean_weights = flattened_weights.mean(dim=0)
        alignment = torch.dot(mean_weights, normalized_share) / (
            torch.linalg.vector_norm(mean_weights)
            * torch.linalg.vector_norm(normalized_share).clamp_min(1.0e-12)
        )
        rms_values = torch.stack(
            [torch.sqrt(output.square().mean(dim=-1) + 1.0e-8) for output in branch_outputs],
            dim=-1,
        ).mean(dim=(0, 1))
        scale_dominance = rms_values.max() / rms_values.min().clamp_min(1.0e-12)
    mean_prl, mean_fir, mean_mlp = _mean_branch_values(flattened_weights)
    norm_prl, norm_fir, norm_mlp = _mean_branch_values(contribution_norms.view(1, -1))
    normalized_prl, normalized_fir, normalized_mlp = _mean_branch_values(
        normalized_norms.view(1, -1),
    )
    return HybridGateDiagnostic(
        mean_prl_weight=mean_prl,
        mean_fir_weight=mean_fir,
        mean_mlp_weight=mean_mlp,
        mean_gate_entropy=float(entropy.mean().detach().cpu().item()),
        prl_contribution_norm=norm_prl,
        fir_contribution_norm=norm_fir,
        mlp_contribution_norm=norm_mlp,
        normalized_prl_contribution_norm=normalized_prl,
        normalized_fir_contribution_norm=normalized_fir,
        normalized_mlp_contribution_norm=normalized_mlp,
        gate_contribution_alignment=float(alignment.detach().cpu().item()),
        scale_dominance_ratio=float(scale_dominance.detach().cpu().item()),
    )


def _rms_normalize(branch: Tensor) -> Tensor:
    rms = torch.sqrt(branch.square().mean(dim=-1, keepdim=True) + 1.0e-8)
    return branch / rms


def tapped_prl_tap_diagnostic(
    branch: TappedPRLBranch,
    teacher_metadata: TeacherMetadata,
) -> TapDiagnostic:
    tap_scores = branch.effective_tap_weights().detach().abs().cpu()
    output_residue = torch.complex(
        branch.output_residue_real.detach(),
        branch.output_residue_imag.detach(),
    )
    mode_weights = torch.linalg.vector_norm(output_residue, dim=1).cpu() * tap_scores.sum(dim=1)
    dominant_mode = int(torch.argmax(mode_weights).item())
    peak_index = int(torch.argmax(tap_scores[dominant_mode]).item())
    true_delay = teacher_metadata.true_delay
    if true_delay is None:
        return TapDiagnostic(
            dominant_mode=dominant_mode,
            tap_peak_index=peak_index,
            tap_mass_near_true_delay=None,
            tap_peak_error=None,
            metadata_status=teacher_metadata.metadata_status,
        )
    lower = max(0, true_delay - 1)
    upper = min(branch.tap_kernel_size - 1, true_delay + 1)
    total_mass = float(tap_scores[dominant_mode].sum().item())
    near_mass = float(tap_scores[dominant_mode, lower : upper + 1].sum().item())
    return TapDiagnostic(
        dominant_mode=dominant_mode,
        tap_peak_index=peak_index,
        tap_mass_near_true_delay=near_mass / max(total_mass, 1.0e-12),
        tap_peak_error=abs(peak_index - true_delay),
        metadata_status=teacher_metadata.metadata_status,
    )


def tapped_prl_pole_diagnostic(
    branch: TappedPRLBranch,
    teacher_metadata: TeacherMetadata,
) -> PoleDiagnostic:
    teacher_poles = _teacher_discrete_poles(teacher_metadata)
    if not teacher_poles:
        return PoleDiagnostic(
            mean_pole_error=None,
            max_pole_error=None,
            metadata_status=teacher_metadata.metadata_status,
            matched_pole_count=0,
        )
    learned_poles = tuple(torch.exp(branch.continuous_poles() * branch.dt).detach().cpu().tolist())
    matched_errors = _greedy_pole_errors(learned_poles=learned_poles, teacher_poles=teacher_poles)
    return PoleDiagnostic(
        mean_pole_error=sum(matched_errors) / len(matched_errors),
        max_pole_error=max(matched_errors),
        metadata_status=teacher_metadata.metadata_status,
        matched_pole_count=len(matched_errors),
    )


def _teacher_discrete_poles(teacher_metadata: TeacherMetadata) -> tuple[complex, ...]:
    match teacher_metadata.metadata_status:
        case "full_ground_truth" | "pole_only":
            pass
        case "delay_only" | "proxy_only":
            return ()
        case unreachable:
            assert_never(unreachable)
    radius = teacher_metadata.damping_radius
    omega = teacher_metadata.angular_frequency
    if radius is not None and omega is not None and omega != 0.0:
        base = complex(radius * cos(omega), radius * sin(omega))
        return (base, complex(base.real, -base.imag))
    real = teacher_metadata.discrete_pole_real
    imag = teacher_metadata.discrete_pole_imag
    if real is None or imag is None:
        return ()
    return (complex(real, imag),)


def _greedy_pole_errors(
    *,
    learned_poles: tuple[complex, ...],
    teacher_poles: tuple[complex, ...],
) -> tuple[float, ...]:
    remaining = list(learned_poles)
    errors: list[float] = []
    ordered_teacher_poles = sorted(
        teacher_poles,
        key=lambda pole: (-abs(pole), atan2(pole.imag, pole.real)),
    )
    for teacher_pole in ordered_teacher_poles:
        best_index = min(
            range(len(remaining)),
            key=lambda index: abs(remaining[index] - teacher_pole),
        )
        errors.append(abs(remaining.pop(best_index) - teacher_pole))
        if not remaining:
            break
    return tuple(errors)
