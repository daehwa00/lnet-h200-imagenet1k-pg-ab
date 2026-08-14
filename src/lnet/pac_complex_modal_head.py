from __future__ import annotations

from typing import TYPE_CHECKING, Literal, assert_never

import torch
from torch import Tensor

from .pac_real2d_math import discrete_pole_real2d
from .pac_recurrence import recurrence_real2d

if TYPE_CHECKING:
    from .pac_model import PACHybridPRLBlock

ComplexModalPooling = Literal["complex_stats", "hermitian", "complex_pyramid"]
ComplexModalSource = Literal["all", "last"]


def complex_modal_output_dim(
    *,
    model_dim: int,
    modes: int,
    depth: int,
    pooling: ComplexModalPooling,
    source: ComplexModalSource,
) -> int:
    modal_depth = depth if source == "all" else 1
    match pooling:
        case "complex_stats":
            return modal_depth * 7 * modes + 2 * model_dim
        case "hermitian":
            return modal_depth * modes * modes + 2 * model_dim
        case "complex_pyramid":
            return modal_depth * 7 * modes + 14 * model_dim
        case unreachable:
            assert_never(unreachable)


def complex_modal_pool(
    inputs: Tensor,
    blocks: tuple[PACHybridPRLBlock, ...],
    pooling: ComplexModalPooling,
    source: ComplexModalSource,
) -> tuple[Tensor, Tensor]:
    features = inputs
    summaries: list[Tensor] = []
    last_index = len(blocks) - 1
    for index, block in enumerate(blocks):
        if source == "all" or index == last_index:
            states = _block_states(block, features)
            summaries.append(_state_summary(states, pooling))
        features = block(features)
    return features, torch.cat(summaries, dim=-1)


def combine_complex_pool(
    features: Tensor, modal_summary: Tensor, pooling: ComplexModalPooling
) -> Tensor:
    match pooling:
        case "complex_stats" | "hermitian":
            return torch.cat((modal_summary, features.mean(dim=1), features.amax(dim=1)), dim=-1)
        case "complex_pyramid":
            return torch.cat((modal_summary, torch.cat(_pyramid_pool(features), dim=-1)), dim=-1)
        case unreachable:
            assert_never(unreachable)


def _block_states(block: PACHybridPRLBlock, inputs: Tensor) -> tuple[Tensor, Tensor]:
    projected = block.input_projection(inputs)
    prl = block.require_prl_branch()
    damping = prl.effective_damping_values(projected)
    frequency = prl.frequency_values().to(device=projected.device, dtype=damping.dtype)
    frequency = frequency.view(1, 1, prl.modes).expand_as(damping)
    decay_real, decay_imag, gamma_real, gamma_imag = discrete_pole_real2d(
        damping, frequency, prl.dt
    )
    instant_drive = torch.einsum("bnd,md->bnm", projected.to(dtype=prl.reader.dtype), prl.reader)
    tapped_drive = prl.tapped_drive_sequence(instant_drive)
    return recurrence_real2d(
        decay_real,
        decay_imag,
        gamma_real * tapped_drive,
        gamma_imag * tapped_drive,
        prl.recurrence_backend,
    )


def _state_summary(states: tuple[Tensor, Tensor], pooling: ComplexModalPooling) -> Tensor:
    real, imag = states
    match pooling:
        case "complex_stats" | "complex_pyramid":
            return _complex_stats(real, imag)
        case "hermitian":
            return _hermitian_features(real, imag)
        case unreachable:
            assert_never(unreachable)


def _complex_stats(real: Tensor, imag: Tensor) -> Tensor:
    energy = real.square() + imag.square()
    return torch.cat(
        (
            real[:, -1, :],
            imag[:, -1, :],
            real.mean(dim=1),
            imag.mean(dim=1),
            energy.mean(dim=1),
            torch.sqrt(energy.clamp_min(1.0e-12)).amax(dim=1),
            _phase_drift(real, imag),
        ),
        dim=-1,
    )


def _phase_drift(real: Tensor, imag: Tensor) -> Tensor:
    if real.shape[1] < 2:
        return torch.zeros(real.shape[0], real.shape[2], device=real.device, dtype=real.dtype)
    current_real = real[:, 1:, :]
    current_imag = imag[:, 1:, :]
    previous_real = real[:, :-1, :]
    previous_imag = imag[:, :-1, :]
    cross_real = current_real * previous_real + current_imag * previous_imag
    cross_imag = current_imag * previous_real - current_real * previous_imag
    return torch.atan2(cross_imag, cross_real).mean(dim=1)


def _hermitian_features(real: Tensor, imag: Tensor) -> Tensor:
    sequence_length = real.shape[1]
    cov_real = (
        torch.einsum("bnm,bnl->bml", real, real) + torch.einsum("bnm,bnl->bml", imag, imag)
    ) / sequence_length
    cov_imag = (
        torch.einsum("bnm,bnl->bml", imag, real) - torch.einsum("bnm,bnl->bml", real, imag)
    ) / sequence_length
    modes = real.shape[2]
    diag_index = torch.arange(modes, device=real.device)
    upper = torch.triu_indices(modes, modes, offset=1, device=real.device)
    return torch.cat(
        (
            cov_real[:, diag_index, diag_index],
            cov_real[:, upper[0], upper[1]],
            cov_imag[:, upper[0], upper[1]],
        ),
        dim=-1,
    )


def _pyramid_pool(features: Tensor) -> list[Tensor]:
    pooled: list[Tensor] = []
    for segments in (1, 2, 4):
        chunks = torch.tensor_split(features, segments, dim=1)
        for chunk in chunks:
            pooled.extend((chunk.mean(dim=1), chunk.amax(dim=1)))
    return pooled
