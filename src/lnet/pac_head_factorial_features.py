from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, assert_never

import torch
from torch import Tensor, nn

from .pac_head_factorial_spec import LAGS, SEGMENTS, ModalFeature, Source
from .pac_real2d_math import discrete_pole_real2d
from .pac_recurrence import recurrence_real2d

if TYPE_CHECKING:
    from .pac_model import PACHybridPRLBlock


@dataclass(frozen=True, slots=True)
class BlockContext:
    block: PACHybridPRLBlock
    states_real: Tensor
    states_imag: Tensor
    damping: Tensor
    decay_abs: Tensor
    instant_drive: Tensor
    tapped_drive: Tensor
    projected: Tensor
    output: Tensor


def block_context(block: PACHybridPRLBlock, inputs: Tensor) -> BlockContext:
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
    states_real, states_imag = recurrence_real2d(
        decay_real,
        decay_imag,
        gamma_real * tapped_drive,
        gamma_imag * tapped_drive,
        prl.recurrence_backend,
    )
    return BlockContext(
        block,
        states_real,
        states_imag,
        damping,
        torch.sqrt(decay_real.square() + decay_imag.square()),
        instant_drive,
        tapped_drive,
        projected,
        block(inputs),
    )


def modal_summary(
    contexts: tuple[BlockContext, ...],
    source: Source,
    feature: ModalFeature,
    mix_logits: Tensor | None,
    attention: nn.Linear | None,
) -> Tensor:
    if source == "cross_block":
        return cross_summary(contexts[0], contexts[-1], feature)
    summaries = tuple(single_summary(context, feature, attention) for context in contexts)
    if source == "all_concat":
        return torch.cat(summaries, dim=-1)
    if source == "all_learned_mix":
        logits = _require_mix(mix_logits, len(summaries))
        weights = torch.softmax(logits, dim=0).view(1, len(summaries), 1)
        return (torch.stack(summaries, dim=1) * weights).sum(dim=1)
    return summaries[-1]


def single_summary(  # noqa: C901, PLR0912
    context: BlockContext, feature: ModalFeature, attention: nn.Linear | None
) -> Tensor:
    real = context.states_real
    imag = context.states_imag
    envelope = torch.sqrt((real.square() + imag.square()).clamp_min(1.0e-12))
    match feature:
        case "final_state":
            summary = torch.cat((real[:, -1], imag[:, -1], envelope[:, -1]), dim=-1)
        case "envelope_phase":
            phase = _phase_drift(real, imag)
            summary = torch.cat(
                (
                    envelope.mean(dim=1),
                    envelope.amax(dim=1),
                    envelope.std(dim=1, unbiased=False),
                    phase.mean(dim=1),
                    phase.std(dim=1, unbiased=False),
                ),
                dim=-1,
            )
        case "complex_stats":
            summary = torch.cat(
                (
                    real[:, -1],
                    imag[:, -1],
                    real.mean(dim=1),
                    imag.mean(dim=1),
                    envelope.square().mean(dim=1),
                    envelope.amax(dim=1),
                    envelope.std(dim=1, unbiased=False),
                    _phase_drift(real, imag).mean(dim=1),
                ),
                dim=-1,
            )
        case "hermitian":
            summary = covariance_features(real, imag)
        case "normalized_hermitian":
            summary = normalized_covariance_features(real, imag)
        case "lagged_hermitian":
            summary = torch.cat([lagged_covariance(real, imag, lag) for lag in LAGS], dim=-1)
        case "temporal_hermitian_pyramid":
            summary = _temporal_covariance(real, imag, lagged=False)
        case "lagged_temporal_hermitian":
            summary = _temporal_covariance(real, imag, lagged=True)
        case "modal_attention":
            summary = _modal_attention(real, imag, envelope, attention)
        case "drive_stats":
            summary = _drive_stats(context.instant_drive, context.tapped_drive)
        case "modal_dynamics":
            summary = _modal_dynamics(real, imag, envelope)
        case "hermitian_dynamics_lite":
            summary = _hermitian_dynamics(real, imag, envelope, lite=True)
        case "hermitian_dynamics":
            summary = _hermitian_dynamics(real, imag, envelope, lite=False)
        case "cross_block_hermitian" | "cross_block_lagged_hermitian":
            message = f"{feature} requires cross_block source"
            raise RuntimeError(message)
        case unreachable:
            assert_never(unreachable)
    return summary


def cross_summary(first: BlockContext, second: BlockContext, feature: ModalFeature) -> Tensor:
    match feature:
        case "cross_block_hermitian":
            return covariance_features(
                second.states_real, second.states_imag, first.states_real, first.states_imag
            )
        case "cross_block_lagged_hermitian":
            return torch.cat(
                [
                    lagged_cross_covariance(
                        second.states_real,
                        second.states_imag,
                        first.states_real,
                        first.states_imag,
                        lag,
                    )
                    for lag in LAGS
                ],
                dim=-1,
            )
        case unsupported:
            message = f"{unsupported} is not a cross-block feature"
            raise RuntimeError(message)


def covariance_features(
    real_a: Tensor, imag_a: Tensor, real_b: Tensor | None = None, imag_b: Tensor | None = None
) -> Tensor:
    real_right = real_a if real_b is None else real_b
    imag_right = imag_a if imag_b is None else imag_b
    length = real_a.shape[1]
    cov_real = (
        torch.einsum("bnm,bnl->bml", real_a, real_right)
        + torch.einsum("bnm,bnl->bml", imag_a, imag_right)
    ) / length
    cov_imag = (
        torch.einsum("bnm,bnl->bml", imag_a, real_right)
        - torch.einsum("bnm,bnl->bml", real_a, imag_right)
    ) / length
    modes = real_a.shape[2]
    diag = torch.arange(modes, device=real_a.device)
    upper = torch.triu_indices(modes, modes, offset=1, device=real_a.device)
    return torch.cat(
        (cov_real[:, diag, diag], cov_real[:, upper[0], upper[1]], cov_imag[:, upper[0], upper[1]]),
        dim=-1,
    )


def normalized_covariance_features(real: Tensor, imag: Tensor) -> Tensor:
    energy = (real.square() + imag.square()).mean(dim=1).clamp_min(1.0e-8)
    cov = covariance_features(real, imag)
    modes = real.shape[2]
    off = cov[:, modes:]
    upper = torch.triu_indices(modes, modes, offset=1, device=real.device)
    scale = torch.sqrt(energy[:, upper[0]] * energy[:, upper[1]]).repeat(1, 2)
    return torch.cat((energy, off / scale.clamp_min(1.0e-8)), dim=-1)


def lagged_covariance(real: Tensor, imag: Tensor, lag: int) -> Tensor:
    if real.shape[1] <= lag:
        return torch.zeros(real.shape[0], real.shape[2] ** 2, device=real.device, dtype=real.dtype)
    return covariance_features(real[:, lag:], imag[:, lag:], real[:, :-lag], imag[:, :-lag])


def lagged_cross_covariance(
    real_a: Tensor, imag_a: Tensor, real_b: Tensor, imag_b: Tensor, lag: int
) -> Tensor:
    if real_a.shape[1] <= lag:
        return torch.zeros(
            real_a.shape[0],
            real_a.shape[2] ** 2,
            device=real_a.device,
            dtype=real_a.dtype,
        )
    return covariance_features(real_a[:, lag:], imag_a[:, lag:], real_b[:, :-lag], imag_b[:, :-lag])


def _temporal_covariance(real: Tensor, imag: Tensor, *, lagged: bool) -> Tensor:
    chunks = [
        (real_chunk, imag_chunk)
        for parts in SEGMENTS
        for real_chunk, imag_chunk in zip(
            torch.tensor_split(real, parts, dim=1),
            torch.tensor_split(imag, parts, dim=1),
            strict=True,
        )
    ]
    if lagged:
        return torch.cat([lagged_covariance(r, i, lag) for r, i in chunks for lag in LAGS], dim=-1)
    return torch.cat([covariance_features(r, i) for r, i in chunks], dim=-1)


def _phase_drift(real: Tensor, imag: Tensor) -> Tensor:
    if real.shape[1] < 2:
        return torch.zeros_like(real)
    current_real = real[:, 1:]
    current_imag = imag[:, 1:]
    previous_real = real[:, :-1]
    previous_imag = imag[:, :-1]
    phase = torch.atan2(
        current_imag * previous_real - current_real * previous_imag,
        current_real * previous_real + current_imag * previous_imag,
    )
    return torch.nn.functional.pad(phase, (0, 0, 1, 0))


def _modal_dynamics(real: Tensor, imag: Tensor, envelope: Tensor) -> Tensor:
    velocity_real = torch.diff(real, dim=1)
    velocity_imag = torch.diff(imag, dim=1)
    amplitude_delta = torch.diff(envelope, dim=1)
    phase = _phase_drift(real, imag)[:, 1:]
    return torch.cat(
        (
            covariance_features(real, imag),
            covariance_features(velocity_real, velocity_imag),
            _real_second_moment(envelope),
            _real_second_moment(amplitude_delta),
            _real_second_moment(phase),
        ),
        dim=-1,
    )


def _hermitian_dynamics(real: Tensor, imag: Tensor, envelope: Tensor, *, lite: bool) -> Tensor:
    velocity_energy = _velocity_energy(real, imag)
    phase = _phase_drift(real, imag)[:, 1:]
    pieces = [
        covariance_features(real, imag),
        velocity_energy,
        phase.mean(dim=1),
        phase.std(dim=1, unbiased=False),
    ]
    if not lite:
        amplitude_delta = torch.diff(envelope, dim=1)
        pieces.extend(
            (
                amplitude_delta.square().mean(dim=1),
                envelope.mean(dim=1),
                envelope.amax(dim=1),
            )
        )
    return torch.cat(pieces, dim=-1)


def _velocity_energy(real: Tensor, imag: Tensor) -> Tensor:
    velocity_real = torch.diff(real, dim=1)
    velocity_imag = torch.diff(imag, dim=1)
    if velocity_real.shape[1] == 0:
        return torch.zeros(real.shape[0], real.shape[2], device=real.device, dtype=real.dtype)
    return velocity_real.square().add(velocity_imag.square()).mean(dim=1)


def _real_second_moment(values: Tensor) -> Tensor:
    if values.shape[1] == 0:
        return torch.zeros(
            values.shape[0],
            values.shape[2] ** 2,
            device=values.device,
            dtype=values.dtype,
        )
    moment = torch.einsum("bnm,bnl->bml", values, values) / values.shape[1]
    return moment.flatten(start_dim=1)


def _modal_attention(
    real: Tensor, imag: Tensor, envelope: Tensor, attention: nn.Linear | None
) -> Tensor:
    if attention is None:
        message = "modal attention layer is not initialized"
        raise RuntimeError(message)
    tokens = torch.cat((real, imag, envelope), dim=-1)
    weights = torch.softmax(attention(tokens), dim=1)
    return (tokens * weights).sum(dim=1)


def _drive_stats(instant: Tensor, tapped: Tensor) -> Tensor:
    return torch.cat(
        (
            instant.mean(dim=1),
            instant.amax(dim=1),
            instant.std(dim=1, unbiased=False),
            tapped.mean(dim=1),
            tapped.amax(dim=1),
            tapped.std(dim=1, unbiased=False),
        ),
        dim=-1,
    )


def _require_mix(mix_logits: Tensor | None, count: int) -> Tensor:
    if mix_logits is None or mix_logits.shape[0] < count:
        message = "block mix logits are not initialized"
        raise RuntimeError(message)
    return mix_logits[:count]
