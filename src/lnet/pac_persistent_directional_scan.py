"""Persistent D4 memory stages with one matching scan per direction."""

from __future__ import annotations

# The positive-only scan reuses the established recompute backward primitive.
# pyright: reportArgumentType=false, reportCallIssue=false, reportMissingParameterType=false
# pyright: reportPrivateUsage=false
# ruff: noqa: ANN001, EM101, N803, SLF001, TRY003
from typing import TYPE_CHECKING, Protocol, cast

import torch
import triton
import triton.language as tl
from torch import Tensor, nn
from torch.library import triton_op, wrap_triton

from .pac_kernel_launch_config import (
    LaunchGeometry,
    autotuned,
    make_launch_scope,
    register_default,
)
from .pac_product_scan_contracts import DEFAULT_EPSILON
from .pac_product_scan_normalization import static_variance_tables
from .pac_product_scan_pipeline import run_product_scan_pipeline
from .pac_product_scan_reference import bidirectional_product_scan_reference
from .pac_triton_bidirectional_product_scan import (
    _compose_complex,
    _recomputed_backward_op,
    _validate,
)

if TYPE_CHECKING:
    from .complex_scan_stage import ComplexScanStage
    from .pac_phase_gated_transition import (
        PhaseGatedModePathResidualComplexLinearCollapse,
        PhaseGatedS2DPostFusionTransition,
    )
    from .pac_product_scan_contracts import ProductGainNormalization

ComplexField = tuple[Tensor, Tensor]
Pole = tuple[Tensor, Tensor, Tensor, Tensor]

POSITIVE_FORWARD_LAUNCH_NAME = "positive_product_scan_forward"
_LAUNCH_CANDIDATES = tuple(
    LaunchGeometry.build(num_warps=warps, blocks={"BLOCK_MODES": block_modes})
    for block_modes, warps in (
        (8, 4),
        (8, 8),
        (16, 4),
        (16, 8),
        (32, 4),
        (32, 8),
        (64, 8),
    )
)
register_default(
    POSITIVE_FORWARD_LAUNCH_NAME,
    LaunchGeometry.build(num_warps=4, blocks={"BLOCK_MODES": 16}),
    candidates=_LAUNCH_CANDIDATES,
)


class _AutogradContext(Protocol):
    saved_tensors: tuple[Tensor, ...]

    def save_for_backward(self, *tensors: Tensor) -> None: ...


@triton.jit
def _positive_forward_kernel(
    decay_real,
    decay_imag,
    gamma_real,
    gamma_imag,
    source_real,
    source_imag,
    output_real,
    output_imag,
    height: int,
    width: int,
    line_count: int,
    modes: int,
    BLOCK_WIDTH: tl.constexpr,
    BLOCK_MODES: tl.constexpr,
) -> None:
    line = tl.program_id(0)
    batch = line // height
    y = line - batch * height
    x = tl.arange(0, BLOCK_WIDTH)[:, None]
    mode = tl.program_id(1) * BLOCK_MODES + tl.arange(0, BLOCK_MODES)[None, :]
    active = (line < line_count) & (x < width) & (mode < modes)
    offset = ((batch * height + y) * width + x) * modes + mode
    ar = tl.load(decay_real + mode, mask=mode < modes, other=0.0).to(tl.float32)
    ai = tl.load(decay_imag + mode, mask=mode < modes, other=0.0).to(tl.float32)
    gr = tl.load(gamma_real + mode, mask=mode < modes, other=0.0).to(tl.float32)
    gi = tl.load(gamma_imag + mode, mask=mode < modes, other=0.0).to(tl.float32)
    sr = tl.load(source_real + offset, mask=active, other=0.0).to(tl.float32)
    si = tl.load(source_imag + offset, mask=active, other=0.0).to(tl.float32)
    result = tl.associative_scan(
        (
            tl.where(active, ar, 1.0),
            tl.where(active, ai, 0.0),
            gr * sr - gi * si,
            gr * si + gi * sr,
        ),
        axis=0,
        combine_fn=_compose_complex,
    )
    tl.store(output_real + offset, result[2], mask=active)
    tl.store(output_imag + offset, result[3], mask=active)


@triton_op("lnet::pac_positive_product_scan", mutates_args={})
def _positive_forward_op(
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
    source_real: Tensor,
    source_imag: Tensor,
) -> ComplexField:
    pole = decay_real, decay_imag, gamma_real, gamma_imag
    source = source_real, source_imag
    _validate(*pole, *source)
    if not source_real.is_cuda:
        return bidirectional_product_scan_reference(pole, source)[:2]
    outputs = torch.empty_like(source_real), torch.empty_like(source_imag)
    batch, height, width, modes = source_real.shape
    line_count = batch * height
    scope = make_launch_scope(
        _positive_forward_kernel,
        source_real,
        shape={
            "batch": batch,
            "height": height,
            "width": width,
            "modes": modes,
        },
    )
    kernel = autotuned(
        _positive_forward_kernel,
        POSITIVE_FORWARD_LAUNCH_NAME,
        key=("height", "width", "line_count", "modes"),
        scope=scope,
    )

    def grid(metadata: dict[str, int]) -> tuple[int, int]:
        return line_count, int(triton.cdiv(modes, metadata["BLOCK_MODES"]))

    wrap_triton(kernel)[grid](
        *(value.contiguous() for value in (*pole, *source)),
        *outputs,
        height,
        width,
        line_count,
        modes,
        BLOCK_WIDTH=triton.next_power_of_2(width),
    )
    return outputs


def _setup_positive_context(
    ctx: _AutogradContext,
    inputs: tuple[Tensor, ...],
    output: ComplexField,
) -> None:
    del output
    ctx.save_for_backward(*inputs)


def _positive_backward(
    ctx: _AutogradContext,
    grad_real: Tensor | None,
    grad_imag: Tensor | None,
) -> tuple[Tensor | None, ...]:
    inputs = ctx.saved_tensors
    source_real = inputs[4]
    active_gradients = (
        torch.zeros_like(source_real) if grad_real is None else grad_real.contiguous(),
        torch.zeros_like(source_real) if grad_imag is None else grad_imag.contiguous(),
    )
    if not source_real.is_cuda:
        differentiable = tuple(value.detach().requires_grad_() for value in inputs)
        with torch.enable_grad():
            outputs = bidirectional_product_scan_reference(
                cast("Pole", differentiable[:4]),
                cast("ComplexField", differentiable[4:]),
            )[:2]
            return torch.autograd.grad(outputs, differentiable, active_gradients)
    zero = torch.zeros_like(source_real)
    return _recomputed_backward_op(*inputs, *active_gradients, zero, zero)


torch.library.register_autograd(
    "lnet::pac_positive_product_scan",
    _positive_backward,
    setup_context=_setup_positive_context,
)


def positive_product_scan(pole: Pole, source: ComplexField) -> ComplexField:
    """Run one forward associative recurrence without materializing a reverse path."""
    return _positive_forward_op(*pole, *source)


def _conjugate_pole(pole: Pole) -> Pole:
    return pole[0], -pole[1], pole[2], -pole[3]


def _pack_directions(value: Tensor, first: int, second: int) -> Tensor:
    batch, height, width, _, modes = value.shape
    selected = torch.stack((value[..., first, :], value[..., second, :]), dim=1)
    return selected.reshape(2 * batch, height, width, modes)


def _unpack_pair(value: Tensor, batch: int) -> tuple[Tensor, Tensor]:
    return cast("tuple[Tensor, Tensor]", value.reshape(batch, 2, *value.shape[1:]).unbind(1))


def matching_direction_states(
    pole_x: Pole,
    pole_y: Pole,
    source: ComplexField,
) -> ComplexField:
    """Apply (++,-+,+-,--) scans to their matching persistent branches only."""
    source_real, source_imag = source
    if (
        source_real.shape != source_imag.shape
        or source_real.ndim != 5
        or source_real.shape[-2] != 4
    ):
        raise ValueError("matching-direction scan requires matching NHWD4M states")
    batch = source_real.shape[0]

    horizontal_positive = positive_product_scan(
        pole_x,
        (
            _pack_directions(source_real, 0, 2),
            _pack_directions(source_imag, 0, 2),
        ),
    )
    horizontal_negative = positive_product_scan(
        _conjugate_pole(pole_x),
        (
            _pack_directions(source_real, 1, 3).flip(2),
            _pack_directions(source_imag, 1, 3).flip(2),
        ),
    )
    positive_real = _unpack_pair(horizontal_positive[0], batch)
    positive_imag = _unpack_pair(horizontal_positive[1], batch)
    negative_real = _unpack_pair(horizontal_negative[0].flip(2), batch)
    negative_imag = _unpack_pair(horizontal_negative[1].flip(2), batch)
    horizontal_real = torch.stack(
        (positive_real[0], negative_real[0], positive_real[1], negative_real[1]),
        dim=-2,
    )
    horizontal_imag = torch.stack(
        (positive_imag[0], negative_imag[0], positive_imag[1], negative_imag[1]),
        dim=-2,
    )

    vertical_positive = positive_product_scan(
        pole_y,
        (
            _pack_directions(horizontal_real, 0, 1).transpose(1, 2),
            _pack_directions(horizontal_imag, 0, 1).transpose(1, 2),
        ),
    )
    vertical_negative = positive_product_scan(
        _conjugate_pole(pole_y),
        (
            _pack_directions(horizontal_real, 2, 3).flip(1).transpose(1, 2),
            _pack_directions(horizontal_imag, 2, 3).flip(1).transpose(1, 2),
        ),
    )
    vertical_positive = (
        vertical_positive[0].transpose(1, 2),
        vertical_positive[1].transpose(1, 2),
    )
    vertical_negative = (
        vertical_negative[0].transpose(1, 2).flip(1),
        vertical_negative[1].transpose(1, 2).flip(1),
    )
    positive_real = _unpack_pair(vertical_positive[0], batch)
    positive_imag = _unpack_pair(vertical_positive[1], batch)
    negative_real = _unpack_pair(vertical_negative[0], batch)
    negative_imag = _unpack_pair(vertical_negative[1], batch)
    return (
        torch.stack((*positive_real, *negative_real), dim=-2),
        torch.stack((*positive_imag, *negative_imag), dim=-2),
    )


def _normalized_direction_states(
    pole_x: Pole,
    pole_y: Pole,
    states: ComplexField,
    *,
    gain_normalization: ProductGainNormalization,
    epsilon: float,
) -> ComplexField:
    real, imag = states
    _, height, width, _, _ = real.shape
    variance_x, variance_y = static_variance_tables(
        *(value.detach() for value in (*pole_x, *pole_y)),
        width,
        height,
    )
    directions = ((0, 0), (1, 0), (0, 1), (1, 1))
    normalized_real = []
    normalized_imag = []
    for direction, (x_sign, y_sign) in enumerate(directions):
        variance = variance_y[y_sign, :, None, :] * variance_x[x_sign, None, :, :]
        if gain_normalization == "global":
            variance = variance.mean((0, 1), keepdim=True)
        inverse = torch.rsqrt(variance.clamp_min(epsilon)).to(real.dtype)
        normalized_real.append(real[..., direction, :] * inverse)
        normalized_imag.append(imag[..., direction, :] * inverse)
    return torch.stack(normalized_real, dim=-2), torch.stack(normalized_imag, dim=-2)


def _raw_descriptor(real: Tensor, imag: Tensor) -> Tensor:
    return torch.cat(
        [
            torch.log1p(
                real[..., direction, :]
                .float()
                .square()
                .add(imag[..., direction, :].float().square())
                .mean((1, 2))
            )
            for direction in range(4)
        ],
        dim=-1,
    )


def matching_direction_scan(
    pole_x: Pole,
    pole_y: Pole,
    source: ComplexField,
    *,
    emit_coarse: bool,
    gain_normalization: ProductGainNormalization = "pointwise",
    epsilon: float = DEFAULT_EPSILON,
) -> tuple[Tensor, Tensor, Tensor] | Tensor:
    """Return normalized matching-direction endpoints and the unchanged D4 Q."""
    normalized = _normalized_direction_states(
        pole_x,
        pole_y,
        matching_direction_states(pole_x, pole_y, source),
        gain_normalization=gain_normalization,
        epsilon=epsilon,
    )
    descriptor = _raw_descriptor(*normalized)
    if not emit_coarse:
        return descriptor
    real, imag = normalized
    if real.shape[1] % 2 or real.shape[2] % 2:
        raise ValueError("matching-direction coarsening requires even spatial dimensions")
    directions = ((1, 1), (0, 1), (1, 0), (0, 0))
    coarse_real = torch.stack(
        [real[:, y::2, x::2, direction, :] for direction, (x, y) in enumerate(directions)],
        dim=-2,
    )
    coarse_imag = torch.stack(
        [imag[:, y::2, x::2, direction, :] for direction, (x, y) in enumerate(directions)],
        dim=-2,
    )
    return coarse_real, coarse_imag, descriptor


class PersistentPhaseGatedTransition(nn.Module):
    """Mode PG, path PG, direction-preserving S2D carry, then post PG."""

    def __init__(
        self,
        mixer: PhaseGatedModePathResidualComplexLinearCollapse,
        augmented: PhaseGatedS2DPostFusionTransition,
    ) -> None:
        super().__init__()
        self.modes = mixer.modes
        self.mode = mixer.mode
        self.path = mixer.path
        self.post = augmented.post
        self.carry_weight = augmented.carry_weight

    def _carry(self, real: Tensor, imag: Tensor) -> ComplexField:
        if real.shape != imag.shape or real.shape[-1] != self.modes:
            raise ValueError("persistent S2D carry requires matching mode states")
        if real.ndim == 4:
            batch, height, width, modes = real.shape
            shape = (batch, height // 2, 2, width // 2, 2, modes)
            cells_real = (
                real.reshape(shape)
                .permute(0, 1, 3, 2, 4, 5)
                .reshape(batch, height // 2, width // 2, 4, modes)
            )
            cells_imag = (
                imag.reshape(shape)
                .permute(0, 1, 3, 2, 4, 5)
                .reshape(batch, height // 2, width // 2, 4, modes)
            )
        elif real.ndim == 5 and real.shape[-2] == 4:
            batch, height, width, paths, modes = real.shape
            shape = (batch, height // 2, 2, width // 2, 2, paths, modes)
            cells_real = (
                real.reshape(shape)
                .permute(0, 1, 3, 5, 2, 4, 6)
                .reshape(batch, height // 2, width // 2, paths, 4, modes)
            )
            cells_imag = (
                imag.reshape(shape)
                .permute(0, 1, 3, 5, 2, 4, 6)
                .reshape(batch, height // 2, width // 2, paths, 4, modes)
            )
        else:
            raise ValueError("persistent S2D carry requires NHWM or NHW4M inputs")
        weight = self.carry_weight.transpose(0, 1).to(real.dtype)
        return (
            (cells_real * weight).sum(dim=-2),
            (cells_imag * weight).sum(dim=-2),
        )

    def forward(
        self,
        state_real: Tensor,
        state_imag: Tensor,
        excitation_real: Tensor,
        excitation_imag: Tensor,
    ) -> ComplexField:
        mode_real, mode_imag = self.mode(state_real, state_imag)
        path_real, path_imag = self.path(
            mode_real.transpose(-2, -1),
            mode_imag.transpose(-2, -1),
        )
        mixed_real = path_real.transpose(-2, -1)
        mixed_imag = path_imag.transpose(-2, -1)
        carry_real, carry_imag = self._carry(excitation_real, excitation_imag)
        if carry_real.ndim == 4:
            carry_real = carry_real.unsqueeze(-2)
            carry_imag = carry_imag.unsqueeze(-2)
        return self.post(mixed_real + carry_real, mixed_imag + carry_imag)


def _take_transition(stage: ComplexScanStage) -> PersistentPhaseGatedTransition:
    from .pac_phase_gated_transition import (  # noqa: PLC0415
        PhaseGatedModePathResidualComplexLinearCollapse,
        PhaseGatedS2DPostFusionTransition,
    )

    mixer = stage.quadrant_path_mode_combiner
    augmented = stage.augmented
    if not isinstance(mixer, PhaseGatedModePathResidualComplexLinearCollapse) or not isinstance(
        augmented, PhaseGatedS2DPostFusionTransition
    ):
        raise TypeError(
            "persistent D4 stages require Phase-Gated path mixing and post-fusion"
        )
    stage.quadrant_path_mode_combiner = None
    stage.augmented = None
    return PersistentPhaseGatedTransition(mixer, augmented)


class InitialPersistentD4Stage(nn.Module):
    """Create D4 once from the scalar stem excitation and preserve it."""

    def __init__(self, pole_stage: ComplexScanStage) -> None:
        super().__init__()
        self.pole_stage = pole_stage
        self.transition = _take_transition(pole_stage)
        self.modes = pole_stage.modes
        self.output_modes = pole_stage.output_modes

    def forward(self, real: Tensor, imag: Tensor) -> tuple[ComplexField, Tensor]:
        if real.shape != imag.shape or real.ndim != 4 or real.shape[-1] != self.modes:
            raise ValueError("initial persistent D4 stage requires matching NHWM inputs")
        shape = cast("tuple[int, int, int, int]", tuple(real.shape))
        pole_x, pole_y = self.pole_stage._pole_coefficients(shape)
        coarse_real, coarse_imag, descriptor = cast(
            "tuple[Tensor, Tensor, Tensor]",
            run_product_scan_pipeline(
                pole_x,
                pole_y,
                (real, imag),
                epilogue="coarse",
                gain_normalization=self.pole_stage.product_gain_normalization,
                memory_policy=self.pole_stage.scan_memory_policy,
            ),
        )
        return self.transition(coarse_real, coarse_imag, real, imag), descriptor


class MatchingPersistentD4Stage(nn.Module):
    """Coarsen four persistent memories using only their matching directions."""

    def __init__(self, pole_stage: ComplexScanStage) -> None:
        super().__init__()
        self.pole_stage = pole_stage
        self.transition = _take_transition(pole_stage)
        self.modes = pole_stage.modes
        self.output_modes = pole_stage.output_modes

    def forward(self, real: Tensor, imag: Tensor) -> tuple[ComplexField, Tensor]:
        if real.shape != imag.shape or real.ndim != 5 or tuple(real.shape[-2:]) != (4, self.modes):
            raise ValueError("persistent D4 stage requires matching NHW4M inputs")
        shape = (real.shape[0], real.shape[1], real.shape[2], self.modes)
        pole_x, pole_y = self.pole_stage._pole_coefficients(shape)
        coarse_real, coarse_imag, descriptor = cast(
            "tuple[Tensor, Tensor, Tensor]",
            matching_direction_scan(
                pole_x,
                pole_y,
                (real, imag),
                emit_coarse=True,
                gain_normalization=self.pole_stage.product_gain_normalization,
            ),
        )
        return self.transition(coarse_real, coarse_imag, real, imag), descriptor


class TerminalMatchingPersistentD4Stage(nn.Module):
    """Read Q from four terminal memories without creating new directions."""

    def __init__(self, pole_stage: ComplexScanStage) -> None:
        super().__init__()
        self.pole_stage = pole_stage
        self.modes = pole_stage.modes
        self.output_modes = None

    def forward(self, real: Tensor, imag: Tensor) -> tuple[None, Tensor]:
        if real.shape != imag.shape or real.ndim != 5 or tuple(real.shape[-2:]) != (4, self.modes):
            raise ValueError("terminal persistent D4 stage requires matching NHW4M inputs")
        shape = (real.shape[0], real.shape[1], real.shape[2], self.modes)
        pole_x, pole_y = self.pole_stage._pole_coefficients(shape)
        descriptor = cast(
            "Tensor",
            matching_direction_scan(
                pole_x,
                pole_y,
                (real, imag),
                emit_coarse=False,
                gain_normalization=self.pole_stage.product_gain_normalization,
            ),
        )
        return None, descriptor


__all__ = [
    "POSITIVE_FORWARD_LAUNCH_NAME",
    "InitialPersistentD4Stage",
    "MatchingPersistentD4Stage",
    "PersistentPhaseGatedTransition",
    "TerminalMatchingPersistentD4Stage",
    "matching_direction_scan",
    "matching_direction_states",
    "positive_product_scan",
]
