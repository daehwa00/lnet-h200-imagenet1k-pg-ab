from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

PoleGammaFn = Callable[
    [Tensor, Tensor, Tensor, float, float],
    tuple[Tensor, Tensor, Tensor, Tensor],
]
_compiled_pole_gamma: PoleGammaFn | None = None
DiscretePoleFn = Callable[
    [Tensor, Tensor, float | Tensor, float],
    tuple[Tensor, Tensor, Tensor, Tensor],
]
_compiled_discrete_pole: DiscretePoleFn | None = None


def pole_transition_real2d(
    damping: Tensor,
    frequency: Tensor,
    dt: float | Tensor,
) -> tuple[Tensor, Tensor]:
    """Return the real 2D representation of exp((-damping + i frequency) dt)."""
    step = dt.to(dtype=damping.dtype, device=damping.device) if isinstance(dt, Tensor) else dt
    scaled_decay = torch.exp(-damping * step)
    phase = frequency * step
    return scaled_decay * torch.cos(phase), scaled_decay * torch.sin(phase)


def discrete_pole_real2d(
    damping: Tensor,
    frequency: Tensor,
    dt: float | Tensor,
    threshold: float = 1.0e-6,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    step = dt.to(dtype=damping.dtype, device=damping.device) if isinstance(dt, Tensor) else dt
    decay_real, decay_imag = pole_transition_real2d(damping, frequency, step)
    shifted_real = decay_real - 1.0
    shifted_imag = decay_imag
    pole_real = -damping
    pole_imag = frequency
    denominator = pole_real.square() + pole_imag.square()
    gamma_real = (shifted_real * pole_real + shifted_imag * pole_imag) / denominator
    gamma_imag = (shifted_imag * pole_real - shifted_real * pole_imag) / denominator
    small = torch.sqrt((pole_real * step).square() + (pole_imag * step).square()) < threshold
    return (
        decay_real,
        decay_imag,
        torch.where(small, torch.ones_like(gamma_real) * step, gamma_real),
        torch.where(small, torch.zeros_like(gamma_imag), gamma_imag),
    )


def compiled_discrete_pole_real2d(
    damping: Tensor,
    frequency: Tensor,
    dt: float | Tensor,
    threshold: float = 1.0e-6,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Fuse the fixed-shape training pole/gamma pointwise island.

    CUDA Graph callers warm this callable before capture.  Disabling Inductor's
    inner cudagraphs keeps ownership with the outer full-step graph runtime.
    """
    if not damping.is_cuda:
        return discrete_pole_real2d(damping, frequency, dt, threshold)
    global _compiled_discrete_pole  # noqa: PLW0603
    if _compiled_discrete_pole is None:
        _compiled_discrete_pole = torch.compile(
            discrete_pole_real2d,
            fullgraph=True,
            mode="max-autotune-no-cudagraphs",
        )
    return _compiled_discrete_pole(damping, frequency, dt, threshold)


def pole_gamma_from_control_real2d(
    raw_decay: Tensor,
    frequency: Tensor,
    control: Tensor,
    min_decay: float,
    dt: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    damping = min_decay + torch.nn.functional.softplus(raw_decay.view(1, 1, -1) + control)
    expanded_frequency = frequency.view(1, 1, -1).expand_as(damping)
    return discrete_pole_real2d(damping, expanded_frequency, dt)


def compiled_pole_gamma_from_control_real2d(
    raw_decay: Tensor,
    frequency: Tensor,
    control: Tensor,
    min_decay: float,
    dt: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if not control.is_cuda:
        return pole_gamma_from_control_real2d(raw_decay, frequency, control, min_decay, dt)
    global _compiled_pole_gamma  # noqa: PLW0603
    if _compiled_pole_gamma is None:
        _compiled_pole_gamma = torch.compile(pole_gamma_from_control_real2d)
    return _compiled_pole_gamma(raw_decay, frequency, control, min_decay, dt)


def modal_output_real2d(
    states_real: Tensor,
    states_imag: Tensor,
    writer_real: Tensor,
    writer_imag: Tensor,
) -> Tensor:
    real_part = torch.einsum("bnm,md->bnd", states_real, writer_real)
    imag_part = torch.einsum("bnm,md->bnd", states_imag, writer_imag)
    return 2.0 * (real_part - imag_part)
