"""Static variance and global-gain normalization for D4 product scans."""

from __future__ import annotations

# pyright: reportArgumentType=false, reportCallIssue=false, reportMissingParameterType=false
# pyright: reportOptionalSubscript=false
# pyright: reportUnknownLambdaType=false
import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton
from triton.language.extra import libdevice

from .pac_kernel_launch_config import LaunchGeometry, autotuned, make_launch_scope, register_default
from .pac_product_scan_contracts import GLOBAL_GAIN

VARIANCE_LAUNCH_NAME = "product_scan_coarse4_static_variance"
GLOBAL_GAIN_LAUNCH_NAME = "product_scan_coarse4_global_gain"
_AUXILIARY_LAUNCH_CANDIDATES = tuple(
    LaunchGeometry.build(num_warps=warps, blocks={"BLOCK_MODES": modes})
    for warps in (2, 4)
    for modes in (16, 32, 64)
)
_AUXILIARY_DEFAULT = LaunchGeometry.build(num_warps=4, blocks={"BLOCK_MODES": 32})

register_default(
    VARIANCE_LAUNCH_NAME,
    _AUXILIARY_DEFAULT,
    candidates=_AUXILIARY_LAUNCH_CANDIDATES,
)
register_default(
    GLOBAL_GAIN_LAUNCH_NAME,
    _AUXILIARY_DEFAULT,
    candidates=_AUXILIARY_LAUNCH_CANDIDATES,
)


@triton.jit
def _geometric_sum(decay, count):
    delta = 1.0 - decay
    near_one = tl.abs(delta) < 1.0e-4
    denominator = tl.where(near_one, 1.0, delta)
    power = tl.exp2(tl.log2(tl.maximum(decay, 1.0e-30)) * count)
    regular = (1.0 - power) / denominator
    expansion = (
        count
        - delta * count * (count - 1.0) * 0.5
        + delta * delta * count * (count - 1.0) * (count - 2.0) * (1.0 / 6.0)
    )
    return tl.where(near_one, expansion, regular)


@triton.jit
def _geometric_state_total(decay, gamma, length: tl.constexpr):
    delta = 1.0 - decay
    near_one = tl.abs(delta) < 1.0e-4
    denominator = tl.where(near_one, 1.0, delta)
    count = length * 1.0
    regular = gamma * (count - decay * _geometric_sum(decay, count)) / denominator
    expansion = gamma * (
        count * (count + 1.0) * 0.5
        - delta * count * (count + 1.0) * (count - 1.0) * (1.0 / 6.0)
        + delta * delta * count * (count + 1.0) * (count - 1.0) * (count - 2.0) * (1.0 / 24.0)
    )
    return tl.where(near_one, expansion, regular)


@triton.jit
def _static_separable_variance_kernel(
    decay_x_real,
    decay_x_imag,
    gamma_x_real,
    gamma_x_imag,
    decay_y_real,
    decay_y_imag,
    gamma_y_real,
    gamma_y_imag,
    variance_x,
    variance_y,
    length_x: tl.constexpr,
    length_y: tl.constexpr,
    modes: int,
    BLOCK_MODES: tl.constexpr,
    BLOCK_STEPS: tl.constexpr,
) -> None:
    step = tl.program_id(0) * BLOCK_STEPS + tl.arange(0, BLOCK_STEPS)
    mode = tl.program_id(1) * BLOCK_MODES + tl.arange(0, BLOCK_MODES)
    valid_mode = mode < modes
    axr = tl.load(decay_x_real + mode, mask=valid_mode, other=0.0).to(tl.float32)
    axi = tl.load(decay_x_imag + mode, mask=valid_mode, other=0.0).to(tl.float32)
    gxr = tl.load(gamma_x_real + mode, mask=valid_mode, other=0.0).to(tl.float32)
    gxi = tl.load(gamma_x_imag + mode, mask=valid_mode, other=0.0).to(tl.float32)
    ayr = tl.load(decay_y_real + mode, mask=valid_mode, other=0.0).to(tl.float32)
    ayi = tl.load(decay_y_imag + mode, mask=valid_mode, other=0.0).to(tl.float32)
    gyr = tl.load(gamma_y_real + mode, mask=valid_mode, other=0.0).to(tl.float32)
    gyi = tl.load(gamma_y_imag + mode, mask=valid_mode, other=0.0).to(tl.float32)
    decay_x = libdevice.add_rn(libdevice.mul_rn(axr, axr), libdevice.mul_rn(axi, axi))
    gamma_x = libdevice.add_rn(libdevice.mul_rn(gxr, gxr), libdevice.mul_rn(gxi, gxi))
    decay_y = libdevice.add_rn(libdevice.mul_rn(ayr, ayr), libdevice.mul_rn(ayi, ayi))
    gamma_y = libdevice.add_rn(libdevice.mul_rn(gyr, gyr), libdevice.mul_rn(gyi, gyi))
    count = step[:, None].to(tl.float32) + 1.0
    state_x = gamma_x[None, :] * _geometric_sum(decay_x[None, :], count)
    state_y = gamma_y[None, :] * _geometric_sum(decay_y[None, :], count)
    positive = step[:, None] * modes + mode[None, :]
    negative_x = (2 * length_x - 1 - step[:, None]) * modes + mode[None, :]
    negative_y = (2 * length_y - 1 - step[:, None]) * modes + mode[None, :]
    valid_x = (step[:, None] < length_x) & valid_mode[None, :]
    valid_y = (step[:, None] < length_y) & valid_mode[None, :]
    tl.store(variance_x + positive, state_x, mask=valid_x)
    tl.store(variance_x + negative_x, state_x, mask=valid_x)
    tl.store(variance_y + positive, state_y, mask=valid_y)
    tl.store(variance_y + negative_y, state_y, mask=valid_y)


@triton.jit
def _static_global_inverse_gain_kernel(
    decay_x_real,
    decay_x_imag,
    gamma_x_real,
    gamma_x_imag,
    decay_y_real,
    decay_y_imag,
    gamma_y_real,
    gamma_y_imag,
    inverse_gain,
    length_x: tl.constexpr,
    length_y: tl.constexpr,
    modes: int,
    epsilon: tl.constexpr,
    BLOCK_MODES: tl.constexpr,
) -> None:
    mode = tl.program_id(0) * BLOCK_MODES + tl.arange(0, BLOCK_MODES)
    valid = mode < modes
    axr = tl.load(decay_x_real + mode, mask=valid, other=0.0).to(tl.float32)
    axi = tl.load(decay_x_imag + mode, mask=valid, other=0.0).to(tl.float32)
    gxr = tl.load(gamma_x_real + mode, mask=valid, other=0.0).to(tl.float32)
    gxi = tl.load(gamma_x_imag + mode, mask=valid, other=0.0).to(tl.float32)
    ayr = tl.load(decay_y_real + mode, mask=valid, other=0.0).to(tl.float32)
    ayi = tl.load(decay_y_imag + mode, mask=valid, other=0.0).to(tl.float32)
    gyr = tl.load(gamma_y_real + mode, mask=valid, other=0.0).to(tl.float32)
    gyi = tl.load(gamma_y_imag + mode, mask=valid, other=0.0).to(tl.float32)
    decay_x = libdevice.add_rn(libdevice.mul_rn(axr, axr), libdevice.mul_rn(axi, axi))
    gamma_x = libdevice.add_rn(libdevice.mul_rn(gxr, gxr), libdevice.mul_rn(gxi, gxi))
    decay_y = libdevice.add_rn(libdevice.mul_rn(ayr, ayr), libdevice.mul_rn(ayi, ayi))
    gamma_y = libdevice.add_rn(libdevice.mul_rn(gyr, gyr), libdevice.mul_rn(gyi, gyi))
    sum_x = _geometric_state_total(decay_x, gamma_x, length_x)
    sum_y = _geometric_state_total(decay_y, gamma_y, length_y)
    mean_product = (sum_x * (1.0 / length_x)) * (sum_y * (1.0 / length_y))
    tl.store(inverse_gain + mode, tl.rsqrt(tl.maximum(mean_product, epsilon)), mask=valid)


def _positive_variance_table(
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
    length: int,
) -> Tensor:
    modes = decay_real.shape[-1]
    decay = decay_real.float().square().add(decay_imag.float().square()).reshape(modes)
    gamma = gamma_real.float().square().add(gamma_imag.float().square()).reshape(modes)
    counts = torch.arange(1, length + 1, device=decay.device, dtype=decay.dtype).unsqueeze(1)
    decay = decay.unsqueeze(0)
    denominator = 1.0 - decay
    near_one = denominator.abs() < 1.0e-4
    safe_denominator = torch.where(near_one, torch.ones_like(denominator), denominator)
    series = (1.0 - decay.clamp_min(1.0e-30).pow(counts)) / safe_denominator
    expansion = (
        counts
        - denominator * counts * (counts - 1.0) * 0.5
        + denominator.square() * counts * (counts - 1.0) * (counts - 2.0) / 6.0
    )
    series = torch.where(near_one, expansion, series)
    return gamma.unsqueeze(0) * series


@triton_op("lnet::pac_product_static_global_inverse_gain", mutates_args={})
def static_global_inverse_gain(
    decay_x_real: Tensor,
    decay_x_imag: Tensor,
    gamma_x_real: Tensor,
    gamma_x_imag: Tensor,
    decay_y_real: Tensor,
    decay_y_imag: Tensor,
    gamma_y_real: Tensor,
    gamma_y_imag: Tensor,
    length_x: int,
    length_y: int,
    epsilon: float,
) -> Tensor:
    coefficients = (
        decay_x_real,
        decay_x_imag,
        gamma_x_real,
        gamma_x_imag,
        decay_y_real,
        decay_y_imag,
        gamma_y_real,
        gamma_y_imag,
    )
    if length_x <= 0 or length_y <= 0:
        raise ValueError("static global gain lengths must be positive")
    if epsilon <= 0.0:
        raise ValueError("static global gain epsilon must be positive")
    if decay_x_real.ndim != 4 or tuple(decay_x_real.shape[:3]) != (1, 1, 1):
        raise ValueError("static global gain requires compact 111M poles")
    if any(value.shape != decay_x_real.shape for value in coefficients[1:]):
        raise ValueError("static global gain poles must have matching shapes")
    if any(value.device != decay_x_real.device for value in coefficients[1:]):
        raise ValueError("static global gain poles must share one device")
    modes = decay_x_real.shape[-1]
    if not decay_x_real.is_cuda:
        variance_x = _positive_variance_table(*coefficients[:4], length_x)
        variance_y = _positive_variance_table(*coefficients[4:], length_y)
        return torch.rsqrt(
            (variance_x.mean(dim=0) * variance_y.mean(dim=0)).clamp_min(epsilon)
        ).contiguous()
    inverse_gain = torch.empty((modes,), dtype=torch.float32, device=decay_x_real.device)
    gain_kernel = autotuned(
        _static_global_inverse_gain_kernel,
        GLOBAL_GAIN_LAUNCH_NAME,
        key=("length_x", "length_y", "modes"),
        scope=make_launch_scope(
            _static_global_inverse_gain_kernel,
            decay_x_real,
            shape={"length_x": length_x, "length_y": length_y, "modes": modes},
        ),
    )
    wrap_triton(gain_kernel)[lambda metadata: (triton.cdiv(modes, metadata["BLOCK_MODES"]),)](
        *(value.contiguous() for value in coefficients),
        inverse_gain,
        length_x=length_x,
        length_y=length_y,
        modes=modes,
        epsilon=epsilon,
    )
    return inverse_gain


@triton_op("lnet::pac_product_static_variance_tables", mutates_args={})
def static_variance_tables(
    decay_x_real: Tensor,
    decay_x_imag: Tensor,
    gamma_x_real: Tensor,
    gamma_x_imag: Tensor,
    decay_y_real: Tensor,
    decay_y_imag: Tensor,
    gamma_y_real: Tensor,
    gamma_y_imag: Tensor,
    length_x: int,
    length_y: int,
) -> tuple[Tensor, Tensor]:
    coefficients = (
        decay_x_real,
        decay_x_imag,
        gamma_x_real,
        gamma_x_imag,
        decay_y_real,
        decay_y_imag,
        gamma_y_real,
        gamma_y_imag,
    )
    if length_x <= 0 or length_y <= 0:
        raise ValueError("static variance table lengths must be positive")
    if decay_x_real.ndim != 4 or tuple(decay_x_real.shape[:3]) != (1, 1, 1):
        raise ValueError("static variance tables require compact 111M poles")
    if any(value.shape != decay_x_real.shape for value in coefficients[1:]):
        raise ValueError("static variance table poles must have matching shapes")
    if any(value.device != decay_x_real.device for value in coefficients[1:]):
        raise ValueError("static variance table poles must share one device")
    modes = decay_x_real.shape[-1]
    if not decay_x_real.is_cuda:
        positive_x = _positive_variance_table(*coefficients[:4], length_x)
        positive_y = _positive_variance_table(*coefficients[4:], length_y)
        return (
            torch.stack((positive_x, positive_x.flip(0))),
            torch.stack((positive_y, positive_y.flip(0))),
        )
    variance_x = torch.empty((2, length_x, modes), dtype=torch.float32, device=decay_x_real.device)
    variance_y = torch.empty((2, length_y, modes), dtype=torch.float32, device=decay_x_real.device)
    max_length = max(length_x, length_y)
    scope = make_launch_scope(
        _static_separable_variance_kernel,
        decay_x_real,
        shape={"length_x": length_x, "length_y": length_y, "modes": modes},
    )
    variance_kernel = autotuned(
        _static_separable_variance_kernel,
        VARIANCE_LAUNCH_NAME,
        key=("length_x", "length_y", "modes"),
        scope=scope,
    )
    wrap_triton(variance_kernel)[
        lambda metadata: (
            triton.cdiv(max_length, 32),
            triton.cdiv(modes, metadata["BLOCK_MODES"]),
        )
    ](
        *(value.contiguous() for value in coefficients),
        variance_x,
        variance_y,
        length_x=length_x,
        length_y=length_y,
        modes=modes,
        BLOCK_STEPS=32,
    )
    return variance_x, variance_y


def static_product_scan_auxiliary(
    pole_x: tuple[Tensor, Tensor, Tensor, Tensor],
    pole_y: tuple[Tensor, Tensor, Tensor, Tensor],
    source: Tensor,
    *,
    epsilon: float,
    gain_kind: int,
) -> tuple[Tensor, Tensor, Tensor]:
    _, height, width, _ = source.shape
    detached_poles = tuple(value.detach() for value in (*pole_x, *pole_y))
    if gain_kind == GLOBAL_GAIN and source.is_cuda:
        global_inverse_gain = static_global_inverse_gain(
            *detached_poles,
            width,
            height,
            epsilon,
        )
        empty = source.new_empty((0,), dtype=torch.float32)
        return empty, empty, global_inverse_gain
    variance_x, variance_y = static_variance_tables(*detached_poles, width, height)
    global_inverse_gain = variance_x.new_empty((0,))
    return variance_x, variance_y, global_inverse_gain


__all__ = [
    "GLOBAL_GAIN_LAUNCH_NAME",
    "VARIANCE_LAUNCH_NAME",
    "static_global_inverse_gain",
    "static_product_scan_auxiliary",
    "static_variance_tables",
]
