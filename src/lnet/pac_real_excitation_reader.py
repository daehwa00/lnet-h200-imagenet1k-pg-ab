"""RMS-matched readers from a real excitation carrier to a complex pole drive."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional

from .pac_complex_scan_reader import PackedComplexConv2dReader

type ComplexField = tuple[Tensor, Tensor]
type ReaderVariant = Literal[
    "R0_JIT_COMPLEX_K3",
    "R1_REAL_U",
    "R2_DUAL_FULL_K3",
    "R3_CONTENT_DWQ",
    "R4_FIXED_CONTRAST_Q",
    "R5_CONTENT_PWQ",
]

READER_VARIANTS: tuple[ReaderVariant, ...] = (
    "R0_JIT_COMPLEX_K3",
    "R1_REAL_U",
    "R2_DUAL_FULL_K3",
    "R3_CONTENT_DWQ",
    "R4_FIXED_CONTRAST_Q",
    "R5_CONTENT_PWQ",
)


def _validate_dimensions(modes: int, kernel_size: int, variance_epsilon: float) -> None:
    if modes <= 0:
        message = "real-excitation reader modes must be positive"
        raise ValueError(message)
    if kernel_size <= 0 or kernel_size % 2 == 0:
        message = "real-excitation reader kernel size must be a positive odd integer"
        raise ValueError(message)
    if variance_epsilon <= 0.0:
        message = "real-excitation reader variance epsilon must be positive"
        raise ValueError(message)


def _initialize_content_kernel_(convolution: nn.Conv2d, *, center: int) -> None:
    """Initialize the center slice as an orthonormal-row projection."""
    with torch.no_grad():
        convolution.weight.zero_()
        nn.init.orthogonal_(convolution.weight[:, :, center, center])


def _initialize_identity_kernel_(convolution: nn.Conv2d, *, center: int) -> None:
    """Initialize the center slice as identity and every neighbor as zero."""
    output_channels, input_channels = convolution.weight.shape[:2]
    if output_channels != input_channels:
        message = "identity kernel initialization requires equal input and output widths"
        raise ValueError(message)
    with torch.no_grad():
        convolution.weight.zero_()
        convolution.weight[:, :, center, center].copy_(
            torch.eye(
                output_channels,
                dtype=convolution.weight.dtype,
                device=convolution.weight.device,
            )
        )


class RMSMatchedRealExcitationReader(nn.Module, ABC):
    """Base class for tokenwise real-carrier-to-complex readers.

    The two inputs are storage halves of one real vector, not persistent real and
    imaginary coordinates.  Their concatenation therefore has width ``2 * modes``.
    Every subclass only defines the raw complex measurement; this base applies the
    shared tokenwise RMS contract against the full real carrier.
    """

    real_rms: Tensor
    imag_rms: Tensor
    imag_real_rms_ratio: Tensor
    phase_circular_variance: Tensor
    diagnostic_updates: Tensor

    def __init__(self, modes: int, *, variance_epsilon: float = 1.0e-12) -> None:
        super().__init__()
        _validate_dimensions(modes, 1, variance_epsilon)
        self.modes = modes
        self.input_modes = 2 * modes
        self.output_modes = modes
        self.variance_epsilon = float(variance_epsilon)
        self.diagnostics_enabled = False
        for name in (
            "real_rms",
            "imag_rms",
            "imag_real_rms_ratio",
            "phase_circular_variance",
            "diagnostic_updates",
        ):
            self.register_buffer(name, torch.zeros(()), persistent=False)
            setattr(self, name, self.get_buffer(name))

    def _validate_input(self, first_half: Tensor, second_half: Tensor) -> None:
        if first_half.shape != second_half.shape or first_half.ndim != 4:
            message = "real-excitation reader requires matching NHWM storage halves"
            raise ValueError(message)
        if first_half.shape[-1] != self.modes:
            message = "real-excitation reader input has an incompatible half width"
            raise ValueError(message)

    @abstractmethod
    def _raw_forward(self, first_half: Tensor, second_half: Tensor) -> ComplexField:
        """Return an unmatched complex pole drive in NHWM layout."""

    def _match_rms(
        self,
        first_half: Tensor,
        second_half: Tensor,
        output_real: Tensor,
        output_imag: Tensor,
    ) -> ComplexField:
        input_energy = (
            first_half.float().square().sum(dim=-1, keepdim=True)
            + second_half.float().square().sum(dim=-1, keepdim=True)
        ) / self.input_modes
        output_energy = output_real.float().square().add(output_imag.float().square()).mean(
            dim=-1,
            keepdim=True,
        )
        input_rms = torch.sqrt(input_energy)
        output_rms = torch.sqrt(output_energy)
        scale = (input_rms / (output_rms + self.variance_epsilon)).to(
            dtype=output_real.dtype
        )
        return output_real * scale, output_imag * scale

    @torch.no_grad()
    def _update_diagnostics(self, real: Tensor, imag: Tensor) -> None:
        sampled_real = real.detach().reshape(-1, self.modes)[:1024].float()
        sampled_imag = imag.detach().reshape(-1, self.modes)[:1024].float()
        real_rms = sampled_real.square().mean().sqrt()
        imag_rms = sampled_imag.square().mean().sqrt()
        magnitude = sampled_real.square().add(sampled_imag.square()).add(
            self.variance_epsilon
        ).sqrt()
        phase_mean_real = (sampled_real / magnitude).mean(dim=0)
        phase_mean_imag = (sampled_imag / magnitude).mean(dim=0)
        circular_variance = (
            1.0
            - torch.sqrt(
                phase_mean_real.square() + phase_mean_imag.square()
            ).clamp(max=1.0)
        ).mean()
        values = (
            real_rms,
            imag_rms,
            imag_rms / real_rms.clamp_min(self.variance_epsilon),
            circular_variance,
        )
        count = self.diagnostic_updates
        decay = torch.where(count > 0, count.new_tensor(0.95), count.new_zeros(()))
        for target, value in zip(
            (
                self.real_rms,
                self.imag_rms,
                self.imag_real_rms_ratio,
                self.phase_circular_variance,
            ),
            values,
            strict=True,
        ):
            target.mul_(decay).add_(value * (1.0 - decay))
        self.diagnostic_updates.add_(1)

    def forward(self, first_half: Tensor, second_half: Tensor) -> ComplexField:
        self._validate_input(first_half, second_half)
        output_real, output_imag = self._raw_forward(first_half, second_half)
        expected_shape = (*first_half.shape[:-1], self.modes)
        if output_real.shape != expected_shape or output_imag.shape != expected_shape:
            message = "real-excitation reader produced an incompatible complex field"
            raise RuntimeError(message)
        matched = self._match_rms(
            first_half,
            second_half,
            output_real,
            output_imag,
        )
        if self.diagnostics_enabled:
            self._update_diagnostics(*matched)
        return matched

    def set_diagnostics_enabled(self, *, enabled: bool) -> None:
        """Collect bounded response statistics only during explicit probes."""
        self.diagnostics_enabled = bool(enabled)

    @torch.no_grad()
    def diagnostic_metrics(self) -> dict[str, float]:
        """Return response diagnostics without changing reader state."""
        return {
            "real_rms": float(self.real_rms),
            "imag_rms": float(self.imag_rms),
            "imag_real_rms_ratio": float(self.imag_real_rms_ratio),
            "phase_circular_variance": float(self.phase_circular_variance),
        }


class JustInTimeComplexK3Reader(RMSMatchedRealExcitationReader):
    """Interpret the real carrier halves as complex only at the strict K3 reader."""

    def __init__(
        self,
        modes: int,
        *,
        kernel_size: int = 3,
        variance_epsilon: float = 1.0e-12,
    ) -> None:
        _validate_dimensions(modes, kernel_size, variance_epsilon)
        super().__init__(modes, variance_epsilon=variance_epsilon)
        self.reader = PackedComplexConv2dReader(
            modes,
            modes,
            kernel_size=kernel_size,
            variance_epsilon=variance_epsilon,
            match_input_rms=False,
        )

    def _raw_forward(self, first_half: Tensor, second_half: Tensor) -> ComplexField:
        return self.reader(first_half, second_half)


class _ContentK3Reader(RMSMatchedRealExcitationReader, ABC):
    """Shared full-real K3 content projection for structured quadrature readers."""

    def __init__(
        self,
        modes: int,
        *,
        kernel_size: int = 3,
        variance_epsilon: float = 1.0e-12,
    ) -> None:
        _validate_dimensions(modes, kernel_size, variance_epsilon)
        super().__init__(modes, variance_epsilon=variance_epsilon)
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.content = nn.Conv2d(
            2 * modes,
            modes,
            kernel_size,
            padding=self.padding,
            bias=False,
        )
        _initialize_content_kernel_(self.content, center=self.padding)

    def _content(self, first_half: Tensor, second_half: Tensor) -> Tensor:
        packed = torch.cat((first_half, second_half), dim=-1).movedim(-1, 1)
        return self.content(packed)


class RealOnlyK3Reader(_ContentK3Reader):
    """Read one real content response and let pole dynamics create phase."""

    def _raw_forward(self, first_half: Tensor, second_half: Tensor) -> ComplexField:
        real = self._content(first_half, second_half).movedim(1, -1).contiguous()
        return real, torch.zeros_like(real)


class DualFullK3Reader(RMSMatchedRealExcitationReader):
    """Use an unconstrained full-real K3 projection for both complex coordinates."""

    def __init__(
        self,
        modes: int,
        *,
        kernel_size: int = 3,
        variance_epsilon: float = 1.0e-12,
    ) -> None:
        _validate_dimensions(modes, kernel_size, variance_epsilon)
        super().__init__(modes, variance_epsilon=variance_epsilon)
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.projection = nn.Conv2d(
            2 * modes,
            2 * modes,
            kernel_size,
            padding=self.padding,
            bias=False,
        )
        _initialize_identity_kernel_(self.projection, center=self.padding)

    def _raw_forward(self, first_half: Tensor, second_half: Tensor) -> ComplexField:
        packed = torch.cat((first_half, second_half), dim=-1).movedim(-1, 1)
        projected = self.projection(packed)
        real, imag = projected.split(self.modes, dim=1)
        return real.movedim(1, -1).contiguous(), imag.movedim(1, -1).contiguous()


class ContentDepthwiseQuadratureReader(_ContentK3Reader):
    """Pair a full K3 content response with a learned depthwise spatial companion."""

    def __init__(
        self,
        modes: int,
        *,
        kernel_size: int = 3,
        variance_epsilon: float = 1.0e-12,
    ) -> None:
        super().__init__(
            modes,
            kernel_size=kernel_size,
            variance_epsilon=variance_epsilon,
        )
        self.quadrature = nn.Conv2d(
            modes,
            modes,
            kernel_size,
            padding=self.padding,
            groups=modes,
            bias=False,
        )
        nn.init.zeros_(self.quadrature.weight)

    def _raw_forward(self, first_half: Tensor, second_half: Tensor) -> ComplexField:
        real_nchw = self._content(first_half, second_half)
        imag_nchw = self.quadrature(real_nchw)
        return (
            real_nchw.movedim(1, -1).contiguous(),
            imag_nchw.movedim(1, -1).contiguous(),
        )


class FixedContrastQuadratureReader(_ContentK3Reader):
    """Use local content minus its fixed neighborhood mean as the imaginary part."""

    def _raw_forward(self, first_half: Tensor, second_half: Tensor) -> ComplexField:
        real_nchw = self._content(first_half, second_half)
        local_mean = functional.avg_pool2d(
            real_nchw,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.padding,
            count_include_pad=True,
        )
        imag_nchw = real_nchw - local_mean
        return (
            real_nchw.movedim(1, -1).contiguous(),
            imag_nchw.movedim(1, -1).contiguous(),
        )


class ContentPointwiseQuadratureReader(_ContentK3Reader):
    """Pair K3 content with a learned pointwise channel companion."""

    def __init__(
        self,
        modes: int,
        *,
        kernel_size: int = 3,
        variance_epsilon: float = 1.0e-12,
    ) -> None:
        super().__init__(
            modes,
            kernel_size=kernel_size,
            variance_epsilon=variance_epsilon,
        )
        self.quadrature = nn.Linear(modes, modes, bias=False)
        nn.init.zeros_(self.quadrature.weight)

    def _raw_forward(self, first_half: Tensor, second_half: Tensor) -> ComplexField:
        real = self._content(first_half, second_half).movedim(1, -1).contiguous()
        return real, self.quadrature(real)


def build_real_excitation_reader(
    variant: ReaderVariant,
    modes: int,
    *,
    kernel_size: int = 3,
    variance_epsilon: float = 1.0e-12,
) -> RMSMatchedRealExcitationReader:
    """Build one of the six controlled real-excitation reader variants."""
    if variant == "R0_JIT_COMPLEX_K3":
        return JustInTimeComplexK3Reader(
            modes,
            kernel_size=kernel_size,
            variance_epsilon=variance_epsilon,
        )
    if variant == "R1_REAL_U":
        return RealOnlyK3Reader(
            modes,
            kernel_size=kernel_size,
            variance_epsilon=variance_epsilon,
        )
    if variant == "R2_DUAL_FULL_K3":
        return DualFullK3Reader(
            modes,
            kernel_size=kernel_size,
            variance_epsilon=variance_epsilon,
        )
    if variant == "R3_CONTENT_DWQ":
        return ContentDepthwiseQuadratureReader(
            modes,
            kernel_size=kernel_size,
            variance_epsilon=variance_epsilon,
        )
    if variant == "R4_FIXED_CONTRAST_Q":
        return FixedContrastQuadratureReader(
            modes,
            kernel_size=kernel_size,
            variance_epsilon=variance_epsilon,
        )
    if variant == "R5_CONTENT_PWQ":
        return ContentPointwiseQuadratureReader(
            modes,
            kernel_size=kernel_size,
            variance_epsilon=variance_epsilon,
        )
    message = f"unknown real-excitation reader variant: {variant}"
    raise ValueError(message)


__all__ = [
    "READER_VARIANTS",
    "ContentDepthwiseQuadratureReader",
    "ContentPointwiseQuadratureReader",
    "DualFullK3Reader",
    "FixedContrastQuadratureReader",
    "JustInTimeComplexK3Reader",
    "RMSMatchedRealExcitationReader",
    "ReaderVariant",
    "RealOnlyK3Reader",
    "build_real_excitation_reader",
]
