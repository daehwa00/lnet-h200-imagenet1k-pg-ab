"""Continuous complex-state scan backbone."""

from __future__ import annotations

# pyright: reportPrivateUsage=false
import math
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional
from torch.nn.utils import parametrize
from torch.nn.utils.parametrizations import orthogonal

from . import complex_scan_config as _config
from . import complex_scan_stage as _bank
from . import complex_scan_transitions as _transitions
from . import complex_scan_types as _contracts
from . import pac_complex_layers as _complex_layers
from . import pac_path_cffn as _path_cffn
from . import pac_triton_product_scan_coarse4 as _coarse4
from .complex_scan_stage import (
    ComplexScanStage as _ComplexScanStage,
)
from .complex_scan_stage import _phase_atlas
from .complex_scan_transitions import (
    ComplexRMSNorm,
    _dtype_aligned_rms_norm,
)
from .image_layers import CifarConvStem, LayerNorm2d, LowRankQuadraticModalHead

AffineQuadrantPathModeCFFNCombiner = _path_cffn.AffineQuadrantPathModeCFFNCombiner
FactorizedQuadrantPathModeCFFNCombiner = _path_cffn.FactorizedQuadrantPathModeCFFNCombiner
IdentityQuadrantPathModeCombiner = _path_cffn.IdentityQuadrantPathModeCombiner
JointPathModeCFFNCombiner = _path_cffn.JointPathModeCFFNCombiner
ComplexLinear = _complex_layers.ComplexLinear
WidelyLinear = _complex_layers.WidelyLinear
ComplexScanStage = _ComplexScanStage
ComplexCarryBasis = _contracts.ComplexCarryBasis
ComplexCarryMerge = _contracts.ComplexCarryMerge
ComplexField = _contracts.ComplexField
ComplexStem = _contracts.ComplexStem
DirectionalState = _contracts.DirectionalState
_DIRECTIONS = _contracts.DIRECTIONS
ComplexCCCNDownsample = _bank.ComplexCCCNDownsample
ProductGainNormalization = _coarse4.ProductGainNormalization
complex_carry_coordinates = _bank.complex_carry_coordinates
pole_aligned_complex_downsample = _bank.pole_aligned_complex_downsample
AugmentedComplexTransition = _transitions.AugmentedComplexTransition
ComplexInteractionTransition = _transitions.ComplexInteractionTransition
ComplexModulatedTransition = _transitions.ComplexModulatedTransition
ComplexResidualFFN = _transitions.ComplexResidualFFN
ComplexScanConfig = _config.ComplexScanConfig
FixedComplexRMSNorm = _transitions.FixedComplexRMSNorm
S2DCarryMainTransition = _transitions.S2DCarryMainTransition
S2DCleanProjectedResidualPostFusionCFFNTransition = (
    _transitions.S2DCleanProjectedResidualPostFusionCFFNTransition
)
S2DDirectPostFusionCFFNTransition = _transitions.S2DDirectPostFusionCFFNTransition
S2DJointPathResidualPostFusionCFFNTransition = (
    _transitions.S2DJointPathResidualPostFusionCFFNTransition
)
S2DPostCFFNCarryMainTransition = _transitions.S2DPostCFFNCarryMainTransition
S2DPostFusionCFFNTransition = _transitions.S2DPostFusionCFFNTransition
S2DProjectedResidualPostFusionCFFNTransition = (
    _transitions.S2DProjectedResidualPostFusionCFFNTransition
)
S2DStrictComplexPostCarryTransition = _transitions.S2DStrictComplexPostCarryTransition
S2DUnnormalizedPolePostFusionCFFNTransition = (
    _transitions.S2DUnnormalizedPolePostFusionCFFNTransition
)


class ModalFusionHead(nn.Module):
    """Fuse all stage descriptors before the final affine classifier."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or output_dim <= 0:
            message = "modal fusion head dimensions must be positive"
            raise ValueError(message)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.standardizer = nn.BatchNorm1d(input_dim, affine=False)
        self.fusion = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.GELU()
        self.norm = nn.RMSNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, descriptor: Tensor) -> Tensor:
        standardized = self.standardizer(descriptor)
        fused = self.norm(self.activation(self.fusion(standardized)))
        return self.classifier(fused)


class ParallelFusionLRQHead(nn.Module):
    """Combine nonlinear stage fusion with an initially disabled LRQ branch."""

    def __init__(
        self,
        input_dim: int,
        fusion_width: int,
        output_dim: int,
        quadratic_rank: int,
    ) -> None:
        super().__init__()
        self.fusion = ModalFusionHead(input_dim, fusion_width, output_dim)
        self.quadratic = LowRankQuadraticModalHead(
            input_dim,
            output_dim,
            quadratic_rank,
        )
        self.beta = nn.Parameter(torch.zeros(()))

    def forward(self, descriptor: Tensor) -> Tensor:
        return self.fusion(descriptor) + self.beta * self.quadratic(descriptor)


def _privatize_parametrized_class_(child: nn.Module) -> None:
    """Give one parametrized module its own copy of the injected class.

    ``parametrize`` injects a per-instance subclass and hangs the managing
    property off it, but ``copy.deepcopy`` copies the instance while sharing
    that class object.  ``remove_parametrizations`` then ``delattr``s the
    property from the shared class and silently breaks every sibling copy.
    Cloning the class first confines the removal to this module.
    """
    shared = type(child)
    namespace = {
        key: value for key, value in vars(shared).items() if key not in ("__dict__", "__weakref__")
    }
    private = type(shared.__name__, shared.__bases__, namespace)
    # ``remove_parametrizations`` restores ``__bases__[0]``, so the clone must
    # keep the same bases as the injected class it replaces.
    cast("object", child).__class__ = private


_UNTRACED_FLAG = "_lnet_untraced_parametrization"


def untraced_parametrizations_(module: nn.Module) -> int:
    """Force a compile graph break around every parametrization, in place.

    Returns the number of parametrizations marked.  The computation is
    unchanged -- this only tells Dynamo not to trace into it.

    ``matrix_exp`` selects its Pade order from a host-visible norm, and that
    synchronization aborts CUDA Graph capture.  Since
    ``torch.compile(mode="reduce-overhead")`` captures graphs, a traced
    parametrization makes the whole compiled training step fail.  Breaking the
    graph around it keeps the orthogonality constraint and its gradient path
    intact while letting everything downstream be captured.

    Unlike :func:`bake_parametrizations_` this is safe during training: the
    constraint survives, ``state_dict`` is unchanged, and it is reversible in
    the sense that it only affects compilation.
    """
    marked = 0
    for child in module.modules():
        if not parametrize.is_parametrized(child):
            continue
        for entry in cast("nn.ModuleDict", child.parametrizations).values():
            # A ParametrizationList holds its steps as ordinary child modules.
            for step in entry.children():
                if getattr(step, _UNTRACED_FLAG, False):
                    continue
                object.__setattr__(step, "forward", torch.compiler.disable(step.forward))
                object.__setattr__(step, _UNTRACED_FLAG, True)
                marked += 1
    return marked


def bake_parametrizations_(module: nn.Module) -> int:
    """Replace every reparametrized weight with its realized tensor, in place.

    Returns the number of parametrizations removed.  The realized values are
    exactly what the parametrization would have produced, so outputs do not
    change; only the host-side cost of rebuilding them disappears.  The
    conversion is one-way and drops the constraint, so it belongs to inference
    preparation rather than to any training path.
    """
    removed = 0
    for child in module.modules():
        if not parametrize.is_parametrized(child):
            continue
        _privatize_parametrized_class_(child)
        parametrized = cast("nn.ModuleDict", child.parametrizations)
        for name in list(parametrized):
            parametrize.remove_parametrizations(child, name, leave_parametrized=True)
            removed += 1
    return removed


class CifarConvOnlyStem(nn.Module):
    """Linear two-convolution CIFAR stem with no norm or activation inside."""

    def __init__(
        self,
        output_width: int = 64,
        strides: tuple[int, int] = (1, 1),
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=strides[0], padding=1, bias=False),
            nn.Conv2d(32, output_width, 3, stride=strides[1], padding=1, bias=False),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs).permute(0, 2, 3, 1)


class CifarNormalizedLinearStem(nn.Module):
    """Two convolutions with normalization but no activation function."""

    def __init__(
        self,
        output_width: int = 64,
        strides: tuple[int, int] = (1, 1),
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=strides[0], padding=1, bias=False),
            LayerNorm2d(32),
            nn.Conv2d(
                32,
                output_width,
                3,
                stride=strides[1],
                padding=1,
                bias=False,
            ),
            LayerNorm2d(output_width),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs).permute(0, 2, 3, 1)


class LocalComplexFourierStem(nn.Module):
    """Fuse RGB locally, then lift selected sliding Fourier modes to complex fields."""

    def __init__(
        self,
        modes: int,
        *,
        color_stride: int = 1,
        fourier_hop: int = 4,
        window_size: int = 8,
        maximum_phase: float = math.pi * 0.75,
    ) -> None:
        super().__init__()
        if modes <= 0 or modes % 4:
            message = "local Fourier stem requires a positive mode count divisible by four"
            raise ValueError(message)
        if color_stride <= 0 or fourier_hop <= 0 or window_size <= 1:
            message = "local Fourier stem strides and window must be positive"
            raise ValueError(message)
        if window_size % 2:
            message = "local Fourier stem currently requires an even window"
            raise ValueError(message)
        self.modes = modes
        self.fourier_hop = fourier_hop
        self.window_size = window_size
        self.padding = (window_size - fourier_hop) // 2
        if self.padding < 0:
            message = "local Fourier hop cannot exceed its analysis window"
            raise ValueError(message)
        self.color = nn.Sequential(
            nn.Conv2d(3, modes, 3, stride=color_stride, padding=1, bias=False),
            LayerNorm2d(modes),
            nn.GELU(),
        )
        phase_x, phase_y = _phase_atlas(modes, maximum_phase)
        self.phase_x = nn.Parameter(phase_x)
        self.phase_y = nn.Parameter(phase_y)
        self.output_norm = ComplexRMSNorm(modes)
        window = torch.hann_window(window_size, periodic=False)
        analysis_window = torch.outer(window, window)
        self.register_buffer(
            "analysis_window",
            analysis_window / analysis_window.sum(),
            persistent=True,
        )

    def _weights(self, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        coordinate = torch.arange(
            self.window_size,
            device=self.phase_x.device,
            dtype=self.phase_x.dtype,
        )
        coordinate = coordinate - (self.window_size - 1) / 2.0
        offset_y, offset_x = torch.meshgrid(coordinate, coordinate, indexing="ij")
        angle = (
            self.phase_x[:, None, None] * offset_x[None, :, :]
            + self.phase_y[:, None, None] * offset_y[None, :, :]
        )
        window = cast("Tensor", self.analysis_window).to(dtype=self.phase_x.dtype)[None, :, :]
        weight_real = window * torch.cos(angle)
        weight_imag = -window * torch.sin(angle)
        return weight_real[:, None].to(dtype=dtype), weight_imag[:, None].to(dtype=dtype)

    def forward(self, inputs: Tensor) -> ComplexField:
        features = self.color(inputs)
        weight_real, weight_imag = self._weights(features.dtype)
        real = functional.conv2d(
            features,
            weight_real,
            stride=self.fourier_hop,
            padding=self.padding,
            groups=self.modes,
        )
        imag = functional.conv2d(
            features,
            weight_imag,
            stride=self.fourier_hop,
            padding=self.padding,
            groups=self.modes,
        )
        return self.output_norm(
            real.permute(0, 2, 3, 1),
            imag.permute(0, 2, 3, 1),
        )


class ComplexDepthwiseConv2d(nn.Module):
    """Strict complex depthwise convolution evaluated as one grouped real op."""

    def __init__(self, modes: int, kernel_size: int = 3) -> None:
        super().__init__()
        if modes <= 0 or kernel_size <= 0 or kernel_size % 2 == 0:
            message = "complex depthwise convolution requires positive modes and odd kernel"
            raise ValueError(message)
        self.modes = modes
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.weight_real = nn.Parameter(torch.empty(modes, 1, kernel_size, kernel_size))
        self.weight_imag = nn.Parameter(torch.empty(modes, 1, kernel_size, kernel_size))
        nn.init.xavier_uniform_(self.weight_real)
        nn.init.xavier_uniform_(self.weight_imag)
        with torch.no_grad():
            self.weight_real.mul_(math.sqrt(0.5))
            self.weight_imag.mul_(math.sqrt(0.5))

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        if real.shape != imag.shape or real.ndim != 4 or real.shape[1] != self.modes:
            message = "complex depthwise convolution requires matching NCHW fields"
            raise ValueError(message)
        real_kernel = torch.cat((self.weight_real, -self.weight_imag), dim=1)
        imag_kernel = torch.cat((self.weight_imag, self.weight_real), dim=1)
        kernel = torch.stack((real_kernel, imag_kernel), dim=1).reshape(
            2 * self.modes,
            2,
            self.kernel_size,
            self.kernel_size,
        )
        packed = torch.stack((real, imag), dim=2).reshape(
            real.shape[0],
            2 * self.modes,
            real.shape[2],
            real.shape[3],
        )
        output = functional.conv2d(
            packed,
            kernel,
            padding=self.padding,
            groups=self.modes,
        )
        output = output.reshape(
            real.shape[0],
            self.modes,
            2,
            real.shape[2],
            real.shape[3],
        )
        return output[:, :, 0], output[:, :, 1]


class GatedWidelyLinearConv2d(nn.Module):
    """Pointwise Wz + beta V*conj(z), initialized as a strict complex map."""

    def __init__(self, input_modes: int, output_modes: int) -> None:
        super().__init__()
        if input_modes <= 0 or output_modes <= 0:
            message = "widely-linear complex convolution dimensions must be positive"
            raise ValueError(message)
        self.input_modes = input_modes
        self.output_modes = output_modes
        shape = (output_modes, input_modes, 1, 1)
        self.weight_real = nn.Parameter(torch.empty(shape))
        self.weight_imag = nn.Parameter(torch.empty(shape))
        self.conjugate_real = nn.Parameter(torch.empty(shape))
        self.conjugate_imag = nn.Parameter(torch.empty(shape))
        for weight in (
            self.weight_real,
            self.weight_imag,
            self.conjugate_real,
            self.conjugate_imag,
        ):
            nn.init.xavier_uniform_(weight)
            with torch.no_grad():
                weight.mul_(0.5)
        self.conjugate_gate = nn.Parameter(torch.zeros(output_modes))

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        if real.shape != imag.shape or real.ndim != 4 or real.shape[1] != self.input_modes:
            message = "widely-linear complex convolution requires matching NCHW fields"
            raise ValueError(message)
        gate = torch.tanh(self.conjugate_gate).view(-1, 1, 1, 1)
        top_left = self.weight_real + gate * self.conjugate_real
        top_right = -self.weight_imag + gate * self.conjugate_imag
        bottom_left = self.weight_imag + gate * self.conjugate_imag
        bottom_right = self.weight_real - gate * self.conjugate_real
        kernel = torch.cat(
            (
                torch.cat((top_left, top_right), dim=1),
                torch.cat((bottom_left, bottom_right), dim=1),
            ),
            dim=0,
        )
        output = functional.conv2d(torch.cat((real, imag), dim=1), kernel)
        output_real, output_imag = output.split(self.output_modes, dim=1)
        return output_real, output_imag


class ComplexMagnitudeGate2d(nn.Module):
    """Bounded phase-preserving nonlinearity, initialized as the identity."""

    def __init__(self, modes: int) -> None:
        super().__init__()
        if modes <= 0:
            message = "complex magnitude gate requires positive modes"
            raise ValueError(message)
        self.modes = modes
        self.slope = nn.Parameter(torch.zeros(modes))
        self.bias = nn.Parameter(torch.zeros(modes))

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        if real.shape != imag.shape or real.ndim != 4 or real.shape[1] != self.modes:
            message = "complex magnitude gate requires matching NCHW fields"
            raise ValueError(message)
        energy = torch.log1p(real.float().square() + imag.float().square())
        logits = self.slope.view(1, -1, 1, 1) * energy + self.bias.view(1, -1, 1, 1)
        gain = (2.0 * torch.sigmoid(logits)).to(dtype=real.dtype)
        return gain * real, gain * imag


class ComplexPixelResidualConvBlock(nn.Module):
    """Complex spatial refinement kept inside an unnormalized identity path."""

    def __init__(self, modes: int, layer_scale_initial: float = 0.1) -> None:
        super().__init__()
        if layer_scale_initial <= 0.0:
            message = "complex pixel residual LayerScale must be positive"
            raise ValueError(message)
        self.modes = modes
        self.norm = ComplexRMSNorm(modes)
        self.depthwise = ComplexDepthwiseConv2d(modes)
        self.activation = ComplexMagnitudeGate2d(modes)
        self.mixing = GatedWidelyLinearConv2d(modes, modes)
        self.layer_scale = nn.Parameter(torch.full((modes,), layer_scale_initial))

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        if real.shape != imag.shape or real.ndim != 4 or real.shape[1] != self.modes:
            message = "complex pixel residual block requires matching NCHW fields"
            raise ValueError(message)
        normalized_real, normalized_imag = self.norm(
            real.permute(0, 2, 3, 1),
            imag.permute(0, 2, 3, 1),
        )
        branch_real, branch_imag = self.depthwise(
            normalized_real.permute(0, 3, 1, 2),
            normalized_imag.permute(0, 3, 1, 2),
        )
        branch_real, branch_imag = self.activation(branch_real, branch_imag)
        branch_real, branch_imag = self.mixing(branch_real, branch_imag)
        scale = self.layer_scale.to(dtype=real.dtype).view(1, -1, 1, 1)
        return real + scale * branch_real, imag + scale * branch_imag


class ComplexPixelStem(nn.Module):
    """Lossless RGB complex encoding and S2D packing with residual CConv refinement."""

    def __init__(
        self,
        modes: int,
        *,
        reductions: tuple[int, int] = (2, 2),
        layer_scale_initial: float = 0.1,
    ) -> None:
        super().__init__()
        if len(reductions) != 2 or any(factor <= 0 for factor in reductions):
            message = "complex pixel stem requires two positive S2D reductions"
            raise ValueError(message)
        first_modes = 2 * reductions[0] ** 2
        output_modes = first_modes * reductions[1] ** 2
        if modes != output_modes:
            message = "complex pixel stem output modes must equal 2 * product(reductions)^2"
            raise ValueError(message)
        self.modes = modes
        self.reductions = reductions
        self.first_block = ComplexPixelResidualConvBlock(
            first_modes,
            layer_scale_initial,
        )
        self.second_block = ComplexPixelResidualConvBlock(
            output_modes,
            layer_scale_initial,
        )

    @staticmethod
    def encode_pixels(inputs: Tensor) -> ComplexField:
        """Map RGB to the orthonormal coordinates L and C1+iC2."""
        if inputs.ndim != 4 or inputs.shape[1] != 3:
            message = "complex pixel encoding requires a BCHW RGB tensor"
            raise ValueError(message)
        red, green, blue = inputs.unbind(dim=1)
        luminance = (red + green + blue) / math.sqrt(3.0)
        chroma_real = (2.0 * red - green - blue) / math.sqrt(6.0)
        chroma_imag = (green - blue) / math.sqrt(2.0)
        return (
            torch.stack((luminance, chroma_real), dim=1),
            torch.stack((torch.zeros_like(luminance), chroma_imag), dim=1),
        )

    @staticmethod
    def decode_pixels(real: Tensor, imag: Tensor) -> Tensor:
        """Invert the constrained two-complex-channel RGB encoding."""
        if real.shape != imag.shape or real.ndim != 4 or real.shape[1] != 2:
            message = "complex pixel decoding requires matching two-channel fields"
            raise ValueError(message)
        luminance, chroma_real = real.unbind(dim=1)
        chroma_imag = imag[:, 1]
        red = luminance / math.sqrt(3.0) + 2.0 * chroma_real / math.sqrt(6.0)
        green = (
            luminance / math.sqrt(3.0) - chroma_real / math.sqrt(6.0) + chroma_imag / math.sqrt(2.0)
        )
        blue = (
            luminance / math.sqrt(3.0) - chroma_real / math.sqrt(6.0) - chroma_imag / math.sqrt(2.0)
        )
        return torch.stack((red, green, blue), dim=1)

    def pack_pixels(self, real: Tensor, imag: Tensor) -> ComplexField:
        for factor in self.reductions:
            real = functional.pixel_unshuffle(real, factor)
            imag = functional.pixel_unshuffle(imag, factor)
        return real, imag

    def unpack_pixels(self, real: Tensor, imag: Tensor) -> ComplexField:
        for factor in reversed(self.reductions):
            real = functional.pixel_shuffle(real, factor)
            imag = functional.pixel_shuffle(imag, factor)
        return real, imag

    def forward(self, inputs: Tensor) -> ComplexField:
        real, imag = self.encode_pixels(inputs)
        first_factor, second_factor = self.reductions
        real = functional.pixel_unshuffle(real, first_factor)
        imag = functional.pixel_unshuffle(imag, first_factor)
        real, imag = self.first_block(real, imag)
        real = functional.pixel_unshuffle(real, second_factor)
        imag = functional.pixel_unshuffle(imag, second_factor)
        real, imag = self.second_block(real, imag)
        return real.permute(0, 2, 3, 1), imag.permute(0, 2, 3, 1)


class ComplexScanBackbone(nn.Module):
    """Real stem and head connected by one continuous complex modal backbone."""

    def __init__(  # noqa: PLR0915
        self,
        config: ComplexScanConfig | None = None,
    ) -> None:
        super().__init__()
        active = config or ComplexScanConfig()
        active.validate()
        self.config = active
        self.stem: nn.Module
        self.input_norm: nn.Module | None
        self.precomplex_fc: nn.Module | None
        self.analysis: nn.Linear | None

        def make_precomplex_fc() -> nn.Module:
            if not active.use_precomplex_fc:
                return nn.Identity()
            layers: list[nn.Module] = []
            for _ in range(active.precomplex_fc_layers):
                layers.extend(
                    (
                        nn.Linear(active.stem_width, active.stem_width),
                        nn.GELU(),
                    )
                )
            return nn.Sequential(*layers)

        if active.stem == "local_fourier":
            self.stem = LocalComplexFourierStem(
                active.modes[0],
                color_stride=active.stem_strides[0],
                fourier_hop=active.stem_strides[1],
            )
            self.input_norm = None
            self.precomplex_fc = None
            self.analysis = None
        elif active.stem == "complex_pixel":
            self.stem = ComplexPixelStem(
                active.modes[0],
                reductions=active.stem_strides,
            )
            self.input_norm = None
            self.precomplex_fc = None
            self.analysis = None
        elif active.stem == "conv_only":
            self.stem = CifarConvOnlyStem(active.stem_width, active.stem_strides)
            self.input_norm = (
                nn.RMSNorm(active.stem_width) if active.use_post_stem_rmsnorm else nn.Identity()
            )
            self.precomplex_fc = make_precomplex_fc()
            self.analysis = nn.Linear(active.stem_width, 2 * active.modes[0], bias=False)
        elif active.stem == "normalized_no_activation":
            self.stem = CifarNormalizedLinearStem(active.stem_width, active.stem_strides)
            self.input_norm = (
                nn.RMSNorm(active.stem_width) if active.use_post_stem_rmsnorm else nn.Identity()
            )
            self.precomplex_fc = make_precomplex_fc()
            self.analysis = nn.Linear(active.stem_width, 2 * active.modes[0], bias=False)
        else:
            self.stem = CifarConvStem(
                active.stem_width,
                active.stem_strides,
                bias=True,
            )
            self.input_norm = (
                nn.RMSNorm(active.stem_width) if active.use_post_stem_rmsnorm else nn.Identity()
            )
            self.precomplex_fc = make_precomplex_fc()
            self.analysis = nn.Linear(active.stem_width, 2 * active.modes[0], bias=False)
        if self.analysis is not None:
            nn.init.orthogonal_(self.analysis.weight)
            orthogonal(
                self.analysis,
                "weight",
                orthogonal_map="matrix_exp",
                use_trivialization=True,
            )
        maximum_phases = (math.pi * 0.75, math.pi * 0.70, math.pi * 0.65)
        transition_widths = active.transition_widths or (None, None)
        interaction_ranks = active.interaction_ranks or (None, None)
        augmented_widths = active.augmented_widths or (None, None)
        quadrant_path_mode_cffn_widths = active.quadrant_path_mode_cffn_widths or (
            None,
            None,
        )
        quadrant_path_cffn_widths = active.quadrant_path_cffn_widths or (None, None)
        post_transition_widths = active.post_transition_widths or (None, None)

        def make_stage(index: int) -> ComplexScanStage:
            return ComplexScanStage(
                active.modes[index],
                maximum_phase=maximum_phases[index],
                output_modes=active.modes[index + 1],
                transition_width=transition_widths[index],
                interaction_rank=interaction_ranks[index],
                widely_linear=active.widely_linear_bridges,
                augmented_width=augmented_widths[index],
                carry_basis=active.carry_bases[index],
                carry_merge=active.carry_merge,
                carry_scale_initial=active.carry_scale_initial,
                coherence_gated_carry=active.coherence_gated_carry,
                use_pole_aligned_shortcut=active.use_pole_aligned_shortcuts,
                use_cccn_shortcut=active.use_cccn_shortcuts,
                use_zero_gated_pole_aligned_residual=active.zero_gated_pole_aligned_residuals,
                quadrant_path_mode_cffn_width=quadrant_path_mode_cffn_widths[index],
                quadrant_path_cffn_width=quadrant_path_cffn_widths[index],
                post_transition_width=post_transition_widths[index],
                stage_residual_scale_initial=active.stage_residual_scale_initial,
                scan_memory_policy=active.scan_memory_policy,
                damping_min=active.damping_min,
                damping_max=active.damping_max,
            )

        self.stage1 = make_stage(0)
        self.stage2 = make_stage(1)
        self.terminal = ComplexScanStage(
            active.modes[2],
            maximum_phase=maximum_phases[2],
            output_modes=None,
            scan_memory_policy=active.scan_memory_policy,
            damping_min=active.damping_min,
            damping_max=active.damping_max,
        )
        self.descriptor_dim = 4 * sum(active.modes)
        self.classifier = (
            ParallelFusionLRQHead(
                self.descriptor_dim,
                active.fusion_width,
                active.output_dim,
                active.quadratic_rank,
            )
            if active.dual_fusion_lrq_head and active.fusion_width is not None
            else ModalFusionHead(
                self.descriptor_dim,
                active.fusion_width,
                active.output_dim,
            )
            if active.fusion_width is not None
            else LowRankQuadraticModalHead(
                self.descriptor_dim,
                active.output_dim,
                active.quadratic_rank,
            )
        )

    def prepare_for_inference_(self) -> ComplexScanBackbone:
        """Materialize every reparametrized weight in place for inference.

        The orthogonal analysis frame is produced by a ``matrix_exp``
        parametrization that is rebuilt on every access to ``.weight``.  Its
        arithmetic is negligible, but the parametrization machinery around it
        costs several milliseconds of host time per forward, which dominates a
        dispatch-bound model.  Baking the realized weight removes that cost
        without changing a single output value.

        It also makes the whole forward CUDA-Graph capturable: ``matrix_exp``
        picks its Pade order from a host-visible norm, and that synchronization
        invalidates capture.  Replaying a captured prepared model is where most
        of the available inference speedup lives.

        This is a one-way, inference-only conversion.  It drops the
        orthogonality constraint, so the model must not be trained afterwards,
        and it renames ``analysis.parametrizations.weight.*`` to
        ``analysis.weight`` in ``state_dict``.  Call it only after checkpoint
        loading and the final device/dtype move.
        """
        bake_parametrizations_(self)
        self.eval()
        return self

    def prepare_for_compiled_training_(self) -> ComplexScanBackbone:
        """Make the model capturable by ``torch.compile(mode="reduce-overhead")``.

        The orthogonal analysis frame stays parametrized and trainable; only
        Dynamo tracing stops at it, so the rest of the step can be recorded as
        a CUDA Graph.  Without this, ``reduce-overhead`` aborts capture and the
        step falls back to the roughly 2x slower non-captured compile.

        Nothing about the computation, the parameters, or ``state_dict``
        changes, so this is safe to call before training.
        """
        untraced_parametrizations_(self)
        return self

    @staticmethod
    def _require_state(state: ComplexField | None) -> ComplexField:
        if state is None:
            message = "non-terminal complex pole stage returned no state"
            raise RuntimeError(message)
        return state

    def _initial_excitation(self, inputs: Tensor) -> ComplexField:
        if isinstance(self.stem, (LocalComplexFourierStem, ComplexPixelStem)):
            return self.stem(inputs)
        if self.input_norm is None or self.precomplex_fc is None or self.analysis is None:
            message = "real complex-scan stem is missing its analysis projection"
            raise RuntimeError(message)
        transformed = self.precomplex_fc(self.stem(inputs))
        normalized = _dtype_aligned_rms_norm(transformed, self.input_norm)
        return self.analysis(normalized).chunk(2, dim=-1)

    def complex_features(self, inputs: Tensor) -> tuple[ComplexField, ComplexField]:
        excitation_real, excitation_imag = self._initial_excitation(inputs)
        state2, _ = self.stage1(excitation_real, excitation_imag)
        state2 = self._require_state(state2)
        state3, _ = self.stage2(*state2)
        return state2, self._require_state(state3)

    def raw_descriptor(self, inputs: Tensor) -> Tensor:
        excitation_real, excitation_imag = self._initial_excitation(inputs)
        state2, descriptor1 = self.stage1(excitation_real, excitation_imag)
        state2 = self._require_state(state2)
        state3, descriptor2 = self.stage2(*state2)
        state3 = self._require_state(state3)
        _, descriptor3 = self.terminal(*state3)
        return torch.cat((descriptor1, descriptor2, descriptor3), dim=-1)

    def forward(self, inputs: Tensor) -> Tensor:
        descriptor = self.raw_descriptor(inputs)
        with torch.autocast(device_type=descriptor.device.type, enabled=False):
            return self.classifier(descriptor.float())
