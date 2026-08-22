"""Low-rank strict-complex spatial readers for product-scan inputs."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional

from .complex_scan_transitions import ComplexRMSNorm

ComplexField = tuple[Tensor, Tensor]


class FactorizedComplexConv2dReader(nn.Module):
    """Apply complex pointwise analysis followed by pole-wise spatial filters.

    The effective kernel is

        W[p, k, dy, dx] = sum_r D[p, r, dy, dx] * A[p, r, k],

    where ``A`` and ``D`` are strict-complex weights.  ``rank`` controls how
    many independently detected local events each output pole can combine.
    """

    def __init__(
        self,
        input_modes: int,
        output_modes: int,
        *,
        rank: int = 2,
        kernel_size: int = 3,
        variance_epsilon: float = 1.0e-12,
        normalize_input: bool = False,
        match_input_rms: bool = False,
        rms_reference_modes: int | None = None,
    ) -> None:
        super().__init__()
        if input_modes <= 0 or output_modes <= 0:
            message = "factorized complex reader dimensions must be positive"
            raise ValueError(message)
        if rank <= 0:
            message = "factorized complex reader rank must be positive"
            raise ValueError(message)
        if kernel_size <= 0 or kernel_size % 2 == 0:
            message = "factorized complex reader kernel size must be a positive odd integer"
            raise ValueError(message)
        if variance_epsilon <= 0.0:
            message = "factorized complex reader variance epsilon must be positive"
            raise ValueError(message)

        self.input_modes = int(input_modes)
        self.output_modes = int(output_modes)
        self.rank = int(rank)
        self.kernel_size = int(kernel_size)
        self.padding = kernel_size // 2
        self.variance_epsilon = float(variance_epsilon)
        self.normalize_input = bool(normalize_input)
        self.match_input_rms = bool(match_input_rms)
        active_rms_modes = output_modes if rms_reference_modes is None else int(rms_reference_modes)
        if active_rms_modes <= 0 or active_rms_modes > output_modes:
            message = "reader RMS reference modes must be within the output width"
            raise ValueError(message)
        self.rms_reference_modes = active_rms_modes
        self.input_norm: ComplexRMSNorm | None = (
            ComplexRMSNorm(input_modes) if self.normalize_input else None
        )

        point_shape = (output_modes, rank, input_modes)
        spatial_shape = (output_modes, rank, kernel_size, kernel_size)
        self.point_weight_real = nn.Parameter(torch.empty(point_shape))
        self.point_weight_imag = nn.Parameter(torch.empty(point_shape))
        self.spatial_weight_real = nn.Parameter(torch.empty(spatial_shape))
        self.spatial_weight_imag = nn.Parameter(torch.empty(spatial_shape))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Use a stable generic initialization for non-square readers."""
        nn.init.xavier_uniform_(self.point_weight_real)
        nn.init.xavier_uniform_(self.point_weight_imag)
        with torch.no_grad():
            scale = math.sqrt(0.5)
            self.point_weight_real.mul_(scale)
            self.point_weight_imag.mul_(scale)
            self.spatial_weight_real.zero_()
            self.spatial_weight_imag.zero_()
            self.spatial_weight_real[:, :, self.padding, self.padding].fill_(
                1.0 / math.sqrt(self.rank)
            )

    def initialize_orthogonal_(self) -> None:
        """Start near identity with active, mutually orthogonal rank components.

        The first component is the identity detector and a centered impulse.
        Every remaining component receives an independent complex point detector
        orthogonal to that identity channel and an independent complex spatial
        filter orthogonal to the centered impulse.  Their total initial energy is
        derived from the input width, so no model shape is special-cased.
        """
        if self.input_modes != self.output_modes:
            message = "orthogonal initialization requires equal input and output modes"
            raise ValueError(message)
        maximum_rank = min(self.input_modes, self.kernel_size * self.kernel_size)
        if self.rank > maximum_rank:
            message = (
                "orthogonal initialization requires rank no greater than both "
                "the input width and spatial support"
            )
            raise ValueError(message)
        with torch.no_grad():
            self.point_weight_real.zero_()
            self.point_weight_imag.zero_()
            self.spatial_weight_real.zero_()
            self.spatial_weight_imag.zero_()
            indices = torch.arange(self.input_modes, device=self.point_weight_real.device)
            self.point_weight_real[indices, 0, indices] = 1.0
            self.spatial_weight_real[:, 0, self.padding, self.padding] = 1.0

            if self.rank == 1:
                return

            residual_rank = self.rank - 1
            point_candidates = torch.complex(
                torch.randn(
                    self.output_modes,
                    self.input_modes,
                    residual_rank,
                    device=self.point_weight_real.device,
                    dtype=self.point_weight_real.dtype,
                ),
                torch.randn(
                    self.output_modes,
                    self.input_modes,
                    residual_rank,
                    device=self.point_weight_real.device,
                    dtype=self.point_weight_real.dtype,
                ),
            )
            point_candidates[indices, indices] = 0.0
            point_basis = torch.linalg.qr(point_candidates, mode="reduced").Q.movedim(-1, -2)
            self.point_weight_real[:, 1:].copy_(point_basis.real)
            self.point_weight_imag[:, 1:].copy_(point_basis.imag)

            spatial_size = self.kernel_size * self.kernel_size
            spatial_candidates = torch.complex(
                torch.randn(
                    self.output_modes,
                    spatial_size,
                    residual_rank,
                    device=self.spatial_weight_real.device,
                    dtype=self.spatial_weight_real.dtype,
                ),
                torch.randn(
                    self.output_modes,
                    spatial_size,
                    residual_rank,
                    device=self.spatial_weight_real.device,
                    dtype=self.spatial_weight_real.dtype,
                ),
            )
            center = self.padding * self.kernel_size + self.padding
            spatial_candidates[:, center] = 0.0
            spatial_basis = torch.linalg.qr(spatial_candidates, mode="reduced").Q
            spatial_basis = spatial_basis.movedim(-1, -2).reshape(
                self.output_modes,
                residual_rank,
                self.kernel_size,
                self.kernel_size,
            )
            residual_scale = 1.0 / math.sqrt(self.input_modes)
            self.spatial_weight_real[:, 1:].copy_(spatial_basis.real * residual_scale)
            self.spatial_weight_imag[:, 1:].copy_(spatial_basis.imag * residual_scale)

    def initialize_semi_orthogonal_(self, *, identity_when_square: bool = True) -> None:
        """Initialize a rectangular reader without privileging a fixed width.

        Square readers keep the exact identity-centered initialization above.
        Rectangular readers instead receive the closest available isometry:
        orthonormal rows when compressing and orthonormal columns when
        expanding.  Remaining rank components are independently orthogonal to
        each pole's primary detector and to the centered spatial impulse.

        Set ``identity_when_square=False`` to sample one joint orthogonal square
        analysis basis while retaining the same rank-component contract.
        """
        if self.input_modes == self.output_modes and identity_when_square:
            self.initialize_orthogonal_()
            return
        maximum_rank = min(self.input_modes, self.kernel_size * self.kernel_size)
        if self.rank > maximum_rank:
            message = (
                "semi-orthogonal initialization requires rank no greater than "
                "both the input width and spatial support"
            )
            raise ValueError(message)

        with torch.no_grad():
            self.point_weight_real.zero_()
            self.point_weight_imag.zero_()
            self.spatial_weight_real.zero_()
            self.spatial_weight_imag.zero_()

            primary = torch.empty(
                self.output_modes,
                self.input_modes,
                device=self.point_weight_real.device,
                dtype=self.point_weight_real.dtype,
            )
            nn.init.orthogonal_(primary)
            self.point_weight_real[:, 0].copy_(primary)
            self.spatial_weight_real[:, 0, self.padding, self.padding] = 1.0
            if self.rank == 1:
                return

            residual_rank = self.rank - 1
            primary_complex = torch.complex(primary, torch.zeros_like(primary))
            point_candidates = torch.complex(
                torch.randn(
                    self.output_modes,
                    residual_rank,
                    self.input_modes,
                    device=self.point_weight_real.device,
                    dtype=self.point_weight_real.dtype,
                ),
                torch.randn(
                    self.output_modes,
                    residual_rank,
                    self.input_modes,
                    device=self.point_weight_real.device,
                    dtype=self.point_weight_real.dtype,
                ),
            )
            overlap = (primary_complex[:, None].conj() * point_candidates).sum(dim=-1, keepdim=True)
            primary_energy = primary.square().sum(dim=-1, keepdim=True)[:, None]
            point_candidates = point_candidates - (
                overlap * primary_complex[:, None] / primary_energy.clamp_min(1.0e-12)
            )
            point_candidates = point_candidates / torch.linalg.vector_norm(
                point_candidates,
                dim=-1,
                keepdim=True,
            ).clamp_min(1.0e-12)
            self.point_weight_real[:, 1:].copy_(point_candidates.real)
            self.point_weight_imag[:, 1:].copy_(point_candidates.imag)

            spatial_size = self.kernel_size * self.kernel_size
            spatial_candidates = torch.complex(
                torch.randn(
                    self.output_modes,
                    residual_rank,
                    spatial_size,
                    device=self.spatial_weight_real.device,
                    dtype=self.spatial_weight_real.dtype,
                ),
                torch.randn(
                    self.output_modes,
                    residual_rank,
                    spatial_size,
                    device=self.spatial_weight_real.device,
                    dtype=self.spatial_weight_real.dtype,
                ),
            )
            center = self.padding * self.kernel_size + self.padding
            spatial_candidates[..., center] = 0.0
            spatial_candidates = spatial_candidates / torch.linalg.vector_norm(
                spatial_candidates,
                dim=-1,
                keepdim=True,
            ).clamp_min(1.0e-12)
            spatial_candidates = spatial_candidates.reshape(
                self.output_modes,
                residual_rank,
                self.kernel_size,
                self.kernel_size,
            )
            residual_scale = 1.0 / math.sqrt(self.input_modes)
            self.spatial_weight_real[:, 1:].copy_(spatial_candidates.real * residual_scale)
            self.spatial_weight_imag[:, 1:].copy_(spatial_candidates.imag * residual_scale)

    def synthesized_kernel(self) -> ComplexField:
        """Synthesize the strict-complex K-by-K kernel represented by the factors."""
        point_real = self.point_weight_real.float()
        point_imag = self.point_weight_imag.float()
        spatial_real = self.spatial_weight_real.float()
        spatial_imag = self.spatial_weight_imag.float()
        full_real = torch.einsum("prij,prk->pkij", spatial_real, point_real)
        full_real = full_real - torch.einsum("prij,prk->pkij", spatial_imag, point_imag)
        full_imag = torch.einsum("prij,prk->pkij", spatial_real, point_imag)
        full_imag = full_imag + torch.einsum("prij,prk->pkij", spatial_imag, point_real)
        return full_real, full_imag

    def unit_energy_kernel(self) -> ComplexField:
        """Return the synthesized kernel with exact unit row energy."""
        full_real, full_imag = self.synthesized_kernel()
        inverse_rms = torch.rsqrt(
            full_real.square()
            .add(full_imag.square())
            .sum(dim=(1, 2, 3), keepdim=True)
            .clamp_min(self.variance_epsilon)
        )
        return full_real * inverse_rms, full_imag * inverse_rms

    def _validate_input(self, real: Tensor, imag: Tensor) -> None:
        if real.shape != imag.shape or real.ndim != 4:
            message = "factorized complex reader requires matching NHWM fields"
            raise ValueError(message)
        if real.shape[-1] != self.input_modes:
            message = "factorized complex reader input has an incompatible mode dimension"
            raise ValueError(message)

    def _apply_packed_kernel(
        self,
        real: Tensor,
        imag: Tensor,
        packed_kernel: Tensor,
    ) -> ComplexField:
        """Apply one packed Cartesian convolution and the optional RMS match."""
        packed_input = torch.cat(
            (real.movedim(-1, 1), imag.movedim(-1, 1)),
            dim=1,
        )
        output = functional.conv2d(
            packed_input,
            packed_kernel,
            padding=self.padding,
        )

        if self.match_input_rms:
            input_energy = (
                packed_input.float().square().sum(dim=1, keepdim=True).div(self.input_modes)
            )
            output_energy = (
                output.float()
                .reshape(
                    output.shape[0],
                    2,
                    self.output_modes,
                    output.shape[2],
                    output.shape[3],
                )[:, :, : self.rms_reference_modes]
                .square()
                .sum(dim=(1, 2), keepdim=False)
                .unsqueeze(1)
                .div(self.rms_reference_modes)
            )
            token_scale = torch.sqrt(
                (input_energy + self.variance_epsilon) / (output_energy + self.variance_epsilon)
            ).to(dtype=output.dtype)
            output = output * token_scale

        return (
            output[:, : self.output_modes].movedim(1, -1).contiguous(),
            output[:, self.output_modes :].movedim(1, -1).contiguous(),
        )

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        self._validate_input(real, imag)
        if self.input_norm is not None:
            real, imag = self.input_norm(real, imag)
        kernel_real, kernel_imag = self.unit_energy_kernel()
        kernel_top = torch.cat((kernel_real, -kernel_imag), dim=1)
        kernel_bottom = torch.cat((kernel_imag, kernel_real), dim=1)
        packed_kernel = torch.cat((kernel_top, kernel_bottom), dim=0)
        return self._apply_packed_kernel(real, imag, packed_kernel)


class GatedWidelyLinearFactorizedComplexConv2dReader(nn.Module):
    """Add a zero-gated factorized conjugate branch to a strict reader."""

    def __init__(self, strict_reader: FactorizedComplexConv2dReader) -> None:
        super().__init__()
        if type(strict_reader) is not FactorizedComplexConv2dReader:
            message = "gated widely-linear reader requires a factorized strict reader"
            raise TypeError(message)
        self.strict_reader = strict_reader
        self.conjugate_point_weight_real = nn.Parameter(
            torch.empty_like(strict_reader.point_weight_real)
        )
        self.conjugate_point_weight_imag = nn.Parameter(
            torch.empty_like(strict_reader.point_weight_imag)
        )
        self.conjugate_spatial_weight_real = nn.Parameter(
            torch.empty_like(strict_reader.spatial_weight_real)
        )
        self.conjugate_spatial_weight_imag = nn.Parameter(
            torch.empty_like(strict_reader.spatial_weight_imag)
        )
        self.conjugate_gate = nn.Parameter(torch.zeros(strict_reader.output_modes))
        with torch.no_grad():
            self.conjugate_point_weight_real.copy_(strict_reader.point_weight_real)
            self.conjugate_point_weight_imag.copy_(strict_reader.point_weight_imag)
            self.conjugate_spatial_weight_real.copy_(strict_reader.spatial_weight_real)
            self.conjugate_spatial_weight_imag.copy_(strict_reader.spatial_weight_imag)

    @classmethod
    def from_strict(
        cls,
        source: FactorizedComplexConv2dReader,
    ) -> GatedWidelyLinearFactorizedComplexConv2dReader:
        return cls(source)

    @property
    def input_modes(self) -> int:
        return self.strict_reader.input_modes

    @property
    def output_modes(self) -> int:
        return self.strict_reader.output_modes

    @property
    def rank(self) -> int:
        return self.strict_reader.rank

    @property
    def kernel_size(self) -> int:
        return self.strict_reader.kernel_size

    def synthesized_conjugate_kernel(self) -> ComplexField:
        point_real = self.conjugate_point_weight_real.float()
        point_imag = self.conjugate_point_weight_imag.float()
        spatial_real = self.conjugate_spatial_weight_real.float()
        spatial_imag = self.conjugate_spatial_weight_imag.float()
        full_real = torch.einsum("prij,prk->pkij", spatial_real, point_real)
        full_real = full_real - torch.einsum("prij,prk->pkij", spatial_imag, point_imag)
        full_imag = torch.einsum("prij,prk->pkij", spatial_real, point_imag)
        full_imag = full_imag + torch.einsum("prij,prk->pkij", spatial_imag, point_real)
        return full_real, full_imag

    def joint_unit_energy_kernels(self) -> tuple[ComplexField, ComplexField]:
        weight_real, weight_imag = self.strict_reader.synthesized_kernel()
        conjugate_real, conjugate_imag = self.synthesized_conjugate_kernel()
        gate = torch.tanh(self.conjugate_gate.float()).view(-1, 1, 1, 1)
        conjugate_real = conjugate_real * gate
        conjugate_imag = conjugate_imag * gate
        inverse_rms = torch.rsqrt(
            weight_real.square()
            .add(weight_imag.square())
            .add(conjugate_real.square())
            .add(conjugate_imag.square())
            .sum(dim=(1, 2, 3), keepdim=True)
            .clamp_min(self.strict_reader.variance_epsilon)
        )
        return (
            (weight_real * inverse_rms, weight_imag * inverse_rms),
            (conjugate_real * inverse_rms, conjugate_imag * inverse_rms),
        )

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        strict = self.strict_reader
        strict._validate_input(real, imag)
        if strict.input_norm is not None:
            real, imag = strict.input_norm(real, imag)
        (weight_real, weight_imag), (conjugate_real, conjugate_imag) = (
            self.joint_unit_energy_kernels()
        )
        top_left = weight_real + conjugate_real
        top_right = conjugate_imag - weight_imag
        bottom_left = weight_imag + conjugate_imag
        bottom_right = weight_real - conjugate_real
        packed_kernel = torch.cat(
            (
                torch.cat((top_left, top_right), dim=1),
                torch.cat((bottom_left, bottom_right), dim=1),
            ),
            dim=0,
        )
        return strict._apply_packed_kernel(real, imag, packed_kernel)


__all__ = [
    "FactorizedComplexConv2dReader",
    "GatedWidelyLinearFactorizedComplexConv2dReader",
]
