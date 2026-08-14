"""Multiple learned pole dynamics per shared excitation mode.

The scan is evaluated once per pole replica with the existing optimized D4
product-scan kernels.  Replica responses are fused before the established
stage transition and before the exact full-grid Q measurement.  Keeping the
replica loop outside the kernel avoids materializing an NHW4MR tensor while
remaining mathematically identical to a mode-wise strict complex R-to-1 map.
"""

from __future__ import annotations

# This experiment deliberately composes private stage primitives without
# changing the baseline stage implementation.
# pyright: reportPrivateUsage=false
# ruff: noqa: ANN201, C901, EM101, SLF001, TRY003
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor, nn

from .complex_scan_stage import ComplexScanStage, complex_carry_coordinates
from .pac_product_scan_normalization import static_variance_tables
from .pac_product_scan_pipeline import run_product_scan_pipeline
from .pac_real2d_math import discrete_pole_real2d
from .pac_triton_bidirectional_product_scan import (
    pac_triton_bidirectional_product_scan,
)

if TYPE_CHECKING:
    from .complex_scan_types import ComplexField

Pole = tuple[Tensor, Tensor, Tensor, Tensor]

_DIRECTIONS = ((0, 0), (1, 0), (0, 1), (1, 1))
# ``full16`` stores each 2x2 cell in direction-relative order.  These
# permutations align all four paths to the physical ordering used by (++).
_PHYSICAL_CELL_ORDER = (
    (0, 1, 2, 3),
    (2, 3, 0, 1),
    (1, 0, 3, 2),
    (3, 2, 1, 0),
)


def _descriptor_from_full_states(
    real: Tensor,
    imag: Tensor,
) -> Tensor:
    """Measure exact full-grid Q after pole-replica fusion.

    Inputs use ``NHW4M`` for terminal scans or ``NHW44M`` for the full16
    non-terminal epilogue.  The local-cell axis is realigned before raw
    directional energies are accumulated.
    """
    if (
        real.shape != imag.shape
        or real.ndim not in {5, 6}
        or real.shape[-2 if real.ndim == 5 else -3] != 4
    ):
        raise ValueError("multi-pole Q requires matching NHW4M or NHW44M states")
    if real.ndim == 5:
        # Add a singleton physical-cell axis for a common implementation.
        aligned_real = real.unsqueeze(-2)
        aligned_imag = imag.unsqueeze(-2)
    else:
        aligned_real = torch.stack(
            [
                real[..., direction, list(order), :]
                for direction, order in enumerate(_PHYSICAL_CELL_ORDER)
            ],
            dim=-3,
        )
        aligned_imag = torch.stack(
            [
                imag[..., direction, list(order), :]
                for direction, order in enumerate(_PHYSICAL_CELL_ORDER)
            ],
            dim=-3,
        )

    # NHW4CM -> NHWCM4, where C is one or four physical cells.
    packed_real = aligned_real.permute(0, 1, 2, 4, 5, 3)
    packed_imag = aligned_imag.permute(0, 1, 2, 4, 5, 3)
    raw_energy = packed_real.float().square().add(packed_imag.float().square())
    # B,C,M,D -> B,D,M to retain the established direction-major layout.
    raw = raw_energy.mean((1, 2, 3)).permute(0, 2, 1)
    return torch.log1p(raw).flatten(1)


class MultiPoleExcitationStage(nn.Module):
    """Fan each excitation mode out to R poles, then fuse strict-complexly."""

    response_correlation_mean: Tensor
    response_correlation_max: Tensor
    diagnostic_updates: Tensor

    def __init__(self, stage: ComplexScanStage, multiplicity: int) -> None:
        super().__init__()
        if multiplicity <= 0 or stage.modes % multiplicity:
            raise ValueError("pole multiplicity must be a positive divisor of the mode width")
        if (
            stage.use_pole_aligned_shortcut
            or stage.use_cccn_shortcut
            or stage.use_zero_gated_pole_aligned_residual
        ):
            raise TypeError("multi-pole experiment requires the shortcut-free matched control")
        if stage.quadrant_path_mode_combiner is not None and bool(
            stage.quadrant_path_mode_combiner.requires_full_product_cells
        ):
            raise TypeError("multi-pole experiment requires the coarse D4 transition control")
        self.stage = stage
        self.multiplicity = multiplicity
        self.modes = stage.modes
        self.output_modes = stage.output_modes

        offsets = tuple(replica * self.modes // multiplicity for replica in range(1, multiplicity))

        def replicas(source: Tensor) -> Tensor:
            if not offsets:
                return source.new_empty((0, self.modes))
            return torch.stack([torch.roll(source.detach(), shifts=offset) for offset in offsets])

        # Parameter names deliberately retain the geometry markers used by the
        # matched pole-aware optimizer grouping.
        self.damping_logits_x_extra = nn.Parameter(replicas(stage.damping_logits_x))
        self.damping_logits_y_extra = nn.Parameter(replicas(stage.damping_logits_y))
        self.phase_x_extra = nn.Parameter(replicas(stage.phase_x))
        self.phase_y_extra = nn.Parameter(replicas(stage.phase_y))
        self.fusion_weight_real = nn.Parameter(
            torch.full((self.modes, multiplicity), 1.0 / multiplicity)
        )
        self.fusion_weight_imag = nn.Parameter(torch.zeros(self.modes, multiplicity))
        self.register_buffer("response_correlation_mean", torch.zeros(()), persistent=False)
        self.register_buffer("response_correlation_max", torch.zeros(()), persistent=False)
        self.register_buffer(
            "diagnostic_updates", torch.zeros((), dtype=torch.int64), persistent=False
        )

    @property
    def quadrant_path_mode_combiner(self):
        return self.stage.quadrant_path_mode_combiner

    @property
    def augmented(self):
        return self.stage.augmented

    def _geometry(self, replica: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if replica == 0:
            return (
                self.stage.damping_logits_x,
                self.stage.damping_logits_y,
                self.stage.phase_x,
                self.stage.phase_y,
            )
        index = replica - 1
        return (
            self.damping_logits_x_extra[index],
            self.damping_logits_y_extra[index],
            self.phase_x_extra[index],
            self.phase_y_extra[index],
        )

    def _pole_coefficients(
        self,
        shape: tuple[int, int, int, int],
        replica: int,
    ) -> tuple[Pole, Pole]:
        _, height, width, _ = shape
        logits_x, logits_y, phase_x, phase_y = self._geometry(replica)
        ratio_x = torch.sigmoid(logits_x).view(1, 1, 1, -1)
        ratio_y = torch.sigmoid(logits_y).view(1, 1, 1, -1)
        span = self.stage.damping_max - self.stage.damping_min
        damping_x = self.stage.damping_min + span * ratio_x
        damping_y = self.stage.damping_min + span * ratio_y
        spacing_x, spacing_y = 1.0 / width, 1.0 / height
        pole_x = discrete_pole_real2d(
            damping_x / spacing_x,
            (phase_x / spacing_x).view(1, 1, 1, -1),
            spacing_x,
        )
        pole_y = discrete_pole_real2d(
            damping_y / spacing_y,
            (phase_y / spacing_y).view(1, 1, 1, -1),
            spacing_y,
        )
        return pole_x, pole_y

    def _terminal_full_states(
        self,
        pole_x: Pole,
        pole_y: Pole,
        source: ComplexField,
    ) -> ComplexField:
        horizontal = pac_triton_bidirectional_product_scan(pole_x, source)

        def vertical(active: ComplexField) -> tuple[Tensor, Tensor, Tensor, Tensor]:
            transposed = active[0].transpose(1, 2), active[1].transpose(1, 2)
            result = pac_triton_bidirectional_product_scan(pole_y, transposed)
            return cast(
                "tuple[Tensor, Tensor, Tensor, Tensor]",
                tuple(value.transpose(1, 2) for value in result),
            )

        positive_x = vertical((horizontal[0], horizontal[1]))
        negative_x = vertical((horizontal[2], horizontal[3]))
        directional = (
            (positive_x[0], positive_x[1]),
            (negative_x[0], negative_x[1]),
            (positive_x[2], positive_x[3]),
            (negative_x[2], negative_x[3]),
        )
        _, height, width, _ = source[0].shape
        variance_x, variance_y = static_variance_tables(
            *(value.detach() for value in (*pole_x, *pole_y)),
            width,
            height,
        )
        real_paths = []
        imag_paths = []
        for (real, imag), (x_sign, y_sign) in zip(directional, _DIRECTIONS, strict=True):
            variance = variance_y[y_sign, :, None, :] * variance_x[x_sign, None, :, :]
            if self.stage.product_gain_normalization == "global":
                variance = variance.mean((0, 1), keepdim=True)
            inverse = torch.rsqrt(variance.clamp_min(1.0e-8)).to(real.dtype)
            real_paths.append(real * inverse)
            imag_paths.append(imag * inverse)
        return torch.stack(real_paths, dim=-2), torch.stack(imag_paths, dim=-2)

    def _scan_replica(
        self,
        real: Tensor,
        imag: Tensor,
        replica: int,
    ) -> ComplexField:
        shape = cast("tuple[int, int, int, int]", tuple(real.shape))
        pole_x, pole_y = self._pole_coefficients(shape, replica)
        if self.output_modes is None:
            return self._terminal_full_states(pole_x, pole_y, (real, imag))
        full_real, full_imag, _ = cast(
            "tuple[Tensor, Tensor, Tensor]",
            run_product_scan_pipeline(
                pole_x,
                pole_y,
                (real, imag),
                epilogue="full16",
                gain_normalization=self.stage.product_gain_normalization,
                memory_policy=self.stage.scan_memory_policy,
            ),
        )
        return full_real, full_imag

    def _weighted(self, state: ComplexField, replica: int) -> ComplexField:
        weight_real = self.fusion_weight_real[:, replica].to(state[0].dtype)
        weight_imag = self.fusion_weight_imag[:, replica].to(state[0].dtype)
        return (
            state[0] * weight_real - state[1] * weight_imag,
            state[0] * weight_imag + state[1] * weight_real,
        )

    @torch.no_grad()
    def _update_response_diagnostics(self, samples: list[ComplexField]) -> None:
        if len(samples) < 2:
            return
        correlations = []
        epsilon = 1.0e-8
        for left in range(len(samples)):
            left_real = samples[left][0].float().reshape(-1, self.modes)
            left_imag = samples[left][1].float().reshape(-1, self.modes)
            left_energy = left_real.square().add(left_imag.square()).sum(0)
            for right in range(left + 1, len(samples)):
                right_real = samples[right][0].float().reshape(-1, self.modes)
                right_imag = samples[right][1].float().reshape(-1, self.modes)
                inner_real = (left_real * right_real + left_imag * right_imag).sum(0)
                inner_imag = (left_real * right_imag - left_imag * right_real).sum(0)
                numerator = torch.sqrt(inner_real.square() + inner_imag.square())
                right_energy = right_real.square().add(right_imag.square()).sum(0)
                denominator = torch.sqrt(left_energy * right_energy).clamp_min(epsilon)
                correlations.append(numerator / denominator)
        packed = torch.stack(correlations)
        self.response_correlation_mean.copy_(packed.mean())
        self.response_correlation_max.copy_(packed.max())
        self.diagnostic_updates.add_(1)

    def diagnostic_metrics(self) -> dict[str, float]:
        geometries = [self._geometry(replica) for replica in range(self.multiplicity)]
        damping_x = torch.stack([values[0].detach().float() for values in geometries], dim=-1)
        damping_y = torch.stack([values[1].detach().float() for values in geometries], dim=-1)
        phase_x = torch.stack([values[2].detach().float() for values in geometries], dim=-1)
        phase_y = torch.stack([values[3].detach().float() for values in geometries], dim=-1)
        weight_real = self.fusion_weight_real.detach().float()
        weight_imag = self.fusion_weight_imag.detach().float()
        uniform = 1.0 / self.multiplicity
        return {
            "response_correlation_mean": float(self.response_correlation_mean),
            "response_correlation_max": float(self.response_correlation_max),
            "pole_damping_logit_spread": float(
                torch.cat(
                    (damping_x.std(-1, unbiased=False), damping_y.std(-1, unbiased=False))
                ).mean()
            ),
            "pole_phase_spread": float(
                torch.cat((phase_x.std(-1, unbiased=False), phase_y.std(-1, unbiased=False))).mean()
            ),
            "fusion_real_rms": float(weight_real.square().mean().sqrt()),
            "fusion_imag_rms": float(weight_imag.square().mean().sqrt()),
            "fusion_uniform_deviation_rms": float(
                (weight_real - uniform).square().add(weight_imag.square()).mean().sqrt()
            ),
        }

    def forward(self, real: Tensor, imag: Tensor) -> tuple[ComplexField | None, Tensor]:
        if real.shape != imag.shape or real.ndim != 4 or real.shape[-1] != self.modes:
            raise ValueError("multi-pole stage inputs must be matching NHWM tensors")
        fused_real: Tensor | None = None
        fused_imag: Tensor | None = None
        diagnostic_samples: list[ComplexField] = []
        for replica in range(self.multiplicity):
            state = self._scan_replica(real, imag, replica)
            if self.training:
                height = min(4, state[0].shape[1])
                width = min(4, state[0].shape[2])
                diagnostic_samples.append(
                    (
                        state[0][:1, :height, :width].detach(),
                        state[1][:1, :height, :width].detach(),
                    )
                )
            weighted_real, weighted_imag = self._weighted(state, replica)
            fused_real = weighted_real if fused_real is None else fused_real + weighted_real
            fused_imag = weighted_imag if fused_imag is None else fused_imag + weighted_imag
        if fused_real is None or fused_imag is None:
            raise RuntimeError("multi-pole stage produced no replicas")
        if self.training:
            self._update_response_diagnostics(diagnostic_samples)

        descriptor = _descriptor_from_full_states(fused_real, fused_imag)
        if self.output_modes is None:
            return None, descriptor

        # Local index three is the exact direction-aligned 2x2 endpoint.
        # The baseline coarse kernel emits contiguous NHW4M.  Selecting the
        # endpoint from full16 leaves a strided cell view, so materialize the
        # same row-major contract before the packed PG kernels.
        coarse_real = fused_real[..., 3, :].contiguous()
        coarse_imag = fused_imag[..., 3, :].contiguous()
        mixer = self.stage.quadrant_path_mode_combiner
        path_real, path_imag = coarse_real, coarse_imag
        collapse_paths = False
        if mixer is not None:
            path_real, path_imag = mixer.forward_packed(path_real, path_imag)
            collapse_paths = bool(getattr(mixer, "collapses_product_paths", False))
        if collapse_paths:
            if tuple(path_real.shape[-2:]) != (1, self.modes):
                raise RuntimeError("multi-pole path collapse changed its output contract")
            projected_input = path_real.squeeze(-2), path_imag.squeeze(-2)
        else:
            raise TypeError("multi-pole experiment requires the matched collapsed-path control")

        carry = (
            complex_carry_coordinates(real, imag, self.stage.carry_basis)
            if self.stage.carry_basis != "none"
            else None
        )
        output = self.stage._project_coarse(*projected_input, carry)
        if self.stage.post_transition_ffn is not None:
            output = self.stage.post_transition_ffn(*output)
        return output, descriptor


__all__ = ["MultiPoleExcitationStage"]
