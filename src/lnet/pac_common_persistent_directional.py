"""Common excitation with persistent directional memory across scales."""

from __future__ import annotations

# pyright: reportPrivateUsage=false
# ruff: noqa: EM101, SLF001, TRY003
from typing import TYPE_CHECKING, cast

from torch import Tensor, nn

from .pac_modewise_path_collapse import (
    ModeWiseComplexLinearCollapse,
    PhaseGatedModePathResidualModeWiseCollapse,
)
from .pac_phase_gated_transition import PhaseGatedS2DPostFusionTransition
from .pac_product_scan_pipeline import run_product_scan_pipeline

if TYPE_CHECKING:
    from .complex_scan_stage import ComplexScanStage
    from .pac_phase_gated_cffn import PhaseGatedComplexFFN

ComplexField = tuple[Tensor, Tensor]
CommonPersistentState = tuple[Tensor, Tensor, Tensor, Tensor]


class CommonPersistentTransition(nn.Module):
    """Update directional memory, then read one shared excitation from it."""

    def __init__(
        self,
        mixer: PhaseGatedModePathResidualModeWiseCollapse,
        augmented: PhaseGatedS2DPostFusionTransition,
    ) -> None:
        super().__init__()
        self.modes = mixer.modes
        self.mode: PhaseGatedComplexFFN = mixer.mode
        self.path: PhaseGatedComplexFFN = mixer.path
        self.readout: ModeWiseComplexLinearCollapse = mixer.collapse
        self.post: PhaseGatedComplexFFN = augmented.post
        self.carry_weight = augmented.carry_weight

    def _s2d(self, real: Tensor, imag: Tensor) -> ComplexField:
        """Apply one shared mode-wise 2x2 operator, preserving D4 when present."""
        if real.shape != imag.shape or real.shape[-1] != self.modes:
            raise ValueError("common/persistent S2D requires matching mode states")
        if real.shape[1] % 2 or real.shape[2] % 2:
            raise ValueError("common/persistent S2D requires even spatial dimensions")
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
            batch, height, width, directions, modes = real.shape
            shape = (batch, height // 2, 2, width // 2, 2, directions, modes)
            cells_real = (
                real.reshape(shape)
                .permute(0, 1, 3, 5, 2, 4, 6)
                .reshape(batch, height // 2, width // 2, directions, 4, modes)
            )
            cells_imag = (
                imag.reshape(shape)
                .permute(0, 1, 3, 5, 2, 4, 6)
                .reshape(batch, height // 2, width // 2, directions, 4, modes)
            )
        else:
            raise ValueError("common/persistent S2D requires NHWM or NHW4M inputs")
        weight = self.carry_weight.transpose(0, 1).to(real.dtype)
        return (
            (cells_real * weight).sum(dim=-2),
            (cells_imag * weight).sum(dim=-2),
        )

    def _update_memory(
        self,
        fresh_real: Tensor,
        fresh_imag: Tensor,
        previous_real: Tensor | None,
        previous_imag: Tensor | None,
    ) -> ComplexField:
        if fresh_real.shape != fresh_imag.shape or tuple(fresh_real.shape[-2:]) != (
            4,
            self.modes,
        ):
            raise ValueError("fresh directional memory must use NHW4M coordinates")
        if (previous_real is None) != (previous_imag is None):
            raise ValueError("persistent directional memory must provide both coordinates")
        update_real, update_imag = fresh_real, fresh_imag
        if previous_real is not None and previous_imag is not None:
            carried_real, carried_imag = self._s2d(previous_real, previous_imag)
            if carried_real.shape != fresh_real.shape:
                raise ValueError("persistent memory S2D shape does not match the fresh scan")
            update_real = update_real + carried_real
            update_imag = update_imag + carried_imag
        mode_real, mode_imag = self.mode(update_real, update_imag)
        path_real, path_imag = self.path(
            mode_real.transpose(-2, -1),
            mode_imag.transpose(-2, -1),
        )
        return path_real.transpose(-2, -1), path_imag.transpose(-2, -1)

    def _read_common(self, memory_real: Tensor, memory_imag: Tensor) -> ComplexField:
        read_real, read_imag = self.readout(
            memory_real.transpose(-2, -1),
            memory_imag.transpose(-2, -1),
        )
        return read_real.squeeze(-1), read_imag.squeeze(-1)

    def forward(
        self,
        fresh_real: Tensor,
        fresh_imag: Tensor,
        excitation_real: Tensor,
        excitation_imag: Tensor,
        previous_real: Tensor | None = None,
        previous_imag: Tensor | None = None,
    ) -> CommonPersistentState:
        memory_real, memory_imag = self._update_memory(
            fresh_real,
            fresh_imag,
            previous_real,
            previous_imag,
        )
        common_real, common_imag = self._read_common(memory_real, memory_imag)
        carry_real, carry_imag = self._s2d(excitation_real, excitation_imag)
        excitation_next = self.post(common_real + carry_real, common_imag + carry_imag)
        return *excitation_next, memory_real, memory_imag


def _take_transition(stage: ComplexScanStage) -> CommonPersistentTransition:
    mixer = stage.quadrant_path_mode_combiner
    augmented = stage.augmented
    if not isinstance(mixer, PhaseGatedModePathResidualModeWiseCollapse) or not isinstance(
        augmented,
        PhaseGatedS2DPostFusionTransition,
    ):
        raise TypeError(
            "common/persistent stages require Phase-Gated path mixing and post-fusion"
        )
    stage.quadrant_path_mode_combiner = None
    stage.augmented = None
    return CommonPersistentTransition(mixer, augmented)


class _FreshD4Stage(nn.Module):
    def __init__(self, pole_stage: ComplexScanStage) -> None:
        super().__init__()
        self.pole_stage = pole_stage
        self.transition = _take_transition(pole_stage)
        self.modes = pole_stage.modes
        self.output_modes = pole_stage.output_modes

    def _fresh_scan(
        self,
        real: Tensor,
        imag: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if real.shape != imag.shape or real.ndim != 4 or real.shape[-1] != self.modes:
            raise ValueError("common excitation must use matching NHWM coordinates")
        shape = cast("tuple[int, int, int, int]", tuple(real.shape))
        pole_x, pole_y = self.pole_stage._pole_coefficients(shape)
        return cast(
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


class InitialCommonPersistentStage(_FreshD4Stage):
    """Create the first persistent D4 memory from one common excitation."""

    def forward(self, real: Tensor, imag: Tensor) -> tuple[CommonPersistentState, Tensor]:
        fresh_real, fresh_imag, descriptor = self._fresh_scan(real, imag)
        state = self.transition(fresh_real, fresh_imag, real, imag)
        return state, descriptor


class CommonPersistentStage(_FreshD4Stage):
    """Refresh D4 from common E while carrying the previous D4 side-memory."""

    def forward(
        self,
        excitation_real: Tensor,
        excitation_imag: Tensor,
        memory_real: Tensor,
        memory_imag: Tensor,
    ) -> tuple[CommonPersistentState, Tensor]:
        fresh_real, fresh_imag, descriptor = self._fresh_scan(
            excitation_real,
            excitation_imag,
        )
        state = self.transition(
            fresh_real,
            fresh_imag,
            excitation_real,
            excitation_imag,
            memory_real,
            memory_imag,
        )
        return state, descriptor


class CommonExcitationTerminal(nn.Module):
    """Keep terminal Q tied to four fresh measurements of the shared excitation."""

    def __init__(self, pole_stage: ComplexScanStage) -> None:
        super().__init__()
        self.pole_stage = pole_stage
        self.modes = pole_stage.modes
        self.output_modes = None

    def forward(
        self,
        excitation_real: Tensor,
        excitation_imag: Tensor,
        memory_real: Tensor,
        memory_imag: Tensor,
    ) -> tuple[None, Tensor]:
        if (
            memory_real.shape != memory_imag.shape
            or memory_real.ndim != 5
            or tuple(memory_real.shape[-2:]) != (4, self.modes)
            or memory_real.shape[:3] != excitation_real.shape[:3]
        ):
            raise ValueError("terminal persistent memory must align with common excitation")
        return self.pole_stage(excitation_real, excitation_imag)


__all__ = [
    "CommonExcitationTerminal",
    "CommonPersistentStage",
    "CommonPersistentState",
    "CommonPersistentTransition",
    "InitialCommonPersistentStage",
]
