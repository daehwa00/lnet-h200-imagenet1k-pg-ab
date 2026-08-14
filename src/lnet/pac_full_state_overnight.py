"""Declarative full-cell transition family for the overnight campaign."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from .pac_complex_layers import ComplexLinear, WidelyLinear
from .pac_full_state_operators import (
    GroupedComplexLinear,
    GroupedPhaseGatedComplexFFN,
    GroupedWidelyLinearResidual,
)
from .pac_full_state_transition import (
    Pole,
    direction_relative_pointwise_gains,
    raw_to_pole_aligned_innovations,
)
from .pac_grouped_path_cffn import GroupedWidelyLinear
from .pac_path_cffn import D4PathModeCombiner
from .pac_phase_gated_cffn import PhaseGatedComplexFFN

ComplexField = tuple[Tensor, Tensor]
Basis = Literal["raw", "innovation_detached"]
AxialOperator = Literal["pg", "gwl"]
AxialSharing = Literal["mode_specific", "shared"]
OperationOrder = Literal["axial_compress_mode", "mode_axial_compress"]
Collapse = Literal["gwl", "complex", "rank2_joint_complex", "rank2_joint_wl", "full_joint_wl"]
PostType = Literal["base", "phase_gated"]

BASE_MODEL = "Pgv2-H192-All3e-3"
DIRECTION_ORDERING = "direction_relative_q00_q10_q01_q11"


@dataclass(frozen=True, slots=True)
class FullStateExperimentSpec:
    """One orthogonal architecture choice in the full-state campaign."""

    code: str
    variant: str
    basis: Basis
    state_count: int
    axial_operator: AxialOperator = "pg"
    axial_weight_sharing: AxialSharing = "mode_specific"
    operation_order: OperationOrder = "axial_compress_mode"
    collapse: Collapse = "gwl"
    state_pg: bool = False
    stage_mask: tuple[bool, bool, bool] = (True, True, True)
    post_type: PostType = "base"

    def signature(self) -> dict[str, object]:
        operator = "PGv2_4x8x4" if self.axial_operator == "pg" else "GWL_4x8x4"
        collapse_names = {
            "gwl": f"GWL_{self.state_count}x1",
            "complex": f"CL_{self.state_count}x1_bias_free",
            "rank2_joint_complex": "per_mode_CL_8x2_then_joint_CL_192x96",
            "rank2_joint_wl": "per_mode_CL_8x2_then_joint_WL_192x96",
            "full_joint_wl": "joint_WL_768x96",
        }
        return {
            "signature_schema": "lnet.full_state_transition.v2",
            "base_model": BASE_MODEL,
            "base_contract": "Pgv2-H192-All3e-3.v1",
            "coarsen_type": "full_2x2",
            "basis": self.basis,
            "basis_normalization": "post_pointwise_gain_effective_decay",
            "pole_gradient_policy": "detached_basis",
            "direction_ordering": DIRECTION_ORDERING,
            "state_count": self.state_count,
            "local_operator": operator,
            "local_hidden": 8,
            "direction_operator": operator,
            "direction_hidden": 8,
            "axial_weight_sharing": self.axial_weight_sharing,
            "operation_order": self.operation_order,
            "mode_operator": "PGv2_96x192x96",
            "mode_hidden": 192,
            "mode_state_initialization": "exact_stagewise_base_copy",
            "state_operator": "PGv2_8x16x8" if self.state_pg else "none",
            "collapse_type": collapse_names[self.collapse],
            "collapse_bias": self.collapse in {"gwl", "rank2_joint_wl", "full_joint_wl"},
            "joint_rank": 2 if self.collapse.startswith("rank2") else None,
            "stage_mask": [int(value) for value in self.stage_mask],
            "post_type": "PGv2_H192" if self.post_type == "phase_gated" else "base_WL",
        }

    def signature_hash(self) -> str:
        encoded = json.dumps(self.signature(), separators=(",", ":"), sort_keys=True).encode()
        return hashlib.sha256(encoded).hexdigest()


EXPERIMENT_SPECS = (
    FullStateExperimentSpec("E01", "FS-I4-AxPG", "innovation_detached", 4),
    FullStateExperimentSpec("E02", "FS-I8-AxPG", "innovation_detached", 8),
    FullStateExperimentSpec("E05", "FS-R8-AxPG", "raw", 8),
    FullStateExperimentSpec("E03", "FS-I12-AxPG", "innovation_detached", 12),
    FullStateExperimentSpec(
        "E07",
        "FS-I8-AxPG-Shared",
        "innovation_detached",
        8,
        axial_weight_sharing="shared",
    ),
    FullStateExperimentSpec(
        "E06",
        "FS-I8-AxGWL",
        "innovation_detached",
        8,
        axial_operator="gwl",
    ),
    FullStateExperimentSpec(
        "E10",
        "FS-I8-CollapseCL",
        "innovation_detached",
        8,
        collapse="complex",
    ),
    FullStateExperimentSpec(
        "E11",
        "FS-I8-Rank2JointCL",
        "innovation_detached",
        8,
        collapse="rank2_joint_complex",
    ),
    FullStateExperimentSpec(
        "E12",
        "FS-I8-Rank2JointWL",
        "innovation_detached",
        8,
        collapse="rank2_joint_wl",
    ),
    FullStateExperimentSpec(
        "E09",
        "FS-I8-StatePG",
        "innovation_detached",
        8,
        state_pg=True,
    ),
    FullStateExperimentSpec(
        "E14",
        "FS-I8-Stage1Only",
        "innovation_detached",
        8,
        stage_mask=(True, False, False),
    ),
    FullStateExperimentSpec(
        "E15",
        "FS-I8-PGPost",
        "innovation_detached",
        8,
        post_type="phase_gated",
    ),
    FullStateExperimentSpec(
        "E08",
        "FS-I8-ModeFirst",
        "innovation_detached",
        8,
        operation_order="mode_axial_compress",
    ),
    FullStateExperimentSpec("E04", "FS-I16-AxPG", "innovation_detached", 16),
    FullStateExperimentSpec(
        "E13",
        "FS-I8-FullJointWL",
        "innovation_detached",
        8,
        collapse="full_joint_wl",
    ),
)
SPECS_BY_VARIANT = {spec.variant: spec for spec in EXPERIMENT_SPECS}


def experiment_manifest() -> list[dict[str, object]]:
    """Return the ordered, JSON-safe architecture manifest."""
    return [
        {
            **asdict(spec),
            "signature": spec.signature(),
            "signature_sha256": spec.signature_hash(),
        }
        for spec in EXPERIMENT_SPECS
    ]


def select_novel_variants(
    known_signatures: set[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split the campaign by canonical signature rather than run name."""
    selected = []
    skipped = []
    observed = set(known_signatures)
    for spec in EXPERIMENT_SPECS:
        digest = spec.signature_hash()
        target = skipped if digest in observed else selected
        target.append(spec.variant)
        observed.add(digest)
    return tuple(selected), tuple(skipped)


class StructuredFullStateTransition(D4PathModeCombiner):
    """Axially process, compress, and canonicalize one full 2x2 D4 cell."""

    collapses_product_paths = True
    requires_full_product_cells = True

    def __init__(
        self,
        modes: int,
        spec: FullStateExperimentSpec,
        *,
        mode_hidden: int = 192,
        gain_normalization: Literal["pointwise", "global"] = "pointwise",
    ) -> None:
        super().__init__()
        if modes <= 0 or mode_hidden <= 0:
            message = "structured full-state dimensions must be positive"
            raise ValueError(message)
        self.modes = modes
        self.spec = spec
        self.path_count = 16
        self.output_paths = 1
        self.input_modes = 16 * modes
        self.gain_normalization: Literal["pointwise", "global"] = gain_normalization
        self.local = self._make_axial_operator()
        self.direction = self._make_axial_operator()
        self.compression = self._make_compression()
        self.mode = PhaseGatedComplexFFN(modes, mode_hidden)
        self.state = (
            PhaseGatedComplexFFN(spec.state_count, 2 * spec.state_count) if spec.state_pg else None
        )
        self.collapse = self._make_collapse()
        self.raw_rms: Tensor
        self.pre_normalization_rms: Tensor
        self.normalized_scan_rms: Tensor
        self.finite_gain_max: Tensor
        self.memory_gram: Tensor
        self.local_gram: Tensor
        self.direction_gram: Tensor
        self.basis_energy: Tensor
        self.register_buffer("raw_rms", torch.zeros(()), persistent=False)
        self.register_buffer("pre_normalization_rms", torch.zeros(()), persistent=False)
        self.register_buffer("normalized_scan_rms", torch.zeros(()), persistent=False)
        self.register_buffer("finite_gain_max", torch.zeros(()), persistent=False)
        self.register_buffer(
            "memory_gram", torch.zeros(16, 16, dtype=torch.complex64), persistent=False
        )
        self.register_buffer(
            "local_gram", torch.zeros(4, 4, dtype=torch.complex64), persistent=False
        )
        self.register_buffer(
            "direction_gram",
            torch.zeros(4, 4, dtype=torch.complex64),
            persistent=False,
        )
        self.register_buffer("basis_energy", torch.zeros(4), persistent=False)
        self.diagnostics_enabled = False

    def set_diagnostics_enabled(self, *, enabled: bool) -> None:
        """Enable sampled statistics only for an explicit eager diagnostic pass."""
        self.diagnostics_enabled = bool(enabled)
        for operator in (self.local, self.direction):
            if isinstance(operator, GroupedPhaseGatedComplexFFN):
                operator.diagnostics_enabled = bool(enabled)

    def _make_axial_operator(self) -> nn.Module:
        if self.spec.axial_operator == "gwl":
            return GroupedWidelyLinearResidual(self.modes, 4, 8)
        if self.spec.axial_weight_sharing == "shared":
            return PhaseGatedComplexFFN(4, 8)
        return GroupedPhaseGatedComplexFFN(self.modes, 4, 8)

    def _make_compression(self) -> nn.Module | None:
        if self.spec.state_count == 16:
            return None
        if self.spec.basis == "raw":
            return GroupedComplexLinear(self.modes, 4, self.spec.state_count // 4)
        if self.spec.state_count == 4:
            return GroupedComplexLinear(self.modes, 4, 1)
        return GroupedComplexLinear(self.modes, 3, self.spec.state_count // 4 - 1)

    def _make_collapse(self) -> nn.Module:
        states = self.spec.state_count
        if self.spec.collapse == "gwl":
            return GroupedWidelyLinear(self.modes, states, 1, bias=True)
        if self.spec.collapse == "complex":
            return GroupedComplexLinear(self.modes, states, 1)
        if self.spec.collapse == "rank2_joint_complex":
            return nn.ModuleDict(
                {
                    "reduce": GroupedComplexLinear(self.modes, states, 2),
                    "joint": ComplexLinear(2 * self.modes, self.modes),
                }
            )
        if self.spec.collapse == "rank2_joint_wl":
            return nn.ModuleDict(
                {
                    "reduce": GroupedComplexLinear(self.modes, states, 2),
                    "joint": WidelyLinear(2 * self.modes, self.modes, bias=True),
                }
            )
        return WidelyLinear(states * self.modes, self.modes, bias=True)

    def copy_mode_from(self, baseline: PhaseGatedComplexFFN) -> None:
        """Copy the unchanged Base mode-PG parameters exactly."""
        if baseline.modes != self.modes or baseline.hidden_modes != self.mode.hidden_modes:
            message = "cannot copy a mismatched Base mode PG block"
            raise ValueError(message)
        self.mode.load_state_dict(baseline.state_dict())

    @staticmethod
    def _apply_axis(
        operator: nn.Module,
        real: Tensor,
        imag: Tensor,
        *,
        direction_axis: bool,
    ) -> ComplexField:
        active_real = real.transpose(-3, -2) if direction_axis else real
        active_imag = imag.transpose(-3, -2) if direction_axis else imag
        if isinstance(operator, PhaseGatedComplexFFN):
            output = operator(active_real.transpose(-2, -1), active_imag.transpose(-2, -1))
            output = output[0].transpose(-2, -1), output[1].transpose(-2, -1)
        else:
            output = operator(active_real, active_imag)
        if direction_axis:
            return output[0].transpose(-3, -2), output[1].transpose(-3, -2)
        return output

    def _compress(self, real: Tensor, imag: Tensor) -> ComplexField:
        if self.compression is None:
            return real.flatten(-3, -2), imag.flatten(-3, -2)
        if not isinstance(self.compression, GroupedComplexLinear):
            message = "full-state compression has an invalid operator"
            raise TypeError(message)
        if self.spec.basis == "innovation_detached" and self.spec.state_count in {8, 12}:
            detail_real, detail_imag = self.compression(real[..., 1:, :], imag[..., 1:, :])
            real = torch.cat((real[..., :1, :], detail_real), dim=-2)
            imag = torch.cat((imag[..., :1, :], detail_imag), dim=-2)
        else:
            real, imag = self.compression(real, imag)
        return real.flatten(-3, -2), imag.flatten(-3, -2)

    def _collapse(self, real: Tensor, imag: Tensor) -> ComplexField:
        if isinstance(self.collapse, (GroupedWidelyLinear, GroupedComplexLinear)):
            return self.collapse(real, imag)
        if isinstance(self.collapse, nn.ModuleDict):
            reducer = self.collapse["reduce"]
            joint = self.collapse["joint"]
            if not isinstance(reducer, GroupedComplexLinear):
                message = "rank-2 collapse lost its grouped reducer"
                raise TypeError(message)
            reduced_real, reduced_imag = reducer(real, imag)
            leading = reduced_real.shape[:-2]
            joint_real, joint_imag = joint(
                reduced_real.reshape(*leading, 2 * self.modes),
                reduced_imag.reshape(*leading, 2 * self.modes),
            )
            return joint_real.unsqueeze(-2), joint_imag.unsqueeze(-2)
        leading = real.shape[:-2]
        output_real, output_imag = self.collapse(
            real.reshape(*leading, self.spec.state_count * self.modes),
            imag.reshape(*leading, self.spec.state_count * self.modes),
        )
        return output_real.unsqueeze(-2), output_imag.unsqueeze(-2)

    @torch.no_grad()
    def _update_memory_diagnostics(
        self,
        raw_real: Tensor,
        raw_imag: Tensor,
        basis_real: Tensor,
        basis_imag: Tensor,
        *,
        pole_x: Pole,
        pole_y: Pole,
    ) -> None:
        rows = raw_real.reshape(-1, 16, self.modes)[:32]
        rows_real = rows.permute(0, 2, 1).reshape(-1, 16).float()
        rows_imag = (
            raw_imag.reshape(-1, 16, self.modes)[:32].permute(0, 2, 1).reshape(-1, 16).float()
        )
        rows_real = rows_real - rows_real.mean(dim=0, keepdim=True)
        rows_imag = rows_imag - rows_imag.mean(dim=0, keepdim=True)
        rows_complex = torch.complex(rows_real, rows_imag)
        count = max(rows_complex.shape[0], 1)
        self.memory_gram.copy_((rows_complex.mH @ rows_complex) / count)
        local_real = raw_real.reshape(-1, 4, 4, self.modes)[:32]
        local_imag = raw_imag.reshape(-1, 4, 4, self.modes)[:32]
        local_matrix_real = local_real.permute(0, 1, 3, 2).reshape(-1, 4).float()
        local_matrix_imag = local_imag.permute(0, 1, 3, 2).reshape(-1, 4).float()
        local_matrix_real = local_matrix_real - local_matrix_real.mean(dim=0, keepdim=True)
        local_matrix_imag = local_matrix_imag - local_matrix_imag.mean(dim=0, keepdim=True)
        local_complex = torch.complex(local_matrix_real, local_matrix_imag)
        self.local_gram.copy_((local_complex.mH @ local_complex) / max(local_complex.shape[0], 1))
        direction_matrix_real = local_real.permute(0, 2, 3, 1).reshape(-1, 4).float()
        direction_matrix_imag = local_imag.permute(0, 2, 3, 1).reshape(-1, 4).float()
        direction_matrix_real = direction_matrix_real - direction_matrix_real.mean(
            dim=0,
            keepdim=True,
        )
        direction_matrix_imag = direction_matrix_imag - direction_matrix_imag.mean(
            dim=0,
            keepdim=True,
        )
        direction_complex = torch.complex(direction_matrix_real, direction_matrix_imag)
        self.direction_gram.copy_(
            (direction_complex.mH @ direction_complex) / max(direction_complex.shape[0], 1)
        )
        raw_energy = raw_real.float().square() + raw_imag.float().square()
        self.raw_rms.copy_(torch.sqrt(raw_energy.mean()))
        self.normalized_scan_rms.copy_(torch.sqrt(raw_energy.mean()))
        if self.gain_normalization == "pointwise":
            gains = direction_relative_pointwise_gains(
                raw_real,
                pole_x=pole_x,
                pole_y=pole_y,
            ).float()
            pre_energy = (raw_real.float() * gains).square() + (raw_imag.float() * gains).square()
            self.pre_normalization_rms.copy_(torch.sqrt(pre_energy.mean()))
            self.finite_gain_max.copy_(gains.max())
        else:
            self.pre_normalization_rms.fill_(float("nan"))
            self.finite_gain_max.fill_(float("nan"))
        basis_energy = basis_real.float().square() + basis_imag.float().square()
        self.basis_energy.copy_(basis_energy.mean(dim=(0, 1, 2, 3, 5)))

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        del real, imag
        message = "structured full-state transitions require direction-relative cells and poles"
        raise RuntimeError(message)

    def forward_packed(self, source_real: Tensor, source_imag: Tensor) -> ComplexField:
        del source_real, source_imag
        message = "structured full-state transitions cannot discard the local-cell axis"
        raise RuntimeError(message)

    def forward_full_state(
        self,
        source_real: Tensor,
        source_imag: Tensor,
        *,
        pole_x: Pole,
        pole_y: Pole,
    ) -> ComplexField:
        if (
            source_real.shape != source_imag.shape
            or source_real.ndim != 6
            or tuple(source_real.shape[-3:]) != (4, 4, self.modes)
        ):
            message = "structured full-state transition requires NHW-direction-local-mode inputs"
            raise ValueError(message)
        active_real, active_imag = source_real, source_imag
        if self.spec.basis == "innovation_detached":
            active_real, active_imag = raw_to_pole_aligned_innovations(
                active_real,
                active_imag,
                pole_x=pole_x,
                pole_y=pole_y,
                gain_normalization=self.gain_normalization,
            )
        if self.diagnostics_enabled:
            self._update_memory_diagnostics(
                source_real,
                source_imag,
                active_real,
                active_imag,
                pole_x=pole_x,
                pole_y=pole_y,
            )
        if self.spec.operation_order == "mode_axial_compress":
            leading = active_real.shape[:-3]
            active_real, active_imag = self.mode(
                active_real.reshape(*leading, 16, self.modes),
                active_imag.reshape(*leading, 16, self.modes),
            )
            active_real = active_real.reshape(*leading, 4, 4, self.modes)
            active_imag = active_imag.reshape(*leading, 4, 4, self.modes)
        active_real, active_imag = self._apply_axis(
            self.local,
            active_real,
            active_imag,
            direction_axis=False,
        )
        active_real, active_imag = self._apply_axis(
            self.direction,
            active_real,
            active_imag,
            direction_axis=True,
        )
        active_real, active_imag = self._compress(active_real, active_imag)
        if self.spec.operation_order == "axial_compress_mode":
            active_real, active_imag = self.mode(active_real, active_imag)
        if self.state is not None:
            state_real, state_imag = self.state(
                active_real.transpose(-2, -1),
                active_imag.transpose(-2, -1),
            )
            active_real = state_real.transpose(-2, -1)
            active_imag = state_imag.transpose(-2, -1)
        return self._collapse(active_real, active_imag)

    @staticmethod
    def _correlation(gram: Tensor) -> Tensor:
        diagonal = gram.diag().real.clamp_min(1.0e-12).sqrt()
        return gram / (diagonal[:, None] * diagonal[None, :])

    def diagnostic_metrics(self) -> dict[str, float]:
        gram = self.memory_gram.detach().cpu()
        eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0.0)
        probabilities = eigenvalues / eigenvalues.sum().clamp_min(1.0e-12)
        effective_rank = torch.exp(-(probabilities * probabilities.clamp_min(1.0e-12).log()).sum())
        metrics = {
            "memory/raw16_rms": float(self.raw_rms),
            "scan/pre_normalization_rms": float(self.pre_normalization_rms),
            "scan/normalized_rms": float(self.normalized_scan_rms),
            "scan/finite_gain_max": float(self.finite_gain_max),
            "memory/effective_rank": float(effective_rank),
        }
        local = self._correlation(self.local_gram.detach().cpu())
        direction = self._correlation(self.direction_gram.detach().cpu())
        for row in range(4):
            for column in range(4):
                metrics[f"memory/local_corr_{row}{column}"] = float(local[row, column].abs())
                metrics[f"memory/direction_corr_{row}{column}"] = float(
                    direction[row, column].abs()
                )
        names = (
            ("M", "Ix", "Iy", "Ixy")
            if self.spec.basis == "innovation_detached"
            else (
                "q00",
                "q10",
                "q01",
                "q11",
            )
        )
        energy = self.basis_energy.detach().float()
        total = energy.sum().clamp_min(1.0e-12)
        for name, value in zip(names, energy, strict=True):
            metrics[f"basis/{name}_rms"] = float(torch.sqrt(value))
            metrics[f"basis/{name}_energy_fraction"] = float(value / total)
        for axis, operator in (("local", self.local), ("direction", self.direction)):
            if isinstance(operator, (GroupedPhaseGatedComplexFFN, PhaseGatedComplexFFN)):
                metrics.update(
                    {
                        f"{axis}_pg/{name}": value
                        for name, value in operator.diagnostic_metrics().items()
                    }
                )
        metrics.update(
            {f"mode_pg/{name}": value for name, value in self.mode.diagnostic_metrics().items()}
        )
        if self.state is not None:
            metrics.update(
                {
                    f"state_pg/{name}": value
                    for name, value in self.state.diagnostic_metrics().items()
                }
            )
        return metrics


__all__ = [
    "BASE_MODEL",
    "EXPERIMENT_SPECS",
    "SPECS_BY_VARIANT",
    "FullStateExperimentSpec",
    "StructuredFullStateTransition",
    "experiment_manifest",
    "select_novel_variants",
]
