from __future__ import annotations

from typing import TYPE_CHECKING, assert_never

import torch
from torch import Tensor, nn

from .laplace import LaplaceParameterError, LaplaceShapeError
from .pac_hybrid_backend import HybridBackend, is_pac_lite_fused_candidate, selected_hybrid_backend
from .pac_prl_branch import PACControlledTappedPRLBranch
from .pac_prl_branch import stable_expm1_over_p as _stable_expm1_over_p
from .pac_triton_lite_fused import pac_lite_prl_fused_output

if TYPE_CHECKING:
    from .pac_recurrence import RecurrenceBackend
    from .pac_types import PACBranchName, PACFusion

BRANCH_ORDER: tuple[str, ...] = ("prl", "fir", "mlp")
stable_expm1_over_p = _stable_expm1_over_p

__all__ = [
    "PACControlledTappedPRLBranch",
    "PACHybridPRLBlock",
    "PACSequenceClassifier",
    "stable_expm1_over_p",
]


class PACHybridPRLBlock(nn.Module):
    def __init__(
        self,
        *,
        raw_input_dim: int,
        model_dim: int,
        output_dim: int,
        modes: int,
        tap_kernel_size: int,
        fir_kernel_size: int,
        use_mlp_branch: bool = True,
        active_branches: tuple[PACBranchName, ...] | None = None,
        fusion: PACFusion = "learned_scalar_sum",
        damping_control_range: float = 1.0,
        recurrence_backend: RecurrenceBackend = "auto",
        hybrid_backend: HybridBackend = "auto",
    ) -> None:
        super().__init__()
        self.raw_input_dim = raw_input_dim
        self.model_dim = model_dim
        self.output_dim = output_dim
        self.fusion = fusion
        self.hybrid_backend: HybridBackend = hybrid_backend
        self.fir_kernel_size = fir_kernel_size
        default_branches = ("prl", "fir", "mlp") if use_mlp_branch else ("prl", "fir")
        self.active_branches = active_branches or default_branches
        self.input_projection = nn.Linear(raw_input_dim, model_dim)
        self.prl_branch = (
            PACControlledTappedPRLBranch(
                model_dim=model_dim,
                modes=modes,
                tap_kernel_size=tap_kernel_size,
                damping_control_range=damping_control_range,
                recurrence_backend=recurrence_backend,
            )
            if "prl" in self.active_branches
            else None
        )
        self.fir_depthwise = (
            nn.Conv1d(
                model_dim,
                model_dim,
                fir_kernel_size,
                groups=model_dim,
                padding=fir_kernel_size - 1,
            )
            if "fir" in self.active_branches
            else None
        )
        self.fir_pointwise = (
            nn.Linear(model_dim, model_dim) if "fir" in self.active_branches else None
        )
        self.mlp_branch = (
            nn.Sequential(
                nn.Linear(model_dim, model_dim * 2), nn.GELU(), nn.Linear(model_dim * 2, model_dim)
            )
            if "mlp" in self.active_branches
            else None
        )
        self.branch_scales = nn.Parameter(torch.tensor([1.0, 1.0, 0.5], dtype=torch.float32))
        self.branch_gate = _branch_gate(fusion, model_dim)
        self.residual_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.output_projection = nn.Linear(model_dim, model_dim)
        self.readout_projection = nn.Linear(model_dim, output_dim)
        self.activation = nn.GELU()

    def branch_scale_values(self) -> Tensor:
        return self.branch_scales.detach().cpu()

    def branch_outputs(
        self, projected: Tensor, disabled: tuple[PACBranchName, ...] = ()
    ) -> dict[PACBranchName, Tensor]:
        return self._branch_outputs(projected, disabled)

    def require_prl_branch(self) -> PACControlledTappedPRLBranch:
        if self.prl_branch is None:
            message = "PRL branch is not initialized"
            raise RuntimeError(message)
        return self.prl_branch

    def forward(self, inputs: Tensor) -> Tensor:
        return self.forward_with_disabled(inputs, ())

    def forward_with_disabled(self, inputs: Tensor, disabled: tuple[PACBranchName, ...]) -> Tensor:
        _check_raw(inputs, self.raw_input_dim)
        projected = self.input_projection(inputs)
        selected = self._selected_hybrid_backend(projected, disabled)
        if selected in {"pac_lite_prl_fused", "pac_lite_block_fused"}:
            return self._forward_pac_lite_fused(projected)
        fused = self._fused(projected, disabled)
        residual = projected + self.residual_scale * self.activation(self.output_projection(fused))
        return self.readout_projection(residual)

    def _selected_hybrid_backend(
        self, projected: Tensor, disabled: tuple[PACBranchName, ...]
    ) -> HybridBackend:
        if disabled:
            return "generic"
        prl_branch = self.prl_branch
        is_candidate = is_pac_lite_fused_candidate(
            active_branches=tuple(self.active_branches),
            fusion=self.fusion,
            has_gate=self.branch_gate is not None,
            inputs=projected,
            model_dim=self.model_dim,
            modes=prl_branch.modes if prl_branch is not None else 0,
            tap_kernel_size=prl_branch.tap_kernel_size if prl_branch is not None else 0,
            fir_kernel_size=self.fir_kernel_size,
        )
        return selected_hybrid_backend(self.hybrid_backend, is_candidate=is_candidate)

    def _forward_pac_lite_fused(self, projected: Tensor) -> Tensor:
        prl = self.require_prl_branch()
        scales = self.branch_scales.to(device=projected.device, dtype=projected.dtype)
        fused = pac_lite_prl_fused_output(prl, projected) * scales[0]
        if "fir" in self.active_branches:
            fused = fused + self._fir_output(projected) * scales[1]
        residual = projected + self.residual_scale * self.activation(self.output_projection(fused))
        return self.readout_projection(residual)

    def _branch_outputs(
        self, projected: Tensor, disabled: tuple[PACBranchName, ...]
    ) -> dict[PACBranchName, Tensor]:
        branches: dict[PACBranchName, Tensor] = {}
        if "prl" in self.active_branches and "prl" not in disabled:
            branches["prl"] = self._prl_output(projected)
        if "fir" in self.active_branches and "fir" not in disabled:
            branches["fir"] = self._fir_output(projected)
        if "mlp" in self.active_branches and "mlp" not in disabled:
            branches["mlp"] = self._mlp_output(projected)
        return branches

    def _fused(self, projected: Tensor, disabled: tuple[PACBranchName, ...]) -> Tensor:
        if self._uses_scalar_fast_path(disabled):
            return self._fused_scalar_fast(projected)
        branches = self._branch_outputs(projected, disabled)
        if not branches:
            return torch.zeros_like(projected)
        branch_names = tuple(branches)
        mask = _active_mask(branch_names, (), projected.device, projected.dtype)
        if self.fusion == "sum":
            weights = mask
        elif self.fusion == "learned_scalar_sum":
            weights = mask * self.branch_scales.to(device=projected.device, dtype=projected.dtype)
        elif self.fusion in {"softmax", "temperature_softmax", "sigmoid_gates"}:
            weights = self._gated_weights(projected, mask)
        else:
            raise LaplaceParameterError(reason=f"unsupported fusion: {self.fusion}")
        fused = torch.zeros_like(projected)
        for branch, output in branches.items():
            fused = fused + output * weights[_branch_index(branch)]
        return fused

    def _uses_scalar_fast_path(self, disabled: tuple[PACBranchName, ...]) -> bool:
        return (
            self.fusion == "learned_scalar_sum"
            and not disabled
            and self.branch_gate is None
            and "mlp" not in self.active_branches
        )

    def _fused_scalar_fast(self, projected: Tensor) -> Tensor:
        scales = self.branch_scales.to(device=projected.device, dtype=projected.dtype)
        fused = torch.zeros_like(projected)
        if "prl" in self.active_branches:
            fused = fused + self._prl_output(projected) * scales[0]
        if "fir" in self.active_branches:
            fused = fused + self._fir_output(projected) * scales[1]
        return fused

    def _prl_output(self, projected: Tensor) -> Tensor:
        return self.require_prl_branch()(projected)

    def _fir_output(self, projected: Tensor) -> Tensor:
        if self.fir_depthwise is None or self.fir_pointwise is None:
            message = "FIR branch is not initialized"
            raise RuntimeError(message)
        convolved = self.fir_depthwise(projected.transpose(1, 2))[..., : projected.shape[1]]
        return self.fir_pointwise(convolved.transpose(1, 2))

    def _mlp_output(self, projected: Tensor) -> Tensor:
        if self.mlp_branch is None:
            message = "MLP branch is not initialized"
            raise RuntimeError(message)
        return self.mlp_branch(projected)

    def _gated_weights(self, projected: Tensor, mask: Tensor) -> Tensor:
        if self.branch_gate is None:
            message = "branch gate is not initialized"
            raise RuntimeError(message)
        logits = self.branch_gate(projected)
        if self.fusion == "sigmoid_gates":
            return torch.sigmoid(logits).mean(dim=(0, 1)) * mask
        divisor = 0.5 if self.fusion == "temperature_softmax" else 1.0
        masked = torch.where(
            mask.view(1, 1, -1) > 0, logits / divisor, torch.full_like(logits, -torch.inf)
        )
        return torch.softmax(masked, dim=-1).mean(dim=(0, 1))


class PACSequenceClassifier(nn.Module):
    def __init__(self, *, raw_input_dim: int, model_dim: int, modes: int, class_count: int) -> None:
        super().__init__()
        self.encoder = PACHybridPRLBlock(
            raw_input_dim=raw_input_dim,
            model_dim=model_dim,
            output_dim=model_dim,
            modes=modes,
            tap_kernel_size=8,
            fir_kernel_size=5,
        )
        self.classifier = nn.Linear(model_dim, class_count)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.classifier(self.encoder(inputs).mean(dim=1))


def _branch_gate(fusion: PACFusion, model_dim: int) -> nn.Linear | None:
    if fusion in {"softmax", "temperature_softmax", "sigmoid_gates"}:
        return nn.Linear(model_dim, len(BRANCH_ORDER))
    return None


def _branch_index(branch: PACBranchName) -> int:
    match branch:
        case "prl":
            return 0
        case "fir":
            return 1
        case "mlp":
            return 2
        case unreachable:
            assert_never(unreachable)


def _active_mask(
    active: tuple[PACBranchName, ...],
    disabled: tuple[PACBranchName, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    return torch.tensor(
        [branch in active and branch not in disabled for branch in BRANCH_ORDER],
        device=device,
        dtype=dtype,
    )


def _check_raw(inputs: Tensor, raw_input_dim: int) -> None:
    if inputs.ndim != 3 or inputs.shape[-1] != raw_input_dim:
        raise LaplaceShapeError(tuple(inputs.shape), 3, raw_input_dim)
