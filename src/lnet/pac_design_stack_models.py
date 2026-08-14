from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import torch
from torch import Tensor, nn
from torch.nn import functional

from .pac_prl_branch import PACControlledTappedPRLBranch

if TYPE_CHECKING:
    from .pac_types import PACExperimentConfig


PAC_DESIGN_MODELS: Final[tuple[str, ...]] = (
    "pac_design_depth4_concat_pyramid_evidence",
    "pac_design_depth6_concat_pyramid_evidence",
)


@dataclass(frozen=True, slots=True)
class PACDesignSpec:
    name: str
    block_kinds: tuple[str, ...]


_SPECS: Final[tuple[PACDesignSpec, ...]] = (
    PACDesignSpec("pac_design_depth4_concat_pyramid_evidence", ("lite", "full", "lite", "full")),
    PACDesignSpec(
        "pac_design_depth6_concat_pyramid_evidence",
        ("lite", "lite", "full", "lite", "lite", "full"),
    ),
)


class PACDesignStackClassifier(nn.Module):
    def __init__(self, config: PACExperimentConfig, class_count: int, spec: PACDesignSpec) -> None:
        super().__init__()
        self.model_dim = max(config.model_dim, 4 * config.modes)
        self.stem = _AmplitudeStem(config.raw_input_dim, self.model_dim)
        tap_size = min(config.tap_kernel_size, 5)
        self.blocks = nn.ModuleList(
            _DesignPACBlock(
                model_dim=self.model_dim,
                modes=config.modes,
                tap_kernel_size=tap_size,
                fir_kernel_size=config.fir_kernel_size,
                full_block=kind == "full",
            )
            for kind in spec.block_kinds
        )
        self.final_norm = _RMSNorm(self.model_dim)
        self.head = _PyramidEvidenceHead(self.model_dim, class_count)

    def forward(self, inputs: Tensor) -> Tensor:
        features = self.stem(inputs)
        for block in self.blocks:
            features = block(features)
        return self.head(self.final_norm(features))


class _DesignPACBlock(nn.Module):
    def __init__(
        self,
        *,
        model_dim: int,
        modes: int,
        tap_kernel_size: int,
        fir_kernel_size: int,
        full_block: bool,
    ) -> None:
        super().__init__()
        self.norm = _RMSNorm(model_dim)
        prl_dim = max(1, min(model_dim - 1, round(model_dim * 0.625)))
        self.prl = PACControlledTappedPRLBranch(
            model_dim=model_dim,
            modes=modes,
            tap_kernel_size=tap_kernel_size,
            damping_control_range=1.0,
            recurrence_backend="auto",
        )
        self.prl_project = nn.Linear(model_dim, prl_dim)
        self.fir = _FIRBranch(model_dim, model_dim - prl_dim, fir_kernel_size)
        self.fusion = nn.Linear(model_dim, model_dim)
        self.temporal_scale = nn.Parameter(torch.full((model_dim,), 1.0e-2))
        self.channel_norm = _RMSNorm(model_dim) if full_block else None
        self.channel_mlp = _SwiGLU(model_dim) if full_block else None
        self.channel_scale = nn.Parameter(torch.full((model_dim,), 1.0e-2)) if full_block else None
        _disable_prl_direct(self.prl)
        _init_quadrature_projection(self.prl, self.prl_project)

    def forward(self, inputs: Tensor) -> Tensor:
        normalized = self.norm(inputs)
        temporal = self.fusion(
            torch.cat((self.prl_project(self.prl(normalized)), self.fir(normalized)), dim=-1)
        )
        outputs = inputs + self.temporal_scale.view(1, 1, -1) * temporal
        if self.channel_norm is None or self.channel_mlp is None or self.channel_scale is None:
            return outputs
        mixed = self.channel_mlp(self.channel_norm(outputs))
        return outputs + self.channel_scale.view(1, 1, -1) * mixed


class _AmplitudeStem(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.depthwise = _CausalConv1d(input_dim, input_dim, 5, groups=input_dim)
        self.pointwise = nn.Linear(input_dim, output_dim)
        self.norm = _RMSNorm(output_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        return functional.silu(self.norm(self.pointwise(self.depthwise(inputs))))


class _FIRBranch(nn.Module):
    def __init__(self, model_dim: int, output_dim: int, kernel_size: int) -> None:
        super().__init__()
        self.depthwise = _CausalConv1d(model_dim, model_dim, kernel_size, groups=model_dim)
        self.pointwise = nn.Linear(model_dim, output_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.pointwise(self.depthwise(inputs))


class _CausalConv1d(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, kernel_size: int, groups: int = 1) -> None:
        super().__init__()
        self.left_pad = kernel_size - 1
        self.conv = nn.Conv1d(input_dim, output_dim, kernel_size, groups=groups)

    def forward(self, inputs: Tensor) -> Tensor:
        padded = functional.pad(inputs.transpose(1, 2), (self.left_pad, 0))
        return self.conv(padded).transpose(1, 2)


class _RMSNorm(nn.Module):
    def __init__(self, feature_dim: int, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(feature_dim))
        self.eps = eps

    def forward(self, inputs: Tensor) -> Tensor:
        scale = torch.rsqrt(inputs.square().mean(dim=-1, keepdim=True) + self.eps)
        return inputs * scale * self.weight


class _SwiGLU(nn.Module):
    def __init__(self, model_dim: int) -> None:
        super().__init__()
        hidden = 2 * model_dim
        self.input = nn.Linear(model_dim, 2 * hidden)
        self.output = nn.Linear(hidden, model_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        gate, value = self.input(inputs).chunk(2, dim=-1)
        return self.output(functional.silu(gate) * value)


class _PyramidEvidenceHead(nn.Module):
    def __init__(self, model_dim: int, class_count: int) -> None:
        super().__init__()
        input_dim = 14 * model_dim
        hidden = max(32, min(256, input_dim // 2))
        self.global_head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, class_count),
        )
        self.timestep_head = nn.Linear(model_dim, class_count)
        self.local_scale = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.temperature = 1.0

    def forward(self, sequence: Tensor) -> Tensor:
        global_logits = self.global_head(_temporal_pyramid(sequence))
        local_scores = self.timestep_head(sequence)
        local_logits = self.temperature * torch.logsumexp(local_scores / self.temperature, dim=1)
        return global_logits + self.local_scale * local_logits


def build_pac_design_classifier(
    name: str, config: PACExperimentConfig, class_count: int
) -> nn.Module | None:
    spec = _find_spec(name)
    if spec is None:
        return None
    return PACDesignStackClassifier(config, class_count, spec)


def _temporal_pyramid(sequence: Tensor) -> Tensor:
    pieces: list[Tensor] = []
    for segments in (1, 2, 4):
        for chunk in torch.tensor_split(sequence, segments, dim=1):
            pieces.append(chunk.mean(dim=1))
            pieces.append(torch.sqrt(chunk.square().mean(dim=1) + 1.0e-6))
    return torch.cat(pieces, dim=-1)


def _disable_prl_direct(prl: PACControlledTappedPRLBranch) -> None:
    prl.direct_term.requires_grad = False
    prl.bias.requires_grad = False


def _init_quadrature_projection(prl: PACControlledTappedPRLBranch, projection: nn.Linear) -> None:
    with torch.no_grad():
        prl.writer_real.zero_()
        prl.writer_imag.zero_()
        projection.weight.zero_()
        projection.bias.zero_()
        for mode in range(prl.modes):
            if mode < prl.model_dim:
                prl.writer_real[mode, mode] = 0.5
            imag_index = prl.modes + mode
            if imag_index < prl.model_dim:
                prl.writer_imag[mode, imag_index] = -0.5
        rows = min(projection.out_features, projection.in_features)
        projection.weight[:rows, :rows] = torch.eye(rows, dtype=projection.weight.dtype)


def _find_spec(name: str) -> PACDesignSpec | None:
    for spec in _SPECS:
        if spec.name == name:
            return spec
    return None
