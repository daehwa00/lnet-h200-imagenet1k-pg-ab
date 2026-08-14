from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final, Literal, assert_never

import torch
from torch import Tensor, nn
from torch.nn import functional

from .pac_head_factorial_model import PACHeadFactorialClassifier
from .pac_head_factorial_spec import Branch, PACHeadSpec

if TYPE_CHECKING:
    from .pac_types import PACExperimentConfig

StemKind = Literal["convstem1", "convstem2", "dwconvstem", "msstem", "resconvstem", "tcnstem"]


@dataclass(frozen=True, slots=True)
class LocalStemSpec:
    name: str
    stem: StemKind
    branch: Branch | None


@dataclass(frozen=True, slots=True)
class CausalConvSpec:
    input_dim: int
    output_dim: int
    kernel_size: int
    dilation: int = 1
    groups: int = 1


LOCAL_STEM_SPECS: Final[tuple[LocalStemSpec, ...]] = (
    LocalStemSpec("pac_convstem1_full_depth2_hermitian_realmeanmax", "convstem1", "full"),
    LocalStemSpec("pac_dwconvstem_full_depth2_hermitian_realmeanmax", "dwconvstem", "full"),
    LocalStemSpec("pac_msstem_full_depth2_hermitian_realmeanmax", "msstem", "full"),
    LocalStemSpec("pac_resconvstem_full_depth2_hermitian_realmeanmax", "resconvstem", "full"),
    LocalStemSpec("pac_tcnstem_full_depth2_hermitian_realmeanmax", "tcnstem", "full"),
    LocalStemSpec("pac_convstem1_lite_depth2_hermitian_realmeanmax", "convstem1", "lite"),
    LocalStemSpec("pac_convstem2_full_depth2_hermitian_realmeanmax", "convstem2", "full"),
    LocalStemSpec("pac_convstem2_lite_depth2_hermitian_realmeanmax", "convstem2", "lite"),
    LocalStemSpec("pac_dwconvstem_lite_depth2_hermitian_realmeanmax", "dwconvstem", "lite"),
    LocalStemSpec("pac_msstem_lite_depth2_hermitian_realmeanmax", "msstem", "lite"),
    LocalStemSpec("convstem1_only", "convstem1", None),
    LocalStemSpec("dwconvstem_only", "dwconvstem", None),
    LocalStemSpec("msstem_only", "msstem", None),
    LocalStemSpec("resconvstem_only", "resconvstem", None),
    LocalStemSpec("tcnstem_only", "tcnstem", None),
)
LOCAL_STEM_MODELS: Final[tuple[str, ...]] = tuple(spec.name for spec in LOCAL_STEM_SPECS)
CONVSTEM2_MODELS: Final[tuple[str, ...]] = (
    "pac_convstem2_full_depth2_hermitian_realmeanmax",
    "pac_convstem2_lite_depth2_hermitian_realmeanmax",
)


class LocalStemPACClassifier(nn.Module):
    def __init__(self, config: PACExperimentConfig, class_count: int, spec: LocalStemSpec) -> None:
        super().__init__()
        if spec.branch is None:
            message = "PAC local-stem classifier requires a PAC branch"
            raise RuntimeError(message)
        self.stem = _build_stem(spec.stem, config.raw_input_dim, config.model_dim)
        stem_config = replace(config, raw_input_dim=config.model_dim)
        head_spec = PACHeadSpec(
            branch=spec.branch,
            depth=2,
            direction="causal",
            source="all_learned_mix",
            modal_feature="hermitian",
            real_pool="mean_max",
            damping_aux=False,
            fir_aux=False,
            branch_aux=False,
        )
        self.classifier = PACHeadFactorialClassifier(stem_config, class_count, head_spec)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.classifier(self.stem(inputs))


class StemOnlyClassifier(nn.Module):
    def __init__(self, config: PACExperimentConfig, class_count: int, spec: LocalStemSpec) -> None:
        super().__init__()
        self.stem = _build_stem(spec.stem, config.raw_input_dim, config.model_dim)
        input_dim = 2 * config.model_dim
        hidden = max(16, min(128, input_dim))
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, class_count),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        features = self.stem(inputs)
        pooled = torch.cat((features.mean(dim=1), features.amax(dim=1)), dim=-1)
        return self.classifier(pooled)


class _CausalConv1d(nn.Module):
    def __init__(self, spec: CausalConvSpec) -> None:
        super().__init__()
        self.left_pad = (spec.kernel_size - 1) * spec.dilation
        self.conv = nn.Conv1d(
            spec.input_dim,
            spec.output_dim,
            kernel_size=spec.kernel_size,
            dilation=spec.dilation,
            groups=spec.groups,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        channels = inputs.transpose(1, 2)
        padded = functional.pad(channels, (self.left_pad, 0))
        return self.conv(padded).transpose(1, 2)


class _ConvStem(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.conv = _CausalConv1d(CausalConvSpec(input_dim, output_dim, 5))
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        return functional.gelu(self.norm(self.conv(inputs)))


class _ConvStem2(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.first = _CausalConv1d(CausalConvSpec(input_dim, output_dim, 5))
        self.second = _CausalConv1d(CausalConvSpec(output_dim, output_dim, 5))
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        first = functional.gelu(self.first(inputs))
        return functional.gelu(self.norm(first + self.second(first)))


class _DepthwiseStem(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.depthwise = _CausalConv1d(CausalConvSpec(input_dim, input_dim, 9, groups=input_dim))
        self.pointwise = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        return functional.gelu(self.norm(self.pointwise(self.depthwise(inputs))))


class _MultiScaleStem(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.convs = nn.ModuleList(
            _CausalConv1d(CausalConvSpec(input_dim, output_dim, kernel)) for kernel in (3, 5, 9, 17)
        )
        self.projection = nn.Linear(4 * output_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        features = torch.cat([conv(inputs) for conv in self.convs], dim=-1)
        return functional.gelu(self.norm(self.projection(features)))


class _ResidualStem(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.base = nn.Linear(input_dim, output_dim)
        self.local = _ConvStem(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        return functional.gelu(self.norm(self.base(inputs) + self.local(inputs)))


class _TinyTCNStem(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.first = _CausalConv1d(CausalConvSpec(input_dim, output_dim, 5))
        self.second = _CausalConv1d(CausalConvSpec(output_dim, output_dim, 5, dilation=2))
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        first = functional.gelu(self.first(inputs))
        return functional.gelu(self.norm(first + self.second(first)))


def build_local_stem_classifier(
    name: str, config: PACExperimentConfig, class_count: int
) -> nn.Module | None:
    spec = _find_spec(name)
    if spec is None:
        return None
    if spec.branch is None:
        return StemOnlyClassifier(config, class_count, spec)
    return LocalStemPACClassifier(config, class_count, spec)


def _find_spec(name: str) -> LocalStemSpec | None:
    for spec in LOCAL_STEM_SPECS:
        if spec.name == name:
            return spec
    return None


def _build_stem(kind: StemKind, input_dim: int, output_dim: int) -> nn.Module:
    match kind:
        case "convstem1":
            return _ConvStem(input_dim, output_dim)
        case "convstem2":
            return _ConvStem2(input_dim, output_dim)
        case "dwconvstem":
            return _DepthwiseStem(input_dim, output_dim)
        case "msstem":
            return _MultiScaleStem(input_dim, output_dim)
        case "resconvstem":
            return _ResidualStem(input_dim, output_dim)
        case "tcnstem":
            return _TinyTCNStem(input_dim, output_dim)
        case unreachable:
            assert_never(unreachable)
