from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Final, Literal, assert_never

import torch
from torch import Tensor, nn

from .pac_complex_modal_head import (
    ComplexModalPooling,
    ComplexModalSource,
    combine_complex_pool,
    complex_modal_output_dim,
    complex_modal_pool,
)
from .pac_model import PACHybridPRLBlock
from .pac_overnight_models import build_overnight_classifier

StandardPoolingName = Literal["mean", "mean_max", "pyramid", "attention"]
PoolingName = StandardPoolingName | ComplexModalPooling
POOLING_SUFFIXES: Final[tuple[tuple[str, PoolingName], ...]] = (
    ("_complex_stats", "complex_stats"),
    ("_hermitian", "hermitian"),
    ("_complex_pyramid", "complex_pyramid"),
    ("_mean_max", "mean_max"),
    ("_pyramid", "pyramid"),
    ("_attention", "attention"),
)

if TYPE_CHECKING:
    from .pac_types import PACExperimentConfig


@dataclass(frozen=True, slots=True)
class UpgradeSpec:
    branch: Literal["lite", "full"]
    pooling: PoolingName
    bidirectional: bool
    multi_scale_fir: bool
    depth: int
    modal_source: ComplexModalSource


class PoolingHead(nn.Module):
    def __init__(self, feature_dim: int, pooling: StandardPoolingName) -> None:
        super().__init__()
        self.pooling: StandardPoolingName = pooling
        self.attention = nn.Linear(feature_dim, 1) if pooling == "attention" else None
        self.output_dim = feature_dim * _pool_multiplier(pooling)

    def forward(self, features: Tensor) -> Tensor:
        match self.pooling:
            case "mean":
                return features.mean(dim=1)
            case "mean_max":
                return torch.cat((features.mean(dim=1), features.amax(dim=1)), dim=-1)
            case "pyramid":
                return torch.cat(_pyramid_pool(features), dim=-1)
            case "attention":
                if self.attention is None:
                    message = "attention pool is not initialized"
                    raise RuntimeError(message)
                weights = torch.softmax(self.attention(features), dim=1)
                return (features * weights).sum(dim=1)
            case unreachable:
                assert_never(unreachable)


class MultiScaleFIRAdapter(nn.Module):
    def __init__(self, feature_dim: int, kernels: tuple[int, ...] = (3, 5, 9, 17)) -> None:
        super().__init__()
        self.kernels = kernels
        self.depthwise = nn.ModuleList(
            nn.Conv1d(feature_dim, feature_dim, kernel_size=k, groups=feature_dim, padding=k - 1)
            for k in kernels
        )
        self.pointwise = nn.Linear(feature_dim * len(kernels), feature_dim)

    def forward(self, features: Tensor) -> Tensor:
        channels = features.transpose(1, 2)
        outputs = [
            conv(channels)[..., : features.shape[1]].transpose(1, 2) for conv in self.depthwise
        ]
        return self.pointwise(torch.cat(outputs, dim=-1))


class PACUpgradeClassifier(nn.Module):
    def __init__(self, config: PACExperimentConfig, class_count: int, spec: UpgradeSpec) -> None:
        super().__init__()
        self.forward_stack = _PACStack(config.raw_input_dim, config, spec)
        self.backward_stack = (
            _PACStack(config.raw_input_dim, config, spec) if spec.bidirectional else None
        )
        feature_dim = config.model_dim * (2 if spec.bidirectional else 1)
        modal_pooling = _complex_pooling(spec.pooling)
        self.modal_pooling: ComplexModalPooling | None = modal_pooling
        self.modal_source: ComplexModalSource = spec.modal_source
        if modal_pooling is None:
            pool = PoolingHead(feature_dim, _standard_pooling(spec.pooling))
            self.pool: PoolingHead | None = pool
            output_dim = pool.output_dim
        else:
            self.pool = None
            output_dim = complex_modal_output_dim(
                model_dim=config.model_dim,
                modes=config.modes,
                depth=spec.depth,
                pooling=modal_pooling,
                source=spec.modal_source,
            ) * (2 if spec.bidirectional else 1)
        hidden = max(16, min(128, output_dim // 2))
        self.classifier = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.Linear(output_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, class_count),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        modal_pooling = self.modal_pooling
        if modal_pooling is not None:
            pooled = self.forward_stack.modal_pool(inputs, modal_pooling, self.modal_source)
            if self.backward_stack is not None:
                backward = self.backward_stack.modal_pool(
                    torch.flip(inputs, dims=(1,)), modal_pooling, self.modal_source
                )
                pooled = torch.cat((pooled, backward), dim=-1)
            return self.classifier(pooled)
        pool = self.require_pool()
        features = self.forward_stack(inputs)
        if self.backward_stack is not None:
            backward = torch.flip(self.backward_stack(torch.flip(inputs, dims=(1,))), dims=(1,))
            features = torch.cat((features, backward), dim=-1)
        return self.classifier(pool(features))

    def require_pool(self) -> PoolingHead:
        if self.pool is None:
            message = "standard pooling head is not initialized"
            raise RuntimeError(message)
        return self.pool


def build_upgrade_classifier(name: str, config: PACExperimentConfig, class_count: int) -> nn.Module:
    if name == "mamba_ssm":
        return _MambaClassifier(config.raw_input_dim, config.model_dim, class_count)
    spec = parse_upgrade_name(name)
    if spec is not None:
        return PACUpgradeClassifier(config, class_count, spec)
    return build_overnight_classifier(name, config, class_count)


def parse_upgrade_name(name: str) -> UpgradeSpec | None:
    if not name.startswith(("pac_lite", "pac_full")):
        return None
    branch: Literal["lite", "full"] = "full" if name.startswith("pac_full") else "lite"
    return UpgradeSpec(
        branch=branch,
        pooling=_pooling_from_name(name),
        bidirectional="_bidir" in name,
        multi_scale_fir="_msfir" in name,
        depth=2 if "_stack2" in name else 1,
        modal_source="last" if "_last_" in name else "all",
    )


class _PACStack(nn.Module):
    def __init__(self, raw_input_dim: int, config: PACExperimentConfig, spec: UpgradeSpec) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            _block(raw_input_dim if index == 0 else config.model_dim, config, spec)
            for index in range(spec.depth)
        )
        self.adapter = MultiScaleFIRAdapter(config.model_dim) if spec.multi_scale_fir else None

    def forward(self, inputs: Tensor) -> Tensor:
        features = inputs
        for block in self.blocks:
            features = block(features)
        if self.adapter is not None:
            features = features + self.adapter(features)
        return features

    def modal_pool(
        self, inputs: Tensor, pooling: ComplexModalPooling, source: ComplexModalSource
    ) -> Tensor:
        blocks = tuple(_require_block(block) for block in self.blocks)
        features, modal_summary = complex_modal_pool(inputs, blocks, pooling, source)
        if self.adapter is not None:
            features = features + self.adapter(features)
        return combine_complex_pool(features, modal_summary, pooling)


class _MambaClassifier(nn.Module):
    def __init__(self, input_dim: int, model_dim: int, class_count: int) -> None:
        super().__init__()
        mamba_module = import_module("mamba_ssm")
        self.input_projection = nn.Linear(input_dim, model_dim)
        self.mamba = mamba_module.Mamba(d_model=model_dim, d_state=16, d_conv=4, expand=2)
        self.classifier = nn.Linear(model_dim, class_count)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.classifier(self.mamba(self.input_projection(inputs)).mean(dim=1))


def _block(raw_input_dim: int, config: PACExperimentConfig, spec: UpgradeSpec) -> PACHybridPRLBlock:
    branches = ("prl", "mlp") if spec.multi_scale_fir and spec.branch == "full" else None
    if spec.multi_scale_fir and spec.branch == "lite":
        branches = ("prl",)
    return PACHybridPRLBlock(
        raw_input_dim=raw_input_dim,
        model_dim=config.model_dim,
        output_dim=config.model_dim,
        modes=config.modes,
        tap_kernel_size=config.tap_kernel_size,
        fir_kernel_size=config.fir_kernel_size,
        use_mlp_branch=spec.branch == "full",
        active_branches=branches,
        damping_control_range=1.0,
    )


def _pool_multiplier(pooling: StandardPoolingName) -> int:
    match pooling:
        case "mean" | "attention":
            return 1
        case "mean_max":
            return 2
        case "pyramid":
            return 14


def _pooling_from_name(name: str) -> PoolingName:
    for suffix, pooling in POOLING_SUFFIXES:
        if name.endswith(suffix):
            return pooling
    return "mean"


def _complex_pooling(pooling: PoolingName) -> ComplexModalPooling | None:
    match pooling:
        case "complex_stats" | "hermitian" | "complex_pyramid":
            return pooling
        case "mean" | "mean_max" | "pyramid" | "attention":
            return None
        case unreachable:
            assert_never(unreachable)


def _standard_pooling(pooling: PoolingName) -> StandardPoolingName:
    match pooling:
        case "mean" | "mean_max" | "pyramid" | "attention":
            return pooling
        case "complex_stats" | "hermitian" | "complex_pyramid":
            message = f"{pooling} is not a standard pooling head"
            raise RuntimeError(message)
        case unreachable:
            assert_never(unreachable)


def _require_block(module: nn.Module) -> PACHybridPRLBlock:
    match module:
        case PACHybridPRLBlock():
            return module
        case _:
            message = "PAC stack contains a non-PAC block"
            raise RuntimeError(message)


def _pyramid_pool(features: Tensor) -> list[Tensor]:
    pooled: list[Tensor] = []
    for segments in (1, 2, 4):
        chunks = torch.tensor_split(features, segments, dim=1)
        for chunk in chunks:
            pooled.extend((chunk.mean(dim=1), chunk.amax(dim=1)))
    return pooled
