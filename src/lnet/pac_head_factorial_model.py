from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from .pac_head_factorial_features import BlockContext, block_context, modal_summary
from .pac_head_factorial_spec import PACHeadSpec, RealPool, feature_dim
from .pac_model import PACHybridPRLBlock

if TYPE_CHECKING:
    from .pac_types import PACBranchName, PACExperimentConfig


class PACHeadFactorialClassifier(nn.Module):
    def __init__(self, config: PACExperimentConfig, class_count: int, spec: PACHeadSpec) -> None:
        super().__init__()
        self.spec = spec
        self.forward_stack = _FactorialStack(config.raw_input_dim, config, spec)
        self.backward_stack = (
            _FactorialStack(config.raw_input_dim, config, spec)
            if spec.direction == "bidirectional"
            else None
        )
        input_dim = feature_dim(spec, config.model_dim, config.modes)
        hidden = max(16, min(128, input_dim // 2))
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, class_count),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        features = self.forward_stack.head_features(inputs)
        if self.backward_stack is not None:
            backward = self.backward_stack.head_features(torch.flip(inputs, dims=(1,)))
            features = torch.cat((features, backward), dim=-1)
        return self.classifier(features)


class _FactorialStack(nn.Module):
    def __init__(self, raw_input_dim: int, config: PACExperimentConfig, spec: PACHeadSpec) -> None:
        super().__init__()
        self.spec = spec
        self.blocks = nn.ModuleList(
            _block(raw_input_dim if index == 0 else config.model_dim, config, spec)
            for index in range(spec.depth)
        )
        self.mix_logits = (
            nn.Parameter(torch.zeros(spec.depth, dtype=torch.float32))
            if spec.source == "all_learned_mix" and spec.depth > 1
            else None
        )
        self.attention = (
            nn.Linear(3 * config.modes, 1) if spec.modal_feature == "modal_attention" else None
        )

    def head_features(self, inputs: Tensor) -> Tensor:
        contexts = self._contexts(inputs)
        pieces = [
            modal_summary(
                contexts,
                self.spec.source,
                self.spec.modal_feature,
                self.mix_logits,
                self.attention,
            )
        ]
        pieces.extend(_auxiliary_features(contexts[-1], self.spec))
        return torch.cat(pieces, dim=-1)

    def _contexts(self, inputs: Tensor) -> tuple[BlockContext, ...]:
        features = inputs
        contexts: list[BlockContext] = []
        for module in self.blocks:
            block = _require_block(module)
            context = block_context(block, features)
            contexts.append(context)
            features = context.output
        return tuple(contexts)


def _auxiliary_features(context: BlockContext, spec: PACHeadSpec) -> list[Tensor]:
    pieces: list[Tensor] = []
    if spec.real_pool != "none":
        pieces.append(_real_pool(context.output, spec.real_pool))
    if spec.damping_aux:
        pieces.append(_damping_aux(context))
    if spec.fir_aux:
        pieces.append(_branch_pool(_require_fir(context)))
    if spec.branch_aux:
        pieces.extend(_branch_aux(context))
    return pieces


def _real_pool(features: Tensor, mode: RealPool) -> Tensor:
    match mode:
        case "mean_max":
            return torch.cat((features.mean(dim=1), features.amax(dim=1)), dim=-1)
        case "pyramid":
            pooled: list[Tensor] = []
            for segments in (1, 2, 4):
                for chunk in torch.tensor_split(features, segments, dim=1):
                    pooled.extend((chunk.mean(dim=1), chunk.amax(dim=1)))
            return torch.cat(pooled, dim=-1)
        case "none":
            message = "none real pool should be handled before _real_pool"
            raise RuntimeError(message)


def _damping_aux(context: BlockContext) -> Tensor:
    damping = context.damping
    decay = context.decay_abs
    return torch.cat(
        (
            damping.mean(dim=1),
            damping.std(dim=1, unbiased=False),
            damping.amin(dim=1),
            damping.amax(dim=1),
            decay.mean(dim=1),
            decay.amax(dim=1),
        ),
        dim=-1,
    )


def _require_fir(context: BlockContext) -> Tensor:
    branches = _branch_outputs(context)
    output = branches.get("fir")
    if output is None:
        message = "FIR auxiliary requested but FIR branch is unavailable"
        raise RuntimeError(message)
    return output


def _branch_aux(context: BlockContext) -> list[Tensor]:
    return [_branch_pool(output) for _, output in _branch_outputs(context).items()]


def _branch_outputs(context: BlockContext) -> dict[PACBranchName, Tensor]:
    return context.block.branch_outputs(context.projected)


def _branch_pool(output: Tensor) -> Tensor:
    return torch.cat((output.mean(dim=1), output.amax(dim=1)), dim=-1)


def _block(raw_input_dim: int, config: PACExperimentConfig, spec: PACHeadSpec) -> PACHybridPRLBlock:
    return PACHybridPRLBlock(
        raw_input_dim=raw_input_dim,
        model_dim=config.model_dim,
        output_dim=config.model_dim,
        modes=config.modes,
        tap_kernel_size=config.tap_kernel_size,
        fir_kernel_size=config.fir_kernel_size,
        use_mlp_branch=spec.branch == "full",
        damping_control_range=1.0,
    )


def _require_block(module: nn.Module) -> PACHybridPRLBlock:
    match module:
        case PACHybridPRLBlock():
            return module
        case _:
            message = "factorial stack contains a non-PAC block"
            raise RuntimeError(message)
