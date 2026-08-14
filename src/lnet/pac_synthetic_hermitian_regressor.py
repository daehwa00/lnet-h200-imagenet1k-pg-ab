from __future__ import annotations

from typing import TYPE_CHECKING, Final

import torch
from torch import Tensor, nn

from .pac_head_factorial_features import block_context, modal_summary
from .pac_head_factorial_spec import PACHeadSpec, feature_dim
from .pac_model import PACHybridPRLBlock

if TYPE_CHECKING:
    from .pac_types import PACExperimentConfig

HERMITIAN_REGRESSION_SPECS: Final[dict[str, PACHeadSpec]] = {
    "pac_lite_depth1_last_hermitian_realnone_seqreg": PACHeadSpec(
        branch="lite",
        depth=1,
        direction="causal",
        source="last",
        modal_feature="hermitian",
        real_pool="none",
        damping_aux=False,
        fir_aux=False,
        branch_aux=False,
    ),
    "pac_lite_depth1_last_hermitian_realmean_max_seqreg": PACHeadSpec(
        branch="lite",
        depth=1,
        direction="causal",
        source="last",
        modal_feature="hermitian",
        real_pool="mean_max",
        damping_aux=False,
        fir_aux=False,
        branch_aux=False,
    ),
    "pac_full_depth1_last_hermitian_realnone_seqreg": PACHeadSpec(
        branch="full",
        depth=1,
        direction="causal",
        source="last",
        modal_feature="hermitian",
        real_pool="none",
        damping_aux=False,
        fir_aux=False,
        branch_aux=False,
    ),
    "pac_full_depth1_last_hermitian_realmean_max_seqreg": PACHeadSpec(
        branch="full",
        depth=1,
        direction="causal",
        source="last",
        modal_feature="hermitian",
        real_pool="mean_max",
        damping_aux=False,
        fir_aux=False,
        branch_aux=False,
    ),
    "pac_lite_depth2_all_learned_mix_hermitian_realmean_max_seqreg": PACHeadSpec(
        branch="lite",
        depth=2,
        direction="causal",
        source="all_learned_mix",
        modal_feature="hermitian",
        real_pool="mean_max",
        damping_aux=False,
        fir_aux=False,
        branch_aux=False,
    ),
    "pac_full_depth2_all_learned_mix_hermitian_realmean_max_seqreg": PACHeadSpec(
        branch="full",
        depth=2,
        direction="causal",
        source="all_learned_mix",
        modal_feature="hermitian",
        real_pool="mean_max",
        damping_aux=False,
        fir_aux=False,
        branch_aux=False,
    ),
}


class PACSyntheticHermitianRegressor(nn.Module):
    def __init__(self, config: PACExperimentConfig, spec: PACHeadSpec) -> None:
        super().__init__()
        self.spec = spec
        self.blocks = nn.ModuleList(
            _block(config.raw_input_dim if index == 0 else config.model_dim, config, spec)
            for index in range(spec.depth)
        )
        self.mix_logits = (
            nn.Parameter(torch.zeros(spec.depth, dtype=torch.float32))
            if spec.source == "all_learned_mix" and spec.depth > 1
            else None
        )
        summary_dim = feature_dim(spec, config.model_dim, config.modes)
        self.context_projection = nn.Sequential(
            nn.LayerNorm(summary_dim),
            nn.Linear(summary_dim, config.model_dim),
            nn.GELU(),
        )
        self.output_projection = nn.Linear(2 * config.model_dim, config.output_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        features = inputs
        contexts = []
        for block in self.blocks:
            context = block_context(_require_block(block), features)
            contexts.append(context)
            features = context.output
        summary = modal_summary(
            tuple(contexts),
            self.spec.source,
            "hermitian",
            self.mix_logits,
            None,
        )
        if self.spec.real_pool == "mean_max":
            summary = torch.cat((summary, features.mean(dim=1), features.amax(dim=1)), dim=-1)
        global_context = self.context_projection(summary).unsqueeze(1)
        global_context = global_context.expand(-1, features.shape[1], -1)
        return self.output_projection(torch.cat((features, global_context), dim=-1))


def build_synthetic_hermitian_regressor(name: str, config: PACExperimentConfig) -> nn.Module:
    return PACSyntheticHermitianRegressor(config, HERMITIAN_REGRESSION_SPECS[name])


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
            message = "synthetic Hermitian stack contains a non-PAC block"
            raise RuntimeError(message)
