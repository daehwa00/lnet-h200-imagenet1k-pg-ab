from __future__ import annotations

from collections.abc import Callable
from typing import Final

from torch import Tensor, nn

from .models import (
    FIRSequenceBaseline,
    GRUSequenceBaseline,
    LinearRecurrentBaseline,
    PerStepMLPBaseline,
    TransformerSequenceBaseline,
)
from .pac_baselines import LSTMSequenceBaseline, SelectiveDiagonalSSMBaseline
from .pac_model import PACHybridPRLBlock
from .pac_types import PACBranchName, PACExperimentConfig, PACModelName

RegressionFactory = Callable[[PACExperimentConfig], nn.Module]


def build_regression_model(name: PACModelName, config: PACExperimentConfig) -> nn.Module:
    return REGRESSION_FACTORIES[name](config)


def build_classifier_model(
    name: PACModelName, config: PACExperimentConfig, class_count: int
) -> nn.Module:
    encoder = build_regression_model(name, _classification_config(config, config.model_dim))
    return SequenceMeanClassifier(encoder, config.model_dim, class_count)


class SequenceMeanClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, feature_dim: int, class_count: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(feature_dim, class_count)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.classifier(self.encoder(inputs).mean(dim=1))


def _common(config: PACExperimentConfig) -> dict[str, int]:
    return {
        "raw_input_dim": config.raw_input_dim,
        "model_dim": config.model_dim,
        "output_dim": config.output_dim,
    }


def _pac(
    config: PACExperimentConfig,
    *,
    use_mlp: bool,
    branches: tuple[PACBranchName, ...],
    damping_range: float = 1.0,
    tap_size: int | None = None,
) -> PACHybridPRLBlock:
    return PACHybridPRLBlock(
        raw_input_dim=config.raw_input_dim,
        model_dim=config.model_dim,
        output_dim=config.output_dim,
        modes=config.modes,
        tap_kernel_size=tap_size or config.tap_kernel_size,
        fir_kernel_size=config.fir_kernel_size,
        use_mlp_branch=use_mlp,
        active_branches=branches,
        damping_control_range=damping_range,
    )


def _pac_full(config: PACExperimentConfig) -> nn.Module:
    return _pac(config, use_mlp=True, branches=("prl", "fir", "mlp"))


def _pac_lite(config: PACExperimentConfig) -> nn.Module:
    return _pac(config, use_mlp=False, branches=("prl", "fir"))


def _controlled_prl(config: PACExperimentConfig) -> nn.Module:
    return _pac(config, use_mlp=False, branches=("prl",))


def _fixed_tapped_prl(config: PACExperimentConfig) -> nn.Module:
    return _pac(config, use_mlp=False, branches=("prl",), damping_range=0.0)


def _fixed_prl(config: PACExperimentConfig) -> nn.Module:
    return _pac(config, use_mlp=False, branches=("prl",), damping_range=0.0, tap_size=1)


def _fir(config: PACExperimentConfig) -> nn.Module:
    return FIRSequenceBaseline(**_common(config), kernel_size=config.fir_kernel_size)


def _mlp(config: PACExperimentConfig) -> nn.Module:
    return PerStepMLPBaseline(**_common(config))


def _linear_recurrent(config: PACExperimentConfig) -> nn.Module:
    return LinearRecurrentBaseline(**_common(config), state_dim=config.model_dim)


def _gru(config: PACExperimentConfig) -> nn.Module:
    return GRUSequenceBaseline(**_common(config))


def _lstm(config: PACExperimentConfig) -> nn.Module:
    return LSTMSequenceBaseline(**_common(config))


def _transformer(config: PACExperimentConfig) -> nn.Module:
    return TransformerSequenceBaseline(
        **_common(config), attention_heads=2 if config.model_dim % 2 == 0 else 1
    )


def _selective_ssm(config: PACExperimentConfig) -> nn.Module:
    return SelectiveDiagonalSSMBaseline(**_common(config), modes=config.modes)


REGRESSION_FACTORIES: Final[dict[PACModelName, RegressionFactory]] = {
    "pac_full": _pac_full,
    "pac_lite": _pac_lite,
    "controlled_tapped_prl_only": _controlled_prl,
    "tapped_prl_fixed": _fixed_tapped_prl,
    "fixed_prl": _fixed_prl,
    "fir_only": _fir,
    "mlp_only": _mlp,
    "linear_recurrent": _linear_recurrent,
    "gru": _gru,
    "lstm": _lstm,
    "transformer": _transformer,
    "selective_diagonal_ssm": _selective_ssm,
}


def _classification_config(config: PACExperimentConfig, output_dim: int) -> PACExperimentConfig:
    return PACExperimentConfig(
        sample_count=config.sample_count,
        validation_count=config.validation_count,
        test_count=config.test_count,
        sequence_length=config.sequence_length,
        raw_input_dim=1,
        output_dim=output_dim,
        model_dim=config.model_dim,
        modes=config.modes,
        tap_kernel_size=config.tap_kernel_size,
        fir_kernel_size=config.fir_kernel_size,
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        grad_clip_norm=config.grad_clip_norm,
        seeds=config.seeds,
        device=config.device,
        output_dir=config.output_dir,
    )
