from __future__ import annotations

from dataclasses import replace
from importlib import import_module
from typing import TYPE_CHECKING

from torch import Tensor, nn

from .models import FIRSequenceBaseline, TransformerSequenceBaseline
from .pac_builders import build_regression_model
from .pac_overnight_models import build_overnight_classifier
from .pac_synthetic_hermitian_regressor import (
    HERMITIAN_REGRESSION_SPECS,
    build_synthetic_hermitian_regressor,
)

if TYPE_CHECKING:
    from .pac_types import PACExperimentConfig


class CNN1DRegressor(nn.Module):
    def __init__(
        self, *, raw_input_dim: int, channels: tuple[int, int, int], output_dim: int
    ) -> None:
        super().__init__()
        c1, c2, c3 = channels
        self.net = nn.Sequential(
            nn.Conv1d(raw_input_dim, c1, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(c1, c2, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(c2, c3, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.output_projection = nn.Linear(c3, output_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        encoded = self.net(inputs.transpose(1, 2)).transpose(1, 2)
        return self.output_projection(encoded)


class TCNRegressor(nn.Module):
    def __init__(self, *, raw_input_dim: int, channels: int, levels: int, output_dim: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = raw_input_dim
        for level in range(levels):
            padding = (2**level) * 4
            layers.extend(
                (
                    nn.Conv1d(current, channels, 5, padding=padding, dilation=2**level),
                    _Trim(padding),
                    nn.GELU(),
                )
            )
            current = channels
        self.net = nn.Sequential(*layers)
        self.output_projection = nn.Linear(channels, output_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        encoded = self.net(inputs.transpose(1, 2)).transpose(1, 2)
        return self.output_projection(encoded)


class MambaSSMRegressor(nn.Module):
    def __init__(self, *, raw_input_dim: int, model_dim: int, output_dim: int) -> None:
        super().__init__()
        mamba_module = import_module("mamba_ssm")
        self.input_projection = nn.Linear(raw_input_dim, model_dim)
        self.mamba = mamba_module.Mamba(d_model=model_dim, d_state=16, d_conv=4, expand=2)
        self.output_projection = nn.Linear(model_dim, output_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.output_projection(self.mamba(self.input_projection(inputs)))


class _Trim(nn.Module):
    def __init__(self, trim: int) -> None:
        super().__init__()
        self.trim = trim

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs if self.trim == 0 else inputs[..., : -self.trim]


def build_paper_regressor(name: str, config: PACExperimentConfig) -> nn.Module:
    model: nn.Module
    special = _special_regressor(name, config)
    if special is not None:
        return special
    match name:
        case (
            "pac_lite"
            | "pac_full"
            | "controlled_tapped_prl_only"
            | "tapped_prl_fixed"
            | "fixed_prl"
            | "gru"
            | "lstm"
            | "transformer"
            | "selective_diagonal_ssm"
            | "fir_only"
            | "mlp_only"
            | "linear_recurrent"
        ):
            model = build_regression_model(name, config)
        case "cnn1d":
            model = CNN1DRegressor(
                raw_input_dim=config.raw_input_dim,
                channels=(32, 64, 64),
                output_dim=config.output_dim,
            )
        case "cnn1d_small":
            model = CNN1DRegressor(
                raw_input_dim=config.raw_input_dim,
                channels=(16, 32, 32),
                output_dim=config.output_dim,
            )
        case "tcn":
            model = TCNRegressor(
                raw_input_dim=config.raw_input_dim,
                channels=32,
                levels=4,
                output_dim=config.output_dim,
            )
        case "tcn_small":
            model = TCNRegressor(
                raw_input_dim=config.raw_input_dim,
                channels=16,
                levels=3,
                output_dim=config.output_dim,
            )
        case "transformer_tiny":
            model = TransformerSequenceBaseline(
                raw_input_dim=config.raw_input_dim,
                model_dim=16,
                output_dim=config.output_dim,
                attention_heads=2,
            )
        case "fir_classifier":
            model = FIRSequenceBaseline(
                raw_input_dim=config.raw_input_dim,
                model_dim=config.model_dim,
                output_dim=config.output_dim,
                kernel_size=config.fir_kernel_size,
            )
        case "mamba_ssm":
            model = MambaSSMRegressor(
                raw_input_dim=config.raw_input_dim,
                model_dim=config.model_dim,
                output_dim=config.output_dim,
            )
        case _:
            message = f"unsupported paper regressor: {name}"
            raise KeyError(message)
    return model


def _special_regressor(name: str, config: PACExperimentConfig) -> nn.Module | None:
    if name in HERMITIAN_REGRESSION_SPECS:
        return build_synthetic_hermitian_regressor(name, config)
    return _matched_regressor(name, config)


def _matched_config(config: PACExperimentConfig, model_dim: int) -> PACExperimentConfig:
    return replace(config, model_dim=model_dim)


def _matched_gru(config: PACExperimentConfig) -> nn.Module:
    return build_regression_model("gru", _matched_config(config, 36))


def _matched_lstm(config: PACExperimentConfig) -> nn.Module:
    return build_regression_model("lstm", _matched_config(config, 31))


def _matched_transformer(config: PACExperimentConfig) -> nn.Module:
    return build_regression_model("transformer", _matched_config(config, 31))


def _matched_mamba(config: PACExperimentConfig) -> nn.Module:
    return MambaSSMRegressor(
        raw_input_dim=config.raw_input_dim,
        model_dim=28,
        output_dim=config.output_dim,
    )


def _matched_tcn(config: PACExperimentConfig) -> nn.Module:
    return TCNRegressor(
        raw_input_dim=config.raw_input_dim,
        channels=39,
        levels=2,
        output_dim=config.output_dim,
    )


def _matched_cnn(config: PACExperimentConfig) -> nn.Module:
    return CNN1DRegressor(
        raw_input_dim=config.raw_input_dim,
        channels=(28, 32, 32),
        output_dim=config.output_dim,
    )


SEQREG_MATCHED_FACTORIES = {
    "gru_seqreg_matched": _matched_gru,
    "lstm_seqreg_matched": _matched_lstm,
    "transformer_seqreg_matched": _matched_transformer,
    "mamba_ssm_seqreg_matched": _matched_mamba,
    "tcn_seqreg_matched": _matched_tcn,
    "cnn1d_seqreg_matched": _matched_cnn,
}


def _matched_regressor(name: str, config: PACExperimentConfig) -> nn.Module | None:
    factory = SEQREG_MATCHED_FACTORIES.get(name)
    return None if factory is None else factory(config)


def build_paper_classifier(name: str, config: PACExperimentConfig, class_count: int) -> nn.Module:
    return build_overnight_classifier(name, config, class_count)
