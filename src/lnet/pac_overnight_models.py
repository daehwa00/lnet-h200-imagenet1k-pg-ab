from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor, nn

from .models import TransformerSequenceBaseline
from .pac_builders import SequenceMeanClassifier, build_classifier_model
from .pac_model import PACHybridPRLBlock

if TYPE_CHECKING:
    from .pac_types import PACExperimentConfig


class CNN1DClassifier(nn.Module):
    def __init__(self, *, channels: tuple[int, int, int], class_count: int) -> None:
        super().__init__()
        c1, c2, c3 = channels
        self.net = nn.Sequential(
            nn.Conv1d(1, c1, kernel_size=7, padding=3),
            nn.BatchNorm1d(c1),
            nn.GELU(),
            nn.Conv1d(c1, c2, kernel_size=5, padding=2),
            nn.BatchNorm1d(c2),
            nn.GELU(),
            nn.Conv1d(c2, c3, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.classifier = nn.Linear(c3, class_count)

    def forward(self, inputs: Tensor) -> Tensor:
        encoded = self.net(inputs.transpose(1, 2)).mean(dim=-1)
        return self.classifier(encoded)


class TCNClassifier(nn.Module):
    def __init__(self, *, channels: int, levels: int, class_count: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_channels = 1
        for level in range(levels):
            dilation = 2**level
            padding = dilation * 4
            layers.extend(
                (
                    nn.Conv1d(
                        current_channels,
                        channels,
                        kernel_size=5,
                        padding=padding,
                        dilation=dilation,
                    ),
                    _CausalTrim(padding),
                    nn.GELU(),
                )
            )
            current_channels = channels
        self.net = nn.Sequential(*layers)
        self.classifier = nn.Linear(channels, class_count)

    def forward(self, inputs: Tensor) -> Tensor:
        encoded = self.net(inputs.transpose(1, 2)).mean(dim=-1)
        return self.classifier(encoded)


class _CausalTrim(nn.Module):
    def __init__(self, trim: int) -> None:
        super().__init__()
        self.trim = trim

    def forward(self, inputs: Tensor) -> Tensor:
        if self.trim == 0:
            return inputs
        return inputs[..., : -self.trim]


def build_overnight_classifier(
    name: str,
    config: PACExperimentConfig,
    class_count: int,
) -> nn.Module:
    factories = {
        "cnn1d": lambda: CNN1DClassifier(channels=(32, 64, 64), class_count=class_count),
        "cnn1d_small": lambda: CNN1DClassifier(channels=(16, 32, 32), class_count=class_count),
        "tcn": lambda: TCNClassifier(channels=32, levels=4, class_count=class_count),
        "tcn_small": lambda: TCNClassifier(channels=16, levels=3, class_count=class_count),
        "transformer_tiny": lambda: _transformer_tiny(class_count),
        "fir_classifier": lambda: _fir_classifier(config, class_count),
        "pac_no_damping_control": lambda: _pac_classifier(
            config, class_count, damping_range=0.0, use_mlp=True
        ),
        "pac_lite": lambda: _pac_classifier(config, class_count, damping_range=1.0, use_mlp=False),
        "pac_full": lambda: _pac_classifier(config, class_count, damping_range=1.0, use_mlp=True),
        "gru": lambda: build_classifier_model("gru", config, class_count),
        "lstm": lambda: build_classifier_model("lstm", config, class_count),
        "selective_diagonal_ssm": lambda: build_classifier_model(
            "selective_diagonal_ssm", config, class_count
        ),
        "controlled_tapped_prl_only": lambda: build_classifier_model(
            "controlled_tapped_prl_only", config, class_count
        ),
        "fixed_prl": lambda: build_classifier_model("fixed_prl", config, class_count),
        "tapped_prl_fixed": lambda: build_classifier_model("tapped_prl_fixed", config, class_count),
    }
    if name in factories:
        return factories[name]()
    message = f"unsupported overnight model: {name}"
    raise KeyError(message)


def _transformer_tiny(class_count: int) -> nn.Module:
    encoder = TransformerSequenceBaseline(
        raw_input_dim=1, model_dim=16, output_dim=16, attention_heads=2
    )
    return SequenceMeanClassifier(encoder, 16, class_count)


def _pac_classifier(
    config: PACExperimentConfig,
    class_count: int,
    *,
    damping_range: float,
    use_mlp: bool,
) -> nn.Module:
    encoder = PACHybridPRLBlock(
        raw_input_dim=1,
        model_dim=config.model_dim,
        output_dim=config.model_dim,
        modes=config.modes,
        tap_kernel_size=config.tap_kernel_size,
        fir_kernel_size=config.fir_kernel_size,
        use_mlp_branch=use_mlp,
        active_branches=("prl", "fir", "mlp") if use_mlp else ("prl", "fir"),
        damping_control_range=damping_range,
    )
    return SequenceMeanClassifier(encoder, config.model_dim, class_count)


def _fir_classifier(config: PACExperimentConfig, class_count: int) -> nn.Module:
    encoder = nn.Sequential(
        nn.Conv1d(
            1,
            32,
            kernel_size=config.fir_kernel_size,
            padding=config.fir_kernel_size - 1,
        ),
        _CausalTrim(config.fir_kernel_size - 1),
        nn.GELU(),
    )
    return _ConvMeanClassifier(encoder, 32, class_count)


class _ConvMeanClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, feature_dim: int, class_count: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(feature_dim, class_count)

    def forward(self, inputs: Tensor) -> Tensor:
        encoded = self.encoder(inputs.transpose(1, 2)).mean(dim=-1)
        return self.classifier(encoded)
