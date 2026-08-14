from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING, Final, TypeVar

from .pac_builders import build_classifier_model
from .pac_classifier_upgrade_models import build_upgrade_classifier
from .pac_design_stack_models import build_pac_design_classifier
from .pac_head_factorial_model import PACHeadFactorialClassifier
from .pac_head_factorial_spec import PACHeadSpec
from .pac_headroom_efficient_models import build_efficient_headroom_classifier
from .pac_implicit_complex_models import build_implicit_complex_classifier
from .pac_local_stem_models import build_local_stem_classifier
from .pac_overnight_models import CNN1DClassifier, TCNClassifier
from .pac_qprl_models import build_qprl_classifier
from .pac_tight_frame_models import build_tight_frame_classifier
from .pac_unified_models import build_unified_pac_classifier

if TYPE_CHECKING:
    from torch import nn

    from .pac_types import PACExperimentConfig

RECOMMENDED_MODEL: Final[str] = "pac_full_depth2_causal_all_learned_mix_hermitian_realmean_max"
RECOMMENDED_SPEC: Final[PACHeadSpec] = PACHeadSpec(
    branch="full",
    depth=2,
    direction="causal",
    source="all_learned_mix",
    modal_feature="hermitian",
    real_pool="mean_max",
    damping_aux=False,
    fir_aux=False,
    branch_aux=False,
)
MATCHED_PAC_D64_M16_CONTROL: Final[str] = "pac_stiefel_depth2_norm_autocorr_d64_m16"
MATCHED_PAC_D64_M16_BASELINES: Final[tuple[str, ...]] = (
    "cnn1d_matched_pac_d64_m16",
    "tcn_matched_pac_d64_m16",
    "gru_matched_pac_d64_m16",
    "lstm_matched_pac_d64_m16",
    "mamba_ssm_matched_pac_d64_m16",
    "transformer_matched_pac_d64_m16",
)
MATCHED_PAC_D64_M16_MODELS: Final[tuple[str, ...]] = (
    MATCHED_PAC_D64_M16_CONTROL,
    *MATCHED_PAC_D64_M16_BASELINES,
)
# All 13 benchmark datasets have 2--7 classes. Widths below keep each baseline
# within 5% of the class-count-specific D64/M16 PAC trainable-parameter count.
MATCHED_BASELINE_RELATIVE_TOLERANCE: Final[float] = 0.05
_CNN_CHANNELS_BY_CLASSES: Final[dict[int, tuple[int, int, int]]] = {
    2: (13, 39, 39),
    3: (16, 32, 48),
    4: (17, 34, 51),
    5: (20, 40, 40),
    6: (18, 36, 54),
    7: (21, 42, 42),
}
_TCN_WIDTH_BY_CLASSES: Final[dict[int, int]] = {2: 22, 3: 23, 4: 23, 5: 24, 6: 25, 7: 26}
_GRU_WIDTH_BY_CLASSES: Final[dict[int, int]] = {2: 32, 3: 33, 4: 34, 5: 35, 6: 36, 7: 37}
_LSTM_WIDTH_BY_CLASSES: Final[dict[int, int]] = {2: 28, 3: 29, 4: 30, 5: 31, 6: 32, 7: 33}
_MAMBA_WIDTH_BY_CLASSES: Final[dict[int, int]] = {2: 26, 3: 28, 4: 29, 5: 30, 6: 31, 7: 32}
_TRANSFORMER_WIDTH_BY_CLASSES: Final[dict[int, int]] = {
    2: 28,
    3: 29,
    4: 30,
    5: 31,
    6: 32,
    7: 33,
}
ClassifierFactory = Callable[[], "nn.Module"]
T = TypeVar("T")


def build_low_data_classifier(
    name: str, config: PACExperimentConfig, class_count: int
) -> nn.Module:
    if name in {"WP", "PA2WP"}:
        return build_efficient_headroom_classifier(
            name,
            config,
            class_count,
            objective="classification",
        )
    for builder in (
        build_unified_pac_classifier,
        build_tight_frame_classifier,
        build_implicit_complex_classifier,
        build_qprl_classifier,
        build_pac_design_classifier,
        build_local_stem_classifier,
    ):
        specialized = builder(name, config, class_count)
        if specialized is not None:
            return specialized
    if name == RECOMMENDED_MODEL:
        return PACHeadFactorialClassifier(config, class_count, RECOMMENDED_SPEC)
    matched_factories: dict[str, ClassifierFactory] = {
        "tcn_matched10k": lambda: TCNClassifier(channels=27, levels=4, class_count=class_count),
        "cnn1d_matched10k": lambda: CNN1DClassifier(channels=(20, 40, 56), class_count=class_count),
        "gru_matched10k": lambda: build_classifier_model(
            "gru", replace(config, model_dim=39, output_dim=39), class_count
        ),
        "lstm_matched10k": lambda: build_classifier_model(
            "lstm", replace(config, model_dim=35, output_dim=35), class_count
        ),
        "mamba_ssm_matched10k": lambda: build_upgrade_classifier(
            "mamba_ssm", replace(config, model_dim=34, output_dim=34), class_count
        ),
        "transformer_matched10k": lambda: build_classifier_model(
            "transformer", replace(config, model_dim=34, output_dim=34), class_count
        ),
        "cnn1d_matched_pac_d64_m16": lambda: CNN1DClassifier(
            channels=_matched_value(_CNN_CHANNELS_BY_CLASSES, class_count),
            class_count=class_count,
        ),
        "tcn_matched_pac_d64_m16": lambda: TCNClassifier(
            channels=_matched_value(_TCN_WIDTH_BY_CLASSES, class_count),
            levels=4,
            class_count=class_count,
        ),
        "gru_matched_pac_d64_m16": lambda: build_classifier_model(
            "gru",
            _matched_config(config, _matched_value(_GRU_WIDTH_BY_CLASSES, class_count)),
            class_count,
        ),
        "lstm_matched_pac_d64_m16": lambda: build_classifier_model(
            "lstm",
            _matched_config(config, _matched_value(_LSTM_WIDTH_BY_CLASSES, class_count)),
            class_count,
        ),
        "mamba_ssm_matched_pac_d64_m16": lambda: build_upgrade_classifier(
            "mamba_ssm",
            _matched_config(config, _matched_value(_MAMBA_WIDTH_BY_CLASSES, class_count)),
            class_count,
        ),
        "transformer_matched_pac_d64_m16": lambda: build_classifier_model(
            "transformer",
            _matched_config(
                config,
                _matched_value(_TRANSFORMER_WIDTH_BY_CLASSES, class_count),
            ),
            class_count,
        ),
    }
    if name in matched_factories:
        return matched_factories[name]()
    return build_upgrade_classifier(name, config, class_count)


def _matched_config(config: PACExperimentConfig, model_dim: int) -> PACExperimentConfig:
    return replace(config, model_dim=model_dim, output_dim=model_dim)


def _matched_value(values: dict[int, T], class_count: int) -> T:  # noqa: UP047 - B200 py3.10
    try:
        return values[class_count]
    except KeyError as error:
        message = "D64/M16 matched baselines support benchmark class counts 2 through 7"
        raise ValueError(message) from error
