"""Public package exports without importing optional model families eagerly."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lnet.alphabet import Alphabet
    from lnet.alphabet2d import Alphabet2D, Alphabet2DConfig, ProductPoleField2D
    from lnet.experiment import (
        SyntheticLaplaceTask,
        SyntheticTaskConfig,
        TrainedLaplaceModel,
        TrainingConfig,
        TrainingSummary,
        make_synthetic_task,
        train_synthetic_model,
    )
    from lnet.hybrid import HybridModalPRLBlock
    from lnet.laplace import PoleResidueTemporalMixing, ProjectedPRLBlock
    from lnet.models import LinearRecurrentBaseline, PerStepMLPBaseline
    from lnet.pac_global_pole_gain import (
        GlobalFiniteGridGainNorm,
        factorized_product_mean_variance,
    )
    from lnet.tapped_prl import TappedPRLBranch


_LAZY_MODULES = {
    "Alphabet": "lnet.alphabet",
    "Alphabet2D": "lnet.alphabet2d",
    "Alphabet2DConfig": "lnet.alphabet2d",
    "GlobalFiniteGridGainNorm": "lnet.pac_global_pole_gain",
    "HybridModalPRLBlock": "lnet.hybrid",
    "LinearRecurrentBaseline": "lnet.models",
    "PerStepMLPBaseline": "lnet.models",
    "PoleResidueTemporalMixing": "lnet.laplace",
    "ProductPoleField2D": "lnet.alphabet2d",
    "ProjectedPRLBlock": "lnet.laplace",
    "SyntheticLaplaceTask": "lnet.experiment",
    "SyntheticTaskConfig": "lnet.experiment",
    "TappedPRLBranch": "lnet.tapped_prl",
    "TrainedLaplaceModel": "lnet.experiment",
    "TrainingConfig": "lnet.experiment",
    "TrainingSummary": "lnet.experiment",
    "factorized_product_mean_variance": "lnet.pac_global_pole_gain",
    "make_synthetic_task": "lnet.experiment",
    "train_synthetic_model": "lnet.experiment",
}

__all__ = [
    "Alphabet",
    "Alphabet2D",
    "Alphabet2DConfig",
    "GlobalFiniteGridGainNorm",
    "HybridModalPRLBlock",
    "LinearRecurrentBaseline",
    "PerStepMLPBaseline",
    "PoleResidueTemporalMixing",
    "ProductPoleField2D",
    "ProjectedPRLBlock",
    "SyntheticLaplaceTask",
    "SyntheticTaskConfig",
    "TappedPRLBranch",
    "TrainedLaplaceModel",
    "TrainingConfig",
    "TrainingSummary",
    "factorized_product_mean_variance",
    "make_synthetic_task",
    "train_synthetic_model",
]


def __getattr__(name: str) -> object:
    """Resolve a public export on first use and cache the original object."""
    try:
        module_name = _LAZY_MODULES[name]
    except KeyError:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message) from None
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include unresolved public exports in interactive discovery."""
    return sorted(set(globals()) | set(__all__))
