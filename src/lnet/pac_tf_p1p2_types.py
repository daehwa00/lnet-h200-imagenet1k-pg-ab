from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal

if TYPE_CHECKING:
    from .pac_types import PACDevice

DEFAULT_SELECTION_PATH: Final = Path(
    ".omx/results/pac-tf-confirmatory-unseen-20260711/reports/confirmatory_baseline_selection.json"
)
DEFAULT_UNSEEN_ROOT: Final = Path(".omx/results/pac-tf-confirmatory-unseen-20260711")

EvidencePackage = Literal[
    "low_data",
    "synthetic_ood",
    "real_diagnostics",
    "real_domain_ood",
    "efficiency",
]
EvidenceRuntime = Literal["train", "eager", "compiled"]
SyntheticEstimand = Literal["sequence", "endpoint"]
RatioOneFitPolicy = Literal[
    "legacy_model_default",
    "frozen_full_train",
    "optimization_fold_validation",
]

ALL_EVIDENCE_PACKAGES: Final[tuple[EvidencePackage, ...]] = (
    "low_data",
    "synthetic_ood",
    "real_diagnostics",
    "real_domain_ood",
    "efficiency",
)


@dataclass(frozen=True, slots=True)
class P1P2Config:
    output_root: Path = Path(".omx/results/pac-tf-p1p2-confirmatory-20260711")
    protocol_path: Path = Path(".omx/protocols/pac_tf_confirmatory_20260711.json")
    selection_path: Path = DEFAULT_SELECTION_PATH
    unseen_root: Path = DEFAULT_UNSEEN_ROOT
    device: PACDevice = "auto"
    workers: int = 8
    total_slots: int = 16
    models: tuple[str, ...] = (
        "pac_tf",
        "tcn",
        "cnn1d",
        "gru",
        "lstm",
        "transformer",
        "mamba",
        "s4d",
        "inception_time",
    )
    packages: tuple[EvidencePackage, ...] = ALL_EVIDENCE_PACKAGES
    synthetic_estimand: SyntheticEstimand = "sequence"
    synthetic_target_params: int | None = None

    def __post_init__(self) -> None:
        if self.workers < 1 or self.total_slots < 1:
            message = "P1/P2 workers and total_slots must both be positive"
            raise ValueError(message)
        if not self.models or len(set(self.models)) != len(self.models):
            message = "P1/P2 models must be a non-empty unique tuple"
            raise ValueError(message)
        if not self.packages or len(set(self.packages)) != len(self.packages):
            message = "P1/P2 packages must be a non-empty unique tuple"
            raise ValueError(message)
        unknown_packages = set(self.packages) - set(ALL_EVIDENCE_PACKAGES)
        if unknown_packages:
            message = f"unsupported P1/P2 packages: {sorted(unknown_packages)}"
            raise ValueError(message)
        if self.synthetic_target_params is not None and self.synthetic_target_params < 1:
            message = "synthetic_target_params must be positive when provided"
            raise ValueError(message)


@dataclass(frozen=True, slots=True)
class P1P2Job:
    key: str
    package: EvidencePackage
    seed: int
    model: str = "pac_tf"
    reference_model: str = ""
    dataset: str = ""
    ratio: float = 1.0
    length: int = 0
    batch_size: int = 0
    grad_clip_norm: float = 0.0
    runtime: EvidenceRuntime = "train"
    slots: int = 2
    selection_trial: int = 0
    architecture_metadata_json: str = ""
    refit_epochs: int = 0
    learning_rate: float = 0.0
    weight_decay: float = 0.0
    protocol_sha256: str = ""
    p0_job_key: str = ""
    p0_checkpoint_path: str = ""
    p0_checkpoint_sha256: str = ""
    p0_metrics_json: str = ""
    ratio_one_fit_policy: RatioOneFitPolicy = "legacy_model_default"
    parameter_match_tolerance: float | None = None
    parameter_match_policy: str = ""
    parameter_match_default_tolerance: float | None = None
    parameter_match_max_tolerance: float | None = None
    parameter_match_exceptions_json: str = ""
    capacity_policy: str = ""
    reference_model_dim: int = 0
    selected_model_width: int = 0
    selection_source: str = ""
    selection_config_key: str = ""
    selection_artifact_sha256: str = ""
    synthetic_estimand: SyntheticEstimand = "sequence"
    synthetic_target_params: int | None = None
