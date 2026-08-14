# ruff: noqa: EM102, FBT003, PERF401, TRY003
"""Frozen Stage-1/Stage-2/Final contract for the broad ALPHABET benchmark.

The registry deliberately separates scientific scope from current execution
readiness.  Every requested dataset-model cell remains in the declared matrix;
missing adapters, runtime integrations, and restricted data become explicit
blockers instead of silent omissions.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path
from typing import Final, Literal, cast

from .pac_campaign_utils import write_once

Suite = Literal["irregular", "regular", "forecasting", "external"]
Access = Literal["public", "restricted"]
Stage = Literal["stage1", "stage2", "final"]
EvaluationSplit = Literal["validation", "test"]
JobClass = Literal["short", "medium", "long"]

DEFAULT_ROOT: Final = Path(".omx/results/alphabet-broad-benchmark-3gpu-20260727")
SEARCH_SEED: Final = 7
CONFIRMATION_SEEDS: Final = (11, 19)
FINAL_SEEDS: Final = (23, 31, 43, 47, 59)
TOP_K: Final = 6


@dataclass(frozen=True, slots=True)
class OptimizerRecipe:
    name: Literal["A", "B", "C"]
    learning_rate: float
    weight_decay: float
    effective_batch_size: int
    grad_clip_norm: float


OPTIMIZER_RECIPES: Final = (
    OptimizerRecipe("A", 1.0e-3, 1.0e-4, 64, 0.5),
    OptimizerRecipe("B", 3.0e-3, 1.0e-4, 64, 1.0),
    OptimizerRecipe("C", 1.0e-2, 1.0e-4, 64, 2.0),
)
ALPHABET_CAPACITIES: Final = (
    (32, 8),
    (32, 16),
    (64, 16),
    (64, 32),
    (128, 16),
    (128, 32),
)
BASELINE_WIDTHS: Final = (32, 64, 128)
STANDARD_SEQUENCE_MODELS: Final = (
    "cnn1d",
    "tcn",
    "transformer",
    "mamba",
    "s4d",
    "s5",
    "lru",
    "gru",
    "lstm",
)


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    key: str
    suite: Suite
    family: str
    endpoint: str
    access: Access
    source: str
    primary_variant: str
    execution_backend: str | None
    adapter_ready: bool
    estimated_seconds: float

    @property
    def cell_key_prefix(self) -> str:
        return f"{self.suite}:{self.key}:{self.endpoint}"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    key: str
    suite: Suite
    runtime_profile: str
    architecture_variants: tuple[str, str]
    implementation_ready: bool
    execution_backends: tuple[str, ...]
    runtime_factor: float = 1.0


@dataclass(frozen=True, slots=True)
class CampaignCell:
    suite: Suite
    dataset: str
    endpoint: str
    model: str
    runtime_profile: str
    blockers: tuple[str, ...]

    @property
    def key(self) -> str:
        return f"{self.suite}:{self.dataset}:{self.endpoint}:{self.model}"

    @property
    def implementation_ready(self) -> bool:
        return not self.blockers


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    candidate_id: str
    ordinal: int
    recipe: OptimizerRecipe
    width: int
    modes: int | None
    architecture: str


@dataclass(frozen=True, slots=True)
class BenchmarkJob:
    key: str
    stage: Stage
    suite: Suite
    dataset: str
    endpoint: str
    model: str
    candidate_id: str
    candidate_rank: int
    recipe: OptimizerRecipe
    width: int
    modes: int | None
    architecture: str
    split_seed: int
    train_seed: int
    epochs: int
    evaluation_split: EvaluationSplit
    official_test_accessed: bool
    runtime_profile: str
    comparison_group: str
    data_shard: str
    job_class: JobClass
    estimated_seconds: float
    estimated_peak_memory_mb: int
    microbatch_size: int
    gradient_accumulation_steps: int
    blockers: tuple[str, ...] = ()

    @property
    def runnable(self) -> bool:
        return not self.blockers

    @property
    def cell_key(self) -> str:
        return f"{self.suite}:{self.dataset}:{self.endpoint}:{self.model}"

    @property
    def config_key(self) -> str:
        return self.candidate_id

    def payload(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> BenchmarkJob:
        recipe = cast("dict[str, object]", payload["recipe"])
        return cls(
            key=str(payload["key"]),
            stage=cast("Stage", payload["stage"]),
            suite=cast("Suite", payload["suite"]),
            dataset=str(payload["dataset"]),
            endpoint=str(payload["endpoint"]),
            model=str(payload["model"]),
            candidate_id=str(payload["candidate_id"]),
            candidate_rank=int(cast("str | int", payload["candidate_rank"])),
            recipe=OptimizerRecipe(
                cast("Literal['A', 'B', 'C']", recipe["name"]),
                float(cast("str | int | float", recipe["learning_rate"])),
                float(cast("str | int | float", recipe["weight_decay"])),
                int(cast("str | int", recipe["effective_batch_size"])),
                float(cast("str | int | float", recipe["grad_clip_norm"])),
            ),
            width=int(cast("str | int", payload["width"])),
            modes=(
                None if payload.get("modes") is None else int(cast("str | int", payload["modes"]))
            ),
            architecture=str(payload["architecture"]),
            split_seed=int(cast("str | int", payload["split_seed"])),
            train_seed=int(cast("str | int", payload["train_seed"])),
            epochs=int(cast("str | int", payload["epochs"])),
            evaluation_split=cast("EvaluationSplit", payload["evaluation_split"]),
            official_test_accessed=bool(payload["official_test_accessed"]),
            runtime_profile=str(payload["runtime_profile"]),
            comparison_group=str(payload["comparison_group"]),
            data_shard=str(payload["data_shard"]),
            job_class=cast("JobClass", payload["job_class"]),
            estimated_seconds=float(cast("str | int | float", payload["estimated_seconds"])),
            estimated_peak_memory_mb=int(cast("str | int", payload["estimated_peak_memory_mb"])),
            microbatch_size=int(cast("str | int", payload["microbatch_size"])),
            gradient_accumulation_steps=int(
                cast("str | int", payload["gradient_accumulation_steps"])
            ),
            blockers=tuple(str(value) for value in cast("list[object]", payload["blockers"])),
        )


UCR_128: Final = (
    "ACSF1",
    "Adiac",
    "AllGestureWiimoteX",
    "AllGestureWiimoteY",
    "AllGestureWiimoteZ",
    "ArrowHead",
    "Beef",
    "BeetleFly",
    "BirdChicken",
    "BME",
    "Car",
    "CBF",
    "Chinatown",
    "ChlorineConcentration",
    "CinCECGTorso",
    "Coffee",
    "Computers",
    "CricketX",
    "CricketY",
    "CricketZ",
    "Crop",
    "DiatomSizeReduction",
    "DistalPhalanxOutlineAgeGroup",
    "DistalPhalanxOutlineCorrect",
    "DistalPhalanxTW",
    "DodgerLoopDay",
    "DodgerLoopGame",
    "DodgerLoopWeekend",
    "Earthquakes",
    "ECG200",
    "ECG5000",
    "ECGFiveDays",
    "ElectricDevices",
    "EOGHorizontalSignal",
    "EOGVerticalSignal",
    "EthanolLevel",
    "FaceAll",
    "FaceFour",
    "FacesUCR",
    "FiftyWords",
    "Fish",
    "FordA",
    "FordB",
    "FreezerRegularTrain",
    "FreezerSmallTrain",
    "Fungi",
    "GestureMidAirD1",
    "GestureMidAirD2",
    "GestureMidAirD3",
    "GesturePebbleZ1",
    "GesturePebbleZ2",
    "GunPoint",
    "GunPointAgeSpan",
    "GunPointMaleVersusFemale",
    "GunPointOldVersusYoung",
    "Ham",
    "HandOutlines",
    "Haptics",
    "Herring",
    "HouseTwenty",
    "InlineSkate",
    "InsectEPGRegularTrain",
    "InsectEPGSmallTrain",
    "InsectWingbeatSound",
    "ItalyPowerDemand",
    "LargeKitchenAppliances",
    "Lightning2",
    "Lightning7",
    "Mallat",
    "Meat",
    "MedicalImages",
    "MelbournePedestrian",
    "MiddlePhalanxOutlineAgeGroup",
    "MiddlePhalanxOutlineCorrect",
    "MiddlePhalanxTW",
    "MixedShapesRegularTrain",
    "MixedShapesSmallTrain",
    "MoteStrain",
    "NonInvasiveFetalECGThorax1",
    "NonInvasiveFetalECGThorax2",
    "OSULeaf",
    "OliveOil",
    "PLAID",
    "PhalangesOutlinesCorrect",
    "Phoneme",
    "PickupGestureWiimoteZ",
    "PigAirwayPressure",
    "PigArtPressure",
    "PigCVP",
    "Plane",
    "PowerCons",
    "ProximalPhalanxOutlineAgeGroup",
    "ProximalPhalanxOutlineCorrect",
    "ProximalPhalanxTW",
    "RefrigerationDevices",
    "Rock",
    "ScreenType",
    "SemgHandGenderCh2",
    "SemgHandMovementCh2",
    "SemgHandSubjectCh2",
    "ShakeGestureWiimoteZ",
    "ShapeletSim",
    "ShapesAll",
    "SmallKitchenAppliances",
    "SmoothSubspace",
    "SonyAIBORobotSurface1",
    "SonyAIBORobotSurface2",
    "StarLightCurves",
    "Strawberry",
    "SwedishLeaf",
    "Symbols",
    "SyntheticControl",
    "ToeSegmentation1",
    "ToeSegmentation2",
    "Trace",
    "TwoLeadECG",
    "TwoPatterns",
    "UMD",
    "UWaveGestureLibraryAll",
    "UWaveGestureLibraryX",
    "UWaveGestureLibraryY",
    "UWaveGestureLibraryZ",
    "Wafer",
    "Wine",
    "WordSynonyms",
    "Worms",
    "WormsTwoClass",
    "Yoga",
)

UEA_30: Final = (
    "ArticularyWordRecognition",
    "AtrialFibrillation",
    "BasicMotions",
    "CharacterTrajectories",
    "Cricket",
    "DuckDuckGeese",
    "EigenWorms",
    "Epilepsy",
    "ERing",
    "EthanolConcentration",
    "FaceDetection",
    "FingerMovements",
    "HandMovementDirection",
    "Handwriting",
    "Heartbeat",
    "InsectWingbeat",
    "JapaneseVowels",
    "Libras",
    "LSST",
    "MotorImagery",
    "NATOPS",
    "PEMS-SF",
    "PenDigits",
    "PhonemeSpectra",
    "RacketSports",
    "SelfRegulationSCP1",
    "SelfRegulationSCP2",
    "SpokenArabicDigits",
    "StandWalkJump",
    "UWaveGestureLibrary",
)

SPECTRAL_DATASETS: Final = (
    ("sleepedf-78", "sleep-stage-5", "public"),
    ("isruc-sleep", "sleep-stage", "public"),
    ("tuab", "normal-abnormal", "restricted"),
    ("tusz", "seizure-detection", "restricted"),
    ("chb-mit", "seizure-detection", "public"),
    ("bci-iv-2a", "motor-imagery-4", "public"),
    ("shhs", "sleep-stage", "restricted"),
)
VIBRATION_DATASETS: Final = (
    ("mfpt-bearing", "fault-classification"),
    ("paderborn-kat", "fault-classification"),
    ("xjtu-sy", "fault-classification"),
    ("ims-bearing", "fault-classification"),
)
ECG_DATASETS: Final = (
    ("chapman-shaoxing", "rhythm-11"),
    ("cpsc-2018", "arrhythmia-9"),
    ("icentia11k", "wearable-ecg-classification"),
)
IRREGULAR_DATASETS: Final = (
    ("physionet-2012", "mortality", "public", "physionet2012", True),
    ("physionet-2019", "early-sepsis", "public", "raindrop_fixed", True),
    ("human-activity", "activity-classification", "public", None, False),
    ("ushcn-daily", "interpolation-forecasting", "public", None, False),
    ("pam", "activity-8", "public", "raindrop_fixed", True),
    ("mimic-benchmark", "clinical-benchmark", "restricted", None, False),
    ("eicu", "clinical-benchmark", "restricted", None, False),
)
STANDARD_FORECAST_DATASETS: Final = (
    ("ettm1", "ettm", True),
    ("ettm2", "ettm", True),
    ("electricity", "ratio", True),
    ("weather", "ratio", True),
    ("etth1", "etth", True),
    ("etth2", "etth", True),
    ("traffic", "standard", True),
    ("ili", "standard", True),
    ("exchange-rate", "standard", True),
)
MONASH_CANONICAL_30: Final = (
    ("monash-m1", "monthly"),
    ("monash-m3", "monthly"),
    ("monash-m4", "monthly"),
    ("monash-tourism", "monthly"),
    ("monash-cif2016", "monthly"),
    ("monash-london-smart-meters", "without-missing"),
    ("monash-australian-electricity-demand", "half-hourly"),
    ("monash-wind-farms", "without-missing"),
    ("monash-dominick", "weekly"),
    ("monash-bitcoin", "without-missing"),
    ("monash-pedestrian-counts", "hourly"),
    ("monash-vehicle-trips", "without-missing"),
    ("monash-kdd-cup-2018", "without-missing"),
    ("monash-weather", "daily"),
    ("monash-nn5", "daily-without-missing"),
    ("monash-web-traffic", "daily-without-missing"),
    ("monash-solar", "10-minutes"),
    ("monash-electricity", "hourly"),
    ("monash-car-parts", "without-missing"),
    ("monash-fred-md", "monthly"),
    ("monash-san-francisco-traffic", "hourly"),
    ("monash-rideshare", "without-missing"),
    ("monash-hospital", "monthly"),
    ("monash-covid-deaths", "daily"),
    ("monash-temperature-rain", "without-missing"),
    ("monash-sunspot", "without-missing"),
    ("monash-saugeen-river-flow", "daily"),
    ("monash-us-births", "daily"),
    ("monash-solar-power", "4-seconds"),
    ("monash-wind-power", "4-seconds"),
)

_CLASSIFICATION_SOURCE: Final = "https://www.timeseriesclassification.com/"
_MONASH_SOURCE: Final = "https://forecastingdata.org/"


@cache
def dataset_registry() -> tuple[DatasetSpec, ...]:
    datasets: list[DatasetSpec] = [
        DatasetSpec(
            key=name,
            suite="regular",
            family="ucr-128",
            endpoint="classification",
            access="public",
            source=_CLASSIFICATION_SOURCE,
            primary_variant="official-train-test",
            execution_backend="balanced_hpo",
            adapter_ready=True,
            estimated_seconds=90.0,
        )
        for name in UCR_128
    ]
    datasets.extend(
        DatasetSpec(
            key=name,
            suite="regular",
            family="uea-30",
            endpoint="classification",
            access="public",
            source=_CLASSIFICATION_SOURCE,
            primary_variant="official-train-test",
            execution_backend=None,
            adapter_ready=False,
            estimated_seconds=600.0,
        )
        for name in UEA_30
    )
    datasets.extend(
        DatasetSpec(
            key=name,
            suite="regular",
            family="spectral",
            endpoint=endpoint,
            access=cast("Access", access),
            source="dataset-native official release",
            primary_variant="preregistered-official-split",
            execution_backend=None,
            adapter_ready=False,
            estimated_seconds=1_800.0,
        )
        for name, endpoint, access in SPECTRAL_DATASETS
    )
    datasets.extend(
        DatasetSpec(
            key=name,
            suite="regular",
            family="vibration",
            endpoint=endpoint,
            access="public",
            source="dataset-native official release",
            primary_variant="source-group-disjoint",
            execution_backend=None,
            adapter_ready=False,
            estimated_seconds=900.0,
        )
        for name, endpoint in VIBRATION_DATASETS
    )
    datasets.extend(
        DatasetSpec(
            key=name,
            suite="regular",
            family="ecg",
            endpoint=endpoint,
            access="public",
            source="dataset-native official release",
            primary_variant="patient-disjoint",
            execution_backend=None,
            adapter_ready=False,
            estimated_seconds=1_800.0,
        )
        for name, endpoint in ECG_DATASETS
    )
    datasets.extend(
        DatasetSpec(
            key=name,
            suite="irregular",
            family="irregular-standard",
            endpoint=endpoint,
            access=cast("Access", access),
            source="literature-standard official release",
            primary_variant="paper-standard-split",
            execution_backend=backend,
            adapter_ready=ready,
            estimated_seconds=2_400.0,
        )
        for name, endpoint, access, backend, ready in IRREGULAR_DATASETS
    )
    datasets.extend(
        DatasetSpec(
            key=name,
            suite="forecasting",
            family="standard-forecast",
            endpoint="forecasting",
            access="public",
            source="Autoformer/PatchTST standard corpus",
            primary_variant=variant,
            execution_backend="external" if ready else None,
            adapter_ready=ready,
            estimated_seconds=1_200.0,
        )
        for name, variant, ready in STANDARD_FORECAST_DATASETS
    )
    datasets.extend(
        DatasetSpec(
            key=name,
            suite="forecasting",
            family="monash-30",
            endpoint="forecasting",
            access="public",
            source=_MONASH_SOURCE,
            primary_variant=variant,
            execution_backend=None,
            adapter_ready=False,
            estimated_seconds=3_600.0,
        )
        for name, variant in MONASH_CANONICAL_30
    )
    return tuple(datasets)


def model_registry() -> dict[Suite, tuple[ModelSpec, ...]]:
    regular = (
        ModelSpec(
            "alphabet", "regular", "core", ("radial-log", "radial-log"), True, ("balanced_hpo",)
        ),
        ModelSpec("cnn1d", "regular", "core", ("d2-k3", "d4-k5"), True, ("balanced_hpo",)),
        ModelSpec("tcn", "regular", "core", ("d3-k3", "d5-k5"), True, ("balanced_hpo",)),
        ModelSpec(
            "transformer",
            "regular",
            "core",
            ("d1-h2", "d2-h4"),
            True,
            ("balanced_hpo",),
            1.5,
        ),
        ModelSpec("mamba", "regular", "mamba", ("s16-c3", "s32-c4"), True, ("balanced_hpo",), 1.3),
        ModelSpec("s4d", "regular", "core", ("d1-s16", "d3-s16"), True, ("balanced_hpo",), 1.4),
        ModelSpec("s5", "regular", "core", ("d1-s16", "d2-s32"), True, ("balanced_hpo",), 1.5),
        ModelSpec("lru", "regular", "core", ("d1-s16", "d2-s32"), True, ("balanced_hpo",), 1.5),
        ModelSpec("gru", "regular", "core", ("d1-s16", "d2-s32"), True, ("balanced_hpo",), 1.3),
        ModelSpec("lstm", "regular", "core", ("d1-s16", "d2-s32"), True, ("balanced_hpo",), 1.5),
    )
    irregular = (
        ModelSpec(
            "alphabet",
            "irregular",
            "core",
            ("radial-log", "radial-log"),
            True,
            ("physionet2012", "raindrop_fixed"),
        ),
        ModelSpec(
            "cnn1d",
            "irregular",
            "core",
            ("d2-k3", "d4-k5"),
            True,
            ("physionet2012", "raindrop_fixed"),
        ),
        ModelSpec(
            "tcn",
            "irregular",
            "core",
            ("d3-k3", "d5-k5"),
            True,
            ("physionet2012", "raindrop_fixed"),
        ),
        ModelSpec(
            "transformer",
            "irregular",
            "core",
            ("d1-h2", "d2-h4"),
            True,
            ("physionet2012", "raindrop_fixed"),
            1.5,
        ),
        ModelSpec(
            "mamba",
            "irregular",
            "mamba",
            ("s16-c3", "s32-c4"),
            True,
            ("physionet2012", "raindrop_fixed"),
            1.3,
        ),
        ModelSpec(
            "s4d",
            "irregular",
            "core",
            ("d1-s16", "d3-s16"),
            True,
            ("physionet2012", "raindrop_fixed"),
            1.4,
        ),
        ModelSpec(
            "s5",
            "irregular",
            "core",
            ("d1-s16", "d2-s32"),
            True,
            ("physionet2012", "raindrop_fixed"),
            1.5,
        ),
        ModelSpec(
            "lru",
            "irregular",
            "core",
            ("d1-s16", "d2-s32"),
            True,
            ("physionet2012", "raindrop_fixed"),
            1.5,
        ),
        ModelSpec(
            "gru",
            "irregular",
            "core",
            ("d1-s16", "d2-s32"),
            True,
            ("physionet2012", "raindrop_fixed"),
            1.3,
        ),
        ModelSpec(
            "lstm",
            "irregular",
            "core",
            ("d1-s16", "d2-s32"),
            True,
            ("physionet2012", "raindrop_fixed"),
            1.5,
        ),
    )
    forecasting = (
        ModelSpec(
            "alphabet", "forecasting", "core", ("radial-log", "radial-log"), True, ("external",)
        ),
        ModelSpec("cnn1d", "forecasting", "core", ("d2-k3", "d4-k5"), True, ("external",)),
        ModelSpec("tcn", "forecasting", "core", ("d3-k3", "d5-k5"), True, ("external",)),
        ModelSpec(
            "transformer",
            "forecasting",
            "core",
            ("d1-h2", "d2-h4"),
            True,
            ("external",),
            1.5,
        ),
        ModelSpec("mamba", "forecasting", "mamba", ("s16-c3", "s32-c4"), True, ("external",), 1.3),
        ModelSpec("s4d", "forecasting", "core", ("d1-s16", "d3-s16"), True, ("external",), 1.4),
        ModelSpec("s5", "forecasting", "core", ("d1-s16", "d2-s32"), True, ("external",), 1.5),
        ModelSpec("lru", "forecasting", "core", ("d1-s16", "d2-s32"), True, ("external",), 1.5),
        ModelSpec("gru", "forecasting", "core", ("d1-s16", "d2-s32"), True, ("external",), 1.3),
        ModelSpec("lstm", "forecasting", "core", ("d1-s16", "d2-s32"), True, ("external",), 1.5),
    )
    return {
        "regular": regular,
        "irregular": irregular,
        "forecasting": forecasting,
    }


@cache
def campaign_cells() -> tuple[CampaignCell, ...]:
    models = model_registry()
    cells: list[CampaignCell] = []
    for dataset in dataset_registry():
        for model in models[dataset.suite]:
            blockers: list[str] = []
            if dataset.access == "restricted":
                blockers.append("blocked_access")
            if not dataset.adapter_ready or dataset.execution_backend is None:
                blockers.append("blocked_dataset_adapter")
            if not model.implementation_ready:
                blockers.append("blocked_model_implementation")
            if (
                dataset.execution_backend is not None
                and model.implementation_ready
                and dataset.execution_backend not in model.execution_backends
            ):
                blockers.append("blocked_runner_integration")
            cells.append(
                CampaignCell(
                    suite=dataset.suite,
                    dataset=dataset.key,
                    endpoint=dataset.endpoint,
                    model=model.key,
                    runtime_profile=model.runtime_profile,
                    blockers=tuple(blockers),
                )
            )
    return tuple(cells)


@cache
def candidate_specs(model: ModelSpec) -> tuple[CandidateSpec, ...]:
    candidates: list[CandidateSpec] = []
    if model.key == "alphabet":
        for width, modes in ALPHABET_CAPACITIES:
            for recipe in OPTIMIZER_RECIPES:
                candidates.append(
                    CandidateSpec(
                        candidate_id=f"d{width}-m{modes}-recipe{recipe.name.lower()}",
                        ordinal=len(candidates),
                        recipe=recipe,
                        width=width,
                        modes=modes,
                        architecture="radial-log-r-affine",
                    )
                )
    else:
        for width in BASELINE_WIDTHS:
            for architecture in model.architecture_variants:
                for recipe in OPTIMIZER_RECIPES:
                    candidates.append(
                        CandidateSpec(
                            candidate_id=(f"w{width}-{architecture}-recipe{recipe.name.lower()}"),
                            ordinal=len(candidates),
                            recipe=recipe,
                            width=width,
                            modes=None,
                            architecture=architecture,
                        )
                    )
    if len(candidates) != 18 or len({item.candidate_id for item in candidates}) != 18:
        raise RuntimeError(f"{model.suite}/{model.key} does not define 18 unique candidates")
    return tuple(candidates)


@cache
def _model_lookup() -> dict[tuple[Suite, str], ModelSpec]:
    return {
        (suite, model.key): model for suite, models in model_registry().items() for model in models
    }


@cache
def _dataset_lookup() -> dict[tuple[Suite, str, str], DatasetSpec]:
    return {
        (dataset.suite, dataset.key, dataset.endpoint): dataset for dataset in dataset_registry()
    }


def _job_class(seconds: float) -> JobClass:
    if seconds < 180.0:
        return "short"
    if seconds <= 1_200.0:
        return "medium"
    return "long"


def _microbatch(width: int) -> tuple[int, int]:
    microbatch = 32 if width >= 128 else 64
    return microbatch, 64 // microbatch


def _make_job(
    cell: CampaignCell,
    candidate: CandidateSpec,
    *,
    stage: Stage,
    train_seed: int,
    candidate_rank: int,
) -> BenchmarkJob:
    dataset = _dataset_lookup()[(cell.suite, cell.dataset, cell.endpoint)]
    model = _model_lookup()[(cell.suite, cell.model)]
    capacity_factor = max(0.35, candidate.width / 64.0)
    stage_factor = {"stage1": 1.0, "stage2": 1.0, "final": 1.15}[stage]
    seconds = dataset.estimated_seconds * model.runtime_factor * capacity_factor * stage_factor
    microbatch, accumulation = _microbatch(candidate.width)
    official_test = stage == "final"
    return BenchmarkJob(
        key=(
            f"broad:{stage}:{cell.suite}:{cell.dataset}:{cell.endpoint}:"
            f"{cell.model}:{candidate.candidate_id}:split{SEARCH_SEED}:seed{train_seed}"
        ),
        stage=stage,
        suite=cell.suite,
        dataset=cell.dataset,
        endpoint=cell.endpoint,
        model=cell.model,
        candidate_id=candidate.candidate_id,
        candidate_rank=candidate_rank,
        recipe=candidate.recipe,
        width=candidate.width,
        modes=candidate.modes,
        architecture=candidate.architecture,
        split_seed=SEARCH_SEED,
        train_seed=train_seed,
        epochs=100 if cell.suite == "regular" else 60,
        evaluation_split="test" if official_test else "validation",
        official_test_accessed=official_test,
        runtime_profile=cell.runtime_profile,
        comparison_group=(
            f"{stage}:{cell.suite}:{cell.dataset}:{cell.endpoint}"
        ),
        data_shard=f"{cell.suite}:{cell.dataset}:{cell.endpoint}",
        job_class=_job_class(seconds),
        estimated_seconds=seconds,
        estimated_peak_memory_mb=2_048 + candidate.width * 32,
        microbatch_size=microbatch,
        gradient_accumulation_steps=accumulation,
        blockers=cell.blockers,
    )


@cache
def stage1_jobs() -> tuple[BenchmarkJob, ...]:
    models = _model_lookup()
    jobs = [
        _make_job(
            cell,
            candidate,
            stage="stage1",
            train_seed=SEARCH_SEED,
            candidate_rank=candidate.ordinal,
        )
        for cell in campaign_cells()
        for candidate in candidate_specs(models[(cell.suite, cell.model)])
    ]
    return tuple(jobs)


def stage2_jobs(
    selections: dict[str, tuple[str, ...]],
) -> tuple[BenchmarkJob, ...]:
    models = _model_lookup()
    jobs: list[BenchmarkJob] = []
    for cell in campaign_cells():
        selected = selections.get(cell.key)
        if selected is None or len(selected) != TOP_K or len(set(selected)) != TOP_K:
            raise ValueError(f"{cell.key} must select exactly {TOP_K} unique candidates")
        candidates = {
            candidate.candidate_id: candidate
            for candidate in candidate_specs(models[(cell.suite, cell.model)])
        }
        unknown = set(selected) - set(candidates)
        if unknown:
            raise ValueError(f"{cell.key} selected unknown candidates: {sorted(unknown)}")
        for rank, candidate_id in enumerate(selected, start=1):
            for seed in CONFIRMATION_SEEDS:
                jobs.append(
                    _make_job(
                        cell,
                        candidates[candidate_id],
                        stage="stage2",
                        train_seed=seed,
                        candidate_rank=rank,
                    )
                )
    return tuple(jobs)


def final_jobs(
    selections: dict[str, str],
) -> tuple[BenchmarkJob, ...]:
    models = _model_lookup()
    jobs: list[BenchmarkJob] = []
    for cell in campaign_cells():
        selected = selections.get(cell.key)
        candidates = {
            candidate.candidate_id: candidate
            for candidate in candidate_specs(models[(cell.suite, cell.model)])
        }
        if selected is None or selected not in candidates:
            raise ValueError(f"{cell.key} must freeze one known candidate")
        for seed in FINAL_SEEDS:
            jobs.append(
                _make_job(
                    cell,
                    candidates[selected],
                    stage="final",
                    train_seed=seed,
                    candidate_rank=1,
                )
            )
    return tuple(jobs)


def expected_counts() -> dict[str, int]:
    datasets = dataset_registry()
    cells = campaign_cells()
    return {
        "datasets": len(datasets),
        "irregular_datasets": sum(item.suite == "irregular" for item in datasets),
        "regular_datasets": sum(item.suite == "regular" for item in datasets),
        "forecasting_datasets": sum(item.suite == "forecasting" for item in datasets),
        "cells": len(cells),
        "implementation_ready_cells": sum(cell.implementation_ready for cell in cells),
        "blocked_cells": sum(not cell.implementation_ready for cell in cells),
        "stage1": len(cells) * 18,
        "stage2": len(cells) * TOP_K * len(CONFIRMATION_SEEDS),
        "final": len(cells) * len(FINAL_SEEDS),
        "total_fits": len(cells) * (18 + TOP_K * len(CONFIRMATION_SEEDS) + len(FINAL_SEEDS)),
    }


def audit_registry() -> dict[str, object]:
    datasets = dataset_registry()
    cells = campaign_cells()
    counts = expected_counts()
    dataset_keys = [(dataset.suite, dataset.key, dataset.endpoint) for dataset in datasets]
    cell_keys = [cell.key for cell in cells]
    stage1 = stage1_jobs()
    problems: list[str] = []
    if len(dataset_keys) != len(set(dataset_keys)):
        problems.append("duplicate dataset key")
    if len(cell_keys) != len(set(cell_keys)):
        problems.append("duplicate cell key")
    if len({job.key for job in stage1}) != len(stage1):
        problems.append("duplicate Stage-1 logical key")
    if any(job.official_test_accessed or job.evaluation_split != "validation" for job in stage1):
        problems.append("Stage 1 accesses TEST")
    if counts["datasets"] != 218:
        problems.append(f"expected 218 datasets, got {counts['datasets']}")
    if counts["cells"] != 2_180:
        problems.append(f"expected 2,180 cells, got {counts['cells']}")
    if counts["total_fits"] != 76_300:
        problems.append(f"expected 76,300 fits, got {counts['total_fits']}")
    return {
        "schema": "alphabet.broad_benchmark.registry_audit.v1",
        "ok": not problems,
        "problems": problems,
        "counts": counts,
        "blocker_counts": {
            blocker: sum(blocker in cell.blockers for cell in cells)
            for blocker in (
                "blocked_access",
                "blocked_dataset_adapter",
                "blocked_model_implementation",
                "blocked_runner_integration",
            )
        },
    }


def write_campaign_contract(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    audit = audit_registry()
    if not audit["ok"]:
        raise RuntimeError(f"broad benchmark registry audit failed: {audit['problems']}")
    datasets = dataset_registry()
    models = model_registry()
    cells = campaign_cells()
    payload: dict[str, object] = {
        "schema": "alphabet.broad_benchmark.contract.v1",
        "state": "prepared_not_released",
        "architecture": {
            "name": "ALPHABET",
            "implementation": "lnet.alphabet.Alphabet",
            "descriptor": "writer-reader radial-log R(0,1,2,4)",
            "head": "affine",
        },
        "seeds": {
            "stage1": [SEARCH_SEED],
            "stage2": list(CONFIRMATION_SEEDS),
            "final": list(FINAL_SEEDS),
        },
        "selection": {
            "stage1_candidates_per_cell": 18,
            "stage2_top_k": TOP_K,
            "final_candidates_per_cell": 1,
            "tie_break": "lexical candidate_id",
        },
        "test_access_policy": {
            "stage1": "forbidden",
            "stage2": "forbidden",
            "final": "allowed only after configuration freeze",
        },
        "datasets": [asdict(dataset) for dataset in datasets],
        "models": {
            suite: [asdict(model) for model in suite_models]
            for suite, suite_models in models.items()
        },
        "expected": expected_counts(),
        "registry_sha256": hashlib.sha256(
            json.dumps(
                {
                    "datasets": [asdict(dataset) for dataset in datasets],
                    "models": {
                        suite: [asdict(model) for model in suite_models]
                        for suite, suite_models in models.items()
                    },
                    "cells": [asdict(cell) for cell in cells],
                },
                sort_keys=True,
            ).encode()
        ).hexdigest(),
        "audit": audit,
    }
    write_once(
        root / "contract.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    blocked_rows = "".join(
        json.dumps(asdict(cell), sort_keys=True) + "\n"
        for cell in cells
        if not cell.implementation_ready
    )
    write_once(root / "blocked" / "cells.jsonl", blocked_rows)
    return payload


__all__ = [
    "ALPHABET_CAPACITIES",
    "BASELINE_WIDTHS",
    "CONFIRMATION_SEEDS",
    "DEFAULT_ROOT",
    "FINAL_SEEDS",
    "MONASH_CANONICAL_30",
    "OPTIMIZER_RECIPES",
    "SEARCH_SEED",
    "TOP_K",
    "UCR_128",
    "UEA_30",
    "BenchmarkJob",
    "CampaignCell",
    "CandidateSpec",
    "DatasetSpec",
    "ModelSpec",
    "audit_registry",
    "campaign_cells",
    "candidate_specs",
    "dataset_registry",
    "expected_counts",
    "final_jobs",
    "model_registry",
    "stage1_jobs",
    "stage2_jobs",
    "write_campaign_contract",
]
