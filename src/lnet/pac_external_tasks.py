# ruff: noqa: EM101, EM102, TRY003
from __future__ import annotations

import ast
import csv
import hashlib
import math
import wave
from dataclasses import dataclass, field
from importlib import import_module
from typing import TYPE_CHECKING, Final, Literal, assert_never

import torch
from torch import Tensor
from torch.nn import functional

if TYPE_CHECKING:
    from pathlib import Path

ExternalObjective = Literal["multiclass", "multilabel", "forecasting"]
ExternalInputEncoding = Literal["continuous", "tokens", "token_pair"]
ExternalDatasetName = Literal[
    "ptb-xl",
    "mit-bih",
    "cwru",
    "speech-commands",
    "pathfinder",
    "ettm1",
    "ettm2",
    "electricity",
    "weather",
    "etth1",
    "etth2",
    "traffic",
    "ili",
    "exchange-rate",
    "human-activity",
    "ushcn-daily",
    "lra-listops",
    "lra-text",
    "lra-retrieval",
    "lra-image",
    "sequential-mnist",
    "permuted-mnist",
    "sequential-cifar",
    "audioset-balanced",
]

PTBXL_CLASSES: Final = ("NORM", "MI", "STTC", "CD", "HYP")
MITBIH_CLASSES: Final = ("N", "S", "V", "F", "Q")
MITBIH_DS1: Final = (
    "101",
    "106",
    "108",
    "109",
    "112",
    "114",
    "115",
    "116",
    "118",
    "119",
    "122",
    "124",
    "201",
    "203",
    "205",
    "207",
    "208",
    "209",
    "215",
    "220",
    "223",
    "230",
)
MITBIH_DS2: Final = (
    "100",
    "103",
    "105",
    "111",
    "113",
    "117",
    "121",
    "123",
    "200",
    "202",
    "210",
    "212",
    "213",
    "214",
    "219",
    "221",
    "222",
    "228",
    "231",
    "232",
    "233",
    "234",
)
_MITBIH_SYMBOLS: Final = {
    "N": "N",
    "L": "N",
    "R": "N",
    "e": "N",
    "j": "N",
    "A": "S",
    "a": "S",
    "J": "S",
    "S": "S",
    "V": "V",
    "E": "V",
    "F": "F",
    "/": "Q",
    "f": "Q",
    "Q": "Q",
    "?": "Q",
}


class ExternalDatasetError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ExternalTemporalMetadata:
    """Optional interval and missingness tensors aligned with one task split."""

    time_delta: Tensor | None = None
    observation_mask: Tensor | None = None
    valid_mask: Tensor | None = None

    @property
    def is_empty(self) -> bool:
        return all(
            value is None for value in (self.time_delta, self.observation_mask, self.valid_mask)
        )

    def model_kwargs(self) -> dict[str, Tensor]:
        return {
            name: value
            for name, value in (
                ("time_delta", self.time_delta),
                ("observation_mask", self.observation_mask),
                ("valid_mask", self.valid_mask),
            )
            if value is not None
        }

    def index_select(self, indices: Tensor) -> ExternalTemporalMetadata:
        return ExternalTemporalMetadata(
            **{
                name: None if value is None else value[indices]
                for name, value in (
                    ("time_delta", self.time_delta),
                    ("observation_mask", self.observation_mask),
                    ("valid_mask", self.valid_mask),
                )
            }
        )

    def batch_slice(self, start: int, stop: int) -> ExternalTemporalMetadata:
        return ExternalTemporalMetadata(
            **{
                name: None if value is None else value[start:stop]
                for name, value in (
                    ("time_delta", self.time_delta),
                    ("observation_mask", self.observation_mask),
                    ("valid_mask", self.valid_mask),
                )
            }
        )

    def to(self, *, device: str | torch.device) -> ExternalTemporalMetadata:
        return ExternalTemporalMetadata(
            **{
                name: None if value is None else value.to(device=device)
                for name, value in (
                    ("time_delta", self.time_delta),
                    ("observation_mask", self.observation_mask),
                    ("valid_mask", self.valid_mask),
                )
            }
        )


@dataclass(frozen=True, slots=True)
class ExternalTask:
    name: str
    objective: ExternalObjective
    train_inputs: Tensor
    train_targets: Tensor
    validation_inputs: Tensor
    validation_targets: Tensor
    test_inputs: Tensor
    test_targets: Tensor
    output_dim: int
    class_names: tuple[str, ...] = ()
    train_groups: tuple[str, ...] = ()
    validation_groups: tuple[str, ...] = ()
    test_groups: tuple[str, ...] = ()
    sample_rate_hz: float | None = None
    characteristic_time_scale: float | None = None
    input_encoding: ExternalInputEncoding = "continuous"
    vocab_size: int | None = None
    train_metadata: ExternalTemporalMetadata = field(default_factory=ExternalTemporalMetadata)
    validation_metadata: ExternalTemporalMetadata = field(default_factory=ExternalTemporalMetadata)
    test_metadata: ExternalTemporalMetadata = field(default_factory=ExternalTemporalMetadata)

    def __post_init__(self) -> None:
        validate_external_task(self)

    @property
    def input_dim(self) -> int:
        return int(self.train_inputs.shape[-1])

    @property
    def sequence_length(self) -> int:
        return int(self.train_inputs.shape[1])

    @property
    def has_temporal_metadata(self) -> bool:
        return not all(
            metadata.is_empty
            for metadata in (
                self.train_metadata,
                self.validation_metadata,
                self.test_metadata,
            )
        )


@dataclass(frozen=True, slots=True)
class ExternalSelectionTask:
    """TRAIN/validation-only task container with no held-out tensor attributes."""

    name: str
    objective: ExternalObjective
    train_inputs: Tensor
    train_targets: Tensor
    validation_inputs: Tensor
    validation_targets: Tensor
    output_dim: int
    test_count: int
    selection_split_sha256: str
    class_names: tuple[str, ...] = ()
    train_groups: tuple[str, ...] = ()
    validation_groups: tuple[str, ...] = ()
    sample_rate_hz: float | None = None
    characteristic_time_scale: float | None = None
    input_encoding: ExternalInputEncoding = "continuous"
    vocab_size: int | None = None
    train_metadata: ExternalTemporalMetadata = field(default_factory=ExternalTemporalMetadata)
    validation_metadata: ExternalTemporalMetadata = field(default_factory=ExternalTemporalMetadata)

    def __post_init__(self) -> None:
        validate_external_selection_task(self)

    @property
    def input_dim(self) -> int:
        return int(self.train_inputs.shape[-1])

    @property
    def sequence_length(self) -> int:
        return int(self.train_inputs.shape[1])

    @property
    def has_temporal_metadata(self) -> bool:
        return not self.train_metadata.is_empty or not self.validation_metadata.is_empty


def _validate_temporal_metadata(  # noqa: C901, PLR0912
    task_name: str,
    split: str,
    inputs: Tensor,
    metadata: ExternalTemporalMetadata,
) -> None:
    prefix = f"{task_name} {split}"
    time_delta = metadata.time_delta
    observation_mask = metadata.observation_mask
    valid_mask = metadata.valid_mask
    if time_delta is not None:
        if time_delta.shape not in (inputs.shape[:2], (*inputs.shape[:2], 1)):
            raise ValueError(f"{prefix} time_delta must have shape [S,N] or [S,N,1]")
        if time_delta.is_complex() or not torch.isfinite(time_delta).all():
            raise ValueError(f"{prefix} time_delta must contain finite real values")
        if bool((time_delta < 0).any()):
            raise ValueError(f"{prefix} time_delta must be non-negative")
    if observation_mask is not None:
        allowed_shapes = (
            inputs.shape[:2],
            (*inputs.shape[:2], 1),
            inputs.shape,
        )
        if observation_mask.shape not in allowed_shapes:
            raise ValueError(
                f"{prefix} observation_mask must have shape [S,N], [S,N,1], or [S,N,C]"
            )
        if observation_mask.is_complex() or not torch.isfinite(observation_mask).all():
            raise ValueError(f"{prefix} observation_mask must contain finite real values")
        if bool(((observation_mask < 0) | (observation_mask > 1)).any()):
            raise ValueError(f"{prefix} observation_mask must lie in [0,1]")
    if valid_mask is not None:
        if valid_mask.shape not in (inputs.shape[:2], (*inputs.shape[:2], 1)):
            raise ValueError(f"{prefix} valid_mask must have shape [S,N] or [S,N,1]")
        if valid_mask.is_complex() or not torch.isfinite(valid_mask).all():
            raise ValueError(f"{prefix} valid_mask must contain finite real values")
        if bool(((valid_mask < 0) | (valid_mask > 1)).any()):
            raise ValueError(f"{prefix} valid_mask must lie in [0,1]")
    if observation_mask is not None and valid_mask is not None:
        active_valid = valid_mask
        if observation_mask.ndim == 3 and active_valid.ndim == 2:
            active_valid = active_valid.unsqueeze(-1)
        if bool((observation_mask > active_valid).any()):
            raise ValueError(f"{prefix} observation_mask cannot mark padded events as observed")


def validate_external_task(task: ExternalTask) -> None:  # noqa: C901, PLR0912
    if task.characteristic_time_scale is not None and (
        not math.isfinite(task.characteristic_time_scale)
        or task.characteristic_time_scale <= 0
    ):
        raise ValueError("characteristic_time_scale must be finite and positive")
    splits = (
        ("train", task.train_inputs, task.train_targets, task.train_metadata),
        (
            "validation",
            task.validation_inputs,
            task.validation_targets,
            task.validation_metadata,
        ),
        ("test", task.test_inputs, task.test_targets, task.test_metadata),
    )
    for split, inputs, targets, metadata in splits:
        if inputs.ndim != 3 or inputs.shape[0] == 0:
            message = f"{task.name} {split} inputs must be non-empty [S,N,C]"
            raise ValueError(message)
        if targets.shape[0] != inputs.shape[0]:
            message = f"{task.name} {split} input/target counts differ"
            raise ValueError(message)
        if not torch.isfinite(inputs).all():
            message = f"{task.name} {split} inputs contain non-finite values"
            raise ValueError(message)
        _validate_temporal_metadata(task.name, split, inputs, metadata)
    if task.input_encoding != "continuous" and task.has_temporal_metadata:
        raise ValueError("temporal metadata is supported only for continuous inputs")
    if task.input_encoding == "continuous":
        if task.vocab_size is not None:
            raise ValueError("continuous tasks cannot declare vocab_size")
    else:
        expected_channels = 1 if task.input_encoding == "tokens" else 2
        if task.input_dim != expected_channels:
            message = (
                f"{task.input_encoding} inputs must have {expected_channels} channel(s), "
                f"got {task.input_dim}"
            )
            raise ValueError(message)
        if task.vocab_size is None or task.vocab_size < 2:
            raise ValueError("token tasks require vocab_size >= 2")
        for split, inputs, _, _ in splits:
            if inputs.is_floating_point() or inputs.is_complex():
                raise ValueError(f"{task.name} {split} token inputs must use an integer dtype")
            if int(inputs.min().item()) < 0 or int(inputs.max().item()) >= task.vocab_size:
                raise ValueError(f"{task.name} {split} token id is outside vocab_size")
    if task.objective == "multiclass":
        if any(targets.ndim != 1 or targets.dtype != torch.long for _, _, targets, _ in splits):
            message = "multiclass targets must be rank-1 torch.long tensors"
            raise ValueError(message)
    elif task.objective == "multilabel":
        if any(
            targets.ndim != 2 or targets.shape[1] != task.output_dim for _, _, targets, _ in splits
        ):
            message = "multilabel targets must be [S, output_dim]"
            raise ValueError(message)
    elif task.objective == "forecasting":
        if any(
            targets.ndim != 3 or targets.shape[1] * targets.shape[2] != task.output_dim
            for _, _, targets, _ in splits
        ):
            message = "forecast targets must be [S, horizon, channels]"
            raise ValueError(message)
    else:
        assert_never(task.objective)
    _validate_group_disjointness(task)


def validate_external_selection_task(  # noqa: C901, PLR0912
    task: ExternalSelectionTask,
) -> None:
    if task.characteristic_time_scale is not None and (
        not math.isfinite(task.characteristic_time_scale)
        or task.characteristic_time_scale <= 0
    ):
        raise ValueError("characteristic_time_scale must be finite and positive")
    if task.test_count < 1:
        raise ValueError("selection artifact must record a positive held-out count")
    if len(task.selection_split_sha256) != 64:
        raise ValueError("selection artifact must record a SHA-256 split digest")
    splits = (
        ("train", task.train_inputs, task.train_targets, task.train_metadata),
        (
            "validation",
            task.validation_inputs,
            task.validation_targets,
            task.validation_metadata,
        ),
    )
    for split, inputs, targets, metadata in splits:
        if inputs.ndim != 3 or inputs.shape[0] == 0:
            raise ValueError(f"{task.name} {split} inputs must be non-empty [S,N,C]")
        if targets.shape[0] != inputs.shape[0]:
            raise ValueError(f"{task.name} {split} input/target counts differ")
        if not torch.isfinite(inputs).all():
            raise ValueError(f"{task.name} {split} inputs contain non-finite values")
        _validate_temporal_metadata(task.name, split, inputs, metadata)
    if task.input_encoding != "continuous" and task.has_temporal_metadata:
        raise ValueError("temporal metadata is supported only for continuous inputs")
    if task.objective == "multiclass":
        if any(targets.ndim != 1 or targets.dtype != torch.long for _, _, targets, _ in splits):
            raise ValueError("multiclass targets must be rank-1 torch.long tensors")
    elif task.objective == "multilabel":
        if any(
            targets.ndim != 2 or targets.shape[1] != task.output_dim for _, _, targets, _ in splits
        ):
            raise ValueError("multilabel targets must be [S, output_dim]")
    elif task.objective == "forecasting":
        if any(
            targets.ndim != 3 or targets.shape[1] * targets.shape[2] != task.output_dim
            for _, _, targets, _ in splits
        ):
            raise ValueError("forecast targets must be [S, horizon, channels]")
    else:
        assert_never(task.objective)


def save_prepared_task(task: ExternalTask, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 3,
            "name": task.name,
            "objective": task.objective,
            "train_inputs": task.train_inputs,
            "train_targets": task.train_targets,
            "validation_inputs": task.validation_inputs,
            "validation_targets": task.validation_targets,
            "test_inputs": task.test_inputs,
            "test_targets": task.test_targets,
            "output_dim": task.output_dim,
            "class_names": task.class_names,
            "train_groups": task.train_groups,
            "validation_groups": task.validation_groups,
            "test_groups": task.test_groups,
            "sample_rate_hz": task.sample_rate_hz,
            "characteristic_time_scale": task.characteristic_time_scale,
            "input_encoding": task.input_encoding,
            "vocab_size": task.vocab_size,
            "temporal_metadata": {
                "train": _temporal_metadata_payload(task.train_metadata),
                "validation": _temporal_metadata_payload(task.validation_metadata),
                "test": _temporal_metadata_payload(task.test_metadata),
            },
        },
        path,
    )


def write_external_selection_task(task: ExternalTask, path: Path) -> None:
    """Materialize an auditable artifact that physically omits held-out tensors."""
    split_sha256 = _selection_split_sha256(task)
    payload: dict[str, object] = {
        "format_version": 2,
        "selection_only": True,
        "name": task.name,
        "objective": task.objective,
        "train_inputs": task.train_inputs,
        "train_targets": task.train_targets,
        "validation_inputs": task.validation_inputs,
        "validation_targets": task.validation_targets,
        "output_dim": task.output_dim,
        "test_count": int(task.test_inputs.shape[0]),
        "selection_split_sha256": split_sha256,
        "class_names": task.class_names,
        "train_groups": task.train_groups,
        "validation_groups": task.validation_groups,
        "sample_rate_hz": task.sample_rate_hz,
        "characteristic_time_scale": task.characteristic_time_scale,
        "input_encoding": task.input_encoding,
        "vocab_size": task.vocab_size,
        "temporal_metadata": {
            "train": _temporal_metadata_payload(task.train_metadata),
            "validation": _temporal_metadata_payload(task.validation_metadata),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_external_selection_task(name: str, root: Path) -> ExternalSelectionTask:
    """Load a mandatory TRAIN/validation-only snapshot; never fall back to a full task."""
    path = root / "selection-only" / f"{name}.pt"
    if not path.exists():
        raise ExternalDatasetError(
            f"selection-only artifact does not exist: {path}; prepare it before model selection"
        )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or payload.get("format_version") not in (1, 2)
        or payload.get("selection_only") is not True
    ):
        raise ExternalDatasetError(f"unsupported selection-only task format: {path}")
    forbidden = {"test_inputs", "test_targets", "test_groups"}.intersection(payload)
    temporal_payload = payload.get("temporal_metadata")
    if isinstance(temporal_payload, dict) and "test" in temporal_payload:
        forbidden.add("temporal_metadata.test")
    if forbidden:
        raise ExternalDatasetError(
            f"selection-only artifact contains forbidden held-out fields: {sorted(forbidden)}"
        )
    objective = str(payload["objective"])
    if objective not in ("multiclass", "multilabel", "forecasting"):
        raise ExternalDatasetError(f"invalid objective in {path}: {objective}")
    return ExternalSelectionTask(
        name=str(payload["name"]),
        objective=objective,
        train_inputs=_tensor(payload, "train_inputs"),
        train_targets=_tensor(payload, "train_targets"),
        validation_inputs=_tensor(payload, "validation_inputs"),
        validation_targets=_tensor(payload, "validation_targets"),
        output_dim=int(payload["output_dim"]),
        test_count=int(payload["test_count"]),
        selection_split_sha256=str(payload["selection_split_sha256"]),
        class_names=tuple(str(value) for value in payload.get("class_names", ())),
        train_groups=tuple(str(value) for value in payload.get("train_groups", ())),
        validation_groups=tuple(str(value) for value in payload.get("validation_groups", ())),
        sample_rate_hz=_optional_float(payload.get("sample_rate_hz")),
        characteristic_time_scale=_optional_float(
            payload.get("characteristic_time_scale")
        ),
        input_encoding=_input_encoding(payload.get("input_encoding", "continuous"), path),
        vocab_size=_optional_int(payload.get("vocab_size")),
        train_metadata=_temporal_metadata_from_payload(payload, "train", path),
        validation_metadata=_temporal_metadata_from_payload(payload, "validation", path),
    )


def load_prepared_task(path: Path) -> ExternalTask:
    if not path.exists():
        raise ExternalDatasetError(f"prepared task does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format_version") not in (1, 2, 3):
        raise ExternalDatasetError(f"unsupported prepared task format: {path}")
    objective = str(payload["objective"])
    if objective not in ("multiclass", "multilabel", "forecasting"):
        raise ExternalDatasetError(f"invalid objective in {path}: {objective}")
    return ExternalTask(
        name=str(payload["name"]),
        objective=objective,
        train_inputs=_tensor(payload, "train_inputs"),
        train_targets=_tensor(payload, "train_targets"),
        validation_inputs=_tensor(payload, "validation_inputs"),
        validation_targets=_tensor(payload, "validation_targets"),
        test_inputs=_tensor(payload, "test_inputs"),
        test_targets=_tensor(payload, "test_targets"),
        output_dim=int(payload["output_dim"]),
        class_names=tuple(str(value) for value in payload.get("class_names", ())),
        train_groups=tuple(str(value) for value in payload.get("train_groups", ())),
        validation_groups=tuple(str(value) for value in payload.get("validation_groups", ())),
        test_groups=tuple(str(value) for value in payload.get("test_groups", ())),
        sample_rate_hz=_optional_float(payload.get("sample_rate_hz")),
        characteristic_time_scale=_optional_float(
            payload.get("characteristic_time_scale")
        ),
        input_encoding=_input_encoding(payload.get("input_encoding", "continuous"), path),
        vocab_size=_optional_int(payload.get("vocab_size")),
        train_metadata=_temporal_metadata_from_payload(payload, "train", path),
        validation_metadata=_temporal_metadata_from_payload(payload, "validation", path),
        test_metadata=_temporal_metadata_from_payload(payload, "test", path),
    )


def load_external_task(  # noqa: C901, PLR0911, PLR0912
    name: ExternalDatasetName,
    root: Path,
    *,
    mitbih_beat_length: int = 256,
    cwru_window_length: int = 2048,
    forecast_context_length: int | None = None,
    prediction_length: int | None = None,
) -> ExternalTask:
    prepared = root / f"{name}.pt"
    if prepared.exists():
        return load_prepared_task(prepared)
    match name:
        case "ptb-xl":
            return load_ptbxl(root / "ptb-xl")
        case "mit-bih":
            return load_mitbih(root / "mit-bih", beat_length=mitbih_beat_length)
        case "cwru":
            return load_cwru(root / "cwru", window_length=cwru_window_length)
        case "speech-commands":
            return load_speech_commands(root / "speech-commands")
        case "pathfinder":
            return load_pathfinder(root / "pathfinder")
        case "ettm1" | "ettm2" | "etth1" | "etth2" | "traffic" | "ili" | "exchange-rate":
            default_context = 36 if name == "ili" else 96
            default_prediction = 24 if name == "ili" else 96
            filename = {
                "exchange-rate": "exchange-rate.csv",
            }.get(name, f"{name}.csv")
            split_kind: Literal["etth", "ettm", "ratio"]
            if name in ("ettm1", "ettm2"):
                split_kind = "ettm"
            elif name in ("etth1", "etth2"):
                split_kind = "etth"
            else:
                split_kind = "ratio"
            return load_forecasting_csv(
                root / "forecasting" / filename,
                name=name,
                context_length=(
                    default_context
                    if forecast_context_length is None
                    else forecast_context_length
                ),
                prediction_length=(
                    default_prediction if prediction_length is None else prediction_length
                ),
                split_kind=split_kind,
            )
        case "electricity" | "weather":
            return load_forecasting_csv(
                root / "forecasting" / f"{name}.csv",
                name=name,
                context_length=96 if forecast_context_length is None else forecast_context_length,
                prediction_length=96 if prediction_length is None else prediction_length,
                split_kind="ratio",
            )
        case (
            "lra-listops"
            | "lra-text"
            | "lra-retrieval"
            | "lra-image"
            | "sequential-mnist"
            | "permuted-mnist"
            | "sequential-cifar"
            | "audioset-balanced"
            | "human-activity"
            | "ushcn-daily"
        ):
            message = f"{name} requires a prepared task at {prepared}; run "
            raise ExternalDatasetError(message + "scripts/prepare_pac_additional_tasks.py")
        case unreachable:
            assert_never(unreachable)


def load_ptbxl(root: Path, *, sampling_rate: Literal[100, 500] = 100) -> ExternalTask:
    metadata_path = root / "ptbxl_database.csv"
    statements_path = root / "scp_statements.csv"
    _require_files(metadata_path, statements_path)
    code_to_class = _ptbxl_diagnostic_classes(statements_path)
    rows = list(csv.DictReader(metadata_path.open(encoding="utf-8", newline="")))
    records: list[tuple[dict[str, str], Tensor, Tensor]] = []
    filename_key = "filename_lr" if sampling_rate == 100 else "filename_hr"
    for row in rows:
        labels = torch.zeros(len(PTBXL_CLASSES), dtype=torch.float32)
        scp_codes = ast.literal_eval(row["scp_codes"])
        for code, likelihood in scp_codes.items():
            diagnostic_class = code_to_class.get(str(code))
            if diagnostic_class is not None and float(likelihood) > 0.0:
                labels[PTBXL_CLASSES.index(diagnostic_class)] = 1.0
        if not labels.any():
            continue
        signal = _read_wfdb_format16(root / row[filename_key])
        records.append((row, signal, labels))
    if not records:
        raise ExternalDatasetError(f"no labeled PTB-XL records found under {root}")
    expected_shape = records[0][1].shape
    if any(signal.shape != expected_shape for _, signal, _ in records):
        raise ExternalDatasetError("PTB-XL records do not share a fixed shape")
    inputs = torch.stack([signal for _, signal, _ in records])
    targets = torch.stack([labels for _, _, labels in records])
    folds = torch.tensor([int(row["strat_fold"]) for row, _, _ in records])
    patients = tuple(str(row["patient_id"]) for row, _, _ in records)
    train = folds <= 8
    validation = folds == 9
    test = folds == 10
    inputs = _normalize_from_train(inputs, train)
    return _task_from_masks(
        "ptb-xl",
        "multilabel",
        inputs,
        targets,
        train,
        validation,
        test,
        len(PTBXL_CLASSES),
        PTBXL_CLASSES,
        patients,
        float(sampling_rate),
    )


def load_mitbih(root: Path, *, beat_length: int = 256) -> ExternalTask:
    try:
        wfdb = import_module("wfdb")
    except ModuleNotFoundError as error:
        message = (
            "MIT-BIH raw loading requires the optional 'wfdb' package; alternatively place "
            f"a prepared bundle at {root.parent / 'mit-bih.pt'}"
        )
        raise ExternalDatasetError(message) from error
    if beat_length < 8:
        raise ValueError("beat_length must be at least 8")
    validation_records = MITBIH_DS1[-4:]
    train_records = MITBIH_DS1[:-4]
    signals: list[Tensor] = []
    labels: list[int] = []
    splits: list[str] = []
    groups: list[str] = []
    left = beat_length // 3
    right = beat_length - left
    for record in (*MITBIH_DS1, *MITBIH_DS2):
        record_path = str(root / record)
        signal_values, _ = wfdb.rdsamp(record_path)
        annotations = wfdb.rdann(record_path, "atr")
        signal = torch.as_tensor(signal_values, dtype=torch.float32)
        split = (
            "validation"
            if record in validation_records
            else "train"
            if record in train_records
            else "test"
        )
        for sample, symbol in zip(annotations.sample, annotations.symbol, strict=True):
            mapped = _MITBIH_SYMBOLS.get(str(symbol))
            if mapped is None or sample < left or sample + right > signal.shape[0]:
                continue
            signals.append(signal[sample - left : sample + right])
            labels.append(MITBIH_CLASSES.index(mapped))
            splits.append(split)
            groups.append(record)
    return _split_classification_records(
        "mit-bih",
        torch.stack(signals),
        torch.tensor(labels, dtype=torch.long),
        splits,
        groups,
        MITBIH_CLASSES,
        360.0,
    )


def load_cwru(
    root: Path,
    *,
    window_length: int = 2048,
    window_stride: int | None = None,
) -> ExternalTask:
    manifest_path = root / "manifest.csv"
    _require_files(manifest_path)
    scipy_io = import_module("scipy.io")
    stride = window_stride or window_length
    signals: list[Tensor] = []
    labels: list[str] = []
    splits: list[str] = []
    groups: list[str] = []
    rows = list(csv.DictReader(manifest_path.open(encoding="utf-8", newline="")))
    class_names = tuple(sorted({row["label"] for row in rows}))
    for row in rows:
        split = _checked_split(row["split"])
        path = root / row["path"]
        payload = scipy_io.loadmat(path)
        key = row.get("signal_key") or _cwru_signal_key(payload)
        raw_signal = torch.as_tensor(payload[key], dtype=torch.float32).reshape(-1, 1)
        for start in range(0, raw_signal.shape[0] - window_length + 1, stride):
            signals.append(raw_signal[start : start + window_length])
            labels.append(row["label"])
            splits.append(split)
            groups.append(row.get("group") or row["path"])
    encoded = torch.tensor([class_names.index(label) for label in labels], dtype=torch.long)
    return _split_classification_records(
        "cwru",
        torch.stack(signals),
        encoded,
        splits,
        groups,
        class_names,
        None,
    )


def load_speech_commands(
    root: Path,
    *,
    target_rate: int = 16_000,
    patch_samples: int = 16,
) -> ExternalTask:
    validation_files = _path_set(root / "validation_list.txt")
    test_files = _path_set(root / "testing_list.txt")
    paths = sorted(path for path in root.glob("*/*.wav") if not path.parent.name.startswith("_"))
    if not paths:
        raise ExternalDatasetError(f"no Speech Commands wav files found under {root}")
    class_names = tuple(sorted({path.parent.name for path in paths}))
    inputs: list[Tensor] = []
    labels: list[int] = []
    splits: list[str] = []
    groups: list[str] = []
    for path in paths:
        relative = path.relative_to(root).as_posix()
        signal, sample_rate = _read_wave(path)
        signal = _resample_1d(signal, sample_rate, target_rate)
        signal = _pad_or_trim(signal, target_rate)
        if target_rate % patch_samples != 0:
            message = "target_rate must be divisible by patch_samples"
            raise ValueError(message)
        signal = signal.reshape(-1, patch_samples)
        inputs.append(signal)
        labels.append(class_names.index(path.parent.name))
        splits.append(
            "validation"
            if relative in validation_files
            else "test"
            if relative in test_files
            else "train"
        )
        groups.append(path.stem.split("_nohash_", maxsplit=1)[0])
    return _split_classification_records(
        "speech-commands",
        torch.stack(inputs),
        torch.tensor(labels, dtype=torch.long),
        splits,
        groups,
        class_names,
        float(target_rate),
    )


def load_pathfinder(root: Path) -> ExternalTask:
    manifest_path = root / "manifest.csv"
    _require_files(manifest_path)
    rows = list(csv.DictReader(manifest_path.open(encoding="utf-8", newline="")))
    inputs: list[Tensor] = []
    labels: list[int] = []
    splits: list[str] = []
    groups: list[str] = []
    for row in rows:
        path = root / row["path"]
        image = _read_pathfinder_image(path)
        inputs.append(image.reshape(-1, 1))
        labels.append(int(row["label"]))
        splits.append(_checked_split(row["split"]))
        groups.append(row.get("group") or row["path"])
    return _split_classification_records(
        "pathfinder",
        torch.stack(inputs),
        torch.tensor(labels, dtype=torch.long),
        splits,
        groups,
        ("disconnected", "connected"),
        None,
        enforce_group_disjointness=False,
    )


def load_forecasting_csv(
    path: Path,
    *,
    name: str,
    context_length: int,
    prediction_length: int,
    split_kind: Literal["etth", "ettm", "ratio"],
) -> ExternalTask:
    _require_files(path)
    if context_length < 1 or prediction_length < 1:
        raise ValueError("forecast context and prediction lengths must be positive")
    rows = list(csv.reader(path.open(encoding="utf-8", newline="")))
    if len(rows) < 4:
        raise ExternalDatasetError(f"forecasting CSV is empty: {path}")
    header = rows[0]
    value_indices = [index for index, column in enumerate(header) if column.lower() != "date"]
    values = torch.tensor(
        [[float(row[index]) for index in value_indices] for row in rows[1:]],
        dtype=torch.float32,
    )
    train_end, validation_end, test_end = _forecast_boundaries(values.shape[0], split_kind)
    mean = values[:train_end].mean(dim=0, keepdim=True)
    std = values[:train_end].std(dim=0, keepdim=True).clamp_min(1.0e-6)
    values = (values - mean) / std
    train_x, train_y = _forecast_split_windows(
        values, 0, train_end, context_length, prediction_length
    )
    validation_x, validation_y = _forecast_split_windows(
        values, train_end, validation_end, context_length, prediction_length
    )
    test_x, test_y = _forecast_split_windows(
        values, validation_end, test_end, context_length, prediction_length
    )
    return ExternalTask(
        name=name,
        objective="forecasting",
        train_inputs=train_x,
        train_targets=train_y,
        validation_inputs=validation_x,
        validation_targets=validation_y,
        test_inputs=test_x,
        test_targets=test_y,
        output_dim=prediction_length * len(value_indices),
        class_names=tuple(header[index] for index in value_indices),
    )


def synthetic_external_task(
    objective: ExternalObjective,
    *,
    seed: int = 7,
    sequence_length: int = 32,
    input_dim: int = 2,
) -> ExternalTask:
    generator = torch.Generator().manual_seed(seed)
    counts = (12, 6, 6)
    inputs = tuple(
        torch.randn(count, sequence_length, input_dim, generator=generator) for count in counts
    )
    if objective == "multiclass":
        targets = tuple((values.mean(dim=(1, 2)) > 0).to(torch.long) for values in inputs)
        return ExternalTask(
            "synthetic-multiclass",
            objective,
            inputs[0],
            targets[0],
            inputs[1],
            targets[1],
            inputs[2],
            targets[2],
            2,
            ("negative", "positive"),
        )
    if objective == "multilabel":
        targets = tuple(
            torch.stack(
                (
                    (values[..., 0].mean(dim=1) > 0),
                    (values[..., -1].square().mean(dim=1) > 1),
                ),
                dim=-1,
            ).to(torch.float32)
            for values in inputs
        )
        return ExternalTask(
            "synthetic-multilabel",
            objective,
            inputs[0],
            targets[0],
            inputs[1],
            targets[1],
            inputs[2],
            targets[2],
            2,
            ("mean", "energy"),
        )
    if objective == "forecasting":
        targets = tuple(values[:, -4:].clone() for values in inputs)
        return ExternalTask(
            "synthetic-forecasting",
            objective,
            inputs[0],
            targets[0],
            inputs[1],
            targets[1],
            inputs[2],
            targets[2],
            4 * input_dim,
        )
    assert_never(objective)


def _validate_group_disjointness(task: ExternalTask) -> None:
    groups = (task.train_groups, task.validation_groups, task.test_groups)
    if not any(groups):
        return
    expected_counts = (
        task.train_inputs.shape[0],
        task.validation_inputs.shape[0],
        task.test_inputs.shape[0],
    )
    if any(
        group and len(group) != count for group, count in zip(groups, expected_counts, strict=True)
    ):
        raise ValueError("group identifiers must be empty or match their split sample count")
    sets = tuple(set(group) for group in groups)
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise ValueError(f"{task.name} contains group leakage across splits")


def _task_from_masks(
    name: str,
    objective: ExternalObjective,
    inputs: Tensor,
    targets: Tensor,
    train: Tensor,
    validation: Tensor,
    test: Tensor,
    output_dim: int,
    class_names: tuple[str, ...],
    groups: tuple[str, ...],
    sample_rate_hz: float | None,
) -> ExternalTask:
    indexed_groups = tuple(
        tuple(group for group, selected in zip(groups, mask.tolist(), strict=True) if selected)
        for mask in (train, validation, test)
    )
    return ExternalTask(
        name,
        objective,
        inputs[train],
        targets[train],
        inputs[validation],
        targets[validation],
        inputs[test],
        targets[test],
        output_dim,
        class_names,
        indexed_groups[0],
        indexed_groups[1],
        indexed_groups[2],
        sample_rate_hz,
    )


def _split_classification_records(
    name: str,
    inputs: Tensor,
    targets: Tensor,
    splits: list[str],
    groups: list[str],
    class_names: tuple[str, ...],
    sample_rate_hz: float | None,
    *,
    enforce_group_disjointness: bool = True,
) -> ExternalTask:
    masks = tuple(
        torch.tensor([value == split for value in splits])
        for split in ("train", "validation", "test")
    )
    inputs = _normalize_from_train(inputs, masks[0])
    task_groups = tuple(
        tuple(group for group, selected in zip(groups, mask.tolist(), strict=True) if selected)
        for mask in masks
    )
    if not enforce_group_disjointness:
        task_groups = ((), (), ())
    return ExternalTask(
        name,
        "multiclass",
        inputs[masks[0]],
        targets[masks[0]],
        inputs[masks[1]],
        targets[masks[1]],
        inputs[masks[2]],
        targets[masks[2]],
        len(class_names),
        class_names,
        task_groups[0],
        task_groups[1],
        task_groups[2],
        sample_rate_hz,
    )


def _normalize_from_train(inputs: Tensor, train_mask: Tensor) -> Tensor:
    train_values = inputs[train_mask]
    mean = train_values.mean(dim=(0, 1), keepdim=True)
    std = train_values.std(dim=(0, 1), keepdim=True).clamp_min(1.0e-6)
    return (inputs - mean) / std


def _ptbxl_diagnostic_classes(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return mapping
        code_column = reader.fieldnames[0]
        for row in reader:
            diagnostic_class = row.get("diagnostic_class", "")
            diagnostic_flag = float(row.get("diagnostic") or 0.0)
            if diagnostic_flag == 1.0 and diagnostic_class in PTBXL_CLASSES:
                mapping[row[code_column]] = diagnostic_class
    return mapping


def _read_wfdb_format16(record_path: Path) -> Tensor:
    header_path = record_path.with_suffix(".hea")
    _require_files(header_path)
    lines = [
        line
        for line in header_path.read_text(encoding="ascii").splitlines()
        if line and not line.startswith("#")
    ]
    header = lines[0].split()
    channel_count = int(header[1])
    sample_count = int(header[3])
    signal_lines = [line.split() for line in lines[1 : 1 + channel_count]]
    data_files = {fields[0] for fields in signal_lines}
    formats = {fields[1].split("+", maxsplit=1)[0] for fields in signal_lines}
    if len(data_files) != 1 or formats != {"16"}:
        raise ExternalDatasetError(
            f"native WFDB reader only supports one-file format-16 records: {header_path}"
        )
    data_path = header_path.parent / next(iter(data_files))
    raw = torch.from_file(
        str(data_path), shared=False, size=sample_count * channel_count, dtype=torch.int16
    )
    raw = raw.reshape(sample_count, channel_count).to(torch.float32)
    gains: list[float] = []
    baselines: list[float] = []
    for fields in signal_lines:
        gain_field = fields[2]
        gain_text = gain_field.split("(", maxsplit=1)[0].split("/", maxsplit=1)[0]
        gains.append(float(gain_text))
        baseline = float(fields[4]) if len(fields) > 4 else 0.0
        if "(" in gain_field and ")" in gain_field:
            baseline = float(gain_field.split("(", maxsplit=1)[1].split(")", maxsplit=1)[0])
        baselines.append(baseline)
    return (raw - torch.tensor(baselines)) / torch.tensor(gains).clamp_min(1.0e-12)


def _cwru_signal_key(payload: dict[str, object]) -> str:
    candidates = sorted(key for key in payload if key.endswith("_DE_time"))
    if not candidates:
        candidates = sorted(key for key in payload if key.endswith("_FE_time"))
    if not candidates:
        raise ExternalDatasetError("CWRU MAT file has no DE/FE vibration signal")
    return candidates[0]


def _path_set(path: Path) -> set[str]:
    _require_files(path)
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _read_wave(path: Path) -> tuple[Tensor, int]:
    with wave.open(str(path), "rb") as stream:
        if stream.getsampwidth() != 2:
            raise ExternalDatasetError(f"only 16-bit PCM wav is supported: {path}")
        channels = stream.getnchannels()
        sample_rate = stream.getframerate()
        frames = stream.readframes(stream.getnframes())
    signal = torch.frombuffer(bytearray(frames), dtype=torch.int16).to(torch.float32)
    signal = signal.reshape(-1, channels).mean(dim=1) / 32768.0
    return signal, sample_rate


def _resample_1d(signal: Tensor, source_rate: int, target_rate: int) -> Tensor:
    if source_rate == target_rate:
        return signal
    target_length = max(1, round(signal.numel() * target_rate / source_rate))
    return functional.interpolate(
        signal.reshape(1, 1, -1), size=target_length, mode="linear", align_corners=False
    ).reshape(-1)


def _pad_or_trim(signal: Tensor, length: int) -> Tensor:
    if signal.numel() >= length:
        return signal[:length]
    return functional.pad(signal, (0, length - signal.numel()))


def _read_pathfinder_image(path: Path) -> Tensor:
    if path.suffix == ".pt":
        value = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(value, Tensor):
            raise ExternalDatasetError(f"Pathfinder tensor file is not a Tensor: {path}")
        image = value
    elif path.suffix == ".npy":
        numpy = import_module("numpy")
        image = torch.as_tensor(numpy.load(path), dtype=torch.float32)
    else:
        io = import_module("torchvision.io")
        image = io.read_image(str(path)).to(torch.float32) / 255.0
    if image.ndim == 3:
        image = image.mean(dim=0)
    return image.to(torch.float32)


def _forecast_boundaries(
    length: int,
    split_kind: Literal["etth", "ettm", "ratio"],
) -> tuple[int, int, int]:
    if split_kind in ("etth", "ettm"):
        samples_per_hour = 4 if split_kind == "ettm" else 1
        train = 12 * 30 * 24 * samples_per_hour
        validation = 4 * 30 * 24 * samples_per_hour
        if length < train + 2 * validation:
            minimum = train + 2 * validation
            raise ExternalDatasetError(
                f"{split_kind.upper()} data has {length} rows; expected at least {minimum}"
            )
        return train, train + validation, train + 2 * validation
    return math.floor(length * 0.7), math.floor(length * 0.8), length


def _forecast_split_windows(
    values: Tensor,
    target_start: int,
    target_end: int,
    context: int,
    horizon: int,
) -> tuple[Tensor, Tensor]:
    # Validation and test may read historical context, but every target remains
    # inside its assigned split.
    context_start = max(0, target_start - context)
    inputs, targets = _forecast_windows(values[context_start:target_end], context, horizon)
    if target_start == 0:
        return inputs, targets
    first_target_offset = target_start - context_start - context
    if first_target_offset < 0:
        raise ExternalDatasetError("forecast split does not contain enough historical context")
    return inputs[first_target_offset:], targets[first_target_offset:]


def _forecast_windows(values: Tensor, context: int, horizon: int) -> tuple[Tensor, Tensor]:
    count = values.shape[0] - context - horizon + 1
    if count < 1:
        raise ExternalDatasetError(
            f"split length {values.shape[0]} is too short for context={context}, horizon={horizon}"
        )
    windows = values.unfold(0, context + horizon, 1).permute(0, 2, 1).contiguous()
    return windows[:, :context], windows[:, context:]


def _checked_split(value: str) -> str:
    if value not in ("train", "validation", "test"):
        raise ExternalDatasetError(f"invalid split {value!r}; expected train/validation/test")
    return value


def _require_files(*paths: Path) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise ExternalDatasetError("missing required files: " + ", ".join(map(str, missing)))


def _tensor(payload: dict[str, object], key: str) -> Tensor:
    value = payload[key]
    if not isinstance(value, Tensor):
        raise ExternalDatasetError(f"prepared task field {key!r} is not a Tensor")
    return value


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (float, int, str)):
        return float(value)
    raise ExternalDatasetError(f"invalid floating-point metadata: {value!r}")


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, str)):
        return int(value)
    raise ExternalDatasetError(f"invalid integer metadata: {value!r}")


def _temporal_metadata_payload(metadata: ExternalTemporalMetadata) -> dict[str, Tensor]:
    return metadata.model_kwargs()


def _temporal_metadata_from_payload(
    payload: dict[str, object],
    split: str,
    path: Path,
) -> ExternalTemporalMetadata:
    container = payload.get("temporal_metadata")
    if container is None:
        return ExternalTemporalMetadata()
    if not isinstance(container, dict):
        raise ExternalDatasetError(f"invalid temporal_metadata in {path}")
    split_payload = container.get(split, {})
    if not isinstance(split_payload, dict):
        raise ExternalDatasetError(f"invalid {split} temporal metadata in {path}")
    allowed = {"time_delta", "observation_mask", "valid_mask"}
    extra = set(split_payload) - allowed
    if extra:
        raise ExternalDatasetError(
            f"unknown {split} temporal metadata fields in {path}: {sorted(extra)}"
        )
    values: dict[str, Tensor | None] = {}
    for name in sorted(allowed):
        value = split_payload.get(name)
        if value is not None and not isinstance(value, Tensor):
            raise ExternalDatasetError(f"{split} {name} in {path} is not a tensor")
        values[name] = value
    return ExternalTemporalMetadata(**values)


def _input_encoding(value: object, path: Path) -> ExternalInputEncoding:
    if value in ("continuous", "tokens", "token_pair"):
        return value
    raise ExternalDatasetError(f"invalid input_encoding in {path}: {value!r}")


def _selection_split_sha256(task: ExternalTask) -> str:
    digest = hashlib.sha256(b"external_selection_split.v3")
    digest.update(repr(task.characteristic_time_scale).encode())
    digest.update(b"\0")
    for name, tensor in (
        ("train_inputs", task.train_inputs),
        ("train_targets", task.train_targets),
        ("validation_inputs", task.validation_inputs),
        ("validation_targets", task.validation_targets),
    ):
        values = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(values.shape)).encode())
        digest.update(str(values.dtype).encode())
        digest.update(memoryview(values.numpy()))
    for name, groups in (
        ("train_groups", task.train_groups),
        ("validation_groups", task.validation_groups),
    ):
        digest.update(name.encode())
        for group in groups:
            digest.update(group.encode())
            digest.update(b"\0")
    for split, metadata in (
        ("train", task.train_metadata),
        ("validation", task.validation_metadata),
    ):
        for name, tensor in sorted(metadata.model_kwargs().items()):
            values = tensor.detach().cpu().contiguous()
            digest.update(f"{split}_{name}".encode())
            digest.update(str(tuple(values.shape)).encode())
            digest.update(str(values.dtype).encode())
            digest.update(memoryview(values.numpy()))
    return digest.hexdigest()


def stable_split_key(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "little")
