# ruff: noqa: EM101, EM102, TRY003
from __future__ import annotations

import csv
import hashlib
import math
import struct
import tarfile
from importlib import import_module
from typing import TYPE_CHECKING, Final

import torch
from torch import Tensor

from .pac_external_tasks import ExternalDatasetError, ExternalTask, save_prepared_task

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from pathlib import Path
    from typing import IO

_BYTE_VOCAB_SIZE: Final = 257
_AUDIOSET_CLASSES: Final = 527
_AUDIOSET_FRAMES: Final = 10
_AUDIOSET_FEATURES: Final = 128


def prepare_vision_tasks(source_root: Path, output_root: Path) -> tuple[Path, ...]:
    datasets = import_module("torchvision.datasets")
    mnist_train = datasets.MNIST(source_root, train=True, download=True)
    mnist_test = datasets.MNIST(source_root, train=False, download=True)
    cifar_train = datasets.CIFAR10(source_root, train=True, download=True)
    cifar_test = datasets.CIFAR10(source_root, train=False, download=True)

    mnist_inputs = mnist_train.data.to(torch.float32).reshape(-1, 784, 1) / 255.0
    mnist_targets = mnist_train.targets.to(torch.long)
    mnist_test_inputs = mnist_test.data.to(torch.float32).reshape(-1, 784, 1) / 255.0
    mnist_test_targets = mnist_test.targets.to(torch.long)
    split = math.floor(mnist_inputs.shape[0] * 0.9)
    sequential_mnist = _vision_task(
        "sequential-mnist",
        mnist_inputs[:split],
        mnist_targets[:split],
        mnist_inputs[split:],
        mnist_targets[split:],
        mnist_test_inputs,
        mnist_test_targets,
    )
    permutation = torch.randperm(784, generator=torch.Generator().manual_seed(0))
    permuted_mnist = _vision_task(
        "permuted-mnist",
        mnist_inputs[:split, permutation],
        mnist_targets[:split],
        mnist_inputs[split:, permutation],
        mnist_targets[split:],
        mnist_test_inputs[:, permutation],
        mnist_test_targets,
    )

    cifar_rows, cifar_grayscale, cifar_targets = _cifar_views(cifar_train)
    cifar_test_rows, cifar_test_grayscale, cifar_test_targets = _cifar_views(cifar_test)
    split = math.floor(cifar_rows.shape[0] * 0.9)
    sequential_cifar = _vision_task(
        "sequential-cifar",
        cifar_rows[:split],
        cifar_targets[:split],
        cifar_rows[split:],
        cifar_targets[split:],
        cifar_test_rows,
        cifar_test_targets,
    )
    lra_image = ExternalTask(
        "lra-image",
        "multiclass",
        cifar_grayscale[:split].to(torch.int16) + 1,
        cifar_targets[:split],
        cifar_grayscale[split:].to(torch.int16) + 1,
        cifar_targets[split:],
        cifar_test_grayscale.to(torch.int16) + 1,
        cifar_test_targets,
        10,
        class_names=tuple(str(index) for index in range(10)),
        input_encoding="tokens",
        vocab_size=_BYTE_VOCAB_SIZE,
    )

    tasks = (sequential_mnist, permuted_mnist, sequential_cifar, lra_image)
    paths = tuple(output_root / f"{task.name}.pt" for task in tasks)
    for task, path in zip(tasks, paths, strict=True):
        save_prepared_task(task, path)
    return paths


def prepare_imdb_task(archive: Path, output: Path, *, max_length: int = 1000) -> Path:
    train_examples: list[tuple[str, bytes, int]] = []
    test_examples: list[tuple[str, bytes, int]] = []
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle:
            if not member.isfile() or not member.name.endswith(".txt"):
                continue
            parts = member.name.split("/")
            if len(parts) != 4 or parts[2] not in ("pos", "neg"):
                continue
            stream = bundle.extractfile(member)
            if stream is None:
                continue
            example = (member.name, stream.read(), int(parts[2] == "pos"))
            if parts[1] == "train":
                train_examples.append(example)
            elif parts[1] == "test":
                test_examples.append(example)
    if len(train_examples) != 25_000 or len(test_examples) != 25_000:
        raise ExternalDatasetError(
            f"IMDb archive has {len(train_examples)} train and {len(test_examples)} test reviews"
        )
    train_examples.sort(key=lambda item: item[0])
    test_examples.sort(key=lambda item: item[0])
    validation = [item for item in train_examples if _stable_bucket(item[0], 10) == 0]
    training = [item for item in train_examples if _stable_bucket(item[0], 10) != 0]
    task = ExternalTask(
        "lra-text",
        "multiclass",
        _encode_byte_examples(training, max_length),
        torch.tensor([item[2] for item in training], dtype=torch.long),
        _encode_byte_examples(validation, max_length),
        torch.tensor([item[2] for item in validation], dtype=torch.long),
        _encode_byte_examples(test_examples, max_length),
        torch.tensor([item[2] for item in test_examples], dtype=torch.long),
        2,
        class_names=("negative", "positive"),
        input_encoding="tokens",
        vocab_size=_BYTE_VOCAB_SIZE,
    )
    save_prepared_task(task, output)
    return output


def prepare_listops_task(
    train_path: Path,
    validation_path: Path,
    test_path: Path,
    output: Path,
    *,
    max_length: int = 2000,
) -> Path:
    splits = tuple(_read_listops(path) for path in (train_path, validation_path, test_path))
    vocabulary = sorted({token for split in splits for source, _ in split for token in source})
    token_to_id = {token: index + 1 for index, token in enumerate(vocabulary)}
    encoded = tuple(_encode_token_examples(split, token_to_id, max_length) for split in splits)
    targets = tuple(
        torch.tensor([target for _, target in split], dtype=torch.long) for split in splits
    )
    task = ExternalTask(
        "lra-listops",
        "multiclass",
        encoded[0],
        targets[0],
        encoded[1],
        targets[1],
        encoded[2],
        targets[2],
        10,
        class_names=tuple(str(index) for index in range(10)),
        input_encoding="tokens",
        vocab_size=len(token_to_id) + 1,
    )
    save_prepared_task(task, output)
    return output


def prepare_retrieval_task(
    train_path: Path,
    validation_path: Path,
    test_path: Path,
    output: Path,
    *,
    max_length: int = 4000,
) -> Path:
    splits = tuple(_read_retrieval(path) for path in (train_path, validation_path, test_path))
    inputs = tuple(_encode_retrieval_examples(split, max_length) for split in splits)
    targets = tuple(
        torch.tensor([example[0] for example in split], dtype=torch.long) for split in splits
    )
    task = ExternalTask(
        "lra-retrieval",
        "multiclass",
        inputs[0],
        targets[0],
        inputs[1],
        targets[1],
        inputs[2],
        targets[2],
        2,
        class_names=("different", "related"),
        input_encoding="token_pair",
        vocab_size=_BYTE_VOCAB_SIZE,
    )
    save_prepared_task(task, output)
    return output


def prepare_audioset_task(
    feature_archive: Path,
    balanced_csv: Path,
    evaluation_csv: Path,
    class_csv: Path,
    output: Path,
) -> Path:
    class_names = _audioset_class_names(class_csv)
    balanced = _audioset_labels(balanced_csv, class_csv)
    evaluation = _audioset_labels(evaluation_csv, class_csv)
    wanted = set(balanced) | set(evaluation)
    features: dict[str, Tensor] = {}
    with tarfile.open(feature_archive, "r:gz") as bundle:
        for member in bundle:
            if not member.isfile():
                continue
            stream = bundle.extractfile(member)
            if stream is None:
                continue
            for record in _tfrecord_records(stream):
                video_id, embedding = _audioset_sequence_example(record)
                if video_id in wanted:
                    features[video_id] = embedding
    missing = wanted - features.keys()
    if missing:
        sample = ", ".join(sorted(missing)[:5])
        message = f"AudioSet archive is missing {len(missing)} requested IDs: {sample}"
        raise ExternalDatasetError(message)
    train_ids = sorted(video_id for video_id in balanced if _stable_bucket(video_id, 10) != 0)
    validation_ids = sorted(video_id for video_id in balanced if _stable_bucket(video_id, 10) == 0)
    test_ids = sorted(evaluation)
    train_inputs = torch.stack([features[video_id] for video_id in train_ids])
    validation_inputs = torch.stack([features[video_id] for video_id in validation_ids])
    test_inputs = torch.stack([features[video_id] for video_id in test_ids])
    mean = train_inputs.mean(dim=(0, 1), keepdim=True)
    scale = train_inputs.std(dim=(0, 1), keepdim=True).clamp_min(1.0e-6)
    task = ExternalTask(
        "audioset-balanced",
        "multilabel",
        (train_inputs - mean) / scale,
        _audioset_target_tensor(train_ids, balanced),
        (validation_inputs - mean) / scale,
        _audioset_target_tensor(validation_ids, balanced),
        (test_inputs - mean) / scale,
        _audioset_target_tensor(test_ids, evaluation),
        _AUDIOSET_CLASSES,
        class_names=class_names,
    )
    save_prepared_task(task, output)
    return output


def _vision_task(
    name: str,
    train_inputs: Tensor,
    train_targets: Tensor,
    validation_inputs: Tensor,
    validation_targets: Tensor,
    test_inputs: Tensor,
    test_targets: Tensor,
) -> ExternalTask:
    return ExternalTask(
        name,
        "multiclass",
        train_inputs,
        train_targets,
        validation_inputs,
        validation_targets,
        test_inputs,
        test_targets,
        10,
        class_names=tuple(str(index) for index in range(10)),
    )


def _cifar_views(dataset: object) -> tuple[Tensor, Tensor, Tensor]:
    data = torch.as_tensor(getattr(dataset, "data"), dtype=torch.float32)  # noqa: B009
    grayscale = torch.round(
        0.2989 * data[..., 0] + 0.5870 * data[..., 1] + 0.1140 * data[..., 2]
    )
    rows = data.reshape(-1, 32, 96) / 255.0
    targets = torch.tensor(getattr(dataset, "targets"), dtype=torch.long)  # noqa: B009
    return rows, grayscale.to(torch.uint8).reshape(-1, 1024, 1), targets


def _encode_byte_examples(examples: list[tuple[str, bytes, int]], max_length: int) -> Tensor:
    encoded = torch.zeros(len(examples), max_length, 1, dtype=torch.int16)
    for index, (_, text, _) in enumerate(examples):
        values = torch.tensor(list(text[:max_length]), dtype=torch.int16) + 1
        encoded[index, : values.numel(), 0] = values
    return encoded


def _read_listops(path: Path) -> list[tuple[tuple[str, ...], int]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle, delimiter="\t")
        return [
            (
                tuple(row["Source"].replace("]", "X").replace("(", "").replace(")", "").split()),
                int(row["Target"]),
            )
            for row in rows
        ]


def _encode_token_examples(
    examples: list[tuple[tuple[str, ...], int]],
    token_to_id: dict[str, int],
    max_length: int,
) -> Tensor:
    encoded = torch.zeros(len(examples), max_length, 1, dtype=torch.int16)
    for index, (tokens, _) in enumerate(examples):
        values = torch.tensor(
            [token_to_id[token] for token in tokens[:max_length]], dtype=torch.int16
        )
        encoded[index, : values.numel(), 0] = values
    return encoded


def _read_retrieval(path: Path) -> list[tuple[int, bytes, bytes]]:
    examples: list[tuple[int, bytes, bytes]] = []
    csv.field_size_limit(64 * 1024 * 1024)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if not row or row[0].lower() in ("label", "target"):
                continue
            if len(row) < 5:
                raise ExternalDatasetError(f"retrieval row in {path} has {len(row)} columns")
            label = int(float(row[0]))
            if label not in (0, 1):
                raise ExternalDatasetError(f"retrieval label must be binary, got {row[0]!r}")
            examples.append((label, row[-2].encode(), row[-1].encode()))
    return examples


def _encode_retrieval_examples(examples: list[tuple[int, bytes, bytes]], max_length: int) -> Tensor:
    encoded = torch.zeros(len(examples), max_length, 2, dtype=torch.int16)
    for index, (_, left, right) in enumerate(examples):
        for side, document in enumerate((left, right)):
            values = torch.tensor(list(document[:max_length]), dtype=torch.int16) + 1
            encoded[index, : values.numel(), side] = values
    return encoded


def _stable_bucket(value: str, buckets: int) -> int:
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "little") % buckets


def _audioset_class_names(path: Path) -> tuple[str, ...]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda row: int(row["index"]))
    if len(rows) != _AUDIOSET_CLASSES:
        raise ExternalDatasetError(f"expected 527 AudioSet classes, found {len(rows)}")
    return tuple(row["display_name"] for row in rows)


def _audioset_labels(segment_path: Path, class_path: Path) -> dict[str, tuple[int, ...]]:
    with class_path.open(encoding="utf-8", newline="") as handle:
        mid_to_index = {row["mid"]: int(row["index"]) for row in csv.DictReader(handle)}
    labels: dict[str, tuple[int, ...]] = {}
    with segment_path.open(encoding="utf-8", newline="") as handle:
        rows = csv.reader(line for line in handle if not line.startswith("#"))
        for row in rows:
            video_id = row[0].strip()
            mids = row[3].strip().strip('"').split(",")
            labels[video_id] = tuple(mid_to_index[mid] for mid in mids)
    return labels


def _audioset_target_tensor(
    identifiers: list[str], labels: dict[str, tuple[int, ...]]
) -> Tensor:
    targets = torch.zeros(len(identifiers), _AUDIOSET_CLASSES, dtype=torch.float32)
    for row, video_id in enumerate(identifiers):
        targets[row, list(labels[video_id])] = 1.0
    return targets


def _tfrecord_records(stream: IO[bytes]) -> Iterator[bytes]:
    while header := stream.read(12):
        if len(header) != 12:
            raise ExternalDatasetError("truncated TFRecord header")
        length = struct.unpack("<Q", header[:8])[0]
        record = stream.read(length)
        footer = stream.read(4)
        if len(record) != length or len(footer) != 4:
            raise ExternalDatasetError("truncated TFRecord payload")
        yield record


def _audioset_sequence_example(data: bytes) -> tuple[str, Tensor]:
    top = _protobuf_fields(data)
    context = _protobuf_map(_first_bytes(top, 1))
    feature_lists = _protobuf_map(_first_bytes(top, 2))
    video_feature = _protobuf_fields(context["video_id"])
    video_id = _first_bytes(_protobuf_fields(_first_bytes(video_feature, 1)), 1).decode()
    feature_list = _protobuf_fields(feature_lists["audio_embedding"])
    rows: list[Tensor] = []
    for feature in _all_bytes(feature_list, 1):
        bytes_list = _protobuf_fields(_first_bytes(_protobuf_fields(feature), 1))
        raw = _first_bytes(bytes_list, 1)
        rows.append(torch.tensor(list(raw), dtype=torch.float32))
    embedding = torch.zeros(_AUDIOSET_FRAMES, _AUDIOSET_FEATURES, dtype=torch.float32)
    if rows:
        stacked = torch.stack(rows[:_AUDIOSET_FRAMES])
        if stacked.shape[1] != _AUDIOSET_FEATURES:
            raise ExternalDatasetError(f"AudioSet embedding width is {stacked.shape[1]}")
        embedding[: stacked.shape[0]] = stacked
    return video_id, embedding


def _protobuf_map(data: bytes) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    for entry in _all_bytes(_protobuf_fields(data), 1):
        fields = _protobuf_fields(entry)
        result[_first_bytes(fields, 1).decode()] = _first_bytes(fields, 2)
    return result


def _protobuf_fields(data: bytes) -> list[tuple[int, int, int | bytes]]:
    fields: list[tuple[int, int, int | bytes]] = []
    offset = 0
    while offset < len(data):
        tag, offset = _read_varint(data, offset)
        number, wire = tag >> 3, tag & 7
        if wire == 0:
            value, offset = _read_varint(data, offset)
        elif wire == 1:
            value = data[offset : offset + 8]
            offset += 8
        elif wire == 2:
            length, offset = _read_varint(data, offset)
            value = data[offset : offset + length]
            offset += length
        elif wire == 5:
            value = data[offset : offset + 4]
            offset += 4
        else:
            raise ExternalDatasetError(f"unsupported protobuf wire type {wire}")
        fields.append((number, wire, value))
    return fields


def _read_varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while offset < len(data):
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, offset
        shift += 7
    raise ExternalDatasetError("truncated protobuf varint")


def _all_bytes(fields: Iterable[tuple[int, int, int | bytes]], number: int) -> list[bytes]:
    values = [value for field, wire, value in fields if field == number and wire == 2]
    if not all(isinstance(value, bytes) for value in values):
        raise ExternalDatasetError(f"protobuf field {number} is not length-delimited")
    return [value for value in values if isinstance(value, bytes)]


def _first_bytes(fields: Iterable[tuple[int, int, int | bytes]], number: int) -> bytes:
    values = _all_bytes(fields, number)
    if not values:
        raise ExternalDatasetError(f"protobuf field {number} is missing")
    return values[0]
