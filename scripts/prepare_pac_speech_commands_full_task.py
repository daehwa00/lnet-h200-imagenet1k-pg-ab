"""Materialize the Q2-final Speech Commands tensor without changing its data.

The raw loader transiently holds per-file waveforms, their stacked tensor, a
normalized copy, and split copies.  Serializing the already-normalized split
tensors once removes that construction peak for every Q2-final worker.  The
TRAIN/validation content is accepted only when its SHA-256 fingerprint equals
the sealed selection-only artifact used by calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from lnet.pac_external_tasks import (
    ExternalTask,
    _selection_split_sha256,  # pyright: ignore[reportPrivateUsage]
    load_external_selection_task,
    load_prepared_task,
    load_speech_commands,
    save_prepared_task,
)

SCHEMA = "pac_speech_commands_full_preparation.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_selection_identity(task: ExternalTask, expected_sha256: str) -> str:
    """Fail closed unless TRAIN/validation tensors match the sealed snapshot."""
    actual = _selection_split_sha256(task)
    if actual != expected_sha256:
        message = (
            "Speech Commands full-task TRAIN/validation fingerprint changed: "
            f"{actual} != {expected_sha256}"
        )
        raise RuntimeError(message)
    return actual


def prepare(data_root: Path) -> dict[str, object]:
    output = data_root / "speech-commands.pt"
    selection = load_external_selection_task("speech-commands", data_root)
    if output.is_file():
        task = load_prepared_task(output)
        source = "existing prepared tensor"
    else:
        task = load_speech_commands(data_root / "speech-commands")
        verify_selection_identity(task, selection.selection_split_sha256)
        save_prepared_task(task, output)
        source = "raw WAV loader, then one-time serialization"

    selection_sha256 = verify_selection_identity(
        task, selection.selection_split_sha256
    )
    payload: dict[str, object] = {
        "schema": SCHEMA,
        "prepared_path": str(output),
        "prepared_sha256": _sha256(output),
        "selection_split_sha256": selection_sha256,
        "selection_identity_verified": True,
        "source": source,
        "train_count": int(task.train_inputs.shape[0]),
        "validation_count": int(task.validation_inputs.shape[0]),
        "test_count": int(task.test_inputs.shape[0]),
        "sequence_length": task.sequence_length,
        "input_dim": task.input_dim,
        "output_dim": task.output_dim,
        "scientific_contract": (
            "serialization-only optimization; TRAIN/validation tensors equal the "
            "sealed calibration artifact by full-content SHA-256"
        ),
    }
    report = data_root / "speech-commands.pt.audit.json"
    temporary = report.with_suffix(report.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(report)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/external"))
    args = parser.parse_args()
    sys.stdout.write(json.dumps(prepare(args.data_root), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
