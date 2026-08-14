# ruff: noqa: EM102, T201, TRY003
"""Extract leakage-safe Wave-2 manifests from downloaded public archives."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import re
import zipfile
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
from scipy.io import loadmat

from lnet.pac_edf import read_annotations, read_signals
from lnet.pac_gdf import read_gdf


def _stratified_splits(rows: list[dict[str, str]]) -> None:
    by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)
    group_split: dict[str, str] = {}
    for records in by_label.values():
        groups = sorted({row["group"] for row in records if row["group"] not in group_split})
        count = len(groups)
        held_out = max(1, round(0.15 * count)) if count >= 3 else 0
        for index, group in enumerate(groups):
            if held_out and index >= count - held_out:
                group_split[group] = "test"
            elif held_out and index >= count - 2 * held_out:
                group_split[group] = "validation"
            else:
                group_split[group] = "train"
    for row in rows:
        row["split"] = group_split[row["group"]]


def _write(rows: list[dict[str, str]], path: Path) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty Wave-2 manifest: {path}")
    _stratified_splits(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "path",
        "label",
        "group",
        "split",
        "sample_rate_hz",
        "signal_key",
        "channels",
        "start_sample",
        "stop_sample",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def extract_mfpt(raw_root: Path, manifest: Path) -> int:
    archive = (
        raw_root
        / "mfpt/source/dataset/Diagnostics/MFPT_data/MFPT_data.zip"
    )
    extracted = raw_root / "mfpt/extracted"
    extracted.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as payload:
        payload.extractall(extracted)
    rows = []
    for path in sorted(extracted.glob("*.txt")):
        lowered = path.name.lower()
        label = (
            "inner-race"
            if "inner" in lowered
            else "outer-race"
            if "outter" in lowered or "outer" in lowered
            else "normal"
        )
        rows.append(
            {
                "path": os.path.relpath(path, manifest.parent),
                "label": label,
                "group": path.stem,
                "split": "",
                "sample_rate_hz": "48828",
                "signal_key": "",
                "channels": "0",
            }
        )
    _write(rows, manifest)
    return len(rows)


def _header_value(path: Path, key: str) -> str:
    prefix = f"# {key}:"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix):
            return line.removeprefix(prefix).strip()
    raise ValueError(f"{path} has no {key} field")


def extract_cpsc(raw_root: Path, manifest: Path) -> int:
    root = raw_root / "cpsc2018"
    diagnosis_names = {
        "426783006": "normal",
        "164889003": "atrial-fibrillation",
        "270492004": "first-degree-av-block",
        "164909002": "left-bundle-branch-block",
        "59118001": "right-bundle-branch-block",
        "284470004": "premature-atrial-contraction",
        "427172004": "premature-ventricular-contraction",
        "429622005": "st-depression",
        "164931005": "st-elevation",
    }
    rows = []
    for header in sorted(root.glob("g*/*.hea")):
        signal = header.with_suffix(".mat")
        if not signal.is_file():
            continue
        diagnoses = tuple(
            sorted(re.findall(r"\d+", _header_value(header, "Dx")))
        )
        target_diagnoses = [
            diagnosis_names[code] for code in diagnoses if code in diagnosis_names
        ]
        if len(target_diagnoses) != 1:
            continue
        label = target_diagnoses[0]
        rows.append(
            {
                "path": os.path.relpath(signal, manifest.parent),
                "label": label,
                "group": signal.stem,
                "split": "",
                "sample_rate_hz": "500",
                "signal_key": "val",
                "channels": ";".join(str(index) for index in range(12)),
            }
        )
    _write(rows, manifest)
    return len(rows)


def _xlsx_rows(path: Path) -> list[list[str]]:
    """Read the first worksheet without adding an Excel dependency."""
    namespace = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with zipfile.ZipFile(path) as workbook:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in workbook.namelist():
            root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))  # noqa: S314
            shared = [
                "".join(node.text or "" for node in item.iter() if node.tag.endswith("}t"))
                for item in root
            ]
        sheet = ET.fromstring(workbook.read("xl/worksheets/sheet1.xml"))  # noqa: S314
    result = []
    for row in sheet.iter(namespace + "row"):
        values: dict[int, str] = {}
        for cell in row:
            if not cell.tag.endswith("}c"):
                continue
            column = re.match(r"[A-Z]+", cell.attrib["r"])
            if column is None:
                continue
            index = 0
            for character in column.group():
                index = index * 26 + ord(character) - ord("A") + 1
            value_node = next(
                (node for node in cell if node.tag.endswith("}v")),
                None,
            )
            value = "" if value_node is None else value_node.text or ""
            if cell.attrib.get("t") == "s" and value:
                value = shared[int(value)]
            values[index - 1] = value
        width = max(values, default=-1) + 1
        result.append([values.get(index, "") for index in range(width)])
    return result


def extract_chapman(raw_root: Path, manifest: Path) -> int:
    root = raw_root / "chapman"
    table = _xlsx_rows(root / "Diagnostics.xlsx")
    columns = {name: index for index, name in enumerate(table[0])}
    labels = {
        row[columns["FileName"]]: row[columns["Rhythm"]]
        for row in table[1:]
        if len(row) > max(columns["FileName"], columns["Rhythm"])
    }
    signal_root = root / "extracted/ECGDataDenoised"
    rows = []
    for signal in sorted(signal_root.glob("*.csv")):
        label = labels.get(signal.stem)
        if not label:
            continue
        rows.append(
            {
                "path": os.path.relpath(signal, manifest.parent),
                "label": label,
                "group": signal.stem,
                "split": "",
                "sample_rate_hz": "500",
                "signal_key": "",
                "channels": ";".join(str(index) for index in range(12)),
            }
        )
    _write(rows, manifest)
    return len(rows)


def extract_paderborn(raw_root: Path, manifest: Path) -> int:
    source = raw_root / "paderborn/extracted"
    cache = raw_root / "paderborn-selected"
    cache.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(source.rglob("*.mat")):
        bearing = path.stem.split("_")[-2]
        if bearing.startswith("K0"):
            label = "healthy"
        elif bearing.startswith("KA"):
            label = "outer-race"
        elif bearing.startswith("KI"):
            label = "inner-race"
        else:
            # KB bearings have mixed real damages and do not define a single
            # inner/outer class, so they are excluded from this endpoint.
            continue
        destination = cache / bearing / f"{path.stem}.npy"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.is_file():
            payload = loadmat(path, squeeze_me=True, struct_as_record=False)
            measurement = next(
                value for key, value in payload.items() if not key.startswith("__")
            )
            vibration = next(
                channel.Data
                for channel in np.atleast_1d(measurement.Y)
                if channel.Name == "vibration_1"
            )
            np.save(destination, np.asarray(vibration, dtype=np.float32)[:, None])
        rows.append(
            {
                "path": os.path.relpath(destination, manifest.parent),
                "label": label,
                "group": bearing,
                "split": "",
                "sample_rate_hz": "64000",
                "signal_key": "",
                "channels": "0",
            }
        )
    _write(rows, manifest)
    return len(rows)


def extract_ims(raw_root: Path, manifest: Path) -> int:
    source = raw_root / "ims/runs"
    rows = []
    for run_root in sorted(path for path in source.iterdir() if path.is_dir()):
        signals = sorted(
            path
            for path in run_root.rglob("*")
            if (
                path.is_file()
                and path.stat().st_size > 0
                and not path.name.startswith(".")
            )
        )
        count = len(signals)
        if count < 3:
            continue
        for index, path in enumerate(signals):
            progress = index / (count - 1)
            label = (
                "healthy"
                if progress < 0.50
                else "degrading"
                if progress < 0.80
                else "late-life"
            )
            rows.append(
                {
                    "path": os.path.relpath(path, manifest.parent),
                    "label": label,
                    "group": run_root.name,
                    "split": "",
                    "sample_rate_hz": "20000",
                    "signal_key": "text",
                    "channels": "0",
                }
            )
    _write(rows, manifest)
    return len(rows)


def extract_xjtu(raw_root: Path, manifest: Path) -> int:
    source = raw_root / "xjtu/extracted/XJTU-SY_Bearing_Datasets"
    rows = []
    for bearing_root in sorted(path for path in source.glob("*/*") if path.is_dir()):
        signals = sorted(
            bearing_root.glob("*.csv"),
            key=lambda path: int(path.stem),
        )
        count = len(signals)
        if count < 3:
            continue
        group = f"{bearing_root.parent.name}/{bearing_root.name}"
        for index, path in enumerate(signals):
            progress = index / (count - 1)
            label = (
                "healthy"
                if progress < 0.50
                else "degrading"
                if progress < 0.80
                else "late-life"
            )
            rows.append(
                {
                    "path": os.path.relpath(path, manifest.parent),
                    "label": label,
                    "group": group,
                    "split": "",
                    "sample_rate_hz": "25600",
                    "signal_key": "",
                    "channels": "0;1",
                }
            )
    _write(rows, manifest)
    return len(rows)


_BCI_LABELS = {
    769: "left-hand",
    770: "right-hand",
    771: "feet",
    772: "tongue",
}


def extract_bci(raw_root: Path, manifest: Path) -> int:
    source = raw_root / "bci-iv-2a"
    cache = raw_root / "bci-iv-2a-selected"
    cache.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(source.glob("A??T.gdf")):
        subject = path.stem[:3]
        recording = read_gdf(path)
        length = round(4.0 * recording.sample_rate_hz)
        for trial, (position, event_type) in enumerate(
            zip(
                recording.event_positions,
                recording.event_types,
                strict=True,
            )
        ):
            label = _BCI_LABELS.get(int(event_type))
            if label is None or position + length > recording.signals.shape[0]:
                continue
            destination = cache / subject / f"{trial:04d}.npy"
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.is_file():
                np.save(
                    destination,
                    np.asarray(
                        recording.signals[position : position + length, :22],
                        dtype=np.float32,
                    ),
                )
            rows.append(
                {
                    "path": os.path.relpath(destination, manifest.parent),
                    "label": label,
                    "group": subject,
                    "split": "",
                    "sample_rate_hz": str(recording.sample_rate_hz),
                    "signal_key": "",
                    "channels": "",
                }
            )
    _write(rows, manifest)
    return len(rows)


_ISRUC_LABELS = {
    "0": "W",
    "1": "N1",
    "2": "N2",
    "3": "N3",
    "4": "REM",
}


def _isruc_channel(recording_root: Path) -> np.ndarray:
    data_root = recording_root / "data"
    metadata = json.loads((data_root / "zarr.json").read_text(encoding="utf-8"))
    epochs, channels, samples = metadata["shape"]
    chunk_epochs = metadata["chunk_grid"]["configuration"]["chunk_shape"][0]
    pieces = []
    for chunk_index in range((epochs + chunk_epochs - 1) // chunk_epochs):
        payload = gzip.decompress(
            (data_root / f"c/{chunk_index}/0/0").read_bytes()
        )
        chunk = np.frombuffer(payload, dtype="<f4")
        chunk = chunk.reshape(-1, channels, samples)
        pieces.append(np.asarray(chunk[:, 0, :], dtype=np.float32))
    return np.concatenate(pieces, axis=0)[:epochs]


def extract_isruc(raw_root: Path, manifest: Path) -> int:
    source = raw_root / "isruc-hf/sourcedata/braindecode/dataset.zarr"
    cache = raw_root / "isruc-selected"
    cache.mkdir(parents=True, exist_ok=True)
    rows = []
    for recording_root in sorted(source.glob("recording_*")):
        if not (recording_root / "data/zarr.json").is_file():
            continue
        recording_metadata = json.loads(
            (recording_root / "zarr.json").read_text(encoding="utf-8")
        )
        group = recording_metadata["attributes"]["description"]["subject"]
        table = list(
            csv.DictReader(
                (recording_root / "metadata.tsv").open(encoding="utf-8"),
                delimiter="\t",
            )
        )
        destination = cache / f"{recording_root.name}.npy"
        if not destination.is_file():
            epochs = _isruc_channel(recording_root)
            np.save(destination, epochs.reshape(-1, 1))
        for epoch, item in enumerate(table):
            label = _ISRUC_LABELS.get(item["target"])
            if label is None:
                continue
            rows.append(
                {
                    "path": os.path.relpath(destination, manifest.parent),
                    "label": label,
                    "group": group,
                    "split": "",
                    "sample_rate_hz": "200",
                    "signal_key": "",
                    "channels": "0",
                    "start_sample": str(epoch * 6000),
                    "stop_sample": str((epoch + 1) * 6000),
                }
            )
    _write(rows, manifest)
    return len(rows)


_SLEEP_LABELS = {
    "Sleep stage W": "W",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",
    "Sleep stage R": "REM",
}


def extract_sleepedf(raw_root: Path, manifest: Path) -> int:
    source = raw_root / "sleep-edfx/sleep-cassette"
    cache = raw_root / "sleep-edfx-selected"
    cache.mkdir(parents=True, exist_ok=True)
    pairs = []
    for psg in sorted(source.glob("*-PSG.edf")):
        hypnogram = psg.with_name(psg.name.replace("-PSG.edf", "-Hypnogram.edf"))
        if hypnogram.is_file():
            pairs.append((psg, hypnogram))
    subject_ids = sorted({psg.name[:5] for psg, _ in pairs})[:78]
    allowed = set(subject_ids)
    rows = []
    for psg, hypnogram in pairs:
        subject = psg.name[:5]
        if subject not in allowed:
            continue
        destination = cache / f"{psg.stem}.npy"
        if not destination.is_file():
            signal, sample_rate = read_signals(psg, ("EEG Fpz-Cz",))
            if round(sample_rate) != 100:
                raise ValueError(f"unexpected SleepEDF rate {sample_rate} in {psg}")
            np.save(destination, signal)
        for onset, duration, raw_label in read_annotations(hypnogram):
            label = _SLEEP_LABELS.get(raw_label)
            if label is None:
                continue
            epochs = max(1, round(duration / 30.0))
            for epoch in range(epochs):
                start = round((onset + 30.0 * epoch) * 100)
                rows.append(
                    {
                        "path": os.path.relpath(destination, manifest.parent),
                        "label": label,
                        "group": subject,
                        "split": "",
                        "sample_rate_hz": "100",
                        "signal_key": "",
                        "channels": "0",
                        "start_sample": str(start),
                        "stop_sample": str(start + 3000),
                    }
                )
    _write(rows, manifest)
    return len(rows)


_CHB_CHANNELS = (
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "FP2-F8",
    "F8-T8",
    "T8-P8",
    "P8-O2",
)


def _chb_seizures(summary: Path) -> dict[str, list[tuple[float, float]]]:
    result: dict[str, list[tuple[float, float]]] = defaultdict(list)
    current = ""
    starts: list[float] = []
    for line in summary.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("File Name:"):
            current = line.split(":", 1)[1].strip()
            starts = []
        elif "Seizure Start Time:" in line:
            starts.append(float(re.findall(r"\d+", line)[-1]))
        elif "Seizure End Time:" in line and starts:
            result[current].append((starts.pop(0), float(re.findall(r"\d+", line)[-1])))
    return result


def extract_chbmit(raw_root: Path, manifest: Path) -> int:
    source = raw_root / "chbmit"
    cache = raw_root / "chbmit-selected"
    cache.mkdir(parents=True, exist_ok=True)
    rows = []
    for subject_root in sorted(path for path in source.glob("chb*") if path.is_dir()):
        summary = next(subject_root.glob("*-summary.txt"), None)
        seizures = {} if summary is None else _chb_seizures(summary)
        for edf in sorted(subject_root.glob("*.edf")):
            destination = cache / subject_root.name / f"{edf.stem}.npy"
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                if not destination.is_file():
                    signal, sample_rate = read_signals(edf, _CHB_CHANNELS)
                    np.save(destination, signal)
                else:
                    sample_rate = 256.0
            except ValueError:
                continue
            length = round(4.0 * sample_rate)
            stride = round(2.0 * sample_rate)
            samples = int(np.load(destination, mmap_mode="r").shape[0])
            intervals = seizures.get(edf.name, ())
            positive_rows = []
            background_rows = []
            for start in range(0, samples - length + 1, stride):
                left, right = start / sample_rate, (start + length) / sample_rate
                positive = any(max(left, begin) < min(right, end) for begin, end in intervals)
                row = {
                    "path": os.path.relpath(destination, manifest.parent),
                    "label": "seizure" if positive else "background",
                    "group": subject_root.name,
                    "split": "",
                    "sample_rate_hz": str(sample_rate),
                    "signal_key": "",
                    "channels": "",
                    "start_sample": str(start),
                    "stop_sample": str(start + length),
                }
                (positive_rows if positive else background_rows).append(row)
            # Preserve every seizure-overlapping window and use an evenly
            # distributed deterministic background subset.  This prevents
            # multi-million-window caches while retaining full event coverage.
            keep_background = min(
                len(background_rows),
                max(64, 4 * len(positive_rows)),
            )
            if keep_background and len(background_rows) > keep_background:
                indices = np.linspace(
                    0,
                    len(background_rows) - 1,
                    keep_background,
                    dtype=int,
                )
                background_rows = [background_rows[index] for index in indices]
            rows.extend(positive_rows)
            rows.extend(background_rows)
    _write(rows, manifest)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=(
            "mfpt-bearing",
            "cpsc-2018",
            "sleepedf-78",
            "chb-mit",
            "chapman-shaoxing",
            "paderborn-kat",
            "ims-bearing",
            "xjtu-sy",
            "bci-iv-2a",
            "isruc-sleep",
        ),
        required=True,
    )
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    extractors = {
        "mfpt-bearing": extract_mfpt,
        "cpsc-2018": extract_cpsc,
        "sleepedf-78": extract_sleepedf,
        "chb-mit": extract_chbmit,
        "chapman-shaoxing": extract_chapman,
        "paderborn-kat": extract_paderborn,
        "ims-bearing": extract_ims,
        "xjtu-sy": extract_xjtu,
        "bci-iv-2a": extract_bci,
        "isruc-sleep": extract_isruc,
    }
    count = extractors[args.dataset](args.raw_root, args.manifest)
    print(f"{args.dataset}: wrote {count} records to {args.manifest}")


if __name__ == "__main__":
    main()
