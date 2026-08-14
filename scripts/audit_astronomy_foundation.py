from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import torch

from lnet.astronomy.phase0 import Phase0RunConfig, build_model
from lnet.astronomy.plasticc import (
    PHASE0_CLASS_NAMES,
    PHASE0_TARGETS,
    read_light_curves,
    read_phase0_labels,
    stratified_object_split,
)
from lnet.astronomy.poles import AstronomyPoleRange

OFFICIAL_ZENODO_RECORD = "https://zenodo.org/records/2539456"
OFFICIAL_FILES = {
    "plasticc_train_metadata.csv.gz": {
        "bytes": 370_350,
        "md5": "8c6b00fd503d6cf3d9a42bfb53046e0f",
    },
    "plasticc_train_lightcurves.csv.gz": {
        "bytes": 21_553_100,
        "md5": "1aa1605908b5a6398bd46bf9120b6400",
    },
}
SOURCE_PATHS = (
    "src/lnet/alphabet.py",
    "src/lnet/astronomy/plasticc.py",
    "src/lnet/astronomy/poles.py",
    "src/lnet/astronomy/phase0.py",
    "scripts/run_astronomy_phase0.py",
    "scripts/audit_astronomy_foundation.py",
)


class _PoleBlock(Protocol):
    damping_min: float
    damping_max: float
    frequency_bound: float

    def damping_values(self) -> torch.Tensor: ...


def _digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_digest(object_ids: tuple[int, ...]) -> str:
    payload = json.dumps(object_ids, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=20260729)
    args = parser.parse_args()

    repository = Path(__file__).resolve().parents[1]
    metadata_path = args.data_dir / "plasticc_train_metadata.csv.gz"
    light_curves_path = args.data_dir / "plasticc_train_lightcurves.csv.gz"
    file_audit: dict[str, dict[str, object]] = {}
    for path in (metadata_path, light_curves_path):
        expected = OFFICIAL_FILES[path.name]
        actual = {
            "bytes": path.stat().st_size,
            "md5": _digest(path, "md5"),
            "sha256": _digest(path, "sha256"),
        }
        file_audit[path.name] = {
            **actual,
            "official_expected": expected,
            "official_match": (
                actual["bytes"] == expected["bytes"] and actual["md5"] == expected["md5"]
            ),
        }

    labels = read_phase0_labels(metadata_path, seed=args.split_seed)
    curves = read_light_curves(light_curves_path, labels)
    split = stratified_object_split(labels, seed=args.split_seed)
    split_sets = {
        "train": set(split.train),
        "validation": set(split.validation),
        "test": set(split.test),
    }
    observed_flux = np.concatenate(
        [curve.flux[curve.observation_mask] for curve in curves.values()]
    )
    epoch_counts = np.asarray(
        [curve.flux.shape[0] for curve in curves.values()],
        dtype=np.int64,
    )

    model = build_model(
        Phase0RunConfig(model="alphabet", seed=7),
        sequence_length=int(epoch_counts.max()),
    )
    pole_range = AstronomyPoleRange()
    blocks: dict[str, object] = {}
    for name, raw_block in (
        ("forward", model.forward_block),
        ("backward", model.backward_block),
    ):
        block = cast("_PoleBlock", cast("object", raw_block))
        damping = block.damping_values().detach().cpu()
        blocks[name] = {
            "damping_min_per_day": block.damping_min,
            "damping_max_per_day": block.damping_max,
            "frequency_bound_rad_per_day": block.frequency_bound,
            "initialized_damping_min_per_day": float(damping.min()),
            "initialized_damping_max_per_day": float(damping.max()),
        }

    class_counts = {
        name: sum(class_index == index for class_index in labels.values())
        for index, name in enumerate(PHASE0_CLASS_NAMES)
    }
    source_sha256 = {
        relative_path: _digest(repository / relative_path, "sha256")
        for relative_path in SOURCE_PATHS
    }
    report = {
        "schema": "lnet.astronomy.foundation_audit.v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "host": socket.gethostname(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        },
        "official_source": {
            "record": OFFICIAL_ZENODO_RECORD,
            "files": file_audit,
            "all_files_match": all(bool(entry["official_match"]) for entry in file_audit.values()),
        },
        "representation": {
            "object_count": len(curves),
            "target_ids": PHASE0_TARGETS,
            "class_counts": class_counts,
            "epoch_token_contract": "one sparse six-band tensor per unique MJD",
            "time_unit": "days",
            "first_interval_zero_for_every_object": all(
                curve.time_delta[0] == 0.0 for curve in curves.values()
            ),
            "signed_raw_flux_preserved": True,
            "per_object_flux_normalization": False,
            "observed_flux_count": int(observed_flux.size),
            "negative_observed_flux_count": int((observed_flux < 0.0).sum()),
            "epoch_tokens": {
                "minimum": int(epoch_counts.min()),
                "median": float(np.median(epoch_counts)),
                "maximum": int(epoch_counts.max()),
            },
        },
        "split": {
            "seed": args.split_seed,
            "counts": {
                "train": len(split.train),
                "validation": len(split.validation),
                "test": len(split.test),
            },
            "object_disjoint": not (
                split_sets["train"] & split_sets["validation"]
                or split_sets["train"] & split_sets["test"]
                or split_sets["validation"] & split_sets["test"]
            ),
            "covers_every_selected_object": set(labels) == set().union(*split_sets.values()),
            "object_id_sha256": {
                "train": _split_digest(split.train),
                "validation": _split_digest(split.validation),
                "test": _split_digest(split.test),
            },
        },
        "pole_range": {
            "requested": {
                "damping_min_per_day": pole_range.damping_min_per_day,
                "damping_max_per_day": pole_range.damping_max_per_day,
                "frequency_max_rad_per_day": pole_range.frequency_max_rad_per_day,
                "minimum_period_days": 0.05,
                "maximum_memory_timescale_days": 3000.0,
            },
            "blocks": blocks,
        },
        "source_sha256": source_sha256,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
