from __future__ import annotations

# pyright: reportMissingImports=false
import argparse
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
import torch

from lnet.astronomy.features import broker_features
from lnet.astronomy.metrics import MetricAccumulator
from lnet.astronomy.plasticc import (
    PLASTICC_KNOWN_TARGETS,
    LightCurve,
    iter_light_curves,
    read_phase0_labels,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


class ProbabilisticClassifier(Protocol):
    def predict_proba(self, features: NDArray[np.float32]) -> NDArray[np.float64]: ...


def _load_trusted_model(path: Path, root: Path) -> ProbabilisticClassifier:
    """Load only an explicitly trusted, local model artifact.

    Joblib uses pickle internally, so this is intentionally opt-in at the CLI
    boundary rather than treating a results directory as an untrusted upload.
    """
    resolved_root = root.resolve()
    if path.is_symlink():
        raise ValueError(f"model artifact must not be a symlink: {path}")
    resolved = path.resolve()
    if not resolved.is_file() or not resolved.is_relative_to(resolved_root):
        raise ValueError(f"model artifact must be a regular file below --results-dir: {path}")
    if resolved.stat().st_size <= 0 or resolved.stat().st_size > 512 * 1024 * 1024:
        raise ValueError(f"model artifact has an invalid size: {path}")
    import joblib  # noqa: PLC0415

    return cast("ProbabilisticClassifier", joblib.load(resolved))


def _evaluate_batch(
    examples: list[LightCurve],
    labels: dict[int, int],
    models: list[ProbabilisticClassifier],
    accumulators: list[MetricAccumulator],
    ensemble: MetricAccumulator,
) -> None:
    features = np.stack([broker_features(curve) for curve in examples])
    target = torch.tensor([labels[curve.object_id] for curve in examples])
    probabilities = [
        torch.from_numpy(model.predict_proba(features)).to(torch.float32)
        for model in models
    ]
    for accumulator, probability in zip(accumulators, probabilities, strict=True):
        accumulator.update(probability, target)
    ensemble.update(torch.stack(probabilities).mean(dim=0), target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--expected-shards", type=int, default=11)
    parser.add_argument(
        "--allow-unsafe-local-models",
        action="store_true",
        help="opt in to loading locally produced joblib/pickle model artifacts",
    )
    args = parser.parse_args()
    if not args.allow_unsafe_local_models:
        raise SystemExit(
            "joblib model loading is disabled by default; pass "
            "--allow-unsafe-local-models only for trusted local artifacts"
        )

    labels = read_phase0_labels(
        args.data_dir / "plasticc_test_metadata.csv.gz",
        targets=PLASTICC_KNOWN_TARGETS,
        max_objects_per_class=10_000_000,
        seed=20260729,
        target_column="true_target",
    )
    paths = tuple(sorted(args.data_dir.glob("plasticc_test_lightcurves_*.csv.gz")))
    if len(paths) != args.expected_shards:
        message = f"expected {args.expected_shards} official test shards, found {len(paths)}"
        raise ValueError(message)

    models = [
        _load_trusted_model(args.results_dir / f"rf-seed{seed}.joblib", args.results_dir)
        for seed in (7, 11, 19, 23, 31)
    ]
    accumulators = [
        MetricAccumulator.create(len(PLASTICC_KNOWN_TARGETS)) for _ in models
    ]
    ensemble = MetricAccumulator.create(len(PLASTICC_KNOWN_TARGETS))
    started = time.perf_counter()
    examples: list[LightCurve] = []
    for curve in iter_light_curves(paths, set(labels)):
        examples.append(curve)
        if len(examples) < args.batch_size:
            continue
        _evaluate_batch(examples, labels, models, accumulators, ensemble)
        examples.clear()
    if examples:
        _evaluate_batch(examples, labels, models, accumulators, ensemble)
    elapsed = time.perf_counter() - started
    evaluated_objects = int(ensemble.class_count.sum())
    if evaluated_objects != len(labels):
        message = (
            f"official test object mismatch: evaluated {evaluated_objects}, "
            f"expected {len(labels)}"
        )
        raise RuntimeError(message)
    payload = {
        "model": "compact_broker_rf",
        "official_unknown_classes_excluded": [991, 992, 993, 994],
        "metric_scope": "known 14 classes; not the 15-class class-99 competition metric",
        "baseline_scope": "compact statistical features; not ALeRCE or Avocado",
        "per_seed": [accumulator.finalize() for accumulator in accumulators],
        "ensemble": ensemble.finalize(),
        "elapsed_seconds": elapsed,
        "expected_known14_objects": len(labels),
        "objects_per_second": evaluated_objects / elapsed,
    }
    output = args.results_dir / "official-test-known14.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
