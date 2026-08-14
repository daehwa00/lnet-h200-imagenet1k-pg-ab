# pyright: reportExplicitAny=false, reportPrivateUsage=false
# pyright: reportUnknownLambdaType=false, reportUnnecessaryCast=false

from __future__ import annotations

import argparse
import hashlib
import json
import math
import socket
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch

from lnet.astronomy.descriptor_alignment import (
    balanced_accuracy,
    select_ridge_alpha,
    variance_weighted_r2,
    within_group_permutation_pvalue,
)
from lnet.astronomy.fourier_shape import (
    FourierShape,
    alternating_observation_views,
    estimate_fourier_shape,
    is_reliable_fourier_shape,
)
from lnet.astronomy.phase0 import Phase0RunConfig, build_model
from lnet.astronomy.plasticc import (
    LightCurveBatch,
    PlasticcDataset,
    collate_light_curves,
    read_light_curves,
    read_phase0_labels,
    stratified_object_split,
)
from lnet.astronomy.pole_audit import modal_representations

if TYPE_CHECKING:
    from lnet.alphabet import Alphabet
    from lnet.astronomy.plasticc import LightCurve, ObjectSplit

RIDGE_ALPHAS = (1.0e-4, 1.0e-3, 1.0e-2, 0.1, 1.0, 10.0, 100.0, 1000.0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 19, 23, 31])
    parser.add_argument("--permutation-draws", type=int, default=999)
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],  # noqa: S607
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _load_contract(
    results_dir: Path,
    seed: int,
) -> tuple[Phase0RunConfig, dict[str, Any]]:
    result = cast(
        "dict[str, Any]",
        json.loads((results_dir / f"alphabet-seed{seed}.json").read_text()),
    )
    stored = cast("dict[str, Any]", result["config"])
    if stored.get("model") != "alphabet":
        message = "descriptor audit requires ALPHABET checkpoints"
        raise ValueError(message)
    config = Phase0RunConfig(
        model="alphabet",
        seed=int(stored["seed"]),
        epochs=int(stored.get("epochs", 50)),
        batch_size=int(stored.get("batch_size", 64)),
        learning_rate=float(stored.get("learning_rate", 3.0e-3)),
        weight_decay=float(stored.get("weight_decay", 1.0e-4)),
        patience=int(stored.get("patience", 8)),
        model_dim=int(stored.get("model_dim", 64)),
        modes=int(stored.get("modes", 16)),
        classes=int(stored.get("classes", 2)),
        lag_mode=cast("Any", stored.get("lag_mode", "physical")),
        injection_mode=cast("Any", stored.get("injection_mode", "zoh")),
        near_undamped_modes=int(stored.get("near_undamped_modes", 0)),
        near_undamped_alpha_per_day=float(
            stored.get("near_undamped_alpha_per_day", 1.0e-6)
        ),
        point_sample_local_convolution=bool(
            stored.get("point_sample_local_convolution", False)
        ),
        class_weights=tuple(cast("list[float]", stored.get("class_weights", []))),
    )
    manifest = cast(
        "dict[str, Any]",
        json.loads((results_dir / "split-manifest.json").read_text()),
    )
    if manifest.get("time_mode") != "actual":
        message = "cross-view Fourier audit is preregistered for actual-time checkpoints"
        raise ValueError(message)
    return config, manifest


def _validate_data(data_dir: Path, manifest: dict[str, Any]) -> None:
    for key, filename in (
        ("metadata_sha256", "plasticc_train_metadata.csv.gz"),
        ("light_curves_sha256", "plasticc_train_lightcurves.csv.gz"),
    ):
        path = data_dir / filename
        if _digest(path) != manifest[key]:
            message = f"data file does not match training manifest: {path}"
            raise ValueError(message)


def _move(batch: LightCurveBatch) -> LightCurveBatch:
    return LightCurveBatch(
        flux=batch.flux.cuda(non_blocking=True),
        time_delta=batch.time_delta.cuda(non_blocking=True),
        observation_mask=batch.observation_mask.cuda(non_blocking=True),
        valid_mask=batch.valid_mask.cuda(non_blocking=True),
        target=batch.target.cuda(non_blocking=True),
        object_id=batch.object_id.cuda(non_blocking=True),
    )


def _capture(
    model: Alphabet,
    curves: dict[int, LightCurve],
    labels: dict[int, int],
    object_ids: tuple[int, ...],
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    dataset = PlasticcDataset(curves, labels, object_ids)
    descriptors: dict[int, np.ndarray] = {}
    logits: dict[int, np.ndarray] = {}
    with torch.no_grad():
        for start in range(0, len(dataset), 64):
            examples = [
                dataset[index]
                for index in range(start, min(start + 64, len(dataset)))
            ]
            cpu_batch = collate_light_curves(examples)
            batch = _move(cpu_batch)
            writer, reader = modal_representations(model, batch)
            descriptor = torch.cat((writer, reader), dim=-1)
            output = model.head(writer, reader)
            for index, object_id in enumerate(cpu_batch.object_id.tolist()):
                descriptors[object_id] = descriptor[index].cpu().numpy().astype(np.float64)
                logits[object_id] = output[index].cpu().numpy().astype(np.float64)
    return descriptors, logits


def _nuisance(
    descriptor_curve: LightCurve,
    target_curve: LightCurve,
    target: int,
    shape: FourierShape,
    classes: int,
) -> np.ndarray:
    descriptor_counts = descriptor_curve.observation_mask.sum(axis=0)
    target_counts = target_curve.observation_mask.sum(axis=0)
    target_time = np.cumsum(target_curve.time_delta, dtype=np.float64)
    baseline = float(target_time[-1] - target_time[0]) if target_time.size else 0.0
    one_hot = np.eye(classes, dtype=np.float64)[target]
    return np.concatenate(
        (
            one_hot,
            np.asarray(
                [
                    math.log(shape.period_days),
                    math.log1p(shape.observation_count),
                    math.log1p(int(descriptor_curve.observation_mask.sum())),
                    math.log1p(max(baseline, 0.0)),
                    shape.explained_variance,
                ]
            ),
            np.log1p(descriptor_counts),
            np.log1p(target_counts),
        )
    )


def _arrays(
    object_ids: tuple[int, ...],
    descriptor_curves: dict[int, LightCurve],
    target_curves: dict[int, LightCurve],
    shapes: dict[int, FourierShape | None],
    descriptors: dict[int, np.ndarray],
    logits: dict[int, np.ndarray],
    labels: dict[int, int],
    classes: int,
) -> dict[str, np.ndarray]:
    retained = [
        object_id
        for object_id in object_ids
        if is_reliable_fourier_shape(shapes[object_id])
    ]
    reliable = cast("dict[int, FourierShape]", shapes)
    return {
        "object_id": np.asarray(retained, dtype=np.int64),
        "label": np.asarray([labels[object_id] for object_id in retained]),
        "descriptor": np.stack([descriptors[object_id] for object_id in retained]),
        "logits": np.stack([logits[object_id] for object_id in retained]),
        "target": np.stack([reliable[object_id].audit_target() for object_id in retained]),
        "nuisance": np.stack(
            [
                _nuisance(
                    descriptor_curves[object_id],
                    target_curves[object_id],
                    labels[object_id],
                    reliable[object_id],
                    classes,
                )
                for object_id in retained
            ]
        ),
        "period": np.asarray(
            [reliable[object_id].period_days for object_id in retained]
        ),
        "observations": np.asarray(
            [reliable[object_id].observation_count for object_id in retained]
        ),
    }


def _standardize_targets(
    train: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
) -> None:
    mean = train["target"].mean(axis=0)
    scale = train["target"].std(axis=0)
    scale = np.where(scale > 1.0e-8, scale, 1.0)
    for arrays in (train, validation, test):
        arrays["target"] = (arrays["target"] - mean) / scale


def _coordinate_indices(modes: int) -> tuple[np.ndarray, np.ndarray]:
    energy = np.concatenate(
        (
            np.arange(modes),
            7 * modes + np.arange(modes),
        )
    )
    all_indices = np.arange(14 * modes)
    lag = np.setdiff1d(all_indices, energy)
    return energy, lag


def _probe_metrics(
    train: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    modes: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    energy, lag = _coordinate_indices(modes)
    feature_sets = {
        "nuisance": lambda row: row["nuisance"],
        "logits": lambda row: np.column_stack((row["nuisance"], row["logits"])),
        "energy": lambda row: np.column_stack(
            (row["nuisance"], row["descriptor"][:, energy])
        ),
        "lag": lambda row: np.column_stack(
            (row["nuisance"], row["descriptor"][:, lag])
        ),
        "full": lambda row: np.column_stack((row["nuisance"], row["descriptor"])),
    }
    metrics: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}
    for name, features in feature_sets.items():
        alpha, probe = select_ridge_alpha(
            features(train),
            train["target"],
            features(validation),
            validation["target"],
            RIDGE_ALPHAS,
        )
        prediction = probe.predict(features(test))
        predictions[name] = prediction
        metrics[name] = {
            "alpha": alpha,
            "test_q2": variance_weighted_r2(test["target"], prediction),
            "class_q2": {
                str(class_id): variance_weighted_r2(
                    test["target"][test["label"] == class_id],
                    prediction[test["label"] == class_id],
                )
                for class_id in np.unique(test["label"])
            },
        }
    nuisance_q2 = metrics["nuisance"]["test_q2"]
    for name in ("logits", "energy", "lag", "full"):
        metrics[name]["incremental_q2"] = metrics[name]["test_q2"] - nuisance_q2
    classifier_alpha, classifier = select_ridge_alpha(
        train["target"],
        np.eye(2)[train["label"]],
        validation["target"],
        np.eye(2)[validation["label"]],
        RIDGE_ALPHAS,
    )
    classifier_prediction = classifier.predict(test["target"]).argmax(axis=1)
    metrics["fourier_classifier"] = {
        "alpha": classifier_alpha,
        "balanced_accuracy": balanced_accuracy(
            test["label"],
            classifier_prediction,
        ),
    }
    return metrics, predictions


def _audit_direction(
    split: ObjectSplit,
    descriptor_curves: dict[int, LightCurve],
    target_curves: dict[int, LightCurve],
    shapes: dict[int, FourierShape | None],
    descriptors: dict[int, np.ndarray],
    logits: dict[int, np.ndarray],
    labels: dict[int, int],
    classes: int,
    modes: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    arrays = [
        _arrays(
            object_ids,
            descriptor_curves,
            target_curves,
            shapes,
            descriptors,
            logits,
            labels,
            classes,
        )
        for object_ids in (split.train, split.validation, split.test)
    ]
    train, validation, test = arrays
    _standardize_targets(train, validation, test)
    metrics, predictions = _probe_metrics(train, validation, test, modes)
    return (
        {
            "retained": {
                "train": len(train["object_id"]),
                "validation": len(validation["object_id"]),
                "test": len(test["object_id"]),
            },
            "metrics": metrics,
        },
        {
            **test,
            **{f"prediction_{name}": value for name, value in predictions.items()},
        },
    )


def _bootstrap_ci(
    targets: np.ndarray,
    predictions: list[np.ndarray],
    baseline_predictions: list[np.ndarray],
    *,
    draws: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        selected = rng.integers(0, targets.shape[0], size=targets.shape[0])
        values.append(
            float(
                np.median(
                    [
                        variance_weighted_r2(
                            targets[selected],
                            prediction[selected],
                        )
                        - variance_weighted_r2(
                            targets[selected],
                            baseline[selected],
                        )
                        for prediction, baseline in zip(
                            predictions,
                            baseline_predictions,
                            strict=True,
                        )
                    ]
                )
            )
        )
    return tuple(np.quantile(values, (0.025, 0.975)).tolist())


def _permutation_groups(test: dict[str, np.ndarray]) -> np.ndarray:
    return test["label"]


def _fourier_only_classifier(
    split: ObjectSplit,
    shapes: dict[int, FourierShape | None],
    labels: dict[int, int],
) -> dict[str, Any]:
    arrays: list[tuple[np.ndarray, np.ndarray]] = []
    counts: list[int] = []
    for object_ids in (split.train, split.validation, split.test):
        retained = [
            object_id
            for object_id in object_ids
            if is_reliable_fourier_shape(shapes[object_id])
        ]
        reliable = cast("dict[int, FourierShape]", shapes)
        arrays.append(
            (
                np.stack(
                    [reliable[object_id].audit_target() for object_id in retained]
                ),
                np.asarray([labels[object_id] for object_id in retained]),
            )
        )
        counts.append(len(retained))
    train, validation, test = arrays
    mean = train[0].mean(axis=0)
    scale = np.where(train[0].std(axis=0) > 1.0e-8, train[0].std(axis=0), 1.0)
    train_x = (train[0] - mean) / scale
    validation_x = (validation[0] - mean) / scale
    test_x = (test[0] - mean) / scale
    alpha, classifier = select_ridge_alpha(
        train_x,
        np.eye(2)[train[1]],
        validation_x,
        np.eye(2)[validation[1]],
        RIDGE_ALPHAS,
    )
    prediction = classifier.predict(test_x).argmax(axis=1)
    return {
        "retained": dict(zip(("train", "validation", "test"), counts, strict=True)),
        "alpha": alpha,
        "balanced_accuracy": balanced_accuracy(test[1], prediction),
    }


def main() -> None:  # noqa: PLR0915
    args = _parse_args()
    if not torch.cuda.is_available():
        message = "descriptor extraction requires a CUDA host"
        raise RuntimeError(message)
    first_config, manifest = _load_contract(args.results_dir, args.seeds[0])
    _validate_data(args.data_dir, manifest)
    targets = tuple(int(value) for value in manifest["targets"])
    labels = read_phase0_labels(
        args.data_dir / "plasticc_train_metadata.csv.gz",
        targets=targets,
        seed=int(manifest["split_seed"]),
    )
    curves = read_light_curves(
        args.data_dir / "plasticc_train_lightcurves.csv.gz",
        labels,
    )
    split = stratified_object_split(labels, seed=int(manifest["split_seed"]))
    manifest_ids = cast("dict[str, list[int]]", manifest["object_ids"])
    regenerated_ids = {
        "train": list(split.train),
        "validation": list(split.validation),
        "test": list(split.test),
    }
    if regenerated_ids != manifest_ids:
        message = "regenerated object split does not match training manifest"
        raise ValueError(message)
    views = {
        object_id: alternating_observation_views(curve)
        for object_id, curve in curves.items()
    }
    view_curves = (
        {object_id: pair[0] for object_id, pair in views.items()},
        {object_id: pair[1] for object_id, pair in views.items()},
    )
    shapes = (
        {
            object_id: estimate_fourier_shape(pair[0])
            for object_id, pair in views.items()
        },
        {
            object_id: estimate_fourier_shape(pair[1])
            for object_id, pair in views.items()
        },
    )
    full_shapes = {
        object_id: estimate_fourier_shape(curve)
        for object_id, curve in curves.items()
    }
    object_ids = tuple(sorted(labels))
    seed_rows: list[dict[str, Any]] = []
    primary_tests: list[dict[str, np.ndarray]] = []
    for seed in args.seeds:
        config, current_manifest = _load_contract(args.results_dir, seed)
        if current_manifest != manifest:
            message = "seed checkpoints do not share an identical manifest"
            raise ValueError(message)
        variant = (
            config.model_dim,
            config.modes,
            config.classes,
            config.lag_mode,
            config.injection_mode,
            config.near_undamped_modes,
            config.near_undamped_alpha_per_day,
            config.point_sample_local_convolution,
        )
        first_variant = (
            first_config.model_dim,
            first_config.modes,
            first_config.classes,
            first_config.lag_mode,
            first_config.injection_mode,
            first_config.near_undamped_modes,
            first_config.near_undamped_alpha_per_day,
            first_config.point_sample_local_convolution,
        )
        if config.seed != seed or variant != first_variant:
            message = "checkpoint configuration is inconsistent"
            raise ValueError(message)
        model = cast(
            "Alphabet",
            build_model(
                config,
                max(curve.flux.shape[0] for curve in curves.values()),
            ),
        ).cuda()
        checkpoint = args.results_dir / f"alphabet-seed{seed}.pt"
        model.load_state_dict(
            torch.load(checkpoint, map_location="cuda", weights_only=True)
        )
        model.eval()
        captures = [
            _capture(model, view, labels, object_ids)
            for view in view_curves
        ]
        primary, primary_test = _audit_direction(
            split,
            view_curves[0],
            view_curves[1],
            shapes[1],
            *captures[0],
            labels,
            len(targets),
            config.modes,
        )
        reverse, _ = _audit_direction(
            split,
            view_curves[1],
            view_curves[0],
            shapes[0],
            *captures[1],
            labels,
            len(targets),
            config.modes,
        )
        seed_rows.append(
            {
                "seed": seed,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _digest(checkpoint),
                "primary_a_to_b": primary,
                "sensitivity_b_to_a": reverse,
            }
        )
        primary_tests.append(primary_test)
    reference_test = primary_tests[0]
    if any(
        not np.array_equal(test["object_id"], reference_test["object_id"])
        for test in primary_tests[1:]
    ):
        message = "test retention differs across model seeds"
        raise RuntimeError(message)
    full_predictions = [
        test["prediction_full"]
        for test in primary_tests
    ]
    nuisance_predictions = [
        test["prediction_nuisance"]
        for test in primary_tests
    ]
    observed, permutation_pvalue = within_group_permutation_pvalue(
        reference_test["target"],
        full_predictions,
        _permutation_groups(reference_test),
        baseline_predictions=nuisance_predictions,
        draws=args.permutation_draws,
        seed=20260730,
    )
    bootstrap_ci = _bootstrap_ci(
        reference_test["target"],
        full_predictions,
        nuisance_predictions,
        draws=args.bootstrap_draws,
        seed=20260731,
    )
    incremental_q2 = [
        float(row["primary_a_to_b"]["metrics"]["full"]["incremental_q2"])
        for row in seed_rows
    ]
    energy_incremental_q2 = [
        float(row["primary_a_to_b"]["metrics"]["energy"]["incremental_q2"])
        for row in seed_rows
    ]
    fourier_ba = [
        float(
            row["primary_a_to_b"]["metrics"]["fourier_classifier"][
                "balanced_accuracy"
            ]
        )
        for row in seed_rows
    ]
    class_q2 = {
        str(class_id): [
            float(
                row["primary_a_to_b"]["metrics"]["full"]["class_q2"][str(class_id)]
                - row["primary_a_to_b"]["metrics"]["nuisance"]["class_q2"][
                    str(class_id)
                ]
            )
            for row in seed_rows
        ]
        for class_id in range(len(targets))
    }
    success = {
        "fourier_classifier_ba_ge_0_80": float(np.median(fourier_ba)) >= 0.80,
        "incremental_q2_ge_0_15": float(np.median(incremental_q2)) >= 0.15,
        "bootstrap_lower_gt_0": bootstrap_ci[0] > 0.0,
        "positive_conditional_permutation_p_le_0_01": (
            observed > 0.0 and permutation_pvalue <= 0.01
        ),
        "four_of_five_positive": sum(value > 0.0 for value in incremental_q2) >= 4,
        "both_classes_positive": all(
            float(np.median(values)) > 0.0 for values in class_q2.values()
        ),
        "lag_mechanism_delta_ge_0_03": float(
            np.median(np.asarray(incremental_q2) - energy_incremental_q2)
        )
        >= 0.03,
    }
    payload = {
        "schema": "lnet.astronomy.fourier_descriptor_audit.v1",
        "execution_host": socket.gethostname(),
        "code_revision": _git_revision(),
        "source_sha256": {
            "audit_script": _digest(Path(__file__)),
            "fourier_shape": _digest(Path("src/lnet/astronomy/fourier_shape.py")),
            "descriptor_alignment": _digest(
                Path("src/lnet/astronomy/descriptor_alignment.py")
            ),
        },
        "targets": targets,
        "seeds": args.seeds,
        "contract": {
            "views": "deterministic alternating observations within each passband",
            "primary_direction": "ALPHABET view A -> Fourier view B",
            "base_period_search_days": [0.05, 10.0],
            "accepted_alias_range_days": [0.05, 20.0],
            "alias_candidates": ["P/2", "P", "2P"],
            "target_scope": "pooled achromatic multiband Fourier proxy",
            "fourier_targets": [
                "log_R21",
                "log_R31",
                "cos_phi21",
                "sin_phi21",
                "cos_phi31",
                "sin_phi31",
            ],
            "quality": {
                "minimum_observations": 30,
                "minimum_explained_variance": 0.10,
            },
            "probe_fit": "training split only",
            "regularization_selection": "validation split only",
            "evaluation": "locked test split only",
        },
        "seed_results": seed_rows,
        "aggregate": {
            "incremental_full_q2": incremental_q2,
            "median_incremental_full_q2": float(np.median(incremental_q2)),
            "incremental_energy_q2": energy_incremental_q2,
            "median_incremental_full_minus_energy_q2": float(
                np.median(np.asarray(incremental_q2) - energy_incremental_q2)
            ),
            "fourier_classifier_ba": fourier_ba,
            "median_fourier_classifier_ba": float(np.median(fourier_ba)),
            "class_q2": class_q2,
            "full_view_fourier_classifier": _fourier_only_classifier(
                split,
                full_shapes,
                labels,
            ),
            "conditional_permutation": {
                "observed_median_q2": observed,
                "pvalue": permutation_pvalue,
                "draws": args.permutation_draws,
            },
            "object_bootstrap_95_ci": bootstrap_ci,
        },
        "success_criteria": success,
        "overall_pass": all(success.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    sys.stdout.write(json.dumps(payload["aggregate"], indent=2) + "\n")
    sys.stdout.write(
        json.dumps(
            {"success": success, "overall_pass": all(success.values())},
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
