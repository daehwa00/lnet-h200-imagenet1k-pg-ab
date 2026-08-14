# pyright: reportExplicitAny=false

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import socket
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional
from torch.utils.data import DataLoader, TensorDataset

from lnet.astronomy.fourier_shape import estimate_fourier_shape
from lnet.astronomy.harmonic_branch import (
    HarmonicFeatureHead,
    fuse_logits,
    harmonic_feature_vector,
)
from lnet.astronomy.phase0 import Phase0RunConfig, build_model
from lnet.astronomy.plasticc import (
    LightCurveBatch,
    PlasticcDataset,
    collate_light_curves,
    read_light_curves,
    read_phase0_labels,
)

BETA_GRID = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)
CONFIDENCE_GRID = (0.5, 0.6, 0.7, 0.8, 0.9)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--alphabet-results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 19, 23, 31])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_config(results_dir: Path, seed: int) -> Phase0RunConfig:
    payload = cast(
        "dict[str, Any]",
        json.loads((results_dir / f"alphabet-seed{seed}.json").read_text()),
    )
    stored = cast("dict[str, Any]", payload["config"])
    return Phase0RunConfig(
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
        lag_mode=stored.get("lag_mode", "physical"),
        injection_mode=stored.get("injection_mode", "zoh"),
        near_undamped_modes=int(stored.get("near_undamped_modes", 0)),
        near_undamped_alpha_per_day=float(
            stored.get("near_undamped_alpha_per_day", 1.0e-6)
        ),
        point_sample_local_convolution=bool(
            stored.get("point_sample_local_convolution", False)
        ),
        class_weights=tuple(cast("list[float]", stored.get("class_weights", []))),
    )


def _validate_checkpoint_config(
    results_dir: Path,
    seed: int,
    config: Phase0RunConfig,
    manifest: dict[str, Any],
    classes: int,
) -> None:
    payload = cast(
        "dict[str, Any]",
        json.loads((results_dir / f"alphabet-seed{seed}.json").read_text()),
    )
    stored = cast("dict[str, Any]", payload["config"])
    expected = {
        "model": "alphabet",
        "seed": seed,
        "classes": classes,
        "lag_mode": manifest["lag_mode"],
        "injection_mode": "impulse",
        "near_undamped_modes": manifest["near_undamped_modes"],
        "point_sample_local_convolution": manifest[
            "point_sample_local_convolution"
        ],
    }
    mismatches = {
        key: (stored.get(key), value)
        for key, value in expected.items()
        if stored.get(key) != value
    }
    if (
        manifest.get("injection_mode") != "impulse"
        or config.injection_mode != "impulse"
        or mismatches
    ):
        message = f"checkpoint is not the declared impulse variant: {mismatches}"
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


def _alphabet_logits(
    model: torch.nn.Module,
    curves: dict[int, Any],
    labels: dict[int, int],
    object_ids: list[int],
) -> Tensor:
    dataset = PlasticcDataset(curves, labels, object_ids)
    chunks: list[Tensor] = []
    with torch.no_grad():
        for start in range(0, len(dataset), 64):
            examples = [
                dataset[index]
                for index in range(start, min(start + 64, len(dataset)))
            ]
            batch = _move(collate_light_curves(examples))
            chunks.append(
                model(
                    batch.flux,
                    time_delta=batch.time_delta,
                    observation_mask=batch.observation_mask,
                    valid_mask=batch.valid_mask,
                ).cpu()
            )
    return torch.cat(chunks)


def _balanced_accuracy(logits: Tensor, targets: Tensor) -> float:
    prediction = logits.argmax(dim=-1)
    recalls = [
        float((prediction[targets == class_id] == class_id).float().mean())
        for class_id in targets.unique(sorted=True)
    ]
    return float(np.mean(recalls))


def _metrics(logits: Tensor, targets: Tensor) -> dict[str, float]:
    prediction = logits.argmax(dim=-1)
    return {
        "loss": float(functional.cross_entropy(logits, targets)),
        "balanced_accuracy": _balanced_accuracy(logits, targets),
        **{
            f"class_{int(class_id)}_recall": float(
                (prediction[targets == class_id] == class_id).float().mean()
            )
            for class_id in targets.unique(sorted=True)
        },
    }


def _complementarity(
    base_logits: Tensor,
    harmonic_logits: Tensor,
    targets: Tensor,
) -> dict[str, float]:
    base = base_logits.argmax(dim=-1)
    harmonic = harmonic_logits.argmax(dim=-1)
    base_correct = base == targets
    harmonic_correct = harmonic == targets
    oracle = torch.where(base_correct, base, harmonic)
    base_errors = ~base_correct
    return {
        "prediction_disagreement": float((base != harmonic).float().mean()),
        "harmonic_recovers_base_errors": (
            float(harmonic_correct[base_errors].float().mean())
            if base_errors.any()
            else 0.0
        ),
        "harmonic_damages_base_correct": float(
            (~harmonic_correct[base_correct]).float().mean()
        ),
        "oracle_union_balanced_accuracy": _balanced_accuracy(
            functional.one_hot(oracle, num_classes=2).float(),
            targets,
        ),
    }


def _train_head(
    train_features: Tensor,
    train_targets: Tensor,
    validation_features: Tensor,
    validation_targets: Tensor,
    *,
    seed: int,
    epochs: int,
    patience: int,
    batch_size: int,
) -> tuple[HarmonicFeatureHead, dict[str, float | int]]:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    head = HarmonicFeatureHead(train_features.shape[1], 2).cuda()
    counts = torch.bincount(train_targets, minlength=2).float()
    class_weight = (counts.sum() / (2.0 * counts)).cuda()
    loader = DataLoader(
        TensorDataset(train_features, train_targets),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=3.0e-3, weight_decay=1.0e-4)
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    best_state: dict[str, Tensor] | None = None
    for epoch in range(1, epochs + 1):
        head.train()
        for features, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = functional.cross_entropy(
                head(features.cuda()),
                targets.cuda(),
                weight=class_weight,
            )
            loss.backward()
            optimizer.step()
        head.eval()
        with torch.no_grad():
            validation_logits = head(validation_features.cuda())
            validation_loss = float(
                functional.cross_entropy(
                    validation_logits,
                    validation_targets.cuda(),
                    weight=class_weight,
                )
            )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            stale = 0
            best_state = copy.deepcopy(head.state_dict())
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        message = "harmonic head failed to produce a checkpoint"
        raise RuntimeError(message)
    head.load_state_dict(best_state)
    head.eval()
    return head, {"best_epoch": best_epoch, "validation_weighted_loss": best_loss}


def _select_beta(
    base_logits: Tensor,
    harmonic_logits: Tensor,
    targets: Tensor,
) -> tuple[float, list[dict[str, float]]]:
    rows = [
        {
            "beta": beta,
            "balanced_accuracy": _balanced_accuracy(
                fuse_logits(base_logits, harmonic_logits, beta),
                targets,
            ),
        }
        for beta in BETA_GRID
    ]
    best = max(rows, key=lambda row: (row["balanced_accuracy"], -row["beta"]))
    return best["beta"], rows


def _apply_confidence_gate(
    base_logits: Tensor,
    harmonic_logits: Tensor,
    base_maximum: float,
    harmonic_minimum: float,
) -> Tensor:
    base_confidence = torch.softmax(base_logits, dim=-1).max(dim=-1).values
    harmonic_confidence = torch.softmax(harmonic_logits, dim=-1).max(dim=-1).values
    disagree = base_logits.argmax(dim=-1) != harmonic_logits.argmax(dim=-1)
    switch = (
        disagree
        & (base_confidence <= base_maximum)
        & (harmonic_confidence >= harmonic_minimum)
    )
    return torch.where(switch.unsqueeze(-1), harmonic_logits, base_logits)


def _select_confidence_gate(
    base_logits: Tensor,
    harmonic_logits: Tensor,
    targets: Tensor,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    rows = [
        {
            "base_maximum": base_maximum,
            "harmonic_minimum": harmonic_minimum,
            "balanced_accuracy": _balanced_accuracy(
                _apply_confidence_gate(
                    base_logits,
                    harmonic_logits,
                    base_maximum,
                    harmonic_minimum,
                ),
                targets,
            ),
        }
        for base_maximum in CONFIDENCE_GRID
        for harmonic_minimum in CONFIDENCE_GRID
    ]
    best = max(
        rows,
        key=lambda row: (
            row["balanced_accuracy"],
            -row["base_maximum"],
            row["harmonic_minimum"],
        ),
    )
    return best, rows


def _standardize(
    train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
) -> tuple[Tensor, Tensor, Tensor, dict[str, list[float]]]:
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale = np.where(scale > 1.0e-8, scale, 1.0)
    arrays = [
        torch.from_numpy(((values - mean) / scale).astype(np.float32))
        for values in (train, validation, test)
    ]
    return (
        arrays[0],
        arrays[1],
        arrays[2],
        {"mean": mean.tolist(), "scale": scale.tolist()},
    )


def main() -> None:  # noqa: PLR0915
    args = _parse_args()
    if not torch.cuda.is_available():
        message = "harmonic hybrid campaign requires a CUDA host"
        raise RuntimeError(message)
    manifest = cast(
        "dict[str, Any]",
        json.loads((args.alphabet_results_dir / "split-manifest.json").read_text()),
    )
    if manifest.get("time_mode") != "actual":
        message = "harmonic hybrid is preregistered for actual-time checkpoints"
        raise ValueError(message)
    for key, filename in (
        ("metadata_sha256", "plasticc_train_metadata.csv.gz"),
        ("light_curves_sha256", "plasticc_train_lightcurves.csv.gz"),
    ):
        if _digest(args.data_dir / filename) != manifest[key]:
            message = f"data file does not match checkpoint manifest: {filename}"
            raise ValueError(message)
    targets = tuple(int(value) for value in manifest["targets"])
    labels = read_phase0_labels(
        args.data_dir / "plasticc_train_metadata.csv.gz",
        targets=targets,
        seed=int(manifest["split_seed"]),
    )
    object_ids = cast("dict[str, list[int]]", manifest["object_ids"])
    if set(labels) != set().union(*map(set, object_ids.values())):
        message = "selected labels do not match manifest object ids"
        raise ValueError(message)
    curves = read_light_curves(
        args.data_dir / "plasticc_train_lightcurves.csv.gz",
        labels,
    )
    shapes = {
        object_id: estimate_fourier_shape(curve)
        for object_id, curve in curves.items()
    }
    split_targets = {
        split: torch.tensor([labels[object_id] for object_id in ids], dtype=torch.long)
        for split, ids in object_ids.items()
    }
    feature_sets: dict[str, dict[str, Tensor]] = {}
    normalizations: dict[str, dict[str, list[float]]] = {}
    for variant, phase_coupling in (("amplitude", False), ("phase", True)):
        arrays = [
            np.stack(
                [
                    harmonic_feature_vector(
                        shapes[object_id],
                        phase_coupling=phase_coupling,
                    )
                    for object_id in object_ids[split]
                ]
            )
            for split in ("train", "validation", "test")
        ]
        train, validation, test, normalization = _standardize(*arrays)
        feature_sets[variant] = {
            "train": train,
            "validation": validation,
            "test": test,
        }
        normalizations[variant] = normalization
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_results: list[dict[str, Any]] = []
    for seed in args.seeds:
        config = _load_config(args.alphabet_results_dir, seed)
        _validate_checkpoint_config(
            args.alphabet_results_dir,
            seed,
            config,
            manifest,
            len(targets),
        )
        model = build_model(
            config,
            max(curve.flux.shape[0] for curve in curves.values()),
        ).cuda()
        checkpoint = args.alphabet_results_dir / f"alphabet-seed{seed}.pt"
        model.load_state_dict(
            torch.load(checkpoint, map_location="cuda", weights_only=True)
        )
        model.eval()
        base_logits = {
            split: _alphabet_logits(model, curves, labels, ids)
            for split, ids in object_ids.items()
        }
        variant_results: dict[str, Any] = {}
        for variant in ("amplitude", "phase"):
            features = feature_sets[variant]
            head, training = _train_head(
                features["train"],
                split_targets["train"],
                features["validation"],
                split_targets["validation"],
                seed=seed,
                epochs=args.epochs,
                patience=args.patience,
                batch_size=args.batch_size,
            )
            with torch.no_grad():
                harmonic_logits = {
                    split: head(values.cuda()).cpu()
                    for split, values in features.items()
                }
            beta, beta_rows = _select_beta(
                base_logits["validation"],
                harmonic_logits["validation"],
                split_targets["validation"],
            )
            fused_test = fuse_logits(
                base_logits["test"],
                harmonic_logits["test"],
                beta,
            )
            gate, gate_rows = _select_confidence_gate(
                base_logits["validation"],
                harmonic_logits["validation"],
                split_targets["validation"],
            )
            gated_test = _apply_confidence_gate(
                base_logits["test"],
                harmonic_logits["test"],
                gate["base_maximum"],
                gate["harmonic_minimum"],
            )
            head_path = args.output_dir / f"{variant}-head-seed{seed}.pt"
            torch.save(head.state_dict(), head_path)
            variant_results[variant] = {
                **training,
                "head_checkpoint": str(head_path),
                "head_checkpoint_sha256": _digest(head_path),
                "parameter_count": sum(
                    parameter.numel() for parameter in head.parameters()
                ),
                "harmonic_test": _metrics(
                    harmonic_logits["test"],
                    split_targets["test"],
                ),
                "test_complementarity": _complementarity(
                    base_logits["test"],
                    harmonic_logits["test"],
                    split_targets["test"],
                ),
                "selected_beta": beta,
                "validation_beta_grid": beta_rows,
                "fusion_test": _metrics(fused_test, split_targets["test"]),
                "exploratory_confidence_gate": {
                    "selected": gate,
                    "validation_grid": gate_rows,
                    "test": _metrics(gated_test, split_targets["test"]),
                },
            }
        seed_results.append(
            {
                "seed": seed,
                "alphabet_checkpoint": str(checkpoint),
                "alphabet_checkpoint_sha256": _digest(checkpoint),
                "alphabet_test": _metrics(
                    base_logits["test"],
                    split_targets["test"],
                ),
                "variants": variant_results,
            }
        )
    aggregate: dict[str, Any] = {
        metric: [
            float(
                row["variants"][variant][section]["balanced_accuracy"]
                if variant != "alphabet"
                else row["alphabet_test"]["balanced_accuracy"]
            )
            for row in seed_results
        ]
        for metric, variant, section in (
            ("alphabet_ba", "alphabet", ""),
            ("amplitude_ba", "amplitude", "harmonic_test"),
            ("phase_ba", "phase", "harmonic_test"),
            ("amplitude_fusion_ba", "amplitude", "fusion_test"),
            ("phase_fusion_ba", "phase", "fusion_test"),
        )
    }
    aggregate["amplitude_gated_ba"] = [
        float(
            row["variants"]["amplitude"]["exploratory_confidence_gate"]["test"][
                "balanced_accuracy"
            ]
        )
        for row in seed_results
    ]
    aggregate["phase_gated_ba"] = [
        float(
            row["variants"]["phase"]["exploratory_confidence_gate"]["test"][
                "balanced_accuracy"
            ]
        )
        for row in seed_results
    ]
    aggregate["median_phase_minus_amplitude_pp"] = 100.0 * float(
        np.median(
            np.asarray(aggregate["phase_ba"])
            - np.asarray(aggregate["amplitude_ba"])
        )
    )
    aggregate["median_phase_fusion_minus_alphabet_pp"] = 100.0 * float(
        np.median(
            np.asarray(aggregate["phase_fusion_ba"])
            - np.asarray(aggregate["alphabet_ba"])
        )
    )
    aggregate["median_phase_gated_minus_alphabet_pp"] = 100.0 * float(
        np.median(
            np.asarray(aggregate["phase_gated_ba"])
            - np.asarray(aggregate["alphabet_ba"])
        )
    )
    payload = {
        "schema": "lnet.astronomy.harmonic_hybrid.v1",
        "evidence_status": (
            "adaptive exploratory: the same locked test split was inspected in "
            "earlier rounds; no confirmatory claim is permitted"
        ),
        "execution_host": socket.gethostname(),
        "targets": targets,
        "seeds": args.seeds,
        "contract": {
            "encoder": "label-free pooled achromatic Fourier fit",
            "period_aliases": ["P/2", "P", "2P"],
            "amplitude_features": [
                "log_period",
                "log_A1",
                "log_R21",
                "log_R31",
                "fit_explained_variance",
                "log_observation_count",
                "reliability",
            ],
            "phase_features": [
                "cos_phi21",
                "sin_phi21",
                "cos_phi31",
                "sin_phi31",
            ],
            "alphabet": "frozen impulse checkpoint",
            "fusion_beta_grid": BETA_GRID,
            "beta_selection": "validation balanced accuracy; smallest beta wins ties",
            "exploratory_gate": {
                "status": "validation-selected exploratory gate",
                "confidence_grid": CONFIDENCE_GRID,
                "selection": "validation only",
            },
        },
        "source_sha256": {
            "script": _digest(Path(__file__)),
            "harmonic_branch": _digest(
                Path("src/lnet/astronomy/harmonic_branch.py")
            ),
            "fourier_shape": _digest(Path("src/lnet/astronomy/fourier_shape.py")),
        },
        "normalization": normalizations,
        "reliable_objects": int(
            sum(
                harmonic_feature_vector(shape, phase_coupling=False)[6] > 0.5
                for shape in shapes.values()
            )
        ),
        "seed_results": seed_results,
        "aggregate": aggregate,
        "success": {
            "standalone_phase_ba_ge_0_80": float(
                np.median(aggregate["phase_ba"])
            )
            >= 0.80,
            "phase_minus_amplitude_ge_0_03": aggregate[
                "median_phase_minus_amplitude_pp"
            ]
            >= 3.0,
            "fusion_not_worse_than_1pp": aggregate[
                "median_phase_fusion_minus_alphabet_pp"
            ]
            >= -1.0,
            "fusion_improves": aggregate[
                "median_phase_fusion_minus_alphabet_pp"
            ]
            > 0.0,
        },
        "exploratory": {
            "confidence_gate_improves": aggregate[
                "median_phase_gated_minus_alphabet_pp"
            ]
            > 0.0,
        },
    }
    output = args.output_dir / "summary.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    sys.stdout.write(json.dumps(payload["aggregate"], indent=2) + "\n")
    sys.stdout.write(json.dumps(payload["success"], indent=2) + "\n")


if __name__ == "__main__":
    main()
