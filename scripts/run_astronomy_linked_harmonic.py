# pyright: reportExplicitAny=false

"""Exploratory linked-impulse harmonic-bank ablation on a frozen split."""

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
from torch.utils.data import DataLoader

from lnet.astronomy.linked_harmonic import (
    LinkedImpulseHarmonicBranch,
    fuse_linked_logits,
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--alphabet-results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 19, 23, 31])
    parser.add_argument("--base-modes", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _move(batch: LightCurveBatch) -> LightCurveBatch:
    return LightCurveBatch(
        flux=batch.flux.cuda(non_blocking=True),
        time_delta=batch.time_delta.cuda(non_blocking=True),
        observation_mask=batch.observation_mask.cuda(non_blocking=True),
        valid_mask=batch.valid_mask.cuda(non_blocking=True),
        target=batch.target.cuda(non_blocking=True),
        object_id=batch.object_id.cuda(non_blocking=True),
    )


def _balanced_accuracy(logits: Tensor, targets: Tensor) -> float:
    prediction = logits.argmax(dim=-1)
    return float(
        np.mean(
            [
                float((prediction[targets == class_id] == class_id).float().mean())
                for class_id in targets.unique(sorted=True)
            ]
        )
    )


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


def _branch_logits(
    model: LinkedImpulseHarmonicBranch,
    dataset: PlasticcDataset,
    batch_size: int,
) -> Tensor:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_light_curves,
    )
    chunks: list[Tensor] = []
    model.eval()
    with torch.no_grad():
        for cpu_batch in loader:
            batch = _move(cpu_batch)
            chunks.append(
                model(
                    batch.flux,
                    time_delta=batch.time_delta,
                    observation_mask=batch.observation_mask,
                    valid_mask=batch.valid_mask,
                ).cpu()
            )
    return torch.cat(chunks)


def _alphabet_logits(
    model: torch.nn.Module,
    dataset: PlasticcDataset,
    batch_size: int,
) -> Tensor:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_light_curves,
    )
    chunks: list[Tensor] = []
    model.eval()
    with torch.no_grad():
        for cpu_batch in loader:
            batch = _move(cpu_batch)
            chunks.append(
                model(
                    batch.flux,
                    time_delta=batch.time_delta,
                    observation_mask=batch.observation_mask,
                    valid_mask=batch.valid_mask,
                ).cpu()
            )
    return torch.cat(chunks)


def _train_branch(
    train: PlasticcDataset,
    validation: PlasticcDataset,
    *,
    phase_coupling: bool,
    capacity_matched: bool,
    seed: int,
    base_modes: int,
    epochs: int,
    patience: int,
    batch_size: int,
) -> tuple[LinkedImpulseHarmonicBranch, dict[str, float | int]]:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = LinkedImpulseHarmonicBranch(
        6,
        base_modes,
        2,
        phase_coupling=phase_coupling,
        capacity_matched=capacity_matched,
    ).cuda()
    targets = torch.tensor([train[index][1] for index in range(len(train))])
    counts = torch.bincount(targets, minlength=2).float()
    class_weight = (counts.sum() / (2.0 * counts)).cuda()
    loader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_light_curves,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-3, weight_decay=1.0e-4)
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for cpu_batch in loader:
            batch = _move(cpu_batch)
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                batch.flux,
                time_delta=batch.time_delta,
                observation_mask=batch.observation_mask,
                valid_mask=batch.valid_mask,
            )
            loss = functional.cross_entropy(logits, batch.target, weight=class_weight)
            loss.backward()
            optimizer.step()
        validation_logits = _branch_logits(model, validation, batch_size)
        validation_targets = torch.tensor(
            [validation[index][1] for index in range(len(validation))]
        )
        validation_loss = float(
            functional.cross_entropy(
                validation_logits,
                validation_targets,
                weight=class_weight.cpu(),
            )
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        message = "linked harmonic branch failed to produce a checkpoint"
        raise RuntimeError(message)
    model.load_state_dict(best_state)
    model.eval()
    return model, {"best_epoch": best_epoch, "validation_weighted_loss": best_loss}


def _select_beta(
    base: Tensor,
    branch: Tensor,
    targets: Tensor,
) -> tuple[float, list[dict[str, float]]]:
    rows = [
        {
            "beta": beta,
            "balanced_accuracy": _balanced_accuracy(
                fuse_linked_logits(base, branch, beta),
                targets,
            ),
        }
        for beta in BETA_GRID
    ]
    best = max(rows, key=lambda row: (row["balanced_accuracy"], -row["beta"]))
    return best["beta"], rows


def _load_config(results_dir: Path, seed: int) -> tuple[dict[str, Any], Phase0RunConfig]:
    payload = cast(
        "dict[str, Any]",
        json.loads((results_dir / f"alphabet-seed{seed}.json").read_text()),
    )
    stored = cast("dict[str, Any]", payload["config"])
    config = Phase0RunConfig(
        model=stored["model"],
        seed=int(stored["seed"]),
        epochs=int(stored["epochs"]),
        batch_size=int(stored["batch_size"]),
        learning_rate=float(stored["learning_rate"]),
        weight_decay=float(stored["weight_decay"]),
        patience=int(stored["patience"]),
        model_dim=int(stored["model_dim"]),
        modes=int(stored["modes"]),
        classes=int(stored["classes"]),
        lag_mode=stored["lag_mode"],
        injection_mode=stored["injection_mode"],
        near_undamped_modes=int(stored["near_undamped_modes"]),
        near_undamped_alpha_per_day=float(stored["near_undamped_alpha_per_day"]),
        point_sample_local_convolution=bool(stored["point_sample_local_convolution"]),
        class_weights=tuple(cast("list[float]", stored["class_weights"])),
    )
    return stored, config


def _validate_config(
    stored: dict[str, Any],
    manifest: dict[str, Any],
    *,
    seed: int,
    classes: int,
) -> None:
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
    if manifest.get("injection_mode") != "impulse" or mismatches:
        message = f"checkpoint is not the declared impulse variant: {mismatches}"
        raise ValueError(message)


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        message = "linked harmonic campaign requires a CUDA host"
        raise RuntimeError(message)
    manifest = cast(
        "dict[str, Any]",
        json.loads((args.alphabet_results_dir / "split-manifest.json").read_text()),
    )
    if manifest.get("time_mode") != "actual":
        message = "linked harmonic campaign requires actual-time checkpoints"
        raise ValueError(message)
    for key, filename in (
        ("metadata_sha256", "plasticc_train_metadata.csv.gz"),
        ("light_curves_sha256", "plasticc_train_lightcurves.csv.gz"),
    ):
        if _digest(args.data_dir / filename) != manifest[key]:
            message = f"data file does not match manifest: {filename}"
            raise ValueError(message)
    targets = tuple(int(value) for value in manifest["targets"])
    labels = read_phase0_labels(
        args.data_dir / "plasticc_train_metadata.csv.gz",
        targets=targets,
        seed=int(manifest["split_seed"]),
    )
    object_ids = cast("dict[str, list[int]]", manifest["object_ids"])
    curves = read_light_curves(
        args.data_dir / "plasticc_train_lightcurves.csv.gz",
        labels,
    )
    datasets = {
        split: PlasticcDataset(curves, labels, ids)
        for split, ids in object_ids.items()
    }
    targets_by_split = {
        split: torch.tensor([labels[object_id] for object_id in ids])
        for split, ids in object_ids.items()
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_results: list[dict[str, Any]] = []
    for seed in args.seeds:
        stored, config = _load_config(args.alphabet_results_dir, seed)
        _validate_config(stored, manifest, seed=seed, classes=len(targets))
        alphabet = build_model(
            config,
            max(curve.flux.shape[0] for curve in curves.values()),
        ).cuda()
        alphabet_path = args.alphabet_results_dir / f"alphabet-seed{seed}.pt"
        alphabet.load_state_dict(
            torch.load(alphabet_path, map_location="cuda", weights_only=True)
        )
        base_logits = {
            split: _alphabet_logits(alphabet, dataset, args.batch_size)
            for split, dataset in datasets.items()
        }
        variants: dict[str, Any] = {}
        for name, phase_coupling, capacity_matched in (
            ("amplitude", False, False),
            ("amplitude_matched", False, True),
            ("phase", True, False),
        ):
            branch, training = _train_branch(
                datasets["train"],
                datasets["validation"],
                phase_coupling=phase_coupling,
                capacity_matched=capacity_matched,
                seed=seed,
                base_modes=args.base_modes,
                epochs=args.epochs,
                patience=args.patience,
                batch_size=args.batch_size,
            )
            branch_logits = {
                split: _branch_logits(branch, dataset, args.batch_size)
                for split, dataset in datasets.items()
            }
            beta, beta_grid = _select_beta(
                base_logits["validation"],
                branch_logits["validation"],
                targets_by_split["validation"],
            )
            checkpoint = args.output_dir / f"linked-{name}-seed{seed}.pt"
            torch.save(branch.state_dict(), checkpoint)
            variants[name] = {
                **training,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _digest(checkpoint),
                "parameter_count": sum(p.numel() for p in branch.parameters()),
                "test": _metrics(
                    branch_logits["test"],
                    targets_by_split["test"],
                ),
                "selected_beta": beta,
                "validation_beta_grid": beta_grid,
                "fusion_test": _metrics(
                    fuse_linked_logits(
                        base_logits["test"],
                        branch_logits["test"],
                        beta,
                    ),
                    targets_by_split["test"],
                ),
            }
        seed_results.append(
            {
                "seed": seed,
                "alphabet_checkpoint": str(alphabet_path),
                "alphabet_checkpoint_sha256": _digest(alphabet_path),
                "alphabet_test": _metrics(
                    base_logits["test"],
                    targets_by_split["test"],
                ),
                "variants": variants,
            }
        )
    aggregate: dict[str, Any] = {
        "alphabet_ba": [
            row["alphabet_test"]["balanced_accuracy"] for row in seed_results
        ],
        "amplitude_ba": [
            row["variants"]["amplitude"]["test"]["balanced_accuracy"]
            for row in seed_results
        ],
        "phase_ba": [
            row["variants"]["phase"]["test"]["balanced_accuracy"]
            for row in seed_results
        ],
        "amplitude_matched_ba": [
            row["variants"]["amplitude_matched"]["test"]["balanced_accuracy"]
            for row in seed_results
        ],
        "phase_fusion_ba": [
            row["variants"]["phase"]["fusion_test"]["balanced_accuracy"]
            for row in seed_results
        ],
    }
    aggregate["median_phase_minus_amplitude_pp"] = 100.0 * float(
        np.median(np.asarray(aggregate["phase_ba"]) - np.asarray(aggregate["amplitude_ba"]))
    )
    aggregate["median_phase_minus_capacity_matched_pp"] = 100.0 * float(
        np.median(
            np.asarray(aggregate["phase_ba"])
            - np.asarray(aggregate["amplitude_matched_ba"])
        )
    )
    aggregate["median_phase_fusion_minus_alphabet_pp"] = 100.0 * float(
        np.median(
            np.asarray(aggregate["phase_fusion_ba"])
            - np.asarray(aggregate["alphabet_ba"])
        )
    )
    payload = {
        "schema": "lnet.astronomy.linked_harmonic.v1",
        "evidence_status": (
            "adaptive exploratory: the same locked test split was inspected in "
            "earlier rounds; no confirmatory claim is permitted"
        ),
        "execution_host": socket.gethostname(),
        "seeds": args.seeds,
        "contract": {
            "injection": "point-sample impulse",
            "frequency_linkage": "hard omega_k = k * omega_1 for k=1..4",
            "family_selection": "object-wise maximum linked-family modal energy",
            "phase_invariants": "Re/Im of normalized z_k * conj(z_1)^k",
            "capacity_control": (
                "same 12D RMSNorm and classifier as phase, with six zero phase coordinates"
            ),
            "base_modes": args.base_modes,
            "period_range_days": [0.05, 10.0],
            "alphabet": "frozen validated impulse checkpoint",
            "fusion": "validation-selected nonnegative scalar beta",
        },
        "source_sha256": {
            "script": _digest(Path(__file__)),
            "linked_harmonic": _digest(
                Path("src/lnet/astronomy/linked_harmonic.py")
            ),
        },
        "seed_results": seed_results,
        "aggregate": aggregate,
    }
    output = args.output_dir / "summary.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    sys.stdout.write(json.dumps(aggregate, indent=2) + "\n")


if __name__ == "__main__":
    main()
