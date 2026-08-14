"""Leakage-safe linear probes for astronomy descriptor audits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class RidgeProbe:
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: np.ndarray
    coefficients: np.ndarray

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        standardized = (inputs - self.x_mean) / self.x_scale
        return standardized @ self.coefficients + self.y_mean


def fit_ridge_probe(
    inputs: np.ndarray,
    targets: np.ndarray,
    alpha: float,
) -> RidgeProbe:
    """Fit a centered multi-output ridge map without penalizing an intercept."""
    x_mean = inputs.mean(axis=0)
    x_scale = inputs.std(axis=0)
    x_scale = np.where(x_scale > 1.0e-8, x_scale, 1.0)
    y_mean = targets.mean(axis=0)
    x = (inputs - x_mean) / x_scale
    y = targets - y_mean
    gram = x.T @ x
    coefficients = np.linalg.solve(
        gram + alpha * np.eye(gram.shape[0], dtype=np.float64),
        x.T @ y,
    )
    return RidgeProbe(x_mean, x_scale, y_mean, coefficients)


def variance_weighted_r2(targets: np.ndarray, predictions: np.ndarray) -> float:
    """Return pooled variance-weighted multi-output R²."""
    residual = float(np.sum(np.square(targets - predictions)))
    centered = targets - targets.mean(axis=0)
    total = float(np.sum(np.square(centered)))
    return 1.0 - residual / total if total > np.finfo(np.float64).eps else float("nan")


def select_ridge_alpha(
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    validation_inputs: np.ndarray,
    validation_targets: np.ndarray,
    candidates: tuple[float, ...],
) -> tuple[float, RidgeProbe]:
    """Choose regularization on validation data and retain a train-only probe."""
    rows = [
        (
            variance_weighted_r2(
                validation_targets,
                (probe := fit_ridge_probe(train_inputs, train_targets, alpha)).predict(
                    validation_inputs
                ),
            ),
            alpha,
            probe,
        )
        for alpha in candidates
    ]
    _, alpha, probe = max(rows, key=lambda row: row[0])
    return alpha, probe


def balanced_accuracy(targets: np.ndarray, predictions: np.ndarray) -> float:
    recalls = [
        float(np.mean(predictions[targets == class_id] == class_id))
        for class_id in np.unique(targets)
    ]
    return float(np.mean(recalls))


def within_group_permutation_pvalue(
    targets: np.ndarray,
    predictions: list[np.ndarray],
    groups: np.ndarray,
    *,
    baseline_predictions: list[np.ndarray] | None = None,
    draws: int,
    seed: int,
) -> tuple[float, float]:
    """Test incremental R² with Freedman-Lane nuisance-residual permutations."""
    baselines = (
        baseline_predictions
        if baseline_predictions is not None
        else [np.broadcast_to(targets.mean(axis=0), targets.shape)] * len(predictions)
    )
    observed = float(
        np.median(
            [
                variance_weighted_r2(targets, prediction)
                - variance_weighted_r2(targets, baseline)
                for prediction, baseline in zip(predictions, baselines, strict=True)
            ]
        )
    )
    rng = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(draws):
        order = np.arange(targets.shape[0])
        for group in np.unique(groups):
            selected = np.flatnonzero(groups == group)
            order[selected] = rng.permutation(selected)
        statistic = float(
            np.median(
                [
                    variance_weighted_r2(
                        baseline + (targets - baseline)[order],
                        prediction,
                    )
                    - variance_weighted_r2(
                        baseline + (targets - baseline)[order],
                        baseline,
                    )
                    for prediction, baseline in zip(
                        predictions,
                        baselines,
                        strict=True,
                    )
                ]
            )
        )
        exceedances += statistic >= observed
    return observed, (exceedances + 1.0) / (draws + 1.0)
