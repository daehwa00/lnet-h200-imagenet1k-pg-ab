from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
from scipy.stats import wilcoxon

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


class _WilcoxonResult(Protocol):
    pvalue: float


def bootstrap_summary(
    values: Sequence[float],
    *,
    iterations: int,
    seed: int,
    label: str,
) -> dict[str, object]:
    clean = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if clean.size == 0:
        return {
            "mean": None,
            "ci95_low": None,
            "ci95_high": None,
            "observations": 0,
            "ci_method": "not_estimable",
        }
    generator = np.random.default_rng(_derived_seed(seed, label))
    draws = generator.choice(clean, size=(iterations, clean.size), replace=True).mean(axis=1)
    low, high = np.quantile(draws, (0.025, 0.975))
    return {
        "mean": float(clean.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "observations": int(clean.size),
        "ci_method": "seed_bootstrap_percentile_95",
    }


def hierarchical_bootstrap_summary(
    grouped_values: Mapping[str, Sequence[float]],
    *,
    iterations: int,
    seed: int,
    label: str,
) -> dict[str, object]:
    groups = {
        key: np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
        for key, values in grouped_values.items()
    }
    groups = {key: values for key, values in groups.items() if values.size}
    if not groups:
        return {
            "mean": None,
            "ci95_low": None,
            "ci95_high": None,
            "observations": 0,
            "groups": 0,
            "ci_method": "not_estimable",
        }
    names = sorted(groups)
    generator = np.random.default_rng(_derived_seed(seed, label))
    draws = np.empty(iterations, dtype=np.float64)
    for draw_index in range(iterations):
        selected = generator.choice(len(names), size=len(names), replace=True)
        group_means = []
        for index in selected:
            values = groups[names[int(index)]]
            sampled = generator.choice(values, size=values.size, replace=True)
            group_means.append(float(sampled.mean()))
        draws[draw_index] = float(np.mean(group_means))
    point = float(np.mean([values.mean() for values in groups.values()]))
    low, high = np.quantile(draws, (0.025, 0.975))
    return {
        "mean": point,
        "ci95_low": float(low),
        "ci95_high": float(high),
        "observations": int(sum(values.size for values in groups.values())),
        "groups": len(groups),
        "ci_method": "hierarchical_group_then_seed_bootstrap_percentile_95",
    }


def paired_test_summary(
    grouped_differences: Mapping[str, Sequence[float]],
    *,
    iterations: int,
    seed: int,
    label: str,
) -> dict[str, object]:
    clean = {
        key: tuple(value for value in values if np.isfinite(value))
        for key, values in grouped_differences.items()
    }
    clean = {key: values for key, values in clean.items() if values}
    summary = hierarchical_bootstrap_summary(
        clean,
        iterations=iterations,
        seed=seed,
        label=f"{label}:hierarchical",
    )
    group_means = [float(np.mean(clean[key])) for key in sorted(clean)]
    summary.update(
        {
            "paired_observations": sum(len(values) for values in clean.values()),
            "paired_groups": len(clean),
            "wilcoxon_unit": "group_mean",
            "wilcoxon_alternative": "two-sided",
            "wilcoxon_p": _wilcoxon_p(group_means) if group_means else None,
        }
    )
    return summary


def bh_fdr(
    rows: Sequence[dict[str, object]],
    *,
    p_field: str = "wilcoxon_p",
    alpha: float = 0.05,
) -> list[dict[str, object]]:
    output = [dict(row) for row in rows]
    indexed: list[tuple[int, float]] = []
    for index, row in enumerate(output):
        value = row.get(p_field)
        if not isinstance(value, (int, float)) or not np.isfinite(value):
            continue
        indexed.append((index, float(value)))
    if not indexed:
        return output
    ranked = sorted(indexed, key=lambda item: (item[1], item[0]))
    adjusted = [1.0] * len(ranked)
    running = 1.0
    total = len(ranked)
    for reverse_index in range(total - 1, -1, -1):
        _, p_value = ranked[reverse_index]
        rank = reverse_index + 1
        running = min(running, p_value * total / rank)
        adjusted[reverse_index] = running
    for rank_index, ((row_index, _), q_value) in enumerate(
        zip(ranked, adjusted, strict=True), start=1
    ):
        output[row_index]["fdr_rank"] = rank_index
        output[row_index]["fdr_q"] = float(min(1.0, q_value))
        output[row_index]["fdr_reject_0_05"] = bool(q_value <= alpha)
    return output


def equivalence_test(differences: Sequence[float], margin: float | None) -> dict[str, object]:
    if margin is None:
        return {
            "status": "not_performed",
            "reason": "no separately locked equivalence margin is present",
            "equivalence_claim": False,
        }
    clean = [value for value in differences if np.isfinite(value)]
    if not clean:
        return {
            "status": "not_estimable",
            "margin": margin,
            "equivalence_claim": False,
        }
    lower_p = _wilcoxon_p([value + margin for value in clean], alternative="greater")
    upper_p = _wilcoxon_p([margin - value for value in clean], alternative="greater")
    return {
        "status": "performed",
        "method": "paired_wilcoxon_tost",
        "margin": margin,
        "lower_one_sided_p": lower_p,
        "upper_one_sided_p": upper_p,
        "equivalence_claim": bool(lower_p <= 0.05 and upper_p <= 0.05),
    }


def _wilcoxon_p(values: Sequence[float], *, alternative: str = "two-sided") -> float:
    clean = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if clean.size == 0 or bool(np.allclose(clean, 0.0)):
        return 1.0
    result = cast("_WilcoxonResult", cast("object", wilcoxon(clean, alternative=alternative)))
    return float(result.pvalue)


def _derived_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "little", signed=False)
