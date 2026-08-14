from __future__ import annotations

from statistics import mean
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tapped_prl_followup_schema import JsonRow, JsonValue

ADAPTIVE_TASKS = (
    "context_damped_exponential",
    "delayed_context_damped_exponential",
    "switching_teacher",
)
CONTROL_TASKS = ("modal_teacher", "random_fir_teacher", "strict_delay_6", "delayed_exponential_4")


def summary_rows(rows: list[JsonRow]) -> list[JsonRow]:
    keys = sorted(
        {
            (_as_str(row["task"]), _as_str(row["model"]), _optional_float(row["damping_beta"]))
            for row in rows
        },
    )
    return [
        _summary_row(task, model, beta, _matching_rows(rows, task, model, beta))
        for task, model, beta in keys
    ]


def conclusion(summary: list[JsonRow], runs: list[JsonRow]) -> JsonRow:
    adaptive_improvements = sum(_adaptive_improved(summary, task) for task in ADAPTIVE_TASKS)
    control_regressions = sum(_control_regressed(summary, task) for task in CONTROL_TASKS)
    interpretable = sum(
        _interpretable(summary, task)
        for task in ("context_damped_exponential", "delayed_context_damped_exponential")
    )
    stable = all(
        _optional_float(row["max_abs_discrete_decay"]) is None
        or _as_float(row["max_abs_discrete_decay"]) < 1.0
        for row in runs
    )
    if not stable or control_regressions > (len(CONTROL_TASKS) // 2):
        status = "does_not_support"
    elif adaptive_improvements >= 2 and interpretable >= 2:
        status = "supports"
    elif adaptive_improvements >= 1:
        status = "mixed"
    else:
        status = "does_not_support"
    return {
        "status": status,
        "rationale": (
            f"adaptive improvements={adaptive_improvements}/{len(ADAPTIVE_TASKS)}; "
            f"control regressions={control_regressions}/{len(CONTROL_TASKS)}; "
            f"interpretable damping regimes={interpretable}/2; stable={stable}."
        ),
        "decision_rule": (
            "supports if controlled damping improves >=2 nonstationary tasks, aligns with both "
            "controlled teacher regimes, and keeps all discrete decays stable"
        ),
    }


def _matching_rows(
    rows: list[JsonRow],
    task: str,
    model: str,
    beta: float | None,
) -> list[JsonRow]:
    return [
        row
        for row in rows
        if row["task"] == task
        and row["model"] == model
        and _optional_float(row["damping_beta"]) == beta
    ]


def _summary_row(task: str, model: str, beta: float | None, rows: list[JsonRow]) -> JsonRow:
    return {
        "task": task,
        "model": model,
        "damping_beta": beta,
        "mean_validation_loss": mean(_float_values(rows, "validation_loss")),
        "seed_count": len(rows),
        "mean_params": mean(_float_values(rows, "params")),
        "mean_max_abs_discrete_decay": _optional_mean(rows, "max_abs_discrete_decay"),
        "mean_fast_damping": _optional_mean(rows, "fast_damping_mean"),
        "mean_slow_damping": _optional_mean(rows, "slow_damping_mean"),
        "mean_damping_regime_auc": _optional_mean(rows, "damping_regime_auc"),
        "mean_damping_regime_correlation": _optional_mean(rows, "damping_regime_correlation"),
    }


def _adaptive_improved(rows: list[JsonRow], task: str) -> bool:
    selective = _loss_for(rows, task, "selective_full")
    controlled = _best_controlled_loss(rows, task)
    return controlled is not None and controlled <= (0.90 * selective)


def _control_regressed(rows: list[JsonRow], task: str) -> bool:
    selective = _loss_for(rows, task, "selective_full")
    controlled = _best_controlled_loss(rows, task)
    return controlled is not None and controlled > (1.10 * selective)


def _interpretable(rows: list[JsonRow], task: str) -> bool:
    candidates = [
        row for row in rows if row["task"] == task and row["model"] in {"damping", "damping_full"}
    ]
    if not candidates:
        return False
    best = min(candidates, key=lambda row: _as_float(row["mean_validation_loss"]))
    fast = _optional_float(best["mean_fast_damping"])
    slow = _optional_float(best["mean_slow_damping"])
    auc = _optional_float(best["mean_damping_regime_auc"])
    correlation = _optional_float(best["mean_damping_regime_correlation"])
    return (
        fast is not None
        and slow is not None
        and auc is not None
        and correlation is not None
        and fast > slow
        and auc >= 0.55
        and correlation > 0.0
    )


def _best_controlled_loss(rows: list[JsonRow], task: str) -> float | None:
    values = [
        _as_float(row["mean_validation_loss"])
        for row in rows
        if row["task"] == task and row["model"] in {"damping", "damping_full"}
    ]
    return min(values) if values else None


def _loss_for(rows: list[JsonRow], task: str, model: str) -> float:
    for row in rows:
        if row["task"] == task and row["model"] == model:
            return _as_float(row["mean_validation_loss"])
    message = f"missing loss for {task}/{model}"
    raise RuntimeError(message)


def _float_values(rows: list[JsonRow], key: str) -> list[float]:
    return [_as_float(row[key]) for row in rows]


def _optional_mean(rows: list[JsonRow], key: str) -> float | None:
    values = [_optional_float(row[key]) for row in rows]
    finite_values = [value for value in values if value is not None]
    return mean(finite_values) if finite_values else None


def _as_str(value: JsonValue) -> str:
    if isinstance(value, str):
        return value
    message = f"expected string scalar, got {type(value).__name__}"
    raise TypeError(message)


def _as_float(value: JsonValue) -> float:
    if isinstance(value, int | float | str):
        return float(value)
    message = f"expected numeric scalar, got {type(value).__name__}"
    raise TypeError(message)


def _optional_float(value: JsonValue) -> float | None:
    if value is None:
        return None
    return _as_float(value)
