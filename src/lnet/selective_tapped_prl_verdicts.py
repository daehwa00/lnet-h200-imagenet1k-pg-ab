from __future__ import annotations

from math import isfinite
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tapped_prl_followup_schema import JsonRow, JsonValue


def selectivity_verdict(rows: list[JsonRow]) -> JsonRow:
    tasks = sorted({_as_str(row["task"]) for row in rows})
    improved = 0
    near_hybrid = 0
    for task in tasks:
        fixed = loss_for(rows, task, "selective_fixed")
        full = loss_for(rows, task, "selective_full")
        hybrid = loss_for(rows, task, "hybrid_prl_fir_mlp")
        improved += int(full < fixed)
        near_hybrid += int(full <= 1.20 * hybrid)
    support_threshold = improved >= len(tasks) // 2 + 1 and near_hybrid >= len(tasks) // 2
    status = "supports" if support_threshold else "mixed"
    if improved == 0:
        status = "does_not_support"
    return {
        "status": status,
        "rationale": (
            f"full beats fixed on {improved}/{len(tasks)} tasks; "
            f"within 20% of hybrid on {near_hybrid}/{len(tasks)}."
        ),
    }


def delay_verdict(rows: list[JsonRow]) -> JsonRow:
    groups = sorted({(_as_int(row["true_delay"]), _as_str(row["model"])) for row in rows})
    wins = 0
    total = 0
    for delay, model in groups:
        insufficient = _finite_losses(rows, delay, model, horizon_satisfied=False)
        sufficient = _finite_losses(rows, delay, model, horizon_satisfied=True)
        if insufficient and sufficient:
            total += 1
            wins += int(min(sufficient) < min(insufficient))
    status = "supports" if total > 0 and wins == total else "mixed"
    if wins == 0:
        status = "does_not_support"
    return {
        "status": status,
        "rationale": f"sufficient K improves best loss in {wins}/{total} delay/model groups.",
    }


def parameter_verdict(rows: list[JsonRow]) -> JsonRow:
    tasks = sorted({_as_str(row["task"]) for row in rows})
    competitive = 0
    smaller_than_two = 0
    for task in tasks:
        task_rows = [row for row in rows if row["task"] == task]
        best = min(_as_float(row["validation_loss"]) for row in task_rows)
        selective = next(row for row in task_rows if row["model"] == "selective_full")
        baselines = [row for row in task_rows if row["model"] != "selective_full"]
        competitive += int(_as_float(selective["validation_loss"]) <= 1.10 * best)
        smaller_than_two += int(
            sum(_as_int(selective["params"]) < _as_int(row["params"]) for row in baselines) >= 2,
        )
    status = "supports" if competitive >= 3 and smaller_than_two >= 3 else "mixed"
    if competitive == 0:
        status = "does_not_support"
    return {
        "status": status,
        "rationale": (
            f"selective_full within 10% best on {competitive}/{len(tasks)} tasks and "
            f"smaller than >=2 baselines on {smaller_than_two}/{len(tasks)}."
        ),
    }


def loss_for(rows: list[JsonRow], task: str, model: str) -> float:
    return min(
        _as_float(row["validation_loss"])
        for row in rows
        if row["task"] == task and row["model"] == model
    )


def _finite_losses(
    rows: list[JsonRow],
    delay: int,
    model: str,
    *,
    horizon_satisfied: bool,
) -> list[float]:
    losses: list[float] = []
    for row in rows:
        if _as_int(row["true_delay"]) != delay:
            continue
        if row["model"] != model:
            continue
        if _as_bool(row["horizon_satisfied"]) != horizon_satisfied:
            continue
        loss = _as_float(row["validation_loss"])
        if isfinite(loss):
            losses.append(loss)
    return losses


def _as_float(value: JsonValue) -> float:
    if isinstance(value, int | float | str):
        return float(value)
    message = f"expected numeric JSON scalar, got {type(value).__name__}"
    raise TypeError(message)


def _as_int(value: JsonValue) -> int:
    if isinstance(value, int | float | str):
        return int(value)
    message = f"expected integer JSON scalar, got {type(value).__name__}"
    raise TypeError(message)


def _as_bool(value: JsonValue) -> bool:
    if isinstance(value, bool):
        return value
    message = f"expected boolean JSON scalar, got {type(value).__name__}"
    raise TypeError(message)


def _as_str(value: JsonValue) -> str:
    if isinstance(value, str):
        return value
    message = f"expected string JSON scalar, got {type(value).__name__}"
    raise TypeError(message)
