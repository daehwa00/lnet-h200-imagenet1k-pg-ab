from __future__ import annotations

from math import isfinite
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from .tapped_prl_followup_schema import JsonRow, JsonValue

MEANINGFUL_LOSS_FRACTION: Final = 0.10


def mlp_necessity_conclusion(summary_rows: list[JsonRow], knockout_rows: list[JsonRow]) -> JsonRow:
    tasks = sorted({_as_str(row["task"]) for row in summary_rows})
    within_ten = 0
    full_meaningfully_better = 0
    for task in tasks:
        prl_fir = _loss(summary_rows, task, "prl_fir")
        full = _loss(summary_rows, task, "prl_fir_mlp")
        if not isfinite(prl_fir) or not isfinite(full):
            continue
        within_ten += int(prl_fir <= (1.0 + MEANINGFUL_LOSS_FRACTION) * full)
        full_meaningfully_better += int(_relative_improvement(prl_fir, full) > 0.10)
    knockout_small = _small_mlp_knockout_count(knockout_rows)
    status = _status(
        task_count=len(tasks),
        within_ten=within_ten,
        full_meaningfully_better=full_meaningfully_better,
        knockout_small=knockout_small,
    )
    return {
        "status": status,
        "rationale": (
            f"prl_fir within 10% of full on {within_ten}/{len(tasks)} tasks; "
            f"full improves >10% on {full_meaningfully_better}/{len(tasks)} tasks; "
            f"MLP-off knockout small on {knockout_small}/{len(tasks)} tasks."
        ),
        "decision_rule": "remove if prl_fir is near full and MLP knockout is small on most tasks",
    }


def task_mlp_delta_rows(summary_rows: list[JsonRow]) -> list[JsonRow]:
    rows: list[JsonRow] = []
    for task in sorted({_as_str(row["task"]) for row in summary_rows}):
        prl_fir = _loss(summary_rows, task, "prl_fir")
        full = _loss(summary_rows, task, "prl_fir_mlp")
        rows.append(
            {
                "task": task,
                "prl_fir_loss": prl_fir,
                "prl_fir_mlp_loss": full,
                "relative_mlp_improvement": _relative_improvement(prl_fir, full),
                "prl_fir_within_10_percent": prl_fir <= 1.10 * full,
            },
        )
    return rows


def _status(
    *,
    task_count: int,
    within_ten: int,
    full_meaningfully_better: int,
    knockout_small: int,
) -> str:
    most_tasks = max(1, (task_count * 2 + 2) // 3)
    if within_ten >= most_tasks and knockout_small >= most_tasks:
        return "remove_mlp"
    if full_meaningfully_better >= max(1, task_count // 2):
        return "keep_mlp"
    return "optional_mlp"


def _small_mlp_knockout_count(rows: list[JsonRow]) -> int:
    tasks = sorted(
        {_as_str(row["task"]) for row in rows if _as_str(row["knockout"]) == "mlp_off"},
    )
    small_count = 0
    for task in tasks:
        deltas = [
            _as_float(row["relative_delta"])
            for row in rows
            if row["task"] == task and row["knockout"] == "mlp_off"
        ]
        small_count += int(sum(deltas) / len(deltas) <= 0.10)
    return small_count


def _loss(rows: list[JsonRow], task: str, model: str) -> float:
    matches = [
        _as_float(row["mean_validation_loss"])
        for row in rows
        if row["task"] == task and row["model"] == model
    ]
    return matches[0] if matches else float("nan")


def _relative_improvement(reference: float, candidate: float) -> float:
    return (reference - candidate) / max(reference, 1.0e-12)


def _as_str(value: JsonValue) -> str:
    if isinstance(value, str):
        return value
    message = f"expected string JSON scalar, got {type(value).__name__}"
    raise TypeError(message)


def _as_float(value: JsonValue) -> float:
    if isinstance(value, int | float | str):
        return float(value)
    message = f"expected numeric JSON scalar, got {type(value).__name__}"
    raise TypeError(message)
