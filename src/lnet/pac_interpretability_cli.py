from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from .pac_interpretability_report import write_interpretability_report
from .pac_interpretability_runner import enqueue_jobs, run_sanity, run_workers
from .pac_interpretability_types import (
    InterpretabilityPreset,
    InterpretabilityQueueConfig,
    InterpretabilityStage,
)
from .pac_types import PACDevice  # noqa: TC001 - Typer resolves annotations at runtime.

app = typer.Typer(add_completion=False)
DEFAULT_ROOT = Path(".omx/results/pac-hybrid-prl/interpretability-evidence-20260709")


@app.callback(invoke_without_command=True)
def main(
    stage: Annotated[InterpretabilityStage, typer.Option("--stage")] = "sanity",
    device: Annotated[PACDevice, typer.Option("--device")] = "auto",
    output_root: Annotated[Path | None, typer.Option("--output-root")] = None,
    preset: Annotated[InterpretabilityPreset, typer.Option("--preset")] = "full",
    workers: Annotated[int, typer.Option("--workers")] = 4,
    total_slots: Annotated[int, typer.Option("--total-slots")] = 8,
    max_jobs: Annotated[int | None, typer.Option("--max-jobs")] = None,
    seeds: Annotated[list[int] | None, typer.Option("--seeds")] = None,
) -> None:
    root = output_root or DEFAULT_ROOT
    config = InterpretabilityQueueConfig(
        output_root=root,
        preset=preset,
        seeds=tuple(seeds or (7, 11, 19, 23, 31)),
        device=device,
        workers=workers,
        total_slots=total_slots,
    )
    match stage:
        case "sanity":
            run_sanity(config)
        case "enqueue":
            enqueue_jobs(config)
        case "workers":
            run_workers(config, max_jobs=max_jobs)
        case "report":
            write_interpretability_report(root)


if __name__ == "__main__":
    app()
