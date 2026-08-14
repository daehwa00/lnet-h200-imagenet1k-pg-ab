from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from .pac_paper_queue_report import write_paper_queue_reports
from .pac_paper_queue_runner import enqueue_jobs, run_sanity, run_workers
from .pac_paper_queue_types import PaperPreset, PaperQueueConfig, PaperStage
from .pac_types import PACDevice  # noqa: TC001 - Typer resolves CLI annotations at runtime.

app = typer.Typer(add_completion=False)


@app.callback(invoke_without_command=True)
def main(
    stage: Annotated[PaperStage, typer.Option("--stage")] = "sanity",
    device: Annotated[PACDevice, typer.Option("--device")] = "auto",
    output_root: Annotated[Path | None, typer.Option("--output-root")] = None,
    preset: Annotated[PaperPreset, typer.Option("--preset")] = "full",
    workers: Annotated[int, typer.Option("--workers")] = 4,
    total_slots: Annotated[int, typer.Option("--total-slots")] = 4,
    max_jobs: Annotated[int | None, typer.Option("--max-jobs")] = None,
    seeds: Annotated[list[int] | None, typer.Option("--seeds")] = None,
) -> None:
    config = PaperQueueConfig(
        output_root=output_root or Path(".omx/results/pac-hybrid-prl/paper-queue-20260706"),
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
            write_paper_queue_reports(
                output_root or Path(".omx/results/pac-hybrid-prl/paper-queue-20260706")
            )


if __name__ == "__main__":
    app()
