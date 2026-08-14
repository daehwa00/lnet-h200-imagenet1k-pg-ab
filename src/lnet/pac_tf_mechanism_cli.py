from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from .pac_tf_mechanism_runner import enqueue_jobs, run_sanity, run_workers, write_report
from .pac_tf_mechanism_types import MechanismQueueConfig, MechanismStage
from .pac_types import PACDevice  # noqa: TC001 - Typer resolves annotations at runtime.

app = typer.Typer(add_completion=False)
DEFAULT_ROOT = Path(".omx/results/pac-tf-mechanism-recovery-20260710")


@app.callback(invoke_without_command=True)
def main(
    stage: Annotated[MechanismStage, typer.Option("--stage")] = "sanity",
    device: Annotated[PACDevice, typer.Option("--device")] = "auto",
    output_root: Annotated[Path, typer.Option("--output-root")] = DEFAULT_ROOT,
    workers: Annotated[int, typer.Option("--workers")] = 8,
    total_slots: Annotated[int, typer.Option("--total-slots")] = 16,
    max_jobs: Annotated[int | None, typer.Option("--max-jobs")] = None,
    seeds: Annotated[list[int] | None, typer.Option("--seeds")] = None,
) -> None:
    config = MechanismQueueConfig(
        output_root=output_root,
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
            write_report(output_root)


if __name__ == "__main__":
    app()
