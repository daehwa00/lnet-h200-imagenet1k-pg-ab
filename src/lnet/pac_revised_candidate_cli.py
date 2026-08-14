from __future__ import annotations

from pathlib import Path  # noqa: TC003 - Typer resolves annotations at runtime
from typing import Annotated, Literal

import typer

from .pac_revised_candidate import DEFAULT_ROOT, enqueue_jobs, run_workers, write_report

app = typer.Typer(add_completion=False)


@app.callback(invoke_without_command=True)
def main(
    stage: Annotated[
        Literal["enqueue", "workers", "report", "all"], typer.Option("--stage")
    ] = "all",
    output_root: Annotated[Path, typer.Option("--output-root")] = DEFAULT_ROOT,
    collection: Annotated[
        Literal["development", "untouched"], typer.Option("--collection")
    ] = "development",
    device: Annotated[Literal["cpu", "cuda"], typer.Option("--device")] = "cuda",
    workers: Annotated[int, typer.Option("--workers")] = 8,
    total_slots: Annotated[int, typer.Option("--total-slots")] = 16,
    max_jobs: Annotated[int | None, typer.Option("--max-jobs")] = None,
) -> None:
    if stage in {"enqueue", "all"}:
        typer.echo(f"enqueued jobs: {enqueue_jobs(output_root, collection=collection)}")
    if stage in {"workers", "all"}:
        run_workers(
            output_root,
            device=device,
            workers=workers,
            total_slots=total_slots,
            max_jobs=max_jobs,
        )
    if stage in {"report", "all"}:
        report = write_report(output_root)
        typer.echo(f"candidate status: {report['status']}")


if __name__ == "__main__":
    app()
