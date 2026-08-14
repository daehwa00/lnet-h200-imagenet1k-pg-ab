from __future__ import annotations

from pathlib import Path  # noqa: TC003 - Typer resolves annotations at runtime.
from typing import Annotated, Literal

import typer

from .pac_tf_evidence_queue import (
    CAPACITY_SELECTION_PATH,
    DEFAULT_ROOT,
    PROTOCOL_PATH,
    write_exploratory_mechanism_statistics,
    write_manifests,
    write_mechanism_statistics,
    write_validation_statistics,
)
from .pac_tf_evidence_runner import run_workers, write_status

Stage = Literal["enqueue", "workers", "report", "all"]
app = typer.Typer(add_completion=False)


@app.callback(invoke_without_command=True)
def main(
    stage: Annotated[Stage, typer.Option("--stage")] = "all",
    output_root: Annotated[Path, typer.Option("--output-root")] = DEFAULT_ROOT,
    protocol: Annotated[Path, typer.Option("--protocol")] = PROTOCOL_PATH,
    capacity_selection: Annotated[
        Path, typer.Option("--capacity-selection")
    ] = CAPACITY_SELECTION_PATH,
    kind: Annotated[
        Literal[
            "core_ablation",
            "mechanism_checkpoint",
            "interpretability",
            "sensitivity",
        ],
        typer.Option("--kind"),
    ] = "core_ablation",
    device: Annotated[Literal["auto", "cpu", "cuda"], typer.Option("--device")] = "auto",
    workers: Annotated[int, typer.Option("--workers")] = 4,
    total_slots: Annotated[int, typer.Option("--total-slots")] = 8,
    max_jobs: Annotated[int | None, typer.Option("--max-jobs")] = None,
    exploratory_mechanism_artifact: Annotated[
        Path | None,
        typer.Option(
            "--exploratory-mechanism-artifact",
            help=(
                "Explicit historical artifact; produces a labelled exploratory "
                "mechanism report only."
            ),
        ),
    ] = None,
) -> None:
    """Prepare or resume locked PAC-TF evidence queues."""
    if exploratory_mechanism_artifact is not None and stage != "report":
        message = "--exploratory-mechanism-artifact is only valid with --stage report"
        raise typer.BadParameter(message)
    if stage in {"enqueue", "all"}:
        counts = write_manifests(
            output_root,
            protocol,
            capacity_selection,
            None,
        )
        typer.echo(f"manifest counts: {counts}")
    if stage == "workers":
        run_workers(
            output_root,
            kind=kind,
            device=device,
            workers=workers,
            total_slots=total_slots,
            max_jobs=max_jobs,
        )
    if stage in {"report", "all"}:
        if exploratory_mechanism_artifact is not None:
            report = write_exploratory_mechanism_statistics(
                output_root,
                protocol,
                exploratory_mechanism_artifact,
            )
            typer.echo(
                f"exploratory paired comparisons: {len(report['comparisons'])}"
            )
        else:
            report = write_mechanism_statistics(output_root, protocol)
            write_validation_statistics(output_root, protocol)
            write_status(output_root)
            typer.echo(f"selected paired comparisons: {len(report['comparisons'])}")


if __name__ == "__main__":
    app()
