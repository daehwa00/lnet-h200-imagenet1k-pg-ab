# ruff: noqa: EM101, TRY003
from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import typer

from .pac_external_benchmarks import ExternalBenchmarkConfig, run_external_benchmarks
from .pac_unified_campaign import (
    DEFAULT_ROOT,
    campaign_status,
    enqueue_phase1,
    enqueue_ucr_test,
    run_manifest,
)
from .pac_unified_models import PAC_UNIFIED_MODEL

app = typer.Typer(add_completion=False)


@app.callback(invoke_without_command=True)
def main(
    stage: Annotated[
        Literal["smoke", "enqueue", "worker", "enqueue-test", "status"],
        typer.Option("--stage"),
    ] = "status",
    output_root: Annotated[Path, typer.Option("--output-root")] = DEFAULT_ROOT,
    manifest: Annotated[Path | None, typer.Option("--manifest")] = None,
    device: Annotated[Literal["cpu", "cuda"], typer.Option("--device")] = "cuda",
    workers: Annotated[int, typer.Option("--workers")] = 2,
) -> None:
    match stage:
        case "smoke":
            smoke_root = output_root / "smoke"
            run_external_benchmarks(
                ExternalBenchmarkConfig(
                    data_root=Path("data/external"),
                    output_root=smoke_root,
                    datasets=("lra-listops",),
                    models=("pac",),
                    epochs=1,
                    batch_size=4,
                    patience=1,
                    seeds=(7,),
                    device=device,
                    smoke=True,
                    latency_warmup=0,
                    latency_iterations=1,
                    pac_model=PAC_UNIFIED_MODEL,
                )
            )
            typer.echo(f"smoke={smoke_root}")
        case "enqueue":
            typer.echo(f"phase1_jobs={enqueue_phase1(output_root, workers=workers)}")
        case "worker":
            if manifest is None:
                raise typer.BadParameter("--manifest is required for worker stage")
            run_manifest(output_root, manifest, device=device)
        case "enqueue-test":
            jobs, refit_epochs = enqueue_ucr_test(output_root, workers=workers)
            typer.echo(f"phase2_jobs={jobs} refit_epochs={refit_epochs}")
        case "status":
            typer.echo(campaign_status(output_root))


if __name__ == "__main__":
    app()
