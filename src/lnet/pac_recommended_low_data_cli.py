from __future__ import annotations

from pathlib import Path
from typing import Annotated, assert_never

import typer

from .pac_recommended_low_data_report import write_low_data_report
from .pac_recommended_low_data_runner import (
    enqueue_jobs,
    enqueue_selected_test_jobs,
    enqueue_unseen_final_jobs,
    enqueue_unseen_validation_jobs,
    run_sanity,
    run_workers,
)
from .pac_recommended_low_data_types import LowDataPreset, LowDataQueueConfig, LowDataStage
from .pac_types import (  # noqa: TC001 - Typer needs runtime annotations.
    PACCompileMode,
    PACDevice,
    PACOptimizerMode,
    PACPrecision,
)

app = typer.Typer(add_completion=False)


@app.callback(invoke_without_command=True)
def main(
    stage: Annotated[LowDataStage, typer.Option("--stage")] = "sanity",
    device: Annotated[PACDevice, typer.Option("--device")] = "auto",
    output_root: Annotated[Path | None, typer.Option("--output-root")] = None,
    preset: Annotated[LowDataPreset, typer.Option("--preset")] = "full",
    workers: Annotated[int, typer.Option("--workers")] = 4,
    total_slots: Annotated[int, typer.Option("--total-slots")] = 8,
    max_jobs: Annotated[int | None, typer.Option("--max-jobs")] = None,
    seeds: Annotated[list[int] | None, typer.Option("--seeds")] = None,
    compile_mode: Annotated[PACCompileMode, typer.Option("--compile-mode")] = "none",
    precision: Annotated[PACPrecision, typer.Option("--precision")] = "fp32",
    optimizer_mode: Annotated[PACOptimizerMode, typer.Option("--optimizer-mode")] = "default",
    selection_root: Annotated[Path | None, typer.Option("--selection-root")] = None,
    final_datasets: Annotated[list[str] | None, typer.Option("--final-datasets")] = None,
    protocol_path: Annotated[Path | None, typer.Option("--protocol-path")] = None,
) -> None:
    root = output_root or Path(".omx/results/pac-hybrid-prl/recommended-low-data-20260708")
    config = LowDataQueueConfig(
        output_root=root,
        preset=preset,
        seeds=tuple(seeds or (7, 11, 19, 23, 31)),
        device=device,
        workers=workers,
        total_slots=total_slots,
        compile_mode=compile_mode,
        precision=precision,
        optimizer_mode=optimizer_mode,
    )
    active_protocol = protocol_path or Path(".omx/protocols/pac_tf_confirmatory_20260711.json")
    match stage:
        case "sanity":
            run_sanity(config)
        case "enqueue":
            enqueue_jobs(config)
        case "workers":
            run_workers(config, max_jobs=max_jobs)
        case "report":
            write_low_data_report(root)
        case "enqueue-selected-test":
            enqueue_selected_test_jobs(config)
        case "enqueue-unseen-final":
            if selection_root is None:
                message = "--selection-root is required for unseen final enqueue"
                raise typer.BadParameter(message)
            enqueue_unseen_final_jobs(
                config,
                selection_root=selection_root,
                protocol_path=active_protocol,
                datasets=tuple(final_datasets or ()),
            )
        case "enqueue-unseen-validation":
            if selection_root is None:
                message = "--selection-root is required for unseen validation enqueue"
                raise typer.BadParameter(message)
            enqueue_unseen_validation_jobs(
                config,
                selection_root=selection_root,
                protocol_path=active_protocol,
            )
        case unreachable:
            assert_never(unreachable)


if __name__ == "__main__":
    app()
