from __future__ import annotations

from pathlib import Path
from typing import Annotated, Final, Literal, cast

import typer

from .pac_tf_p1p2_runner import enqueue, run_workers, write_report
from .pac_tf_p1p2_types import EvidencePackage, P1P2Config, SyntheticEstimand
from .pac_types import PACDevice  # noqa: TC001

app = typer.Typer(add_completion=False)
DEFAULT_OUTPUT_ROOT: Final = Path(".omx/results/pac-tf-p1p2-confirmatory-20260711")
DEFAULT_PROTOCOL_PATH: Final = Path(".omx/protocols/pac_tf_confirmatory_20260711.json")
DEFAULT_SELECTION_PATH: Final = Path(
    ".omx/results/pac-tf-confirmatory-unseen-20260711/reports/confirmatory_baseline_selection.json"
)
DEFAULT_UNSEEN_ROOT: Final = Path(".omx/results/pac-tf-confirmatory-unseen-20260711")


@app.callback(invoke_without_command=True)
def main(
    stage: Annotated[Literal["enqueue", "workers", "report"], typer.Option("--stage")] = "enqueue",
    output_root: Annotated[Path, typer.Option("--output-root")] = DEFAULT_OUTPUT_ROOT,
    protocol_path: Annotated[Path, typer.Option("--protocol-path")] = DEFAULT_PROTOCOL_PATH,
    selection_path: Annotated[Path, typer.Option("--selection-path")] = DEFAULT_SELECTION_PATH,
    unseen_root: Annotated[Path, typer.Option("--unseen-root")] = DEFAULT_UNSEEN_ROOT,
    device: Annotated[PACDevice, typer.Option("--device")] = "auto",
    workers: Annotated[int, typer.Option("--workers")] = 8,
    total_slots: Annotated[int, typer.Option("--total-slots")] = 16,
    max_jobs: Annotated[int | None, typer.Option("--max-jobs")] = None,
    package: Annotated[EvidencePackage | None, typer.Option("--package")] = None,
    manifest_package: Annotated[list[str] | None, typer.Option("--manifest-package")] = None,
    synthetic_estimand: Annotated[
        SyntheticEstimand, typer.Option("--synthetic-estimand")
    ] = "sequence",
    synthetic_target_params: Annotated[
        int | None, typer.Option("--synthetic-target-params")
    ] = None,
    model: Annotated[list[str] | None, typer.Option("--model")] = None,
) -> None:
    config = P1P2Config(
        output_root=output_root,
        protocol_path=protocol_path,
        selection_path=selection_path,
        unseen_root=unseen_root,
        device=device,
        workers=workers,
        total_slots=total_slots,
        models=tuple(model) if model else P1P2Config().models,
        packages=(
            cast("tuple[EvidencePackage, ...]", tuple(manifest_package))
            if manifest_package
            else P1P2Config().packages
        ),
        synthetic_estimand=synthetic_estimand,
        synthetic_target_params=synthetic_target_params,
    )
    if stage == "enqueue":
        enqueue(config)
    elif stage == "workers":
        run_workers(config, max_jobs=max_jobs, package=package)
    else:
        write_report(output_root)


if __name__ == "__main__":
    app()
