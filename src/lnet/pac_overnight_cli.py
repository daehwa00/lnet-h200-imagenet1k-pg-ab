from __future__ import annotations

from pathlib import Path
from typing import Annotated, Final, Literal

import typer

from .pac_overnight_report import write_overnight_summary
from .pac_overnight_runner import run_queue, run_sanity
from .pac_overnight_types import OvernightConfig

StageOption = Literal["sanity", "queue", "report"]
DeviceOption = Literal["auto", "cpu", "cuda"]
DEFAULT_OUTPUT_ROOT: Final[Path] = Path(".omx/results/pac-hybrid-prl/overnight-20260705")

app = typer.Typer(add_completion=False)


@app.callback(invoke_without_command=True)
def main(
    stage: Annotated[StageOption, typer.Option("--stage")] = "sanity",
    device: Annotated[DeviceOption, typer.Option("--device")] = "auto",
    output_root: Annotated[Path, typer.Option("--output-root")] = DEFAULT_OUTPUT_ROOT,
    wait_for_current_synthetic: Annotated[
        bool | None, typer.Option("--wait-for-current-synthetic")
    ] = None,
) -> None:
    config = OvernightConfig(output_root=output_root, device=device)
    match stage:
        case "sanity":
            run_sanity(config)
        case "queue":
            run_queue(config, wait_for_current_synthetic=wait_for_current_synthetic is True)
        case "report":
            write_overnight_summary(output_root)
    typer.echo(f"overnight stage completed: {stage}")


if __name__ == "__main__":
    app()
