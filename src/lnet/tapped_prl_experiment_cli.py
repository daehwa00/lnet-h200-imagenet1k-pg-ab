from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from .tapped_prl_experiment_schema import CheckpointSchemaError
from .tapped_prl_experiments import run_all, run_stage

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def main(
    *,
    stage: Annotated[str, typer.Option("--stage")] = "all",
    smoke: Annotated[bool, typer.Option("--smoke/--full")] = False,
    gate_choice: Annotated[str | None, typer.Option("--gate-choice")] = None,
    gate_selection_checkpoint: Annotated[
        Path | None, typer.Option("--gate-selection-checkpoint")
    ] = None,
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
) -> None:
    console = Console()
    resolved_output_dir = output_dir or (Path(".omx") / "results")
    try:
        match stage:
            case "all":
                checkpoints = run_all(output_dir=resolved_output_dir, smoke=smoke)
                message = (
                    f"Wrote {len(checkpoints)} stage checkpoints and final reports "
                    f"to {resolved_output_dir}"
                )
                console.print(message)
            case "stage1" | "stage2" | "stage3" as concrete_stage:
                checkpoint = run_stage(
                    concrete_stage,
                    output_dir=resolved_output_dir,
                    smoke=smoke,
                    gate_choice=gate_choice,
                    gate_selection_checkpoint=gate_selection_checkpoint,
                )
                console.print(f"Wrote {checkpoint.stage} checkpoint to {resolved_output_dir}")
            case _:
                _raise_exit(
                    console,
                    "stage must be one of: stage1, stage2, stage3, all",
                )
    except CheckpointSchemaError as error:
        _raise_exit(console, str(error))


def _raise_exit(console: Console, message: str) -> None:
    console.print(message, style="bold red")
    raise typer.Exit(code=1) from None


if __name__ == "__main__":
    app()
