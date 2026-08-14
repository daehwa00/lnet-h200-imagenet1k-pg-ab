from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, NoReturn

import typer
from rich.console import Console

from .controlled_damping_experiments import config_for_mode, run_controlled_damping_suite
from .controlled_damping_reports import write_controlled_damping_artifacts

if TYPE_CHECKING:
    from .controlled_damping_types import ControlledDampingDevice, ControlledDampingMode

app = typer.Typer(add_completion=False, no_args_is_help=False)


@app.command()
def main(
    *,
    mode: Annotated[str, typer.Option("--mode")] = "full",
    device: Annotated[str, typer.Option("--device")] = "auto",
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
) -> None:
    console = Console()
    try:
        parsed_mode = _parse_mode(mode)
        parsed_device = _parse_device(device)
    except ValueError as error:
        _raise_exit(console, str(error))
    resolved_output_dir = output_dir or (Path(".omx") / "results" / "controlled-damping" / mode)
    payload = run_controlled_damping_suite(config_for_mode(parsed_mode, parsed_device))
    json_path, markdown_path = write_controlled_damping_artifacts(payload, resolved_output_dir)
    console.print(f"Wrote controlled-damping report JSON to {json_path}")
    console.print(f"Wrote controlled-damping report Markdown to {markdown_path}")
    conclusion = payload["conclusion"]
    if isinstance(conclusion, dict):
        console.print(f"conclusion: {conclusion.get('status')}")


def _parse_mode(value: str) -> ControlledDampingMode:
    match value:
        case "smoke" | "full":
            return value
        case _:
            message = "mode must be one of: smoke, full"
            raise ValueError(message)


def _parse_device(value: str) -> ControlledDampingDevice:
    match value:
        case "auto" | "cpu" | "cuda":
            return value
        case _:
            message = "device must be one of: auto, cpu, cuda"
            raise ValueError(message)


def _raise_exit(console: Console, message: str) -> NoReturn:
    console.print(message, style="bold red")
    raise typer.Exit(code=1) from None


if __name__ == "__main__":
    app()
